# situp_minimal.py
# Minimal, robust sit-up counter (MediaPipe Pose).
# Close the window to exit.

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time

# --------- Tunable constants (edit here) ----------
UP_TH = 95            # deg: curled <= this is "UP" (more forgiving)
DOWN_TH = 145         # deg: flat   >= this is "DOWN" (more reachable)
SMOOTH = 7            # frames for moving average
MINVIS = 0.35         # min avg visibility on chosen side
TOP_HOLD_MS = 250     # hold at top before counting on the way down

# --------- MediaPipe (version-safe drawing) ---------
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def _lm_style():
    try:
        return mp_styles.get_default_pose_landmarks_style()
    except Exception:
        return mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)

def _conn_style():
    if hasattr(mp_styles, "get_default_pose_connections_style"):
        try:
            return mp_styles.get_default_pose_connections_style()
        except Exception:
            pass
    return mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

LM_STYLE = _lm_style()
CONN_STYLE = _conn_style()

# --------- Helpers ---------
def angle_abc(a, b, c):
    """Return angle at point b (degrees) for triangle a-b-c."""
    a, b, c = np.array(a, np.float32), np.array(b, np.float32), np.array(c, np.float32)
    ba, bc = a - b, c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def side_visibility(lms, side):
    """Average visibility of shoulder-hip-knee on a side."""
    pl = mp_pose.PoseLandmark
    idx = [pl.LEFT_SHOULDER.value, pl.LEFT_HIP.value,  pl.LEFT_KNEE.value] if side=="LEFT" \
        else [pl.RIGHT_SHOULDER.value, pl.RIGHT_HIP.value, pl.RIGHT_KNEE.value]
    return float(np.mean([lms[i].visibility for i in idx]))

def get_points(hw, lms, side):
    """(S, H, K) pixel coords for shoulder, hip, knee of the chosen side."""
    w, h = hw; pl = mp_pose.PoseLandmark
    if side=="RIGHT":
        S,H,K = pl.RIGHT_SHOULDER.value, pl.RIGHT_HIP.value, pl.RIGHT_KNEE.value
    else:
        S,H,K = pl.LEFT_SHOULDER.value,  pl.LEFT_HIP.value,  pl.LEFT_KNEE.value
    s = (lms[S].x*w, lms[S].y*h); h_ = (lms[H].x*w, lms[H].y*h); k = (lms[K].x*w, lms[K].y*h)
    return s, h_, k

def pick_side_stable(prev_side, lms, minvis=MINVIS):
    """
    Lock a side unless the other is clearly better.
    Returns (side_used, visL, visR).
    """
    pl = mp_pose.PoseLandmark

    def vis(side):
        idx = [pl.LEFT_SHOULDER.value, pl.LEFT_HIP.value, pl.LEFT_KNEE.value] if side=="LEFT" \
              else [pl.RIGHT_SHOULDER.value, pl.RIGHT_HIP.value, pl.RIGHT_KNEE.value]
        return float(np.mean([lms[i].visibility for i in idx]))

    vL, vR = vis("LEFT"), vis("RIGHT")

    # If no previous side, choose the better visible one
    if prev_side is None:
        return ("LEFT" if vL >= vR else "RIGHT"), vL, vR

    # Keep current side unless it's poor and the other is clearly better
    cur_v = vL if prev_side=="LEFT" else vR
    oth_v = vR if prev_side=="LEFT" else vL
    MARGIN = 0.15
    if (cur_v < minvis) and (oth_v > cur_v + MARGIN):
        return ("RIGHT" if prev_side=="LEFT" else "LEFT"), vL, vR
    return prev_side, vL, vR

# --------- State ---------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open webcam. Check camera permissions or index.")

WIN = "Sit-up Counter (minimal)"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

reps = 0
state = "down"
angle_hist = deque(maxlen=SMOOTH)
top_enter_time = None
last_count_time = 0
side_used = None

t0 = time.time()
frames = 0

# --------- Loop ---------
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frames += 1
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = pose.process(rgb)

    angle_display = np.nan
    valid_pose = False
    visL = visR = 0.0

    if res.pose_landmarks:
        valid_pose = True
        lms = res.pose_landmarks.landmark

        # --- choose/lock side stably ---
        side_used, visL, visR = pick_side_stable(side_used, lms, MINVIS)
        side_vis = visL if side_used=="LEFT" else visR
        visibility_ok = side_vis >= MINVIS

        # points & angle on chosen side
        S,H,K = get_points((w, h), lms, side_used)
        ang = angle_abc(S, H, K)
        angle_hist.append(ang)
        sm_ang = float(np.mean(angle_hist))
        angle_display = sm_ang

        # draw pose
        mp_draw.draw_landmarks(
            frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=LM_STYLE, connection_drawing_spec=CONN_STYLE
        )

        # Rep FSM with hold + debounce
        now_ms = int((time.time() - t0) * 1000)
        if visibility_ok:
            # enter UP when sufficiently curled
            if state == "down" and sm_ang <= UP_TH:
                state = "up"
                top_enter_time = now_ms
            elif state == "up":
                held = (top_enter_time is not None) and (now_ms - top_enter_time >= TOP_HOLD_MS)
                # count when you return flat after a real hold at the top
                if sm_ang >= DOWN_TH and held:
                    state = "down"
                    if now_ms - last_count_time >= 300:  # debounce
                        reps += 1
                        last_count_time = now_ms
                        top_enter_time = None

        # mark hip
        cv2.circle(frame, (int(H[0]), int(H[1])), 8,
                   (0,255,0) if side_used=="RIGHT" else (0,170,255), 2)

    # HUD
    fps = frames / (time.time() - t0 + 1e-6)
    cv2.rectangle(frame, (10, 10), (700, 160), (0,0,0), -1)
    cv2.putText(frame, f"Reps: {reps}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    cv2.putText(frame, f"State: {state.upper()}   Side: {side_used if side_used else '...'}   FPS: {fps:.1f}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,255), 2)
    cv2.putText(frame, f"Angle: {angle_display:.1f} deg  (UP_TH {UP_TH} / DOWN_TH {DOWN_TH})",
                (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,255,180), 2)
    if res.pose_landmarks:
        cv2.putText(frame, f"Vis L:{visL:.2f}  R:{visR:.2f}  Using:{side_used}",
                    (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,255,180), 2)

    if not valid_pose:
        cv2.putText(frame, "No pose detected (show shoulder-hip-knee, side view).",
                    (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)

    cv2.imshow(WIN, frame)

    # Close by closing the window
    if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
        break
    cv2.waitKey(1)  # keep window responsive

# --------- Cleanup ---------
cap.release()
cv2.destroyAllWindows()
pose.close()
