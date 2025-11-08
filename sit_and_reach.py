import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
import os

try:
    import joblib
except Exception:
    joblib = None

PIXEL_TO_CM = 0.15
START_MOV_TH = 8.0
RELEASE_DROP_PX = 10.0
SMOOTH = 7
MINVIS = 0.35
SIDE_SWITCH_MARGIN = 0.15
HUD_MARGIN = 8
MODEL_PATH = r"./sitreach_model_R1.pkl"
ML_THRESH = 0.50
TARGET_CM = 10.0

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

def draw_text_pill(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.7, thickness=2,
                   text_color=(240,240,240), bg_color=(0,0,0), alpha=0.55):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    pad = HUD_MARGIN
    x2, y2 = x + tw + 2*pad, y + th + 2*pad
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    cv2.putText(img, text, (x + pad, y + th + pad - 2), font, scale, text_color, thickness)

def side_score(lms, side):
    pl = mp_pose.PoseLandmark
    if side == "LEFT":
        idx = [pl.LEFT_SHOULDER.value, pl.LEFT_HIP.value, pl.LEFT_KNEE.value,
               pl.LEFT_FOOT_INDEX.value, pl.LEFT_INDEX.value]
    else:
        idx = [pl.RIGHT_SHOULDER.value, pl.RIGHT_HIP.value, pl.RIGHT_KNEE.value,
               pl.RIGHT_FOOT_INDEX.value, pl.RIGHT_INDEX.value]
    weights = np.array([1.0, 1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    vis = np.array([lms[i].visibility for i in idx], dtype=np.float32)
    return float(np.sum(vis * weights) / np.sum(weights))

def pick_side_stable(prev_side, lms, minvis=MINVIS, margin=SIDE_SWITCH_MARGIN):
    vL = side_score(lms, "LEFT")
    vR = side_score(lms, "RIGHT")
    if prev_side is None:
        return ("LEFT" if vL >= vR else "RIGHT"), vL, vR
    cur = vL if prev_side == "LEFT" else vR
    oth = vR if prev_side == "LEFT" else vL
    if (cur < minvis and oth > cur + margin) or (oth > cur + 2*margin):
        return ("RIGHT" if prev_side == "LEFT" else "LEFT"), vL, vR
    return prev_side, vL, vR

def hand_toe_x(hw, lms, side):
    w, h = hw
    pl = mp_pose.PoseLandmark
    if side == "RIGHT":
        hand = pl.RIGHT_INDEX.value
        toe  = pl.RIGHT_FOOT_INDEX.value
    else:
        hand = pl.LEFT_INDEX.value
        toe  = pl.LEFT_FOOT_INDEX.value
    return lms[hand].x * w, lms[toe].x * w, lms[hand].visibility, lms[toe].visibility

class HandSpeed:
    def __init__(self):
        self.prev_x = None
    def update(self, x):
        if self.prev_x is None:
            self.prev_x = x
            return 0.0
        v = abs(x - self.prev_x)
        self.prev_x = x
        return float(v)

class ReachModel:
    def __init__(self, path):
        self.enabled = False
        self.model = None
        if joblib is not None and path and os.path.exists(path):
            try:
                self.model = joblib.load(path)
                self.enabled = True
            except Exception:
                self.enabled = False
    def proba(self, feats):
        if not self.enabled:
            return 1.0
        x = np.asarray(feats, dtype=np.float32).reshape(1, -1)
        p = self.model.predict_proba(x)[0][1]
        return float(p)

def make_feats(hx, tx, hvis, tvis, side_vis, hand_speed, w):
    hand_x_n = hx / max(w, 1)
    toe_x_n = tx / max(w, 1)
    reach = toe_x_n - hand_x_n
    vis_min = float(min(hvis, tvis))
    vis_mean = float((hvis + tvis + side_vis) / 3.0)
    return [hand_x_n, toe_x_n, float(hvis), float(tvis), float(side_vis), float(hand_speed), float(reach), float(vis_min), float(vis_mean)]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open webcam. Check camera permissions or index.")

WIN = "Sit & Reach (minimal)"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 1280, 720)

state = "ready"
reach_hist = deque(maxlen=SMOOTH)
side_used = None
visL = visR = 0.0
best_cm = 0.0
max_reach_px = 0.0
prev_hand_x = None
move_count = 0
reach_count = 0
peak_ml = 0.0
last_tx = None

t0 = time.time()
frames = 0
hs = HandSpeed()
ml = ReachModel(MODEL_PATH)

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

    reach_px_disp = np.nan
    valid_pose = False

    if res.pose_landmarks:
        valid_pose = True
        lms = res.pose_landmarks.landmark
        side_used, visL, visR = pick_side_stable(side_used, lms, MINVIS, SIDE_SWITCH_MARGIN)
        side_vis = visL if side_used == "LEFT" else visR
        visibility_ok = side_vis >= MINVIS
        hx, tx, hvis, tvis = hand_toe_x((w, h), lms, side_used)

        if (hvis >= MINVIS*0.8) and (tvis >= MINVIS*0.8):
            reach_px = tx - hx
            reach_hist.append(reach_px)
            sm_reach_px = float(np.mean(reach_hist))
            reach_px_disp = sm_reach_px

            mp_draw.draw_landmarks(
                frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=LM_STYLE, connection_drawing_spec=CONN_STYLE
            )

            cv2.line(frame, (int(hx), int(h*0.12)), (int(hx), int(h*0.88)), (0, 255, 0), 2)
            cv2.line(frame, (int(tx), int(h*0.12)), (int(tx), int(h*0.88)), (0, 0, 255), 2)

            if tvis >= MINVIS*0.8:
                last_tx = tx
            toe_ref = last_tx if last_tx is not None else tx
            target_px = TARGET_CM / PIXEL_TO_CM
            target_x = int(min(max(toe_ref + target_px, 2), w - 3))
            cv2.line(frame, (target_x, int(h*0.12)), (target_x, int(h*0.88)), (255, 0, 0), 2)
            cv2.putText(frame, "Target", (target_x - 40, int(h*0.10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            if visibility_ok:
                hand_speed = hs.update(hx)
                feats = make_feats(hx, tx, hvis, tvis, side_vis, hand_speed, w)
                ml_proba = ml.proba(feats)

                if state == "ready":
                    if prev_hand_x is not None:
                        move_count = move_count + 1 if abs(hx - prev_hand_x) > START_MOV_TH else 0
                        if move_count >= 3:
                            state = "reaching"
                            max_reach_px = -1e9
                            peak_ml = 0.0
                            move_count = 0
                    prev_hand_x = hx
                elif state == "reaching":
                    if sm_reach_px > max_reach_px:
                        max_reach_px = sm_reach_px
                    if ml_proba > peak_ml:
                        peak_ml = ml_proba
                    if sm_reach_px < (max_reach_px - RELEASE_DROP_PX):
                        if peak_ml >= ML_THRESH:
                            val_cm = max_reach_px * PIXEL_TO_CM
                            best_cm = max(best_cm, val_cm)
                            reach_count += 1
                        state = "ready"
                        prev_hand_x = None
                        move_count = 0
                        peak_ml = 0.0
        else:
            reach_hist.clear()
            mp_draw.draw_landmarks(
                frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=LM_STYLE, connection_drawing_spec=CONN_STYLE
            )

    fps = frames / (time.time() - t0 + 1e-6)
    y0 = 12
    line_gap = 6
    lines = [
        f"State: {state.upper()}   Side: {side_used if side_used else '...'}   FPS: {fps:.1f}",
        f"Reach: {0.0 if np.isnan(reach_px_disp) else reach_px_disp*PIXEL_TO_CM:.1f} cm   Best: {best_cm:.1f} cm",
        f"Count: {reach_count}",
    ]
    if res.pose_landmarks:
        lines.append(f"Vis L:{visL:.2f}  R:{visR:.2f}  Using:{side_used}")

    for i, txt in enumerate(lines):
        draw_text_pill(frame, txt, (12, y0))
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        y0 += th + 2*HUD_MARGIN + line_gap

    if not valid_pose:
        draw_text_pill(frame, "No pose detected (side view, show hands & toes).",
                       (12, h - 12 - 24 - 2*HUD_MARGIN))

    cv2.imshow(WIN, frame)
    if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
        break
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
    