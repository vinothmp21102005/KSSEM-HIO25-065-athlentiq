# jump_height_webcam.py
# Real-Time Jump Height Estimation with Dynamic Ground Tracking
# pip install mediapipe opencv-python numpy matplotlib pandas

import time
import collections
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import pandas as pd
import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# PARAMETERS
SMOOTH_WINDOW = 5
MIN_KEYPOINT_CONF = 0.4
MIN_AIRTIME = 0.05
MAX_AIRTIME = 1.5
G = 9.81

# ---------- Angle Helper ----------
def angle_between(p1, p2, p3):
    ax, ay = p1.x - p2.x, p1.y - p2.y
    bx, by = p3.x - p2.x, p3.y - p2.y
    dot = ax * bx + ay * by
    mag_a, mag_b = math.hypot(ax, ay), math.hypot(bx, by)
    if mag_a == 0 or mag_b == 0:
        return 0.0
    cosang = max(-1.0, min(1.0, dot / (mag_a * mag_b)))
    return math.degrees(math.acos(cosang))

# ---------- Standing Detection ----------
def is_user_standing(landmarks, min_conf=0.45, torso_tilt_thresh_deg=25,
                     knee_angle_thresh_deg=150, hip_y_thresh=0.75):
    mp = __import__('mediapipe').solutions.pose
    L_SHOULDER, R_SHOULDER = mp.PoseLandmark.LEFT_SHOULDER, mp.PoseLandmark.RIGHT_SHOULDER
    L_HIP, R_HIP = mp.PoseLandmark.LEFT_HIP, mp.PoseLandmark.RIGHT_HIP
    L_KNEE, R_KNEE = mp.PoseLandmark.LEFT_KNEE, mp.PoseLandmark.RIGHT_KNEE
    L_ANKLE, R_ANKLE = mp.PoseLandmark.LEFT_ANKLE, mp.PoseLandmark.RIGHT_ANKLE

    required = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]
    for lm_idx in required:
        if getattr(landmarks[lm_idx], "visibility", 0.0) < min_conf:
            return False

    left_knee_angle = angle_between(landmarks[L_HIP], landmarks[L_KNEE], landmarks[L_ANKLE])
    right_knee_angle = angle_between(landmarks[R_HIP], landmarks[R_KNEE], landmarks[R_ANKLE])

    shoulder_mid_y = (landmarks[L_SHOULDER].y + landmarks[R_SHOULDER].y) / 2.0
    hip_mid_y = (landmarks[L_HIP].y + landmarks[R_HIP].y) / 2.0
    shoulder_mid_x = (landmarks[L_SHOULDER].x + landmarks[R_SHOULDER].x) / 2.0
    hip_mid_x = (landmarks[L_HIP].x + landmarks[R_HIP].x) / 2.0

    vx, vy = shoulder_mid_x - hip_mid_x, shoulder_mid_y - hip_mid_y
    mag = math.hypot(vx, vy)
    if mag == 0:
        return False
    cosang = max(-1.0, min(1.0, (-vy) / mag))
    torso_tilt_deg = math.degrees(math.acos(cosang))

    hip_y = hip_mid_y
    knees_ok = (left_knee_angle >= knee_angle_thresh_deg) and (right_knee_angle >= knee_angle_thresh_deg)
    torso_ok = torso_tilt_deg <= torso_tilt_thresh_deg
    hip_ok = hip_y < hip_y_thresh

    return knees_ok and torso_ok and hip_ok

# ---------- Jump Estimator ----------
class JumpHeightEstimator:
    def __init__(self, smooth_n=SMOOTH_WINDOW):
        self.takeoff_time = None
        self.landing_time = None
        self.buffer = collections.deque(maxlen=smooth_n)
        self.prev_smoothed = None

    def update_from_rel_y(self, rel_y, timestamp):
        if rel_y is None:
            return None
        self.buffer.append(rel_y)
        smoothed = float(np.mean(self.buffer))
        if self.prev_smoothed is None:
            self.prev_smoothed = smoothed
            return None

        # detect takeoff
        if self.takeoff_time is None and smoothed > self.prev_smoothed + 0.02:
            self.takeoff_time = timestamp
        # detect landing
        elif self.takeoff_time is not None and self.landing_time is None and smoothed < self.prev_smoothed - 0.02:
            self.landing_time = timestamp
        self.prev_smoothed = smoothed

        if self.takeoff_time and self.landing_time:
            airtime = self.landing_time - self.takeoff_time
            if MIN_AIRTIME < airtime < MAX_AIRTIME:
                jump_h = G * (airtime / 2.0) ** 2
                self.reset()
                return airtime, jump_h
            self.reset()
        return None

    def reset(self):
        self.takeoff_time = None
        self.landing_time = None
        self.buffer.clear()
        self.prev_smoothed = None

# ---------- Main ----------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Unable to open webcam.")
        return

    plt.ion()
    fig, ax = plt.subplots()
    x_data, y_data = [], []
    (line,) = ax.plot([], [], "b-", label="ankle_y")
    ax.set_ylim(-0.2, 0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Relative Ankle Height (to ground)")
    ax.legend()

    log_data = []

    # --- Ground tracking variables ---
    ground_calibrated = False
    ground_ref_frames = []
    ground_ref_y = None
    p0 = None
    old_gray = None

    with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
        estimator = JumpHeightEstimator()
        last_jump = 0.0
        frame_idx = 0

        print("🟡 Stand still for 3 seconds to set ground reference...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            now = time.time()
            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            display = frame.copy()
            rel_y = None

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                mp_drawing.draw_landmarks(display, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                left = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
                right = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
                left_conf, right_conf = left.visibility, right.visibility

                if left_conf >= MIN_KEYPOINT_CONF and right_conf >= MIN_KEYPOINT_CONF:
                    norm_y = (left.y + right.y) / 2.0
                elif left_conf >= MIN_KEYPOINT_CONF:
                    norm_y = left.y
                elif right_conf >= MIN_KEYPOINT_CONF:
                    norm_y = right.y
                else:
                    norm_y = None

                if norm_y is not None:
                    ankle_px_x = int(w / 2)
                    ankle_px_y = int(norm_y * h)

                    # --- Calibration ---
                    if not ground_calibrated:
                        ground_ref_frames.append(norm_y)
                        if len(ground_ref_frames) >= 30:
                            ground_ref_y = np.mean(ground_ref_frames)
                            ground_calibrated = True
                            print("✅ Ground reference visually locked")
                            old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, maxCorners=50,
                                                         qualityLevel=0.3, minDistance=3, blockSize=7)
                    else:
                        # --- Optical flow tracking ---
                        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        if p0 is not None:
                            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None,
                                                                   winSize=(15, 15), maxLevel=2,
                                                                   criteria=(cv2.TERM_CRITERIA_EPS |
                                                                             cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                            if p1 is not None:
                                good_new = p1[st == 1]
                                good_old = p0[st == 1]
                                dx = np.mean(good_new[:, 0] - good_old[:, 0])
                                dy = np.mean(good_new[:, 1] - good_old[:, 1])
                                ankle_px_x = int(ankle_px_x + dx)
                                ankle_px_y = int(ankle_px_y + dy)
                                old_gray = frame_gray.copy()
                                p0 = good_new.reshape(-1, 1, 2)

                        rel_y = (ground_ref_y * h - ankle_px_y) / h

                        jump_result = estimator.update_from_rel_y(rel_y, now)
                        if jump_result is not None:
                            airtime, jump_h = jump_result
                            last_jump = jump_h
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                            log_data.append([timestamp, airtime, jump_h])
                            print(f"[{timestamp}] Jump: {jump_h:.3f} m (airtime {airtime:.3f}s)")

                        # Draw moving ground marker
                        cv2.circle(display, (ankle_px_x, ankle_px_y + 20), 8, (0, 255, 255), -1)
                        cv2.putText(display, "Ground (tracked)", (ankle_px_x + 10, ankle_px_y + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # --- Update Plot ---
            frame_idx += 1
            x_data.append(frame_idx)
            y_data.append(rel_y if rel_y is not None else np.nan)
            if len(x_data) > 100:
                x_data, y_data = x_data[-100:], y_data[-100:]
            line.set_xdata(x_data)
            line.set_ydata(y_data)
            ax.set_xlim(max(0, frame_idx - 100), frame_idx)
            fig.canvas.draw()
            fig.canvas.flush_events()

            cv2.putText(display, f"Jump Height: {last_jump:.2f} m", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("Jump Height (press q to quit)", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # --- Save jump logs ---
    if log_data:
        df = pd.DataFrame(log_data, columns=["Timestamp", "Airtime (s)", "Jump Height (m)"])
        df.to_csv("jump_log.csv", index=False)
        print(f"✅ Saved {len(df)} jump records to jump_log.csv")
    else:
        print("⚠️ No jumps detected. No log saved.")

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
