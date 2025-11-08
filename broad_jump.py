# broad_jump_webcam_calibrated.py
# Standing Broad Jump Estimation (Real-World Meters) with Auto Height Calibration
# pip install mediapipe opencv-python numpy matplotlib pandas

import time, collections, cv2, numpy as np, mediapipe as mp, matplotlib.pyplot as plt, pandas as pd, math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# PARAMETERS
SMOOTH_WINDOW = 5
MIN_KEYPOINT_CONF = 0.4
MIN_AIRTIME = 0.05
MAX_AIRTIME = 1.5
G = 9.81

# ---- Helper ----
def angle_between(p1, p2, p3):
    ax, ay = p1.x - p2.x, p1.y - p2.y
    bx, by = p3.x - p2.x, p3.y - p2.y
    dot = ax * bx + ay * by
    ma, mb = math.hypot(ax, ay), math.hypot(bx, by)
    if ma == 0 or mb == 0: return 0.0
    cosang = max(-1.0, min(1.0, dot / (ma * mb)))
    return math.degrees(math.acos(cosang))

# ---- Jump Estimator ----
class BroadJumpEstimator:
    def __init__(self, smooth_n=SMOOTH_WINDOW):
        self.takeoff_x = None
        self.landing_x = None
        self.takeoff_time = None
        self.in_air = False
        self.buffer_y = collections.deque(maxlen=smooth_n)

    def update(self, ankle_x, ankle_y, timestamp, ground_y):
        if ankle_y is None:
            return None
        self.buffer_y.append(ankle_y)
        avg_y = np.mean(self.buffer_y)

        # Detect takeoff
        if not self.in_air and avg_y < ground_y - 0.02:
            self.in_air = True
            self.takeoff_x = ankle_x
            self.takeoff_time = timestamp

        # Detect landing
        elif self.in_air and avg_y >= ground_y - 0.01:
            self.in_air = False
            self.landing_x = ankle_x
            distance_norm = abs(self.landing_x - self.takeoff_x)
            airtime = timestamp - self.takeoff_time
            return airtime, distance_norm
        return None

# ---- Main ----
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Unable to open webcam.")
        return

    athlete_height_m = float(input("Enter athlete height in meters (e.g., 1.75): "))

    plt.ion()
    fig, ax = plt.subplots()
    x_data, y_data = [], []
    (line,) = ax.plot([], [], "r-", label="Jump Distance (m)")
    ax.set_ylim(0, 3)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Jump Distance (m)")
    ax.legend()

    log_data = []
    ground_calibrated = False
    ground_ref_y = None
    pixel_to_meter = None
    p0 = None
    old_gray = None

    with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
        estimator = BroadJumpEstimator()
        last_jump = 0.0
        frame_idx = 0
        print("🟡 Stand still for calibration...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            now = time.time()
            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            display = frame.copy()

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                mp_drawing.draw_landmarks(display, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                left = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
                right = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
                left_conf, right_conf = left.visibility, right.visibility
                if left_conf >= MIN_KEYPOINT_CONF and right_conf >= MIN_KEYPOINT_CONF:
                    ankle_x = (left.x + right.x) / 2.0
                    ankle_y = (left.y + right.y) / 2.0
                elif left_conf >= MIN_KEYPOINT_CONF:
                    ankle_x, ankle_y = left.x, left.y
                elif right_conf >= MIN_KEYPOINT_CONF:
                    ankle_x, ankle_y = right.x, right.y
                else:
                    ankle_x, ankle_y = None, None

                # Calibration
                if not ground_calibrated and ankle_y is not None:
                    # derive pixel-to-meter scale using height
                    hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
                    shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    torso_px = abs(shoulder.y - hip.y) * h
                    torso_m = athlete_height_m * 0.35  # average torso ~35% of height
                    pixel_to_meter = torso_m / torso_px if torso_px > 0 else 0.003
                    ground_ref_y = ankle_y
                    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, maxCorners=50,
                                                 qualityLevel=0.3, minDistance=3, blockSize=7)
                    ground_calibrated = True
                    print(f"✅ Calibration complete (scale={pixel_to_meter:.5f} m/px)")

                elif ground_calibrated and ankle_x is not None:
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
                            old_gray = frame_gray.copy()
                            p0 = good_new.reshape(-1, 1, 2)
                            ankle_x = ankle_x - (dx / w)

                    jump_result = estimator.update(ankle_x, ankle_y, now, ground_ref_y)
                    if jump_result is not None:
                        airtime, distance_norm = jump_result
                        distance_px = distance_norm * w
                        distance_m = distance_px * pixel_to_meter
                        last_jump = distance_m
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        log_data.append([timestamp, airtime, distance_m])
                        print(f"[{timestamp}] Jump Distance: {distance_m:.2f} m | Airtime: {airtime:.3f}s")

                    cv2.circle(display, (int(ankle_x * w), int(ankle_y * h)), 8, (0, 255, 255), -1)
                    cv2.putText(display, "Feet (tracked)", (int(ankle_x * w) + 10, int(ankle_y * h)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            frame_idx += 1
            x_data.append(frame_idx)
            y_data.append(last_jump)
            if len(x_data) > 100:
                x_data, y_data = x_data[-100:], y_data[-100:]
            line.set_xdata(x_data)
            line.set_ydata(y_data)
            ax.set_xlim(max(0, frame_idx - 100), frame_idx)
            fig.canvas.draw()
            fig.canvas.flush_events()

            cv2.putText(display, f"Jump Distance: {last_jump:.2f} m", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("Standing Broad Jump (press q to quit)", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if log_data:
        df = pd.DataFrame(log_data, columns=["Timestamp", "Airtime (s)", "Jump Distance (m)"])
        df.to_csv("broad_jump_log_meters.csv", index=False)
        print(f"✅ Saved {len(df)} records to broad_jump_log_meters.csv")
    else:
        print("⚠️ No jumps detected. No log saved.")

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
