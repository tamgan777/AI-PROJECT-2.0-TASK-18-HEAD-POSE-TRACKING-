import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
import json

# ===== HELPER FUNCTIONS =====
def display_text(frame, text, pos, size=0.8, thickness=2, color=(255, 255, 255)):
    """Display text on frame - reduces code duplication"""
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)

def detect_head_position(yaw, pitch, roll, yaw_limit=12, pitch_limit=12, roll_limit=10):
    """Detect head position based on thresholds - improved detection for all directions"""
    if abs(yaw) > yaw_limit:
        if yaw < 0:
            return "LOOKING LEFT"
        else:
            return "LOOKING RIGHT"
    elif abs(pitch) > pitch_limit:
        if pitch < 0:
            return "LOOKING DOWN"
        else:
            return "LOOKING UP"
    elif abs(roll) > roll_limit:
        if roll < 0:
            return "TILTED LEFT"
        else:
            return "TILTED RIGHT"
    return "NORMAL"

def smooth_angle(old_val, new_val, factor=0.75):
    """Apply exponential smoothing - eliminates repeated smoothing code"""
    return factor * old_val + (1 - factor) * new_val

def apply_alert_overlay(frame, w, h, alpha=0.12):
    """Apply red alert overlay - consolidates alert visualization"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

# ===== SETUP =====
cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

YAW_LIMIT = 12
PITCH_LIMIT = 12
ROLL_LIMIT = 10

smooth_yaw = smooth_pitch = smooth_roll = 0
tracking_enabled = True
prev_time = violation_count = last_beep_time = 0
debug_mode = False

# ===== MAIN LOOP =====
while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    status = "NORMAL"
    color = (0, 255, 0)

    if results.multi_face_landmarks and tracking_enabled:
        for face_landmarks in results.multi_face_landmarks:
            mesh_points = np.array([
                np.multiply([p.x, p.y], [w, h]).astype(int)
                for p in face_landmarks.landmark
            ])

            face_2d = []
            face_3d = []

            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [33, 263, 1, 61, 291, 199]:
                    face_2d.append([int(lm.x * w), int(lm.y * h)])
                    face_3d.append([int(lm.x * w), int(lm.y * h), lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            cam_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float32)
            dist_matrix = np.zeros((4, 1), dtype=np.float32)

            _, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            pitch = angles[0] * 360
            yaw = -angles[1] * 360
            roll = angles[2] * 360

            smooth_yaw = smooth_angle(smooth_yaw, yaw)
            smooth_pitch = smooth_angle(smooth_pitch, pitch)
            smooth_roll = smooth_angle(smooth_roll, roll)

            eye_center = (
                int((mesh_points[33][0] + mesh_points[263][0]) / 2),
                int((mesh_points[33][1] + mesh_points[263][1]) / 2)
            )

            cv2.line(frame, eye_center, (int(eye_center[0] + smooth_yaw * 5), int(eye_center[1] - smooth_pitch * 5)), (255, 0, 0), 3)
            cv2.circle(frame, eye_center, 5, (0, 255, 255), -1)

            current_time = time.time()
            status = detect_head_position(smooth_yaw, smooth_pitch, smooth_roll, YAW_LIMIT, PITCH_LIMIT, ROLL_LIMIT)

            if status != "NORMAL":
                color = (0, 0, 255)
                violation_count += 1
                display_text(frame, "ALERT: LOOK STRAIGHT", (80, 320), 1, 3, (0, 0, 255))
                frame = apply_alert_overlay(frame, w, h)

                if current_time - last_beep_time > 2:
                    winsound.Beep(1000, 300)
                    last_beep_time = current_time

            mp_draw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

            display_text(frame, f"Yaw: {int(smooth_yaw)}", (20, 50))
            display_text(frame, f"Pitch: {int(smooth_pitch)}", (20, 90))
            display_text(frame, f"Roll: {int(smooth_roll)}", (20, 130))
            display_text(frame, f"STATUS: {status}", (20, 180), 1, 3, color)
            display_text(frame, f"Violations: {violation_count}", (20, 230), color=(0, 255, 255))

            print(json.dumps({
                "status": status,
                "yaw": round(float(smooth_yaw), 2),
                "pitch": round(float(smooth_pitch), 2),
                "roll": round(float(smooth_roll), 2),
                "violations": violation_count
            }))

    else:
        display_text(frame, "NO FACE DETECTED", (20, 50), 1, 3, (0, 0, 255))

    current_time = time.time()
    fps = 1 / (current_time - prev_time + 1e-6)
    prev_time = current_time

    display_text(frame, f"FPS: {int(fps)}", (20, 280), color=(255, 255, 0))
    display_text(frame, "S = Stop | R = Resume | Q = Quit", (20, h - 20), 0.7, color=(255, 255, 255))

    if debug_mode:
        display_text(frame, f"DEBUG: Yaw={smooth_yaw:.1f} Pitch={smooth_pitch:.1f} Roll={smooth_roll:.1f}", (20, 310), 0.6, color=(200, 200, 255))

    cv2.imshow("Advanced Multi-Angle Head Tracking", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        tracking_enabled = False
    elif key == ord('r'):
        tracking_enabled = True
    elif key == ord('d'):
        debug_mode = not debug_mode

cap.release()
cv2.destroyAllWindows()
