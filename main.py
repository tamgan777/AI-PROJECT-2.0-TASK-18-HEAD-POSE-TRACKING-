import cv2
import mediapipe as mp
import numpy as np
import time
import json
import platform
YAW_LIMIT = 18
PITCH_LIMIT = 18
ROLL_LIMIT = 15
ALERT_THRESHOLD = 5
WARNING_THRESHOLD = 3
FONT = cv2.FONT_HERSHEY_SIMPLEX
def beep(freq=1000, duration=300):
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(freq, duration)
    else:
        try:
            import subprocess
            subprocess.Popen(["play", "-nq", "-t", "alsa", "synth", f"{duration/1000}", "sine", str(freq)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            print("\a", end="", flush=True)
def txt(frame, text, pos, size=0.45, thickness=1, color=(255, 255, 255)):
    cv2.putText(frame, text, pos, FONT, size, color, thickness, cv2.LINE_AA)
def smooth_angle(old_val, new_val, factor=0.75):
    return factor * old_val + (1 - factor) * new_val
def detect_head_position(yaw, pitch, roll):
    abs_y, abs_p, abs_r = abs(yaw), abs(pitch), abs(roll)
    severe_yaw = abs_y > YAW_LIMIT * 2.2
    severe_pitch = abs_p > PITCH_LIMIT * 2.2
    severe_roll = abs_r > ROLL_LIMIT * 2.2
    if abs_y > YAW_LIMIT and abs_p > PITCH_LIMIT:
        if yaw < 0 and pitch > 0:
            label = "UP-LEFT"
        elif yaw > 0 and pitch > 0:
            label = "UP-RIGHT"
        elif yaw < 0 and pitch < 0:
            label = "DOWN-LEFT"
        else:
            label = "DOWN-RIGHT"
        return label, 2 if (severe_yaw or severe_pitch) else 1
    if abs_y > YAW_LIMIT:
        return ("LOOKING LEFT" if yaw < 0 else "LOOKING RIGHT"), (2 if severe_yaw else 1)
    if abs_p > PITCH_LIMIT:
        return ("LOOKING DOWN" if pitch < 0 else "LOOKING UP"), (2 if severe_pitch else 1)
    if abs_r > ROLL_LIMIT:
        return ("TILTED LEFT" if roll < 0 else "TILTED RIGHT"), (2 if severe_roll else 1)
    if abs_p > PITCH_LIMIT * 1.5:
        return ("CHIN UP" if pitch > 0 else "NOSE DOWN"), 2
    return "NORMAL", 0
def draw_big_warning(frame, w, h, flash_on):
    overlay = frame.copy()
    if flash_on:
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 220), -1)
        frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
    cv2.rectangle(frame, (6, 6), (w - 6, h - 6), (0, 0, 255), 5)
    bx, by, bw, bh = w // 2 - 260, h // 2 - 70, 520, 140
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 0, 0), -1)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 0, 255), 3)
    cv2.putText(frame, "!! FINAL WARNING !!", (bx + 30, by + 55), FONT, 1.05, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, "LOOK STRAIGHT AT SCREEN", (bx + 30, by + 105), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame
def draw_removed_screen(frame, w, h, warning_count):
    frame = cv2.addWeighted(frame, 0.15, np.zeros_like(frame), 0.85, 0)
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 100, cy - 100), (cx + 100, cy + 100), (0, 0, 255), 8)
    cv2.line(frame, (cx + 100, cy - 100), (cx - 100, cy + 100), (0, 0, 255), 8)
    cv2.putText(frame, "ACCESS REVOKED", (cx - 175, cy - 130), FONT, 1.1, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, "Too many violations", (cx - 145, cy + 140), FONT, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Warnings issued: {warning_count}", (cx - 110, cy + 175), FONT, 0.55, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(frame, "Press Q to quit | R to reset", (cx - 165, h - 25), FONT, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
    return frame
cap = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
smooth_yaw = smooth_pitch = smooth_roll = 0.0
tracking_enabled = True
removed = False
prev_time = 0.0
debug_mode = False
violation_count = 0
big_warning_count = 0
flash_toggle = False
flash_timer = 0.0
in_big_warning = False
head_away = False
while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    current_time = time.time()
    fps = 1 / (current_time - prev_time + 1e-6)
    prev_time = current_time
    if removed:
        frame = draw_removed_screen(frame, w, h, big_warning_count)
        cv2.imshow("Head Tracking Monitor", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            removed = False
            violation_count = 0
            big_warning_count = 0
            in_big_warning = False
        continue
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    status = "NORMAL"
    severity = 0
    status_color = (0, 255, 0)
    if results.multi_face_landmarks and tracking_enabled:
        for face_landmarks in results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [w, h]).astype(int) for p in face_landmarks.landmark])
            face_2d, face_3d = [], []
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [33, 263, 1, 61, 291, 199]:
                    face_2d.append([int(lm.x * w), int(lm.y * h)])
                    face_3d.append([int(lm.x * w), int(lm.y * h), lm.z])
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            cam_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float32)
            _, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, np.zeros((4, 1), dtype=np.float32))
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(cv2.Rodrigues(rot_vec)[0])
            smooth_yaw = smooth_angle(smooth_yaw, -angles[1] * 360)
            smooth_pitch = smooth_angle(smooth_pitch, angles[0] * 360)
            smooth_roll = smooth_angle(smooth_roll, angles[2] * 360)
            status, severity = detect_head_position(smooth_yaw, smooth_pitch, smooth_roll)
            eye_center = (int((mesh_points[33][0] + mesh_points[263][0]) / 2), int((mesh_points[33][1] + mesh_points[263][1]) / 2))
            arrow_color = (0, 100, 255) if severity == 2 else (255, 180, 0) if severity == 1 else (0, 255, 100)
            cv2.arrowedLine(frame, eye_center, (int(eye_center[0] + smooth_yaw * 4), int(eye_center[1] - smooth_pitch * 4)), arrow_color, 2, tipLength=0.3)
            cv2.circle(frame, eye_center, 4, (0, 255, 255), -1)
            if not in_big_warning:
                if status != "NORMAL":
                    status_color = (0, 0, 255) if severity == 2 else (0, 165, 255)
                    if not head_away:
                        violation_count += 1
                        head_away = True
                        beep(1000 if severity == 1 else 1400, 300)
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                    frame = cv2.addWeighted(overlay, 0.10, frame, 0.90, 0)
                    txt(frame, "LOOK STRAIGHT!", (w // 2 - 90, h - 50), 0.55, 2, (0, 0, 255))
                else:
                    head_away = False
                if violation_count >= ALERT_THRESHOLD:
                    in_big_warning = True
                    big_warning_count += 1
                    flash_timer = current_time
                    violation_count = 0
                    head_away = False
                    beep(1600, 600)
                    if big_warning_count >= WARNING_THRESHOLD:
                        removed = True
            else:
                if status == "NORMAL" and (current_time - flash_timer) >= 2.0:
                    in_big_warning = False
                    head_away = False
            if not in_big_warning:
                mp_draw.draw_landmarks(frame, face_landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS,
                                       mp_draw.DrawingSpec(color=(80, 80, 80), thickness=1, circle_radius=1),
                                       mp_draw.DrawingSpec(color=(80, 80, 80), thickness=1))
            txt(frame, f"Yaw:{int(smooth_yaw)}  Pit:{int(smooth_pitch)}  Rol:{int(smooth_roll)}", (10, 20), 0.4, 1, (200, 200, 200))
            txt(frame, f"STATUS: {status}", (10, 42), 0.5, 1, status_color)
            txt(frame, f"Alerts: {violation_count}/{ALERT_THRESHOLD}  |  Warnings: {big_warning_count}/{WARNING_THRESHOLD}  ->  3 warnings = REMOVED", (10, 62), 0.36, 1, (0, 220, 220))
            txt(frame, f"FPS:{int(fps)}", (w - 70, 20), 0.4, 1, (200, 200, 0))
            if debug_mode:
                txt(frame, f"DBG Y={smooth_yaw:.1f} P={smooth_pitch:.1f} R={smooth_roll:.1f}", (10, 82), 0.35, 1, (180, 180, 255))
            print(json.dumps({"status": status, "severity": severity, "yaw": round(float(smooth_yaw), 2), "pitch": round(float(smooth_pitch), 2), "roll": round(float(smooth_roll), 2), "alerts": violation_count, "warnings": big_warning_count}))
    else:
        txt(frame, "NO FACE DETECTED", (20, 40), 0.6, 2, (0, 0, 255))
    if in_big_warning:
        flash_toggle = int((current_time - flash_timer) * 4) % 2 == 0
        frame = draw_big_warning(frame, w, h, flash_toggle)
    txt(frame, "S=Stop  R=Resume  D=Debug  Q=Quit", (10, h - 10), 0.35, 1, (140, 140, 140))
    cv2.imshow("Head Tracking Monitor", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        tracking_enabled = False
    elif key == ord('r'):
        tracking_enabled = True
        removed = False
        violation_count = 0
        big_warning_count = 0
        in_big_warning = False
        head_away = False
    elif key == ord('d'):
        debug_mode = not debug_mode
cap.release()
cv2.destroyAllWindows()