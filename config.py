"""
Configuration file for Head Pose Tracking System
Customize thresholds and parameters here
"""

# =========================================================
# DETECTION THRESHOLDS (in degrees)
# =========================================================

YAW_LIMIT = 18        # Horizontal rotation threshold (left/right)
PITCH_LIMIT = 15      # Vertical rotation threshold (up/down)
ROLL_LIMIT = 12       # Tilt rotation threshold (left/right tilt)

# =========================================================
# SMOOTHING PARAMETERS
# =========================================================

SMOOTHING_FACTOR = 0.85  # Exponential smoothing weight (0-1)
                          # Higher = more stable but slower response
                          # Lower = more responsive but noisier

# =========================================================
# ALERT PARAMETERS
# =========================================================

BEEP_INTERVAL = 2.0      # Seconds between alert beeps
ALERT_COLOR = (0, 0, 255)   # BGR format: Red
NORMAL_COLOR = (0, 255, 0)  # BGR format: Green
ALERT_OVERLAY_ALPHA = 0.12  # Transparency of red overlay (0-1)

# =========================================================
# MEDIAPIPE PARAMETERS
# =========================================================

MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# =========================================================
# DISPLAY PARAMETERS
# =========================================================

FONT = "cv2.FONT_HERSHEY_SIMPLEX"
FONT_SIZE = 0.8
TEXT_THICKNESS = 2
LINE_THICKNESS = 3

# Important facial landmarks for pose estimation
LANDMARK_IDS = [33, 263, 1, 61, 291, 199]  # Eyes, nose, mouth corners, chin
LEFT_EYE_ID = 33
RIGHT_EYE_ID = 263
NOSE_ID = 1
