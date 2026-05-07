# Head Pose Tracking and Attention Monitoring System - User Guide

## Quick Start

### Installation

1. **Install Python 3.11** from [https://www.python.org/downloads/release/python-3119/](https://www.python.org/downloads/release/python-3119/)

2. **Install required packages:**
```bash
py -3.11 -m pip install mediapipe==0.10.14 opencv-python numpy
```

### Running the System

```bash
py -3.11 main2.py
```

---

## System Features

### Detection Capabilities

The system detects and monitors:

| Movement | Detection |
|----------|-----------|
| Looking Left | Yaw < -18° |
| Looking Right | Yaw > +18° |
| Looking Up | Pitch < -15° |
| Looking Down | Pitch > +15° |
| Head Tilt Left | Roll < -12° |
| Head Tilt Right | Roll > +12° |

### Real-Time Monitoring

- **Facial Landmarks**: Detects 468 facial landmarks using MediaPipe
- **3D Pose Estimation**: Uses OpenCV solvePnP for accurate head orientation
- **Eye-Centered Tracking**: Stabilizes tracking using eye center point
- **Smoothing**: Exponential smoothing reduces noise (85% previous + 15% current)
- **FPS Display**: Real-time frames per second monitoring

### Alert System

When suspicious movement is detected:

1. **Visual Alert**: Red overlay appears on screen
2. **Text Warning**: "ALERT: LOOK STRAIGHT" message displays
3. **Audio Alert**: 1000Hz beep for 300ms (every 2 seconds)
4. **Violation Count**: Increments for each detection event

---

## Keyboard Controls

| Key | Function |
|-----|----------|
| **Q** | Quit the application |
| **S** | Stop tracking (freeze current state) |
| **R** | Resume tracking |
| **C** | Clear violation counter |

---

## Display Output

The system shows real-time information:

```
Yaw: -22        (Horizontal rotation in degrees)
Pitch: 5        (Vertical rotation in degrees)
Roll: 3         (Tilt rotation in degrees)
Status: LOOKING LEFT   (Current detection status)
Violations: 4   (Total violations detected)
FPS: 30         (Frames per second)
```

---

## Configuration

Edit `config.py` to customize:

```python
YAW_LIMIT = 18       # Adjust horizontal sensitivity
PITCH_LIMIT = 15     # Adjust vertical sensitivity
ROLL_LIMIT = 12      # Adjust tilt sensitivity
SMOOTHING_FACTOR = 0.85  # 0-1 (higher = more stable)
```

---

## JSON Output Data

The system logs movement data as JSON:

```json
{
  "status": "LOOKING LEFT",
  "yaw": -22.35,
  "pitch": 5.12,
  "roll": 3.41,
  "violations": 4,
  "fps": 30,
  "timestamp": 1715076123.456
}
```

---

## Technical Details

### Pose Estimation Algorithm

The system uses OpenCV's `solvePnP()` function:

```
R, t = solvePnP(3D_points, 2D_points, camera_matrix, distortion_matrix)
```

This maps 2D facial landmarks to 3D head rotation vectors.

### Camera Calibration

Camera matrix is automatically calculated:

```python
focal_length = image_width
camera_matrix = [
    [focal_length, 0, width/2],
    [0, focal_length, height/2],
    [0, 0, 1]
]
```

### Smoothing Formula

Exponential smoothing reduces jitter:

```
smooth_angle = 0.85 * previous_angle + 0.15 * current_angle
```

---

## Troubleshooting

### Issue: Camera not detected
- Check webcam is connected and not in use by another application
- Try disconnecting and reconnecting USB camera

### Issue: Poor tracking accuracy
- Ensure good lighting
- Clean webcam lens
- Face webcam directly
- Reduce `MIN_DETECTION_CONFIDENCE` in config.py if landmarks disappear

### Issue: Too many false alerts
- Increase thresholds: `YAW_LIMIT`, `PITCH_LIMIT`, `ROLL_LIMIT`
- Increase `SMOOTHING_FACTOR` (closer to 1.0)

### Issue: Slow performance (low FPS)
- Close other applications
- Reduce image resolution in camera settings
- Check CPU usage

---

## Files

| File | Purpose |
|------|---------|
| `main2.py` | Main tracking system |
| `config.py` | Configuration parameters |
| `main.py` | Basic face mesh detection demo |
| `head.py` | Simple face detection demo |

---

## Performance Metrics

Expected performance:

- **CPU**: 20-30% on dual-core processor
- **FPS**: 30-60 depending on resolution
- **Latency**: <100ms for detection
- **Accuracy**: 90%+ for clear frontal faces

---

## Safety Notes

- **Webcam Privacy**: The system only captures from webcam, no data stored by default
- **Screen Recording**: All alert overlays are visible on screen
- **Audio**: Beep volume depends on system volume settings

---

## Advanced Usage

### Modifying Detection Logic

Edit detection section in main2.py:

```python
looking_left = smooth_yaw < -YAW_LIMIT
looking_right = smooth_yaw > YAW_LIMIT
looking_up = smooth_pitch < -PITCH_LIMIT
looking_down = smooth_pitch > PITCH_LIMIT
tilted_left = smooth_roll < -ROLL_LIMIT
tilted_right = smooth_roll > ROLL_LIMIT
```

### Enabling JSON Logging to File

Add to main2.py after line ~395:

```python
import json
with open('pose_log.json', 'a') as f:
    json.dump(log_data, f)
    f.write('\n')
```

---

## Support

For issues or improvements, check:
- MediaPipe Documentation: https://developers.google.com/mediapipe
- OpenCV Documentation: https://opencv.org/
- Python Documentation: https://docs.python.org/3/

---

**Version**: 1.0  
**Last Updated**: May 2026  
**Python Version**: 3.11+
