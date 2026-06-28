# Skeleton - Video Pose Estimation and Joint Angle Analysis

A Python package for detecting human poses in videos and calculating joint angles and angular velocities using YOLOv8.

## Current State of the Project

Have a little bit of hallucinations for darker video such as false positives and warping skeletons and potentially more due to this being tested in a few circumstances which only include video of a swift movement of upper body without significant lower body movements and the camera is set using tripods in all the cases. For 1 minute video using CPU would take around 5 minutes to complete the analysis (tested on AMD Ryzen 7 3750H laptop)

## Project Structure

```
skeleton/
├── pose_estimation/          # Core pose processing module
│   ├── __init__.py
│   └── processor.py          # Main processing functions
├── demo/
│   └── demo.py               # Streamlit web application
├── __init__.py
├── requirements.txt          # Project dependencies
└── README.md
```

## Installation

1. **Clone or navigate to the project directory**

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

Alternatively, install manually:
```bash
pip install ultralytics torch opencv-python scikit-learn mediapipe pandas scipy streamlit tqdm
```

## Quick Start

### Running the Streamlit Demo

```bash
cd demo
streamlit run demo.py
```

This will open a web browser at `http://localhost:8501` where you can:
- Upload a video file (MP4, MOV, AVI, or MKV)
- Process the video to detect poses and calculate joint angles
- View analytics and download results as CSV

### Using the Package Programmatically

```python
from ultralytics import YOLO
from pose_estimation import process_video

# Load the YOLO model
model = YOLO('yolov8n-pose.pt')

# Process a video
processed_frames, joint_data_df = process_video(
    video_path='your_video.mp4',
    model=model,
    window_size=7,
    poly_order=3
)

# Access the results
print(joint_data_df.head())
```

## Features

### Core Functionality

- **Pose Detection**: Uses YOLOv8n-pose for real-time human pose detection
- **Multi-Person Tracking**: ByteTrack for consistent tracking across frames
- **Joint Angle Calculation**: Calculates angles between joint connections (e.g., knee angles)
- **Angular Velocity**: Computes frame-by-frame angular velocity
- **Noise Smoothing**: Savitzky-Golay filter for smooth angle curves
- **Data Export**: Export results to CSV for further analysis

### YOLO Keypoints (17 total)

| Index | Body Part        | Index | Body Part       |
|-------|------------------|-------|-----------------|
| 0     | Nose             | 9     | Left Wrist      |
| 1     | Left Eye         | 10    | Right Wrist     |
| 2     | Right Eye        | 11    | Left Hip        |
| 3     | Left Ear         | 12    | Right Hip       |
| 4     | Right Ear        | 13    | Left Knee       |
| 5     | Left Shoulder    | 14    | Right Knee      |
| 6     | Right Shoulder   | 15    | Left Ankle      |
| 7     | Left Elbow       | 16    | Right Ankle     |
| 8     | Right Elbow      |       |                 |

## API Documentation

### `process_video()`

Processes a video file to extract pose keypoints and calculate joint angles.

**Parameters:**
- `video_path` (str): Path to the video file
- `model`: YOLO model for pose estimation
- `skeleton_connection` (list, optional): List of tuples defining skeleton connections
- `window_size` (int): Window size for Savitzky-Goyal filter (default: 7, must be odd)
- `poly_order` (int): Polynomial order for smoothing filter (default: 3)
- `l_hip`, `l_knee`, `l_ankle` (int): Left leg keypoint indices (default: 11, 13, 15)
- `r_hip`, `r_knee`, `r_ankle` (int): Right leg keypoint indices (default: 12, 14, 16)

**Returns:**
- `processed_frames` (list): List of video frames
- `joint_data_df` (DataFrame): Joint angle data with columns:
  - `frame_idx`: Frame number
  - `timestamp`: Time in seconds
  - `track_id`: Person ID
  - `left_knee_angle`: Left knee angle in degrees
  - `left_knee_angular_velocity`: Left knee angular velocity
  - `right_knee_angle`: Right knee angle in degrees
  - `right_knee_angular_velocity`: Right knee angular velocity

### `calculate_joint_angle()`

Calculates the angle at a joint given three 2D points.

**Parameters:**
- `p_top`: Top point coordinates (numpy array or list)
- `p_joint`: Joint point coordinates (numpy array or list)
- `p_bottom`: Bottom point coordinates (numpy array or list)

**Returns:**
- float: Angle in degrees, or None if keypoints are invalid

## Requirements

- Python 3.8+
- ultralytics
- torch
- opencv-python
- scikit-learn
- mediapipe
- pandas
- scipy
- streamlit
- tqdm

## Example Workflow

```python
# 1. Load the model
from ultralytics import YOLO
from pose_estimation import process_video
import pandas as pd

model = YOLO('yolov8n-pose.pt')

# 2. Process video
processed_frames, joint_data_df = process_video(
    video_path='sample_video.mp4',
    model=model
)

# 3. Analyze results
print("Average left knee angle:", joint_data_df['left_knee_angle'].mean())
print("Peak angular velocity (right):", joint_data_df['right_knee_angular_velocity'].max())

# 4. Export to CSV
joint_data_df.to_csv('pose_analysis.csv', index=False)
```

## Troubleshooting

### CUDA/GPU Issues
If you encounter GPU-related errors, the code will automatically fall back to CPU processing.

### Memory Issues with Large Videos
For very large videos, consider:
- Processing videos in segments
- Using a lower resolution input
- Adjusting the window_size parameter

### No Detections
- Ensure people are clearly visible in the video
- Try videos with good lighting and clear body poses

## License

This project is provided as-is for educational and research purposes.

## Contributing

Feel free to submit issues and enhancement requests!
