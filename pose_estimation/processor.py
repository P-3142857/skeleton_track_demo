"""
Pose processing module for calculating joint angles and angular velocities from video.
"""

import numpy as np
import pandas as pd
import cv2
from scipy.signal import savgol_filter
from tqdm import tqdm
import torch


def get_device():
    """Determine the best device to use (CUDA or CPU).
    
    Returns:
        str: 'cuda' if available, otherwise 'cpu'
    """
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def calculate_joint_angle(p_top, p_joint, p_bottom):
    """Calculates the angle at a joint given three 2D points.
    
    Args:
        p_top: Top point coordinates (numpy array or list)
        p_joint: Joint point coordinates (numpy array or list)
        p_bottom: Bottom point coordinates (numpy array or list)
        
    Returns:
        float: Angle in degrees, or None if keypoints are invalid
    """
    # Ensure points are numpy arrays for vector operations
    p_top = np.array(p_top)
    p_joint = np.array(p_joint)
    p_bottom = np.array(p_bottom)

    v1 = p_top - p_joint      # Vector Joint -> Top
    v2 = p_bottom - p_joint   # Vector Joint -> Bottom

    # Handle zero vectors which can occur if keypoints are (0,0) or co-located
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Consider points invalid if their norms are too small (e.g., keypoints are very close or at origin)
    if norm_v1 < 1e-6 or norm_v2 < 1e-6:
        return None  # Cannot calculate angle if keypoints are invalid or co-located

    cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip to avoid floating point errors outside [-1, 1]

    return np.degrees(angle)  # Convert radians to degrees


def process_video(video_path, model, skeleton_connection=None, window_size=7, poly_order=3, 
                  l_hip=11, l_knee=13, l_ankle=15, r_hip=12, r_knee=14, r_ankle=16):
    """Process a video to extract pose keypoints and calculate joint angles.
    
    Args:
        video_path: Path to the video file
        model: YOLO model for pose estimation
        skeleton_connection: List of tuples defining skeleton connections (optional)
        window_size: Window size for Savitzky-Golay filter (must be odd)
        poly_order: Polynomial order for Savitzky-Golay filter
        l_hip: Left hip keypoint index
        l_knee: Left knee keypoint index
        l_ankle: Left ankle keypoint index
        r_hip: Right hip keypoint index
        r_knee: Right knee keypoint index
        r_ankle: Right ankle keypoint index
        
    Returns:
        tuple: (processed_frames, joint_data_df)
            - processed_frames: List of video frames
            - joint_data_df: DataFrame containing joint angle data for all frames and persons
    """
    cap = cv2.VideoCapture(video_path)

    # Get video properties for output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    processed_frames = []  # Store frames without overlays
    joint_data = []        # Store collected joint feature data

    person_data_history = {}
    current_frame_idx = 0

    # Get the best device (CUDA or CPU)
    device = get_device()

    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, verbose=False, device=device, tracker='bytetrack.yaml')

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            keypoints_data = results[0].keypoints.xy.cpu().numpy()

            for i, track_id in enumerate(track_ids):
                person_kpts = keypoints_data[i]

                if track_id not in person_data_history:
                    person_data_history[track_id] = {
                        'left_knee_angles': [],
                        'right_knee_angles': [],
                        'timestamps': []
                    }

                left_knee_angle = calculate_joint_angle(person_kpts[l_hip], person_kpts[l_knee], person_kpts[l_ankle])
                right_knee_angle = calculate_joint_angle(person_kpts[r_hip], person_kpts[r_knee], person_kpts[r_ankle])

                person_data_history[track_id]['left_knee_angles'].append(left_knee_angle)
                person_data_history[track_id]['right_knee_angles'].append(right_knee_angle)
                person_data_history[track_id]['timestamps'].append(current_frame_idx / fps)

                for key in ['left_knee_angles', 'right_knee_angles', 'timestamps']:
                    if len(person_data_history[track_id][key]) > window_size:
                        person_data_history[track_id][key].pop(0)

                smoothed_left_knee_angle = np.nan
                smoothed_right_knee_angle = np.nan
                left_knee_angular_velocity = 0.0
                right_knee_angular_velocity = 0.0

                # Process left knee
                current_left_angles = [
                    a for a in person_data_history[track_id]['left_knee_angles']
                    if a is not None and not np.isnan(a)
                ]
                current_timestamps_left = [
                    ts for ts, a in zip(person_data_history[track_id]['timestamps'],
                                        person_data_history[track_id]['left_knee_angles'])
                    if a is not None and not np.isnan(a)
                ]

                if len(current_left_angles) >= window_size and len(current_timestamps_left) >= window_size:
                    current_filter_window = min(len(current_left_angles), window_size)
                    if current_filter_window % 2 == 0:
                        current_filter_window -= 1
                    if current_filter_window >= poly_order + 1:
                        smoothed_left_angles = savgol_filter(current_left_angles, current_filter_window, poly_order)
                        smoothed_left_knee_angle = smoothed_left_angles[-1]
                        if len(smoothed_left_angles) >= 2 and len(current_timestamps_left) >= 2:
                            time_diff = current_timestamps_left[-1] - current_timestamps_left[-2]
                            if time_diff > 0:
                                left_knee_angular_velocity = (smoothed_left_angles[-1] - smoothed_left_angles[-2]) / time_diff

                # Process right knee
                current_right_angles = [
                    a for a in person_data_history[track_id]['right_knee_angles']
                    if a is not None and not np.isnan(a)
                ]
                current_timestamps_right = [
                    ts for ts, a in zip(person_data_history[track_id]['timestamps'],
                                        person_data_history[track_id]['right_knee_angles'])
                    if a is not None and not np.isnan(a)
                ]

                if len(current_right_angles) >= window_size and len(current_timestamps_right) >= window_size:
                    current_filter_window = min(len(current_right_angles), window_size)
                    if current_filter_window % 2 == 0:
                        current_filter_window -= 1
                    if current_filter_window >= poly_order + 1:
                        smoothed_right_angles = savgol_filter(current_right_angles, current_filter_window, poly_order)
                        smoothed_right_knee_angle = smoothed_right_angles[-1]
                        if len(smoothed_right_angles) >= 2 and len(current_timestamps_right) >= 2:
                            time_diff = current_timestamps_right[-1] - current_timestamps_right[-2]
                            if time_diff > 0:
                                right_knee_angular_velocity = (smoothed_right_angles[-1] - smoothed_right_angles[-2]) / time_diff

                # Collect joint data
                joint_data.append({
                    'frame_idx': current_frame_idx,
                    'timestamp': current_frame_idx / fps,
                    'track_id': track_id,
                    'left_knee_angle': smoothed_left_knee_angle,
                    'left_knee_angular_velocity': left_knee_angular_velocity,
                    'right_knee_angle': smoothed_right_knee_angle,
                    'right_knee_angular_velocity': right_knee_angular_velocity
                })

        processed_frames.append(frame)  # Append the original frame (without overlays)
        current_frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # Convert joint_data to a pandas DataFrame
    joint_data_df = pd.DataFrame(joint_data)

    return processed_frames, joint_data_df


# YOLO Keypoint Mapping Documentation
KEYPOINT_MAPPING = {
    0: "Nose",
    1: "Left Eye",
    2: "Right Eye",
    3: "Left Ear",
    4: "Right Ear",
    5: "Left Shoulder",
    6: "Right Shoulder",
    7: "Left Elbow",
    8: "Right Elbow",
    9: "Left Wrist",
    10: "Right Wrist",
    11: "Left Hip",
    12: "Right Hip",
    13: "Left Knee",
    14: "Right Knee",
    15: "Left Ankle",
    16: "Right Ankle"
}

SKELETON_CONNECTION = [
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]
