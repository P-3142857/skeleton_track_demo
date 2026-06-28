"""
Pose processing module for calculating joint angles and angular velocities from video.
"""

import numpy as np
import pandas as pd
import cv2
from scipy.signal import savgol_filter
from tqdm import tqdm
import torch

KEYPOINT_MAPPING = {
    0: "Nose", 1: "Left Eye", 2: "Right Eye", 3: "Left Ear", 4: "Right Ear",
    5: "Left Shoulder", 6: "Right Shoulder", 7: "Left Elbow", 8: "Right Elbow",
    9: "Left Wrist", 10: "Right Wrist", 11: "Left Hip", 12: "Right Hip",
    13: "Left Knee", 14: "Right Knee", 15: "Left Ankle", 16: "Right Ankle"
}

SKELETON_CONNECTION = [
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]

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
    p_top = np.array(p_top)
    p_joint = np.array(p_joint)
    p_bottom = np.array(p_bottom)

    v1 = p_top - p_joint      # Vector Joint -> Top
    v2 = p_bottom - p_joint   # Vector Joint -> Bottom

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 < 1e-6 or norm_v2 < 1e-6:
        return None  

    cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  

    return np.degrees(angle)  


def process_video(video_path, model, Skeleton_connection, WINDOW_SIZE, POLY_ORDER, 
                  L_HIP, L_KNEE, L_ANKLE, R_HIP, R_KNEE, R_ANKLE,
                  L_SHOULDER=5, L_ELBOW=7, L_WRIST=9, R_SHOULDER=6, R_ELBOW=8, R_WRIST=10):
    cap = cv2.VideoCapture(video_path)

    # Get video properties for output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    processed_frames = [] # Store frames with overlays
    joint_data = []       # Store collected joint feature data for arms and legs

    person_data_history = {}
    current_frame_idx = 0

    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break
        if get_device() == 'cuda':
            results = model.track(frame, persist=True, verbose=False, device='0', tracker='bytetrack.yaml')
        else:
            results = model.track(frame, persist=True, verbose=False, device='cpu', tracker='bytetrack.yaml')

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            keypoints_data = results[0].keypoints.xy.cpu().numpy()

            for i, track_id in enumerate(track_ids):
                person_kpts = keypoints_data[i]

                # --- 1. DRAW SKELETON CONNECTIONS & JOINTS ---
                # Draw the bones (lines)
                for partA, partB in Skeleton_connection:
                    x1, y1 = person_kpts[partA]
                    x2, y2 = person_kpts[partB]
                    if (x1 > 0 and y1 > 0) and (x2 > 0 and y2 > 0):
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # Draw the joint keypoints (circles)
                for kpt in person_kpts:
                    x, y = kpt
                    if x > 0 and y > 0:
                        cv2.circle(frame, (int(x), int(y)), 4, (255, 0, 0), -1)

                # Draw the Tracker ID over the person's head (using keypoint 0: Nose)
                nose_x, nose_y = person_kpts[0]
                if nose_x > 0 and nose_y > 0:
                    cv2.putText(frame, f"ID: {track_id}", (int(nose_x) - 20, int(nose_y) - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # --- 2. DATA PROCESSING & MATH LAYER ---
                if track_id not in person_data_history:
                    person_data_history[track_id] = {
                        'left_knee_angles': [], 'right_knee_angles': [],
                        'left_elbow_angles': [], 'right_elbow_angles': [],
                        'timestamps': []
                    }

                # Calculate raw metrics for legs and arms
                left_knee_angle = calculate_joint_angle(person_kpts[L_HIP], person_kpts[L_KNEE], person_kpts[L_ANKLE])
                right_knee_angle = calculate_joint_angle(person_kpts[R_HIP], person_kpts[R_KNEE], person_kpts[R_ANKLE])
                left_elbow_angle = calculate_joint_angle(person_kpts[L_SHOULDER], person_kpts[L_ELBOW], person_kpts[L_WRIST])
                right_elbow_angle = calculate_joint_angle(person_kpts[R_SHOULDER], person_kpts[R_ELBOW], person_kpts[R_WRIST])

                person_data_history[track_id]['left_knee_angles'].append(left_knee_angle)
                person_data_history[track_id]['right_knee_angles'].append(right_knee_angle)
                person_data_history[track_id]['left_elbow_angles'].append(left_elbow_angle)
                person_data_history[track_id]['right_elbow_angles'].append(right_elbow_angle)
                person_data_history[track_id]['timestamps'].append(current_frame_idx / fps)

                # Maintain history windows
                for key in ['left_knee_angles', 'right_knee_angles', 'left_elbow_angles', 'right_elbow_angles', 'timestamps']:
                    if len(person_data_history[track_id][key]) > WINDOW_SIZE:
                        person_data_history[track_id][key].pop(0)

                # Initialization defaults
                smoothed_metrics = {
                    'left_knee_angle': np.nan, 'right_knee_angle': np.nan,
                    'left_elbow_angle': np.nan, 'right_elbow_angle': np.nan,
                    'left_knee_vel': 0.0, 'right_knee_vel': 0.0,
                    'left_elbow_vel': 0.0, 'right_elbow_vel': 0.0
                }

                # Helper nested function to dry up the Savitzky-Golay filtering loop logic
                def smooth_and_derive(history_key):
                    angles = [a for a in person_data_history[track_id][history_key] if a is not None and not np.isnan(a)]
                    times = [ts for ts, a in zip(person_data_history[track_id]['timestamps'], person_data_history[track_id][history_key]) if a is not None and not np.isnan(a)]
                    
                    if len(angles) >= WINDOW_SIZE and len(times) >= WINDOW_SIZE:
                        filter_w = min(len(angles), WINDOW_SIZE)
                        if filter_w % 2 == 0: filter_w -= 1
                        if filter_w >= POLY_ORDER + 1:
                            smoothed = savgol_filter(angles, filter_w, POLY_ORDER)
                            vel = 0.0
                            if len(smoothed) >= 2 and len(times) >= 2:
                                dt = times[-1] - times[-2]
                                if dt > 0:
                                    vel = (smoothed[-1] - smoothed[-2]) / dt
                            return smoothed[-1], vel
                    return np.nan, 0.0

                # Execute smoothing and velocity extraction for all 4 tracked extremities
                smoothed_metrics['left_knee_angle'], smoothed_metrics['left_knee_vel'] = smooth_and_derive('left_knee_angles')
                smoothed_metrics['right_knee_angle'], smoothed_metrics['right_knee_vel'] = smooth_and_derive('right_knee_angles')
                smoothed_metrics['left_elbow_angle'], smoothed_metrics['left_elbow_vel'] = smooth_and_derive('left_elbow_angles')
                smoothed_metrics['right_elbow_angle'], smoothed_metrics['right_elbow_vel'] = smooth_and_derive('right_elbow_angles')

                # Append full tabular data profile
                joint_data.append({
                    'frame_idx': current_frame_idx,
                    'timestamp': current_frame_idx / fps,
                    'track_id': track_id,
                    'left_knee_angle': smoothed_metrics['left_knee_angle'],
                    'left_knee_angular_velocity': smoothed_metrics['left_knee_vel'],
                    'right_knee_angle': smoothed_metrics['right_knee_angle'],
                    'right_knee_angular_velocity': smoothed_metrics['right_knee_vel'],
                    'left_elbow_angle': smoothed_metrics['left_elbow_angle'],
                    'left_elbow_angular_velocity': smoothed_metrics['left_elbow_vel'],
                    'right_elbow_angle': smoothed_metrics['right_elbow_angle'],
                    'right_elbow_angular_velocity': smoothed_metrics['right_elbow_vel']
                })

        processed_frames.append(frame) 
        current_frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    joint_data_df = pd.DataFrame(joint_data)
    return processed_frames, joint_data_df


