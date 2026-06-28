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
    """Determine the best device to use (CUDA or CPU)."""
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def calculate_joint_angle(p_top, p_joint, p_bottom):
    """Calculates the angle at a joint given three 2D points."""
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

    processed_frames = [] 
    joint_data = []       

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
            bboxes = results[0].boxes.xyxy.cpu().numpy()

            for i, track_id in enumerate(track_ids):
                person_kpts = keypoints_data[i]
                bbox = bboxes[i]

                # --- 1. DRAW PERSON BOUNDING BOX ---
                bx1, by1, bx2, by2 = map(int, bbox)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 100, 0), 2)
                
                label = f"ID: {track_id}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (bx1, by1 - h - 10), (bx1 + w + 10, by1), (255, 100, 0), -1)
                cv2.putText(frame, label, (bx1 + 5, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # --- 2. DRAW SKELETON CONNECTIONS & JOINTS ---
                for partA, partB in Skeleton_connection:
                    sx1, sy1 = person_kpts[partA]
                    sx2, sy2 = person_kpts[partB]
                    if (sx1 > 0 and sy1 > 0) and (sx2 > 0 and sy2 > 0):
                        cv2.line(frame, (int(sx1), int(sy1)), (int(sx2), int(sy2)), (0, 255, 0), 2)

                for kpt in person_kpts:
                    kx, ky = kpt
                    if kx > 0 and ky > 0:
                        cv2.circle(frame, (int(kx), int(ky)), 4, (255, 0, 0), -1)

                # --- 3. DATA PROCESSING & MATH LAYER ---
                if track_id not in person_data_history:
                    person_data_history[track_id] = {
                        'left_knee_angles': [], 'right_knee_angles': [],
                        'left_elbow_angles': [], 'right_elbow_angles': [],
                        'left_knee_vels': [], 'right_knee_vels': [],
                        'left_elbow_vels': [], 'right_elbow_vels': [],
                        'timestamps': []
                    }

                left_knee_angle = calculate_joint_angle(person_kpts[L_HIP], person_kpts[L_KNEE], person_kpts[L_ANKLE])
                right_knee_angle = calculate_joint_angle(person_kpts[R_HIP], person_kpts[R_KNEE], person_kpts[R_ANKLE])
                left_elbow_angle = calculate_joint_angle(person_kpts[L_SHOULDER], person_kpts[L_ELBOW], person_kpts[L_WRIST])
                right_elbow_angle = calculate_joint_angle(person_kpts[R_SHOULDER], person_kpts[R_ELBOW], person_kpts[R_WRIST])

                person_data_history[track_id]['left_knee_angles'].append(left_knee_angle)
                person_data_history[track_id]['right_knee_angles'].append(right_knee_angle)
                person_data_history[track_id]['left_elbow_angles'].append(left_elbow_angle)
                person_data_history[track_id]['right_elbow_angles'].append(right_elbow_angle)
                person_data_history[track_id]['timestamps'].append(current_frame_idx / fps)

                # Window size limit for raw angles
                for key in ['left_knee_angles', 'right_knee_angles', 'left_elbow_angles', 'right_elbow_angles', 'timestamps']:
                    if len(person_data_history[track_id][key]) > WINDOW_SIZE:
                        person_data_history[track_id][key].pop(0)

                smoothed_metrics = {
                    'left_knee_angle': np.nan, 'right_knee_angle': np.nan,
                    'left_elbow_angle': np.nan, 'right_elbow_angle': np.nan,
                    'left_knee_vel': 0.0, 'right_knee_vel': 0.0,
                    'left_elbow_vel': 0.0, 'right_elbow_vel': 0.0
                }

                # Helper function for Savitzky-Golay angle filtering and velocity derivation
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

                # Extract smoothed values
                smoothed_metrics['left_knee_angle'], raw_lk_vel = smooth_and_derive('left_knee_angles')
                smoothed_metrics['right_knee_angle'], raw_rk_vel = smooth_and_derive('right_knee_angles')
                smoothed_metrics['left_elbow_angle'], raw_le_vel = smooth_and_derive('left_elbow_angles')
                smoothed_metrics['right_elbow_angle'], raw_re_vel = smooth_and_derive('right_elbow_angles')

                # --- CLIPPING AND SMOOTHING VELOCITIES ---
                # A. Append raw derived velocities to historical queues
                person_data_history[track_id]['left_knee_vels'].append(raw_lk_vel)
                person_data_history[track_id]['right_knee_vels'].append(raw_rk_vel)
                person_data_history[track_id]['left_elbow_vels'].append(raw_le_vel)
                person_data_history[track_id]['right_elbow_vels'].append(raw_re_vel)

                # Keep velocity tracking histories inside an active sliding-window length
                VELOCITY_SMOOTH_WINDOW = 5
                for v_key in ['left_knee_vels', 'right_knee_vels', 'left_elbow_vels', 'right_elbow_vels']:
                    if len(person_data_history[track_id][v_key]) > VELOCITY_SMOOTH_WINDOW:
                        person_data_history[track_id][v_key].pop(0)

                # B. Execute window averaging and clip extreme outliers to eradicate 4000 deg/s spikes
                MAX_HUMAN_VELOCITY = 1200.0  # Caps velocity variations within physiological boundaries
                
                smoothed_metrics['left_knee_vel'] = np.clip(np.mean(person_data_history[track_id]['left_knee_vels']), -MAX_HUMAN_VELOCITY, MAX_HUMAN_VELOCITY)
                smoothed_metrics['right_knee_vel'] = np.clip(np.mean(person_data_history[track_id]['right_knee_vels']), -MAX_HUMAN_VELOCITY, MAX_HUMAN_VELOCITY)
                smoothed_metrics['left_elbow_vel'] = np.clip(np.mean(person_data_history[track_id]['left_elbow_vels']), -MAX_HUMAN_VELOCITY, MAX_HUMAN_VELOCITY)
                smoothed_metrics['right_elbow_vel'] = np.clip(np.mean(person_data_history[track_id]['right_elbow_vels']), -MAX_HUMAN_VELOCITY, MAX_HUMAN_VELOCITY)

                # --- 4. BURN RIGHT-SIDE DATA BOX ---
                box_width = 250
                box_x1 = frame_width - box_width
                box_y1, box_y2 = 20, 290

                overlay = frame.copy()
                cv2.rectangle(overlay, (box_x1, box_y1), (frame_width - 10, box_y2), (40, 40, 40), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                cv2.rectangle(frame, (box_x1, box_y1), (frame_width - 10, box_y2), (0, 255, 255), 1)

                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.45
                color_white = (255, 255, 255)
                color_yellow = (0, 255, 255)
                
                current_ts = current_frame_idx / fps
                cv2.putText(frame, f"ID {track_id}", (box_x1 + 10, 40), font, 0.5, color_yellow, 1, cv2.LINE_AA)
                cv2.putText(frame, f"Time: {current_ts:.2f}s", (box_x1 + 10, 60), font, scale, color_white, 1, cv2.LINE_AA)
                
                lk = smoothed_metrics['left_knee_angle']
                rk = smoothed_metrics['right_knee_angle']
                cv2.putText(frame, f"L-Knee Ang: {'N/A' if np.isnan(lk) else f'{lk:.1f} deg'}", (box_x1 + 10, 95), font, scale, color_white, 1, cv2.LINE_AA)
                cv2.putText(frame, f"L-Knee Vel: {smoothed_metrics['left_knee_vel']:.1f} deg/s", (box_x1 + 10, 115), font, scale, color_white, 1, cv2.LINE_AA)
                cv2.putText(frame, f"R-Knee Ang: {'N/A' if np.isnan(rk) else f'{rk:.1f} deg'}", (box_x1 + 10, 145), font, scale, color_white, 1, cv2.LINE_AA)
                cv2.putText(frame, f"R-Knee Vel: {smoothed_metrics['right_knee_vel']:.1f} deg/s", (box_x1 + 10, 165), font, scale, color_white, 1, cv2.LINE_AA)
                
                le = smoothed_metrics['left_elbow_angle']
                re = smoothed_metrics['right_elbow_angle']
                cv2.putText(frame, f"L-Elbow Ang: {'N/A' if np.isnan(le) else f'{le:.1f} deg'}", (box_x1 + 10, 200), font, scale, color_white, 1, cv2.LINE_AA)
                cv2.putText(frame, f"L-Elbow Vel: {smoothed_metrics['left_elbow_vel']:.1f} deg/s", (box_x1 + 10, 220), font, scale, color_white, 1, cv2.LINE_AA)
                cv2.putText(frame, f"R-Elbow Ang: {'N/A' if np.isnan(re) else f'{re:.1f} deg'}", (box_x1 + 10, 250), font, scale, color_white, 1, cv2.LINE_AA)
                cv2.putText(frame, f"R-Elbow Vel: {smoothed_metrics['right_elbow_vel']:.1f} deg/s", (box_x1 + 10, 270), font, scale, color_white, 1, cv2.LINE_AA)

                joint_data.append({
                    'frame_idx': current_frame_idx,
                    'timestamp': current_ts,
                    'track_id': track_id,
                    'left_knee_angle': lk, 'left_knee_angular_velocity': smoothed_metrics['left_knee_vel'],
                    'right_knee_angle': rk, 'right_knee_angular_velocity': smoothed_metrics['right_knee_vel'],
                    'left_elbow_angle': le, 'left_elbow_angular_velocity': smoothed_metrics['left_elbow_vel'],
                    'right_elbow_angle': re, 'right_elbow_angular_velocity': smoothed_metrics['right_elbow_vel']
                })

        processed_frames.append(frame) 
        current_frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    return processed_frames, pd.DataFrame(joint_data)


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