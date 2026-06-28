"""
Streamlit Web Application for Video Pose Estimation and Joint Angle Analysis
"""
# ==========================================
# CRITICAL: THIS MUST BE LINE 1 & 2 OF YOUR APP
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ==========================================

import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import sys
import pandas as pd
import ffmpeg
# Add parent directory to path to import skeleton module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pose_estimation import process_video, KEYPOINT_MAPPING, SKELETON_CONNECTION, get_device

# Page configuration
st.set_page_config(layout='wide', page_title='Pose Estimation and Kinematic Analysis')

st.title('Video Pose Estimation and Kinematic Analysis')
st.write("""
Upload a video to analyze human poses, joint angles, and angular velocities for both arms and legs.
The application tracks skeletal positions, calculates smoothed knee and elbow angles, and maps their
velocities into an interactive real-time visual coaching dashboard.
""")

# Display device information
device = get_device()
device_color = "🟢" if device == "cuda" else "🔵"
st.info(f"{device_color} **Processing Device**: {device.upper()}")

# Display keypoint information
with st.expander("📋 YOLO Keypoint Mapping & Tracked Joints"):
    st.write("""
The application uses the 17 standard COCO keypoints from the YOLO pose architecture. For joint analytics, the following chains are extracted:
- **Left Arm**: Shoulder (5) → Elbow (7) → Wrist (9)
- **Right Arm**: Shoulder (6) → Elbow (8) → Wrist (10)
- **Left Leg**: Hip (11) → Knee (13) → Ankle (15)
- **Right Leg**: Hip (12) → Knee (14) → Ankle (16)
    """)
    keypoint_df = pd.DataFrame([
        {"Index": k, "Body Part": v} for k, v in sorted(KEYPOINT_MAPPING.items())
    ])
    st.table(keypoint_df)

# Initialize session state
if 'video_path' not in st.session_state:
    st.session_state['video_path'] = None
if 'processed' not in st.session_state:
    st.session_state['processed'] = False
if 'model' not in st.session_state:
    device = get_device()
    with st.spinner(f"Loading YOLO model on {device.upper()}..."):
        st.session_state['model'] = YOLO('yolov8n-pose.pt')
        st.session_state['device'] = device

# Helper function to convert frames into a web-compatible MP4 file
def save_frames_to_mp4(frames, fps=30):
    if not frames:
        return None
    
    height, width, _ = frames[0].shape
    
    # 1. Save an intermediate raw video file using standard 'mp4v' 
    # (This works natively without needing openh264-1.8.0-win64.dll)
    raw_tmp = tempfile.NamedTemporaryFile(delete=False, suffix='_raw.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(raw_tmp.name, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    out.release()
    
    # 2. Define the web-optimized output path for Streamlit
    web_tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    
    try:
        # Re-encode the video stream using FFmpeg to standard H.264 (libx264)
        (
            ffmpeg
            .input(raw_tmp.name)
            .output(web_tmp.name, vcodec='libx264', acodec='aac', loglevel="quiet")
            .overwrite_output()
            .run()
        )
        # Clean up the raw temporary file
        try:
            os.remove(raw_tmp.name)
        except:
            pass
            
        return web_tmp.name
        
    except Exception as e:
        # Fallback warning if system-level FFmpeg isn't discovered in the Windows path
        st.warning("⚠️ Web-optimization failed because FFmpeg isn't installed or configured in your system path. Showing raw fallback.")
        return raw_tmp.name

# File uploader
uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi', 'mkv'])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        video_path = tmp_file.name
    
    st.success(f"✅ Video uploaded successfully: {uploaded_file.name}")
    st.session_state['video_path'] = video_path
    
    # Process button
    if st.button("🚀 Process Video", key="process_button"):
        st.session_state['processed'] = False
        
        try:
            with st.spinner("Processing video... This tracks full skeletal kinematics (arms & legs)."):
                # Get native FPS from original video to match speeds
                orig_cap = cv2.VideoCapture(video_path)
                fps = orig_cap.get(cv2.CAP_PROP_FPS) or 30
                orig_cap.release()
                
                # Process the video (includes knees and newly added elbows)
                processed_frames, joint_data_df = process_video(
                    video_path,
                    st.session_state['model'],
                    Skeleton_connection=SKELETON_CONNECTION,
                    WINDOW_SIZE=7,
                    POLY_ORDER=3,
                    L_HIP=11, L_KNEE=13, L_ANKLE=15,
                    R_HIP=12, R_KNEE=14, R_ANKLE=16
                )
                
                # Compile frames into a video file Streamlit can play
                st.text("Compiling processed frames into video player file...")
                output_video_path = save_frames_to_mp4(processed_frames, fps=fps)
                
                st.session_state['processed_frames'] = processed_frames
                st.session_state['output_video_path'] = output_video_path
                st.session_state['joint_data_df'] = joint_data_df
                st.session_state['processed'] = True
            
            st.success("✅ Video processed successfully!")
        
        except Exception as e:
            st.error(f"❌ Error processing video: {str(e)}")
            st.session_state['processed'] = False
    
    # Display results if processing is complete
    if st.session_state['processed']:
        st.divider()
        
        tab1, tab2, tab3, tab4 = st.tabs(["🎥 Processed Video", "📊 Joint Data", "📈 Analytics", "ℹ️ Video Info"])
        
        with tab1:
            st.subheader("Visual Overlay Playback")
            if 'output_video_path' in st.session_state and st.session_state['output_video_path']:
                st.video(st.session_state['output_video_path'])
                
                with open(st.session_state['output_video_path'], 'rb') as f:
                    st.download_button(
                        label="📥 Download Processed Video (MP4)",
                        data=f,
                        file_name="processed_skeleton_pose.mp4",
                        mime="video/mp4"
                    )
            else:
                st.warning("Video playback file could not be generated.")
        
        with tab2:
            st.subheader("Joint Position Data (Angles & Angular Velocities)")
            st.dataframe(
                st.session_state['joint_data_df'],
                use_container_width=True,
                height=400
            )
            
            # Download button for CSV
            csv = st.session_state['joint_data_df'].to_csv(index=False)
            st.download_button(
                label="📥 Download Joint Data (CSV)",
                data=csv,
                file_name="full_kinematic_data.csv",
                mime="text/csv"
            )
        
        with tab3:
            df = st.session_state['joint_data_df']
            
            # --- LOWER BODY ANALYTICS ROW ---
            st.markdown("### 🦵 Lower Body Analytics (Knees)")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Left Knee Angle Statistics**")
                left_knee_data = df['left_knee_angle'].dropna()
                if len(left_knee_data) > 0:
                    st.columns(4)[0].metric("Mean", f"{left_knee_data.mean():.1f}°")
                    st.columns(4)[1].metric("Min", f"{left_knee_data.min():.1f}°")
                    st.columns(4)[2].metric("Max", f"{left_knee_data.max():.1f}°")
                    st.line_chart(df.set_index('timestamp')[['left_knee_angle', 'left_knee_angular_velocity']])
            
            with col2:
                st.write("**Right Knee Angle Statistics**")
                right_knee_data = df['right_knee_angle'].dropna()
                if len(right_knee_data) > 0:
                    st.columns(4)[0].metric("Mean", f"{right_knee_data.mean():.1f}°")
                    st.columns(4)[1].metric("Min", f"{right_knee_data.min():.1f}°")
                    st.columns(4)[2].metric("Max", f"{right_knee_data.max():.1f}°")
                    st.line_chart(df.set_index('timestamp')[['right_knee_angle', 'right_knee_angular_velocity']])
            
            st.divider()

            # --- UPPER BODY ANALYTICS ROW ---
            st.markdown("###  Upper Body Analytics (Elbows)")
            col3, col4 = st.columns(2)
            
            with col3:
                st.write("**Left Elbow Angle Statistics**")
                left_elbow_data = df['left_elbow_angle'].dropna()
                if len(left_elbow_data) > 0:
                    st.columns(4)[0].metric("Mean", f"{left_elbow_data.mean():.1f}°")
                    st.columns(4)[1].metric("Min", f"{left_elbow_data.min():.1f}°")
                    st.columns(4)[2].metric("Max", f"{left_elbow_data.max():.1f}°")
                    st.line_chart(df.set_index('timestamp')[['left_elbow_angle', 'left_elbow_angular_velocity']])
            
            with col4:
                st.write("**Right Elbow Angle Statistics**")
                right_elbow_data = df['right_elbow_data' if 'right_elbow_data' in df else 'right_elbow_angle'].dropna()
                if len(right_elbow_data) > 0:
                    st.columns(4)[0].metric("Mean", f"{right_elbow_data.mean():.1f}°")
                    st.columns(4)[1].metric("Min", f"{right_elbow_data.min():.1f}°")
                    st.columns(4)[2].metric("Max", f"{right_elbow_data.max():.1f}°")
                    st.line_chart(df.set_index('timestamp')[['right_elbow_angle', 'right_elbow_angular_velocity']])
        
        with tab4:
            st.subheader("Video Information")
            df = st.session_state['joint_data_df']
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Frames", len(st.session_state['processed_frames']))
            with col2:
                st.metric("Data Matrix Rows", len(df))
            with col3:
                unique_people = df['track_id'].nunique()
                st.metric("Unique People Tracked", unique_people)
            with col4:
                if len(df) > 0:
                    st.metric("Duration (seconds)", f"{df['timestamp'].max():.2f}s")

else:
    st.info("👆 Please upload a video file to get started!")

# Footer
st.divider()
st.markdown("""
### Core Pipeline:
1. **Multi-Person Tracking**: Evaluated via ByteTrack frame persistence.
2. **Signal Resampling**: Spatially smoothed using a Savitzky-Golay numerical filter window to bypass video tracking pixel noise.
3. **Kinematic Derivatives**: Instantaneous velocities are extracted from the angular frame step delta changes ($\Delta\theta/\Delta t$).
""")