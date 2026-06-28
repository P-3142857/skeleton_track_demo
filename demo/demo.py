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

# Add parent directory to path to import skeleton module if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our updated process pipeline
from pose_estimation import process_video, KEYPOINT_MAPPING, SKELETON_CONNECTION, get_device

# Page configuration
st.set_page_config(layout='wide', page_title='Pose Estimation and Advanced Kinematic Analysis')

st.title('Video Pose Estimation and Advanced Kinematic Analysis')
st.write("""
Upload a video to analyze comprehensive multi-joint human analytics. 
The system detects full-body poses, tracks unique subjects, and embeds real-time angles 
and smoothed velocities directly into your video streams and downloadable reports.
""")

# Display device information
device = get_device()
device_color = "🟢" if device == "cuda" else "🔵"
st.info(f"{device_color} **Processing Device**: {device.upper()}")

# Display keypoint information
with st.expander("📋 Tracked Joints & Biomechanical Definitions"):
    st.write("""
The application monitors full skeletal kinematics from the standard 17 COCO keypoints:
- **Knee Angle**: Hip → Knee → Ankle
- **Elbow Angle**: Shoulder → Elbow → Wrist
- **Shoulder Angle**: Hip → Shoulder → Elbow *(New)*
- **Thigh Angle**: Cross-Hip → Active Hip → Knee *(New)*
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

# Web-optimized browser compilation function
def save_frames_to_mp4(frames, fps=30):
    if not frames:
        return None
    
    height, width, _ = frames[0].shape
    
    # 1. Save an intermediate raw file using OpenCV's native mp4v
    raw_tmp = tempfile.NamedTemporaryFile(delete=False, suffix='_raw.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(raw_tmp.name, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    out.release()
    
    # 2. Re-encode to H.264 using FFmpeg so web browsers can display it natively
    web_tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    try:
        import ffmpeg
        (
            ffmpeg
            .input(raw_tmp.name)
            .output(web_tmp.name, vcodec='libx264', acodec='aac', loglevel="quiet")
            .overwrite_output()
            .run()
        )
        try:
            os.remove(raw_tmp.name)
        except:
            pass
        return web_tmp.name
    except Exception as e:
        # Fallback to raw path if system-level ffmpeg wrapper errors out
        return raw_tmp.name

# File uploader
uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi', 'mkv'])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        video_path = tmp_file.name
    
    st.success(f"✅ Video uploaded successfully: {uploaded_file.name}")
    st.session_state['video_path'] = video_path
    
    # Process button
    if st.button("🚀 Process Video", key="process_button"):
        st.session_state['processed'] = False
        
        try:
            with st.spinner("Analyzing biomechanical vectors, smoothing velocities, and rendering dashboard box..."):
                orig_cap = cv2.VideoCapture(video_path)
                fps = orig_cap.get(cv2.CAP_PROP_FPS) or 30
                orig_cap.release()
                
                # Execute full processing pipeline
                processed_frames, joint_data_df = process_video(
                    video_path,
                    st.session_state['model'],
                    Skeleton_connection=SKELETON_CONNECTION,
                    WINDOW_SIZE=7,
                    POLY_ORDER=3
                )
                
                st.text("Compiling processed frames into web-optimized media files...")
                output_video_path = save_frames_to_mp4(processed_frames, fps=fps)
                
                st.session_state['processed_frames'] = processed_frames
                st.session_state['output_video_path'] = output_video_path
                st.session_state['joint_data_df'] = joint_data_df
                st.session_state['processed'] = True
            
            st.success("✅ Video processed successfully!")
        
        except Exception as e:
            st.error(f"❌ Error during execution: {str(e)}")
            st.session_state['processed'] = False
    
    # Display results layout if processing is successful
    if st.session_state['processed']:
        st.divider()
        
        tab1, tab2, tab3, tab4 = st.tabs(["🎥 Video Stream & Live Data", "📊 Data Sheets", "📈 Kinematic Charts", "ℹ️ Run Summary"])
        
        with tab1:
            st.subheader("Visual Overlay Playback")
            st.caption("Notice: Bounding boxes, skeletons, and the liveTelemetry block are baked directly into the video pixels.")
            if 'output_video_path' in st.session_state and st.session_state['output_video_path']:
                st.video(st.session_state['output_video_path'])
                
                with open(st.session_state['output_video_path'], 'rb') as f:
                    st.download_button(
                        label="📥 Download Telemetry Video File (MP4)",
                        data=f,
                        file_name="kinematics_telemetry_output.mp4",
                        mime="video/mp4"
                    )
            else:
                st.warning("Web compilation error. Fall back to alternative player configurations.")
        
        with tab2:
            st.subheader("Comprehensive Kinematic Dataset")
            st.dataframe(st.session_state['joint_data_df'], use_container_width=True, height=400)
            
            csv = st.session_state['joint_data_df'].to_csv(index=False)
            st.download_button(
                label="📥 Download Dataset Report (CSV)",
                data=csv,
                file_name="full_body_kinematics.csv",
                mime="text/csv"
            )
            
        with tab3:
            df = st.session_state['joint_data_df']
            
            # Sub-sample timestamps index for plotting
            chart_df = df.set_index('timestamp')
            
            # Row 1: Knees & Elbows
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🦵 Knee Angles & Velocities")
                st.line_chart(chart_df[['left_knee_angle', 'right_knee_angle', 'left_knee_angular_velocity', 'right_knee_angular_velocity']])
            with col2:
                st.markdown("#### 💪 Elbow Angles & Velocities")
                st.line_chart(chart_df[['left_elbow_angle', 'right_elbow_angle', 'left_elbow_angular_velocity', 'right_elbow_angular_velocity']])
                
            st.divider()
            
            # Row 2: Shoulder Angles & Thigh Angles
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("#### 📐 Shoulder Flexion / Extension")
                st.line_chart(chart_df[['left_shoulder_angle', 'right_shoulder_angle', 'left_shoulder_angular_velocity', 'right_shoulder_angular_velocity']])
            with col4:
                st.markdown("#### 🏃 Thigh Segment Workspace Angles")
                st.line_chart(chart_df[['left_thigh_angle', 'right_thigh_angle', 'left_thigh_angular_velocity', 'right_thigh_angular_velocity']])
        
        with tab4:
            st.subheader("Processing Profile Summary")
            df = st.session_state['joint_data_df']
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Frames", len(st.session_state['processed_frames']))
            with col2:
                st.metric("Logged Data Points", len(df))
            with col3:
                st.metric("Tracked Subject IDs", df['track_id'].nunique() if len(df) > 0 else 0)
            with col4:
                if len(df) > 0:
                    st.metric("Recording Length", f"{df['timestamp'].max():.2f}s")
else:
    st.info("👆 Please upload a video file to get started!")