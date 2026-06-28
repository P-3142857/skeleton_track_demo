"""
Streamlit Web Application for Video Pose Estimation and Joint Angle Analysis
"""

import streamlit as st
import tempfile
import os
import sys

# Add parent directory to path to import skeleton module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pose_estimation import process_video, KEYPOINT_MAPPING
from ultralytics import YOLO
import cv2
import pandas as pd


# Page configuration
st.set_page_config(layout='wide', page_title='Pose Estimation and Angle Analysis')

st.title('Video Pose Estimation and Angle Analysis')
st.write("""
Upload a video to analyze human poses, joint angles, and angular velocities.
The application detects poses in the video, calculates knee joint angles and their 
angular velocities, and displays the results in an interactive dashboard.
""")

# Display keypoint information
with st.expander("📋 YOLO Keypoint Mapping (YOLOv8 Pose)"):
    st.write("""
In the YOLOv8 pose model, there are 17 keypoints representing different parts of the human body:
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
    with st.spinner("Loading YOLO model..."):
        st.session_state['model'] = YOLO('yolov8n-pose.pt')

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
            with st.spinner("Processing video... This may take a few minutes depending on video length."):
                # Process the video
                processed_frames, joint_data_df = process_video(
                    video_path,
                    st.session_state['model'],
                    window_size=7,
                    poly_order=3
                )
                
                st.session_state['processed_frames'] = processed_frames
                st.session_state['joint_data_df'] = joint_data_df
                st.session_state['processed'] = True
            
            st.success("✅ Video processed successfully!")
        
        except Exception as e:
            st.error(f"❌ Error processing video: {str(e)}")
            st.session_state['processed'] = False
    
    # Display results if processing is complete
    if st.session_state['processed']:
        st.divider()
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Joint Data", "📈 Analytics", "🎥 Video Info"])
        
        with tab1:
            st.subheader("Joint Angle Data")
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
                file_name="joint_angles_data.csv",
                mime="text/csv"
            )
        
        with tab2:
            st.subheader("Joint Angle Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Left knee statistics
                st.write("**Left Knee Angle Statistics**")
                left_knee_data = st.session_state['joint_data_df']['left_knee_angle'].dropna()
                if len(left_knee_data) > 0:
                    st.metric("Mean", f"{left_knee_data.mean():.2f}°")
                    st.metric("Std Dev", f"{left_knee_data.std():.2f}°")
                    st.metric("Min", f"{left_knee_data.min():.2f}°")
                    st.metric("Max", f"{left_knee_data.max():.2f}°")
                    
                    # Plot left knee angles
                    st.line_chart(
                        st.session_state['joint_data_df'].set_index('timestamp')['left_knee_angle']
                    )
            
            with col2:
                # Right knee statistics
                st.write("**Right Knee Angle Statistics**")
                right_knee_data = st.session_state['joint_data_df']['right_knee_angle'].dropna()
                if len(right_knee_data) > 0:
                    st.metric("Mean", f"{right_knee_data.mean():.2f}°")
                    st.metric("Std Dev", f"{right_knee_data.std():.2f}°")
                    st.metric("Min", f"{right_knee_data.min():.2f}°")
                    st.metric("Max", f"{right_knee_data.max():.2f}°")
                    
                    # Plot right knee angles
                    st.line_chart(
                        st.session_state['joint_data_df'].set_index('timestamp')['right_knee_angle']
                    )
            
            # Angular velocity
            st.write("**Angular Velocity Analysis**")
            col3, col4 = st.columns(2)
            
            with col3:
                st.write("Left Knee Angular Velocity")
                st.line_chart(
                    st.session_state['joint_data_df'].set_index('timestamp')['left_knee_angular_velocity']
                )
            
            with col4:
                st.write("Right Knee Angular Velocity")
                st.line_chart(
                    st.session_state['joint_data_df'].set_index('timestamp')['right_knee_angular_velocity']
                )
        
        with tab3:
            st.subheader("Video Information")
            
            df = st.session_state['joint_data_df']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Frames", len(st.session_state['processed_frames']))
            
            with col2:
                st.metric("Data Points", len(df))
            
            with col3:
                unique_people = df['track_id'].nunique()
                st.metric("Unique People", unique_people)
            
            with col4:
                if len(df) > 0:
                    st.metric("Duration (s)", f"{df['timestamp'].max():.2f}")

else:
    st.info("👆 Please upload a video file to get started!")

# Footer
st.divider()
st.markdown("""
---
### How it works:
1. **Upload** a video file (MP4, MOV, AVI, or MKV)
2. **Process** the video to detect poses and calculate joint angles
3. **Analyze** the extracted joint data with interactive visualizations
4. **Download** the results as CSV for further analysis

### Technical Details:
- **Pose Detection**: YOLOv8n-pose model
- **Tracking**: ByteTrack for multi-person tracking
- **Angle Calculation**: Using vector geometry (joint angles in degrees)
- **Smoothing**: Savitzky-Golay filter for noise reduction
- **Angular Velocity**: Calculated from frame-by-frame angle changes
""")
