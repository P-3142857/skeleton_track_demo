"""
Pose estimation module containing functions for processing videos and calculating joint angles.
"""

from .processor import (
    calculate_joint_angle,
    process_video,
    KEYPOINT_MAPPING,
    SKELETON_CONNECTION
)

__all__ = [
    'calculate_joint_angle',
    'process_video',
    'KEYPOINT_MAPPING',
    'SKELETON_CONNECTION'
]
