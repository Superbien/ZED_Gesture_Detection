# config.py

import os
import sys
import time
import threading
import numpy as np
import pyzed.sl as sl
import tensorflow as tf
import json
import pygame
import cv2
from collections import deque
import traceback
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, font
from ttkthemes import ThemedTk
from PIL import Image, ImageTk
from scipy.spatial.distance import euclidean

DEBUG = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

###############################################################################
# ADJUST THESE TWO LINES TO MATCH YOUR NEW TRAINING:
###############################################################################
MODEL_PATH = r"D:\user\Documents\PLENG\Realtime_Inference\models\models_03\models_lstm\lstm_model_best"
WINDOW_SIZE = 7  # Was 6, now 7 frames
FEATURE_DIM = 70 # Was 54, now 70 features

CAMERA_FPS = 30

BODY_REGIONS = {
    "right_arm": [13, 15, 17],
    "left_arm": [12, 14, 16],
    "full_body": list(range(38))
}

READY_POSE_THRESHOLDS = {
    "arm_extension_ratio": 0.65,
    "wrist_pelvis_angle": 70,
    "ready_frames_required": 4
}
MOTION_THRESHOLDS = {
    "wrist_velocity": 0.15,
    "wrist_acceleration": 0.5,
    "detection_frames": 3,
    "follow_up_frames": 3,
    "significant_velocity_increase": 100.0
}
CLASSIFICATION_THRESHOLDS = {
    "confidence_threshold": 0.5,
    "diversity_penalty": 0.0,
    "window_consistency": 3
}

SKELETON_PAIRS_BODY_38 = [
    # Torso/spine
    (0, 1), (1, 2), (2, 3), (3, 4),

    # Head
    (4, 5),
    (5, 6),
    (5, 7),

    # Left Arm
    (4, 10),
    (10, 12),
    (12, 14),
    (14, 16),

    # Right Arm
    (4, 11),
    (11, 13),
    (13, 15),
    (15, 17),

    # Left Leg
    (0, 20),
    (20, 22),
    (22, 24),
    (24, 26),

    # Right Leg
    (0, 21),
    (21, 23),
    (23, 25),
    (25, 27),

    # Left Hand Fingers (from Left_Wrist)
    (16, 30),
    (16, 32),
    (16, 34),
    (16, 36),

    # Right Hand Fingers (from Right_Wrist)
    (17, 31),
    (17, 33),
    (17, 35),
    (17, 37),
]
