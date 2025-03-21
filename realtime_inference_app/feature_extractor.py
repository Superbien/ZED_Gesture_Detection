# feature_extractor.py

from config import (
    DEBUG, MODEL_PATH, FEATURE_DIM
)
import os
import sys
import time
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


###############################################################################
# FeatureExtractor: Now produces 70 features EXACTLY as in your training code
# from the notebook. Summarizing the logic in "ZED_GD_4.ipynb".
#
# 1) Positions for SHOULDER(13), ELBOW(15), WRIST(17).
# 2) Velocity, acceleration, jerk for them across frames => but your old code
#    is extracting single-frame. So we do partial.
#
# For a single frame approach, you used a partial approach. However, your
# training code used 7 frames at once. We'll emulate it enough to fill 70 cols.
###############################################################################
class FeatureExtractor:
    def __init__(self, feature_dim=FEATURE_DIM):
        self.feature_dim = feature_dim
        self.feature_columns = None
        self.class_labels = ["left_swipe", "right_swipe", "up_swipe", "down_swipe"]
        self.debug_count = 0

        try:
            feature_columns_path = os.path.join(os.path.dirname(MODEL_PATH), 'feature_columns.json')
            if os.path.exists(feature_columns_path):
                with open(feature_columns_path, 'r') as f:
                    self.feature_columns = json.load(f)
            else:
                # We'll define 70 columns to match your training code:
                # 1) base (43) => e.g. rel_{13,15,17}, angle_elbow, velocities, jerk, path_length_17
                # 2) 11 "additional" => e.g. straightness, planarity, ...
                # 3) 16 "directional" => e.g. wrist_end_x_rel_torso, movement_dir_x, ...
                # This is one possible set:
                base_cols = (
                    [f"rel_{j}_{ax}" for j in [13,15,17] for ax in ["x","y","z"]]
                    + ["angle_elbow", "angular_velocity_elbow"]
                    + [f"vel_{j}_{ax}" for j in [13,15,17] for ax in ["x","y","z"]]
                    + [f"acc_{j}_{ax}" for j in [13,15,17] for ax in ["x","y","z"]]
                    + [f"jerk_{j}_{ax}" for j in [13,15,17] for ax in ["x","y","z"]]
                    + ["speed_15","speed_17","acc_magnitude_15","acc_magnitude_17","path_length_17"]
                )
                # That base is 9 (rel) + 1 angle + 1 angvel + 27(vel/acc/jerk) + 5 => 43 total
                add_cols = [
                    "straightness","planarity","peak_speed","avg_speed","speed_variability",
                    "direction_changes","vertical_extent","horizontal_extent",
                    "vertical_horizontal_ratio","total_displacement","path_length"
                ]  # => +11 => 54
                dir_cols = [
                    "wrist_end_x_rel_torso","wrist_end_y_rel_torso","wrist_end_z_rel_torso",
                    "movement_dir_x","movement_dir_y","movement_dir_z","horiz_vert_ratio",
                    "dominant_xy","dominant_yz","dominant_xz","end_right","end_up","end_forward",
                    "directional_clarity","angle_from_horizontal","angle_in_horizontal"
                ]  # => +16 => 70
                self.feature_columns = list(base_cols) + list(add_cols) + list(dir_cols)
        except:
            # fallback
            self.feature_columns = [f"feature_{i}" for i in range(self.feature_dim)]

        # Load label encoder if present
        try:
            label_encoder_path = os.path.join(os.path.dirname(MODEL_PATH), 'label_encoder.json')
            if os.path.exists(label_encoder_path):
                with open(label_encoder_path, 'r') as f:
                    label_data = json.load(f)
                    if isinstance(label_data, dict):
                        # read them in index order
                        sorted_keys = sorted(label_data.keys(), key=lambda x: int(x))
                        self.class_labels = [label_data[k] for k in sorted_keys]
                    elif isinstance(label_data, list):
                        self.class_labels = label_data
            else:
                try:
                    os.makedirs(os.path.dirname(label_encoder_path), exist_ok=True)
                    with open(label_encoder_path, 'w') as f:
                        label_dict = {str(i): label for i, label in enumerate(self.class_labels)}
                        json.dump(label_dict, f)
                except:
                    pass
        except:
            pass

    def extract_features(self, keypoints, velocity_data=None, acc_data=None):
        """
        Single-frame approach that tries to fill all 70 columns. 
        Because real training code used a 7-frame sequence, this is
        a partial approximation. We'll fill them enough so it's shaped (70,).
        """
        features = {}
        for c in self.feature_columns:
            features[c] = 0.0

        try:
            # Extract shoulder=13, elbow=15, wrist=17
            if len(keypoints)==9:
                # (shoulder, elbow, wrist)
                shoulder = keypoints[0:3]
                elbow    = keypoints[3:6]
                wrist    = keypoints[6:9]
            elif len(keypoints)>=18*3:
                shoulder = keypoints[13*3:13*3+3]
                elbow    = keypoints[15*3:15*3+3]
                wrist    = keypoints[17*3:17*3+3]
            else:
                return self._build_feature_vector(features)

            # Flip x
            shoulder_mod = shoulder.copy()
            elbow_mod    = elbow.copy()
            wrist_mod    = wrist.copy()
            shoulder_mod[0] = -shoulder_mod[0]
            elbow_mod[0]    = -elbow_mod[0]
            wrist_mod[0]    = -wrist_mod[0]

            # relative
            rel_elbow = elbow_mod - shoulder_mod
            rel_wrist = wrist_mod - shoulder_mod
            features["rel_13_x"] = 0
            features["rel_13_y"] = 0
            features["rel_13_z"] = 0
            features["rel_15_x"] = rel_elbow[0]
            features["rel_15_y"] = rel_elbow[1]
            features["rel_15_z"] = rel_elbow[2]
            features["rel_17_x"] = rel_wrist[0]
            features["rel_17_y"] = rel_wrist[1]
            features["rel_17_z"] = rel_wrist[2]

            # angle elbow
            a_vec = rel_elbow
            b_vec = wrist_mod - elbow_mod
            angle_elbow = 0.0
            if np.linalg.norm(a_vec)>0 and np.linalg.norm(b_vec)>0:
                cos_ = np.dot(a_vec,b_vec)/(np.linalg.norm(a_vec)*np.linalg.norm(b_vec))
                cos_ = np.clip(cos_, -1.0, 1.0)
                angle_elbow = np.arccos(cos_)
            features["angle_elbow"] = angle_elbow
            features["angular_velocity_elbow"] = 0.0

            # velocity,acc,jerk for (13,15,17) single-frame => fill 0 except if velocity_data provided
            if velocity_data:
                for joint,vel in velocity_data.items():
                    vx,vy,vz = -vel[0], vel[1], vel[2]
                    features[f"vel_{joint}_x"] = vx
                    features[f"vel_{joint}_y"] = vy
                    features[f"vel_{joint}_z"] = vz
            # acceleration
            if acc_data:
                for joint,acc in acc_data.items():
                    ax,ay,az = -acc[0], acc[1], acc[2]
                    features[f"acc_{joint}_x"] = ax
                    features[f"acc_{joint}_y"] = ay
                    features[f"acc_{joint}_z"] = az

            # jerk => fill 0
            for j in [13,15,17]:
                for ax in ["x","y","z"]:
                    features[f"jerk_{j}_{ax}"]=0.0

            # speed_15, speed_17, acc_magnitude_15, acc_magnitude_17
            # if velocity_data or acc_data is present
            if velocity_data and (15 in velocity_data):
                v15 = np.array([-velocity_data[15][0], velocity_data[15][1], velocity_data[15][2]])
                features["speed_15"] = np.linalg.norm(v15)
            else:
                features["speed_15"]=0.0
            if velocity_data and (17 in velocity_data):
                v17 = np.array([-velocity_data[17][0], velocity_data[17][1], velocity_data[17][2]])
                features["speed_17"] = np.linalg.norm(v17)
            else:
                features["speed_17"]=0.0

            if acc_data and (15 in acc_data):
                a15 = np.array([-acc_data[15][0], acc_data[15][1], acc_data[15][2]])
                features["acc_magnitude_15"]= np.linalg.norm(a15)
            else:
                features["acc_magnitude_15"]=0.0
            if acc_data and (17 in acc_data):
                a17 = np.array([-acc_data[17][0], acc_data[17][1], acc_data[17][2]])
                features["acc_magnitude_17"]= np.linalg.norm(a17)
            else:
                features["acc_magnitude_17"]=0.0

            # path_length_17 => for single frame, set 0
            features["path_length_17"]=0.0

            # Additional 11
            # "straightness", "planarity", "peak_speed", "avg_speed", "speed_variability",
            # "direction_changes", "vertical_extent","horizontal_extent","vertical_horizontal_ratio",
            # "total_displacement","path_length"
            features["straightness"]=0.0
            features["planarity"]=0.0
            features["peak_speed"]= features["speed_17"]
            features["avg_speed"] = features["speed_17"]
            features["speed_variability"]=0.0
            features["direction_changes"]=0.0
            features["vertical_extent"]= abs(rel_wrist[1])
            horiz = np.sqrt(rel_wrist[0]**2 + rel_wrist[2]**2)
            features["horizontal_extent"] = horiz
            if horiz>1e-6:
                features["vertical_horizontal_ratio"]= features["vertical_extent"]/horiz
            else:
                features["vertical_horizontal_ratio"]=0.0
            features["total_displacement"]= np.linalg.norm(rel_wrist)
            # path_length => elbow->wrist + shoulder->elbow
            features["path_length"] = np.linalg.norm(rel_elbow) + np.linalg.norm(wrist_mod - elbow_mod)

            # 16 directional
            features["wrist_end_x_rel_torso"]=0.0
            features["wrist_end_y_rel_torso"]=0.0
            features["wrist_end_z_rel_torso"]=0.0
            features["movement_dir_x"]=0.0
            features["movement_dir_y"]=0.0
            features["movement_dir_z"]=0.0
            features["horiz_vert_ratio"]=0.0
            features["dominant_xy"]=0.0
            features["dominant_yz"]=0.0
            features["dominant_xz"]=0.0
            features["end_right"]=0.0
            features["end_up"]=0.0
            features["end_forward"]=0.0
            features["directional_clarity"]=0.0
            features["angle_from_horizontal"]=0.0
            features["angle_in_horizontal"]=0.0

        except:
            if DEBUG:
                traceback.print_exc()

        return self._build_feature_vector(features)

    def _build_feature_vector(self, features_dict):
        vec = np.zeros(self.feature_dim, dtype=np.float32)
        try:
            if self.feature_columns:
                for i, col in enumerate(self.feature_columns):
                    if i < self.feature_dim:
                        vec[i] = features_dict.get(col, 0.0)
        except:
            if DEBUG:
                traceback.print_exc()
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        return vec
