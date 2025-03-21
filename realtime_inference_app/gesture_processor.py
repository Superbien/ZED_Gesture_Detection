# gesture_processor.py

from config import (
    DEBUG, BODY_REGIONS, READY_POSE_THRESHOLDS, MOTION_THRESHOLDS, 
    CLASSIFICATION_THRESHOLDS, WINDOW_SIZE
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

# Import your FeatureExtractor from feature_extractor.py
from feature_extractor import FeatureExtractor

class GestureProcessor:
    STATE_WAITING = "WAITING"
    STATE_READY = "READY"
    STATE_CAPTURING = "CAPTURING"
    STATE_CLASSIFYING = "CLASSIFYING"

    def __init__(self, smoothing_alpha=0.3):
        self.state = self.STATE_WAITING
        self.smoothing_alpha = smoothing_alpha
        self.last_valid_kpts = np.zeros(9, dtype=np.float32)
        self.last_kpts = None
        self.prev_kpts = None
        self.last_time = None
        self.prev_time = None
        self.feature_extractor = FeatureExtractor()

        self.stage_thresholds = {
            "ready_pose_frames": 5,
            "motion_detect_frames": 3,
            "max_capture_frames": 10,
            "min_velocity": 0.15,
            "velocity_spike_ratio": 3.0,
            "min_confidence": 0.7,
            "high_confidence": 0.85,
            "sliding_window_consistency": 3
        }

        self.stage_counters = {"ready_pose": 0, "motion_detect": 0, "gesture_capture": 0}
        self.body_detected = False
        self.no_body_counter = 0

        self.velocities = {}
        self.prev_velocities = {}
        self.current_acceleration = None
        self.velocity_history = deque(maxlen=10)
        self.velocity_values = deque(maxlen=10)
        self.acceleration_history = deque(maxlen=10)

        self.motion_detected = False
        self.recent_velocity_increase = False
        self.ready_pose_detected = False
        self.ready_pose_counter = 0
        self.arm_extension_ratio = 0
        self.wrist_pelvis_angle = 0
        self.ready_pose_timestamp = 0
        self.last_gesture_timestamp = 0
        self.gesture_cooldown = 1.0
        self.ready_frame_count = 0
        self.max_ready_frames = 10
        self.frame_buffer = []
        self.sliding_window_size = WINDOW_SIZE
        self.frame_count = 0
        self.torso_arm_angle = 0.0
        self.forward_dot = 0.0

    def _reset_state(self):
        self.state = self.STATE_WAITING
        for s in self.stage_counters:
            self.stage_counters[s] = 0
        self.ready_pose_detected = False
        self.ready_pose_counter = 0
        self.motion_detected = False
        self.frame_buffer.clear()
        self.velocity_history.clear()
        self.velocity_values.clear()
        self.acceleration_history.clear()
        self.ready_pose_timestamp = 0

    def _detect_ready_pose(self, current_kpts):
        if getattr(self, "full_body_kpts", None) is None or len(self.full_body_kpts) < 3:
            return False
        pelvis = self.full_body_kpts[0:3]
        shoulder = current_kpts[0:3]
        elbow = current_kpts[3:6]
        wrist = current_kpts[6:9]
        ua = elbow - shoulder
        fa = wrist - elbow
        d_sw = np.linalg.norm(wrist - shoulder)
        d_arm = np.linalg.norm(ua) + np.linalg.norm(fa)
        if d_arm > 1e-6:
            self.arm_extension_ratio = d_sw/d_arm
        else:
            self.arm_extension_ratio = 0.0
        torso_vec = shoulder - pelvis
        arm_vec = wrist - shoulder
        tl = np.linalg.norm(torso_vec)
        al = np.linalg.norm(arm_vec)
        if tl > 1e-6 and al > 1e-6:
            c = np.dot(torso_vec, arm_vec)/(tl*al)
            c = np.clip(c, -1.0, 1.0)
            deg = np.degrees(np.arccos(c))
        else:
            deg = 0.0
        angle_ok = (80 <= deg <= 130)
        fv = np.array([0,0,-1], dtype=np.float32)
        if al > 1e-6:
            av = arm_vec/al
            fdot = np.dot(av, fv)
        else:
            fdot = 0.0
        in_front = (fdot > 0.5)
        self.wrist_pelvis_angle = 0
        self.torso_arm_angle = deg
        self.forward_dot = fdot
        ready_ext = (self.arm_extension_ratio >= 0.65)
        return (ready_ext and angle_ok and in_front)

    def _detect_motion(self, velocities):
        wv = np.linalg.norm(velocities.get(17, np.zeros(3)))
        self.velocity_history.append(wv)
        self.velocity_values.append(wv)
        c = [
            wv > self.stage_thresholds["min_velocity"],
            len(self.velocity_values) >= 2 and self.velocity_values[-1] > self.velocity_values[-2]*self.stage_thresholds["velocity_spike_ratio"]
        ]
        return any(c)

    def process_frame(self, current_kpts, timestamp):
        try:
            self.frame_count += 1
            current_kpts = np.nan_to_num(current_kpts)
            has_body = not np.all(np.abs(current_kpts) < 0.001)
            if not has_body:
                self.no_body_counter += 1
                if self.no_body_counter > 3:
                    self._reset_state()
                    return None, {}
                return None, {}
            self.no_body_counter = 0
            self.body_detected = True
            if len(current_kpts) != 9:
                return None, {}
            smooth = 0.3*self.last_valid_kpts + 0.7*current_kpts
            self.last_valid_kpts = smooth
            velocities = {}
            accelerations = {}
            if self.last_kpts is not None and self.last_time is not None:
                dt = timestamp - self.last_time
                if dt>0:
                    for j,idx in enumerate([0,3,6]):
                        jid = [13,15,17][j]
                        v = (current_kpts[idx:idx+3]-self.last_kpts[idx:idx+3])/dt
                        velocities[jid] = v
                    self.motion_detected = self._detect_motion(velocities)
            is_ready_pose = self._detect_ready_pose(smooth)
            # EXTRACT 70 features for 1 frame
            feats = self.feature_extractor.extract_features(smooth, velocities, accelerations)
            result = self._update_state_machine(is_ready_pose, feats, timestamp)
            st = {
                "state": self.state,
                "ready_pose": is_ready_pose,
                "motion_detected": self.motion_detected,
                "arm_extension": self.arm_extension_ratio,
                "wrist_pelvis_angle": self.wrist_pelvis_angle,
                "torso_arm_angle": self.torso_arm_angle,
                "forward_dot": self.forward_dot,
                "buffer_frames": len(self.frame_buffer),
                "velocity": self.velocity_values[-1] if self.velocity_values else 0
            }
            self.last_kpts = current_kpts.copy()
            self.last_time = timestamp
            return result, st
        except:
            if DEBUG:
                traceback.print_exc()
            return None, {"state": "ERROR"}

    def _update_state_machine(self, is_ready_pose, feats, t):
        if self.state == self.STATE_WAITING:
            if is_ready_pose:
                self.stage_counters["ready_pose"] += 1
                if self.stage_counters["ready_pose"]>= self.stage_thresholds["ready_pose_frames"]:
                    if t - self.last_gesture_timestamp>= self.gesture_cooldown:
                        self.state = self.STATE_READY
                        self.ready_pose_timestamp = t
                        self.frame_buffer.clear()
                        return {"event":"ready_pose_detected"}
            else:
                self.stage_counters["ready_pose"] = 0
            return None
        elif self.state == self.STATE_READY:
            if t - self.ready_pose_timestamp<1.0:
                return None
            if self.motion_detected:
                self.stage_counters["motion_detect"] +=1
                if self.stage_counters["motion_detect"]>=self.stage_thresholds["motion_detect_frames"]:
                    self.state = self.STATE_CAPTURING
                    self.frame_buffer = [feats]
                    return {"event":"motion_detected"}
            else:
                self.stage_counters["motion_detect"]=0
            if t - self.ready_pose_timestamp>3.0:
                self._reset_state()
                return {"event":"ready_pose_timeout"}
            return None
        elif self.state==self.STATE_CAPTURING:
            self.frame_buffer.append(feats)
            if len(self.frame_buffer)<=self.stage_thresholds["max_capture_frames"]:
                pass
            if len(self.frame_buffer)== self.stage_thresholds["max_capture_frames"]:
                self.state= self.STATE_CLASSIFYING
                return {"event":"capture_complete","frames":self.frame_buffer}
            return None
        elif self.state==self.STATE_CLASSIFYING:
            self.last_gesture_timestamp= t
            self._reset_state()
            return None
        return None
