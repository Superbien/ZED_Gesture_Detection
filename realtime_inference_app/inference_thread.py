# inference_thread.py

from config import (
    DEBUG, BODY_REGIONS, SKELETON_PAIRS_BODY_38
)
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

# We need the GestureProcessor reference:
from gesture_processor import GestureProcessor
# And the GestureClassifier reference for its usage in the thread (if needed):
from gesture_classifier import GestureClassifier

class InferenceThread(threading.Thread):
    def __init__(self, model, processor, classifier, app):
        super().__init__()
        self.model = model
        self.processor = processor
        self.classifier = classifier
        self.app = app
        self.running = True
        self.zed = None
        self.image_scale = 0.1
        self.small_image_scale = 0.1
        self.last_gesture_time = 0
        self.cooldown_time = 1.0
        self.zed = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD1080
        init.camera_fps = 30
        init.depth_mode = sl.DEPTH_MODE.ULTRA
        s = self.zed.open(init)
        if s != sl.ERROR_CODE.SUCCESS:
            self.app.log(f"Camera initialization failed: {s}")
            self.running = False
            return
        tparam = sl.PositionalTrackingParameters()
        st2 = self.zed.enable_positional_tracking(tparam)
        if st2 != sl.ERROR_CODE.SUCCESS:
            self.app.log(f"Positional tracking error: {st2}")
            self.running = False
            return
        bparam = sl.BodyTrackingParameters()
        bparam.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
        bparam.body_format = sl.BODY_FORMAT.BODY_38
        st3 = self.zed.enable_body_tracking(bparam)
        if st3 != sl.ERROR_CODE.SUCCESS:
            self.app.log(f"Body tracking error: {st3}")
            self.running = False
            return
        self.skeleton_image_scale = 0.25

    def run(self):
        if not self.running:
            return
        runtime = sl.RuntimeParameters()
        body_runtime = sl.BodyTrackingRuntimeParameters()
        bodies = sl.Bodies()
        image = sl.Mat()
        skeleton_mat = sl.Mat()
        frame_count = 0
        startup = 0
        self.app.log("Inference thread started")
        try:
            while self.running:
                err = self.zed.grab(runtime)
                if err == sl.ERROR_CODE.SUCCESS:
                    frame_count += 1
                    self.zed.retrieve_image(image, sl.VIEW.LEFT)
                    frm = image.get_data()[:,:,:3]
                    self.app.update_camera_preview(frm)
                    self.zed.retrieve_bodies(bodies, body_runtime)
                    region = self.app.get_selected_region()
                    kpts = self.extract_keypoints(bodies, region)
                    # self.draw_skeleton_view(bodies)  # (Commented in original)
                    if startup<8:
                        startup+=1
                        continue
                    ts = time.time()
                    r,st = self.processor.process_frame(kpts, ts)
                    self.app.update_ui(st)
                    if r:
                        e = r.get("event")
                        if e=="ready_pose_detected":
                            self.app.log("Ready pose detected")
                            self.app.play_sound("ready")
                        elif e=="frames_collected":
                            f = r.get("frames",[])
                            if f:
                                self.app.log(f"Collected {len(f)} frames for sliding window analysis")
                                ci,co = self.classifier.sliding_window_classify(f)
                                if ci is not None:
                                    name = self.processor.feature_extractor.class_labels[ci]
                                    self.app.log(f"SLIDING WINDOW RESULT: {name.upper()} ({co:.2f})")
                                    self.app.show_gesture_result(name,co)
                                    self.app.play_sound("success")
                                    self.last_gesture_time = time.time()
                                else:
                                    self.app.log("No consistent gesture detected in sliding windows")
                            else:
                                self.app.log("No frames collected for analysis")
                        elif e=="ready_pose_broken":
                            self.app.log("Ready pose broken")
                        elif e=="motion_detected":
                            self.app.log("Motion detected - capturing gesture")
                        elif e in ["capture_complete","capture_timeout"]:
                            f = r.get("frames",[])
                            self.app.log(f"Gesture captured ({len(f)} frames) - classifying...")
                            ct = time.time()
                            if ct - self.last_gesture_time>= self.cooldown_time:
                                ci,co = self.classifier.classify_gesture(f)
                                if ci is not None:
                                    gname = self.processor.feature_extractor.class_labels[ci]
                                    if co>=0.5:
                                        self.app.log(f"GESTURE RECOGNIZED: {gname.upper()} ({co:.2f})")
                                        self.app.show_gesture_result(gname,co)
                                        self.app.play_sound("success")
                                    else:
                                        self.app.log(f"Gesture unclear: {gname} (low confidence: {co:.2f})")
                                        self.app.show_gesture_result("UNCLEAR",co,gname)
                                        self.app.play_sound("error")
                                    self.last_gesture_time=ct
                                else:
                                    self.app.log("Classification failed")
                                    self.app.show_gesture_result("ERROR",0)
                time.sleep(0.001)
        except:
            if DEBUG:
                traceback.print_exc()
        finally:
            if self.zed:
                self.zed.close()
            self.app.log("Inference thread stopped")

    # def draw_skeleton_view(self, bodies):
    #     # (As in original code, commented out in your snippet)
    #     pass

    def extract_keypoints(self, bodies, region):
        try:
            idxs = BODY_REGIONS[region]
            kpts = np.zeros(len(idxs)*3,dtype=np.float32)
            fullk = np.zeros(38*3,dtype=np.float32)
            if bodies.is_new and bodies.body_list and len(bodies.body_list)>0:
                b = bodies.body_list[0]
                for i,j in enumerate(idxs):
                    if j<len(b.keypoint):
                        p = b.keypoint[j]
                        if not np.all(np.abs(p)<0.001):
                            kpts[i*3:(i+1)*3] = p
                for i2 in range(len(b.keypoint)):
                    p2 = b.keypoint[i2]
                    if not np.all(np.abs(p2)<0.001):
                        fullk[i2*3:(i2+1)*3] = p2
                self.processor.full_body_kpts = fullk.copy()
                self.app.frame_count = (self.app.frame_count+1) if hasattr(self.app,"frame_count") else 1
            return kpts
        except:
            if DEBUG:
                traceback.print_exc()
            return np.zeros(len(idxs)*3, dtype=np.float32)

    def stop(self):
        self.running = False
