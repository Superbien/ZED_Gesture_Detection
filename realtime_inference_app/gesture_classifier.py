# gesture_classifier.py

from config import (
    DEBUG, CLASSIFICATION_THRESHOLDS, WINDOW_SIZE
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

class GestureClassifier:
    def __init__(self, model, window_size=WINDOW_SIZE, class_labels=None):
        self.model = model
        self.window_size = window_size
        self.last_prediction = None
        self.last_confidence = 0
        self.class_labels = class_labels or ["left_swipe", "right_swipe", "up_swipe", "down_swipe"]

    def classify_gesture(self, frames):
        if not frames or len(frames)<1:
            return None,0
        try:
            # We want exactly 7 frames, each 70D => shape (7,70).
            if len(frames)<self.window_size:
                pad = [frames[-1]]*(self.window_size-len(frames))
                inp = frames+pad
            elif len(frames)>self.window_size:
                m = len(frames)//2
                s = max(0, m-self.window_size//2)
                inp = frames[s:s+self.window_size]
            else:
                inp = frames

            arr = np.array(inp)  # shape (7,70)
            arr = np.expand_dims(arr,0)  # shape (1,7,70)
            preds = self.model.predict(arr,verbose=0)[0]
            # if you want the direction reweighting from your original code,
            # we can skip or do partial
            c_preds = preds.copy()
            if np.sum(c_preds)>0:
                c_preds/=np.sum(c_preds)

            if self.last_prediction is not None:
                if np.argmax(c_preds)==self.last_prediction:
                    c_preds[self.last_prediction]*= CLASSIFICATION_THRESHOLDS["diversity_penalty"]
                    if np.sum(c_preds)>0:
                        c_preds/=np.sum(c_preds)

            idx = np.argmax(c_preds)
            conf = c_preds[idx]
            self.last_prediction= idx
            self.last_confidence= conf
            return idx, conf
        except:
            if DEBUG:
                traceback.print_exc()
            return None,0

    def _analyze_primary_direction(self, frames):
        try:
            vx,vy,vz= 0,0,0
            for fr in frames:
                # 'fr' is a 70D vector. Suppose velocity ~ indices [15..20], etc.
                pass
            return "unknown"
        except:
            if DEBUG:
                traceback.print_exc()
            return "unknown"

    def sliding_window_classify(self, frames, window_size=WINDOW_SIZE):
        # Same as original
        if len(frames)< window_size:
            return self.classify_gesture(frames)
        results = {"predictions":[],"confidences":[],"probabilities":[],"corrected_probabilities":[]}
        nwin= min(5,len(frames)-window_size+1)
        for i in range(nwin):
            w= frames[i:i+window_size]
            a= np.array(w)
            a= np.expand_dims(a,0)
            raw= self.model.predict(a,verbose=0)[0]
            corr= raw.copy()
            if np.sum(corr)>0:
                corr/=np.sum(corr)
            if self.last_prediction is not None:
                if np.argmax(corr)==self.last_prediction:
                    corr[self.last_prediction]*=CLASSIFICATION_THRESHOLDS["diversity_penalty"]
                    if np.sum(corr)>0:
                        corr/=np.sum(corr)
            cidx= np.argmax(corr)
            cconf= corr[cidx]
            results["predictions"].append(cidx)
            results["confidences"].append(cconf)
            results["probabilities"].append(raw)
            results["corrected_probabilities"].append(corr)
        if results["predictions"]:
            pc={}
            for p in results["predictions"]:
                pc[p]= pc.get(p,0)+1
            mp= max(pc, key= pc.get)
            rc= [c for p,c in zip(results["predictions"], results["confidences"]) if p==mp]
            avgc= np.mean(rc)
            if avgc>= CLASSIFICATION_THRESHOLDS["confidence_threshold"] and pc[mp]>= CLASSIFICATION_THRESHOLDS["window_consistency"]:
                return mp, avgc
        return None,0
