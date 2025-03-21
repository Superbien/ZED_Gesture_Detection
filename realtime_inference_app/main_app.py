# main_app.py

from config import (
    DEBUG, WINDOW_SIZE, BODY_REGIONS, READY_POSE_THRESHOLDS, MODEL_PATH
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

from gesture_processor import GestureProcessor
from gesture_classifier import GestureClassifier
from inference_thread import InferenceThread


class GestureRecognitionApp:
    def __init__(self, root, model):
        self.root= root
        self.model= model
        self.processor= GestureProcessor()
        self.classifier= GestureClassifier(model)
        self.inference_thread= None
        self.sounds= {}
        self.sound_initialized= False
        self.frame_count= 0
        try:
            pygame.mixer.init()
            sf= {"ready":"ready.wav","success":"success.wav","error":"error.wav"}
            found= False
            for stype,fn in sf.items():
                try:
                    if os.path.exists(fn):
                        self.sounds[stype]= pygame.mixer.Sound(fn)
                        found= True
                except:
                    pass
            if not found:
                alt= ["beep.wav","ding.wav","alert.wav","notification.wav"]
                for a in alt:
                    if os.path.exists(a):
                        for stype in sf.keys():
                            self.sounds[stype]= pygame.mixer.Sound(a)
                        found= True
                        break
            self.sound_initialized= found
        except:
            self.sound_initialized= False
        self.setup_ui()
        self.start_inference()

    def setup_ui(self):
        self.root.title("Gesture Recognition System")
        self.root.geometry("1400x800")
        self.root.grid_columnconfigure(0, weight=3)
        self.root.grid_columnconfigure(1, weight=2)
        self.root.grid_rowconfigure(0, weight=3)
        self.root.grid_rowconfigure(1, weight=1)

        preview_frame= ttk.Frame(self.root)
        preview_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)

        self.preview_label= ttk.Label(preview_frame)
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        small_cam_width= 240
        small_cam_height= 180
        self.small_camera_frame= ttk.Frame(self.root)
        self.small_camera_frame.place(x=20,y=20,width=small_cam_width,height=small_cam_height)

        self.small_preview_label= ttk.Label(self.small_camera_frame)
        self.small_preview_label.pack(fill="both", expand=True)

        self.result_frame= ttk.Frame(preview_frame)
        self.result_frame.grid(row=0,column=0,sticky="nsew")
        self.result_frame.grid_rowconfigure(0, weight=1)
        self.result_frame.grid_columnconfigure(0, weight=1)

        fnt= font.Font(family="Arial",size=64,weight="bold")
        self.result_label= ttk.Label(self.result_frame,text="",font=fnt,background="#000000",foreground="white",anchor="center")
        self.result_label.grid(row=0, column=0, sticky="nsew")
        self.result_label.grid_remove()

        control_frame= ttk.Frame(self.root)
        control_frame.grid(row=0,column=1,padx=10,pady=10,sticky="nsew")

        self.region_var= tk.StringVar(value="right_arm")
        ttk.Label(control_frame,text="Body Region:").pack(pady=(10,2))
        ttk.Combobox(control_frame,textvariable=self.region_var,values=list(BODY_REGIONS.keys()),state="readonly").pack(pady=(0,10))

        ttk.Separator(control_frame,orient="horizontal").pack(fill="x",pady=10)

        stf= font.Font(family="Arial",size=20,weight="bold")
        self.big_state_label= ttk.Label(control_frame,text="WAITING",font=stf,foreground="orange",anchor="center")
        self.big_state_label.pack(fill="x",padx=5,pady=10)

        self.status_frame= ttk.LabelFrame(control_frame,text="System Status")
        self.status_frame.pack(fill="x",padx=5,pady=5)

        bf= ttk.Frame(self.status_frame)
        bf.pack(fill="x",padx=5,pady=2)
        ttk.Label(bf,text="Body Detection:").pack(side="left")
        self.body_status_label= ttk.Label(bf,text="No",foreground="red")
        self.body_status_label.pack(side="left",padx=5)

        rf= ttk.Frame(self.status_frame)
        rf.pack(fill="x",padx=5,pady=2)
        ttk.Label(rf,text="Ready Pose:").pack(side="left")
        self.ready_label= ttk.Label(rf,text="No",foreground="red")
        self.ready_label.pack(side="left",padx=5)

        mf= ttk.Frame(self.status_frame)
        mf.pack(fill="x",padx=5,pady=2)
        ttk.Label(mf,text="Motion:").pack(side="left")
        self.motion_label= ttk.Label(mf,text="No",foreground="red")
        self.motion_label.pack(side="left",padx=5)

        bf2= ttk.Frame(self.status_frame)
        bf2.pack(fill="x",padx=5,pady=2)
        ttk.Label(bf2,text="Captured Frames:").pack(side="left")
        self.buffer_label= ttk.Label(bf2,text=f"0/{WINDOW_SIZE}")
        self.buffer_label.pack(side="left",padx=5)

        self.capture_progress= ttk.Progressbar(self.status_frame,orient="horizontal",length=180,mode="determinate",maximum=10,value=0)
        self.capture_progress.pack(fill="x",padx=5,pady=5)

        ttk.Separator(control_frame,orient="horizontal").pack(fill="x",pady=10)

        mm= ttk.LabelFrame(control_frame,text="Arm Metrics")
        mm.pack(fill="x",padx=5,pady=5)

        efr= ttk.Frame(mm)
        efr.pack(fill="x",padx=5,pady=2)
        ttk.Label(efr,text="Extension Ratio:").pack(side="left")
        self.extension_label= ttk.Label(efr,text="0.00")
        self.extension_label.pack(side="left",padx=5)

        wfr= ttk.Frame(mm)
        wfr.pack(fill="x",padx=5,pady=2)
        ttk.Label(wfr,text="Wrist-Pelvis Angle:").pack(side="left")
        self.wpa_label= ttk.Label(wfr,text="0.0°")
        self.wpa_label.pack(side="left",padx=5)

        vfr= ttk.Frame(mm)
        vfr.pack(fill="x",padx=5,pady=2)
        ttk.Label(vfr,text="Wrist Velocity:").pack(side="left")
        self.vel_label= ttk.Label(vfr,text="0.000 m/s")
        self.vel_label.pack(side="left",padx=5)

        afr= ttk.Frame(mm)
        afr.pack(fill="x",padx=5,pady=2)
        ttk.Label(afr,text="Acceleration:").pack(side="left")
        self.acc_label= ttk.Label(afr,text="0.000 m/s²")
        self.acc_label.pack(side="left",padx=5)

        anfr= ttk.Frame(mm)
        anfr.pack(fill="x",padx=5,pady=2)
        ttk.Label(anfr,text="Torso-Arm Angle:").pack(side="left")
        self.torso_arm_label= ttk.Label(anfr,text="0.0°")
        self.torso_arm_label.pack(side="left",padx=5)

        ff= ttk.Frame(mm)
        ff.pack(fill="x",padx=5,pady=2)
        ttk.Label(ff,text="Forward Dot:").pack(side="left")
        self.forward_dot_label= ttk.Label(ff,text="0.000")
        self.forward_dot_label.pack(side="left",padx=5)

        ttk.Separator(control_frame,orient="horizontal").pack(fill="x",pady=10)

        bf3= ttk.Frame(control_frame)
        bf3.pack(pady=10)
        ttk.Button(bf3,text="Reset",command=self.reset_processor).pack(side="left",padx=5)
        ttk.Button(bf3,text="Clear Log",command=self.clear_log).pack(side="left",padx=5)
        ttk.Button(bf3,text="Quit",command=self.on_closing).pack(side="left",padx=5)

        log_frame= ttk.LabelFrame(self.root,text="Log")
        log_frame.grid(row=1,column=0,columnspan=2,sticky="nsew",padx=10,pady=10)
        self.log_console= scrolledtext.ScrolledText(log_frame,height=8)
        self.log_console.pack(fill="both",expand=True)

        # If you want a dedicated Skeleton View frame, as in the original code (commented out),
        # you can add it here.

    def update_ui(self, st):
        try:
            bd= self.processor.body_detected
            self.body_status_label.config(text="Yes" if bd else "No",foreground="green" if bd else "red")
            if not bd:
                self.big_state_label.config(text="WAITING",foreground="orange")
                self.ready_label.config(text="No",foreground="red")
                self.motion_label.config(text="No",foreground="red")
                self.extension_label.config(text="0.00")
                self.capture_progress["value"]=0
                return
            s= st.get("state","WAITING")
            c= {"WAITING":"orange","READY":"blue","CAPTURING":"green","CLASSIFYING":"purple","ERROR":"red"}
            self.big_state_label.config(text=s,foreground=c.get(s,"black"))
            r= st.get("ready_pose",False)
            self.ready_label.config(text="Yes" if r else "No",foreground="green" if r else "red")
            m= st.get("motion_detected",False)
            self.motion_label.config(text="Yes" if m else "No",foreground="green" if m else "red")
            e= st.get("arm_extension",0)
            self.extension_label.config(text=f"{e:.2f}")
            w= st.get("wrist_pelvis_angle",0)
            self.wpa_label.config(text=f"{w:.1f}°")
            v= st.get("velocity",0)
            self.vel_label.config(text=f"{v:.4f} m/s")
            a= st.get("acceleration",0)
            if a is not None:
                self.acc_label.config(text=f"{a:.4f} m/s²")
            ta= st.get("torso_arm_angle",0.0)
            self.torso_arm_label.config(text=f"{ta:.1f}°")
            fd= st.get("forward_dot",0.0)
            self.forward_dot_label.config(text=f"{fd:.3f}")
            bf= st.get("buffer_frames",0)
            self.buffer_label.config(text=f"{bf}/{WINDOW_SIZE}",foreground="green" if bf>0 else "black")
            if s=="CAPTURING":
                self.capture_progress["value"]= bf
            else:
                self.capture_progress["value"]= 0
        except:
            if DEBUG:
                traceback.print_exc()

    def update_camera_preview(self, frame):
        try:
            h,w= frame.shape[:2]
            disp= frame.copy()
            st= self.big_state_label.cget("text")
            isr= (self.ready_label.cget("text")=="Yes")
            ang= float(self.wpa_label.cget("text").replace("°",""))
            cv2.putText(disp,f"State: {st}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            ac= (0,255,0) if ang>= READY_POSE_THRESHOLDS["wrist_pelvis_angle"] else (0,0,255)
            cv2.putText(disp,f"Horizontal Angle: {ang:.1f}°",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,ac,2)
            rt= "READY POSE DETECTED" if isr else "Extend arm horizontally"
            rc= (0,255,0) if isr else (0,0,255)
            cv2.putText(disp,rt,(w//2-150,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,rc,2)
            cx,cy= w//2, h-100
            ll= 150
            cv2.line(disp,(cx,cy),(cx,cy-ll),(150,150,150),1)
            cv2.putText(disp,"0°",(cx+5,cy-ll),cv2.FONT_HERSHEY_SIMPLEX,0.5,(150,150,150),1)
            import math
            a45= math.radians(45)
            ex= int(cx + math.sin(a45)* ll)
            ey= int(cy - math.cos(a45)* ll)
            cv2.line(disp,(cx,cy),(ex,ey),(100,100,255),1)
            cv2.putText(disp,"45°",(ex+5,ey),cv2.FONT_HERSHEY_SIMPLEX,0.5,(100,100,255),1)
            a70= math.radians(70)
            ex2= int(cx+ math.sin(a70)* ll)
            ey2= int(cy- math.cos(a70)* ll)
            cv2.line(disp,(cx,cy),(ex2,ey2),(0,255,0),2)
            cv2.putText(disp,"70° (threshold)",(ex2-45,ey2-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
            cv2.line(disp,(cx,cy),(cx+ll,cy),(255,255,0),1)
            cv2.putText(disp,"90°",(cx+ll+5,cy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
            cv2.putText(disp,"Hold arm horizontally for Ready Pose",(w//2-200,h-30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            mainf= cv2.resize(disp,(0,0),fx=self.inference_thread.image_scale,fy=self.inference_thread.image_scale)
            from PIL import Image, ImageTk
            mim= Image.fromarray(cv2.cvtColor(mainf, cv2.COLOR_BGR2RGB))
            mtk= ImageTk.PhotoImage(image=mim)
            self.preview_label.config(image=mtk)
            self.preview_label.image= mtk
            sf= cv2.resize(frame,(0,0),fx=self.inference_thread.small_image_scale,fy=self.inference_thread.small_image_scale)
            sim= Image.fromarray(cv2.cvtColor(sf, cv2.COLOR_BGR2RGB))
            simtk= ImageTk.PhotoImage(image=sim)
            self.small_preview_label.config(image=simtk)
            self.small_preview_label.image= simtk
        except:
            if DEBUG:
                traceback.print_exc()

    def show_gesture_result(self, g, c, second=None):
        try:
            if g.lower()=="unclear":
                txt= "UNCLEAR GESTURE"
                col= "#FF6666"
            elif g.lower()=="error":
                txt= "ERROR"
                col= "#FF0000"
            else:
                txt= g.upper()
                col= "#66FF66"
            if c>0:
                txt+= f"\n{c*100:.1f}%"
            if second:
                txt+= f"\nPossibly {second.upper()}"
            self.result_label.config(text=txt,foreground=col)
            self.result_label.grid()
            self.root.after(2000,self.hide_gesture_result)
        except:
            pass

    def hide_gesture_result(self):
        self.result_label.grid_remove()

    def play_sound(self, st="success"):
        if self.sound_initialized and st in self.sounds:
            try:
                self.sounds[st].play()
            except:
                pass

    def reset_processor(self):
        self.processor._reset_state()
        self.log("Processor state reset")

    def get_selected_region(self):
        return self.region_var.get()

    def start_inference(self):
        self.inference_thread= InferenceThread(self.model,self.processor,self.classifier,self)
        self.inference_thread.daemon= True
        self.inference_thread.start()

    def log(self, message):
        self.log_console.insert("end",f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.log_console.see("end")

    def log_debug(self, message):
        if DEBUG:
            self.log(f"DEBUG: {message}")

    def clear_log(self):
        self.log_console.delete("1.0","end")
        self.log("Log cleared")

    def on_closing(self):
        if self.inference_thread:
            self.inference_thread.stop()
            self.inference_thread.join(timeout=1.0)
        self.root.destroy()


def load_model():
    try:
        m= tf.keras.models.load_model(MODEL_PATH)
        return m
    except Exception as e:
        messagebox.showerror("Error",f"Failed to load model: {e}")
        sys.exit(1)

if __name__=="__main__":
    root= ThemedTk(theme="equilux")
    model= load_model()
    app= GestureRecognitionApp(root, model)
    root.mainloop()
