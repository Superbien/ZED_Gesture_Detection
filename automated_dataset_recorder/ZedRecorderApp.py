import os
import sys
import time
import threading
import json
from datetime import datetime
import pygame
from tkinter import *
from tkinter import ttk, messagebox, simpledialog
from ttkthemes import ThemedTk

import pyzed.sl as sl
import cv2
from PIL import Image, ImageTk

class ZedRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SUPERBIEN ZED Gesture Recorder")

        # Use a config file to store gestures and body parts
        self.load_config()

        # Load favicon if available
        self.icon_image = None
        if os.path.exists("system_assets/favicon.png"):
            self.icon_image = PhotoImage(file="system_assets/favicon.png")
            self.root.iconphoto(True, self.icon_image)

        # Initialize audio playback
        pygame.mixer.init()

        # Initialize style (dark theme / equilux)
        self.style = ttk.Style(self.root)
        self.style.theme_use("equilux")
        self.style.configure('TCombobox', fieldbackground='#333', background='#333')
        self.style.configure('TSpinbox', fieldbackground='#333', background='#333')

        # Camera objects
        self.camera = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD1080
        self.init_params.depth_mode = sl.DEPTH_MODE.NONE

        # Recording state
        self.recording = False
        self.current_count = 0
        self.total_records = 0
        self.current_gesture = ""
        self.current_body_part = ""

        # For checking IMU sensor
        self.sensors_data = sl.SensorsData()

        self.setup_ui()

    def load_config(self):
        """Load the config file, or create a default one if it doesn't exist."""
        self.config_file = "system_assets/config.json"
        default_config = {
            "body_areas": ["Arm", "Hand", "Leg"],
            "gestures": ["Swipe Up", "Swipe Down", "Circle"]
        }
        if not os.path.exists(self.config_file):
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f)
        with open(self.config_file) as f:
            self.config = json.load(f)

    def save_config(self):
        """Save current config to disk."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f)

    def setup_ui(self):
        """Sets up the main UI elements."""
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.grid(row=0, column=0, sticky=(N, S, E, W))

        # Dark theme colors 
        bg_color = '#333'
        fg_color = 'white'
        main_frame.configure(style='TFrame')
        self.style.configure('TFrame', background=bg_color)
        self.style.configure('TLabel', background=bg_color, foreground=fg_color)
        self.style.configure('TButton', background='#555', foreground=fg_color)
        self.style.map('TButton', background=[('active', '#666')])

        # Body Side
        ttk.Label(main_frame, text="Body Part:").grid(row=0, column=0, sticky=W)
        self.body_side = ttk.Combobox(main_frame, values=["Left", "Right"], state="normal")
        self.body_side.grid(row=0, column=1, sticky=EW)
        self.body_side.set("Left")

        # Body Area
        ttk.Label(main_frame, text="Body Area:").grid(row=1, column=0, sticky=W)
        self.body_area = ttk.Combobox(main_frame, values=self.config["body_areas"] + ["Other"])
        self.body_area.grid(row=1, column=1, sticky=EW)
        self.body_area.set("Arm")
        self.body_area.bind("<<ComboboxSelected>>", self.handle_body_area_select)

        # Gesture
        ttk.Label(main_frame, text="Gesture:").grid(row=2, column=0, sticky=W)
        self.gesture_type = ttk.Combobox(main_frame, values=self.config["gestures"] + ["Other"])
        self.gesture_type.grid(row=2, column=1, sticky=EW)
        self.gesture_type.set("Swipe Up")
        self.gesture_type.bind("<<ComboboxSelected>>", self.handle_gesture_select)

        # Number of Records
        ttk.Label(main_frame, text="Number of Records:").grid(row=3, column=0, sticky=W)
        self.record_count = Spinbox(main_frame, from_=1, to=1000)
        self.record_count.grid(row=3, column=1, sticky=EW)

        # Start Button
        self.start_btn = ttk.Button(main_frame, text="Start Recording", command=self.start_recording)
        self.start_btn.grid(row=4, column=0, columnspan=2, pady=20)

        # Status Label
        self.status_label = ttk.Label(main_frame, text="Ready")
        self.status_label.grid(row=5, column=0, columnspan=2)

        # Grid config so second column expands if window is resized
        main_frame.columnconfigure(1, weight=1)

    def handle_body_area_select(self, event):
        """If 'Other' is selected for body area, prompt for a new one and update config."""
        if self.body_area.get() == "Other":
            new_area = simpledialog.askstring("New Body Area", "Enter new body area name:")
            if new_area:
                self.config["body_areas"].append(new_area)
                self.save_config()
                self.body_area["values"] = self.config["body_areas"] + ["Other"]
                self.body_area.set(new_area)

    def handle_gesture_select(self, event):
        """If 'Other' is selected for gesture, prompt for a new one and update config."""
        if self.gesture_type.get() == "Other":
            new_gesture = simpledialog.askstring("New Gesture", "Enter new gesture name:")
            if new_gesture:
                self.config["gestures"].append(new_gesture)
                self.save_config()
                self.gesture_type["values"] = self.config["gestures"] + ["Other"]
                self.gesture_type.set(new_gesture)

    def start_recording(self):
        """Initialize camera and start the main recording loop."""
        if not self.initialize_camera():
            return

        # Construct a short name for the body part
        self.current_body_part = f"{self.body_side.get()[0]}{self.body_area.get()}"
        # Construct a short name for the gesture (remove spaces)
        self.current_gesture = self.gesture_type.get().replace(" ", "")

        self.total_records = int(self.record_count.get())
        self.current_count = 0
        self.recording = True

        self.start_btn.config(state=DISABLED)
        self.record_loop()

    def record_loop(self):
        """Keeps track of how many recordings have been done and handles stopping."""
        if self.current_count >= self.total_records or not self.recording:
            self.stop_recording()
            return

        self.current_count += 1
        self.status_label.config(text=f"Recording {self.current_count}/{self.total_records}")
        self.root.update()

        # Perform a single recording in a separate thread so UI remains responsive
        recording_thread = threading.Thread(target=self.perform_single_recording)
        recording_thread.start()

    def perform_single_recording(self):
        """
        3-second countdown, then record exactly 7 frames for the SVO.
        No hidden frames. 
        """
        # 1) Attempt to see if IMU is working: if no data after a few tries, show error.
        if not self.check_imu_working():
            self.show_sensor_error_screen()
            self.recording = False
            return

        # 2) Standard countdown + 7 frames
        countdown_window = Toplevel(self.root)
        countdown_window.attributes("-fullscreen", True)

        canvas = Canvas(countdown_window, bg="black")
        canvas.pack(fill=BOTH, expand=True)

        screen_width = countdown_window.winfo_screenwidth()
        screen_height = countdown_window.winfo_screenheight()

        countdown_label = canvas.create_text(
            screen_width // 2,
            screen_height // 2,
            font=("Arial", 100),
            fill="white",
            text="3"
        )
        log_label = canvas.create_text(
            screen_width // 2,
            screen_height - 100,
            font=("Arial", 20),
            fill="white",
            text="Ready..."
        )
        preview_label = Label(canvas, bg="black")
        canvas.create_window(
            screen_width - 220,
            screen_height - 180,
            anchor=NW,
            window=preview_label
        )

        wave_count_label = canvas.create_text(
            10,
            screen_height - 10,
            font=("Arial", 20),
            fill="white",
            anchor=SW,
            text=f"{self.current_count}/{self.total_records}"
        )

        # 3-second countdown (no recording yet)
        for i in range(3, 0, -1):
            canvas.itemconfig(countdown_label, text=str(i))
            self.play_sound("countdown")
            canvas.itemconfig(log_label, text=f"Get ready... {i}")
            canvas.update()

            # Grab 1 frame for preview
            if self.camera.grab() == sl.ERROR_CODE.SUCCESS:
                zed_mat = sl.Mat()
                self.camera.retrieve_image(zed_mat, sl.VIEW.LEFT)
                frame = zed_mat.get_data()
                self.update_preview_label(preview_label, frame, 200, 160)

            time.sleep(0.4)

        # Start recording
        filename = self.get_next_filename()
        # specify 30 fps if you want. Or remove that arg to rely on the default
        recording_param = sl.RecordingParameters(filename, sl.SVO_COMPRESSION_MODE.H264, 30)
        err = self.camera.enable_recording(recording_param)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Error enabling recording: {err}")
            countdown_window.destroy()
            self.record_loop()
            return

        # Now record exactly 7 frames
        canvas.itemconfig(countdown_label, text="GO!")
        canvas.itemconfig(log_label, text="Recording now...")
        self.play_sound("beep")
        canvas.update()

        self.record_visible_frames_with_progress(8, canvas, log_label, preview_label)

        self.camera.disable_recording()

        canvas.itemconfig(countdown_label, text="Saving...")
        canvas.itemconfig(log_label, text="Finalizing file...")
        canvas.update()

        if self.check_svo_save_success(filename):
            canvas.itemconfig(log_label, text="Save successful!")
        else:
            canvas.itemconfig(log_label, text="Save failed or incomplete.")

        time.sleep(0.4)
        countdown_window.destroy()

        self.record_loop()

    def record_visible_frames_with_progress(self, num_frames, canvas, log_label, preview_label):
        """Record exactly `num_frames` with progress bar and preview."""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        bar = canvas.create_rectangle(0, 0, 0, screen_height, fill="green")
        canvas.tag_lower(bar)

        for i in range(num_frames):
            if self.camera.grab() == sl.ERROR_CODE.SUCCESS:
                zed_mat = sl.Mat()
                self.camera.retrieve_image(zed_mat, sl.VIEW.LEFT)
                frame = zed_mat.get_data()

                self.update_preview_label(preview_label, frame, 200, 160)
                canvas.itemconfig(log_label, text=f"Recording frame {i + 1}/{num_frames}")
                progress = int((i + 1) / num_frames * screen_width)
                canvas.coords(bar, 0, 0, progress, screen_height)
                canvas.update()

            time.sleep(0.1)

    def update_preview_label(self, preview_label, frame, target_width=200, target_height=160):
        """Update the preview window with new frame."""
        small_frame = cv2.resize(frame, (target_width, target_height))
        small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGRA2RGB)
        img = Image.fromarray(small_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        preview_label.config(image=imgtk)
        preview_label.image = imgtk

    def check_svo_save_success(self, filename):
        """Verify SVO file was saved correctly."""
        return os.path.exists(filename) and os.path.getsize(filename) > 0

    def get_next_filename(self):
        """Generate filename for new recording."""
        base_dir = os.path.join("datasets", f"{self.current_body_part}_{self.current_gesture}")
        os.makedirs(base_dir, exist_ok=True)

        existing_files = [
            f for f in os.listdir(base_dir)
            if f.startswith(f"SB_{self.current_body_part}_{self.current_gesture}")
        ]
        max_num = max([int(f.split("_")[-1].split(".")[0]) for f in existing_files]) if existing_files else 0
        next_num = max_num + 1

        return os.path.join(
            base_dir,
            f"SB_{self.current_body_part}_{self.current_gesture}_{next_num:09d}.svo2"
        )

    def play_sound(self, sound_type):
        """Play appropriate sound effect."""
        try:
            if sound_type == "countdown":
                pygame.mixer.Sound("system_assets/countdown_beep.wav").play()
            elif sound_type == "beep":
                pygame.mixer.Sound("system_assets/start_beep.wav").play()
        except Exception as e:
            print(f"Sound error ({sound_type}): {e}")

    def initialize_camera(self):
        """Attempt to open the ZED camera."""
        status = self.camera.open(self.init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            messagebox.showerror("Camera Error", f"Failed to open camera: {status}")
            return False
        return True

    def check_imu_working(self, tries=5):
        """
        Attempt up to 'tries' times to see if the IMU returns a non-zero timestamp.
        If each attempt fails, we assume sensor is not working.
        """
        for _ in range(tries):
            if self.camera.grab() == sl.ERROR_CODE.SUCCESS:
                self.camera.get_sensors_data(self.sensors_data, sl.TIME_REFERENCE.CURRENT)
                imu_data = self.sensors_data.get_imu_data()
                if imu_data.timestamp.get_microseconds() > 0:
                    return True
            time.sleep(0.2)
        return False

    def show_sensor_error_screen(self):
        """Display a permanent error message about IMU sensor not returning data."""
        error_window = Toplevel(self.root)
        error_window.attributes("-fullscreen", True)
        error_window.configure(bg='red')
        label = Label(error_window,
                      text="IMU sensor not returning data! Please check ZED or run diagnostic.",
                      font=("Arial", 40), fg='white', bg='red')
        label.place(relx=0.5, rely=0.5, anchor=CENTER)

    def stop_recording(self):
        """Cleanup resources after recording."""
        self.recording = False
        self.camera.close()
        self.start_btn.config(state=NORMAL)
        self.status_label.config(text="Recording Complete!!")


if __name__ == "__main__":
    root = ThemedTk(theme="equilux")
    app = ZedRecorderApp(root)
    root.mainloop()
