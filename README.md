# ZED Gesture Detection

**📌 Check out our detailed technical deck/slides here:**  
[https://docs.google.com/presentation/d/e/2PACX-1vSRz_C2oECtM8RogQvxI8SEyLF_WO-WIrgvVMJc4S7IbUI9_4qK2NHtrP3jeuSx3Q/pub?start=false&loop=false&delayms=3000](https://docs.google.com/presentation/d/e/2PACX-1vSRz_C2oECtM8RogQvxI8SEyLF_WO-WIrgvVMJc4S7IbUI9_4qK2NHtrP3jeuSx3Q/pub?start=false&loop=false&delayms=3000)

## Overview

This R&D project presents a real-time gesture recognition system using 3D skeletal data from the [ZED stereo camera](https://www.stereolabs.com/en-fr). Developed at [SUPERBIEN Studio](https://www.superbien.studio) (by CreaTech team at SUPERLAB) in Paris, France. The system replaces traditional image-based gesture recognition with a low-latency, skeletal-feature-driven AI pipeline suitable for creative and performance-critical applications.

The system uses the [ZED SDK’s BODY_38 keypoints](https://www.stereolabs.com/docs/body-tracking) to extract high-resolution body motion in real time, applying feature-engineered sequence models for gesture classification.

## 1. Automated Dataset Recorder

To build a robust dataset for training, we developed a custom GUI-based ZED recorder that automates the data collection process. This tool allows users to configure recording sessions by:

- Selecting any **body part** (left/right, arm/leg, etc.)
- Choosing or defining any **gesture** (e.g., swiping, stillness, etc.)
- Specifying the **number of repetitions** to record

Each session produces `.svo2` files using the ZED SDK, with filenames encoded by body part, gesture type, and index for structured dataset management.

Example:  
`SB_RArm_SwipeRight_00000026.svo2`

## 2. AI R&D


### Why Skeleton-based?

Compared to CNN-LSTM pipelines on image frames, our skeleton-based approach:

- Operates in real time (38 FPS vs. 14 FPS)
- Achieves high classification accuracy (~98.4%)
- Enables interpretability via engineered motion features

### Feature Engineering

Each frame is extracted to features in 5 categories:

- Position & Angles
- Movement Dynamics
- Speed Characteristics
- Advanced Geometric 
- Temporal Pattern 

Each gesture is represented as a **sequence of 7 frames × 70 features**, forming the model input.

### Model Architectures

We implemented and benchmarked three models: LSTM, Transformer, Hybrid LSTM-Transformer.

### Training Strategy

- Dataset: 640 labeled samples (4 gestures)
- Data Split: 80% train / 20% blind test
- Batch Size: 32
- Early stopping: Patience = 15
- Each model trained over 10 random initializations

### Performance Comparison

<img width="1001" alt="Screenshot 2568-06-12 at 16 53 25" src="https://github.com/user-attachments/assets/19df855b-97e4-45f2-bcad-822a7bd0e3e1" />
<img width="948" alt="Screenshot 2568-06-12 at 16 53 34" src="https://github.com/user-attachments/assets/db3cbf82-2271-40c2-b3e8-12956b05a7a4" />

## 3. Real-Time Inference Engine

- Implements a state machine:
  - WAITING → READY → CAPTURING → CLASSIFYING
- Feature extraction and classification run on a dedicated inference thread
- Sliding window strategy with confidence thresholding
- GUI: Tkinter-based interface
- Supports real-time integration via socket or WebSocket to:
  - TouchDesigner
  - Unreal Engine
  - Unity
  - Web platforms
    
## Applications

- Creative interactive installations
- Gesture-controlled UIs
- Gaming and immersive environments
- Accessible computing interfaces
- AR/VR control systems

## Citation

If you use this project, please cite:

```
Napassorn Litchiowong, Nicolas Désilles. *ZED Gesture Detection* SUPERBIEN, 2025. MIT License.  
Available at: https://github.com/Superbien/ZED_Gesture_Detection
```

## Credits

**Authors:**
- [Napassorn Litchiowong / Pleng](https://www.linkedin.com/in/plengnaps/) (Creative Technologist Intern @ SUPERBIEN)
- [Nicolas Désilles](https://www.linkedin.com/in/nicolasdesilles/) (Creative Technologist @ SUPERBIEN)

**Special Thanks:** Céleste, Silje, Axel, Nicolas Lim, and SuperTeam @ SUPERBIEN

_"Thank you, SUPERBIEN, for allowing us to publish this project as open source."_

Contact: diluxed@gmail.com
