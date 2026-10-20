import os
import cv2
import numpy as np
from src.signal_filters import bandpass_filter

def extract_ppg_from_video(video_path):
    if not os.path.exists(video_path):
        print(f"[Warning] {video_path} not found. Generating synthetic PPG signal.")
        fs = 30.0
        t = np.linspace(0, 10, int(fs * 10))
        synthetic_signal = np.sin(2 * np.pi * 1.2 * t) + np.random.normal(0, 0.05, len(t))
        return bandpass_filter(synthetic_signal, fs=fs), fs

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: 
        fps = 30.0
    
    signal = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        # Center fingertip ROI
        roi = frame[int(h*0.4):int(h*0.6), int(w*0.4):int(w*0.6)]
        mean_green = np.mean(roi[:, :, 1]) # Green channel
        signal.append(mean_green)
        
    cap.release()
    raw_signal = np.array(signal)
    return bandpass_filter(raw_signal, fs=fps), fps