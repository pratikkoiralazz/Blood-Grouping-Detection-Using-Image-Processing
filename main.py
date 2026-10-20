import os
import cv2
import numpy as np
from scipy.signal import butter, lfilter
import tensorflow as tf
from tensorflow.keras import layers, models

# ==========================================
# 1. SIGNAL FILTERING (Butterworth Bandpass)
# ==========================================
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# ==========================================
# 2. PPG SIGNAL EXTRACTION FROM VIDEO
# ==========================================
def extract_ppg(video_path):
    if not os.path.exists(video_path):
        print(f"Warning: {video_path} not found. Generating dummy PPG signal for testing.")
        # Generate synthetic PPG signal if video isn't provided yet
        fs = 30.0
        t = np.linspace(0, 10, int(fs * 10))
        return np.sin(2 * np.pi * 1.2 * t) + np.random.normal(0, 0.1, len(t)), fs

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: 
        fps = 30.0
    
    signal = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Extract center Region of Interest (ROI) - Fingertip
        h, w, _ = frame.shape
        roi = frame[int(h*0.4):int(h*0.6), int(w*0.4):int(w*0.6)]
        # Use Green channel as it contains strongest PPG response
        mean_green = np.mean(roi[:, :, 1])
        signal.append(mean_green)
        
    cap.release()
    raw_signal = np.array(signal)
    
    # Filter signal (Typical heart rate frequency band: 0.75 Hz to 3.0 Hz)
    filtered_signal = bandpass_filter(raw_signal, lowcut=0.75, highcut=3.0, fs=fps, order=3)
    return filtered_signal, fps

# ==========================================
# 3. FEATURE EXTRACTION
# ==========================================
def extract_features(signal):
    features = [
        np.mean(signal),
        np.std(signal),
        np.max(signal),
        np.min(signal),
        np.ptp(signal), # Peak-to-peak amplitude
        np.sqrt(np.mean(signal**2)), # RMS
        np.percentile(signal, 75) - np.percentile(signal, 25) # Interquartile range
    ]
    # Add first and second derivatives (APG features)
    first_deriv = np.diff(signal)
    second_deriv = np.diff(first_deriv)
    
    features.extend([
        np.mean(first_deriv), np.std(first_deriv),
        np.mean(second_deriv), np.std(second_deriv)
    ])
    
    return np.array(features)

# ==========================================
# 4. CNN MODEL ARCHITECTURE FOR REGRESSION
# ==========================================
def create_cnn_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((input_shape[0], 1)), # Reshape for 1D CNN
        layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1) # Regression output for parameter estimation
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# ==========================================
# MAIN EXECUTION SCRIPT
# ==========================================
if __name__ == "__main__":
    print("Initializing Non-Invasive Estimation Pipeline...")
    
    # Step 1: Extract & Filter PPG from video (or dummy fallback)
    video_file = "sample_fingertip.mp4"
    ppg_signal, sampling_rate = extract_ppg(video_file)
    print(f"PPG Signal extracted. Length: {len(ppg_signal)} frames.")
    
    # Step 2: Extract Features
    features = extract_features(ppg_signal)
    print(f"Extracted {len(features)} morphological & statistical features.")
    
    # Step 3: Build and Test CNN Model
    # Simulating training data (Batch size of 10 samples, feature length)
    X_train = np.random.rand(10, len(features))
    y_train = np.random.rand(10, 1) # Target parameter values
    
    model = create_cnn_model(input_shape=(len(features),))
    model.summary()
    
    print("Training mock model...")
    model.fit(X_train, y_train, epochs=5, batch_size=2, verbose=1)
    print("Pipeline executed successfully!")