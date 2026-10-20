import cv2
import numpy as np


def extract_ppg_from_video(video_path):
  cap = cv2.VideoCapture(video_path)
  fps = cap.get(cv2.CAP_PROP_FPS)
  signal = []

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    # Select ROI (e.g., center of the fingertip frame)
    height, width, _ = frame.shape
    roi = frame[
        int(height * 0.4) : int(height * 0.6),
        int(width * 0.4) : int(width * 0.6),
    ]

    # Calculate average green channel intensity (strongest PPG reflection)
    mean_green = np.mean(roi[:, :, 1])
    signal.append(mean_green)

  cap.release()
  return np.array(signal), fps