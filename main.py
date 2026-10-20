import numpy as np
from src.database import init_db, save_prediction, fetch_records
from src.video_processing import extract_ppg_from_video
from src.feature_extraction import extract_features
from src.ga_feature_selection import genetic_algorithm_feature_selection
from src.cnn_model import create_cnn_model

if __name__ == "__main__":
    print("=== Initializing Non-Invasive Estimation System ===")
    
    # 1. Initialize Database
    init_db()
    
    # 2. Extract PPG Signal from Video
    video_path = "sample_fingertip.mp4"
    ppg_signal, fs = extract_ppg_from_video(video_path)
    print(f"PPG Signal extracted successfully. Length: {len(ppg_signal)}")
    
    # 3. Extract Features
    features = extract_features(ppg_signal)
    print(f"Extracted {len(features)} morphological features.")
    
    # 4. Feature Selection using Genetic Algorithm
    X_mock = np.random.rand(20, len(features))
    y_mock = np.random.rand(20, 1)
    best_mask = genetic_algorithm_feature_selection(X_mock, y_mock)
    selected_features = features[best_mask == 1]
    print(f"GA selected {len(selected_features)} optimal features.")
    
    # Ensure minimum shape for CNN
    if len(selected_features) == 0:
        selected_features = features
        
    # 5. Train / Predict using CNN Model
    model = create_cnn_model(input_shape=(len(selected_features),))
    X_train = np.random.rand(10, len(selected_features))
    y_train = np.random.rand(10, 1)
    
    print("Training CNN Model...")
    model.fit(X_train, y_train, epochs=3, batch_size=2, verbose=0)
    
    # Make prediction on current sample
    prediction = model.predict(selected_features.reshape(1, -1))[0][0]
    print(f"Estimated Result Value: {prediction:.4f}")
    
    # 6. Save to SQL Database
    save_prediction("Test_User", float(prediction))
    print("Prediction logged to SQL database successfully!")
    
    # Display saved records
    print("\nRecent Database Records:")
    for row in fetch_records():
        print(row)