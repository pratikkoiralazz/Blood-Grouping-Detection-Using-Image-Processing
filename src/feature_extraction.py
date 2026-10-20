import numpy as np

def extract_features(signal):
    features = [
        np.mean(signal),
        np.std(signal),
        np.max(signal),
        np.min(signal),
        np.ptp(signal), # Peak-to-peak
        np.sqrt(np.mean(signal**2)), # RMS
        np.percentile(signal, 75) - np.percentile(signal, 25) # IQR
    ]
    
    # First and second derivatives (APG)
    first_deriv = np.diff(signal)
    second_deriv = np.diff(first_deriv)
    
    features.extend([
        np.mean(first_deriv), np.std(first_deriv),
        np.mean(second_deriv), np.std(second_deriv)
    ])
    
    return np.array(features)