"""
Step 5: Real-Time Sign Language Detection

This script runs real-time sign language detection using webcam.

Usage:
    conda activate timegan-gpu
    python scripts/step5_realtime_detection.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from collections import deque
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class SignLanguageDetector:
    """Real-time sign language detector."""
    
    def __init__(self, model_path, class_names, img_size=(224, 224), 
                 confidence_threshold=0.8, smoothing_window=5):
        """
        Initialize detector.
        
        Args:
            model_path: Path to trained model
            class_names: List of class names
            img_size: Input image size for model
            confidence_threshold: Minimum confidence for prediction
            smoothing_window: Number of frames for prediction smoothing
        """
        self.model = keras.models.load_model(model_path)
        self.class_names = class_names
        self.img_size = img_size
        self.confidence_threshold = confidence_threshold
        self.prediction_queue = deque(maxlen=smoothing_window)
        
    def preprocess_frame(self, frame):
        """Preprocess frame for model input."""
        # Resize
        img = cv2.resize(frame, self.img_size)
        # Normalize
        img = img / 255.0
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img
    
    def predict(self, frame):
        """Predict sign language gesture from frame."""
        # Preprocess
        img = self.preprocess_frame(frame)
        
        # Predict
        predictions = self.model.predict(img, verbose=0)[0]
        
        # Get top prediction
        class_idx = np.argmax(predictions)
        confidence = predictions[class_idx]
        
        # Add to queue for smoothing
        self.prediction_queue.append((class_idx, confidence))
        
        # Get smoothed prediction (most common in queue)
        if len(self.prediction_queue) >= 3:
            # Get most frequent prediction with high confidence
            valid_preds = [(idx, conf) for idx, conf in self.prediction_queue 
                          if conf >= self.confidence_threshold]
            if valid_preds:
                # Most common class
                class_counts = {}
                for idx, conf in valid_preds:
                    class_counts[idx] = class_counts.get(idx, 0) + 1
                smoothed_idx = max(class_counts, key=class_counts.get)
                smoothed_conf = np.mean([conf for idx, conf in valid_preds if idx == smoothed_idx])
                return smoothed_idx, smoothed_conf
        
        return class_idx, confidence
    
    def draw_prediction(self, frame, class_idx, confidence):
        """Draw prediction on frame."""
        h, w = frame.shape[:2]
        
        # Create overlay
        overlay = frame.copy()
        
        # Draw semi-transparent box at top
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Draw prediction text
        if confidence >= self.confidence_threshold:
            text = f"{self.class_names[class_idx]}"
            color = (0, 255, 0)  # Green
        else:
            text = "..."
            color = (0, 165, 255)  # Orange
        
        # Main text
        cv2.putText(frame, text, (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 3)
        
        # Confidence
        conf_text = f"Confidence: {confidence:.2%}"
        cv2.putText(frame, conf_text, (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit", (w - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("STEP 5: REAL-TIME SIGN LANGUAGE DETECTION")
    print("="*70 + "\n")
    
    # Define paths
    model_path = project_root / 'models' / 'saved_models' / 'best_model.keras'
    
    if not model_path.exists():
        print(f"[X] Model not found at: {model_path}")
        print("\nPlease train the model first:")
        print("  python scripts/step3_train_model.py")
        return
    
    # Load class names (ASL alphabet)
    class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                   'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                   'del', 'nothing', 'space']
    
    print("[>>] Loading model...")
    detector = SignLanguageDetector(
        model_path=model_path,
        class_names=class_names,
        img_size=(224, 224),
        confidence_threshold=0.8,
        smoothing_window=5
    )
    print("[OK] Model loaded successfully!")
    
    print("\n[>>] Starting webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[X] Could not open webcam!")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("[OK] Webcam started!")
    print("\n" + "="*70)
    print("REAL-TIME DETECTION ACTIVE")
    print("="*70)
    print("\nInstructions:")
    print("  - Show hand signs to the camera")
    print("  - Press 'q' to quit")
    print("  - Green text = High confidence")
    print("  - Orange text = Low confidence")
    print("\n")
    
    fps_queue = deque(maxlen=30)
    
    try:
        while True:
            start_time = time.time()
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("[X] Failed to read frame!")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Predict
            class_idx, confidence = detector.predict(frame)
            
            # Draw prediction
            frame = detector.draw_prediction(frame, class_idx, confidence)
            
            # Calculate and display FPS
            fps = 1.0 / (time.time() - start_time)
            fps_queue.append(fps)
            avg_fps = np.mean(fps_queue)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Sign Language Detection', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[>>] Quitting...")
                break
                
    except KeyboardInterrupt:
        print("\n[>>] Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[OK] Webcam released")
    
    print("\n" + "="*70)
    print("[OK] STEP 5 COMPLETED!")
    print("="*70)
    print("\nProject completed successfully!")


if __name__ == "__main__":
    main()
