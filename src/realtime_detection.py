"""
Real-time detection utilities for webcam-based sign language recognition.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
from typing import Tuple, Optional, List
import time


class SignLanguageDetector:
    """Real-time sign language detector using webcam."""
    
    def __init__(self,
                 model_path: str,
                 class_names: List[str],
                 img_size: Tuple[int, int] = (224, 224),
                 confidence_threshold: float = 0.8,
                 smoothing_window: int = 5):
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
        self.fps_queue = deque(maxlen=30)
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for model input.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Preprocessed frame
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(frame_rgb, self.img_size)
        
        # Normalize
        img = img / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, frame: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Predict sign language gesture from frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (class_idx, confidence, all_predictions)
        """
        # Preprocess
        img = self.preprocess_frame(frame)
        
        # Predict
        predictions = self.model.predict(img, verbose=0)[0]
        
        # Get top prediction
        class_idx = np.argmax(predictions)
        confidence = predictions[class_idx]
        
        # Add to queue for smoothing
        self.prediction_queue.append((class_idx, confidence))
        
        # Get smoothed prediction
        smoothed_idx, smoothed_conf = self._smooth_predictions()
        
        return smoothed_idx, smoothed_conf, predictions
    
    def _smooth_predictions(self) -> Tuple[int, float]:
        """
        Smooth predictions using queue.
        
        Returns:
            Tuple of (smoothed_class_idx, smoothed_confidence)
        """
        if len(self.prediction_queue) < 3:
            # Not enough predictions yet
            return self.prediction_queue[-1]
        
        # Get predictions with high confidence
        valid_preds = [(idx, conf) for idx, conf in self.prediction_queue
                      if conf >= self.confidence_threshold]
        
        if not valid_preds:
            # No high confidence predictions
            return self.prediction_queue[-1]
        
        # Get most common class
        class_counts = {}
        for idx, conf in valid_preds:
            class_counts[idx] = class_counts.get(idx, 0) + 1
        
        smoothed_idx = max(class_counts, key=class_counts.get)
        
        # Average confidence for smoothed class
        smoothed_conf = np.mean([conf for idx, conf in valid_preds 
                                if idx == smoothed_idx])
        
        return smoothed_idx, smoothed_conf
    
    def draw_prediction(self,
                       frame: np.ndarray,
                       class_idx: int,
                       confidence: float,
                       fps: float) -> np.ndarray:
        """
        Draw prediction overlay on frame.
        
        Args:
            frame: Input frame
            class_idx: Predicted class index
            confidence: Prediction confidence
            fps: Current FPS
            
        Returns:
            Frame with overlay
        """
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Draw prediction text
        if confidence >= self.confidence_threshold:
            text = f"{self.class_names[class_idx]}"
            color = (0, 255, 0)  # Green
        else:
            text = "..."
            color = (0, 165, 255)  # Orange
        
        # Main prediction text
        cv2.putText(frame, text, (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 3)
        
        # Confidence text
        conf_text = f"Confidence: {confidence:.1%}"
        cv2.putText(frame, conf_text, (20, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # FPS counter
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit", (w - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def update_fps(self, frame_time: float) -> float:
        """
        Update FPS calculation.
        
        Args:
            frame_time: Time taken for frame processing
            
        Returns:
            Current average FPS
        """
        fps = 1.0 / frame_time if frame_time > 0 else 0
        self.fps_queue.append(fps)
        return np.mean(self.fps_queue)


class WebcamCapture:
    """Webcam capture utility."""
    
    def __init__(self,
                 camera_id: int = 0,
                 width: int = 640,
                 height: int = 480):
        """
        Initialize webcam capture.
        
        Args:
            camera_id: Camera device ID
            width: Frame width
            height: Frame height
        """
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam!")
    
    def read_frame(self, flip: bool = True) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read frame from webcam.
        
        Args:
            flip: Whether to flip frame horizontally (mirror effect)
            
        Returns:
            Tuple of (success, frame)
        """
        ret, frame = self.cap.read()
        
        if ret and flip:
            frame = cv2.flip(frame, 1)
        
        return ret, frame
    
    def release(self):
        """Release webcam."""
        self.cap.release()
        cv2.destroyAllWindows()


def run_realtime_detection(model_path: str,
                           class_names: List[str],
                           camera_id: int = 0,
                           img_size: Tuple[int, int] = (224, 224),
                           confidence_threshold: float = 0.8):
    """
    Run real-time sign language detection.
    
    Args:
        model_path: Path to trained model
        class_names: List of class names
        camera_id: Camera device ID
        img_size: Input image size for model
        confidence_threshold: Minimum confidence threshold
    """
    print("\n[>>] Initializing detector...")
    detector = SignLanguageDetector(
        model_path=model_path,
        class_names=class_names,
        img_size=img_size,
        confidence_threshold=confidence_threshold
    )
    
    print("[>>] Starting webcam...")
    webcam = WebcamCapture(camera_id=camera_id)
    
    print("[OK] Real-time detection active!")
    print("\nInstructions:")
    print("  - Show hand signs to the camera")
    print("  - Press 'q' to quit")
    print("  - Green text = High confidence")
    print("  - Orange text = Low confidence\n")
    
    try:
        while True:
            start_time = time.time()
            
            # Read frame
            ret, frame = webcam.read_frame(flip=True)
            if not ret:
                print("[X] Failed to read frame!")
                break
            
            # Predict
            class_idx, confidence, _ = detector.predict(frame)
            
            # Calculate FPS
            frame_time = time.time() - start_time
            fps = detector.update_fps(frame_time)
            
            # Draw overlay
            frame = detector.draw_prediction(frame, class_idx, confidence, fps)
            
            # Display
            cv2.imshow('Sign Language Detection', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[>>] Quitting...")
                break
                
    except KeyboardInterrupt:
        print("\n[>>] Interrupted by user")
    
    finally:
        webcam.release()
        print("[OK] Webcam released")


if __name__ == "__main__":
    print("Real-time detection utilities loaded successfully!")
