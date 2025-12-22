"""
Image preprocessing utilities for sign language recognition.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def load_image(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load an image from file.
    
    Args:
        image_path: Path to image file
        target_size: Optional target size (width, height)
        
    Returns:
        Image as numpy array (RGB)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if target_size:
        img = cv2.resize(img, target_size)
    
    return img


def normalize_image(img: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Normalize image pixel values.
    
    Args:
        img: Input image
        method: Normalization method ('standard', 'minmax')
        
    Returns:
        Normalized image
    """
    if method == 'standard':
        # Scale to [0, 1]
        return img.astype(np.float32) / 255.0
    elif method == 'minmax':
        # Min-max normalization
        img_min = img.min()
        img_max = img.max()
        return (img - img_min) / (img_max - img_min + 1e-7)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def preprocess_for_model(img: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess image for model input.
    
    Args:
        img: Input image (can be any size)
        target_size: Target size for model
        
    Returns:
        Preprocessed image ready for model
    """
    # Resize if needed
    if img.shape[:2] != target_size:
        img = cv2.resize(img, target_size)
    
    # Normalize
    img = normalize_image(img)
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img


def augment_image(img: np.ndarray, 
                 rotation_range: int = 15,
                 brightness_range: Tuple[float, float] = (0.8, 1.2),
                 zoom_range: float = 0.1) -> np.ndarray:
    """
    Apply random augmentation to image.
    
    Args:
        img: Input image
        rotation_range: Max rotation angle in degrees
        brightness_range: Min and max brightness multiplier
        zoom_range: Max zoom factor
        
    Returns:
        Augmented image
    """
    h, w = img.shape[:2]
    
    # Random rotation
    if rotation_range > 0:
        angle = np.random.uniform(-rotation_range, rotation_range)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # Random brightness
    if brightness_range:
        brightness = np.random.uniform(*brightness_range)
        img = np.clip(img * brightness, 0, 255).astype(np.uint8)
    
    # Random zoom
    if zoom_range > 0:
        zoom = np.random.uniform(1 - zoom_range, 1 + zoom_range)
        new_h, new_w = int(h * zoom), int(w * zoom)
        img = cv2.resize(img, (new_w, new_h))
        
        # Crop or pad to original size
        if zoom > 1:
            # Crop center
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            img = img[start_h:start_h+h, start_w:start_w+w]
        else:
            # Pad
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            img = cv2.copyMakeBorder(img, pad_h, h-new_h-pad_h, 
                                    pad_w, w-new_w-pad_w, 
                                    cv2.BORDER_REFLECT)
    
    return img


def extract_hand_region(img: np.ndarray, padding: int = 20) -> Optional[np.ndarray]:
    """
    Extract hand region from image using simple thresholding.
    
    Args:
        img: Input image
        padding: Padding around detected region
        
    Returns:
        Cropped hand region or None if not found
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2 * padding)
    h = min(img.shape[0] - y, h + 2 * padding)
    
    # Crop
    hand_region = img[y:y+h, x:x+w]
    
    return hand_region


if __name__ == "__main__":
    print("Image preprocessing utilities loaded successfully!")
