"""
Dataset management utilities.
"""

import os
import subprocess
import zipfile
from pathlib import Path
from typing import Optional, Tuple, List
import shutil


def check_kaggle_api() -> bool:
    """
    Check if Kaggle API is configured.
    
    Returns:
        True if configured, False otherwise
    """
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    return kaggle_json.exists()


def download_from_kaggle(dataset_name: str, 
                         output_dir: Path,
                         unzip: bool = True) -> bool:
    """
    Download dataset from Kaggle.
    
    Args:
        dataset_name: Kaggle dataset name (e.g., 'grassknoted/asl-alphabet')
        output_dir: Directory to save dataset
        unzip: Whether to unzip after download
        
    Returns:
        True if successful, False otherwise
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        cmd = ['kaggle', 'datasets', 'download', '-d', dataset_name, '-p', str(output_dir)]
        if unzip:
            cmd.append('--unzip')
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        print("[X] Kaggle CLI not found!")
        return False
    except Exception as e:
        print(f"[X] Error: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """
    Extract zip file.
    
    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    except Exception as e:
        print(f"[X] Error extracting zip: {e}")
        return False


def verify_dataset_structure(data_dir: Path, 
                            expected_classes: Optional[int] = None,
                            expected_images: Optional[int] = None) -> Tuple[bool, dict]:
    """
    Verify dataset structure and count files.
    
    Args:
        data_dir: Dataset directory
        expected_classes: Expected number of classes
        expected_images: Expected minimum number of images
        
    Returns:
        Tuple of (is_valid, stats_dict)
    """
    if not data_dir.exists():
        return False, {'error': 'Directory does not exist'}
    
    # Count classes
    classes = [d for d in data_dir.iterdir() if d.is_dir()]
    num_classes = len(classes)
    
    # Count images
    total_images = 0
    for class_dir in classes:
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        total_images += len(images)
    
    stats = {
        'num_classes': num_classes,
        'total_images': total_images,
        'classes': [c.name for c in classes]
    }
    
    # Validate
    is_valid = True
    if expected_classes and num_classes != expected_classes:
        is_valid = False
        stats['error'] = f"Expected {expected_classes} classes, found {num_classes}"
    
    if expected_images and total_images < expected_images:
        is_valid = False
        stats['error'] = f"Expected at least {expected_images} images, found {total_images}"
    
    return is_valid, stats


def get_class_names(data_dir: Path) -> List[str]:
    """
    Get sorted list of class names from directory.
    
    Args:
        data_dir: Dataset directory
        
    Returns:
        List of class names
    """
    return sorted([d.name for d in data_dir.iterdir() if d.is_dir()])


def count_images_per_class(data_dir: Path) -> dict:
    """
    Count images in each class.
    
    Args:
        data_dir: Dataset directory
        
    Returns:
        Dictionary mapping class name to count
    """
    class_counts = {}
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.png')))
            class_counts[class_dir.name] = count
    return class_counts


def create_train_val_split(data_dir: Path,
                           output_dir: Path,
                           val_split: float = 0.2,
                           seed: int = 42):
    """
    Create train/validation split by copying files.
    
    Args:
        data_dir: Source dataset directory
        output_dir: Output directory for split data
        val_split: Validation split ratio
        seed: Random seed
    """
    import random
    random.seed(seed)
    
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    
    for class_dir in data_dir.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        
        # Create class directories
        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
        (val_dir / class_name).mkdir(parents=True, exist_ok=True)
        
        # Get all images
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        random.shuffle(images)
        
        # Split
        split_idx = int(len(images) * (1 - val_split))
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Copy files
        for img in train_images:
            shutil.copy(img, train_dir / class_name / img.name)
        
        for img in val_images:
            shutil.copy(img, val_dir / class_name / img.name)
    
    print(f"[OK] Created train/val split in {output_dir}")


def clean_dataset(data_dir: Path, 
                 min_size: Tuple[int, int] = (50, 50),
                 max_size: Optional[Tuple[int, int]] = None):
    """
    Clean dataset by removing invalid or corrupted images.
    
    Args:
        data_dir: Dataset directory
        min_size: Minimum image size (width, height)
        max_size: Maximum image size (width, height)
    """
    import cv2
    
    removed_count = 0
    
    for class_dir in data_dir.iterdir():
        if not class_dir.is_dir():
            continue
        
        for img_path in class_dir.glob('*'):
            if not img_path.is_file():
                continue
            
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"[!] Removing corrupted image: {img_path}")
                    img_path.unlink()
                    removed_count += 1
                    continue
                
                h, w = img.shape[:2]
                
                # Check minimum size
                if w < min_size[0] or h < min_size[1]:
                    print(f"[!] Removing too small image: {img_path} ({w}x{h})")
                    img_path.unlink()
                    removed_count += 1
                    continue
                
                # Check maximum size
                if max_size and (w > max_size[0] or h > max_size[1]):
                    print(f"[!] Removing too large image: {img_path} ({w}x{h})")
                    img_path.unlink()
                    removed_count += 1
                    continue
                    
            except Exception as e:
                print(f"[!] Error processing {img_path}: {e}")
                img_path.unlink()
                removed_count += 1
    
    print(f"[OK] Removed {removed_count} invalid images")


if __name__ == "__main__":
    print("Dataset utilities loaded successfully!")
