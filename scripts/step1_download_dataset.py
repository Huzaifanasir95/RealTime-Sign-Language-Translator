"""
Step 1: Download ASL Alphabet Dataset from Kaggle

This script downloads the ASL Alphabet dataset and verifies it.

Usage:
    conda activate timegan-gpu
    python scripts/step1_download_dataset.py
"""

import os
import sys
import zipfile
from pathlib import Path
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_kaggle_setup():
    """Check if Kaggle API is properly configured."""
    print("\n" + "="*70)
    print("CHECKING KAGGLE API CONFIGURATION")
    print("="*70 + "\n")
    
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        print("[X] Kaggle API not configured!")
        print("\nPlease follow these steps:")
        print("1. Go to https://www.kaggle.com/")
        print("2. Click on your profile -> Account -> Create New API Token")
        print("3. Download kaggle.json")
        print(f"4. Place it in: {kaggle_json}")
        return False
    
    print(f"[OK] Kaggle API configured at: {kaggle_json}")
    return True


def download_dataset():
    """Download ASL Alphabet dataset from Kaggle."""
    print("\n" + "="*70)
    print("DOWNLOADING ASL ALPHABET DATASET")
    print("="*70 + "\n")
    
    # Create data directory
    data_dir = project_root / 'data' / 'raw'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset already exists
    dataset_path = data_dir / 'asl_alphabet_train'
    if dataset_path.exists() and any(dataset_path.iterdir()):
        print(f"[OK] Dataset already exists at: {dataset_path}")
        print("Skipping download...")
        return True
    
    try:
        # Download dataset
        print("[>>] Downloading dataset from Kaggle...")
        print("     This may take several minutes depending on your connection...")
        
        cmd = [
            'kaggle', 'datasets', 'download',
            '-d', 'grassknoted/asl-alphabet',
            '-p', str(data_dir),
            '--unzip'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("[OK] Dataset downloaded successfully!")
            
            # Check if unzipped correctly
            if dataset_path.exists():
                print(f"[OK] Dataset extracted to: {dataset_path}")
                return True
            else:
                print("[!] Dataset downloaded but extraction path not found")
                print("Looking for zip file...")
                
                # Find and extract zip file manually
                zip_files = list(data_dir.glob('*.zip'))
                if zip_files:
                    print(f"Found zip file: {zip_files[0]}")
                    print("Extracting manually...")
                    with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
                        zip_ref.extractall(data_dir)
                    print("[OK] Extraction complete!")
                    # Remove zip file
                    zip_files[0].unlink()
                    return True
                
        else:
            print(f"[X] Error downloading dataset:")
            print(result.stderr)
            return False
            
    except FileNotFoundError:
        print("[X] Kaggle CLI not found!")
        print("\nPlease install it with:")
        print("    pip install kaggle")
        return False
    except Exception as e:
        print(f"[X] Error: {e}")
        return False


def verify_dataset():
    """Verify dataset was downloaded correctly."""
    print("\n" + "="*70)
    print("VERIFYING DATASET")
    print("="*70 + "\n")
    
    dataset_path = project_root / 'data' / 'raw' / 'asl_alphabet_train' / 'asl_alphabet_train'
    
    if not dataset_path.exists():
        print(f"[X] Dataset not found at: {dataset_path}")
        return False
    
    # Count classes
    classes = [d for d in dataset_path.iterdir() if d.is_dir()]
    num_classes = len(classes)
    
    print(f"[OK] Found {num_classes} classes")
    
    # Count total images
    total_images = 0
    for class_dir in classes:
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        total_images += len(images)
    
    print(f"[OK] Total images: {total_images:,}")
    
    if num_classes == 29 and total_images > 80000:
        print("\n[OK] Dataset verification successful!")
        print(f"[>>] Dataset location: {dataset_path}")
        return True
    else:
        print("\n[!] Dataset may be incomplete")
        print(f"Expected: 29 classes, ~87,000 images")
        print(f"Found: {num_classes} classes, {total_images:,} images")
        return False


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("STEP 1: DOWNLOAD ASL ALPHABET DATASET")
    print("="*70)
    
    # Step 1: Check Kaggle setup
    if not check_kaggle_setup():
        return
    
    # Step 2: Download dataset
    if not download_dataset():
        print("\n[X] Dataset download failed!")
        return
    
    # Step 3: Verify dataset
    if verify_dataset():
        print("\n" + "="*70)
        print("[OK] STEP 1 COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nNext step: Data exploration and analysis")
        print("Command: python scripts/step2_explore_data.py")
    else:
        print("\n[!] Please check the dataset manually")


if __name__ == "__main__":
    main()
