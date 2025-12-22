# Setup Instructions for Real-Time Sign Language Translator

## ⚠️ Current Issue: NumPy Installation Corrupted

Your `timegan-gpu` environment has a corrupted NumPy installation. Here's how to fix it:

### Option 1: Reinstall NumPy (Recommended)

```bash
# Activate environment
conda activate timegan-gpu

# Uninstall numpy completely
pip uninstall numpy -y
conda uninstall numpy -y

# Reinstall compatible version
conda install numpy=1.23.5 -y

# Or use pip
pip install "numpy<1.24"
```

### Option 2: Create Fresh Environment

```bash
# Create new environment
conda create -n sign-language-gpu python=3.9 -y
conda activate sign-language-gpu

# Install CUDA toolkit (if not already installed)
conda install cudatoolkit=11.2 cudnn=8.1 -c conda-forge -y

# Install TensorFlow with GPU support
pip install tensorflow-gpu==2.10.0

# Install other dependencies
pip install "numpy<1.24"
pip install opencv-python pillow matplotlib seaborn kaggle scikit-learn pandas
```

### Option 3: Use Existing Environment with Fixes

```bash
conda activate timegan-gpu

# Force reinstall numpy
pip install --force-reinstall "numpy==1.23.5"

# If that fails, try:
conda install --force-reinstall numpy=1.23.5 -c conda-forge
```

---

## 📋 After Fixing NumPy

Once NumPy is fixed, run the scripts in order:

### Step 1: Download Dataset (✅ Already Completed)
```bash
python scripts/step1_download_dataset.py
```

### Step 2: Explore Data
```bash
python scripts/step2_explore_data.py
```

### Step 3: Train Model
```bash
python scripts/step3_train_model.py
```

### Step 4: Evaluate Model
```bash
python scripts/step4_evaluate_model.py
```

### Step 5: Real-Time Detection
```bash
python scripts/step5_realtime_detection.py
```

---

## 🔍 Verify Installation

After fixing NumPy, verify it works:

```bash
conda activate timegan-gpu
python -c "import numpy; print('NumPy version:', numpy.__version__)"
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

Expected output:
```
NumPy version: 1.23.x
TensorFlow version: 2.x.x
OpenCV version: 4.x.x
```

---

## 🚀 Quick Fix Command (Try This First)

```bash
conda activate timegan-gpu
pip uninstall numpy -y && pip install "numpy==1.23.5"
```

Then retry:
```bash
python scripts/step2_explore_data.py
```

---

## 📝 Notes

- The issue is caused by NumPy 1.24+ incompatibility with TensorFlow 2.10
- NumPy 1.23.5 is the recommended version for TensorFlow 2.10
- If you continue having issues, creating a fresh environment (Option 2) is the safest approach

---

## ✅ What's Already Done

- ✅ Project structure created
- ✅ All source modules created (11 files in `src/`)
- ✅ All scripts created (5 files in `scripts/`)
- ✅ Dataset downloaded (87,000 images, 29 classes)

## ⏳ What's Next

- Fix NumPy installation
- Run data exploration
- Train the model
- Evaluate performance
- Test real-time detection

---

Let me know once you've fixed the NumPy issue and I'll help you continue with the project!
