# Real-Time Sign Language Translator - Scripts

This directory contains the complete workflow scripts for the project.

## 📋 Script Execution Order

Run the scripts in this order:

### 1. **step1_download_dataset.py**
Downloads the ASL Alphabet dataset from Kaggle.

```bash
conda activate timegan-gpu
python scripts/step1_download_dataset.py
```

**What it does:**
- Checks Kaggle API configuration
- Downloads ~87,000 ASL alphabet images
- Verifies dataset integrity (29 classes)

**Requirements:**
- Kaggle API token (`~/.kaggle/kaggle.json`)

---

### 2. **step2_explore_data.py**
Explores and visualizes the dataset.

```bash
python scripts/step2_explore_data.py
```

**What it does:**
- Analyzes class distribution
- Creates visualizations (bar charts, sample images)
- Generates exploration report
- Analyzes image properties

**Outputs:**
- `outputs/exploration/class_distribution.png`
- `outputs/exploration/sample_images.png`
- `outputs/exploration/variations_class_A.png`
- `outputs/exploration/data_exploration_report.txt`

---

### 3. **step3_train_model.py**
Trains the sign language recognition model.

```bash
python scripts/step3_train_model.py
```

**What it does:**
- Configures GPU (if available)
- Creates data generators with augmentation
- Builds transfer learning model (MobileNetV2)
- Trains for 50 epochs with callbacks
- Saves best model and training history

**Outputs:**
- `models/saved_models/best_model.keras`
- `outputs/training/training_history.json`
- `outputs/training/training_history.png`
- `outputs/training/model_summary.txt`
- `logs/training_YYYYMMDD_HHMMSS/` (TensorBoard logs)

**Configuration:**
- Image size: 224x224
- Batch size: 32
- Epochs: 50
- Validation split: 20%
- Model: MobileNetV2 (transfer learning)

---

### 4. **step4_evaluate_model.py**
Evaluates the trained model.

```bash
python scripts/step4_evaluate_model.py
```

**What it does:**
- Loads trained model
- Evaluates on test data
- Generates confusion matrix
- Creates classification report
- Plots per-class accuracy

**Outputs:**
- `outputs/evaluation/confusion_matrix.png`
- `outputs/evaluation/classification_report.txt`
- `outputs/evaluation/per_class_accuracy.png`
- `outputs/evaluation/evaluation_summary.json`

---

### 5. **step5_realtime_detection.py**
Runs real-time sign language detection via webcam.

```bash
python scripts/step5_realtime_detection.py
```

**What it does:**
- Loads trained model
- Opens webcam feed
- Detects sign language gestures in real-time
- Shows predictions with confidence scores
- Displays FPS

**Controls:**
- Press `q` to quit

**Features:**
- Prediction smoothing (5-frame window)
- Confidence threshold (80%)
- Visual feedback (green = high confidence, orange = low)
- FPS counter

---

## 🔧 Requirements

All scripts require the `timegan-gpu` conda environment with:
- TensorFlow (with CUDA support)
- OpenCV
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- Pillow
- Kaggle API

Install missing packages:
```bash
pip install opencv-python pillow matplotlib seaborn kaggle scikit-learn
```

---

## 📊 Expected Results

- **Training Accuracy**: >95%
- **Validation Accuracy**: >90%
- **Real-time FPS**: 20-30 FPS (GPU), 5-10 FPS (CPU)
- **Model Size**: ~15 MB

---

## 🐛 Troubleshooting

### Kaggle API Error
```
[X] Kaggle API not configured!
```
**Solution:** Download `kaggle.json` from Kaggle.com and place in `~/.kaggle/`

### GPU Not Detected
```
[!] No GPU detected. Training will use CPU (slower).
```
**Solution:** Check CUDA installation and TensorFlow-GPU setup

### Low Accuracy
**Solutions:**
- Increase training epochs
- Try EfficientNetB0 instead of MobileNetV2
- Adjust learning rate
- Add more data augmentation

---

## 📝 Notes

- Each script checks for prerequisites and provides clear error messages
- All outputs are saved to `outputs/` directory
- Models are saved to `models/saved_models/`
- TensorBoard logs are in `logs/`
- Scripts can be run independently (if prerequisites are met)

---

## 🚀 Quick Start

Run all steps in sequence:

```bash
conda activate timegan-gpu
python scripts/step1_download_dataset.py
python scripts/step2_explore_data.py
python scripts/step3_train_model.py
python scripts/step4_evaluate_model.py
python scripts/step5_realtime_detection.py
```

Total time: ~2-4 hours (depending on GPU)
