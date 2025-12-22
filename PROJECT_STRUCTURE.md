# Project Structure Documentation

## 📁 Complete Folder Structure

```
RealTime-Sign-Language-Translator/
├── data/                          # Dataset storage
│   ├── raw/                       # Original downloaded data
│   ├── processed/                 # Preprocessed data
│   ├── augmented/                 # Augmented data
│   └── test_samples/              # Test samples
│
├── models/                        # Trained models
│   ├── saved_models/              # Best models (.keras files)
│   ├── checkpoints/               # Training checkpoints
│   └── exports/                   # Exported models (TFLite, ONNX)
│
├── src/                           # Source code modules
│   ├── __init__.py                # Package initializer
│   ├── utils.py                   # General utilities
│   ├── config.py                  # System & GPU configuration
│   ├── dataset_utils.py           # Dataset management
│   ├── data_loader.py             # Data loading & generators
│   ├── preprocessing.py           # Image preprocessing
│   ├── model_builder.py           # Model architectures
│   ├── training_utils.py          # Training callbacks & utilities
│   ├── evaluation.py              # Evaluation metrics
│   ├── visualization.py           # Plotting & visualization
│   └── realtime_detection.py      # Real-time detection
│
├── scripts/                       # Executable scripts
│   ├── README.md                  # Scripts documentation
│   ├── step1_download_dataset.py  # Download dataset
│   ├── step2_explore_data.py      # Data exploration
│   ├── step3_train_model.py       # Model training
│   ├── step4_evaluate_model.py    # Model evaluation
│   └── step5_realtime_detection.py # Real-time detection
│
├── notebooks/                     # Jupyter notebooks (optional)
│   └── 01_data_exploration.ipynb
│
├── app/                           # Application files
│   ├── streamlit_app.py           # Streamlit web interface
│   └── real_time_detector.py      # Standalone detector app
│
├── configs/                       # Configuration files
│   └── config.yaml                # Main configuration
│
├── logs/                          # Training logs
│   └── training_YYYYMMDD_HHMMSS/  # TensorBoard logs
│
├── outputs/                       # Output files
│   ├── exploration/               # Data exploration outputs
│   ├── training/                  # Training outputs
│   └── evaluation/                # Evaluation outputs
│
├── tests/                         # Unit tests
│
├── requirements.txt               # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # Project documentation
```

---

## 🔧 Source Modules (`src/`)

### Core Utilities

#### `utils.py`
- General utility functions
- Config loading (YAML, JSON)
- Directory management
- Class name extraction

#### `config.py`
- System information retrieval
- GPU detection and configuration
- Mixed precision training setup
- Random seed setting
- Optimal batch size calculation

---

### Data Management

#### `dataset_utils.py`
- Kaggle API integration
- Dataset downloading and extraction
- Dataset structure verification
- Train/validation splitting
- Dataset cleaning

#### `data_loader.py`
- ASL dataset loader class
- Data generator creation
- Image counting per class
- Batch loading utilities

#### `preprocessing.py`
- Image loading and resizing
- Normalization (standard, min-max)
- Data augmentation (rotation, brightness, zoom)
- Hand region extraction
- Model input preprocessing

---

### Model Development

#### `model_builder.py`
- Pre-trained model loading (MobileNetV2, EfficientNetB0, ResNet50, VGG16)
- Transfer learning model builder
- Custom CNN architecture
- Model compilation
- Parameter counting

#### `training_utils.py`
- Training callbacks creation
- Custom callbacks (MetricsLogger, ProgressCallback)
- Class weight calculation
- Learning rate scheduling
- Early stopping and checkpointing

---

### Evaluation & Visualization

#### `evaluation.py`
- Comprehensive metrics calculation
- Confusion matrix plotting
- Per-class accuracy visualization
- Classification report generation
- Training history plotting
- Misclassified sample analysis

#### `visualization.py`
- Class distribution plots
- Sample image grids
- Class variation visualization
- Image property analysis
- Augmentation examples
- Prediction visualization

---

### Real-Time Detection

#### `realtime_detection.py`
- SignLanguageDetector class
- Webcam capture utility
- Frame preprocessing
- Prediction smoothing
- FPS tracking
- Overlay drawing
- Real-time detection runner

---

## 🎯 Scripts Workflow (`scripts/`)

### Step 1: Download Dataset
**File**: `step1_download_dataset.py`

**Modules Used**:
- `src.dataset_utils` - Kaggle API, downloading, verification

**What it does**:
1. Checks Kaggle API configuration
2. Downloads ASL Alphabet dataset (~87K images)
3. Extracts and verifies dataset
4. Counts classes and images

**Output**: Dataset in `data/raw/`

---

### Step 2: Explore Data
**File**: `step2_explore_data.py`

**Modules Used**:
- `src.dataset_utils` - Class counting
- `src.visualization` - Plotting functions
- `src.preprocessing` - Image loading

**What it does**:
1. Analyzes class distribution
2. Creates visualizations (bar charts, sample images)
3. Analyzes image properties
4. Generates exploration report

**Outputs**:
- `outputs/exploration/class_distribution.png`
- `outputs/exploration/sample_images.png`
- `outputs/exploration/variations_class_A.png`
- `outputs/exploration/data_exploration_report.txt`

---

### Step 3: Train Model
**File**: `step3_train_model.py`

**Modules Used**:
- `src.config` - GPU configuration
- `src.data_loader` - Data generators
- `src.model_builder` - Model creation
- `src.training_utils` - Callbacks
- `src.evaluation` - History plotting

**What it does**:
1. Configures GPU
2. Creates data generators with augmentation
3. Builds transfer learning model
4. Trains with callbacks (checkpoint, early stopping, etc.)
5. Saves model and training history

**Outputs**:
- `models/saved_models/best_model.keras`
- `outputs/training/training_history.json`
- `outputs/training/training_history.png`
- `outputs/training/model_summary.txt`
- `logs/training_YYYYMMDD_HHMMSS/` (TensorBoard)

---

### Step 4: Evaluate Model
**File**: `step4_evaluate_model.py`

**Modules Used**:
- `src.data_loader` - Test data generator
- `src.evaluation` - Metrics, confusion matrix, reports

**What it does**:
1. Loads trained model
2. Evaluates on test data
3. Generates confusion matrix
4. Creates classification report
5. Plots per-class accuracy

**Outputs**:
- `outputs/evaluation/confusion_matrix.png`
- `outputs/evaluation/classification_report.txt`
- `outputs/evaluation/per_class_accuracy.png`
- `outputs/evaluation/evaluation_summary.json`

---

### Step 5: Real-Time Detection
**File**: `step5_realtime_detection.py`

**Modules Used**:
- `src.realtime_detection` - Detector class, webcam capture

**What it does**:
1. Loads trained model
2. Opens webcam
3. Detects signs in real-time
4. Shows predictions with confidence
5. Displays FPS

**Controls**: Press 'q' to quit

---

## 🔄 Module Dependencies

```
scripts/
  ├── step1_download_dataset.py
  │   └── src.dataset_utils
  │
  ├── step2_explore_data.py
  │   ├── src.dataset_utils
  │   ├── src.visualization
  │   └── src.preprocessing
  │
  ├── step3_train_model.py
  │   ├── src.config
  │   ├── src.data_loader
  │   ├── src.model_builder
  │   ├── src.training_utils
  │   └── src.evaluation
  │
  ├── step4_evaluate_model.py
  │   ├── src.data_loader
  │   └── src.evaluation
  │
  └── step5_realtime_detection.py
      └── src.realtime_detection
```

---

## 📦 Module Reusability

Each module in `src/` is designed to be:
- **Independent**: Can be imported and used separately
- **Reusable**: Functions can be called from any script
- **Well-documented**: Clear docstrings for all functions
- **Type-hinted**: Type hints for better IDE support
- **Tested**: Can be unit tested individually

---

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   conda activate timegan-gpu
   pip install -r requirements.txt
   ```

2. **Run scripts sequentially**:
   ```bash
   python scripts/step1_download_dataset.py
   python scripts/step2_explore_data.py
   python scripts/step3_train_model.py
   python scripts/step4_evaluate_model.py
   python scripts/step5_realtime_detection.py
   ```

3. **Or use modules directly in your code**:
   ```python
   from src.model_builder import build_transfer_learning_model
   from src.config import configure_gpu
   from src.evaluation import calculate_metrics
   
   # Your custom code here
   ```

---

## 📊 Total Files Created

- **Source Modules**: 11 files in `src/`
- **Scripts**: 5 files in `scripts/` + README
- **Config Files**: 1 file in `configs/`
- **Documentation**: 2 README files
- **Total**: ~20 Python files with modular, reusable code

---

## ✨ Benefits of This Structure

1. **Modularity**: Each module has a single responsibility
2. **Reusability**: Functions can be used across different scripts
3. **Maintainability**: Easy to update and debug individual modules
4. **Scalability**: Easy to add new features or models
5. **Testability**: Each module can be unit tested
6. **Readability**: Clear organization and documentation
7. **Professional**: Industry-standard project structure

---

## 🎓 Learning Path

1. Start with `scripts/` to understand the workflow
2. Explore `src/` modules to see implementation details
3. Modify modules for custom functionality
4. Create new scripts using existing modules
5. Add unit tests in `tests/` directory

---

This structure follows best practices for ML/DL projects and is production-ready!
