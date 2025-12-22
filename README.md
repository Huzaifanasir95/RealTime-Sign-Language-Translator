# 🤟 RealTime Sign Language Translator

A Gen-AI powered computer vision application that translates sign language gestures into text in real-time using deep learning and MediaPipe hand tracking.

## 🎯 Project Overview

This project leverages TensorFlow, OpenCV, and Google's MediaPipe to create an accessible communication tool for the deaf and hard-of-hearing community. The system detects hand landmarks in real-time and classifies sign language gestures using a trained deep learning model.

## ✨ Features

- **Real-time Hand Detection**: Uses MediaPipe for accurate 21-point hand landmark tracking
- **Deep Learning Classification**: LSTM/CNN-based model for gesture recognition
- **Custom Dataset Collection**: Tools for collecting and preprocessing your own sign language dataset
- **High Accuracy**: Optimized for >95% accuracy on test data
- **Fast Inference**: Real-time performance at 20+ FPS
- **Jupyter Notebook Workflow**: Interactive development and experimentation

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Webcam (720p minimum, 1080p recommended)
- 8GB RAM minimum (16GB recommended)
- Optional: NVIDIA GPU with CUDA support for faster training

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Huzaifanasir95/RealTime-Sign-Language-Translator.git
   cd RealTime-Sign-Language-Translator
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

## 📁 Project Structure

```
RealTime-Sign-Language-Translator/
├── notebooks/              # Jupyter notebooks for development
├── data/
│   ├── raw/               # Raw collected gesture data
│   └── processed/         # Preprocessed & augmented data
├── models/
│   └── saved_models/      # Trained model checkpoints
├── src/                   # Source code modules
├── outputs/
│   ├── logs/             # Training logs
│   ├── metrics/          # Performance metrics
│   └── visualizations/   # Plots & confusion matrices
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 🎓 Usage

### 1. Data Collection
Start by collecting your own sign language gesture dataset using the provided notebooks.

### 2. Model Training
Train the deep learning model on your collected dataset.

### 3. Real-time Detection
Run the real-time detection system to translate sign language gestures.

See the `notebooks/` directory for detailed step-by-step guides.

## 🛠️ Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **OpenCV**: Computer vision and video processing
- **MediaPipe**: Hand landmark detection
- **NumPy/Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **Jupyter**: Interactive development

## 📊 Performance Metrics

- **Target Accuracy**: >95% on test set
- **Inference Speed**: <50ms per frame (20+ FPS)
- **Model Size**: <50MB for deployment

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google MediaPipe team for the hand tracking solution
- The deaf and hard-of-hearing community for inspiration
- Open-source contributors

## 📧 Contact

**Huzaifa Nasir**
- GitHub: [@Huzaifanasir95](https://github.com/Huzaifanasir95)

---

**Made with ❤️ for accessibility and inclusion**
