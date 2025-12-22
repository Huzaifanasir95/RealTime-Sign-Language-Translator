# Real-Time Sign Language Translator

A Gen-AI powered system that translates sign language gestures into text/speech in real-time using computer vision and deep learning.

## 🚀 Project Status
Currently in development - Setting up project structure

## 📁 Project Structure

```
RealTime-Sign-Language-Translator/
├── data/                      # All dataset files
│   ├── raw/                   # Original downloaded datasets
│   ├── processed/             # Preprocessed images/data
│   ├── augmented/             # Augmented training data
│   └── test_samples/          # Sample images for testing
│
├── models/                    # Model-related files
│   ├── saved_models/          # Trained model files (.h5, .keras)
│   ├── checkpoints/           # Training checkpoints
│   └── exports/               # Exported models (TFLite, ONNX)
│
├── notebooks/                 # Jupyter notebooks for experimentation
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_real_time_testing.ipynb
│
├── src/                       # Source code modules
│   ├── __init__.py
│   ├── data_loader.py         # Dataset loading utilities
│   ├── preprocessing.py       # Image preprocessing functions
│   ├── augmentation.py        # Data augmentation
│   ├── model_builder.py       # Model architectures
│   ├── train.py               # Training pipeline
│   ├── evaluate.py            # Evaluation metrics
│   ├── predict.py             # Inference functions
│   └── utils.py               # Helper functions
│
├── app/                       # Application files
│   ├── streamlit_app.py       # Streamlit web interface
│   ├── real_time_detector.py  # Real-time detection system
│   └── text_to_speech.py      # TTS functionality
│
├── configs/                   # Configuration files
│   ├── config.yaml            # Main configuration
│   └── model_config.yaml      # Model hyperparameters
│
├── tests/                     # Unit tests
│   └── test_preprocessing.py
│
├── logs/                      # Training logs and TensorBoard
│
├── outputs/                   # Output files (predictions, visualizations)
│
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore file
└── README.md                 # This file
```

## 🛠️ Tech Stack

- **Deep Learning**: TensorFlow/Keras with CUDA support
- **Computer Vision**: OpenCV, MediaPipe
- **UI**: Streamlit
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn

## 📊 Dataset

Using ASL Alphabet Dataset from Kaggle (87,000 images, 29 classes)

## 🎯 Features

- Real-time sign language detection via webcam
- High accuracy gesture recognition (Target: >95%)
- Text-to-speech output
- User-friendly web interface
- Support for ASL alphabet (A-Z + Space, Delete, Nothing)

## 🚀 Getting Started

Coming soon...

## 📝 License

MIT License

## 👨‍💻 Author

Huzaifa Nasir
