"""
Step 3: Train Sign Language Recognition Model

This script trains a deep learning model for ASL alphabet recognition using transfer learning.

Usage:
    conda activate timegan-gpu
    python scripts/step3_train_model.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_gpu():
    """Check and configure GPU."""
    print("\n" + "="*70)
    print("GPU CONFIGURATION")
    print("="*70 + "\n")
    
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"CUDA Available: {tf.test.is_built_with_cuda()}")
    
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPU Devices: {gpus}")
    
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[OK] {len(gpus)} GPU(s) configured successfully!")
            return True
        except RuntimeError as e:
            print(f"[!] GPU configuration warning: {e}")
            return False
    else:
        print("[!] No GPU detected. Training will use CPU (slower).")
        return False


def create_data_generators(data_dir, img_size=(224, 224), batch_size=32, validation_split=0.2):
    """Create training and validation data generators with augmentation."""
    print("\n" + "="*70)
    print("CREATING DATA GENERATORS")
    print("="*70 + "\n")
    
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=validation_split
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    print(f"[OK] Training samples: {train_generator.samples}")
    print(f"[OK] Validation samples: {val_generator.samples}")
    print(f"[OK] Number of classes: {train_generator.num_classes}")
    print(f"[OK] Class indices: {list(train_generator.class_indices.keys())[:10]}...")
    
    return train_generator, val_generator


def build_model(num_classes, img_size=(224, 224), model_name='MobileNetV2'):
    """Build transfer learning model."""
    print("\n" + "="*70)
    print(f"BUILDING MODEL: {model_name}")
    print("="*70 + "\n")
    
    # Load pre-trained base model
    if model_name == 'MobileNetV2':
        base_model = MobileNetV2(
            input_shape=(*img_size, 3),
            include_top=False,
            weights='imagenet'
        )
    elif model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(
            input_shape=(*img_size, 3),
            include_top=False,
            weights='imagenet'
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Freeze base model
    base_model.trainable = False
    
    # Build model
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    print(f"[OK] Model built successfully!")
    print(f"[>>] Total parameters: {model.count_params():,}")
    print(f"[>>] Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    
    return model


def train_model(model, train_gen, val_gen, epochs=50, model_dir=None, log_dir=None):
    """Train the model with callbacks."""
    print("\n" + "="*70)
    print("TRAINING MODEL")
    print("="*70 + "\n")
    
    # Create directories
    model_dir = model_dir or project_root / 'models' / 'saved_models'
    log_dir = log_dir or project_root / 'logs' / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=str(model_dir / 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1
        )
    ]
    
    print(f"[>>] Starting training for {epochs} epochs...")
    print(f"[>>] Model checkpoints: {model_dir}")
    print(f"[>>] TensorBoard logs: {log_dir}")
    print()
    
    # Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n[OK] Training completed!")
    
    return history, model_dir, log_dir


def save_training_history(history, output_dir):
    """Save training history and plots."""
    print("\n[>>] Saving training history...")
    
    # Save history as JSON
    history_dict = history.history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_json = {k: [float(v) for v in vals] for k, vals in history_dict.items()}
        json.dump(history_json, f, indent=4)
    print(f"[OK] Saved: {history_path}")
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'training_history.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {plot_path}")
    plt.close()


def save_model_summary(model, output_dir):
    """Save model architecture summary."""
    print("[>>] Saving model summary...")
    
    summary_path = output_dir / 'model_summary.txt'
    with open(summary_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"[OK] Saved: {summary_path}")


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("STEP 3: TRAIN SIGN LANGUAGE RECOGNITION MODEL")
    print("="*70)
    
    # Check GPU
    check_gpu()
    
    # Define paths
    data_dir = project_root / 'data' / 'raw' / 'asl_alphabet_train' / 'asl_alphabet_train'
    
    if not data_dir.exists():
        print(f"\n[X] Dataset not found at: {data_dir}")
        print("\nPlease run the download script first:")
        print("  python scripts/step1_download_dataset.py")
        return
    
    # Configuration
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 50
    VALIDATION_SPLIT = 0.2
    MODEL_NAME = 'MobileNetV2'  # or 'EfficientNetB0'
    
    # Create data generators
    train_gen, val_gen = create_data_generators(
        data_dir,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT
    )
    
    # Build model
    model = build_model(
        num_classes=train_gen.num_classes,
        img_size=IMG_SIZE,
        model_name=MODEL_NAME
    )
    
    # Save model summary
    output_dir = project_root / 'outputs' / 'training'
    output_dir.mkdir(parents=True, exist_ok=True)
    save_model_summary(model, output_dir)
    
    # Train model
    history, model_dir, log_dir = train_model(
        model,
        train_gen,
        val_gen,
        epochs=EPOCHS
    )
    
    # Save training history
    save_training_history(history, output_dir)
    
    # Final results
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])
    
    print("\n" + "="*70)
    print("TRAINING RESULTS")
    print("="*70)
    print(f"  Final Training Accuracy: {final_train_acc:.4f}")
    print(f"  Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"  Best Validation Accuracy: {best_val_acc:.4f}")
    
    print("\n" + "="*70)
    print("[OK] STEP 3 COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n[>>] Model saved to: {model_dir / 'best_model.keras'}")
    print(f"[>>] Training outputs: {output_dir}")
    print(f"[>>] TensorBoard logs: {log_dir}")
    print("\nTo view TensorBoard:")
    print(f"  tensorboard --logdir={log_dir}")
    print("\nNext step: Model evaluation")
    print("Command: python scripts/step4_evaluate_model.py")


if __name__ == "__main__":
    main()
