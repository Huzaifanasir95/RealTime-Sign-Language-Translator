"""
Model building utilities for sign language recognition.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import (
    MobileNetV2, 
    EfficientNetB0, 
    ResNet50,
    VGG16
)
from typing import Tuple, Optional


def create_base_model(model_name: str, 
                      input_shape: Tuple[int, int, int] = (224, 224, 3),
                      include_top: bool = False,
                      weights: str = 'imagenet') -> keras.Model:
    """
    Create a pre-trained base model.
    
    Args:
        model_name: Name of the model architecture
        input_shape: Input shape (height, width, channels)
        include_top: Whether to include top classification layer
        weights: Pre-trained weights to use
        
    Returns:
        Base model
    """
    model_map = {
        'MobileNetV2': MobileNetV2,
        'EfficientNetB0': EfficientNetB0,
        'ResNet50': ResNet50,
        'VGG16': VGG16
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_map.keys())}")
    
    base_model = model_map[model_name](
        input_shape=input_shape,
        include_top=include_top,
        weights=weights
    )
    
    return base_model


def build_transfer_learning_model(num_classes: int,
                                  base_model_name: str = 'MobileNetV2',
                                  input_shape: Tuple[int, int, int] = (224, 224, 3),
                                  dropout_rate: float = 0.5,
                                  dense_units: int = 512,
                                  freeze_base: bool = True) -> keras.Model:
    """
    Build a transfer learning model for sign language recognition.
    
    Args:
        num_classes: Number of output classes
        base_model_name: Name of base model architecture
        input_shape: Input shape
        dropout_rate: Dropout rate for regularization
        dense_units: Number of units in dense layer
        freeze_base: Whether to freeze base model weights
        
    Returns:
        Compiled model
    """
    # Create base model
    base_model = create_base_model(
        model_name=base_model_name,
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model if specified
    base_model.trainable = not freeze_base
    
    # Build model
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(dropout_rate),
        layers.Dense(dense_units, activation='relu'),
        layers.Dropout(dropout_rate * 0.6),
        layers.Dense(num_classes, activation='softmax')
    ], name=f'SignLanguage_{base_model_name}')
    
    return model


def build_custom_cnn_model(num_classes: int,
                           input_shape: Tuple[int, int, int] = (224, 224, 3),
                           dropout_rate: float = 0.5) -> keras.Model:
    """
    Build a custom CNN model from scratch.
    
    Args:
        num_classes: Number of output classes
        input_shape: Input shape
        dropout_rate: Dropout rate
        
    Returns:
        Compiled model
    """
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate * 0.5),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate * 0.5),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate * 0.6),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate * 0.7),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ], name='SignLanguage_CustomCNN')
    
    return model


def compile_model(model: keras.Model,
                 learning_rate: float = 0.001,
                 optimizer: str = 'adam',
                 loss: str = 'categorical_crossentropy',
                 metrics: Optional[list] = None) -> keras.Model:
    """
    Compile a Keras model.
    
    Args:
        model: Model to compile
        learning_rate: Learning rate
        optimizer: Optimizer name
        loss: Loss function
        metrics: List of metrics
        
    Returns:
        Compiled model
    """
    if metrics is None:
        metrics = [
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
        ]
    
    # Create optimizer
    if optimizer == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer == 'rmsprop':
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    
    # Compile
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=metrics
    )
    
    return model


def get_model_summary(model: keras.Model) -> str:
    """
    Get model summary as string.
    
    Args:
        model: Keras model
        
    Returns:
        Model summary string
    """
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    return '\n'.join(summary_lines)


def count_parameters(model: keras.Model) -> dict:
    """
    Count model parameters.
    
    Args:
        model: Keras model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params
    }


if __name__ == "__main__":
    print("Model building utilities loaded successfully!")
