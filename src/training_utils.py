"""
Training utilities and callbacks for model training.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    Callback, 
    ModelCheckpoint, 
    EarlyStopping, 
    ReduceLROnPlateau,
    TensorBoard,
    CSVLogger
)
from pathlib import Path
from datetime import datetime
from typing import Optional, List


def create_callbacks(model_dir: Path,
                    log_dir: Path,
                    monitor: str = 'val_accuracy',
                    patience: int = 10,
                    reduce_lr_patience: int = 5,
                    min_lr: float = 1e-7) -> List[Callback]:
    """
    Create training callbacks.
    
    Args:
        model_dir: Directory to save model checkpoints
        log_dir: Directory for TensorBoard logs
        monitor: Metric to monitor
        patience: Patience for early stopping
        reduce_lr_patience: Patience for learning rate reduction
        min_lr: Minimum learning rate
        
    Returns:
        List of callbacks
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        # Save best model
        ModelCheckpoint(
            filepath=str(model_dir / 'best_model.keras'),
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            mode='max' if 'acc' in monitor else 'min',
            verbose=1
        ),
        
        # Save checkpoints every epoch
        ModelCheckpoint(
            filepath=str(model_dir / 'checkpoint_epoch_{epoch:02d}.keras'),
            save_freq='epoch',
            save_best_only=False,
            verbose=0
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1,
            mode='min'
        ),
        
        # TensorBoard logging
        TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq='epoch'
        ),
        
        # CSV logging
        CSVLogger(
            filename=str(log_dir / 'training_log.csv'),
            separator=',',
            append=False
        )
    ]
    
    return callbacks


class MetricsLogger(Callback):
    """Custom callback to log metrics during training."""
    
    def __init__(self, log_file: Path):
        super().__init__()
        self.log_file = log_file
        self.epoch_logs = []
    
    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at end of each epoch."""
        logs = logs or {}
        log_entry = {
            'epoch': epoch + 1,
            'timestamp': datetime.now().isoformat(),
            **logs
        }
        self.epoch_logs.append(log_entry)
        
        # Print summary
        print(f"\nEpoch {epoch + 1} Summary:")
        for key, value in logs.items():
            print(f"  {key}: {value:.4f}")


class ProgressCallback(Callback):
    """Custom callback to show training progress."""
    
    def __init__(self, total_epochs: int):
        super().__init__()
        self.total_epochs = total_epochs
        self.start_time = None
    
    def on_train_begin(self, logs=None):
        """Called at the beginning of training."""
        self.start_time = datetime.now()
        print(f"\n{'='*70}")
        print(f"Training started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total epochs: {self.total_epochs}")
        print(f"{'='*70}\n")
    
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch."""
        print(f"\n[Epoch {epoch + 1}/{self.total_epochs}]")
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        logs = logs or {}
        elapsed = datetime.now() - self.start_time
        
        # Calculate ETA
        epochs_done = epoch + 1
        epochs_remaining = self.total_epochs - epochs_done
        avg_time_per_epoch = elapsed / epochs_done
        eta = avg_time_per_epoch * epochs_remaining
        
        print(f"\nProgress: {epochs_done}/{self.total_epochs} epochs")
        print(f"Elapsed: {str(elapsed).split('.')[0]}")
        print(f"ETA: {str(eta).split('.')[0]}")
    
    def on_train_end(self, logs=None):
        """Called at the end of training."""
        total_time = datetime.now() - self.start_time
        print(f"\n{'='*70}")
        print(f"Training completed!")
        print(f"Total time: {str(total_time).split('.')[0]}")
        print(f"{'='*70}\n")


def get_class_weights(class_counts: dict) -> dict:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        class_counts: Dictionary mapping class index to count
        
    Returns:
        Dictionary of class weights
    """
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    class_weights = {}
    for class_idx, count in class_counts.items():
        weight = total_samples / (num_classes * count)
        class_weights[class_idx] = weight
    
    return class_weights


def create_learning_rate_schedule(initial_lr: float = 0.001,
                                  decay_steps: int = 1000,
                                  decay_rate: float = 0.96,
                                  schedule_type: str = 'exponential'):
    """
    Create learning rate schedule.
    
    Args:
        initial_lr: Initial learning rate
        decay_steps: Steps for decay
        decay_rate: Decay rate
        schedule_type: Type of schedule ('exponential', 'cosine', 'step')
        
    Returns:
        Learning rate schedule
    """
    if schedule_type == 'exponential':
        return keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=False
        )
    elif schedule_type == 'cosine':
        return keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps
        )
    elif schedule_type == 'step':
        return keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[decay_steps, decay_steps * 2],
            values=[initial_lr, initial_lr * 0.1, initial_lr * 0.01]
        )
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


if __name__ == "__main__":
    print("Training utilities loaded successfully!")
