"""
GPU and system configuration utilities.
"""

import tensorflow as tf
import platform
import psutil
from pathlib import Path
from typing import Optional, Dict


def get_system_info() -> Dict[str, str]:
    """
    Get system information.
    
    Returns:
        Dictionary with system info
    """
    return {
        'platform': platform.system(),
        'platform_release': platform.release(),
        'platform_version': platform.version(),
        'architecture': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(logical=True),
        'ram_total_gb': round(psutil.virtual_memory().total / (1024**3), 2)
    }


def check_tensorflow_gpu() -> Dict[str, any]:
    """
    Check TensorFlow GPU availability and configuration.
    
    Returns:
        Dictionary with GPU info
    """
    gpu_info = {
        'tensorflow_version': tf.__version__,
        'cuda_available': tf.test.is_built_with_cuda(),
        'gpu_devices': [],
        'gpu_count': 0
    }
    
    gpus = tf.config.list_physical_devices('GPU')
    gpu_info['gpu_count'] = len(gpus)
    
    for gpu in gpus:
        gpu_info['gpu_devices'].append({
            'name': gpu.name,
            'device_type': gpu.device_type
        })
    
    return gpu_info


def configure_gpu(memory_growth: bool = True,
                 memory_limit: Optional[int] = None,
                 visible_devices: Optional[list] = None) -> bool:
    """
    Configure GPU settings.
    
    Args:
        memory_growth: Enable memory growth
        memory_limit: Set memory limit in MB
        visible_devices: List of visible GPU indices
        
    Returns:
        True if successful, False otherwise
    """
    try:
        gpus = tf.config.list_physical_devices('GPU')
        
        if not gpus:
            print("[!] No GPU devices found")
            return False
        
        # Set visible devices
        if visible_devices is not None:
            visible_gpus = [gpus[i] for i in visible_devices if i < len(gpus)]
            tf.config.set_visible_devices(visible_gpus, 'GPU')
            gpus = visible_gpus
        
        # Configure each GPU
        for gpu in gpus:
            # Set memory growth
            if memory_growth:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set memory limit
            if memory_limit:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                )
        
        print(f"[OK] Configured {len(gpus)} GPU(s)")
        return True
        
    except RuntimeError as e:
        print(f"[X] GPU configuration error: {e}")
        return False


def print_system_info():
    """Print system and GPU information."""
    print("\n" + "="*70)
    print("SYSTEM INFORMATION")
    print("="*70 + "\n")
    
    sys_info = get_system_info()
    for key, value in sys_info.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("GPU INFORMATION")
    print("="*70 + "\n")
    
    gpu_info = check_tensorflow_gpu()
    print(f"  TensorFlow Version: {gpu_info['tensorflow_version']}")
    print(f"  CUDA Available: {gpu_info['cuda_available']}")
    print(f"  GPU Count: {gpu_info['gpu_count']}")
    
    if gpu_info['gpu_devices']:
        print("\n  GPU Devices:")
        for i, gpu in enumerate(gpu_info['gpu_devices']):
            print(f"    [{i}] {gpu['name']} ({gpu['device_type']})")
    else:
        print("\n  [!] No GPU devices detected")
        print("  [!] Training will use CPU (slower)")


def set_mixed_precision(enabled: bool = True):
    """
    Enable/disable mixed precision training.
    
    Args:
        enabled: Whether to enable mixed precision
    """
    if enabled:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("[OK] Mixed precision enabled (float16)")
    else:
        policy = tf.keras.mixed_precision.Policy('float32')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("[OK] Mixed precision disabled (float32)")


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import os
    import numpy as np
    import random
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    print(f"[OK] Random seeds set to {seed}")


def get_optimal_batch_size(model_size_mb: float = 50,
                           image_size: tuple = (224, 224, 3),
                           available_memory_gb: Optional[float] = None) -> int:
    """
    Estimate optimal batch size based on available memory.
    
    Args:
        model_size_mb: Estimated model size in MB
        image_size: Input image size (H, W, C)
        available_memory_gb: Available GPU memory in GB
        
    Returns:
        Recommended batch size
    """
    if available_memory_gb is None:
        # Try to get GPU memory
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Estimate 80% of typical GPU memory
            available_memory_gb = 8.0  # Conservative estimate
        else:
            # Use RAM
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    # Calculate memory per image (in MB)
    pixels = image_size[0] * image_size[1] * image_size[2]
    bytes_per_image = pixels * 4  # float32
    mb_per_image = bytes_per_image / (1024**2)
    
    # Reserve memory for model and overhead
    available_for_batch = (available_memory_gb * 1024) - model_size_mb - 1000  # 1GB overhead
    
    # Calculate batch size
    batch_size = int(available_for_batch / (mb_per_image * 2))  # *2 for gradients
    
    # Clamp to reasonable range
    batch_size = max(8, min(batch_size, 128))
    
    # Round to nearest power of 2
    batch_size = 2 ** int(np.log2(batch_size))
    
    return batch_size


if __name__ == "__main__":
    print("System configuration utilities loaded successfully!")
    print_system_info()
