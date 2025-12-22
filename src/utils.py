"""
Utility functions for the Sign Language Translator project.
"""

import os
import json
import yaml
import numpy as np
from pathlib import Path


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_json(data, filepath):
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filepath):
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def create_directory(directory):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)
    

def get_class_names(data_dir):
    """Get class names from directory structure."""
    class_names = sorted([d for d in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, d))])
    return class_names


if __name__ == "__main__":
    print("Utility functions loaded successfully!")
