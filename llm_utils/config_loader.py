"""
Configuration loading utilities
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def validate_model_config(config: Dict[str, Any]) -> None:
    """
    Validate that the model configuration contains required fields.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If required fields are missing
    """
    required_fields = ['model_name']
    
    if 'model' not in config:
        raise ValueError("Configuration must contain 'model' section")
    
    model_config = config['model']
    
    for field in required_fields:
        if field not in model_config:
            raise ValueError(f"Model configuration missing required field: {field}")


def validate_dataset_config(config: Dict[str, Any]) -> None:
    """
    Validate that the dataset configuration contains required fields.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If required fields are missing
    """
    if 'dataset' not in config:
        raise ValueError("Configuration must contain 'dataset' section")
    
    dataset_config = config['dataset']
    
    # Must have either 'name' (for HuggingFace) or 'path' (for local)
    if 'name' not in dataset_config and 'path' not in dataset_config:
        raise ValueError("Dataset configuration must contain either 'name' or 'path'")
