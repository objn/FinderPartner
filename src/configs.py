"""
Configuration management for CLIP training pipeline
"""
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # Validate required fields
        required_fields = [
            'model_name', 'train_csv', 'val_csv', 
            'train_images_dir', 'val_images_dir'
        ]
        
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            raise ValueError(f"Missing required config fields: {missing_fields}")
        
        # Convert path strings to Path objects and resolve them
        path_fields = [
            'train_csv', 'val_csv', 'train_images_dir', 'val_images_dir',
            'test_csv', 'test_images_dir', 'output_dir'
        ]
        
        for field in path_fields:
            if field in config and config[field]:
                config[field] = Path(config[field]).resolve()
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML config: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading config: {e}")


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save YAML configuration file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert Path objects to strings for YAML serialization
    config_copy = config.copy()
    for key, value in config_copy.items():
        if isinstance(value, Path):
            config_copy[key] = str(value)
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_copy, f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved configuration to {config_path}")
        
    except Exception as e:
        raise RuntimeError(f"Error saving config: {e}")


def get_default_config() -> Dict[str, Any]:
    """Get default configuration dictionary
    
    Returns:
        Default configuration dictionary
    """
    return {
        # Model configuration
        'model_name': 'openai/clip-vit-base-patch32',
        'image_size': 224,
        'max_length': 77,
        'temperature': 0.07,
        
        # Training configuration
        'batch_size': 32,
        'epochs': 5,
        'lr': 5e-5,
        'weight_decay': 1e-2,
        'warmup_steps': 500,
        'save_steps': 1000,
        'eval_steps': 500,
        'gradient_accumulation_steps': 1,
        'max_grad_norm': 1.0,
        
        # Data configuration
        'num_workers': 4,
        'target_memory_gb': 8.0,
        
        # Paths (to be set by user)
        'train_csv': 'data/captions_train.csv',
        'val_csv': 'data/captions_val.csv',
        'train_images_dir': 'data/images/train',
        'val_images_dir': 'data/images/val',
        'output_dir': 'outputs',
        
        # Optional test data
        'test_csv': None,
        'test_images_dir': None,
        
        # Logging configuration
        'log_wandb': True,
        'project': 'FinderPartner-CLIP',
        'run_name': None,
        'log_dir': 'logs',
        
        # Device configuration
        'device': 'auto',  # 'auto', 'cuda', 'cpu'
        'mixed_precision': True,
        
        # Evaluation configuration
        'compute_retrieval_metrics': True,
        'retrieval_k_values': [1, 5, 10]
    }


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration parameters
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check required fields
    required_fields = [
        'model_name', 'train_csv', 'val_csv', 
        'train_images_dir', 'val_images_dir'
    ]
    
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        raise ValueError(f"Missing required config fields: {missing_fields}")
    
    # Check data paths exist
    path_checks = [
        ('train_csv', 'Training CSV file'),
        ('val_csv', 'Validation CSV file'),
        ('train_images_dir', 'Training images directory'),
        ('val_images_dir', 'Validation images directory')
    ]
    
    for field, description in path_checks:
        path = config.get(field)
        if path and not Path(path).exists():
            raise ValueError(f"{description} not found: {path}")
    
    # Check numeric parameters
    if config.get('batch_size', 0) <= 0:
        raise ValueError("batch_size must be positive")
    
    if config.get('lr', 0) <= 0:
        raise ValueError("Learning rate must be positive")
    
    if config.get('epochs', 0) <= 0:
        raise ValueError("epochs must be positive")
    
    # Check image size
    image_size = config.get('image_size', 224)
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("image_size must be a positive integer")
    
    logger.info("Configuration validation passed")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries
    
    Args:
        base_config: Base configuration dictionary
        override_config: Configuration to override base with
        
    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def setup_experiment_dir(config: Dict[str, Any]) -> Path:
    """Setup experiment directory structure
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Path to experiment directory
    """
    import datetime
    
    # Create run name if not provided
    if not config.get('run_name'):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config['run_name'] = f"clip_training_{timestamp}"
    
    # Setup output directory
    output_dir = Path(config['output_dir']) / config['run_name']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)
    (output_dir / 'configs').mkdir(exist_ok=True)
    
    # Save config to experiment directory
    save_config(config, output_dir / 'configs' / 'config.yaml')
    
    logger.info(f"Experiment directory created: {output_dir}")
    return output_dir
