"""
AI Profile Matcher

A CLIP-based profile evaluation system that matches text descriptions 
with profile images to make LIKE/UNLIKE decisions.

Now includes complete CLIP training pipeline for custom datasets.
"""

__version__ = "2.0.0"
__author__ = "AI Developer"

from .embedding import CLIPEmbedder
from .scorer import ProfileScorer
from .utils import load_images_from_directory, validate_image_path

# Training pipeline components
try:
    from .models import CLIPWrapper, create_model
    from .data import CLIPPairedDataset, create_dataloaders
    from .eval import evaluate_model, MetricsTracker
    from .configs import load_config, get_default_config
    
    __all__ = [
        'CLIPEmbedder',
        'ProfileScorer', 
        'load_images_from_directory',
        'validate_image_path',
        'CLIPWrapper',
        'create_model',
        'CLIPPairedDataset',
        'create_dataloaders',
        'evaluate_model',
        'MetricsTracker',
        'load_config',
        'get_default_config'
    ]
except ImportError:
    # Training components not available (missing dependencies)
    __all__ = [
        'CLIPEmbedder',
        'ProfileScorer', 
        'load_images_from_directory',
        'validate_image_path'
    ]