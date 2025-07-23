"""
AI Profile Matcher

A CLIP-based profile evaluation system that matches text descriptions 
with profile images to make LIKE/UNLIKE decisions.
"""

__version__ = "1.0.0"
__author__ = "AI Developer"

from .embedding import CLIPEmbedder
from .scorer import ProfileScorer
from .utils import load_images_from_directory, validate_image_path

__all__ = [
    'CLIPEmbedder',
    'ProfileScorer', 
    'load_images_from_directory',
    'validate_image_path'
]