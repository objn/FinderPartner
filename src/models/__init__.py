"""
Models module for CLIP training pipeline
"""

from .clip_model import CLIPWrapper, create_model

__all__ = [
    'CLIPWrapper',
    'create_model'
]
