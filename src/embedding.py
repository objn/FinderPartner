"""
CLIP embedding functions for text and image encoding
"""
import logging
from typing import List, Optional, Tuple
import torch
import open_clip
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class CLIPEmbedder:
    """CLIP model wrapper for text and image embedding"""
    
    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai"):
        """Initialize CLIP model
        
        Args:
            model_name: CLIP model architecture name
            pretrained: Pretrained weights to use
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading CLIP model {model_name} ({pretrained}) on {self.device}")
        
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.model.eval()
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to normalized embedding vector
        
        Args:
            text: Input text prompt
            
        Returns:
            Normalized embedding vector as numpy array
        """
        try:
            with torch.no_grad():
                text_tokens = self.tokenizer([text]).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                return text_features.cpu().numpy()[0]
        except Exception as e:
            logger.error(f"Failed to encode text '{text}': {e}")
            raise
    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode PIL Image to normalized embedding vector
        
        Args:
            image: PIL Image object
            
        Returns:
            Normalized embedding vector as numpy array
        """
        try:
            with torch.no_grad():
                image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                image_features = self.model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                return image_features.cpu().numpy()[0]
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            raise
    
    def encode_images_batch(self, images: List[Image.Image], batch_size: int = 32) -> List[np.ndarray]:
        """Encode multiple images in batches for efficiency
        
        Args:
            images: List of PIL Image objects
            batch_size: Number of images to process at once
            
        Returns:
            List of normalized embedding vectors
        """
        embeddings = []
        
        try:
            with torch.no_grad():
                for i in range(0, len(images), batch_size):
                    batch = images[i:i + batch_size]
                    batch_tensors = torch.stack([
                        self.preprocess(img) for img in batch
                    ]).to(self.device)
                    
                    batch_features = self.model.encode_image(batch_tensors)
                    batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
                    
                    embeddings.extend(batch_features.cpu().numpy())
            
            return embeddings
        except Exception as e:
            logger.error(f"Failed to encode image batch: {e}")
            raise