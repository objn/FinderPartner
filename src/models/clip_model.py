"""
CLIP Model wrapper for training with Hugging Face transformers
"""
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, CLIPConfig
from transformers.modeling_outputs import BaseModelOutput

logger = logging.getLogger(__name__)


class CLIPWrapper(nn.Module):
    """Wrapper around Hugging Face CLIP model for training"""
    
    def __init__(self, config: Dict):
        """Initialize CLIP wrapper
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__()
        self.config = config
        self.model_name = config['model_name']
        
        # Load pretrained CLIP model
        try:
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            logger.info(f"Loaded CLIP model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model {self.model_name}: {e}")
            raise
        
        # Model configuration
        self.vision_embed_dim = self.model.config.vision_config.hidden_size
        self.text_embed_dim = self.model.config.text_config.hidden_size
        
        # Temperature parameter for contrastive learning
        self.temperature = nn.Parameter(
            torch.log(torch.tensor(1.0 / config.get('temperature', 0.07)))
        )
        
        logger.info(f"CLIP model initialized - Vision dim: {self.vision_embed_dim}, "
                   f"Text dim: {self.text_embed_dim}")
    
    def forward(
        self, 
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through CLIP model
        
        Args:
            pixel_values: Image tensors [batch_size, 3, height, width]
            input_ids: Text token IDs [batch_size, seq_len]
            attention_mask: Text attention mask [batch_size, seq_len]
            return_loss: Whether to compute and return loss
            
        Returns:
            Dictionary with loss, logits, and embeddings
        """
        # Get vision embeddings
        vision_outputs = self.model.vision_model(pixel_values=pixel_values)
        image_embeds = self.model.visual_projection(vision_outputs.pooler_output)
        
        # Get text embeddings
        text_outputs = self.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_embeds = self.model.text_projection(text_outputs.pooler_output)
        
        # Normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        outputs = {
            'image_embeds': image_embeds,
            'text_embeds': text_embeds
        }
        
        if return_loss:
            # Compute contrastive loss
            loss = self.compute_contrastive_loss(image_embeds, text_embeds)
            outputs['loss'] = loss
            
            # Compute logits for metrics
            logit_scale = self.temperature.exp()
            logits_per_image = torch.matmul(image_embeds, text_embeds.t()) * logit_scale
            logits_per_text = logits_per_image.t()
            
            outputs.update({
                'logits_per_image': logits_per_image,
                'logits_per_text': logits_per_text
            })
        
        return outputs
    
    def compute_contrastive_loss(
        self, 
        image_embeds: torch.Tensor, 
        text_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss between image and text embeddings
        
        Args:
            image_embeds: Normalized image embeddings [batch_size, embed_dim]
            text_embeds: Normalized text embeddings [batch_size, embed_dim]
            
        Returns:
            Contrastive loss tensor
        """
        batch_size = image_embeds.shape[0]
        
        # Compute similarity matrix
        logit_scale = self.temperature.exp()
        logits_per_image = torch.matmul(image_embeds, text_embeds.t()) * logit_scale
        logits_per_text = logits_per_image.t()
        
        # Create labels (diagonal matrix for positive pairs)
        labels = torch.arange(batch_size, device=image_embeds.device)
        
        # Compute cross-entropy loss for both directions
        loss_img = nn.functional.cross_entropy(logits_per_image, labels)
        loss_txt = nn.functional.cross_entropy(logits_per_text, labels)
        
        # Average the two losses
        total_loss = (loss_img + loss_txt) / 2
        
        return total_loss
    
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images to embeddings (for inference)
        
        Args:
            pixel_values: Image tensors [batch_size, 3, height, width]
            
        Returns:
            Normalized image embeddings [batch_size, embed_dim]
        """
        with torch.no_grad():
            vision_outputs = self.model.vision_model(pixel_values=pixel_values)
            image_embeds = self.model.visual_projection(vision_outputs.pooler_output)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            return image_embeds
    
    def encode_text(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode text to embeddings (for inference)
        
        Args:
            input_ids: Text token IDs [batch_size, seq_len]
            attention_mask: Text attention mask [batch_size, seq_len]
            
        Returns:
            Normalized text embeddings [batch_size, embed_dim]
        """
        with torch.no_grad():
            text_outputs = self.model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            text_embeds = self.model.text_projection(text_outputs.pooler_output)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            return text_embeds
    
    def save_pretrained(self, save_directory: str) -> None:
        """Save model and processor to directory
        
        Args:
            save_directory: Directory to save model files
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save the underlying CLIP model and processor
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        
        # Save additional training state
        state_dict = {
            'temperature': self.temperature,
            'config': self.config
        }
        torch.save(state_dict, save_path / 'training_state.pt')
        
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def from_pretrained(
        cls, 
        model_path: str, 
        config: Optional[Dict] = None
    ) -> 'CLIPWrapper':
        """Load model from saved directory
        
        Args:
            model_path: Path to saved model directory
            config: Optional config override
            
        Returns:
            Loaded CLIPWrapper instance
        """
        model_path = Path(model_path)
        
        # Load config if not provided
        if config is None:
            state_path = model_path / 'training_state.pt'
            if state_path.exists():
                try:
                    # Try with weights_only=False for backwards compatibility
                    state = torch.load(state_path, map_location='cpu', weights_only=False)
                except Exception:
                    # Fallback for older PyTorch versions
                    state = torch.load(state_path, map_location='cpu')
                config = state.get('config', {})
            else:
                # Fallback config
                config = {'model_name': str(model_path)}
        
        # Create instance
        instance = cls(config)
        
        # Load the CLIP model
        instance.model = CLIPModel.from_pretrained(model_path)
        instance.processor = CLIPProcessor.from_pretrained(model_path)
        
        # Load training state if available
        state_path = model_path / 'training_state.pt'
        if state_path.exists():
            try:
                # Try with weights_only=False for backwards compatibility
                state = torch.load(state_path, map_location='cpu', weights_only=False)
            except Exception:
                # Fallback for older PyTorch versions
                state = torch.load(state_path, map_location='cpu')
            if 'temperature' in state:
                instance.temperature = nn.Parameter(state['temperature'])
        
        logger.info(f"Model loaded from {model_path}")
        return instance


def create_model(config: Dict) -> CLIPWrapper:
    """Factory function to create CLIP model
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized CLIPWrapper model
    """
    return CLIPWrapper(config)
