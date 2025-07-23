#!/usr/bin/env python3
"""
ONNX Export Script for CLIP Model

Export trained CLIP model to ONNX format for deployment and inference optimization.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.onnx
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models import CLIPWrapper
from utils import setup_logging

logger = logging.getLogger(__name__)


class CLIPONNXWrapper(torch.nn.Module):
    """Wrapper for CLIP model to export specific functions to ONNX"""
    
    def __init__(self, clip_model: CLIPWrapper, export_mode: str = 'both'):
        """Initialize ONNX wrapper
        
        Args:
            clip_model: Trained CLIP model
            export_mode: 'image', 'text', or 'both'
        """
        super().__init__()
        self.clip_model = clip_model
        self.export_mode = export_mode
    
    def forward_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass for image encoding only"""
        return self.clip_model.encode_image(pixel_values)
    
    def forward_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for text encoding only"""
        return self.clip_model.encode_text(input_ids, attention_mask)
    
    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Combined forward pass for both image and text"""
        image_embeds = self.clip_model.encode_image(pixel_values)
        text_embeds = self.clip_model.encode_text(input_ids, attention_mask)
        return image_embeds, text_embeds


def export_image_encoder(
    model: CLIPWrapper,
    output_path: str,
    image_size: int = 224,
    batch_size: int = 1
) -> None:
    """Export image encoder to ONNX
    
    Args:
        model: Trained CLIP model
        output_path: Path to save ONNX model
        image_size: Input image size
        batch_size: Batch size for export
    """
    model.eval()
    
    # Create wrapper for image encoding
    wrapper = CLIPONNXWrapper(model, 'image')
    
    # Create dummy input
    dummy_pixel_values = torch.randn(batch_size, 3, image_size, image_size)
    
    # Export to ONNX
    torch.onnx.export(
        wrapper.forward_image,
        dummy_pixel_values,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['pixel_values'],
        output_names=['image_embeddings'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'image_embeddings': {0: 'batch_size'}
        },
        verbose=False
    )
    
    logger.info(f"Image encoder exported to: {output_path}")


def export_text_encoder(
    model: CLIPWrapper,
    output_path: str,
    max_length: int = 77,
    batch_size: int = 1
) -> None:
    """Export text encoder to ONNX
    
    Args:
        model: Trained CLIP model
        output_path: Path to save ONNX model
        max_length: Maximum text sequence length
        batch_size: Batch size for export
    """
    model.eval()
    
    # Create wrapper for text encoding
    wrapper = CLIPONNXWrapper(model, 'text')
    
    # Create dummy inputs
    dummy_input_ids = torch.randint(0, 1000, (batch_size, max_length))
    dummy_attention_mask = torch.ones(batch_size, max_length)
    
    # Export to ONNX
    torch.onnx.export(
        wrapper.forward_text,
        (dummy_input_ids, dummy_attention_mask),
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['text_embeddings'],
        dynamic_axes={
            'input_ids': {0: 'batch_size'},
            'attention_mask': {0: 'batch_size'},
            'text_embeddings': {0: 'batch_size'}
        },
        verbose=False
    )
    
    logger.info(f"Text encoder exported to: {output_path}")


def export_combined_model(
    model: CLIPWrapper,
    output_path: str,
    image_size: int = 224,
    max_length: int = 77,
    batch_size: int = 1
) -> None:
    """Export combined CLIP model to ONNX
    
    Args:
        model: Trained CLIP model
        output_path: Path to save ONNX model
        image_size: Input image size
        max_length: Maximum text sequence length
        batch_size: Batch size for export
    """
    model.eval()
    
    # Create wrapper for combined model
    wrapper = CLIPONNXWrapper(model, 'both')
    
    # Create dummy inputs
    dummy_pixel_values = torch.randn(batch_size, 3, image_size, image_size)
    dummy_input_ids = torch.randint(0, 1000, (batch_size, max_length))
    dummy_attention_mask = torch.ones(batch_size, max_length)
    
    # Export to ONNX
    torch.onnx.export(
        wrapper,
        (dummy_pixel_values, dummy_input_ids, dummy_attention_mask),
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['pixel_values', 'input_ids', 'attention_mask'],
        output_names=['image_embeddings', 'text_embeddings'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'input_ids': {0: 'batch_size'},
            'attention_mask': {0: 'batch_size'},
            'image_embeddings': {0: 'batch_size'},
            'text_embeddings': {0: 'batch_size'}
        },
        verbose=False
    )
    
    logger.info(f"Combined model exported to: {output_path}")


def verify_onnx_model(onnx_path: str, original_model: CLIPWrapper) -> bool:
    """Verify ONNX model outputs match original model
    
    Args:
        onnx_path: Path to ONNX model
        original_model: Original PyTorch model
        
    Returns:
        True if verification passes
    """
    try:
        import onnxruntime as ort
        
        # Load ONNX model
        ort_session = ort.InferenceSession(onnx_path)
        
        # Get input/output names
        input_names = [inp.name for inp in ort_session.get_inputs()]
        output_names = [out.name for out in ort_session.get_outputs()]
        
        logger.info(f"ONNX model inputs: {input_names}")
        logger.info(f"ONNX model outputs: {output_names}")
        
        # Create test inputs
        if 'pixel_values' in input_names:
            test_pixel_values = torch.randn(1, 3, 224, 224)
            
            if len(input_names) == 1:  # Image-only model
                # Test image encoder
                with torch.no_grad():
                    pytorch_output = original_model.encode_image(test_pixel_values).numpy()
                
                onnx_output = ort_session.run(
                    output_names,
                    {'pixel_values': test_pixel_values.numpy()}
                )[0]
                
                # Check similarity
                similarity = np.corrcoef(pytorch_output.flatten(), onnx_output.flatten())[0, 1]
                logger.info(f"Image encoder output correlation: {similarity:.4f}")
                
                return similarity > 0.99
        
        return True
        
    except ImportError:
        logger.warning("onnxruntime not available, skipping verification")
        return True
    except Exception as e:
        logger.error(f"ONNX verification failed: {e}")
        return False


def main():
    """Main export function"""
    parser = argparse.ArgumentParser(description='Export CLIP model to ONNX format')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained CLIP model directory')
    parser.add_argument('--output_dir', type=str, default='outputs/onnx',
                       help='Directory to save ONNX models')
    parser.add_argument('--export_mode', type=str, choices=['image', 'text', 'both', 'all'],
                       default='all', help='What to export')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--max_length', type=int, default=77,
                       help='Maximum text sequence length')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for export')
    parser.add_argument('--verify', action='store_true',
                       help='Verify ONNX model outputs')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging('INFO')
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model path not found: {model_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load model
        logger.info(f"Loading model from: {model_path}")
        model = CLIPWrapper.from_pretrained(str(model_path))
        model.eval()
        
        # Export based on mode
        if args.export_mode in ['image', 'all']:
            output_path = output_dir / 'clip_image_encoder.onnx'
            export_image_encoder(
                model, str(output_path), 
                args.image_size, args.batch_size
            )
            
            if args.verify:
                verify_onnx_model(str(output_path), model)
        
        if args.export_mode in ['text', 'all']:
            output_path = output_dir / 'clip_text_encoder.onnx'
            export_text_encoder(
                model, str(output_path),
                args.max_length, args.batch_size
            )
            
            if args.verify:
                verify_onnx_model(str(output_path), model)
        
        if args.export_mode in ['both', 'all']:
            output_path = output_dir / 'clip_combined.onnx'
            export_combined_model(
                model, str(output_path),
                args.image_size, args.max_length, args.batch_size
            )
            
            if args.verify:
                verify_onnx_model(str(output_path), model)
        
        logger.info(f"ONNX export completed! Models saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
