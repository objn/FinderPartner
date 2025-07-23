#!/usr/bin/env python3
"""
Gradio Demo for CLIP Model

Interactive web interface to test CLIP model similarity scoring between images and text.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models import CLIPWrapper
from utils import setup_logging

logger = logging.getLogger(__name__)


class CLIPDemo:
    """CLIP model demo interface"""
    
    def __init__(self, model_path: str):
        """Initialize demo with trained model
        
        Args:
            model_path: Path to trained CLIP model
        """
        self.model_path = model_path
        self.model = None
        self.device = None
        self._load_model()
    
    def _load_model(self):
        """Load the CLIP model"""
        try:
            import torch
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Loading model on {self.device}")
            
            self.model = CLIPWrapper.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict_similarity(
        self, 
        image: Image.Image, 
        text: str
    ) -> Tuple[float, str, dict]:
        """Predict similarity between image and text
        
        Args:
            image: PIL Image
            text: Text description
            
        Returns:
            Tuple of (similarity_score, interpretation, detailed_info)
        """
        if self.model is None:
            return 0.0, "Model not loaded", {}
        
        if image is None or not text.strip():
            return 0.0, "Please provide both image and text", {}
        
        try:
            import torch
            
            with torch.no_grad():
                # Process image
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Use the model's processor for consistent preprocessing
                pixel_values = self.model.processor(
                    images=image, return_tensors="pt"
                )["pixel_values"].to(self.device)
                
                # Process text
                text_inputs = self.model.processor(
                    text=[text], return_tensors="pt", padding=True, truncation=True
                ).to(self.device)
                
                # Get embeddings
                image_embeds = self.model.encode_image(pixel_values)
                text_embeds = self.model.encode_text(
                    text_inputs['input_ids'], 
                    text_inputs['attention_mask']
                )
                
                # Calculate similarity
                similarity = torch.cosine_similarity(
                    image_embeds, text_embeds, dim=-1
                ).item()
                
                # Generate interpretation
                if similarity >= 0.4:
                    interpretation = "🎯 Excellent match!"
                elif similarity >= 0.3:
                    interpretation = "✅ Good match"
                elif similarity >= 0.2:
                    interpretation = "🤔 Moderate match"
                elif similarity >= 0.1:
                    interpretation = "❌ Poor match"
                else:
                    interpretation = "💀 No match"
                
                # Detailed info
                detailed_info = {
                    'similarity_score': f"{similarity:.4f}",
                    'image_shape': f"{image.size[0]}x{image.size[1]}",
                    'text_length': len(text),
                    'device': str(self.device),
                    'model_path': str(Path(self.model_path).name)
                }
                
                return similarity, interpretation, detailed_info
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.0, f"Error: {str(e)}", {}
    
    def batch_predict(
        self, 
        image: Image.Image, 
        text_list: list
    ) -> list:
        """Predict similarity for multiple texts with single image
        
        Args:
            image: PIL Image
            text_list: List of text descriptions
            
        Returns:
            List of (text, similarity_score) tuples sorted by score
        """
        if not text_list or image is None:
            return []
        
        results = []
        for text in text_list:
            if text.strip():
                similarity, _, _ = self.predict_similarity(image, text.strip())
                results.append((text.strip(), similarity))
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results


def create_gradio_interface(demo: CLIPDemo):
    """Create Gradio interface
    
    Args:
        demo: CLIPDemo instance
        
    Returns:
        Gradio interface
    """
    try:
        import gradio as gr
    except ImportError:
        logger.error("Gradio not installed. Install with: pip install gradio")
        raise
    
    def predict_wrapper(image, text):
        """Wrapper for Gradio interface"""
        similarity, interpretation, details = demo.predict_similarity(image, text)
        
        # Format details for display
        details_text = "\n".join([f"{k}: {v}" for k, v in details.items()])
        
        return similarity, interpretation, details_text
    
    def batch_predict_wrapper(image, text_input):
        """Wrapper for batch prediction"""
        if not text_input.strip():
            return "Please enter text descriptions (one per line)"
        
        text_list = [line.strip() for line in text_input.split('\n') if line.strip()]
        results = demo.batch_predict(image, text_list)
        
        if not results:
            return "No valid text inputs provided"
        
        # Format results
        output_lines = ["**Results (sorted by similarity):**\n"]
        for i, (text, score) in enumerate(results[:10], 1):  # Top 10
            output_lines.append(f"{i}. **{score:.4f}** - {text}")
        
        return "\n".join(output_lines)
    
    # Create interface
    with gr.Blocks(title="CLIP Similarity Demo", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🤖 CLIP Similarity Demo
        
        Upload an image and enter text to see how well they match according to the CLIP model.
        """)
        
        with gr.Tab("Single Prediction"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="Upload Image")
                    text_input = gr.Textbox(
                        label="Text Description",
                        placeholder="Enter description of what you're looking for...",
                        lines=3
                    )
                    predict_btn = gr.Button("Calculate Similarity", variant="primary")
                
                with gr.Column():
                    similarity_output = gr.Number(label="Similarity Score", precision=4)
                    interpretation_output = gr.Textbox(label="Interpretation")
                    details_output = gr.Textbox(label="Details", lines=6)
            
            predict_btn.click(
                predict_wrapper,
                inputs=[image_input, text_input],
                outputs=[similarity_output, interpretation_output, details_output]
            )
        
        with gr.Tab("Batch Comparison"):
            with gr.Row():
                with gr.Column():
                    batch_image_input = gr.Image(type="pil", label="Upload Image")
                    batch_text_input = gr.Textbox(
                        label="Text Descriptions (one per line)",
                        placeholder="Enter multiple descriptions, one per line:\nBeautiful woman with long hair\nCute girl with glasses\nProfessional portrait",
                        lines=8
                    )
                    batch_predict_btn = gr.Button("Compare All", variant="primary")
                
                with gr.Column():
                    batch_output = gr.Markdown(label="Results")
            
            batch_predict_btn.click(
                batch_predict_wrapper,
                inputs=[batch_image_input, batch_text_input],
                outputs=[batch_output]
            )
        
        with gr.Tab("Examples"):
            gr.Markdown("""
            ## 📝 Example Text Prompts
            
            Try these example prompts with your images:
            
            **People:**
            - "Beautiful Asian woman with long black hair"
            - "Young girl with glasses and a bright smile"  
            - "Professional portrait of a business person"
            - "Casual photo of someone outdoors"
            
            **Style:**
            - "Professional headshot with studio lighting"
            - "Candid photo with natural lighting"
            - "Artistic portrait with dramatic shadows"
            
            **Expressions:**
            - "Person with a warm, genuine smile"
            - "Serious, confident expression"
            - "Playful, fun personality"
            
            ## 🎯 Similarity Score Guide
            
            - **0.4+**: Excellent match
            - **0.3-0.4**: Good match  
            - **0.2-0.3**: Moderate match
            - **0.1-0.2**: Poor match
            - **<0.1**: No match
            """)
    
    return interface


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description='CLIP Similarity Demo')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained CLIP model directory')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Host to run demo on')
    parser.add_argument('--port', type=int, default=7860,
                       help='Port to run demo on')
    parser.add_argument('--share', action='store_true',
                       help='Create public share link')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging('INFO')
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model path not found: {model_path}")
        sys.exit(1)
    
    try:
        # Initialize demo
        logger.info("Initializing CLIP demo...")
        demo = CLIPDemo(str(model_path))
        
        # Create Gradio interface
        logger.info("Creating Gradio interface...")
        interface = create_gradio_interface(demo)
        
        # Launch
        logger.info(f"Launching demo on {args.host}:{args.port}")
        interface.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            debug=False
        )
        
    except KeyboardInterrupt:
        logger.info("Demo stopped by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
