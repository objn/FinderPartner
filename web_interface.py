#!/usr/bin/env python3
"""
CLIP Web Interface with Gradio

Interactive web interface for your trained CLIP model.
Upload images and search with text, or upload text and search images.
"""
import sys
import torch
from pathlib import Path
from PIL import Image
import gradio as gr
import numpy as np
from typing import List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from models.clip_model import CLIPWrapper
    from transformers import CLIPProcessor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure to install requirements: pip install -r requirements.txt")
    sys.exit(1)


class CLIPWebInterface:
    """Web interface for CLIP model"""
    
    def __init__(self, model_path: str):
        """Initialize CLIP web interface
        
        Args:
            model_path: Path to trained CLIP model
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model from: {model_path}")
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = CLIPWrapper.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load processor
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        print("✅ Model loaded successfully!")
    
    def compute_similarity(self, image: Image.Image, text: str) -> float:
        """Compute similarity between image and text
        
        Args:
            image: PIL Image
            text: Text string
            
        Returns:
            Similarity score
        """
        if image is None or not text.strip():
            return 0.0
        
        try:
            # Process image
            image_inputs = self.processor(images=[image], return_tensors="pt", padding=True)
            pixel_values = image_inputs['pixel_values'].to(self.device)
            
            # Process text
            text_inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            input_ids = text_inputs['input_ids'].to(self.device)
            attention_mask = text_inputs['attention_mask'].to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                image_embeds = self.model.model.get_image_features(pixel_values)
                text_embeds = self.model.model.get_text_features(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Normalize
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                
                # Compute similarity
                similarity = torch.matmul(image_embeds, text_embeds.t()).item()
            
            return similarity
            
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return 0.0
    
    def compare_texts(self, image: Image.Image, text1: str, text2: str, text3: str = "") -> str:
        """Compare multiple texts against an image
        
        Args:
            image: PIL Image
            text1, text2, text3: Text options to compare
            
        Returns:
            Formatted comparison results
        """
        if image is None:
            return "Please upload an image first."
        
        texts = [t.strip() for t in [text1, text2, text3] if t.strip()]
        
        if len(texts) < 2:
            return "Please provide at least 2 text options to compare."
        
        results = []
        for i, text in enumerate(texts, 1):
            similarity = self.compute_similarity(image, text)
            results.append((f"Option {i}: {text}", similarity))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        output = "🏆 Similarity Rankings:\n\n"
        for i, (text, score) in enumerate(results, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            output += f"{emoji} {text}\n"
            output += f"   Similarity: {score:.4f}\n\n"
        
        return output
    
    def analyze_image(self, image: Image.Image) -> str:
        """Analyze an image with predefined categories
        
        Args:
            image: PIL Image
            
        Returns:
            Analysis results
        """
        if image is None:
            return "Please upload an image first."
        
        # Predefined categories
        categories = [
            "a photo of a person",
            "a photo of an animal",
            "a photo of a car or vehicle",
            "a photo of food",
            "a landscape or nature scene",
            "a building or architecture",
            "an indoor scene",
            "an outdoor scene",
            "a close-up or macro photo",
            "a wide shot or panorama"
        ]
        
        results = []
        for category in categories:
            similarity = self.compute_similarity(image, category)
            results.append((category, similarity))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        output = "🔍 Image Analysis Results:\n\n"
        for i, (category, score) in enumerate(results[:5], 1):
            confidence = score * 100
            output += f"{i}. {category}\n"
            output += f"   Confidence: {confidence:.1f}%\n\n"
        
        return output


def create_interface():
    """Create Gradio interface"""
    
    # Initialize model
    model_path = "outputs/clip_training_20250729_112717/final_model"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found at: {model_path}")
        print("Available models:")
        outputs_dir = Path("outputs")
        if outputs_dir.exists():
            for exp_dir in outputs_dir.iterdir():
                if exp_dir.is_dir():
                    final_model = exp_dir / "final_model"
                    if final_model.exists():
                        print(f"  - {final_model}")
        return None
    
    try:
        clip_interface = CLIPWebInterface(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Create Gradio interface
    with gr.Blocks(title="CLIP Model Interface", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🔍 CLIP Model Interface")
        gr.Markdown("Upload an image and interact with your trained CLIP model!")
        
        with gr.Tab("Image-Text Similarity"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="Upload Image")
                    text_input = gr.Textbox(
                        label="Enter text description",
                        placeholder="e.g., 'a red car driving on a road'"
                    )
                    similarity_btn = gr.Button("Compute Similarity", variant="primary")
                
                with gr.Column():
                    similarity_output = gr.Textbox(
                        label="Similarity Score",
                        placeholder="Similarity score will appear here..."
                    )
            
            similarity_btn.click(
                fn=clip_interface.compute_similarity,
                inputs=[image_input, text_input],
                outputs=[similarity_output]
            )
        
        with gr.Tab("Compare Descriptions"):
            with gr.Row():
                with gr.Column():
                    image_input2 = gr.Image(type="pil", label="Upload Image")
                    text1_input = gr.Textbox(label="Description 1", placeholder="e.g., 'a dog playing'")
                    text2_input = gr.Textbox(label="Description 2", placeholder="e.g., 'a cat sleeping'")
                    text3_input = gr.Textbox(label="Description 3 (optional)", placeholder="e.g., 'a bird flying'")
                    compare_btn = gr.Button("Compare Descriptions", variant="primary")
                
                with gr.Column():
                    compare_output = gr.Textbox(
                        label="Comparison Results",
                        placeholder="Comparison results will appear here...",
                        lines=10
                    )
            
            compare_btn.click(
                fn=clip_interface.compare_texts,
                inputs=[image_input2, text1_input, text2_input, text3_input],
                outputs=[compare_output]
            )
        
        with gr.Tab("Image Analysis"):
            with gr.Row():
                with gr.Column():
                    image_input3 = gr.Image(type="pil", label="Upload Image")
                    analyze_btn = gr.Button("Analyze Image", variant="primary")
                
                with gr.Column():
                    analysis_output = gr.Textbox(
                        label="Analysis Results",
                        placeholder="Analysis results will appear here...",
                        lines=12
                    )
            
            analyze_btn.click(
                fn=clip_interface.analyze_image,
                inputs=[image_input3],
                outputs=[analysis_output]
            )
        
        gr.Markdown("""
        ### 💡 How to Use:
        1. **Image-Text Similarity**: Upload an image and enter a description to get a similarity score
        2. **Compare Descriptions**: Upload an image and compare multiple descriptions to see which matches best
        3. **Image Analysis**: Upload an image to get automatic categorization and analysis
        
        ### 📊 Understanding Scores:
        - Similarity scores range from -1 to 1
        - Higher scores (closer to 1) = better match
        - Scores above 0.3 typically indicate good matches
        - Scores above 0.5 indicate very good matches
        """)
    
    return interface


def main():
    """Main function to launch the interface"""
    print("🚀 Starting CLIP Web Interface...")
    
    interface = create_interface()
    if interface is None:
        print("❌ Failed to create interface")
        return
    
    print("✅ Interface created successfully!")
    print("🌐 Launching web interface...")
    
    # Launch interface
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,  # Set to True if you want to share publicly
        debug=False
    )


if __name__ == "__main__":
    main()
