#!/usr/bin/env python3
"""
CLIP Model Inference Script

Use your trained CLIP model for:
1. Image-Text similarity scoring
2. Image search with text queries
3. Text search with image queries
4. Feature extraction
"""
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Union, Tuple, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.clip_model import CLIPWrapper
from transformers import CLIPProcessor


class CLIPInference:
    """CLIP model inference wrapper"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """Initialize CLIP inference
        
        Args:
            model_path: Path to trained model directory
            device: Device to run inference on
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Loading model from: {model_path}")
        print(f"🔧 Using device: {self.device}")
        
        # Load model
        self.model = CLIPWrapper.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load processor
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        print("✅ Model loaded successfully!")
    
    def encode_images(self, images: List[Union[str, Image.Image]]) -> torch.Tensor:
        """Encode images to embeddings
        
        Args:
            images: List of image paths or PIL Images
            
        Returns:
            Image embeddings tensor
        """
        # Load images if paths provided
        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img).convert('RGB'))
            else:
                pil_images.append(img)
        
        # Process images
        inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
        pixel_values = inputs['pixel_values'].to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            image_embeds = self.model.model.get_image_features(pixel_values)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        
        return image_embeds
    
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to embeddings
        
        Args:
            texts: List of text strings
            
        Returns:
            Text embeddings tensor
        """
        # Process texts
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            text_embeds = self.model.model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        return text_embeds
    
    def compute_similarity(self, images: List[Union[str, Image.Image]], 
                          texts: List[str]) -> np.ndarray:
        """Compute similarity between images and texts
        
        Args:
            images: List of images
            texts: List of texts
            
        Returns:
            Similarity matrix [num_images, num_texts]
        """
        image_embeds = self.encode_images(images)
        text_embeds = self.encode_texts(texts)
        
        # Compute similarity
        similarity = torch.matmul(image_embeds, text_embeds.t())
        return similarity.cpu().numpy()
    
    def search_images_with_text(self, images: List[Union[str, Image.Image]], 
                               query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Search images using text query
        
        Args:
            images: List of images to search
            query: Text query
            top_k: Number of top results to return
            
        Returns:
            List of (image_index, similarity_score) tuples
        """
        similarities = self.compute_similarity(images, [query]).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(idx, similarities[idx]) for idx in top_indices]
        
        return results
    
    def search_texts_with_image(self, image: Union[str, Image.Image], 
                               texts: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
        """Search texts using image query
        
        Args:
            image: Query image
            texts: List of texts to search
            top_k: Number of top results to return
            
        Returns:
            List of (text_index, similarity_score) tuples
        """
        similarities = self.compute_similarity([image], texts)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(idx, similarities[idx]) for idx in top_indices]
        
        return results
    
    def find_best_match(self, image: Union[str, Image.Image], 
                       texts: List[str]) -> Tuple[int, str, float]:
        """Find best matching text for an image
        
        Args:
            image: Query image
            texts: List of candidate texts
            
        Returns:
            Tuple of (best_index, best_text, similarity_score)
        """
        results = self.search_texts_with_image(image, texts, top_k=1)
        best_idx, best_score = results[0]
        return best_idx, texts[best_idx], best_score


def demo_basic_usage():
    """Demonstrate basic CLIP inference usage"""
    print("🚀 CLIP Inference Demo")
    print("=" * 50)
    
    # Initialize model (use your latest trained model)
    model_path = "outputs/clip_training_20250729_112717/final_model"
    clip_inference = CLIPInference(model_path)
    
    # Example 1: Text-to-image similarity
    print("\n📝 Example 1: Text-to-Image Similarity")
    texts = [
        "a red car",
        "a blue ocean", 
        "a green forest",
        "a yellow flower"
    ]
    
    # If you have test images, uncomment and modify this:
    # image_paths = ["path/to/image1.jpg", "path/to/image2.jpg"]
    # similarities = clip_inference.compute_similarity(image_paths, texts)
    # print(f"Similarity matrix shape: {similarities.shape}")
    # print(f"Similarities:\n{similarities}")
    
    # Example 2: Encode texts for similarity search
    print("\n🔤 Example 2: Text Encoding")
    text_embeds = clip_inference.encode_texts(texts)
    print(f"Text embeddings shape: {text_embeds.shape}")
    print(f"Text embedding norm: {text_embeds.norm(dim=-1)}")
    
    # Example 3: Text-to-text similarity (using CLIP's text encoder)
    text_similarities = torch.matmul(text_embeds, text_embeds.t())
    print(f"\n🔗 Text-to-text similarities:")
    for i, text1 in enumerate(texts):
        for j, text2 in enumerate(texts):
            if i < j:  # Only print upper triangle
                sim = text_similarities[i, j].item()
                print(f"'{text1}' <-> '{text2}': {sim:.3f}")
    
    print("\n✅ Demo completed!")
    print("\n💡 To use with your own images:")
    print("   1. Put images in a folder")
    print("   2. Modify the image_paths list above")
    print("   3. Run the similarity computation")


if __name__ == "__main__":
    demo_basic_usage()
