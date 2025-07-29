#!/usr/bin/env python3
"""
Image Search with CLIP

Search through a collection of images using natural language queries.
"""
import sys
import os
import torch
from pathlib import Path
from PIL import Image
import json
from typing import List, Dict, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.clip_model import CLIPWrapper
from transformers import CLIPProcessor


class ImageSearchEngine:
    """CLIP-based image search engine"""
    
    def __init__(self, model_path: str, image_folder: str = None):
        """Initialize image search engine
        
        Args:
            model_path: Path to trained CLIP model
            image_folder: Folder containing images to search
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"Loading CLIP model from: {model_path}")
        self.model = CLIPWrapper.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load processor
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Image database
        self.image_paths = []
        self.image_embeddings = None
        
        if image_folder:
            self.index_images(image_folder)
    
    def index_images(self, image_folder: str):
        """Index all images in a folder
        
        Args:
            image_folder: Path to folder containing images
        """
        print(f"Indexing images from: {image_folder}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_folder = Path(image_folder)
        
        self.image_paths = []
        for ext in image_extensions:
            self.image_paths.extend(list(image_folder.glob(f"**/*{ext}")))
            self.image_paths.extend(list(image_folder.glob(f"**/*{ext.upper()}")))
        
        if not self.image_paths:
            print(f"❌ No images found in {image_folder}")
            return
        
        print(f"Found {len(self.image_paths)} images")
        
        # Encode all images
        self.image_embeddings = self._encode_images_batch(self.image_paths)
        
        print(f"✅ Indexed {len(self.image_paths)} images")
    
    def _encode_images_batch(self, image_paths: List[Path], batch_size: int = 8) -> torch.Tensor:
        """Encode images in batches
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for encoding
            
        Returns:
            Image embeddings tensor
        """
        all_embeddings = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            # Load images
            images = []
            valid_paths = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    images.append(img)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"⚠️ Error loading {path}: {e}")
            
            if not images:
                continue
            
            # Process batch
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            pixel_values = inputs['pixel_values'].to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                embeddings = self.model.model.get_image_features(pixel_values)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                all_embeddings.append(embeddings.cpu())
            
            print(f"Processed {len(valid_paths)} images ({i + len(valid_paths)}/{len(image_paths)})")
        
        if all_embeddings:
            return torch.cat(all_embeddings, dim=0)
        else:
            return torch.empty(0, 512)  # Empty tensor with correct embedding dimension
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search images with text query
        
        Args:
            query: Text query
            top_k: Number of top results to return
            
        Returns:
            List of (image_path, similarity_score) tuples
        """
        if self.image_embeddings is None or len(self.image_embeddings) == 0:
            print("❌ No images indexed. Please run index_images() first.")
            return []
        
        # Encode query
        inputs = self.processor(text=[query], return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            query_embedding = self.model.model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
        
        # Compute similarities
        similarities = torch.matmul(self.image_embeddings, query_embedding.t()).squeeze(-1)
        
        # Get top-k results
        top_indices = torch.argsort(similarities, descending=True)[:top_k]
        
        results = []
        for idx in top_indices:
            image_path = str(self.image_paths[idx])
            similarity = similarities[idx].item()
            results.append((image_path, similarity))
        
        return results
    
    def save_index(self, index_path: str):
        """Save image index to file
        
        Args:
            index_path: Path to save index
        """
        if self.image_embeddings is None:
            print("❌ No index to save")
            return
        
        index_data = {
            'image_paths': [str(path) for path in self.image_paths],
            'embeddings': self.image_embeddings.numpy().tolist()
        }
        
        with open(index_path, 'w') as f:
            json.dump(index_data, f)
        
        print(f"✅ Index saved to {index_path}")
    
    def load_index(self, index_path: str):
        """Load image index from file
        
        Args:
            index_path: Path to load index from
        """
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        self.image_paths = [Path(path) for path in index_data['image_paths']]
        self.image_embeddings = torch.tensor(index_data['embeddings'])
        
        print(f"✅ Index loaded from {index_path}")
        print(f"   Images: {len(self.image_paths)}")


def demo_image_search():
    """Demonstrate image search functionality"""
    print("🔍 CLIP Image Search Demo")
    print("=" * 50)
    
    # Initialize search engine
    model_path = "outputs/clip_training_20250729_112717/final_model"
    search_engine = ImageSearchEngine(model_path)
    
    # Example: Index images from your data folder
    image_folder = "data/images"  # Change this to your image folder
    
    if Path(image_folder).exists():
        search_engine.index_images(image_folder)
        
        # Example queries
        queries = [
            "a red car",
            "beautiful landscape",
            "person smiling",
            "food and cooking",
            "animals in nature"
        ]
        
        print("\n🔍 Search Results:")
        for query in queries:
            print(f"\nQuery: '{query}'")
            results = search_engine.search(query, top_k=3)
            
            for i, (image_path, score) in enumerate(results, 1):
                print(f"  {i}. {Path(image_path).name} (score: {score:.3f})")
        
        # Save index for future use
        search_engine.save_index("image_index.json")
        
    else:
        print(f"❌ Image folder '{image_folder}' not found")
        print("💡 To use image search:")
        print("   1. Create a folder with your images")
        print("   2. Update the 'image_folder' variable above")
        print("   3. Run this script again")


if __name__ == "__main__":
    demo_image_search()
