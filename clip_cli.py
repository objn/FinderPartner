#!/usr/bin/env python3
"""
Command Line Interface for CLIP Model

Quick command-line tool to use your trained CLIP model.

Usage Examples:
    python clip_cli.py --image path/to/image.jpg --text "a red car"
    python clip_cli.py --image path/to/image.jpg --compare "a dog" "a cat" "a bird"
    python clip_cli.py --batch-images path/to/folder/ --text "beautiful landscape"
"""
import sys
import argparse
import torch
from pathlib import Path
from PIL import Image
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.clip_model import CLIPWrapper
from transformers import CLIPProcessor


class CLIPCommandLine:
    """Command line interface for CLIP model"""
    
    def __init__(self, model_path: str):
        """Initialize CLIP CLI
        
        Args:
            model_path: Path to trained CLIP model
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading model from: {model_path}")
        self.model = CLIPWrapper.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print(f"✅ Model loaded on {self.device}")
    
    def compute_similarity(self, image_path: str, text: str) -> float:
        """Compute similarity between image and text
        
        Args:
            image_path: Path to image file
            text: Text description
            
        Returns:
            Similarity score
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Process inputs
        image_inputs = self.processor(images=[image], return_tensors="pt", padding=True)
        text_inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        
        pixel_values = image_inputs['pixel_values'].to(self.device)
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
    
    def compare_texts(self, image_path: str, texts: list) -> list:
        """Compare multiple texts against an image
        
        Args:
            image_path: Path to image file
            texts: List of text descriptions
            
        Returns:
            List of (text, similarity) tuples sorted by similarity
        """
        results = []
        for text in texts:
            similarity = self.compute_similarity(image_path, text)
            results.append((text, similarity))
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def batch_search(self, image_folder: str, query_text: str, top_k: int = 5) -> list:
        """Search through a folder of images with a text query
        
        Args:
            image_folder: Path to folder containing images
            query_text: Text query
            top_k: Number of top results to return
            
        Returns:
            List of (image_path, similarity) tuples
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_folder = Path(image_folder)
        
        # Find all images
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(list(image_folder.glob(f"*{ext}")))
            image_paths.extend(list(image_folder.glob(f"*{ext.upper()}")))
        
        if not image_paths:
            print(f"No images found in {image_folder}")
            return []
        
        print(f"Found {len(image_paths)} images, computing similarities...")
        
        # Compute similarities
        results = []
        for i, image_path in enumerate(image_paths):
            try:
                similarity = self.compute_similarity(str(image_path), query_text)
                results.append((str(image_path), similarity))
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(image_paths)} images")
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        # Sort and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="CLIP Model Command Line Interface")
    
    # Model path
    parser.add_argument(
        "--model", 
        type=str, 
        default="outputs/clip_training_20250729_112717/final_model",
        help="Path to trained CLIP model"
    )
    
    # Single image operations
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--text", type=str, help="Text description")
    parser.add_argument("--compare", nargs="+", help="Multiple texts to compare against image")
    
    # Batch operations
    parser.add_argument("--batch-images", type=str, help="Folder containing images to search")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top results for batch search")
    
    # Output options
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Check model path
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        print("\nAvailable models:")
        outputs_dir = Path("outputs")
        if outputs_dir.exists():
            for exp_dir in outputs_dir.iterdir():
                if exp_dir.is_dir():
                    final_model = exp_dir / "final_model"
                    if final_model.exists():
                        print(f"  {final_model}")
        return
    
    # Initialize CLI
    try:
        clip_cli = CLIPCommandLine(args.model)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    results = {}
    
    # Single image + text similarity
    if args.image and args.text:
        if not Path(args.image).exists():
            print(f"❌ Image not found: {args.image}")
            return
        
        similarity = clip_cli.compute_similarity(args.image, args.text)
        
        print(f"\n📊 Similarity Results:")
        print(f"Image: {args.image}")
        print(f"Text: '{args.text}'")
        print(f"Similarity: {similarity:.4f}")
        
        if similarity > 0.5:
            print("🟢 Very good match!")
        elif similarity > 0.3:
            print("🟡 Good match")
        elif similarity > 0.1:
            print("🟠 Weak match")
        else:
            print("🔴 Poor match")
        
        results['single_similarity'] = {
            'image': args.image,
            'text': args.text,
            'similarity': similarity
        }
    
    # Single image + multiple text comparison
    elif args.image and args.compare:
        if not Path(args.image).exists():
            print(f"❌ Image not found: {args.image}")
            return
        
        comparisons = clip_cli.compare_texts(args.image, args.compare)
        
        print(f"\n🏆 Comparison Results for: {args.image}")
        for i, (text, similarity) in enumerate(comparisons, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📍"
            print(f"{emoji} {i}. '{text}' - {similarity:.4f}")
        
        results['comparison'] = {
            'image': args.image,
            'results': [{'text': text, 'similarity': sim} for text, sim in comparisons]
        }
    
    # Batch image search
    elif args.batch_images and args.text:
        if not Path(args.batch_images).exists():
            print(f"❌ Folder not found: {args.batch_images}")
            return
        
        search_results = clip_cli.batch_search(args.batch_images, args.text, args.top_k)
        
        print(f"\n🔍 Search Results for: '{args.text}'")
        print(f"Searched in: {args.batch_images}")
        print(f"Top {len(search_results)} results:")
        
        for i, (image_path, similarity) in enumerate(search_results, 1):
            image_name = Path(image_path).name
            print(f"{i}. {image_name} - {similarity:.4f}")
        
        results['batch_search'] = {
            'query': args.text,
            'folder': args.batch_images,
            'results': [{'image': img, 'similarity': sim} for img, sim in search_results]
        }
    
    else:
        print("❌ Invalid arguments. Use --help for usage information.")
        print("\nExamples:")
        print("  python clip_cli.py --image photo.jpg --text 'a red car'")
        print("  python clip_cli.py --image photo.jpg --compare 'a dog' 'a cat' 'a bird'")
        print("  python clip_cli.py --batch-images ./photos/ --text 'sunset landscape'")
        return
    
    # Save results if requested
    if args.output and results:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
