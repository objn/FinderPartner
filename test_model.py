#!/usr/bin/env python3
"""
Quick Test - Verify Your CLIP Model Works

This script quickly tests that your trained model loads and works correctly.
"""
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_model():
    """Test basic model functionality"""
    print("🧪 Testing Your Trained CLIP Model")
    print("=" * 50)
    
    try:
        # Test model loading
        from models.clip_model import CLIPWrapper
        from transformers import CLIPProcessor
        
        model_path = "outputs/clip_training_20250729_112717/final_model"
        
        if not Path(model_path).exists():
            print(f"❌ Model not found: {model_path}")
            print("\nAvailable models:")
            outputs_dir = Path("outputs")
            if outputs_dir.exists():
                for exp_dir in outputs_dir.iterdir():
                    if exp_dir.is_dir():
                        final_model = exp_dir / "final_model"
                        if final_model.exists():
                            print(f"  ✅ {final_model}")
            return False
        
        print(f"📂 Loading model from: {model_path}")
        
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CLIPWrapper.from_pretrained(model_path)
        model.to(device)
        model.eval()
        
        # Load processor
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        print(f"✅ Model loaded successfully on {device}")
        print(f"🔧 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test text encoding
        test_texts = ["a red car", "a blue ocean", "a green forest"]
        text_inputs = processor(text=test_texts, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            text_embeds = model.model.get_text_features(
                input_ids=text_inputs['input_ids'].to(device),
                attention_mask=text_inputs['attention_mask'].to(device)
            )
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        print(f"✅ Text encoding works: {text_embeds.shape}")
        
        # Test text-to-text similarity
        similarities = torch.matmul(text_embeds, text_embeds.t())
        
        print(f"\n📊 Text-to-Text Similarities:")
        for i, text1 in enumerate(test_texts):
            for j, text2 in enumerate(test_texts):
                if i < j:
                    sim = similarities[i, j].item()
                    print(f"  '{text1}' ↔ '{text2}': {sim:.3f}")
        
        # Test if we can create dummy image embeddings
        dummy_image = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            image_embeds = model.model.get_image_features(dummy_image)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        
        print(f"✅ Image encoding works: {image_embeds.shape}")
        
        # Test cross-modal similarity
        cross_sim = torch.matmul(image_embeds, text_embeds.t())
        print(f"✅ Cross-modal similarity works: {cross_sim.shape}")
        
        print(f"\n🎉 All tests passed! Your model is ready to use.")
        print(f"\n📖 Next steps:")
        print(f"  1. Try the web interface: python web_interface.py")
        print(f"  2. Use command line tool: python clip_cli.py --help")
        print(f"  3. Check the usage guide: HOW_TO_USE_MODEL.md")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(f"💡 Solution: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_model()
    if success:
        print(f"\n🚀 Model ready for use!")
    else:
        print(f"\n🔧 Please fix the issues above before using the model.")
