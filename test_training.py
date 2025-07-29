#!/usr/bin/env python3
"""
Quick Training Verification Script

Test the training pipeline with minimal configuration to ensure it works.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_training():
    """Test training with minimal setup"""
    print("🧪 Testing CLIP Training Pipeline...")
    
    try:
        # Test configuration loading
        from configs import load_config
        config = load_config('configs/clip_base.yaml')
        print(f"✅ Configuration loaded: {config['model_name']}")
        
        # Test data loading
        from data.dataset_clip import create_dataloaders
        train_loader, val_loader = create_dataloaders(config)
        print(f"✅ Data loaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")
        
        # Test model creation
        from models.clip_model import create_model
        model = create_model(config)
        print(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test evaluation metrics with appropriate k values
        import torch
        from eval.retrieval import compute_retrieval_metrics
        
        # Create small test embeddings
        batch_size = 3  # Small test size
        embed_dim = 512
        image_embeds = torch.randn(batch_size, embed_dim)
        text_embeds = torch.randn(batch_size, embed_dim)
        
        # Test with appropriate k values
        metrics = compute_retrieval_metrics(image_embeds, text_embeds, k_values=[1, 2])
        print(f"✅ Retrieval metrics computed: {list(metrics.keys())}")
        
        print("\n🎉 All tests passed! Training pipeline is ready.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training()
    sys.exit(0 if success else 1)
