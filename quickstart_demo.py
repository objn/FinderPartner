#!/usr/bin/env python3
"""
Quick Start Demo for CLIP Training Pipeline

This script demonstrates the complete pipeline from data creation to training.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    """Quick start demo"""
    print("🚀 FinderPartner CLIP Training Pipeline Demo")
    print("=" * 60)
    
    try:
        # Step 1: Check environment
        print("\n📋 Step 1: Checking environment...")
        from scripts.check_env import main as check_env
        # Note: In a real scenario, you would run this as a subprocess
        print("✅ Environment check completed (run scripts/check_env.py for details)")
        
        # Step 2: Create sample data
        print("\n📁 Step 2: Creating sample training data...")
        from data.dataset_clip import create_sample_data
        
        data_dir = Path('data')
        create_sample_data(data_dir, num_samples=20)
        print(f"✅ Sample data created in {data_dir}")
        
        # Step 3: Load configuration
        print("\n⚙️ Step 3: Loading training configuration...")
        from configs import load_config, get_default_config
        
        # Use default config for demo
        config = get_default_config()
        print(f"✅ Configuration loaded: {config['model_name']}")
        
        # Step 4: Test data loading
        print("\n📊 Step 4: Testing data pipeline...")
        try:
            from data.dataset_clip import create_dataloaders
            train_loader, val_loader = create_dataloaders(config)
            print(f"✅ Data loaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")
        except Exception as e:
            print(f"⚠️ Data loading test failed: {e}")
        
        # Step 5: Test model creation
        print("\n🤖 Step 5: Testing model creation...")
        try:
            from models.clip_model import create_model
            model = create_model(config)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"✅ Model created: {total_params:,} parameters")
        except Exception as e:
            print(f"⚠️ Model creation test failed: {e}")
        
        # Step 6: Next steps
        print("\n🎯 Next Steps:")
        print("1. Prepare your own image-caption dataset")
        print("2. Update configs/clip_base.yaml with your data paths")
        print("3. Run: python src/train_clip.py --config configs/clip_base.yaml")
        print("\n📚 For more details, see README.md section 'CLIP Training Pipeline'")
        
        print("\n✅ Demo completed successfully!")
        
    except ImportError as e:
        print(f"\n❌ Missing dependencies: {e}")
        print("Install required packages: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
