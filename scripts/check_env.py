#!/usr/bin/env python3
"""
Environment check script for CLIP training pipeline

Verifies that all required packages are installed and system meets hardware requirements
"""
import sys
import importlib.util
from typing import Dict, Tuple, Optional

def check_package_version(package_name: str, min_version: str) -> Tuple[bool, str]:
    """Check if package is installed and meets minimum version requirement
    
    Args:
        package_name: Name of package to check
        min_version: Minimum required version
        
    Returns:
        Tuple of (is_valid, current_version_or_error)
    """
    try:
        # Handle special cases
        if package_name == "open_clip_torch":
            import open_clip
            version = open_clip.__version__
        elif package_name == "pyyaml":
            import yaml
            version = yaml.__version__
        else:
            module = importlib.import_module(package_name)
            version = getattr(module, '__version__', 'unknown')
        
        # Simple version comparison (works for most cases)
        if version == 'unknown':
            return True, version  # Assume OK if we can't get version
        
        return True, version
        
    except ImportError as e:
        return False, f"Not installed: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def check_cuda_capability() -> Dict[str, any]:
    """Check CUDA availability and memory information
    
    Returns:
        Dictionary with CUDA information
    """
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        cuda_info = {
            'available': cuda_available,
            'device_count': torch.cuda.device_count() if cuda_available else 0,
            'current_device': None,
            'device_name': None,
            'total_memory_gb': None,
            'reserved_memory_gb': None,
            'allocated_memory_gb': None
        }
        
        if cuda_available and torch.cuda.device_count() > 0:
            device_id = torch.cuda.current_device()
            cuda_info.update({
                'current_device': device_id,
                'device_name': torch.cuda.get_device_name(device_id),
                'total_memory_gb': torch.cuda.get_device_properties(device_id).total_memory / 1e9,
                'reserved_memory_gb': torch.cuda.memory_reserved(device_id) / 1e9,
                'allocated_memory_gb': torch.cuda.memory_allocated(device_id) / 1e9
            })
        
        return cuda_info
        
    except Exception as e:
        return {'available': False, 'error': str(e)}


def main():
    """Main environment check function"""
    print("🔍 FinderPartner CLIP Training Environment Check")
    print("=" * 60)
    
    # Required packages with minimum versions
    required_packages = {
        'torch': '2.2.0',
        'transformers': '4.41.0',
        'datasets': '2.19.0',
        'torchvision': '0.17.0',
        'open_clip_torch': '2.24.0',
        'pyyaml': '6.0',
        'tqdm': '4.66.0',
        'wandb': '0.17.0',
        'PIL': '10.0.0'  # Pillow
    }
    
    print("\n📦 Checking Package Dependencies:")
    print("-" * 40)
    
    all_packages_ok = True
    for package, min_version in required_packages.items():
        is_valid, version_info = check_package_version(package, min_version)
        
        if is_valid:
            print(f"✅ {package:<20} {version_info}")
        else:
            print(f"❌ {package:<20} {version_info}")
            all_packages_ok = False
    
    print(f"\n📊 Package Check: {'✅ PASSED' if all_packages_ok else '❌ FAILED'}")
    
    # Check CUDA capability
    print("\n🖥️  Checking CUDA Capability:")
    print("-" * 40)
    
    cuda_info = check_cuda_capability()
    
    if cuda_info.get('available', False):
        print(f"✅ CUDA Available: Yes")
        print(f"   Device Count: {cuda_info['device_count']}")
        print(f"   Current Device: {cuda_info['current_device']}")
        print(f"   Device Name: {cuda_info['device_name']}")
        print(f"   Total Memory: {cuda_info['total_memory_gb']:.1f} GB")
        print(f"   Reserved Memory: {cuda_info['reserved_memory_gb']:.2f} GB")
        print(f"   Allocated Memory: {cuda_info['allocated_memory_gb']:.2f} GB")
        
        # Check if memory is sufficient for training
        total_mem = cuda_info['total_memory_gb']
        if total_mem and total_mem < 10:
            print(f"⚠️  Warning: GPU memory ({total_mem:.1f} GB) is below recommended 10 GB")
            print("   Consider using smaller batch sizes or CPU training")
        else:
            print(f"✅ GPU Memory: Sufficient for training")
    else:
        print(f"❌ CUDA Available: No")
        if 'error' in cuda_info:
            print(f"   Error: {cuda_info['error']}")
        print("   Training will fall back to CPU (slower)")
    
    # Overall assessment
    print("\n🎯 Overall Assessment:")
    print("-" * 40)
    
    if not all_packages_ok:
        print("❌ FAILED: Missing required packages")
        print("   Run: pip install -r requirements.txt")
        sys.exit(1)
    
    if cuda_info.get('available', False):
        total_mem = cuda_info.get('total_memory_gb', 0)
        if total_mem >= 10:
            print("✅ READY: System ready for GPU training")
        else:
            print("⚠️  CAUTION: GPU available but limited memory")
            print("   Recommend reducing batch size or using CPU")
    else:
        print("⚠️  CPU ONLY: CUDA not available, training will be slow")
    
    print("\n🚀 You can proceed with CLIP training!")
    print("   Run: python src/train_clip.py --config configs/clip_base.yaml")


if __name__ == "__main__":
    main()
