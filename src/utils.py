"""
Utility functions for image loading and processing
"""
import logging
from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageFile
import os

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

# Supported image extensions
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}


def load_images_from_directory(img_dir: Path) -> List[Tuple[Path, Image.Image]]:
    """Load all valid images from a directory
    
    Args:
        img_dir: Path to directory containing images
        
    Returns:
        List of tuples containing (file_path, PIL_Image)
        
    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If no valid images found
    """
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    
    if not img_dir.is_dir():
        raise ValueError(f"Path is not a directory: {img_dir}")
    
    images = []
    failed_files = []
    
    # Get all files with supported extensions
    image_files = [
        f for f in img_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    
    if not image_files:
        raise ValueError(f"No supported image files found in {img_dir}")
    
    logger.info(f"Found {len(image_files)} potential image files")
    
    for img_path in image_files:
        try:
            # Try to open and verify the image
            with Image.open(img_path) as img:
                # Convert to RGB if necessary (handles RGBA, L, etc.)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Create a copy since we're using 'with' statement
                img_copy = img.copy()
                images.append((img_path, img_copy))
                
        except Exception as e:
            logger.warning(f"Failed to load image {img_path.name}: {e}")
            failed_files.append(img_path.name)
    
    if failed_files:
        logger.info(f"Failed to load {len(failed_files)} files: {', '.join(failed_files[:5])}")
        if len(failed_files) > 5:
            logger.info(f"... and {len(failed_files) - 5} more")
    
    if not images:
        raise ValueError(f"No valid images could be loaded from {img_dir}")
    
    logger.info(f"Successfully loaded {len(images)} images")
    return images


def validate_image_path(img_path: str) -> Path:
    """Validate and convert image path string to Path object
    
    Args:
        img_path: String path to image directory
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If path is invalid
    """
    try:
        path = Path(img_path).resolve()
        return path
    except Exception as e:
        raise ValueError(f"Invalid path '{img_path}': {e}")


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in megabytes
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB
    """
    try:
        size_bytes = file_path.stat().st_size
        return size_bytes / (1024 * 1024)
    except Exception:
        return 0.0


def print_directory_summary(img_dir: Path) -> None:
    """Print summary of directory contents
    
    Args:
        img_dir: Path to directory to summarize
    """
    if not img_dir.exists():
        logger.error(f"Directory not found: {img_dir}")
        return
    
    all_files = list(img_dir.iterdir())
    image_files = [
        f for f in all_files 
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    
    total_size = sum(get_file_size_mb(f) for f in image_files)
    
    print(f"\n📁 Directory: {img_dir}")
    print(f"   Total files: {len(all_files)}")
    print(f"   Image files: {len(image_files)}")
    print(f"   Total size: {total_size:.1f} MB")
    
    if image_files:
        print(f"   Extensions: {', '.join(sorted(set(f.suffix.lower() for f in image_files)))}")