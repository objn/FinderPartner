"""
CLIP Dataset implementation for paired image-caption training data
"""
import csv
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from transformers import CLIPTokenizer
import pandas as pd

logger = logging.getLogger(__name__)


class CLIPPairedDataset(Dataset):
    """Dataset for CLIP training with paired image-caption data"""
    
    def __init__(
        self,
        csv_file: Union[str, Path],
        images_dir: Union[str, Path],
        tokenizer: CLIPTokenizer,
        image_size: int = 224,
        max_length: int = 77,
        transforms: Optional[transforms.Compose] = None
    ):
        """Initialize CLIP paired dataset
        
        Args:
            csv_file: Path to CSV file with columns: filename, caption
            images_dir: Directory containing images
            tokenizer: CLIP tokenizer for text processing
            image_size: Target image size for training
            max_length: Maximum text sequence length
            transforms: Optional custom transforms, if None will use default
        """
        self.csv_file = Path(csv_file)
        self.images_dir = Path(images_dir)
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_length = max_length
        
        # Load data
        self.data = self._load_data()
        logger.info(f"Loaded {len(self.data)} image-caption pairs from {csv_file}")
        
        # Setup image transforms
        if transforms is None:
            self.transforms = self._get_default_transforms()
        else:
            self.transforms = transforms
    
    def _load_data(self) -> List[Dict[str, str]]:
        """Load image-caption pairs from CSV file
        
        Returns:
            List of dictionaries with 'filename' and 'caption' keys
        """
        if not self.csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")
        
        data = []
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'filename' not in row or 'caption' not in row:
                        raise ValueError("CSV must have 'filename' and 'caption' columns")
                    
                    # Check if image file exists
                    img_path = self.images_dir / row['filename']
                    if img_path.exists():
                        data.append({
                            'filename': row['filename'],
                            'caption': row['caption'].strip()
                        })
                    else:
                        logger.warning(f"Image not found: {img_path}")
            
        except Exception as e:
            raise RuntimeError(f"Error loading CSV data: {e}")
        
        if not data:
            raise ValueError("No valid image-caption pairs found")
        
        return data
    
    def _get_default_transforms(self) -> transforms.Compose:
        """Get default image transforms for CLIP training
        
        Returns:
            Composed transforms
        """
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from dataset
        
        Args:
            idx: Index of item to retrieve
            
        Returns:
            Dictionary with 'pixel_values', 'input_ids', and 'attention_mask'
        """
        item = self.data[idx]
        
        # Load and process image
        img_path = self.images_dir / item['filename']
        try:
            image = Image.open(img_path).convert('RGB')
            pixel_values = self.transforms(image)
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return dummy data in case of error
            pixel_values = torch.zeros(3, self.image_size, self.image_size)
        
        # Process text
        text = item['caption']
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'pixel_values': pixel_values,
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }


def auto_batch_size(target_mem_gb: float, base_batch_size: int = 32, dataset_size: int = None) -> int:
    """Auto-tune batch size based on available GPU memory and dataset size
    
    Args:
        target_mem_gb: Target memory usage in GB
        base_batch_size: Base batch size to start with
        dataset_size: Size of dataset (to avoid batch size larger than dataset)
        
    Returns:
        Adjusted batch size
    """
    try:
        import torch
        
        # Start with memory-based adjustment
        if not torch.cuda.is_available():
            # For CPU, use smaller batch size
            adjusted_batch_size = min(base_batch_size, 16)
        else:
            # Get available GPU memory
            device_props = torch.cuda.get_device_properties(0)
            total_mem_gb = device_props.total_memory / 1e9
            
            # Reserved memory for other processes
            reserved_mem_gb = torch.cuda.memory_reserved(0) / 1e9
            available_mem_gb = total_mem_gb - reserved_mem_gb - 2.0  # Keep 2GB buffer
            
            if available_mem_gb < target_mem_gb:
                # Reduce batch size proportionally
                scale_factor = available_mem_gb / target_mem_gb
                adjusted_batch_size = max(1, int(base_batch_size * scale_factor))
            else:
                adjusted_batch_size = base_batch_size
        
        # Adjust based on dataset size
        if dataset_size is not None and dataset_size > 0:
            # Batch size should not be larger than dataset size
            adjusted_batch_size = min(adjusted_batch_size, dataset_size)
            
            # For very small datasets, use even smaller batch sizes
            if dataset_size < 10:
                adjusted_batch_size = min(adjusted_batch_size, 2)
            elif dataset_size < 50:
                adjusted_batch_size = min(adjusted_batch_size, 8)
        
        # Ensure minimum batch size of 1
        adjusted_batch_size = max(1, adjusted_batch_size)
        
        if adjusted_batch_size != base_batch_size:
            logger.info(f"Adjusted batch size from {base_batch_size} to {adjusted_batch_size}")
        
        return adjusted_batch_size
        
    except Exception as e:
        logger.warning(f"Error in auto batch size adjustment: {e}")
        return max(1, min(base_batch_size, 8))  # Safe fallback


def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders
    
    Args:
        config: Configuration dictionary with data paths and parameters
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Initialize tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(config['model_name'])
    
    # Create datasets first to get their sizes
    train_dataset = CLIPPairedDataset(
        csv_file=config['train_csv'],
        images_dir=config['train_images_dir'],
        tokenizer=tokenizer,
        image_size=config['image_size'],
        max_length=config.get('max_length', 77)
    )
    
    val_dataset = CLIPPairedDataset(
        csv_file=config['val_csv'],
        images_dir=config['val_images_dir'],
        tokenizer=tokenizer,
        image_size=config['image_size'],
        max_length=config.get('max_length', 77)
    )
    
    # Auto-adjust batch size based on dataset size
    min_dataset_size = min(len(train_dataset), len(val_dataset))
    batch_size = auto_batch_size(
        target_mem_gb=config.get('target_memory_gb', 8.0),
        base_batch_size=config['batch_size'],
        dataset_size=min_dataset_size
    )
    config['batch_size'] = batch_size  # Update config with adjusted batch size
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=torch.cuda.is_available(),
        drop_last=False  # Don't drop last batch for small datasets
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=torch.cuda.is_available(),
        drop_last=False  # Don't drop last batch for small datasets
    )
    
    logger.info(f"Created dataloaders: train={len(train_dataset)} samples, "
               f"val={len(val_dataset)} samples, batch_size={batch_size}")
    
    return train_loader, val_loader


def create_sample_data(data_dir: Union[str, Path], num_samples: int = 100) -> None:
    """Create sample CSV files for testing (when real data is not available)
    
    Args:
        data_dir: Directory to create sample data
        num_samples: Number of sample entries to create
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample captions
    sample_captions = [
        "A beautiful woman with long black hair",
        "Portrait of a young Asian girl with glasses",
        "Cute girl with short hair and a smile",
        "Woman in casual clothes outdoors",
        "Professional photo of a lady in business attire",
        "Girl with curly hair and bright eyes",
        "Elegant woman wearing a red dress",
        "Young person with a friendly expression",
        "Portrait shot with natural lighting",
        "Casual photo of someone smiling"
    ]
    
    # Create sample CSV files
    for split in ['train', 'val', 'test']:
        csv_path = data_dir / f"captions_{split}.csv"
        split_samples = num_samples if split == 'train' else num_samples // 5
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'caption'])
            
            for i in range(split_samples):
                filename = f"{split}_{i:04d}.jpg"
                caption = sample_captions[i % len(sample_captions)]
                writer.writerow([filename, caption])
        
        logger.info(f"Created sample CSV: {csv_path} with {split_samples} entries")


if __name__ == "__main__":
    # Test the dataset implementation
    import tempfile
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        create_sample_data(temp_dir, num_samples=20)
        print(f"Sample data created in: {temp_dir}")
