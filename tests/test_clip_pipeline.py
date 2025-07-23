"""
Unit tests for CLIP training pipeline
"""
import sys
import tempfile
import unittest
from pathlib import Path
import csv
import os

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestCLIPPipeline(unittest.TestCase):
    """Test CLIP training pipeline components"""
    
    def setUp(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / 'data'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample CSV data
        self.create_sample_csv()
        
        # Create minimal config
        self.config = {
            'model_name': 'openai/clip-vit-base-patch32',
            'image_size': 224,
            'batch_size': 2,
            'max_length': 77,
            'train_csv': self.data_dir / 'train.csv',
            'val_csv': self.data_dir / 'val.csv',
            'train_images_dir': self.data_dir / 'images' / 'train',
            'val_images_dir': self.data_dir / 'images' / 'val',
            'epochs': 1,
            'lr': 1e-4,
            'num_workers': 0,  # Use 0 for testing to avoid multiprocessing issues
            'log_wandb': False
        }
    
    def create_sample_csv(self):
        """Create sample CSV files for testing"""
        import PIL.Image as Image
        import numpy as np
        
        # Create image directories
        for split in ['train', 'val']:
            img_dir = self.data_dir / 'images' / split
            img_dir.mkdir(parents=True, exist_ok=True)
            
            # Create sample CSV
            csv_path = self.data_dir / f'{split}.csv'
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'caption'])
                
                # Create a few sample entries
                num_samples = 4 if split == 'train' else 2
                for i in range(num_samples):
                    filename = f'{split}_{i}.jpg'
                    caption = f'Sample image {i} for testing'
                    
                    # Create dummy image
                    img_path = img_dir / filename
                    dummy_img = Image.fromarray(
                        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                    )
                    dummy_img.save(img_path)
                    
                    writer.writerow([filename, caption])
    
    def test_config_loading(self):
        """Test configuration loading and validation"""
        try:
            from configs import get_default_config, validate_config
            
            # Test default config
            default_config = get_default_config()
            self.assertIsInstance(default_config, dict)
            self.assertIn('model_name', default_config)
            
            # Test config validation with valid config
            validate_config(self.config)  # Should not raise exception
            
        except ImportError as e:
            self.skipTest(f"Required dependencies not available: {e}")
    
    def test_dataset_creation(self):
        """Test CLIP dataset creation (CPU only)"""
        try:
            from data import CLIPPairedDataset
            from transformers import CLIPTokenizer
            
            # Create tokenizer
            tokenizer = CLIPTokenizer.from_pretrained(self.config['model_name'])
            
            # Create dataset
            dataset = CLIPPairedDataset(
                csv_file=self.config['train_csv'],
                images_dir=self.config['train_images_dir'],
                tokenizer=tokenizer,
                image_size=self.config['image_size'],
                max_length=self.config['max_length']
            )
            
            # Test dataset properties
            self.assertGreater(len(dataset), 0)
            
            # Test item retrieval
            item = dataset[0]
            self.assertIn('pixel_values', item)
            self.assertIn('input_ids', item)
            self.assertIn('attention_mask', item)
            
            # Check tensor shapes
            self.assertEqual(item['pixel_values'].shape, 
                           (3, self.config['image_size'], self.config['image_size']))
            self.assertEqual(item['input_ids'].shape, (self.config['max_length'],))
            self.assertEqual(item['attention_mask'].shape, (self.config['max_length'],))
            
        except ImportError as e:
            self.skipTest(f"Required dependencies not available: {e}")
    
    def test_model_creation(self):
        """Test CLIP model creation (CPU only)"""
        try:
            from models import create_model
            
            # Create model
            model = create_model(self.config)
            
            # Test model properties
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(model, 'forward'))
            self.assertTrue(hasattr(model, 'save_pretrained'))
            
        except ImportError as e:
            self.skipTest(f"Required dependencies not available: {e}")
    
    def test_training_step_cpu(self):
        """Test a single training step on CPU"""
        try:
            import torch
            from models import create_model
            from data import create_dataloaders
            
            # Force CPU device
            device = torch.device('cpu')
            
            # Create model and data
            model = create_model(self.config)
            model = model.to(device)
            
            train_loader, _ = create_dataloaders(self.config)
            
            # Get a batch
            batch = next(iter(train_loader))
            
            # Test forward pass
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_loss=True
            )
            
            # Check outputs
            self.assertIn('loss', outputs)
            self.assertIsInstance(outputs['loss'], torch.Tensor)
            self.assertTrue(outputs['loss'].item() > 0)  # Loss should be positive
            
            # Test backward pass
            loss = outputs['loss']
            loss.backward()
            
            # Check gradients exist
            has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
            self.assertTrue(has_gradients)
            
        except ImportError as e:
            self.skipTest(f"Required dependencies not available: {e}")
    
    def test_evaluation_metrics(self):
        """Test evaluation metrics computation"""
        try:
            import torch
            from eval import compute_retrieval_metrics
            
            # Create dummy embeddings
            batch_size = 4
            embed_dim = 512
            
            image_embeds = torch.randn(batch_size, embed_dim)
            text_embeds = torch.randn(batch_size, embed_dim)
            
            # Normalize embeddings
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            
            # Compute metrics
            metrics = compute_retrieval_metrics(image_embeds, text_embeds, k_values=[1, 2])
            
            # Check metric names and values
            expected_keys = [
                'image_to_text_recall@1', 'image_to_text_recall@2',
                'text_to_image_recall@1', 'text_to_image_recall@2',
                'mean_recall@1', 'mean_recall@2'
            ]
            
            for key in expected_keys:
                self.assertIn(key, metrics)
                self.assertIsInstance(metrics[key], float)
                self.assertGreaterEqual(metrics[key], 0.0)
                self.assertLessEqual(metrics[key], 1.0)
            
        except ImportError as e:
            self.skipTest(f"Required dependencies not available: {e}")
    
    def test_smoke_training(self):
        """Smoke test: run minimal training loop"""
        try:
            import torch
            from models import create_model
            from data import create_dataloaders
            
            # Use CPU for smoke test
            device = torch.device('cpu')
            
            # Create model and data
            model = create_model(self.config)
            model = model.to(device)
            
            train_loader, _ = create_dataloaders(self.config)
            
            # Setup simple optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            
            # Run one training step
            model.train()
            batch = next(iter(train_loader))
            
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_loss=True
            )
            
            loss = outputs['loss']
            initial_loss = loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Second forward pass to check loss changed
            outputs2 = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_loss=True
            )
            
            final_loss = outputs2['loss'].item()
            
            # Check that loss is reasonable
            self.assertLess(initial_loss, 100.0)  # Loss shouldn't be too high
            self.assertGreater(initial_loss, 0.0)  # Loss should be positive
            
            # Loss might not decrease in one step, but should be finite
            self.assertTrue(torch.isfinite(torch.tensor(final_loss)))
            
        except ImportError as e:
            self.skipTest(f"Required dependencies not available: {e}")
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass  # Ignore cleanup errors


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
