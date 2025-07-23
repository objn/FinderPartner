"""
Unit tests for CLIP training pipeline
"""
import sys
import unittest
import tempfile
import csv
from pathlib import Path
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestCLIPPipeline(unittest.TestCase):
    """Test CLIP training pipeline components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = {
            'model_name': 'openai/clip-vit-base-patch32',
            'image_size': 224,
            'batch_size': 2,
            'max_length': 77,
            'epochs': 1,
            'lr': 1e-4,
            'train_csv': self.temp_dir / 'train.csv',
            'val_csv': self.temp_dir / 'val.csv',
            'train_images_dir': self.temp_dir / 'train_images',
            'val_images_dir': self.temp_dir / 'val_images',
            'output_dir': self.temp_dir / 'outputs',
            'log_wandb': False,
            'device': 'cpu',
            'mixed_precision': False
        }
        
        # Create directories
        self.config['train_images_dir'].mkdir(parents=True)
        self.config['val_images_dir'].mkdir(parents=True)
        
        # Create sample CSV files
        self._create_sample_data()
    
    def tearDown(self):
        """Clean up test fixtures"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def _create_sample_data(self):
        """Create sample data for testing"""
        from PIL import Image
        import random
        
        # Create sample images
        for split, img_dir in [('train', self.config['train_images_dir']), 
                              ('val', self.config['val_images_dir'])]:
            csv_file = self.config[f'{split}_csv']
            
            # Create CSV
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'caption'])
                
                # Create sample entries
                for i in range(4):  # Small dataset for testing
                    filename = f'{split}_{i:03d}.jpg'
                    caption = f'Sample caption {i} for {split} split'
                    writer.writerow([filename, caption])
                    
                    # Create dummy image
                    img = Image.new('RGB', (224, 224), 
                                  color=(random.randint(0, 255), 
                                        random.randint(0, 255), 
                                        random.randint(0, 255)))
                    img.save(img_dir / filename)
    
    def test_config_loading(self):
        """Test configuration loading and validation"""
        try:
            from configs import get_default_config, validate_config, save_config, load_config
            
            # Test default config
            default_config = get_default_config()
            self.assertIsInstance(default_config, dict)
            self.assertIn('model_name', default_config)
            
            # Test config saving and loading
            config_path = self.temp_dir / 'test_config.yaml'
            save_config(self.config, str(config_path))
            self.assertTrue(config_path.exists())
            
            loaded_config = load_config(str(config_path))
            self.assertEqual(loaded_config['model_name'], self.config['model_name'])
            
        except ImportError as e:
            self.skipTest(f"Required packages not available: {e}")
    
    def test_training_step_smoke(self):
        """Smoke test for training step - assert loss is reasonable"""
        try:
            import torch
            from models.clip_model import CLIPWrapper
            from data.dataset_clip import create_dataloaders
            
            # Setup
            device = torch.device('cpu')
            model = CLIPWrapper(self.config).to(device)
            train_loader, _ = create_dataloaders(self.config)
            
            # Get sample batch
            batch = next(iter(train_loader))
            
            # Setup optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            
            # Training step
            model.train()
            optimizer.zero_grad()
            
            outputs = model(
                pixel_values=batch['pixel_values'].to(device),
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                return_loss=True
            )
            
            loss = outputs['loss']
            self.assertIsNotNone(loss)
            self.assertGreater(loss.item(), 0)
            self.assertLess(loss.item(), 20)  # Loss should be reasonable after 1 batch
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
        except ImportError as e:
            self.skipTest(f"Required packages not available: {e}")


if __name__ == '__main__':
    # Run smoke test
    suite = unittest.TestSuite()
    suite.addTest(TestCLIPPipeline('test_config_loading'))
    suite.addTest(TestCLIPPipeline('test_training_step_smoke'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
