#!/usr/bin/env python3
"""
CLIP Training Script for FinderPartner

Train a CLIP model on paired image-caption data with automatic device detection,
memory management, and comprehensive logging.
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from configs import load_config, validate_config, setup_experiment_dir, get_default_config
from data import create_dataloaders, create_sample_data
from models import create_model
from eval import evaluate_model, MetricsTracker
from utils import setup_logging

logger = logging.getLogger(__name__)


def setup_device(config: Dict) -> torch.device:
    """Setup training device with fallback strategy
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Selected device
    """
    device_config = config.get('device', 'auto')
    
    if device_config == 'cpu':
        device = torch.device('cpu')
        logger.info("Using CPU (forced by config)")
    elif device_config == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA (forced by config): {torch.cuda.get_device_name()}")
        else:
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = torch.device('cpu')
    else:  # auto
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name()
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"Auto-selected CUDA: {gpu_name} ({total_mem:.1f}GB)")
            
            # Enable optimizations for GPU
            torch.backends.cudnn.benchmark = True
        else:
            device = torch.device('cpu')
            logger.info("Auto-selected CPU (CUDA not available)")
    
    return device


def setup_optimizer_and_scheduler(model: nn.Module, config: Dict, num_training_steps: int):
    """Setup optimizer and learning rate scheduler
    
    Args:
        model: Model to optimize
        config: Configuration dictionary
        num_training_steps: Total number of training steps
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Filter parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(
        params,
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 1e-2),
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Setup learning rate scheduler with warmup
    warmup_steps = config.get('warmup_steps', 500)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            # Cosine annealing after warmup
            progress = (step - warmup_steps) / (num_training_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    logger.info(f"Setup optimizer: AdamW (lr={config['lr']}, warmup_steps={warmup_steps})")
    return optimizer, scheduler


def setup_wandb_logging(config: Dict, experiment_dir: Path) -> Optional[object]:
    """Setup Weights & Biases logging
    
    Args:
        config: Configuration dictionary
        experiment_dir: Experiment directory path
        
    Returns:
        wandb run object or None if disabled
    """
    if not config.get('log_wandb', False):
        logger.info("W&B logging disabled")
        return None
    
    try:
        import wandb
        
        # Initialize wandb
        run = wandb.init(
            project=config.get('project', 'FinderPartner-CLIP'),
            name=config.get('run_name'),
            config=config,
            dir=str(experiment_dir)
        )
        
        logger.info(f"W&B logging initialized: {run.name}")
        return run
        
    except ImportError:
        logger.warning("wandb not available, skipping W&B logging")
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize W&B: {e}")
        return None


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    step: int,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: Path
) -> None:
    """Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler
        step: Current training step
        epoch: Current epoch
        metrics: Current metrics
        checkpoint_dir: Directory to save checkpoint
    """
    checkpoint_path = checkpoint_dir / f"checkpoint-{step}.pt"
    
    checkpoint = {
        'step': step,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    # Also save the model using the model's save_pretrained method
    model_dir = checkpoint_dir / f"model-{step}"
    model.save_pretrained(str(model_dir))
    
    logger.info(f"Checkpoint saved: {checkpoint_path}")


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0
) -> Dict[str, float]:
    """Perform single training step
    
    Args:
        model: Model to train
        batch: Training batch
        optimizer: Optimizer
        scaler: Gradient scaler for mixed precision
        device: Training device
        gradient_accumulation_steps: Number of steps to accumulate gradients
        max_grad_norm: Maximum gradient norm for clipping
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    # Move batch to device
    pixel_values = batch['pixel_values'].to(device, non_blocking=True)
    input_ids = batch['input_ids'].to(device, non_blocking=True)
    attention_mask = batch['attention_mask'].to(device, non_blocking=True)
    
    # Forward pass with mixed precision
    if scaler is not None:
        try:
            # Use new API if available
            with torch.amp.autocast('cuda'):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_loss=True
                )
                loss = outputs['loss'] / gradient_accumulation_steps
        except (AttributeError, TypeError):
            # Fallback to old API
            with torch.cuda.amp.autocast():
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_loss=True
                )
                loss = outputs['loss'] / gradient_accumulation_steps
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
    else:
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_loss=True
        )
        loss = outputs['loss'] / gradient_accumulation_steps
        loss.backward()
    
    return {'loss': outputs['loss'].item()}


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train CLIP model for FinderPartner')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--create_sample_data', action='store_true',
                       help='Create sample data for testing')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        validate_config(config)
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)
    
    # Setup logging
    setup_logging(config.get('log_level', 'INFO'))
    
    # Create sample data if requested
    if args.create_sample_data:
        logger.info("Creating sample data for testing...")
        data_dir = Path('data')
        create_sample_data(data_dir, num_samples=100)
        logger.info("Sample data created. Update config paths and run again.")
        return
    
    # Setup experiment directory
    experiment_dir = setup_experiment_dir(config)
    
    # Setup device
    device = setup_device(config)
    
    # Setup mixed precision scaler
    scaler = None
    if device.type == 'cuda' and config.get('mixed_precision', True):
        try:
            # Use new API if available, fallback to old one
            scaler = torch.amp.GradScaler('cuda')
        except AttributeError:
            scaler = torch.cuda.amp.GradScaler()
        logger.info("Mixed precision training enabled")
    
    try:
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader = create_dataloaders(config)
        
        # Calculate training steps
        num_training_steps = len(train_loader) * config['epochs']
        
        # Create model
        logger.info("Creating model...")
        model = create_model(config)
        model = model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Setup optimizer and scheduler
        optimizer, scheduler = setup_optimizer_and_scheduler(model, config, num_training_steps)
        
        # Setup logging
        wandb_run = setup_wandb_logging(config, experiment_dir)
        metrics_tracker = MetricsTracker()
        
        # Training loop
        logger.info("Starting training...")
        step = 0
        best_val_loss = float('inf')
        
        for epoch in range(config['epochs']):
            logger.info(f"Epoch {epoch + 1}/{config['epochs']}")
            
            # Training phase
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Training step
                step_metrics = train_step(
                    model, batch, optimizer, scaler, device,
                    config.get('gradient_accumulation_steps', 1),
                    config.get('max_grad_norm', 1.0)
                )
                
                # Update optimizer every gradient_accumulation_steps
                if (batch_idx + 1) % config.get('gradient_accumulation_steps', 1) == 0:
                    if scaler is not None:
                        # Gradient clipping with mixed precision
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 
                            config.get('max_grad_norm', 1.0)
                        )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Standard gradient clipping and step
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 
                            config.get('max_grad_norm', 1.0)
                        )
                        optimizer.step()
                    
                    optimizer.zero_grad()
                    scheduler.step()  # Step scheduler after optimizer
                    step += 1
                
                # Accumulate metrics
                epoch_loss += step_metrics['loss']
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{step_metrics['loss']:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
                
                # Evaluation
                if step % config.get('eval_steps', 500) == 0 and step > 0:
                    logger.info("Running validation...")
                    val_metrics = evaluate_model(
                        model, val_loader, device,
                        compute_retrieval=config.get('compute_retrieval_metrics', True),
                        k_values=config.get('retrieval_k_values', [1, 5, 10])
                    )
                    
                    # Log metrics
                    train_loss = epoch_loss / max(num_batches, 1)
                    all_metrics = {
                        'train/loss': train_loss,
                        'train/lr': scheduler.get_last_lr()[0],
                        'step': step,
                        'epoch': epoch
                    }
                    
                    # Add validation metrics
                    for key, value in val_metrics.items():
                        all_metrics[f'val/{key}'] = value
                    
                    # Update trackers
                    metrics_tracker.update(all_metrics, step, epoch)
                    
                    if wandb_run:
                        wandb_run.log(all_metrics)
                    
                    logger.info(f"Step {step}: train_loss={train_loss:.4f}, "
                              f"val_loss={val_metrics.get('loss', 0):.4f}")
                
                # Save checkpoint
                if step % config.get('save_steps', 1000) == 0 and step > 0:
                    current_metrics = {
                        'train_loss': epoch_loss / max(num_batches, 1),
                        'val_loss': metrics_tracker.get_latest('val/loss')
                    }
                    
                    save_checkpoint(
                        model, optimizer, scheduler, step, epoch,
                        current_metrics, experiment_dir / 'checkpoints'
                    )
                    
                    # Save best model
                    val_loss = current_metrics['val_loss']
                    if val_loss < best_val_loss and val_loss > 0:
                        best_val_loss = val_loss
                        best_model_dir = experiment_dir / 'best_model'
                        model.save_pretrained(str(best_model_dir))
                        logger.info(f"New best model saved (val_loss: {val_loss:.4f})")
            
            # End of epoch logging
            avg_epoch_loss = epoch_loss / max(num_batches, 1)  # Avoid division by zero
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Handle case where no batches were processed
            if num_batches == 0:
                logger.warning(f"No batches processed in epoch {epoch + 1}. Check batch size and dataset size.")
                logger.warning(f"Dataset sizes: train={len(train_loader.dataset)}, batch_size={config['batch_size']}")
                break
        
        # Final evaluation
        logger.info("Running final evaluation...")
        final_metrics = evaluate_model(
            model, val_loader, device,
            compute_retrieval=True,
            k_values=config.get('retrieval_k_values', [1, 5, 10])
        )
        
        logger.info("Final validation metrics:")
        for key, value in final_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # Save final model
        final_model_dir = experiment_dir / 'final_model'
        model.save_pretrained(str(final_model_dir))
        
        # Save metrics history
        metrics_tracker.save_history(experiment_dir / 'metrics_history.json')
        
        if wandb_run:
            wandb_run.finish()
        
        logger.info(f"Training completed! Results saved to: {experiment_dir}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if wandb_run:
            wandb_run.finish()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        if wandb_run:
            wandb_run.finish()
        sys.exit(1)


if __name__ == "__main__":
    main()
