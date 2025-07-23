
"""
Evaluation utilities for CLIP model
"""
import logging
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def compute_retrieval_metrics(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """Compute image-text retrieval metrics
    
    Args:
        image_embeds: Image embeddings [N, embed_dim]
        text_embeds: Text embeddings [N, embed_dim]
        k_values: List of k values for Recall@k computation
        
    Returns:
        Dictionary with retrieval metrics
    """
    device = image_embeds.device
    batch_size = image_embeds.shape[0]
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(image_embeds, text_embeds.t())
    
    metrics = {}
    
    # Image-to-text retrieval
    for k in k_values:
        # Get top-k text indices for each image
        _, top_k_indices = torch.topk(similarity_matrix, k, dim=1)
        
        # Check if correct text (same index) is in top-k
        correct_indices = torch.arange(batch_size, device=device).unsqueeze(1)
        hits = (top_k_indices == correct_indices).any(dim=1).float()
        
        recall_at_k = hits.mean().item()
        metrics[f'image_to_text_recall@{k}'] = recall_at_k
    
    # Text-to-image retrieval
    for k in k_values:
        # Get top-k image indices for each text
        _, top_k_indices = torch.topk(similarity_matrix.t(), k, dim=1)
        
        # Check if correct image (same index) is in top-k
        correct_indices = torch.arange(batch_size, device=device).unsqueeze(1)
        hits = (top_k_indices == correct_indices).any(dim=1).float()
        
        recall_at_k = hits.mean().item()
        metrics[f'text_to_image_recall@{k}'] = recall_at_k
    
    # Compute mean recall across both directions
    for k in k_values:
        img_to_text = metrics[f'image_to_text_recall@{k}']
        text_to_img = metrics[f'text_to_image_recall@{k}']
        metrics[f'mean_recall@{k}'] = (img_to_text + text_to_img) / 2
    
    return metrics


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    compute_retrieval: bool = True,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """Evaluate model on validation/test set
    
    Args:
        model: CLIP model to evaluate
        dataloader: Validation/test dataloader
        device: Device to run evaluation on
        compute_retrieval: Whether to compute retrieval metrics
        k_values: List of k values for Recall@k computation
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    all_image_embeds = []
    all_text_embeds = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
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
            
            # Accumulate loss
            total_loss += outputs['loss'].item()
            num_batches += 1
            
            # Collect embeddings for retrieval metrics
            if compute_retrieval:
                all_image_embeds.append(outputs['image_embeds'].cpu())
                all_text_embeds.append(outputs['text_embeds'].cpu())
    
    # Compute average loss
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    metrics = {'loss': avg_loss}
    
    # Compute retrieval metrics
    if compute_retrieval and all_image_embeds:
        # Concatenate all embeddings
        image_embeds = torch.cat(all_image_embeds, dim=0)
        text_embeds = torch.cat(all_text_embeds, dim=0)
        
        # Compute retrieval metrics
        retrieval_metrics = compute_retrieval_metrics(
            image_embeds, text_embeds, k_values
        )
        metrics.update(retrieval_metrics)
        
        logger.info(f"Retrieval evaluation completed on {len(image_embeds)} samples")
    
    return metrics


def compute_similarity_distribution(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor
) -> Dict[str, float]:
    """Compute statistics about similarity score distribution
    
    Args:
        image_embeds: Image embeddings [N, embed_dim]
        text_embeds: Text embeddings [N, embed_dim]
        
    Returns:
        Dictionary with similarity statistics
    """
    # Compute similarity matrix
    similarity_matrix = torch.matmul(image_embeds, text_embeds.t())
    
    # Get diagonal (positive pairs) and off-diagonal (negative pairs)
    batch_size = image_embeds.shape[0]
    positive_similarities = torch.diagonal(similarity_matrix)
    
    # Create mask for negative pairs
    mask = torch.eye(batch_size, device=similarity_matrix.device).bool()
    negative_similarities = similarity_matrix[~mask]
    
    # Compute statistics
    stats = {
        'positive_mean': positive_similarities.mean().item(),
        'positive_std': positive_similarities.std().item(),
        'negative_mean': negative_similarities.mean().item(),
        'negative_std': negative_similarities.std().item(),
        'positive_min': positive_similarities.min().item(),
        'positive_max': positive_similarities.max().item(),
        'negative_min': negative_similarities.min().item(),
        'negative_max': negative_similarities.max().item(),
    }
    
    # Compute separation (difference between positive and negative means)
    stats['separation'] = stats['positive_mean'] - stats['negative_mean']
    
    return stats


class MetricsTracker:
    """Helper class to track training metrics"""
    
    def __init__(self):
        self.history = {}
        self.current_epoch = 0
        self.current_step = 0
    
    def update(self, metrics: Dict[str, float], step: int = None, epoch: int = None):
        """Update metrics history
        
        Args:
            metrics: Dictionary with metric values
            step: Current training step
            epoch: Current epoch
        """
        if step is not None:
            self.current_step = step
        if epoch is not None:
            self.current_epoch = epoch
        
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append({
                'value': value,
                'step': self.current_step,
                'epoch': self.current_epoch
            })
    
    def get_latest(self, metric_name: str) -> float:
        """Get latest value for a metric
        
        Args:
            metric_name: Name of metric
            
        Returns:
            Latest metric value or 0.0 if not found
        """
        if metric_name in self.history and self.history[metric_name]:
            return self.history[metric_name][-1]['value']
        return 0.0
    
    def get_best(self, metric_name: str, mode: str = 'max') -> Tuple[float, int]:
        """Get best value and step for a metric
        
        Args:
            metric_name: Name of metric
            mode: 'max' for maximum value, 'min' for minimum value
            
        Returns:
            Tuple of (best_value, best_step)
        """
        if metric_name not in self.history or not self.history[metric_name]:
            return 0.0, 0
        
        values = [(entry['value'], entry['step']) for entry in self.history[metric_name]]
        
        if mode == 'max':
            best_value, best_step = max(values, key=lambda x: x[0])
        else:
            best_value, best_step = min(values, key=lambda x: x[0])
        
        return best_value, best_step
    
    def save_history(self, filepath: str):
        """Save metrics history to file
        
        Args:
            filepath: Path to save metrics
        """
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"Metrics history saved to {filepath}")
    
    def summary(self) -> Dict[str, Dict]:
        """Get summary of all metrics
        
        Returns:
            Dictionary with metric summaries
        """
        summary = {}
        
        for metric_name in self.history:
            values = [entry['value'] for entry in self.history[metric_name]]
            
            if values:
                summary[metric_name] = {
                    'count': len(values),
                    'latest': values[-1],
                    'best': max(values),
                    'worst': min(values),
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
        
        return summary
