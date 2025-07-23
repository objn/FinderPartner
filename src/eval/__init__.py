"""
Evaluation module for CLIP training pipeline
"""

from .retrieval import (
    compute_retrieval_metrics,
    evaluate_model,
    compute_similarity_distribution,
    MetricsTracker
)

__all__ = [
    'compute_retrieval_metrics',
    'evaluate_model', 
    'compute_similarity_distribution',
    'MetricsTracker'
]
