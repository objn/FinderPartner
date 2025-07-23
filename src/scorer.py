"""
Scoring logic for profile matching based on CLIP embeddings
"""
import logging
from typing import List, Dict, Tuple, Literal
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

ScoreMethod = Literal["mean", "max", "weighted_mean", "top_k"]


class ProfileScorer:
    """Calculate profile matching scores from embeddings"""
    
    def __init__(self, threshold: float = 0.25, score_method: ScoreMethod = "mean"):
        """Initialize scorer with threshold and aggregation method
        
        Args:
            threshold: Minimum score for LIKE decision
            score_method: How to aggregate multiple image scores
        """
        self.threshold = threshold
        self.score_method = score_method
        logger.info(f"ProfileScorer initialized: threshold={threshold}, method={score_method}")
    
    def calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector (text embedding)
            vec2: Second vector (image embedding)
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            # Vectors should already be normalized from CLIP
            similarity = np.dot(vec1, vec2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Failed to calculate cosine similarity: {e}")
            return 0.0
    
    def calculate_image_scores(
        self, 
        text_embedding: np.ndarray, 
        image_embeddings: List[np.ndarray],
        image_paths: List[Path]
    ) -> List[Dict]:
        """Calculate similarity scores for all images
        
        Args:
            text_embedding: Normalized text embedding vector
            image_embeddings: List of normalized image embedding vectors
            image_paths: List of corresponding image file paths
            
        Returns:
            List of dictionaries with image info and scores
        """
        scores = []
        
        for i, (img_embedding, img_path) in enumerate(zip(image_embeddings, image_paths)):
            similarity = self.calculate_cosine_similarity(text_embedding, img_embedding)
            
            scores.append({
                'path': img_path,
                'filename': img_path.name,
                'score': similarity,
                'index': i
            })
        
        # Sort by score descending
        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores
    
    def aggregate_scores(self, scores: List[float]) -> float:
        """Aggregate multiple image scores into single profile score
        
        Args:
            scores: List of individual image similarity scores
            
        Returns:
            Aggregated profile score
        """
        if not scores:
            return 0.0
        
        scores_array = np.array(scores)
        
        if self.score_method == "mean":
            return float(np.mean(scores_array))
        
        elif self.score_method == "max":
            return float(np.max(scores_array))
        
        elif self.score_method == "weighted_mean":
            # Give higher weight to top scores
            sorted_scores = np.sort(scores_array)[::-1]  # descending
            weights = np.exp(-0.5 * np.arange(len(sorted_scores)))  # exponential decay
            weights = weights / np.sum(weights)  # normalize weights
            return float(np.sum(sorted_scores * weights))
        
        elif self.score_method == "top_k":
            # Use average of top 3 scores (or all if fewer than 3)
            k = min(3, len(scores_array))
            top_scores = np.sort(scores_array)[-k:]  # top k scores
            return float(np.mean(top_scores))
        
        else:
            logger.warning(f"Unknown score method '{self.score_method}', using mean")
            return float(np.mean(scores_array))
    
    def make_decision(self, profile_score: float) -> Tuple[bool, str]:
        """Make LIKE/UNLIKE decision based on profile score
        
        Args:
            profile_score: Aggregated profile score
            
        Returns:
            Tuple of (is_like, decision_string)
        """
        is_like = profile_score >= self.threshold
        decision = "LIKE" if is_like else "UNLIKE"
        
        confidence = abs(profile_score - self.threshold)
        confidence_level = "HIGH" if confidence > 0.1 else "MEDIUM" if confidence > 0.05 else "LOW"
        
        return is_like, f"{decision} (confidence: {confidence_level})"
    
    def evaluate_profile(
        self,
        text_embedding: np.ndarray,
        image_embeddings: List[np.ndarray],
        image_paths: List[Path]
    ) -> Dict:
        """Complete profile evaluation pipeline
        
        Args:
            text_embedding: Text prompt embedding
            image_embeddings: List of image embeddings
            image_paths: List of image file paths
            
        Returns:
            Dictionary with complete evaluation results
        """
        # Calculate individual image scores
        image_scores = self.calculate_image_scores(
            text_embedding, image_embeddings, image_paths
        )
        
        # Extract scores for aggregation
        raw_scores = [score['score'] for score in image_scores]
        
        # Calculate profile score
        profile_score = self.aggregate_scores(raw_scores)
        
        # Make decision
        is_like, decision_text = self.make_decision(profile_score)
        
        # Calculate statistics
        stats = {
            'count': len(raw_scores),
            'mean': np.mean(raw_scores),
            'max': np.max(raw_scores),
            'min': np.min(raw_scores),
            'std': np.std(raw_scores)
        }
        
        return {
            'profile_score': profile_score,
            'is_like': is_like,
            'decision': decision_text,
            'method': self.score_method,
            'threshold': self.threshold,
            'image_scores': image_scores,
            'statistics': stats
        }


def print_evaluation_results(results: Dict, show_all_images: bool = True) -> None:
    """Print formatted evaluation results
    
    Args:
        results: Results dictionary from evaluate_profile
        show_all_images: Whether to show all image scores or just top 5
    """
    print("\n" + "="*60)
    print("🎯 PROFILE EVALUATION RESULTS")
    print("="*60)
    
    # Main decision
    decision_emoji = "👍" if results['is_like'] else "👎"
    print(f"\n{decision_emoji} DECISION: {results['decision']}")
    print(f"📊 Profile Score: {results['profile_score']:.4f}")
    print(f"🎚️  Threshold: {results['threshold']:.4f}")
    print(f"📈 Method: {results['method']}")
    
    # Statistics
    stats = results['statistics']
    print(f"\n📋 STATISTICS:")
    print(f"   Images processed: {stats['count']}")
    print(f"   Average similarity: {stats['mean']:.4f}")
    print(f"   Best match: {stats['max']:.4f}")
    print(f"   Worst match: {stats['min']:.4f}")
    print(f"   Standard deviation: {stats['std']:.4f}")
    
    # Individual image scores
    image_scores = results['image_scores']
    display_count = len(image_scores) if show_all_images else min(5, len(image_scores))
    
    print(f"\n🖼️  TOP {display_count} IMAGE SCORES:")
    print("-" * 50)
    
    for i, score_info in enumerate(image_scores[:display_count]):
        emoji = "⭐" if score_info['score'] >= results['threshold'] else "📷"
        print(f"{emoji} {score_info['filename']:<30} {score_info['score']:.4f}")
    
    if not show_all_images and len(image_scores) > 5:
        remaining = len(image_scores) - 5
        print(f"   ... and {remaining} more images")
    
    print("\n" + "="*60)