#!/usr/bin/env python3
"""
AI Profile Matcher - CLIP-based profile evaluation system

This tool uses CLIP (Contrastive Language-Image Pre-training) to evaluate
whether a profile (collection of images) matches a text description.
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional
import warnings

# Suppress some warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from embedding import CLIPEmbedder
from scorer import ProfileScorer, print_evaluation_results
from utils import (
    load_images_from_directory,
    validate_image_path,
    setup_logging,
    print_directory_summary
)

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


def load_config_from_env() -> dict:
    """Load configuration from environment variables
    
    Returns:
        Dictionary with configuration values
    """
    config = {
        'model_name': os.getenv('MODEL_NAME', 'ViT-L-14'),
        'pretrained': os.getenv('PRETRAINED', 'openai'),
        'threshold': float(os.getenv('THRESHOLD', '0.25')),
        'score_method': os.getenv('SCORE_METHOD', 'mean'),
        'batch_size': int(os.getenv('BATCH_SIZE', '32')),
        'log_level': os.getenv('LOG_LEVEL', 'INFO')
    }
    return config


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='AI Profile Matcher - Evaluate profile images against text prompt using CLIP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py --prompt "สาวเอเชีย ใส่แว่น ผมยาว" --img_dir ./temp
  python src/main.py --prompt "cute girl with glasses" --img_dir ./profiles/user123 --threshold 0.3
  python src/main.py --interactive --img_dir ./temp --method max
        """
    )
    
    # Main arguments
    parser.add_argument(
        '--prompt', '-p',
        type=str,
        help='Text prompt describing desired profile characteristics'
    )
    
    parser.add_argument(
        '--img_dir', '-i',
        type=str,
        default='./temp',
        help='Directory containing profile images (default: ./temp)'
    )
    
    # Model configuration
    parser.add_argument(
        '--model',
        type=str,
        help='CLIP model name (default: from env or ViT-L-14)'
    )
    
    parser.add_argument(
        '--pretrained',
        type=str,
        help='Pretrained weights (default: from env or openai)'
    )
    
    # Scoring configuration
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        help='Threshold for LIKE decision (default: from env or 0.25)'
    )
    
    parser.add_argument(
        '--method', '-m',
        choices=['mean', 'max', 'weighted_mean', 'top_k'],
        help='Score aggregation method (default: from env or mean)'
    )
    
    # Output options
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive mode - prompt for text input'
    )
    
    parser.add_argument(
        '--show_all',
        action='store_true',
        help='Show all image scores (not just top 5)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed output, show only final decision'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def get_text_prompt(args: argparse.Namespace) -> str:
    """Get text prompt from arguments or interactive input
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Text prompt string
    """
    if args.prompt:
        return args.prompt
    
    if args.interactive:
        print("\n🎯 AI Profile Matcher - Interactive Mode")
        print("=" * 50)
        prompt = input("\nEnter your ideal profile description: ").strip()
        if not prompt:
            print("❌ Empty prompt provided")
            sys.exit(1)
        return prompt
    
    print("❌ Error: No prompt provided. Use --prompt or --interactive")
    sys.exit(1)


def main() -> None:
    """Main application entry point"""
    args = parse_arguments()
    
    # Load configuration
    config = load_config_from_env()
    
    # Override config with command line arguments
    if args.model:
        config['model_name'] = args.model
    if args.pretrained:
        config['pretrained'] = args.pretrained
    if args.threshold is not None:
        config['threshold'] = args.threshold
    if args.method:
        config['score_method'] = args.method
    if args.verbose:
        config['log_level'] = 'DEBUG'
    elif args.quiet:
        config['log_level'] = 'WARNING'
    
    # Setup logging
    setup_logging(config['log_level'])
    
    if not args.quiet:
        print("\n🤖 AI Profile Matcher")
        print("=" * 50)
        print(f"Model: {config['model_name']} ({config['pretrained']})")
        print(f"Threshold: {config['threshold']}")
        print(f"Method: {config['score_method']}")
    
    try:
        # Get text prompt
        text_prompt = get_text_prompt(args)
        if not args.quiet:
            print(f"Prompt: \"{text_prompt}\"")
        
        # Validate image directory
        img_dir = validate_image_path(args.img_dir)
        if not args.quiet:
            print_directory_summary(img_dir)
        
        # Initialize CLIP embedder
        if not args.quiet:
            print("\n⚡ Loading CLIP model...")
        embedder = CLIPEmbedder(
            model_name=config['model_name'],
            pretrained=config['pretrained']
        )
        
        # Load images
        if not args.quiet:
            print("\n📁 Loading images...")
        images_data = load_images_from_directory(img_dir)
        image_paths, images = zip(*images_data)
        
        # Encode text prompt
        if not args.quiet:
            print(f"\n📝 Encoding text prompt...")
        text_embedding = embedder.encode_text(text_prompt)
        
        # Encode images
        if not args.quiet:
            print(f"🖼️  Encoding {len(images)} images...")
        
        # Use batch encoding for efficiency
        image_embeddings = embedder.encode_images_batch(
            list(images), 
            batch_size=config['batch_size']
        )
        
        # Initialize scorer and evaluate
        if not args.quiet:
            print("🧮 Calculating similarity scores...")
        
        scorer = ProfileScorer(
            threshold=config['threshold'],
            score_method=config['score_method']
        )
        
        results = scorer.evaluate_profile(
            text_embedding, 
            image_embeddings, 
            list(image_paths)
        )
        
        # Print results
        if args.quiet:
            # Minimal output for quiet mode
            decision = "MATCHED" if results['is_like'] else "NOT MATCHED"
            print(f"{decision} (score: {results['profile_score']:.4f})")
        else:
            print_evaluation_results(results, show_all_images=args.show_all)
        
        # Exit with appropriate code
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)
        
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()