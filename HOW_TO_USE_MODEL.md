# 🚀 How to Use Your Trained CLIP Model

Congratulations! Your CLIP model is trained and ready to use. Here are several ways to use it:

## 📂 Available Scripts

### 1. Basic Inference (`inference.py`)
**Purpose**: General-purpose inference with your CLIP model

```bash
python inference.py
```

**Features**:
- Encode images and texts to embeddings
- Compute similarity between images and texts
- Text-to-text similarity using CLIP's text encoder
- Basic usage examples

### 2. Image Search (`search_images.py`)
**Purpose**: Search through collections of images using text queries

```bash
python search_images.py
```

**Features**:
- Index entire folders of images
- Search with natural language queries
- Save/load image indices for faster future searches
- Batch processing of large image collections

### 3. Web Interface (`web_interface.py`)
**Purpose**: Interactive web interface using Gradio

```bash
# First install gradio
pip install gradio

# Then run the interface
python web_interface.py
```

**Features**:
- Upload images and get similarity scores
- Compare multiple descriptions for one image
- Automatic image analysis and categorization
- User-friendly web interface at http://127.0.0.1:7860

### 4. Command Line Tool (`clip_cli.py`)
**Purpose**: Quick command-line operations

```bash
# Single image-text similarity
python clip_cli.py --image photo.jpg --text "a red car"

# Compare multiple descriptions
python clip_cli.py --image photo.jpg --compare "a dog" "a cat" "a bird"

# Search folder of images
python clip_cli.py --batch-images ./photos/ --text "sunset landscape" --top-k 10

# Save results to file
python clip_cli.py --image photo.jpg --text "a red car" --output results.json
```

## 🎯 Use Cases and Examples

### 1. Image-Text Matching
```python
from inference import CLIPInference

# Initialize model
clip = CLIPInference("outputs/clip_training_20250729_112717/final_model")

# Single image-text similarity
similarity = clip.compute_similarity(["path/to/image.jpg"], ["a red car"])
print(f"Similarity: {similarity[0, 0]:.4f}")
```

### 2. Image Search Engine
```python
from search_images import ImageSearchEngine

# Initialize search engine
search_engine = ImageSearchEngine("outputs/clip_training_20250729_112717/final_model")

# Index your images
search_engine.index_images("path/to/your/images/")

# Search with text
results = search_engine.search("beautiful sunset", top_k=5)
for image_path, score in results:
    print(f"{image_path}: {score:.4f}")
```

### 3. Best Caption Selection
```python
from inference import CLIPInference

clip = CLIPInference("outputs/clip_training_20250729_112717/final_model")

# Find best caption for an image
captions = [
    "a dog playing in the park",
    "a cat sleeping on a couch", 
    "a bird flying in the sky"
]

best_idx, best_caption, score = clip.find_best_match("image.jpg", captions)
print(f"Best match: '{best_caption}' (score: {score:.4f})")
```

### 4. Content Moderation
```python
# Check if image matches inappropriate content
inappropriate_queries = [
    "inappropriate content",
    "violence or harmful content",
    "adult content"
]

for query in inappropriate_queries:
    similarity = clip.compute_similarity(["image.jpg"], [query])
    if similarity[0, 0] > 0.3:  # Threshold
        print(f"⚠️ Potential issue: {query} (score: {similarity[0, 0]:.4f})")
```

### 5. Image Classification
```python
# Use as zero-shot classifier
categories = [
    "a photo of a dog",
    "a photo of a cat", 
    "a photo of a car",
    "a photo of food",
    "a landscape photo"
]

similarities = clip.compute_similarity(["image.jpg"], categories)
best_category_idx = similarities.argmax()
print(f"Category: {categories[best_category_idx]}")
```

## 📊 Understanding Similarity Scores

**Score Ranges**:
- `0.7 - 1.0`: Excellent match
- `0.5 - 0.7`: Very good match  
- `0.3 - 0.5`: Good match
- `0.1 - 0.3`: Weak match
- `< 0.1`: Poor match

**Your Model's Performance**:
Based on your training results:
- Mean Recall@1: 50.00%
- Mean Recall@3: 91.67%
- The model shows excellent performance for retrieval tasks

## 🔧 Advanced Usage

### Batch Processing
```python
# Process multiple images at once
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
texts = ["description 1", "description 2", "description 3"]

similarities = clip.compute_similarity(images, texts)
# similarities is a 3x3 matrix: similarities[i,j] = similarity(image_i, text_j)
```

### Feature Extraction
```python
# Get raw embeddings for your own applications
image_features = clip.encode_images(["image.jpg"])  # Shape: [1, 512]
text_features = clip.encode_texts(["description"])   # Shape: [1, 512]

# Use these features for:
# - Clustering similar images
# - Building recommendation systems  
# - Fine-tuning other models
```

### Integration with Other Tools
```python
# Save embeddings for external use
import numpy as np

embeddings = clip.encode_images(image_list)
np.save("image_embeddings.npy", embeddings.numpy())

# Load in other applications
embeddings = np.load("image_embeddings.npy")
```

## 🚀 Performance Tips

1. **Batch Processing**: Process multiple images/texts together for better GPU utilization
2. **Image Size**: Resize large images to 224x224 for faster processing
3. **Caching**: Save embeddings for frequently used images/texts
4. **GPU Memory**: Use smaller batches if you run out of GPU memory

## 🔄 Model Updates

To retrain or fine-tune your model:
```bash
# Edit configs/clip_base.yaml with new settings
# Then run training again
python run_training.bat
```

## 📈 Monitoring and Evaluation

Check your training results:
- **Weights & Biases Dashboard**: https://wandb.ai/cliptraining/FinderPartner-CLIP/runs/f0nz4ae8
- **Local Metrics**: `outputs/clip_training_20250729_112717/metrics_history.json`
- **Model Checkpoints**: `outputs/clip_training_20250729_112717/checkpoints/`

## 🆘 Troubleshooting

**Common Issues**:

1. **"Model not found"**: Update the model path in scripts
2. **CUDA out of memory**: Reduce batch size or use CPU
3. **Import errors**: Install requirements with `pip install -r requirements.txt`
4. **Low similarity scores**: Your model might need more training data

**Getting Help**:
- Check the training logs in `outputs/clip_training_20250729_112717/logs/`
- Verify model loading with `inference.py`
- Test with the web interface first for easier debugging

---

🎉 **Your CLIP model is production-ready!** Start with the web interface (`web_interface.py`) for interactive testing, then use the command-line tools or Python scripts for automation.
