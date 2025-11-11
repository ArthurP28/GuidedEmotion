# Quick Start Guide

This guide will help you get started quickly with the GuidedEmotion diffusion model.

## Quick Setup (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
python test_demo.py
```
You should see all tests pass ✓

## Training (2-3 hours on GPU)

### Quick Training Run
Train on the "Smiling" attribute with minimal epochs:
```bash
python train.py --attribute Smiling --classifier_epochs 5 --diffusion_epochs 20
```

### Full Training Run
For better results, use more epochs:
```bash
python train.py --attribute Smiling --classifier_epochs 10 --diffusion_epochs 50
```

### Training Different Attributes
```bash
# Train on "Young" attribute
python train.py --attribute Young --diffusion_epochs 50

# Train on "Male" attribute
python train.py --attribute Male --diffusion_epochs 50

# Train on "Eyeglasses" attribute
python train.py --attribute Eyeglasses --diffusion_epochs 50
```

## Generation (< 1 minute)

### Generate Smiling Faces
```bash
python generate.py --target_label 1 --guidance_scale 3.0 --num_samples 16
```

### Generate Non-Smiling Faces
```bash
python generate.py --target_label 0 --guidance_scale 3.0 --num_samples 16
```

### Experiment with Guidance Scale
```bash
# Weak guidance (subtle effect)
python generate.py --target_label 1 --guidance_scale 1.0

# Medium guidance (balanced)
python generate.py --target_label 1 --guidance_scale 3.0

# Strong guidance (pronounced effect)
python generate.py --target_label 1 --guidance_scale 5.0
```

### Generate Without Guidance
```bash
python generate.py --no_guidance --num_samples 16
```

## Output Files

After training, you'll have:
- `classifier_best.pth` - Trained classifier (~1.7MB)
- `diffusion_final.pth` - Trained diffusion model (~4.4MB)
- `diffusion_epoch_*.pth` - Intermediate checkpoints

After generation, you'll have:
- `generated_samples.png` - Grid of all samples
- `generated_samples/` - Individual sample images

## Tips & Tricks

### Memory Issues?
If you run out of memory:
```bash
# Reduce batch size
python train.py --batch_size 16

# Use CPU (slower but no GPU memory needed)
python train.py --device cpu
```

### Quick Testing
For rapid iteration and testing:
```bash
# Minimal training (just to test the pipeline)
python train.py --classifier_epochs 2 --diffusion_epochs 5
```

### Best Quality Results
For publication-quality results:
```bash
# Train longer with more timesteps
python train.py --diffusion_epochs 100 --timesteps 1000
```

## Example Workflow

Here's a complete workflow from scratch:

```bash
# 1. Verify everything works
python test_demo.py

# 2. Train the models (this will download CelebA on first run)
python train.py --attribute Smiling --diffusion_epochs 50

# 3. Generate smiling faces
python generate.py --target_label 1 --guidance_scale 3.0

# 4. Generate non-smiling faces
python generate.py --target_label 0 --guidance_scale 3.0

# 5. Compare with unconditional generation
python generate.py --no_guidance
```

## Troubleshooting

### Dataset Download Issues
If CelebA download fails:
1. Check your internet connection
2. The dataset is large (~1.4GB), so it may take a while
3. Make sure you have enough disk space (~5GB total)

### CUDA Out of Memory
```bash
# Use smaller batch size
python train.py --batch_size 16

# Or use CPU
python train.py --device cpu
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## Understanding the Results

### Classifier Accuracy
- Good: >80%
- Excellent: >85%
- If <70%, consider training longer or checking data

### Diffusion Loss
- Should decrease steadily during training
- Final loss: typically 0.05-0.15
- If loss doesn't decrease, check learning rate

### Generated Samples
- With guidance: Should show target attribute
- Without guidance: Random mix of attributes
- Higher guidance scale = stronger effect

## Next Steps

1. Experiment with different attributes
2. Try different guidance scales
3. Train for more epochs for better quality
4. Modify the architecture in `diffusion_model.py` and `classifier.py`

## Questions?

Check the main README.md for detailed documentation.
