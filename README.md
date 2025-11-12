# GuidedEmotion

Diffusion-based generative model for counterfactual sample generation using classifier guidance on the CelebA dataset.

## Overview

This project implements a minimal diffusion model that can generate counterfactual samples - images with specific attributes guided by a trained classifier. The implementation uses only 50% of the CelebA dataset to address memory constraints while maintaining a simple, easy-to-understand architecture.

## Features

- **Minimal Diffusion Model**: Simple UNet-based architecture for the diffusion process
- **Classifier Guidance**: Guide generation towards specific attributes (e.g., smiling, young, male)
- **Memory Efficient**: Uses only 50% of CelebA dataset
- **Counterfactual Generation**: Generate samples with desired attributes

## Architecture

### Diffusion Model
- Minimal UNet with encoder-decoder structure
- Sinusoidal position embeddings for timesteps
- Linear noise schedule (1000 timesteps)
- Simple convolutional blocks with time conditioning

### Classifier
- Lightweight CNN for attribute classification
- Used for guiding the diffusion process
- Binary classification (attribute present/absent)

## Installation

```bash
# Clone the repository
git clone https://github.com/ArthurP28/GuidedEmotion.git
cd GuidedEmotion

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Train the Models

Train both the classifier and diffusion model on CelebA dataset (uses only 50% of data):

```bash
python train.py --attribute Smiling --classifier_epochs 10 --diffusion_epochs 50
```

**Arguments:**
- `--data_root`: Root directory for dataset (default: `./data`)
- `--image_size`: Image size (default: 64)
- `--batch_size`: Batch size (default: 32)
- `--attribute`: CelebA attribute to use (default: 'Smiling')
  - Available: 'Smiling', 'Young', 'Male', 'Attractive', 'Eyeglasses', etc.
- `--classifier_epochs`: Epochs for classifier training (default: 10)
- `--diffusion_epochs`: Epochs for diffusion training (default: 50)
- `--timesteps`: Number of diffusion timesteps (default: 1000)
- `--device`: Device to use (default: 'cuda')

**Output:**
- `classifier_best.pth`: Best classifier model
- `diffusion_final.pth`: Final diffusion model
- `diffusion_epoch_*.pth`: Intermediate checkpoints

### 2. Generate Counterfactual Samples

Generate samples with classifier guidance:

```bash
# Generate samples with smiling attribute (label=1)
python generate.py --target_label 1 --guidance_scale 3.0 --num_samples 16

# Generate samples without smiling attribute (label=0)
python generate.py --target_label 0 --guidance_scale 3.0 --num_samples 16

# Generate without guidance (unconditional)
python generate.py --no_guidance --num_samples 16
```

**Arguments:**
- `--diffusion_model`: Path to diffusion model (default: `diffusion_final.pth`)
- `--classifier_model`: Path to classifier model (default: `classifier_best.pth`)
- `--num_samples`: Number of samples to generate (default: 16)
- `--guidance_scale`: Classifier guidance scale (default: 3.0)
  - Higher values = stronger guidance towards target attribute
  - Typical range: 1.0-5.0
- `--target_label`: Target attribute label (0 or 1)
- `--output`: Output file path (default: `generated_samples.png`)
- `--no_guidance`: Generate without classifier guidance

**Output:**
- `generated_samples.png`: Grid of generated images
- `generated_samples/`: Individual sample images

## How It Works

### Training

1. **Classifier Training**: Train a CNN classifier to predict CelebA attributes
2. **Diffusion Training**: Train UNet to denoise images at various noise levels

### Generation with Classifier Guidance

1. Start with random noise
2. Iteratively denoise using the diffusion model
3. At each step, use classifier gradients to guide towards desired attribute
4. Continue until clean image is generated

The guidance equation:
```
noise_pred = noise_pred - guidance_scale * ∇log p(y|x)
```

## Examples

```bash
# Train on "Smiling" attribute
python train.py --attribute Smiling --diffusion_epochs 50

# Generate smiling faces
python generate.py --target_label 1 --guidance_scale 3.0

# Train on "Young" attribute  
python train.py --attribute Young --diffusion_epochs 50

# Generate young faces
python generate.py --target_label 1 --guidance_scale 2.0
```

## Project Structure

```
GuidedEmotion/
├── diffusion_model.py    # Diffusion model implementation
├── classifier.py         # Classifier for guidance
├── dataset.py           # CelebA dataset loader (50% of data)
├── train.py             # Training script
├── generate.py          # Generation script
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Memory Optimization

The implementation uses only **50% of the CelebA dataset** to reduce memory requirements:
- Training set: ~81,000 images (from ~162,000)
- Test set: ~10,000 images (from ~20,000)

This makes the model trainable on systems with limited GPU memory while maintaining good performance.

## Technical Details

### Diffusion Process

- **Forward Process**: Gradually add Gaussian noise to images
- **Reverse Process**: Learn to denoise images step by step
- **Training**: MSE loss between predicted and actual noise

### Classifier Guidance

The classifier provides gradients that guide the generation:
- Trained separately on the same dataset
- Provides signal about attribute presence
- Integrated into the reverse diffusion process

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- CUDA-capable GPU (recommended)
- ~8GB GPU memory for training

## Notes

- First run will download CelebA dataset (~1.4GB)
- Training time depends on GPU (RTX 3090: ~2-3 hours for 50 epochs)
- Image size is 64x64 for computational efficiency
- Adjust batch size based on available GPU memory

## Citation

If you use this code, please cite:

```bibtex
@software{GuidedEmotion,
  author = {ArthurP28},
  title = {GuidedEmotion: Diffusion-based Counterfactual Generation},
  year = {2024},
  url = {https://github.com/ArthurP28/GuidedEmotion}
}
```

## License

MIT License