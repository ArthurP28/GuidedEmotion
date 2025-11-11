"""
Generate Counterfactual Samples using Trained Diffusion Model with Classifier Guidance
"""

import torch
import argparse
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
import os

from diffusion_model import MinimalUNet, DiffusionModel
from classifier import SimpleClassifier


def generate_samples(diffusion, classifier, device, num_samples=16, guidance_scale=3.0, target_label=1, image_size=64):
    """
    Generate counterfactual samples with classifier guidance
    
    Args:
        diffusion: Trained diffusion model
        classifier: Trained classifier for guidance
        device: Device to use
        num_samples: Number of samples to generate
        guidance_scale: Strength of classifier guidance (higher = stronger guidance)
        target_label: Target attribute label (0 or 1)
        image_size: Size of generated images
    """
    print(f"\nGenerating {num_samples} samples with guidance scale {guidance_scale} towards label {target_label}")
    
    classifier.eval()
    diffusion.model.eval()
    
    # Generate samples
    with torch.no_grad():
        shape = (num_samples, 3, image_size, image_size)
        target_labels = torch.full((num_samples,), target_label, dtype=torch.long, device=device)
        
        samples = diffusion.sample(
            shape=shape,
            classifier=classifier,
            guidance_scale=guidance_scale,
            target_label=target_labels
        )
    
    # Denormalize from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    
    return samples


def main():
    parser = argparse.ArgumentParser(description='Generate counterfactual samples')
    parser.add_argument('--diffusion_model', type=str, default='diffusion_final.pth', help='Path to diffusion model')
    parser.add_argument('--classifier_model', type=str, default='classifier_best.pth', help='Path to classifier model')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples to generate')
    parser.add_argument('--guidance_scale', type=float, default=3.0, help='Classifier guidance scale')
    parser.add_argument('--target_label', type=int, default=1, help='Target attribute label (0 or 1)')
    parser.add_argument('--image_size', type=int, default=64, help='Image size')
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--output', type=str, default='generated_samples.png', help='Output file path')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--no_guidance', action='store_true', help='Generate without classifier guidance')
    
    args = parser.parse_args()
    
    # Check device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load models
    print("Loading models...")
    
    # Load diffusion model
    unet = MinimalUNet(in_channels=3, out_channels=3, time_dim=128, base_channels=64)
    unet.load_state_dict(torch.load(args.diffusion_model, map_location=device))
    unet = unet.to(device)
    diffusion = DiffusionModel(unet, timesteps=args.timesteps, device=device)
    print(f"Loaded diffusion model from {args.diffusion_model}")
    
    # Load classifier
    classifier = SimpleClassifier(num_classes=2, in_channels=3)
    classifier.load_state_dict(torch.load(args.classifier_model, map_location=device))
    classifier = classifier.to(device)
    print(f"Loaded classifier from {args.classifier_model}")
    
    # Generate samples
    if args.no_guidance:
        print("\nGenerating samples WITHOUT classifier guidance")
        guidance_scale = 0.0
        classifier_for_gen = None
        target_label = None
    else:
        guidance_scale = args.guidance_scale
        classifier_for_gen = classifier
        target_label = args.target_label
    
    samples = generate_samples(
        diffusion=diffusion,
        classifier=classifier_for_gen,
        device=device,
        num_samples=args.num_samples,
        guidance_scale=guidance_scale,
        target_label=target_label,
        image_size=args.image_size
    )
    
    # Save samples
    grid = make_grid(samples, nrow=4, padding=2, normalize=False)
    save_image(grid, args.output)
    print(f"\nSaved generated samples to {args.output}")
    
    # Also save individual samples
    os.makedirs('generated_samples', exist_ok=True)
    for i, sample in enumerate(samples):
        save_image(sample, f'generated_samples/sample_{i:03d}.png')
    print(f"Saved individual samples to generated_samples/")
    
    # Verify with classifier
    if not args.no_guidance:
        print("\nVerifying generated samples with classifier...")
        with torch.no_grad():
            # Renormalize to [-1, 1] for classifier
            samples_norm = samples * 2 - 1
            logits = classifier(samples_norm)
            predictions = torch.softmax(logits, dim=-1)
            predicted_labels = logits.argmax(dim=-1)
        
        target_count = (predicted_labels == target_label).sum().item()
        print(f"Samples with target label {target_label}: {target_count}/{args.num_samples} ({100*target_count/args.num_samples:.1f}%)")
        print(f"Average confidence for target label: {predictions[:, target_label].mean():.3f}")


if __name__ == '__main__':
    main()
