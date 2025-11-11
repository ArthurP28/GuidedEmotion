"""
Quick Demo Script for Testing the Implementation
This script provides a quick way to verify the models work correctly
"""

import torch
from diffusion_model import MinimalUNet, DiffusionModel
from classifier import SimpleClassifier


def test_models():
    """Test that models can be instantiated and perform forward passes"""
    
    print("="*60)
    print("Testing GuidedEmotion Implementation")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Test configuration
    batch_size = 4
    image_size = 64
    channels = 3
    
    print("\n" + "-"*60)
    print("1. Testing Classifier")
    print("-"*60)
    
    # Create classifier
    classifier = SimpleClassifier(num_classes=2, in_channels=3)
    classifier = classifier.to(device)
    
    # Test forward pass
    test_images = torch.randn(batch_size, channels, image_size, image_size).to(device)
    with torch.no_grad():
        logits = classifier(test_images)
    
    print(f"✓ Classifier initialized")
    print(f"  Parameters: {sum(p.numel() for p in classifier.parameters()):,}")
    print(f"  Input shape: {test_images.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected output shape: ({batch_size}, 2)")
    assert logits.shape == (batch_size, 2), "Classifier output shape mismatch!"
    print("✓ Classifier forward pass successful")
    
    print("\n" + "-"*60)
    print("2. Testing Diffusion Model (UNet)")
    print("-"*60)
    
    # Create UNet
    unet = MinimalUNet(in_channels=3, out_channels=3, time_dim=128, base_channels=64)
    unet = unet.to(device)
    
    # Test forward pass
    test_timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    with torch.no_grad():
        noise_pred = unet(test_images, test_timesteps)
    
    print(f"✓ UNet initialized")
    print(f"  Parameters: {sum(p.numel() for p in unet.parameters()):,}")
    print(f"  Input shape: {test_images.shape}")
    print(f"  Timestep shape: {test_timesteps.shape}")
    print(f"  Output shape: {noise_pred.shape}")
    print(f"  Expected output shape: {test_images.shape}")
    assert noise_pred.shape == test_images.shape, "UNet output shape mismatch!"
    print("✓ UNet forward pass successful")
    
    print("\n" + "-"*60)
    print("3. Testing Diffusion Process")
    print("-"*60)
    
    # Create diffusion model
    diffusion = DiffusionModel(unet, timesteps=1000, device=device)
    
    # Test forward diffusion (add noise)
    t = torch.tensor([500]).to(device)
    noisy_images = diffusion.q_sample(test_images[:1], t)
    
    print(f"✓ Diffusion model initialized")
    print(f"  Timesteps: {diffusion.timesteps}")
    print(f"  Original image range: [{test_images.min():.2f}, {test_images.max():.2f}]")
    print(f"  Noisy image range: [{noisy_images.min():.2f}, {noisy_images.max():.2f}]")
    print("✓ Forward diffusion (q_sample) successful")
    
    # Test reverse diffusion step
    with torch.no_grad():
        denoised = diffusion.p_sample(noisy_images, t.item(), classifier=None)
    
    print(f"  Denoised image shape: {denoised.shape}")
    assert denoised.shape == noisy_images.shape, "Denoising output shape mismatch!"
    print("✓ Reverse diffusion (p_sample) successful")
    
    print("\n" + "-"*60)
    print("4. Testing Classifier-Guided Sampling")
    print("-"*60)
    
    # Test guided sampling (just a few steps for speed)
    target_label = torch.tensor([1]).to(device)
    
    # Create a small diffusion for quick testing
    quick_diffusion = DiffusionModel(unet, timesteps=10, device=device)
    
    with torch.no_grad():
        generated = quick_diffusion.sample(
            shape=(1, channels, image_size, image_size),
            classifier=classifier,
            guidance_scale=1.0,
            target_label=target_label
        )
    
    print(f"✓ Generated sample shape: {generated.shape}")
    print(f"  Generated image range: [{generated.min():.2f}, {generated.max():.2f}]")
    assert generated.shape == (1, channels, image_size, image_size), "Generated sample shape mismatch!"
    print("✓ Classifier-guided sampling successful")
    
    print("\n" + "="*60)
    print("All Tests Passed! ✓")
    print("="*60)
    print("\nThe implementation is working correctly.")
    print("\nNext steps:")
    print("  1. Run 'python train.py' to train the models on CelebA")
    print("  2. Run 'python generate.py' to generate counterfactual samples")
    print("\nNote: Training requires downloading CelebA dataset (~1.4GB)")


if __name__ == '__main__':
    try:
        test_models()
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
