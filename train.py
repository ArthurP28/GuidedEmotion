"""
Training Script for Diffusion Model with Classifier Guidance
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import argparse

from diffusion_model import MinimalUNet, DiffusionModel
from classifier import SimpleClassifier
from dataset import CelebADataset


def train_classifier(classifier, train_loader, test_loader, device, epochs=10, lr=1e-3):
    """Train classifier on CelebA attributes"""
    print("\n" + "="*50)
    print("Training Classifier")
    print("="*50)
    
    classifier = classifier.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        # Training
        classifier.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Classifier Epoch {epoch+1}/{epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
        
        train_acc = 100. * correct / total
        
        # Validation
        classifier.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = classifier(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * test_correct / test_total
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(classifier.state_dict(), 'classifier_best.pth')
            print(f'Saved best classifier with accuracy: {best_acc:.2f}%')
    
    return classifier


def train_diffusion(diffusion, train_loader, device, epochs=50, lr=1e-4, save_interval=10):
    """Train diffusion model"""
    print("\n" + "="*50)
    print("Training Diffusion Model")
    print("="*50)
    
    model = diffusion.model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Diffusion Epoch {epoch+1}/{epochs}')
        for images, _ in pbar:
            images = images.to(device)
            batch_size = images.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()
            
            # Sample noise
            noise = torch.randn_like(images)
            
            # Forward diffusion (add noise)
            x_noisy = diffusion.q_sample(images, t, noise)
            
            # Predict noise
            predicted_noise = model(x_noisy, t)
            
            # Compute loss
            loss = criterion(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}: Average Loss: {avg_loss:.6f}')
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), f'diffusion_epoch_{epoch+1}.pth')
            print(f'Saved checkpoint at epoch {epoch+1}')
    
    # Save final model
    torch.save(model.state_dict(), 'diffusion_final.pth')
    print('Saved final diffusion model')
    
    return diffusion


def main():
    parser = argparse.ArgumentParser(description='Train Diffusion Model with Classifier Guidance')
    parser.add_argument('--data_root', type=str, default='./data', help='Root directory for dataset')
    parser.add_argument('--image_size', type=int, default=64, help='Image size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--attribute', type=str, default='Smiling', help='CelebA attribute (e.g., Smiling, Young, Male)')
    parser.add_argument('--classifier_epochs', type=int, default=10, help='Epochs for classifier training')
    parser.add_argument('--diffusion_epochs', type=int, default=50, help='Epochs for diffusion training')
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--lr_classifier', type=float, default=1e-3, help='Learning rate for classifier')
    parser.add_argument('--lr_diffusion', type=float, default=1e-4, help='Learning rate for diffusion')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Check device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load dataset (only 50% of data)
    print("\n" + "="*50)
    print("Loading CelebA Dataset (50%)")
    print("="*50)
    dataset_loader = CelebADataset(
        root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        attribute=args.attribute
    )
    train_loader, test_loader = dataset_loader.get_dataloaders()
    
    # Initialize models
    print("\n" + "="*50)
    print("Initializing Models")
    print("="*50)
    
    # Classifier
    classifier = SimpleClassifier(num_classes=2, in_channels=3)
    print(f'Classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}')
    
    # Diffusion model
    unet = MinimalUNet(in_channels=3, out_channels=3, time_dim=128, base_channels=64)
    print(f'UNet parameters: {sum(p.numel() for p in unet.parameters()):,}')
    diffusion = DiffusionModel(unet, timesteps=args.timesteps, device=device)
    
    # Train classifier
    classifier = train_classifier(
        classifier, train_loader, test_loader, device,
        epochs=args.classifier_epochs, lr=args.lr_classifier
    )
    
    # Train diffusion model
    diffusion = train_diffusion(
        diffusion, train_loader, device,
        epochs=args.diffusion_epochs, lr=args.lr_diffusion
    )
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print("Saved models:")
    print("  - classifier_best.pth")
    print("  - diffusion_final.pth")
    print("\nUse generate.py to generate counterfactual samples")


if __name__ == '__main__':
    main()
