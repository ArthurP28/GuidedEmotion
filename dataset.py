"""
CelebA Dataset Loader
Loads only 50% of the dataset for memory efficiency
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import os


class CelebADataset:
    """CelebA dataset loader that uses only 50% of data"""
    
    def __init__(self, root='./data', image_size=64, batch_size=32, attribute='Smiling'):
        """
        Args:
            root: Root directory for dataset
            image_size: Size to resize images to
            batch_size: Batch size for dataloaders
            attribute: CelebA attribute to use for classification (e.g., 'Smiling', 'Young', 'Male')
        """
        self.root = root
        self.image_size = image_size
        self.batch_size = batch_size
        self.attribute = attribute
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
    
    def get_dataloaders(self, num_workers=2):
        """Get train and test dataloaders using only 50% of data"""
        
        # Load full dataset
        train_dataset = datasets.CelebA(
            root=self.root,
            split='train',
            target_type='attr',
            transform=self.transform,
            download=True
        )
        
        test_dataset = datasets.CelebA(
            root=self.root,
            split='test',
            target_type='attr',
            transform=self.transform,
            download=True
        )
        
        # Get attribute index
        attr_names = train_dataset.attr_names
        if self.attribute not in attr_names:
            raise ValueError(f"Attribute '{self.attribute}' not found. Available: {attr_names}")
        
        self.attr_idx = attr_names.index(self.attribute)
        print(f"Using attribute: {self.attribute} (index {self.attr_idx})")
        
        # Use only 50% of the data (memory constraint)
        train_size = len(train_dataset)
        train_indices = list(range(0, train_size // 2))  # First 50%
        train_subset = Subset(train_dataset, train_indices)
        
        test_size = len(test_dataset)
        test_indices = list(range(0, test_size // 2))  # First 50%
        test_subset = Subset(test_dataset, test_indices)
        
        print(f"Original train size: {train_size}, Using: {len(train_subset)} (50%)")
        print(f"Original test size: {test_size}, Using: {len(test_subset)} (50%)")
        
        # Create custom dataset wrapper to extract specific attribute
        train_dataset_wrapped = AttributeDataset(train_subset, self.attr_idx)
        test_dataset_wrapped = AttributeDataset(test_subset, self.attr_idx)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset_wrapped,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset_wrapped,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, test_loader


class AttributeDataset(Dataset):
    """Wrapper to extract specific attribute from CelebA"""
    def __init__(self, subset, attr_idx):
        self.subset = subset
        self.attr_idx = attr_idx
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        img, attrs = self.subset[idx]
        # Extract specific attribute (convert from {-1, 1} to {0, 1})
        label = (attrs[self.attr_idx] + 1) // 2
        return img, label
