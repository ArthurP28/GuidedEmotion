"""
Simple Classifier for Guidance
Used to guide diffusion generation towards specific attributes
"""

import torch
import torch.nn as nn


class SimpleClassifier(nn.Module):
    """Minimal CNN classifier for CelebA attributes"""
    def __init__(self, num_classes=2, in_channels=3):
        super().__init__()
        
        self.features = nn.Sequential(
            # Input: 64x64
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
