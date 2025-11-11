"""
Minimal Diffusion Model for Counterfactual Generation
Simple UNet-based architecture for diffusion process
"""

import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Simple sinusoidal position embeddings for timesteps"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SimpleBlock(nn.Module):
    """Simple convolutional block with time embedding"""
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        
        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        
        # Add time embedding
        time_emb = self.time_mlp(t)
        h = h + time_emb[:, :, None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        return h + self.residual(x)


class MinimalUNet(nn.Module):
    """Minimal UNet architecture for diffusion model"""
    def __init__(self, in_channels=3, out_channels=3, time_dim=128, base_channels=64):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim)
        )
        
        # Encoder (downsampling)
        self.down1 = SimpleBlock(in_channels, base_channels, time_dim)
        self.down2 = SimpleBlock(base_channels, base_channels * 2, time_dim)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = SimpleBlock(base_channels * 2, base_channels * 2, time_dim)
        
        # Decoder (upsampling)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 2, stride=2)
        self.up_block1 = SimpleBlock(base_channels * 4, base_channels, time_dim)
        
        self.up2 = nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2)
        self.up_block2 = SimpleBlock(base_channels * 2, base_channels, time_dim)
        
        # Output
        self.out = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Encoder
        d1 = self.down1(x, t_emb)
        d1_pool = self.pool(d1)
        
        d2 = self.down2(d1_pool, t_emb)
        d2_pool = self.pool(d2)
        
        # Bottleneck
        b = self.bottleneck(d2_pool, t_emb)
        
        # Decoder with skip connections
        u1 = self.up1(b)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.up_block1(u1, t_emb)
        
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.up_block2(u2, t_emb)
        
        return self.out(u2)


class DiffusionModel:
    """Simple diffusion model with linear noise schedule"""
    def __init__(self, model, timesteps=1000, device='cuda'):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        
        # Linear schedule for beta
        self.betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion process: add noise to images"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, x_t, t, classifier=None, guidance_scale=1.0, target_label=None):
        """Reverse diffusion process: denoise images with optional classifier guidance"""
        batch_size = x_t.shape[0]
        t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
        
        # Predict noise
        if classifier is not None and guidance_scale > 0 and target_label is not None:
            # Classifier guidance
            x_t.requires_grad_(True)
            predicted_noise = self.model(x_t, t_batch)
            
            # Get classifier gradient
            logits = classifier(x_t)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), target_label]
            grad = torch.autograd.grad(selected.sum(), x_t)[0]
            
            # Add gradient guidance
            predicted_noise = predicted_noise - guidance_scale * grad * self.sqrt_one_minus_alphas_cumprod[t]
            x_t.requires_grad_(False)
        else:
            predicted_noise = self.model(x_t, t_batch)
        
        # Compute previous sample
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]
        
        model_mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
        )
        
        if t > 0:
            noise = torch.randn_like(x_t)
            variance = beta_t
            return model_mean + torch.sqrt(variance) * noise
        else:
            return model_mean

    @torch.no_grad()
    def sample(self, shape, classifier=None, guidance_scale=1.0, target_label=None):
        """Generate samples using reverse diffusion"""
        x = torch.randn(shape).to(self.device)
        
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(x, t, classifier, guidance_scale, target_label)
        
        return x
