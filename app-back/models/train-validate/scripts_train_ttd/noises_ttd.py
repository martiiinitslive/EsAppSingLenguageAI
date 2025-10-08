import torch
import torch.nn.functional as F
from config_ttd import (
    NOISE_STD_GAUSSIAN,
    NOISE_AMOUNT_SALT_PEPPER,
    NOISE_LOW_UNIFORM,
    NOISE_HIGH_UNIFORM,
    NOISE_STD_SPECKLE,
    NOISE_KERNEL_BLUR
)

def gaussian_noise(img, std=NOISE_STD_GAUSSIAN):
    return img + torch.randn_like(img) * std

def salt_and_pepper_noise(img, amount=NOISE_AMOUNT_SALT_PEPPER):
    noisy = img.clone()
    num_salt = int(amount * img.numel())
    coords = [torch.randint(0, i, (num_salt,)) for i in img.shape]
    noisy[tuple(coords)] = 1
    num_pepper = int(amount * img.numel())
    coords = [torch.randint(0, i, (num_pepper,)) for i in img.shape]
    noisy[tuple(coords)] = 0
    return noisy

def uniform_noise(img, low=NOISE_LOW_UNIFORM, high=NOISE_HIGH_UNIFORM):
    return img + (high - low) * torch.rand_like(img) + low

def speckle_noise(img, std=NOISE_STD_SPECKLE):
    return img + img * torch.randn_like(img) * std

def blur_noise(img, kernel_size=NOISE_KERNEL_BLUR):
    # Aplica un desenfoque simple usando un filtro de media
    if kernel_size < 2:
        return img
    channels = img.shape[1]
    kernel = torch.ones((channels, 1, kernel_size, kernel_size), device=img.device) / (kernel_size * kernel_size)
    img = F.pad(img, [kernel_size//2]*4, mode='reflect')
    return F.conv2d(img, kernel, groups=channels)

def latent_noise(batch_size, noise_dim, device):
    """Genera un vector de ruido latente para el generador."""
    return torch.randn(batch_size, noise_dim, device=device)