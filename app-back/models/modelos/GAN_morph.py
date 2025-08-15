"""
GAN para morphing entre dos posiciones de la mano (letras).
Este esqueleto define la estructura básica de una GAN para generar imágenes intermedias entre dos letras.
"""

import torch
import torch.nn as nn

# Generador: recibe imagen inicial, imagen final y un parámetro t (0 a 1) para generar la imagen intermedia
def make_generator(img_size):
    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            # Entrada: img_start, img_end, t (todos como canales)
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 1, kernel_size=3, padding=1),
                nn.Tanh()
            )
        def forward(self, img_start, img_end, t):
            # img_start, img_end: (batch, 1, img_size, img_size)
            # t: (batch, 1)
            # Expandir t a imagen
            batch_size = img_start.size(0)
            t_img = t.view(batch_size, 1, 1, 1).expand(-1, 1, img_start.size(2), img_start.size(3))
            x = torch.cat([img_start, img_end, t_img], dim=1)  # (batch, 3, img_size, img_size)
            out = self.conv(x)
            return out.squeeze(1)  # (batch, img_size, img_size)
    return Generator()

# Discriminador: recibe imagen intermedia y las imágenes de inicio/fin, predice si la transición es real o generada
def make_discriminator(img_size):
    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, padding=2),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 1, kernel_size=3, padding=1),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(img_size*img_size, 1),
                nn.Sigmoid()
            )
        def forward(self, img_start, img_end, img_mid):
            # img_start, img_end, img_mid: (batch, 1, img_size, img_size) o (batch, img_size, img_size)
            # Aseguramos que tengan canal
            if img_start.dim() == 3:
                img_start = img_start.unsqueeze(1)
            if img_end.dim() == 3:
                img_end = img_end.unsqueeze(1)
            if img_mid.dim() == 3:
                img_mid = img_mid.unsqueeze(1)
            x = torch.cat([img_start, img_end, img_mid], dim=1)  # (batch, 3, img_size, img_size)
            out = self.conv(x)
            out = self.fc(out)
            return out
    return Discriminator()

# Ejemplo de uso:
# gen = make_generator(img_size=64)
# disc = make_discriminator(img_size=64)
