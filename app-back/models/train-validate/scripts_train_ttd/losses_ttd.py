"""
losses_ttd.py

Funciones de pérdida para entrenamiento de modelos generativos:
- PerceptualLoss: compara imágenes en espacio de características profundas (VGG16)
- BCELoss: pérdida binaria para GANs
- MSELoss: pérdida cuadrática media para similitud pixel a pixel
"""

# Clase para Perceptual Loss usando VGG16
# Compara características visuales entre imágenes generadas y reales
# Ayuda a que el modelo aprenda detalles visuales más allá del error pixel a pixel.
import torch
import torch.nn as nn
from torchvision import models

class PerceptualLoss(nn.Module):
    """
    Pérdida perceptual usando VGG16.
    Compara imágenes en el espacio de características profundas.
    """
    def __init__(self, resize=True, device=None):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16]
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        if device is not None:
            vgg = vgg.to(device)
        self.vgg = vgg
        self.resize = resize
        self.criterion = nn.MSELoss()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def preprocess(self, img):
        """
        Preprocesa la imagen para VGG16.
        img: [B, C, H, W], valores en [-1, 1]
        """
        if img.dim() == 3:
            img = img.unsqueeze(0)
        if img.size(1) == 1:
            img = img.repeat(1, 3, 1, 1)
        if self.resize:
            img = nn.functional.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
        img = (img + 1) / 2
        img = (img - self.mean) / self.std
        return img

    def forward(self, input, target):
        """
        Calcula la pérdida perceptual entre input y target.
        """
        input_vgg = self.preprocess(input)
        target_vgg = self.preprocess(target)
        feat_input = self.vgg(input_vgg)
        feat_target = self.vgg(target_vgg)
        return self.criterion(feat_input, feat_target)

class BCELoss(nn.Module):
    """
    Pérdida BCE con opción de reducción.
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.criterion = nn.BCELoss(reduction=reduction)
    def forward(self, input, target):
        """
        Calcula la pérdida BCE entre input y target.
        """
        return self.criterion(input, target)

class MSELoss(nn.Module):
    """
    Pérdida MSE con opción de reducción.
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.criterion = nn.MSELoss(reduction=reduction)
    def forward(self, input, target):
        """
        Calcula la pérdida MSE entre input y target.
        """
        return self.criterion(input, target)
