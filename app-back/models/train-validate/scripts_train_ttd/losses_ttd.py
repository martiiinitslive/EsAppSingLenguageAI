# Clase para Perceptual Loss usando VGG16
# Compara características visuales entre imágenes generadas y reales
# Ayuda a que el modelo aprenda detalles visuales más allá del error pixel a pixel.
import torch
import torch.nn as nn
from torchvision import models

class PerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        """
        Args:
            resize (bool): Si True, redimensiona a 224x224 para VGG
        """
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16]
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.resize = resize
        self.criterion = nn.MSELoss()
        # Precompute mean and std for VGG normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def preprocess(self, img):
        # img: [B, C, H, W], C=1 or 3, values in [-1, 1]
        if img.size(1) == 1:
            img = img.repeat(1, 3, 1, 1)
        if self.resize:
            img = nn.functional.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
        img = (img + 1) / 2  # [-1,1] -> [0,1]
        img = (img - self.mean) / self.std
        return img

    def forward(self, input, target):
        """
        Args:
            input: Tensor [B, C, H, W]
            target: Tensor [B, C, H, W]
        Returns:
            MSE loss between VGG features
        """
        with torch.no_grad():
            input_vgg = self.preprocess(input)
            target_vgg = self.preprocess(target)
        feat_input = self.vgg(input_vgg)
        feat_target = self.vgg(target_vgg)
        return self.criterion(feat_input, feat_target)

# --- Otras pérdidas como clases o funciones ---
class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.criterion = nn.BCELoss()
    def forward(self, input, target):
        return self.criterion(input, target)

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.criterion = nn.MSELoss()
    def forward(self, input, target):
        return self.criterion(input, target)
