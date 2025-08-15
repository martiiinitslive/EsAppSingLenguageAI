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
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.resize = resize
        self.criterion = nn.MSELoss()

    def forward(self, input, target):
        # Convierte a 3 canales y normaliza para VGG
        def preprocess(img):
            if img.size(1) == 1:
                img = img.repeat(1, 3, 1, 1)
            if self.resize:
                img = nn.functional.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
            # Normalización VGG
            mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1,3,1,1)
            std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1,3,1,1)
            img = (img + 1) / 2  # De [-1,1] a [0,1]
            img = (img - mean) / std
            return img
        input_vgg = preprocess(input)
        target_vgg = preprocess(target)
        feat_input = self.vgg(input_vgg)
        feat_target = self.vgg(target_vgg)
        return self.criterion(feat_input, feat_target)
