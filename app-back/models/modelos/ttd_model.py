"""
Modelo generador: convierte texto en imágenes de dictadología.
"""

# Importa PyTorch y su módulo de redes neuronales
import torch
import torch.nn as nn
import sys, os

import importlib.util

# Define config_path as the directory containing this file
config_path = os.path.dirname(os.path.abspath(__file__))

config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train-validate', 'scripts_train_ttd'))
config_ttd_path = os.path.join(config_path, 'config_ttd.py')
spec = importlib.util.spec_from_file_location("config_ttd", config_ttd_path)
config_ttd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_ttd)
EMBEDDING_DIM = config_ttd.EMBEDDING_DIM
IMG_SIZE = config_ttd.IMG_SIZE

# Definición de la clase del modelo generador
class TextToDictaModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM, img_size=IMG_SIZE):
        super(TextToDictaModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.img_size = img_size
        self.init_map_size = 8  # Tamaño inicial del mapa de características (8x8)
        self.init_channels = 256  # Número de canales iniciales para el mapa de características

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, self.init_channels * self.init_map_size * self.init_map_size),
            nn.ReLU()
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.init_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # x: tensor de índices de letras (batch,)
        # Paso 1: convertir el índice de la letra en su embedding
        emb = self.embedding(x)  # (batch, embedding_dim)
        if emb.dim() > 2:
            emb = emb.squeeze(1)
        # Paso 2: proyectar el embedding a un vector largo y reestructurarlo a un mapa de características inicial
        out = self.fc(emb)  # (batch, init_channels * init_map_size * init_map_size)
        out = out.view(-1, self.init_channels, self.init_map_size, self.init_map_size)  # (batch, channels, 8, 8)
        # Paso 3: expandir el mapa de características con bloques deconvolucionales hasta obtener la imagen final
        imgs = self.deconv(out)  # (batch, 3, 128, 128)
        # Devuelve las imágenes generadas, normalizadas en [-1, 1]
        return imgs

## Ejemplo de uso:
# model = TextToDictaModel(vocab_size=27, embedding_dim=32, img_size=128)  # Crea el modelo
# texto = torch.tensor([[1, 2, 3]])  # índices de letras

# -------------------------------------------------------------
# Resumen de la arquitectura:
#
# 1. Embedding Layer
#    - Convierte el índice de la letra (por ejemplo, 'd') en un vector denso de tamaño embedding_dim (por defecto 128).
#    - Permite que el modelo aprenda una representación numérica para cada letra.
#
# 2. Proyección Inicial (Fully Connected)
#    - Una capa lineal (nn.Linear) transforma el embedding en un vector largo.
#    - Este vector se reestructura a un mapa de características inicial de tamaño 8x8x256 (8x8 espacial, 256 canales).
#    - Se aplica una activación ReLU.
#
# 3. Bloques Deconvolucionales (ConvTranspose2d)
#    - Cuatro bloques de transposed convolutions expanden el mapa de características:
#      - 8x8x256 → 16x16x128 (ConvTranspose2d + BatchNorm + ReLU)
#      - 16x16x128 → 32x32x64 (ConvTranspose2d + BatchNorm + ReLU)
#      - 32x32x64 → 64x64x32 (ConvTranspose2d + BatchNorm + ReLU)
#      - 64x64x32 → 128x128x16 (ConvTranspose2d + BatchNorm + ReLU)
#    - Cada bloque aumenta el tamaño espacial y reduce el número de canales.
#
# 4. Capa Final
#    - Una convolución normal (nn.Conv2d) reduce los canales a 3 (imagen RGB), manteniendo el tamaño 128x128.
#    - Se aplica una activación Tanh para limitar la salida al rango [-1, 1], compatible con la normalización de tus datos.
#
# Resumen del flujo:
# Índice de letra → Embedding → Mapa de características inicial → Expansión espacial con deconvoluciones → Imagen final 128x128x3 normalizada.
# -------------------------------------------------------------
#
# Diagrama del flujo de datos:
#
# [Índice de letra]
#       │
#       ▼
# [Embedding Layer]
#       │
#       ▼
# [Linear (FC) → ReLU]
#       │
#       ▼
# [Reshape a mapa de características inicial: 8x8x256]
#       │
#       ▼
# [ConvTranspose2d: 8x8x256 → 16x16x128] → [BatchNorm] → [ReLU]
#       │
#       ▼
# [ConvTranspose2d: 16x16x128 → 32x32x64] → [BatchNorm] → [ReLU]
#       │
#       ▼
# [ConvTranspose2d: 32x32x64 → 64x64x32] → [BatchNorm] → [ReLU]
#       │
#       ▼
# [ConvTranspose2d: 64x64x32 → 128x128x16] → [BatchNorm] → [ReLU]
#       │
#       ▼
# [Conv2d: 128x128x16 → 128x128x3] → [Tanh]
#       │
#       ▼
# [Imagen generada 128x128x3 (normalizada en [-1, 1])]
