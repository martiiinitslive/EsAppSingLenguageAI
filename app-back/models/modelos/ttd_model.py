"""
Modelo generador: convierte texto en imágenes de dictadología.
"""

# Importa PyTorch y su módulo de redes neuronales
import torch
import torch.nn as nn

# Definición de la clase del modelo generador
class TextToDictaModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, img_size=64):
        super(TextToDictaModel, self).__init__()
        # Capa de embedding: convierte el índice de la letra en un vector denso de tamaño embedding_dim
        # Esto permite que el modelo aprenda una representación numérica para cada letra
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.img_size = img_size
        self.init_map_size = 8  # Tamaño inicial del mapa de características (8x8)
        self.init_channels = 256  # Número de canales iniciales para el mapa de características

        # Proyección inicial: transforma el embedding en un vector largo y lo reestructura a un mapa de características 8x8x128
        # Esto prepara la información para ser expandida espacialmente por las capas deconvolucionales
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, self.init_channels * self.init_map_size * self.init_map_size),
            nn.ReLU()
        )

        # Bloques deconvolucionales (ConvTranspose2d): expanden el mapa de características hasta el tamaño final de la imagen
        # Cada bloque aumenta el tamaño espacial y reduce el número de canales, añadiendo normalización y activación
        self.deconv = nn.Sequential(
            # Primer bloque: 8x8x256 -> 16x16x128
            nn.ConvTranspose2d(self.init_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Segundo bloque: 16x16x128 -> 32x32x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Tercer bloque: 32x32x64 -> 64x64x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Cuarto bloque: 64x64x32 -> 64x64x16
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Capa final: reduce a 1 canal (imagen en escala de grises), mantiene tamaño 64x64
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # x: tensor de índices de letras (batch,)
        # Paso 1: convertir el índice de la letra en su embedding
        emb = self.embedding(x)  # (batch, embedding_dim)
        # Si el embedding tiene más de 2 dimensiones (por ejemplo, batch x 1 x embedding_dim), lo aplana
        if emb.dim() > 2:
            emb = emb.squeeze(1)
        # Paso 2: proyectar el embedding a un vector largo y reestructurarlo a un mapa de características inicial
        out = self.fc(emb)  # (batch, init_channels * init_map_size * init_map_size)
        out = out.view(-1, self.init_channels, self.init_map_size, self.init_map_size)  # (batch, channels, 8, 8)
        # Paso 3: expandir el mapa de características con bloques deconvolucionales hasta obtener la imagen final
        imgs = self.deconv(out)  # (batch, 1, img_size, img_size)
        # Devuelve las imágenes generadas, normalizadas en [-1, 1]
        return imgs

# Ejemplo de uso:
# model = TextToDictaModel(vocab_size=27, embedding_dim=32, img_size=64)  # Crea el modelo
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
#    - Este vector se reestructura a un mapa de características inicial de tamaño 8x8x128 (8x8 espacial, 128 canales).
#    - Se aplica una activación ReLU.
#
# 3. Bloques Deconvolucionales (ConvTranspose2d)
#    - Tres bloques de transposed convolutions expanden el mapa de características:
#      - 8x8x128 → 16x16x64 (ConvTranspose2d + BatchNorm + ReLU)
#      - 16x16x64 → 32x32x32 (ConvTranspose2d + BatchNorm + ReLU)
#      - 32x32x32 → 64x64x16 (ConvTranspose2d + BatchNorm + ReLU)
#    - Cada bloque aumenta el tamaño espacial y reduce el número de canales.
#
# 4. Capa Final
#    - Una convolución normal (nn.Conv2d) reduce los canales a 1 (imagen en escala de grises), manteniendo el tamaño 64x64.
#    - Se aplica una activación Tanh para limitar la salida al rango [-1, 1], compatible con la normalización de tus datos.
#
# Resumen del flujo:
# Índice de letra → Embedding → Mapa de características inicial → Expansión espacial con deconvoluciones → Imagen final 64x64x1 normalizada.
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
# [Reshape a mapa de características inicial: 8x8x128]
#       │
#       ▼
# [ConvTranspose2d: 8x8x128 → 16x16x64] → [BatchNorm] → [ReLU]
#       │
#       ▼
# [ConvTranspose2d: 16x16x64 → 32x32x32] → [BatchNorm] → [ReLU]
#       │
#       ▼
# [ConvTranspose2d: 32x32x32 → 64x64x16] → [BatchNorm] → [ReLU]
#       │
#       ▼
# [Conv2d: 64x64x16 → 64x64x1] → [Tanh]
#       │
#       ▼
# [Imagen generada 64x64x1 (normalizada en [-1, 1])]
