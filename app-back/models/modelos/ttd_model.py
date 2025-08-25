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
INIT_MAP_SIZE = config_ttd.INIT_MAP_SIZE
INIT_CHANNELS = config_ttd.INIT_CHANNELS
LEAKY_RELU_SLOPE = config_ttd.LEAKY_RELU_SLOPE
DROPOUT_ENCODER = config_ttd.DROPOUT_ENCODER
DROPOUT_DECODER = config_ttd.DROPOUT_DECODER
DROPOUT_DISC = config_ttd.DROPOUT_DISC

# Definición de la clase del modelo generador
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        # Solo aplica atención si el tamaño espacial es pequeño
        if width * height > 1024:  # Por ejemplo, no aplicar si >32x32
            return x
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

class TextToDictaModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM, img_size=IMG_SIZE):
        super(TextToDictaModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.img_size = img_size
        self.init_map_size = INIT_MAP_SIZE
        self.init_channels = INIT_CHANNELS
        self.leaky_relu_slope = LEAKY_RELU_SLOPE

        # Encoder channels
        enc1_channels = 512
        enc2_channels = 256
        enc3_channels = 128

        # Decoder channels (computed from skip connections)
        dec1_in = enc3_channels
        dec1_out = enc3_channels
        dec2_in = dec1_out + enc3_channels
        dec2_out = 64
        dec3_in = dec2_out + enc2_channels
        dec3_out = 32
        dec4_in = dec3_out + enc1_channels
        dec4_out = 16
        dec5_in = dec4_out + self.init_channels  # <-- este depende de INIT_CHANNELS
        dec5_out = 8

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, self.init_channels * self.init_map_size * self.init_map_size),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Dropout(DROPOUT_ENCODER)
        )

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(self.init_channels, enc1_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(enc1_channels),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Dropout2d(DROPOUT_ENCODER)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(enc1_channels, enc2_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(enc2_channels),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Dropout2d(DROPOUT_ENCODER)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(enc2_channels, enc3_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(enc3_channels),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Dropout2d(DROPOUT_ENCODER)
        )

        self.attn_enc = SelfAttention(enc3_channels)
        self.attn_dec = SelfAttention(dec5_out)  # <-- este depende de dec5_out (8 canales)

        # Decoder (channels now depend on config)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(dec1_in, dec1_out, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dec1_out),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Dropout2d(DROPOUT_DECODER)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(dec2_in, dec2_out, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dec2_out),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Dropout2d(DROPOUT_DECODER)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(dec3_in, dec3_out, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dec3_out),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Dropout2d(DROPOUT_DECODER)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(dec4_in, dec4_out, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dec4_out),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Dropout2d(DROPOUT_DECODER)
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(dec5_in, dec5_out, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dec5_out),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Dropout2d(DROPOUT_DECODER)
        )
        self.final_conv = nn.Conv2d(dec5_out, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        emb = self.embedding(x)
        if emb.dim() > 2:
            emb = emb.squeeze(1)
        out = self.fc(emb)
        out = out.view(-1, self.init_channels, self.init_map_size, self.init_map_size)

        # Encoder
        e1 = self.enc1(out)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Self-Attention solo tras el encoder
        e3_attn = self.attn_enc(e3)

        # Decoder y skip connections
        d1 = self.dec1(e3_attn)
        e3_up = nn.functional.interpolate(e3_attn, size=d1.shape[2:], mode='nearest')
        d1_cat = torch.cat([d1, e3_up], dim=1)

        d2 = self.dec2(d1_cat)
        e2_up = nn.functional.interpolate(e2, size=d2.shape[2:], mode='nearest')
        d2_cat = torch.cat([d2, e2_up], dim=1)

        d3 = self.dec3(d2_cat)
        e1_up = nn.functional.interpolate(e1, size=d3.shape[2:], mode='nearest')
        d3_cat = torch.cat([d3, e1_up], dim=1)

        d4 = self.dec4(d3_cat)
        out_up = nn.functional.interpolate(out, size=d4.shape[2:], mode='nearest')
        d4_cat = torch.cat([d4, out_up], dim=1)

        d5 = self.dec5(d4_cat)
        # Atención en el decoder solo si la resolución es baja
        if d5.shape[2] * d5.shape[3] <= 4096:  # 64x64 o menos
            d5_attn = self.attn_dec(d5)
        else:
            d5_attn = d5
        imgs = self.final_conv(d5_attn)
        imgs = self.tanh(imgs)
        return imgs
class DictaDiscriminator(nn.Module):
    """
    Discriminador para GAN: recibe una imagen y predice si es real o generada.
    """
    def __init__(self, img_size=IMG_SIZE, in_channels=3):
        super(DictaDiscriminator, self).__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.dropout_prob = DROPOUT_DISC
        self.leaky_relu_slope = LEAKY_RELU_SLOPE

        # Bloques convolucionales descendentes
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),   # 256x256 -> 128x128
            nn.BatchNorm2d(32),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Dropout2d(self.dropout_prob)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),            # 128x128 -> 64x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Dropout2d(self.dropout_prob)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),           # 64x64 -> 32x32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Dropout2d(self.dropout_prob)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),          # 32x32 -> 16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Dropout2d(self.dropout_prob)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),          # 16x16 -> 8x8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Dropout2d(self.dropout_prob)
        )

        # Capa final: reduce a un solo valor (probabilidad)
        self.final = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        out = self.final(x)
        return out

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
