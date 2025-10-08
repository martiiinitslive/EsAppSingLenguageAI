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
NOISE_DIM = config_ttd.NOISE_DIM

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.tensor(0.1))  # Inicialización mejorada
        self.norm = nn.BatchNorm2d(in_dim)            # Normalización opcional

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        out = self.norm(out)  # Normalización opcional
        return out
class TextToDictaModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM, img_size=IMG_SIZE, init_channels=INIT_CHANNELS, init_map_size=INIT_MAP_SIZE, noise_dim=NOISE_DIM):
        super(TextToDictaModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.noise_dim = noise_dim
        self.img_size = img_size
        self.init_map_size = init_map_size
        self.init_channels = init_channels
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
            nn.Linear(embedding_dim + noise_dim, self.init_channels * self.init_map_size * self.init_map_size),
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

    def forward(self, x, noise=None):
        emb = self.embedding(x)
        if emb.dim() > 2:
            emb = emb.squeeze(1)
        if noise is None:
            noise = torch.randn(emb.size(0), self.noise_dim, device=emb.device)
        emb_noise = torch.cat([emb, noise], dim=1)
        out = self.fc(emb_noise)
        out = out.view(-1, self.init_channels, self.init_map_size, self.init_map_size)

        # Encoder
        e1 = self.enc1(out)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Self-Attention solo tras el encoder
        e3_attn = self.attn_enc(e3)

        # Decoder y skip connections
        d1 = self.dec1(e3_attn)
        e3_up = nn.functional.interpolate(e3_attn, size=d1.shape[2:], mode='bilinear', align_corners=False)
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
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(self.leaky_relu_slope)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Dropout2d(self.dropout_prob * 0.25)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Dropout2d(self.dropout_prob * 0.5)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Dropout2d(self.dropout_prob * 0.75)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Dropout2d(self.dropout_prob)
        )

        # Calcula el tamaño de salida después de las convoluciones
        def conv_out_size(size, kernel_size=4, stride=2, padding=1, n_layers=5):
            for _ in range(n_layers):
                size = (size + 2 * padding - kernel_size) // stride + 1
            return size

        final_spatial = conv_out_size(img_size)
        self.final = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * final_spatial * final_spatial, 1),
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
#    - Este vector se reestructura a un mapa de características inicial de tamaño INIT_MAP_SIZE x INIT_MAP_SIZE x INIT_CHANNELS.
#    - Se aplica una activación LeakyReLU y Dropout.
#
# 3. Encoder convolucional
#    - Tres bloques convolucionales (Conv2d + BatchNorm + LeakyReLU + Dropout) extraen características profundas.
#    - Se usan canales configurables (512, 256, 128).
#    - Se aplica Self-Attention tras el último bloque encoder.
#
# 4. Decoder con skip connections
#    - Cinco bloques deconvolucionales (ConvTranspose2d + BatchNorm + LeakyReLU + Dropout).
#    - En cada bloque se concatenan las características del encoder correspondientes (skip connections).
#    - Se aplica Self-Attention en el último bloque del decoder si la resolución es baja.
#
# 5. Capa Final
#    - Una convolución normal (nn.Conv2d) reduce los canales a 3 (imagen RGB), manteniendo el tamaño final.
#    - Se aplica una activación Tanh para limitar la salida al rango [-1, 1], compatible con la normalización de tus datos.
#
# Resumen del flujo:
# Índice de letra → Embedding → Mapa de características inicial → Encoder convolucional + Self-Attention → Decoder con skip connections + Self-Attention → Imagen final normalizada.
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
# [Linear (FC) → LeakyReLU → Dropout]
#       │
#       ▼
# [Reshape a mapa de características inicial: INIT_MAP_SIZE x INIT_MAP_SIZE x INIT_CHANNELS]
#       │
#       ▼
# [Encoder: Conv2d → BatchNorm → LeakyReLU → Dropout] × 3
#       │
#       ▼
# [Self-Attention (encoder)]
#       │
#       ▼
# [Decoder: ConvTranspose2d → BatchNorm → LeakyReLU → Dropout + Skip Connections] × 5
#       │
#       ▼
# [Self-Attention (decoder, solo si resolución baja)]
#       │
#       ▼
# [Conv2d: canales finales → 3 (RGB)]
#       │
#       ▼
# [Tanh]
#       │
#       ▼
# [Imagen generada (normalizada en [-1, 1])]
