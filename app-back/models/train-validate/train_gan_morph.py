"""
Script de entrenamiento y validación para la GAN de morphing entre letras.
Debes adaptar la carga de datos a tu dataset de transiciones (pares de imágenes y parámetro t).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
from ..gan_morph import make_generator, make_discriminator

# Configuración
DATASET_DIR = '../../data/morphing/'  # Cambia esta ruta a tu dataset
IMG_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 10

# Dataset personalizado para morphing
class MorphDataset(Dataset):
    def __init__(self, data_dir, img_size):
        self.data = []
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        # Espera archivos: letraA_letraB_t.png
        for fname in os.listdir(data_dir):
            if fname.endswith('.png'):
                parts = fname.split('_')
                if len(parts) == 3:
                    img_start = os.path.join(data_dir, f'{parts[0]}.png')
                    img_end = os.path.join(data_dir, f'{parts[1]}.png')
                    t = float(parts[2].replace('.png',''))
                    img_mid = os.path.join(data_dir, fname)
                    if os.path.exists(img_start) and os.path.exists(img_end):
                        self.data.append((img_start, img_end, img_mid, t))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_start, img_end, img_mid, t = self.data[idx]
        start = self.transform(Image.open(img_start))
        end = self.transform(Image.open(img_end))
        mid = self.transform(Image.open(img_mid))
        t_tensor = torch.tensor([[t]], dtype=torch.float32)
        return start, end, mid, t_tensor

# Cargar datos y dividir en entrenamiento y validación (80/20)
full_dataset = MorphDataset(DATASET_DIR, IMG_SIZE)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Modelos
generator = make_generator(IMG_SIZE)
discriminator = make_discriminator(IMG_SIZE)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

# Entrenamiento y validación
train_losses_G = []
train_losses_D = []
val_losses_G = []
val_losses_D = []
for epoch in range(EPOCHS):
    generator.train()
    discriminator.train()
    running_loss_G = 0.0
    running_loss_D = 0.0
    for start, end, mid, t in train_loader:
        batch_size = start.size(0)
        real_label = torch.ones(batch_size, 1)
        fake_label = torch.zeros(batch_size, 1)

        # Generador
        fake_mid = generator(start, end, t)
        output = discriminator(start, end, fake_mid)
        loss_G = criterion(output, real_label)
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
        running_loss_G += loss_G.item()

        # Discriminador
        output_real = discriminator(start, end, mid)
        loss_D_real = criterion(output_real, real_label)
        output_fake = discriminator(start, end, fake_mid.detach())
        loss_D_fake = criterion(output_fake, fake_label)
        loss_D = (loss_D_real + loss_D_fake) / 2
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()
        running_loss_D += loss_D.item()

    avg_train_loss_G = running_loss_G / len(train_loader)
    avg_train_loss_D = running_loss_D / len(train_loader)
    train_losses_G.append(avg_train_loss_G)
    train_losses_D.append(avg_train_loss_D)
    print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss G: {avg_train_loss_G:.4f}, D: {avg_train_loss_D:.4f}')

    # Validación
    generator.eval()
    discriminator.eval()
    val_loss_G = 0.0
    val_loss_D = 0.0
    with torch.no_grad():
        for start, end, mid, t in val_loader:
            batch_size = start.size(0)
            real_label = torch.ones(batch_size, 1)
            fake_label = torch.zeros(batch_size, 1)
            fake_mid = generator(start, end, t)
            output = discriminator(start, end, fake_mid)
            loss_G = criterion(output, real_label)
            val_loss_G += loss_G.item()
            output_real = discriminator(start, end, mid)
            loss_D_real = criterion(output_real, real_label)
            output_fake = discriminator(start, end, fake_mid)
            loss_D_fake = criterion(output_fake, fake_label)
            loss_D = (loss_D_real + loss_D_fake) / 2
            val_loss_D += loss_D.item()
    avg_val_loss_G = val_loss_G / len(val_loader) if len(val_loader) > 0 else 0
    avg_val_loss_D = val_loss_D / len(val_loader) if len(val_loader) > 0 else 0
    val_losses_G.append(avg_val_loss_G)
    val_losses_D.append(avg_val_loss_D)
    print(f'Epoch {epoch+1}/{EPOCHS}, Val Loss G: {avg_val_loss_G:.4f}, D: {avg_val_loss_D:.4f}')

# Guardar modelos
torch.save(generator.state_dict(), 'gan_morph_generator.pth')
torch.save(discriminator.state_dict(), 'gan_morph_discriminator.pth')
print('Entrenamiento y validación finalizados y modelos guardados.')

# ===================
# SEGUIMIENTO DEL ENTRENAMIENTO (puedes comentar o quitar este bloque)
# ===================
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(train_losses_G, label='Train Loss Generator')
plt.plot(train_losses_D, label='Train Loss Discriminator')
plt.plot(val_losses_G, label='Val Loss Generator')
plt.plot(val_losses_D, label='Val Loss Discriminator')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Evolución de la pérdida GAN Morph')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
# ===================
