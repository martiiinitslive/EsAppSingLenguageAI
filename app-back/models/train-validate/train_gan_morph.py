"""
Script de entrenamiento y validación para la GAN de morphing entre letras.
Debes adaptar la carga de datos a tu dataset de transiciones (pares de imágenes y parámetro t).
"""

# Añade la carpeta raíz al path para importar módulos del proyecto
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
from models.gan_morph import make_generator, make_discriminator


# Configuración de hiperparámetros y rutas
DATASET_DIR = '../../data/morphing/'  # Ruta al dataset de morphing
IMG_SIZE = 64  # Tamaño de las imágenes (ancho y alto)
BATCH_SIZE = 8  # Número de muestras por batch
EPOCHS = 10  # Número de épocas de entrenamiento

# Dataset personalizado para morphing

# Dataset personalizado para cargar pares de imágenes de morphing
class MorphDataset(Dataset):
    def __init__(self, data_dir, img_size):
        self.data = []  # Lista de tuplas (img_start, img_end, t)
        self.img_size = img_size
        # Transformaciones: escala a gris, resize y tensor
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        # Recorre cada carpeta de par de letras (A_B, B_C, ...)
        for par in os.listdir(data_dir):
            par_path = os.path.join(data_dir, par)
            if not os.path.isdir(par_path):
                continue
            start_dir = os.path.join(par_path, 'start')
            end_dir = os.path.join(par_path, 'end')
            if not os.path.exists(start_dir) or not os.path.exists(end_dir):
                continue
            # Lista de imágenes de inicio y fin
            start_imgs = [os.path.join(start_dir, f) for f in os.listdir(start_dir) if f.endswith('.png')]
            end_imgs = [os.path.join(end_dir, f) for f in os.listdir(end_dir) if f.endswith('.png')]
            # Empareja cada imagen de inicio con cada imagen de fin
            for img_start in start_imgs:
                for img_end in end_imgs:
                    t = 0.5  # Parámetro de morphing (puedes cambiarlo si tienes imágenes intermedias)
                    self.data.append((img_start, img_end, t))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Devuelve las imágenes de inicio, fin, un tensor de ceros como imagen intermedia y el parámetro t
        img_start, img_end, t = self.data[idx]
        start = self.transform(Image.open(img_start))  # Imagen inicial
        end = self.transform(Image.open(img_end))      # Imagen final
        t_tensor = torch.tensor([[t]], dtype=torch.float32)  # Parámetro de morphing
        mid = torch.zeros_like(start)  # Imagen intermedia (aquí no se usa)
        return start, end, mid, t_tensor


# Carga el dataset y lo divide en entrenamiento/validación (80/20)
full_dataset = MorphDataset(DATASET_DIR, IMG_SIZE)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Instancia los modelos generador y discriminador
generator = make_generator(IMG_SIZE)      # Genera imágenes intermedias
discriminator = make_discriminator(IMG_SIZE)  # Distingue si la imagen intermedia es real o generada

# Función de pérdida y optimizadores
criterion = nn.BCELoss()  # Pérdida binaria para la GAN
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)  # Optimizador del generador
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)  # Optimizador del discriminador


# Listas para guardar la evolución de las pérdidas
train_losses_G = []  # Pérdida del generador en entrenamiento
train_losses_D = []  # Pérdida del discriminador en entrenamiento
val_losses_G = []    # Pérdida del generador en validación
val_losses_D = []    # Pérdida del discriminador en validación



import matplotlib.pyplot as plt
# Carpeta donde se guardarán las gráficas de cada epoch
GRAPH_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'imagenes', 'train-gan-morph-epoch'))
os.makedirs(GRAPH_PATH, exist_ok=True)

# Early stopping: detener si la pérdida de validación no mejora tras N epochs
EARLY_STOPPING_PATIENCE = 5  # Número de epochs sin mejora permitidos
best_val_loss = float('inf')
epochs_no_improve = 0

# Bucle principal de entrenamiento
for epoch in range(EPOCHS):
    generator.train()
    discriminator.train()
    running_loss_G = 0.0
    running_loss_D = 0.0
    # Entrenamiento por batch
    for start, end, mid, t in train_loader:
        batch_size = start.size(0)
        real_label = torch.ones(batch_size, 1)   # Etiqueta para imágenes reales
        fake_label = torch.zeros(batch_size, 1)  # Etiqueta para imágenes generadas

        # Generador: genera imagen intermedia y calcula pérdida
        fake_mid = generator(start, end, t)
        output = discriminator(start, end, fake_mid)
        loss_G = criterion(output, real_label)
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
        running_loss_G += loss_G.item()

        # Discriminador: evalúa imágenes reales y generadas
        output_real = discriminator(start, end, mid)
        loss_D_real = criterion(output_real, real_label)
        output_fake = discriminator(start, end, fake_mid.detach())
        loss_D_fake = criterion(output_fake, fake_label)
        loss_D = (loss_D_real + loss_D_fake) / 2
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()
        running_loss_D += loss_D.item()

    # Calcula pérdidas promedio por epoch
    avg_train_loss_G = running_loss_G / len(train_loader)
    avg_train_loss_D = running_loss_D / len(train_loader)
    train_losses_G.append(avg_train_loss_G)
    train_losses_D.append(avg_train_loss_D)
    print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss G: {avg_train_loss_G:.4f}, D: {avg_train_loss_D:.4f}')

    # Validación por batch
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

    # Early stopping: comprobar si la pérdida de validación mejora
    if avg_val_loss_G < best_val_loss:
        best_val_loss = avg_val_loss_G
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"No mejora en la pérdida de validación ({epochs_no_improve}/{EARLY_STOPPING_PATIENCE})")
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping activado en el epoch {epoch+1}")
            break

    # Guardar y mostrar la gráfica de seguimiento en cada epoch
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
    plt.savefig(os.path.join(GRAPH_PATH, f'gan_morph_loss_epoch_{epoch+1}.png'))
    plt.pause(0.001)  # Muestra la gráfica brevemente y continúa
    plt.close('all')  # Cierra todas las ventanas de la gráfica inmediatamente


# Guardar los modelos entrenados al finalizar
torch.save(generator.state_dict(), 'gan_morph_generator.pth')  # Pesos del generador
torch.save(discriminator.state_dict(), 'gan_morph_discriminator.pth')  # Pesos del discriminador
print('Entrenamiento y validación finalizados y modelos guardados.')

