
# Script de entrenamiento para TextToDictaModel
# Este script entrena un modelo generativo que convierte texto (letra) en una imagen de dictadología.
# Debes adaptar la carga de datos a tu dataset de imágenes de dictadología.


# Añade la carpeta raíz al path para importar módulos del proyecto
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from models.generator_model import TextToDictaModel


# Configuración de hiperparámetros y rutas
DATASET_DIR = 'data/dictadologia'  # Ruta al dataset de dictadología
IMG_SIZE = 64  # Tamaño de las imágenes (ancho y alto)
BATCH_SIZE = 16  # Número de muestras por batch
EPOCHS = 10  # Número de épocas de entrenamiento
EMBEDDING_DIM = 32  # Dimensión del embedding para cada letra
VOCAB = 'ABCDEFGHIJKLMNÑOPQRSTUVWXYZ'  # Todas las letras presentes en el dataset


# Verifica cuántas imágenes hay por cada letra en el dataset
for letter in VOCAB:
    letter_dir = os.path.join(DATASET_DIR, letter)
    if os.path.exists(letter_dir):
        imgs = [f for f in os.listdir(letter_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"{letter}: {len(imgs)} imágenes")
    else:
        print(f"{letter}: carpeta no existe")



# Dataset personalizado para dictadología
class DictaDataset(Dataset):
    def __init__(self, data_dir, vocab, img_size):
        self.data = []  # Lista de tuplas (índice de letra, ruta de imagen)
        self.vocab = vocab
        self.img_size = img_size
        # Transformaciones: escala a gris, resize y tensor
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        # Recorre cada letra y añade sus imágenes
        for idx, letter in enumerate(vocab):
            letter_dir = os.path.join(data_dir, letter)
            if not os.path.exists(letter_dir):
                continue
            for img_name in os.listdir(letter_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(letter_dir, img_name)
                    self.data.append((idx, img_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Devuelve el índice de la letra y la imagen transformada
        label, img_path = self.data[idx]
        image = Image.open(img_path)
        image = self.transform(image)
        return torch.tensor([label]), image



# Carga el dataset y lo divide en entrenamiento/validación (80/20)
from torch.utils.data import random_split
full_dataset = DictaDataset(DATASET_DIR, VOCAB, IMG_SIZE)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Instancia el modelo generador de dictadología
model = TextToDictaModel(vocab_size=len(VOCAB), embedding_dim=EMBEDDING_DIM, img_size=IMG_SIZE)
criterion = nn.MSELoss()  # Pérdida de error cuadrático medio
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizador Adam




# Listas para guardar la evolución de las pérdidas
train_losses = []  # Pérdida en entrenamiento
val_losses = []    # Pérdida en validación

import matplotlib.pyplot as plt

# Bucle principal de entrenamiento
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    # Entrenamiento por batch
    for i, (labels, images) in enumerate(train_loader):
        print(f"Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}")
        optimizer.zero_grad()
        outputs = model(labels)  # Genera imagen a partir de la letra
        # Ajusta el shape para comparar correctamente
        if outputs.shape != images.shape:
            # Si outputs es [batch, 64, 64] y images es [batch, 1, 64, 64], elimina el canal
            images = images.squeeze(1)
        loss = criterion(outputs, images)  # Calcula la pérdida
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss/len(train_loader)
    train_losses.append(avg_train_loss)
    print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}')

    # Validación por batch
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for labels, images in val_loader:
            outputs = model(labels)
            if outputs.shape != images.shape:
                images = images.squeeze(1)
            loss = criterion(outputs, images)
            val_loss += loss.item()
    avg_val_loss = val_loss/len(val_loader) if len(val_loader) > 0 else 0
    val_losses.append(avg_val_loss)
    print(f'Epoch {epoch+1}/{EPOCHS}, Val Loss: {avg_val_loss:.4f}')

    # Guardar y mostrar la gráfica de seguimiento en cada epoch
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Evolución de la pérdida modelo de texto a dictadologia')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'text_to_dicta_loss_epoch_{epoch+1}.png')
    plt.show()
    plt.close()



# Guardar el modelo entrenado al finalizar
torch.save(model.state_dict(), 'text_to_dicta_model.pth')  # Pesos del modelo
print('Entrenamiento finalizado y modelo guardado.')

