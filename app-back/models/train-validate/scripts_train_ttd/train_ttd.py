# Script de entrenamiento para TextToDictaModel (ttd)
# Este script entrena un modelo generativo que convierte texto (letra) en una imagen de dictadología.
# Modularizado: importa dataset, modelo y pérdidas desde archivos separados.
# Incluye barra de progreso, guardado de ejemplos, gráficas de pérdida y comentarios explicativos.

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import models

# Importación absoluta de los módulos
import sys, os
modelos_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'modelos'))
sys.path.append(modelos_path)
from ttd_model import TextToDictaModel
from dataset_ttd import DictaDataset
from losses_ttd import PerceptualLoss
from config_ttd import IMG_SIZE, EMBEDDING_DIM, VOCAB, BATCH_SIZE, EPOCHS, LAMBDA_PERCEPTUAL

DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'dictadologia'))  # Ruta absoluta al dataset

# Carga el dataset y lo divide en entrenamiento/validación (80/20)
full_dataset = DictaDataset(DATASET_DIR, VOCAB, IMG_SIZE)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = TextToDictaModel(vocab_size=len(VOCAB), embedding_dim=EMBEDDING_DIM, img_size=IMG_SIZE)
criterion_mse = nn.MSELoss()  # Pérdida de error cuadrático medio
criterion_perceptual = PerceptualLoss().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizador Adam
lambda_perceptual = LAMBDA_PERCEPTUAL

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("[INFO] Iniciando entrenamiento de TextToDictaModel...")
print(f"[INFO] Entrenando en dispositivo: {device}")
model.to(device)

train_losses = []  # Pérdida en entrenamiento
val_losses = []    # Pérdida en validación


# Configuración de carpetas para guardar resultados
RESULTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'imagenes', 'train-ttd-model-epoch'))
os.makedirs(RESULTS_PATH, exist_ok=True)
EXAMPLES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'imagenes', 'ejemplos-generados-ttd-model'))
os.makedirs(EXAMPLES_PATH, exist_ok=True)
# Carpeta para ejemplos reales del dataset
REAL_EXAMPLES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'imagenes', 'ejemplos-reales-dataset'))
os.makedirs(REAL_EXAMPLES_PATH, exist_ok=True)

# Guardar ejemplos reales del dataset antes de entrenar
import torchvision.utils as vutils
batch = next(iter(train_loader))
_, images = batch
images = (images + 1) / 2  # Desnormaliza a [0, 1]
for i in range(min(4, images.size(0))):
    img = images[i].cpu()
    from torchvision import transforms as T
    img_pil = T.ToPILImage()(img)
    img_name = f'ejemplo_real_{i+1}.png'
    img_path = os.path.join(REAL_EXAMPLES_PATH, img_name)
    img_pil.save(img_path)
print(f"Ejemplos reales del dataset guardados en: {REAL_EXAMPLES_PATH}")

# Bucle principal de entrenamiento
for epoch in range(EPOCHS):
    print(f"[INFO] Comenzando epoch {epoch+1}/{EPOCHS}...")
    model.train()
    running_loss = 0.0
    # Entrenamiento por batch con barra de progreso
    for i, (labels, images) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} - Train")):
        optimizer.zero_grad()
        labels, images = labels.to(device), images.to(device)
        outputs = model(labels)
        loss_mse = criterion_mse(outputs, images)
        loss_perceptual = criterion_perceptual(outputs, images)
        loss = loss_mse + lambda_perceptual * loss_perceptual
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss/len(train_loader)
    train_losses.append(avg_train_loss)
    print(f'[INFO] Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}')

    # Validación por batch
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for labels, images in tqdm(val_loader, desc=f"Epoch {epoch+1} - Val"):
            labels, images = labels.to(device), images.to(device)
            outputs = model(labels)
            loss_mse = criterion_mse(outputs, images)
            loss_perceptual = criterion_perceptual(outputs, images)
            loss = loss_mse + lambda_perceptual * loss_perceptual
            val_loss += loss.item()
    avg_val_loss = val_loss/len(val_loader) if len(val_loader) > 0 else 0
    val_losses.append(avg_val_loss)
    print(f'[INFO] Epoch {epoch+1}/{EPOCHS}, Val Loss: {avg_val_loss:.4f}')

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
    save_path = os.path.join(RESULTS_PATH, f'text_to_dicta_loss_epoch_{epoch+1}.png')
    plt.savefig(save_path)
    print(f"[INFO] Gráfica guardada en: {save_path}")
    plt.close()

    # Guardar ejemplos generados por el modelo en cada epoch
    with torch.no_grad():
        for labels, _ in val_loader:
            labels = labels.to(device)
            outputs = model(labels)
            # Desnormaliza de [-1, 1] a [0, 1] para visualización
            outputs = (outputs + 1) / 2
            for i in range(min(4, outputs.size(0))):
                img = outputs[i].cpu()
                img = img.squeeze(0) if img.dim() == 3 and img.size(0) == 1 else img
                from torchvision import transforms as T
                img_pil = T.ToPILImage()(img)
                img_name = f'epoch_{epoch+1}_example_{i+1}.png'
                img_path = os.path.join(EXAMPLES_PATH, img_name)
                img_pil.save(img_path)
            break  # Solo el primer batch
        print(f"[INFO] Ejemplos generados guardados en: {EXAMPLES_PATH}")

# Guardar el modelo entrenado al finalizar
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modelEntrenado', 'ttd_model_trained.pth'))
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f'[INFO] Entrenamiento finalizado y modelo guardado en: {MODEL_PATH}')
