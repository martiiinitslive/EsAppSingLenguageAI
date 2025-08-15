# Script de entrenamiento para TextToDictaModel (ttd)
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
from ..models.modelos.ttd_model import TextToDictaModel
from tqdm import tqdm

# Perceptual Loss usando VGG16
from torchvision import models

# Clase para Perceptual Loss
class PerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features[:16].eval()
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


DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dictadologia'))  # Ruta absoluta al dataset de dictadología
IMG_SIZE = 64  # Tamaño de las imágenes (ancho y alto)
BATCH_SIZE = 16  # Número de muestras por batch
EPOCHS = 50  # Número de épocas de entrenamiento (más entrenamiento para mejor aprendizaje)
EMBEDDING_DIM = 128  # Dimensión del embedding para cada letra (debe coincidir con ttd_model.py)
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
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)  # Normaliza a [-1, 1] para Tanh
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
        try:
            image = Image.open(img_path)
            image = self.transform(image)
        except Exception as e:
            print(f"[ERROR] Fallo al cargar imagen: {img_path}. Error: {e}")
            # Devuelve un tensor de ceros si la imagen falla
            image = torch.zeros(1, self.img_size, self.img_size)
        return torch.tensor(label), image

# Carga el dataset y lo divide en entrenamiento/validación (80/20)

from torch.utils.data import random_split
print("[INFO] Cargando dataset completo...")
full_dataset = DictaDataset(DATASET_DIR, VOCAB, IMG_SIZE)
print(f"[INFO] Total de muestras en el dataset: {len(full_dataset)}")
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
print(f"[INFO] Dividiendo en {train_size} entrenamiento y {val_size} validación...")
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("[INFO] Dataset cargado y dividido correctamente.")


# Instancia el modelo generador de dictadología
# IMPORTANTE: El valor de embedding_dim debe coincidir con el modelo en ttd_model.py
print("[INFO] Instanciando modelo...")
model = TextToDictaModel(vocab_size=len(VOCAB), embedding_dim=EMBEDDING_DIM, img_size=IMG_SIZE)
criterion_mse = nn.MSELoss()  # Pérdida de error cuadrático medio
criterion_perceptual = PerceptualLoss().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
lambda_perceptual = 0.1  # Peso de la perceptual loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizador Adam
print("[INFO] Modelo instanciado.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("[INFO] Iniciando entrenamiento de TextToDictaModel...")
print(f"[INFO] Entrenando en dispositivo: {device}")
model.to(device)

# Configuración de la carpeta para guardar imágenes de seguimiento
# Listas para guardar la evolución de las pérdidas
train_losses = []  # Pérdida en entrenamiento
val_losses = []    # Pérdida en validación

import matplotlib.pyplot as plt
RESULTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'imagenes', 'train-ttd-model-epoch'))
os.makedirs(RESULTS_PATH, exist_ok=True)
# Carpeta para guardar ejemplos generados por el modelo
EXAMPLES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'imagenes', 'ejemplos-generados-ttd-model'))
os.makedirs(EXAMPLES_PATH, exist_ok=True)

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
                img_pil = transforms.ToPILImage()(img)
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

