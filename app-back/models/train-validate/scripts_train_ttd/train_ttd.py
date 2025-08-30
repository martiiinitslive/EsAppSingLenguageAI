# Script de entrenamiento para TextToDictaModel (ttd)
# Este script entrena un modelo generativo que convierte texto (letra) en una imagen de dictadología.
# Modularizado: importa dataset, modelo y pérdidas desde archivos separados.
# Incluye barra de progreso, guardado de ejemplos, gráficas de pérdida y comentarios explicativos.

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import models
import torch.nn.functional as F

# Importación absoluta de los módulos
import sys, os
modelos_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'modelos'))
sys.path.append(modelos_path)
from ttd_model import TextToDictaModel, DictaDiscriminator
from dataset_ttd import DictaDataset
from losses_ttd import PerceptualLoss
from config_ttd import IMG_SIZE, EMBEDDING_DIM, VOCAB, BATCH_SIZE, EPOCHS, LAMBDA_PERCEPTUAL, NUM_EJEMPLOS, LR_G, LR_D

if __name__ == "__main__":
    # Definir el dispositivo de cómputo (GPU si está disponible, si no CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo seleccionado para entrenamiento: {device}")
    if device.type == "cuda":
        print(f"[INFO] Nombre de GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] Memoria total GPU: {torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)} MB")
    else:
        print("[WARNING] Entrenando en CPU. El entrenamiento será mucho más lento.")

    # BASE_DIR apunta a la raíz del proyecto
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

    DATASET_DIR = os.path.join(BASE_DIR, 'app-back', 'data', 'dictadologia')

    RESULTS_PATH = os.path.join(BASE_DIR, 'app-back', 'models', 'train-validate', 'scripts_train_ttd', 'imagenes', 'train-ttd-model-epoch')
    os.makedirs(RESULTS_PATH, exist_ok=True)

    EXAMPLES_PATH = os.path.join(BASE_DIR, 'app-back', 'models', 'train-validate', 'scripts_train_ttd', 'imagenes', 'ejemplos-generados-ttd-model')
    os.makedirs(EXAMPLES_PATH, exist_ok=True)

    REAL_EXAMPLES_PATH = os.path.join(BASE_DIR, 'app-back', 'models', 'train-validate', 'scripts_train_ttd', 'imagenes', 'ejemplos-reales-dataset')
    os.makedirs(REAL_EXAMPLES_PATH, exist_ok=True)

    print(f"[DEBUG] DATASET_DIR: {DATASET_DIR}")


    # Carga el dataset y lo divide en entrenamiento/validación (80/20)
    full_dataset = DictaDataset(DATASET_DIR, VOCAB, IMG_SIZE)
    print(f"[DEBUG] Total imágenes en dataset: {len(full_dataset)}")
    train_size = int(0.8 * len(full_dataset))
    print(f"[DEBUG] Tamaño del conjunto de entrenamiento: {train_size}")
    val_size = len(full_dataset) - train_size
    print(f"[DEBUG] Tamaño del conjunto de validación: {val_size}")
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,      # Prueba 4, 8 o más según tu CPU
        pin_memory=True     # Solo si usas GPU
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model_G = TextToDictaModel(vocab_size=len(VOCAB), embedding_dim=EMBEDDING_DIM, img_size=IMG_SIZE).to(device)
    model_D = DictaDiscriminator(img_size=IMG_SIZE, in_channels=3).to(device)

    optimizer_G = optim.Adam(model_G.parameters(), lr=LR_G)
    optimizer_D = optim.Adam(model_D.parameters(), lr=LR_D)

    criterion_bce = nn.BCELoss()
    criterion_mse = nn.MSELoss()
    criterion_perceptual = PerceptualLoss().to(device)
    lambda_perceptual = LAMBDA_PERCEPTUAL  # Ajusta este valor en config_ttd.py

    train_losses_G = []
    train_losses_D = []
    val_losses_G = []

    # Inicializar el tiempo de inicio antes del bucle de entrenamiento
    start_time = time.time()

    # Guardar ejemplos reales del dataset antes de entrenar
    import torchvision.utils as vutils
    val_batch = next(iter(val_loader))
    val_labels, val_images = val_batch
    val_images = (val_images + 1) / 2  # Desnormaliza a [0, 1] para que sean visualizables
    for i in range(min(NUM_EJEMPLOS, val_images.size(0))):
        img = val_images[i].cpu()
        idx_letra = val_labels[i].item()
        letra = VOCAB[idx_letra]
        from torchvision import transforms as T
        img_pil = T.ToPILImage()(img)
        img_name = f'ejemplo_real_{i+1}_{letra}.png'
        img_path = os.path.join(REAL_EXAMPLES_PATH, img_name)
        img_pil.save(img_path)
    print(f"[INFO] Ejemplos reales del dataset guardados en: {REAL_EXAMPLES_PATH}")

    # Inicializa listas para guardar las pérdidas por tipo
    train_losses_G_bce = []
    train_losses_G_mse = []
    train_losses_G_perceptual = []

    # Bucle principal de entrenamiento
    for epoch in range(EPOCHS):
        print(f"[INFO] Comenzando epoch {epoch+1}/{EPOCHS}...")
        print(f"[INFO] GPU disponible: {torch.cuda.is_available()} | Dispositivo actual: {device}")
        model_G.train()
        model_D.train()
        running_loss_G = 0.0
        running_loss_D = 0.0
        running_loss_G_bce = 0.0
        running_loss_G_mse = 0.0
        running_loss_G_perceptual = 0.0

        for i, (labels, real_images) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} - Train")):
            labels, real_images = labels.to(device), real_images.to(device)
            batch_size = real_images.size(0)

            # --- Entrenamiento del Discriminador ---
            optimizer_D.zero_grad()
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Imágenes reales
            output_real = model_D(real_images)
            loss_real = criterion_bce(output_real, real_labels)

            # Imágenes generadas (falsas)
            fake_images = model_G(labels)
            output_fake = model_D(fake_images.detach())
            loss_fake = criterion_bce(output_fake, fake_labels)

            # Pérdida total del discriminador
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()
            running_loss_D += loss_D.item()

            # --- Entrenamiento del Generador ---
            optimizer_G.zero_grad()
            output_fake_for_G = model_D(fake_images)
            loss_G_bce = criterion_bce(output_fake_for_G, real_labels)
            loss_G_mse = criterion_mse(fake_images, real_images)
            loss_G_perceptual = criterion_perceptual((fake_images + 1) / 2, (real_images + 1) / 2)
            loss_G = loss_G_bce + loss_G_mse + lambda_perceptual * loss_G_perceptual
            loss_G.backward()
            optimizer_G.step()
            running_loss_G += loss_G.item()
            running_loss_G_bce += loss_G_bce.item()
            running_loss_G_mse += loss_G_mse.item()
            running_loss_G_perceptual += loss_G_perceptual.item()

        avg_train_loss_G = running_loss_G / len(train_loader)
        avg_train_loss_D = running_loss_D / len(train_loader)
        avg_train_loss_G_bce = running_loss_G_bce / len(train_loader)
        avg_train_loss_G_mse = running_loss_G_mse / len(train_loader)
        avg_train_loss_G_perceptual = running_loss_G_perceptual / len(train_loader)
        train_losses_G.append(avg_train_loss_G)
        train_losses_D.append(avg_train_loss_D)
        train_losses_G_bce.append(avg_train_loss_G_bce)
        train_losses_G_mse.append(avg_train_loss_G_mse)
        train_losses_G_perceptual.append(avg_train_loss_G_perceptual)

        # --- Tiempo ---
        current_time = time.time()
        elapsed_time = current_time - start_time
        elapsed_hours = int(elapsed_time // 3600)
        elapsed_minutes = int((elapsed_time % 3600) // 60)
        elapsed_seconds = int(elapsed_time % 60)
        print(f'[INFO] Epoch {epoch+1}/{EPOCHS}, Gen Loss: {avg_train_loss_G:.4f}, Disc Loss: {avg_train_loss_D:.4f}, Tiempo total: {elapsed_hours}:{elapsed_minutes:02d}:{elapsed_seconds:02d}')

        # --- Validación ---
        model_G.eval()
        val_loss_G = 0.0
        with torch.no_grad():
            for labels, images in tqdm(val_loader, desc=f"Epoch {epoch+1} - Val"):
                labels, images = labels.to(device), images.to(device)
                outputs = model_G(labels)
                loss_mse = criterion_mse(outputs, images)
                # CAMBIO: normaliza a [0, 1] para la pérdida perceptual
                loss_perceptual = criterion_perceptual((outputs + 1) / 2, (images + 1) / 2)
                loss = loss_mse + lambda_perceptual * loss_perceptual
                val_loss_G += loss.item()
        avg_val_loss_G = val_loss_G / len(val_loader) if len(val_loader) > 0 else 0
        val_losses_G.append(avg_val_loss_G)
        print(f'[INFO] Epoch {epoch+1}/{EPOCHS}, Val Gen Loss: {avg_val_loss_G:.4f}')

        # --- Gráfica de pérdidas ---
        plt.figure(figsize=(8,5))
        plt.plot(train_losses_G, label='Train Gen Loss')
        plt.plot(train_losses_D, label='Train Disc Loss')
        plt.plot(val_losses_G, label='Val Gen Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Evolución de la pérdida GAN (Generador y Discriminador)')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        save_path = os.path.join(RESULTS_PATH, f'gan_loss_epoch_{epoch+1}.png')
        plt.savefig(save_path)
        print(f"[INFO] Gráfica guardada en: {save_path}")
        plt.close()

        # --- Gráfica de pérdidas por tipo ---
        plt.figure(figsize=(8,5))
        plt.plot(train_losses_G_bce, label='Gen BCE Loss')
        plt.plot(train_losses_G_mse, label='Gen MSE Loss')
        plt.plot(train_losses_G_perceptual, label='Gen Perceptual Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Pérdidas del Generador por tipo')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        save_path_types = os.path.join(RESULTS_PATH, f'gan_gen_loss_types_epoch_{epoch+1}.png')
        plt.savefig(save_path_types)
        print(f"[INFO] Gráfica de pérdidas por tipo guardada en: {save_path_types}")
        plt.close()

        # --- Guardar ejemplos generados ---
        with torch.no_grad():
            val_labels_batch = val_labels.to(device)
            outputs = model_G(val_labels_batch)
            outputs = (outputs + 1) / 2
            for i in range(min(NUM_EJEMPLOS, outputs.size(0))):
                img = outputs[i].cpu()
                img = img.squeeze(0) if img.dim() == 3 and img.size(0) == 1 else img
                from torchvision import transforms as T
                img_pil = T.ToPILImage()(img)
                idx_letra = val_labels[i].item()
                letra = VOCAB[idx_letra]
                img_name = f'epoch_{epoch+1}_example_{i+1}_{letra}.png'
                img_path = os.path.join(EXAMPLES_PATH, img_name)
                img_pil.save(img_path)
            print(f"[INFO] Ejemplos generados guardados en: {EXAMPLES_PATH}")

    # --- Tiempo total ---
    current_time = time.time()
    elapsed_time = current_time - start_time
    elapsed_hours = int(elapsed_time // 3600)
    elapsed_minutes = int((elapsed_time % 3600) // 60)
    elapsed_seconds = int(elapsed_time % 60)
    print(f'[INFO] Tiempo total de entrenamiento: {elapsed_hours}:{elapsed_minutes:02d}:{elapsed_seconds:02d}')

    base_path = os.path.join(BASE_DIR, 'app-back', 'models', 'modelos_trained', 'ttd_model')
    os.makedirs(base_path, exist_ok=True)
    GEN_PATH = os.path.join(base_path, 'GEN_ttd_model.pth')
    torch.save(model_G.state_dict(), GEN_PATH)
    print(f'[INFO] Generador guardado en: {GEN_PATH}')

    # --- Guardar el discriminador entrenado ---
    DISC_PATH = os.path.join(base_path, 'DISC_ttd_model.pth')
    torch.save(model_D.state_dict(), DISC_PATH)
    print(f'[INFO] Discriminador guardado en: {DISC_PATH}')

    # Puedes poner esto en un archivo de pérdidas o directamente en train_ttd.py
    class PerceptualLoss(nn.Module):
        def __init__(self, resize=True):
            super(PerceptualLoss, self).__init__()
            vgg = models.vgg16(pretrained=True).features[:16].eval()
            for param in vgg.parameters():
                param.requires_grad = False
            self.vgg = vgg
            self.resize = resize

        def forward(self, x, y):
            # x, y: [batch, 3, H, W] in [0, 1] range
            if self.resize:
                x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
                y = F.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)
            return F.l1_loss(self.vgg(x), self.vgg(y))
