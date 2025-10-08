# Dataset personalizado para dictadología
# Cada muestra es una imagen y su índice de letra
# Adaptado para cargar imágenes de dictadología, normalizadas para el modelo generativo.
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import torchvision.transforms.functional as F

class DictaDataset(Dataset):
    def __init__(self, data_dir, vocab, img_size):
        """
        Args:
            data_dir (str): Ruta al directorio raíz del dataset
            vocab (str): Cadena con todas las letras presentes
            img_size (int): Tamaño al que se redimensionan las imágenes
        """
        self.data = []  # Lista de tuplas (índice de letra, ruta de imagen)
        self.vocab = vocab
        self.img_size = img_size

        # Recorre cada letra y añade sus imágenes
        for idx, letter in enumerate(vocab):
            letter_dir = os.path.join(data_dir, letter)
            if not os.path.exists(letter_dir):
                continue
            for img_name in os.listdir(letter_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(letter_dir, img_name)
                    self.data.append((idx, img_path))
        print(f"[INFO] Total de imágenes encontradas: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def get_random_jitter_values(self):
        brightness = random.uniform(0.7, 1.3)
        contrast = random.uniform(0.8, 1.2)
        saturation = random.uniform(0.8, 1.2)
        hue = random.uniform(0, 0.35)
        angle = random.uniform(-15, 15)
        return brightness, contrast, saturation, hue, angle

    def __getitem__(self, idx):
        # Devuelve el índice de la letra y la imagen transformada
        label, img_path = self.data[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            # Resize, flip y rotate
            transform_basic = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomRotation(10),
            ])
            image = transform_basic(image)
            # Aplica jitter manual y guarda los valores
            brightness, contrast, saturation, hue, angle = self.get_random_jitter_values()
            image = F.adjust_brightness(image, brightness)
            image = F.adjust_contrast(image, contrast)
            image = F.adjust_saturation(image, saturation)
            image = F.adjust_hue(image, hue)
            image = F.rotate(image, angle)
            image = transforms.ToTensor()(image)
            #image = transforms.Normalize(mean=[0.5], std=[0.5])(image)
        except Exception as e:
            print(f"[ERROR] Fallo al cargar imagen: {img_path}. Error: {e}")
            # Devuelve un tensor de ceros si la imagen falla
            image = torch.zeros(3, self.img_size, self.img_size)
            brightness, contrast, saturation, hue = 1.0, 1.0, 1.0, 0.0  # valores neutros si falla
        return torch.tensor(label), image, (brightness, contrast, saturation, hue, angle)
