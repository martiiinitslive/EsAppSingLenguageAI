
# Dataset personalizado para dictadología
# Cada muestra es una imagen y su índice de letra
# Adaptado para cargar imágenes de dictadología, normalizadas para el modelo generativo.
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

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
        # Transformaciones: escala a gris, resize y tensor normalizado a [-1, 1]
        self.transform = transforms.Compose([
            # transforms.Grayscale(),
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
