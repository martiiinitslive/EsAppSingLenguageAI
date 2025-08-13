import os
import shutil
from itertools import product

# Configuración
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DICTADOLOGIA = os.path.join(BASE_DIR, '..', 'data', 'dictadologia')
MORPHING = os.path.join(BASE_DIR, '..', 'data', 'morphing')

# Obtener todas las letras presentes en dictadologia
letras = [l for l in os.listdir(DICTADOLOGIA) if os.path.isdir(os.path.join(DICTADOLOGIA, l))]

# Crear todas las combinaciones posibles (incluyendo consigo misma)
pares = list(product(letras, letras))

# Organizar el dataset de morphing
def organizar_morphing(max_imgs=10):
    if not os.path.exists(MORPHING):
        os.makedirs(MORPHING)
    for letra1, letra2 in pares:
        par_dir = os.path.join(MORPHING, f"{letra1}_{letra2}")
        start_dir = os.path.join(par_dir, 'start')
        end_dir = os.path.join(par_dir, 'end')
        os.makedirs(start_dir, exist_ok=True)
        os.makedirs(end_dir, exist_ok=True)
        # Copiar solo las primeras max_imgs imágenes de la letra inicial a start
        origen_start = os.path.join(DICTADOLOGIA, letra1)
        if os.path.exists(origen_start):
            imgs_start = sorted(os.listdir(origen_start))[:max_imgs]
            for img in imgs_start:
                ext = os.path.splitext(img)[1]
                nuevo_nombre = f"{letra1.upper()}_{img}"
                dest_path = os.path.join(start_dir, nuevo_nombre)
                if not os.path.exists(dest_path):
                    shutil.copy2(os.path.join(origen_start, img), dest_path)
        # Copiar solo las primeras max_imgs imágenes de la letra final a end
        origen_end = os.path.join(DICTADOLOGIA, letra2)
        if os.path.exists(origen_end):
            imgs_end = sorted(os.listdir(origen_end))[:max_imgs]
            for img in imgs_end:
                ext = os.path.splitext(img)[1]
                nuevo_nombre = f"{letra2.upper()}_{img}"
                dest_path = os.path.join(end_dir, nuevo_nombre)
                if not os.path.exists(dest_path):
                    shutil.copy2(os.path.join(origen_end, img), dest_path)
        # Si tienes imágenes intermedias, puedes añadir la lógica aquí
        # intermediate_dir = os.path.join(par_dir, 'intermediate')
        # os.makedirs(intermediate_dir, exist_ok=True)

if __name__ == '__main__':
    organizar_morphing(max_imgs=10)
    print('Dataset de morphing organizado con todos los pares de letras (máx 10 imágenes por letra).')
