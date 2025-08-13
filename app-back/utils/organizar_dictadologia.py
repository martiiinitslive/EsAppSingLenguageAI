import os
import shutil
import json

# Configuración
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGENES = [
    os.path.join(BASE_DIR, '..', 'data', 'dataset-en-bruto', 'lse_static1'),
    os.path.join(BASE_DIR, '..', 'data', 'dataset-en-bruto', 'lse_static2'),
    #os.path.join(BASE_DIR, '..', 'data', 'dataset-en-bruto', 'otra_carpeta'),
    # ...añade aquí más rutas si lo necesitas
]
DESTINO = os.path.join(BASE_DIR, '..', 'data', 'dictadologia')

# Letras con movimiento y su tipo/base
movimientos = {
    'g': {'tipo': 'lateral', 'base': 'g'},
    'h': {'tipo': 'lateral', 'base': 'h'},
    'j': {'tipo': 'circular', 'base': 'j'},
    'ñ': {'tipo': 'lateral', 'base': 'n'},
    'z': {'tipo': 'zigzag', 'base': 'z'}
}

# Crear estructura destino y copiar imágenes
def organizar_dataset():
    if not os.path.exists(DESTINO):
        os.makedirs(DESTINO)
    # Contadores para renombrar imágenes por letra y carpeta destino
    contadores = {}
    for ORIGEN in ORIGENES:
        if not os.path.exists(ORIGEN):
            continue
        letras = [l for l in os.listdir(ORIGEN) if os.path.isdir(os.path.join(ORIGEN, l))]
        for letra in letras:
            origen_letra = os.path.join(ORIGEN, letra)
            if letra not in contadores:
                contadores[letra] = {}
            # Función para obtener el siguiente nombre
            def get_nombre(letra, carpeta):
                if carpeta not in contadores[letra]:
                    contadores[letra][carpeta] = 1
                nombre = f"{letra.upper()}_{contadores[letra][carpeta]}"
                contadores[letra][carpeta] += 1
                return nombre
            if letra in movimientos:
                # Carpeta base
                base = movimientos[letra]['base']
                base_dest = os.path.join(DESTINO, letra, f'base_{base}')
                os.makedirs(base_dest, exist_ok=True)
                for img in os.listdir(origen_letra):
                    ext = os.path.splitext(img)[1]
                    nuevo_nombre = get_nombre(letra, f'base_{base}') + ext
                    dest_path = os.path.join(base_dest, nuevo_nombre)
                    if not os.path.exists(dest_path):
                        shutil.copy2(os.path.join(origen_letra, img), dest_path)
                # Carpeta movimiento
                mov_tipo = movimientos[letra]['tipo']
                mov_dest = os.path.join(DESTINO, letra, f'movimiento_{mov_tipo}')
                os.makedirs(mov_dest, exist_ok=True)
                for img in os.listdir(origen_letra):
                    ext = os.path.splitext(img)[1]
                    nuevo_nombre = get_nombre(letra, f'movimiento_{mov_tipo}') + ext
                    dest_path = os.path.join(mov_dest, nuevo_nombre)
                    if not os.path.exists(dest_path):
                        shutil.copy2(os.path.join(origen_letra, img), dest_path)
            else:
                # Letras estáticas
                letra_dest = os.path.join(DESTINO, letra)
                os.makedirs(letra_dest, exist_ok=True)
                for img in os.listdir(origen_letra):
                    ext = os.path.splitext(img)[1]
                    nuevo_nombre = get_nombre(letra, 'letra') + ext
                    dest_path = os.path.join(letra_dest, nuevo_nombre)
                    if not os.path.exists(dest_path):
                        shutil.copy2(os.path.join(origen_letra, img), dest_path)

# Guardar archivo de mapeo de movimientos
def guardar_mapeo():
    with open(os.path.join(DESTINO, 'movimientos.json'), 'w', encoding='utf-8') as f:
        json.dump(movimientos, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    organizar_dataset()
    guardar_mapeo()
    print('Dataset organizado y mapeo de movimientos guardado.')
