# Configuración centralizada de hiperparámetros y parámetros globales
IMG_SIZE = 128           # Tamaño (alto y ancho) de las imágenes procesadas por el modelo (ej: 128x128 píxeles)
EMBEDDING_DIM = 256      # Dimensión del vector de embedding para cada letra (capacidad de representación del texto)
VOCAB = 'ABCDEFGHIJKLMNÑOPQRSTUVWXYZ'  # Letras reconocidas por el modelo (vocabulario)
BATCH_SIZE = 32          # Número de muestras procesadas en cada batch de entrenamiento
EPOCHS = 10              # Número de veces que el modelo recorre todo el dataset (épocas)
LAMBDA_PERCEPTUAL = 0.2  # Peso de la perceptual loss en la función de coste total (calidad visual vs error pixel)
LR = 0.001               # Tasa de aprendizaje del optimizador (velocidad de ajuste de parámetros)
DATASET_DIR = None       # Ruta al directorio del dataset (se puede completar en el script principal)
DROPOUT_PROB = 0.2       # Probabilidad de dropout en el generador y discriminador
