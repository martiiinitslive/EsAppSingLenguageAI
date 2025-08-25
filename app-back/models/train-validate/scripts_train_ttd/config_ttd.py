# Configuración centralizada de hiperparámetros y parámetros globales

# Parámetros del modelo

## GENERADOR
N = 5  # bloques upsampling de ttd_model
IMG_SIZE = 256  # Tamaño (alto y ancho) de las imágenes procesadas por el modelo
EMBEDDING_DIM = 256  # Dimensión del vector de embedding para cada letra
VOCAB = 'ABCDEFGHIJKLMNÑOPQRSTUVWXYZ'  # Letras reconocidas por el modelo (vocabulario)
INIT_MAP_SIZE = IMG_SIZE // (2 ** N)  # Tamaño inicial del mapa de características
INIT_CHANNELS = 256  # Número de canales iniciales
DROPOUT_ENCODER = 0.1  # Dropout para el encoder del generador
DROPOUT_DECODER = 0.05  # Dropout para el decoder del generador
LEAKY_RELU_SLOPE = 0.2  # Parámetro 'negative_slope' para LeakyReLU

## DISCRIMINADOR
DROPOUT_DISC = 0.1  # Dropout para el discriminador

# Parámetros de entrenamiento
BATCH_SIZE = 32  # Número de muestras procesadas en cada batch de entrenamiento
EPOCHS = 10  # Número de épocas
LAMBDA_PERCEPTUAL = 0.2  # Peso de la perceptual loss en la función de coste total
LR_G = 0.0002  # Tasa de aprendizaje del generador
LR_D = 0.0001  # Tasa de aprendizaje del discriminador
DATASET_DIR = None  # Ruta al directorio del dataset
NUM_EJEMPLOS = 1  # Número de ejemplos reales y generados a guardar por epoch
