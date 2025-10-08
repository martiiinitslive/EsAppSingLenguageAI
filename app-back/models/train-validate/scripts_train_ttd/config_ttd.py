# Configuración centralizada de hiperparámetros y parámetros globales

#---------------------------------------------------------------------------------------------------------------------------------

# Parámetros del modelo

#---------------------------------------------------------------------------------------------------------------------------------

    ## GENERADOR

NOISE_DIM = 32  # Dimensión del vector de ruido latente para el generador ttd_model

N = 5  # bloques upsampling de ttd_model
#   ↑ Subir: más bloques, mayor capacidad y profundidad, pero más coste computacional.
#   ↓ Bajar: menos bloques, modelo más simple, menos capacidad de representación.

IMG_SIZE = 320  # Tamaño (alto y ancho) de las imágenes procesadas por el modelo
#   ↑ Subir: imágenes de mayor resolución, más detalles, pero más memoria y tiempo de entrenamiento.
#   ↓ Bajar: imágenes más pequeñas, menos detalles, entrenamiento más rápido.

EMBEDDING_DIM = 384  # Dimensión del vector de embedding para cada letra
#   ↑ Subir: embeddings más ricos, más capacidad para representar letras, pero más parámetros.
#   ↓ Bajar: embeddings más simples, menos capacidad, menos parámetros.

VOCAB = 'ABCDEFGHIJKLMNÑOPQRSTUVWXYZ'  # Letras reconocidas por el modelo (vocabulario)
#   Modificar: cambia el conjunto de letras que el modelo puede procesar.

INIT_MAP_SIZE = IMG_SIZE // (2 ** N)  # Tamaño inicial del mapa de características
#   Depende de IMG_SIZE y N, controla el tamaño inicial de los mapas de activación.

INIT_CHANNELS = 320  # Número de canales iniciales
#   ↑ Subir: más canales, más capacidad de representación, más memoria.
#   ↓ Bajar: menos canales, menos capacidad, menos memoria.

DROPOUT_ENCODER = 0.15  # Dropout para el encoder del generador
#   ↑ Subir: más regularización, menos sobreajuste, pero puede dificultar el aprendizaje.
#   ↓ Bajar: menos regularización, más riesgo de sobreajuste.

DROPOUT_DECODER = 0.1  # Dropout para el decoder del generador
#   ↑ Subir: más regularización en la decodificación.
#   ↓ Bajar: menos regularización.

LEAKY_RELU_SLOPE = 0.2  # Parámetro 'negative_slope' para LeakyReLU
#   ↑ Subir: activaciones negativas menos penalizadas, puede ayudar a evitar el "dying ReLU".
#   ↓ Bajar: activaciones negativas más penalizadas.


    ## DISCRIMINADOR

DROPOUT_DISC = 0.15  # Dropout para el discriminador
#   ↑ Subir: más regularización, discriminador menos propenso a sobreajuste.
#   ↓ Bajar: menos regularización.

#---------------------------------------------------------------------------------------------------------------------------------

# Parámetros de entrenamiento

#---------------------------------------------------------------------------------------------------------------------------------

BATCH_SIZE = 16  # Número de muestras procesadas en cada batch de entrenamiento
#   ↑ Subir: entrenamiento más rápido (si tienes suficiente memoria), gradientes más estables.
#   ↓ Bajar: entrenamiento más lento, gradientes más ruidosos, útil si tienes poca memoria.

EPOCHS = 50  # Número de épocas
#   ↑ Subir: más tiempo de entrenamiento, posibilidad de mejor convergencia.
#   ↓ Bajar: menos tiempo de entrenamiento, posible underfitting.

LAMBDA_PERCEPTUAL = 0.3  # Peso de la perceptual loss en la función de coste total
#   ↑ Subir: la pérdida perceptual tiene más influencia, imágenes más realistas pero menos fieles pixel a pixel.
#   ↓ Bajar: menos influencia de la perceptual loss, más fidelidad pixel a pixel.

LR_G = 0.0004  # Generador: sube para que aprenda más rápido
#   ↑ Subir: el generador aprende más rápido, pero puede ser inestable.
#   ↓ Bajar: aprendizaje más lento, más estable.

LR_D = 0.00005  # Discriminador: baja para que no domine el entrenamiento
#   ↑ Subir: el discriminador aprende más rápido, puede dominar al generador.
#   ↓ Bajar: aprendizaje más lento, más equilibrado.

DATASET_DIR = None  # Ruta al directorio del dataset
#   Modificar: cambia la fuente de datos.

NUM_EJEMPLOS = 1  # Número de ejemplos reales y generados a guardar por epoch
#   ↑ Subir: más ejemplos guardados, más visualización, más espacio en disco.
#   ↓ Bajar: menos ejemplos guardados.

SAVE_FREQ_GRAPHS = 3  # Guardar gráficas cada x epochs
#   ↑ Subir: menos gráficas guardadas.
#   ↓ Bajar: más gráficas guardadas.

SAVE_FREQ_IMAGES = 1  # Guardar imágenes generadas cada x epochs
#   ↑ Subir: menos imágenes guardadas.
#   ↓ Bajar: más imágenes guardadas.

GENERATOR_STEPS = 3  # Número de pasos del generador por cada paso del discriminador
#   ↑ Subir: el generador se entrena más veces por cada paso del discriminador.
#   ↓ Bajar: el discriminador se entrena más veces por cada paso del generador.

#---------------------------------------------------------------------------------------------------------------------------------

# Parámetros de ruido para entrenamiento

#---------------------------------------------------------------------------------------------------------------------------------

NOISE_STD_GAUSSIAN = 0.012         # Desviación estándar para ruido gaussiano.
#   ↑ Subir: añade más ruido gaussiano, imágenes más "borrosas", puede dificultar el aprendizaje si es muy alto.
#   ↓ Bajar: menos ruido, imágenes más limpias, menos robustez ante perturbaciones.

NOISE_AMOUNT_SALT_PEPPER = 0.0015   # Proporción para ruido sal y pimienta.
#   ↑ Subir: más píxeles aleatorios blancos/negros, puede dificultar la tarea del discriminador.
#   ↓ Bajar: menos píxeles alterados, menos robustez ante este tipo de ruido.

NOISE_LOW_UNIFORM = -0.0055         # Límite inferior para ruido uniforme
NOISE_HIGH_UNIFORM = 0.0055         # Límite superior para ruido uniforme
#   ↑ Ampliar rango: más variación aleatoria en los píxeles, más perturbación.
#   ↓ Reducir rango: menos variación, menos efecto del ruido.

NOISE_STD_SPECKLE = 0.005         # Desviación estándar para ruido speckle.
#   ↑ Subir: más interferencia multiplicativa, puede simular ruido de sensores.
#   ↓ Bajar: menos interferencia, imágenes más limpias.

NOISE_KERNEL_BLUR = 2             # Tamaño del kernel para ruido de desenfoque.
#   ↑ Subir: mayor desenfoque, imágenes más borrosas.
#   ↓ Bajar: menor desenfoque, imágenes más nítidas.
