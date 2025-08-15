# INFORMACIÓN GENERAL SOBRE REDES NEURONALES Y MODELOS

## ¿Qué es una red neuronal?
Una red neuronal es un modelo matemático inspirado en el cerebro humano, compuesto por nodos (neuronas) organizados en capas. Cada neurona recibe entradas, las procesa y transmite una salida a la siguiente capa.

## Tipos de redes neuronales
- **Perceptrón simple:** Una sola capa de neuronas, útil para problemas lineales.
- **Redes multicapa (MLP):** Varias capas densas (fully connected), permiten aprender relaciones no lineales.
- **Redes convolucionales (CNN):** Usan capas convolucionales para procesar datos con estructura espacial (imágenes, vídeo). Aprenden patrones locales y espaciales.
- **Redes recurrentes (RNN):** Procesan secuencias de datos (texto, audio, series temporales). Incluyen variantes como LSTM y GRU.
- **Redes generativas adversariales (GAN):** Dos redes (generador y discriminador) compiten para generar datos realistas.
- **Autoencoders:** Aprenden a comprimir y reconstruir datos, útiles para reducción de dimensionalidad y generación.
- **Transformers:** Usan mecanismos de atención para modelar dependencias complejas en secuencias. Son el estándar en NLP y visión avanzada.

## Partes clave de una red/modelo
- **Capa de entrada:** Recibe los datos originales (imágenes, texto, etc.).
- **Embedding:** Representación densa y continua de datos categóricos (por ejemplo, letras o palabras). Permite que el modelo aprenda relaciones entre categorías.
- **Capas ocultas:** Procesan la información, pueden ser densas, convolucionales, recurrentes, etc.
- **Activaciones:** Funciones no lineales (ReLU, Sigmoid, Tanh) que permiten al modelo aprender relaciones complejas.
- **Normalización:** Técnicas como BatchNorm o LayerNorm estabilizan y aceleran el entrenamiento.
- **Dropout:** Técnica para evitar el sobreajuste, desactivando aleatoriamente neuronas durante el entrenamiento.
- **Capa de salida:** Produce el resultado final (clasificación, imagen generada, etc.).

## Configuraciones y parámetros importantes
- **Número de capas y neuronas/canales:** Determina la capacidad del modelo para aprender patrones complejos.
- **Tamaño del embedding:** Afecta la riqueza de la representación de datos categóricos.
- **Función de pérdida:** Mide la diferencia entre la salida del modelo y el objetivo. Ejemplos: MSELoss, CrossEntropyLoss, Perceptual Loss, Adversarial Loss.
- **Optimizador:** Algoritmo que ajusta los pesos del modelo para minimizar la pérdida. Ejemplos: SGD, Adam, RMSprop.
- **Learning rate (tasa de aprendizaje):** Controla la velocidad de ajuste de los pesos.
- **Batch size:** Número de muestras procesadas en cada paso de entrenamiento.
- **Épocas:** Número de veces que el modelo ve todo el dataset durante el entrenamiento.

## Ejemplo de flujo en un modelo generativo de imágenes
1. **Entrada:** Índice de letra (por ejemplo, 'd').
2. **Embedding:** Vector denso que representa la letra.
3. **Proyección inicial:** Capa lineal que expande el embedding.
4. **Mapa de características inicial:** Reshape a un tensor espacial.
5. **Bloques deconvolucionales:** Expanden el mapa hasta el tamaño final de la imagen.
6. **Capa final:** Conv2d + Tanh para obtener la imagen generada normalizada.

## Recomendaciones generales
- Elige la arquitectura según el tipo de datos y el objetivo.
- Ajusta la capacidad del modelo (capas, canales, embedding) según la complejidad del problema y la cantidad de datos.
- Usa funciones de pérdida y optimizadores adecuados.
- Monitoriza la pérdida y los resultados visuales para ajustar hiperparámetros.

# Funciones de pérdida para modelos generativos de imágenes

1. MSELoss (Error Cuadrático Medio)
- Compara los valores de los píxeles entre la imagen generada y la imagen real.
- Es la función de pérdida más simple y común.
- Ventaja: fácil de implementar y rápida de calcular.
- Desventaja: puede producir imágenes borrosas, ya que solo busca que los valores sean similares, no que la imagen tenga detalles perceptuales.

2. Perceptual Loss
- Compara características extraídas por una red neuronal preentrenada (por ejemplo, VGG).
- En vez de comparar solo los píxeles, compara cómo se ven las imágenes en diferentes niveles de abstracción (bordes, texturas, formas).
- Ventaja: el modelo aprende a generar imágenes que “se ven” más parecidas a las reales para el ojo humano.
- Mejora la calidad visual y los detalles.
- Desventaja: requiere más cálculo y una red preentrenada.

3. L1 Loss (Error Absoluto Medio)
- Similar a MSELoss, pero usa la diferencia absoluta en vez de cuadrática.
- Puede producir imágenes menos borrosas que MSELoss.

4. Adversarial Loss (GAN)
- Usada en redes generativas adversariales (GANs).
- El generador intenta engañar a un discriminador, que aprende a distinguir imágenes reales de generadas.
- Permite generar imágenes muy realistas, pero requiere una arquitectura GAN.

5. Combinaciones
- En muchos modelos avanzados se combinan varias funciones de pérdida (por ejemplo, MSE + perceptual + adversarial) para equilibrar fidelidad de píxeles y calidad visual.

¿Por qué el "val loss" sube y baja?
El "val loss" (pérdida de validación) sube y baja porque el modelo está evaluando su capacidad de generalizar sobre datos que no ha visto durante el entrenamiento. Las razones principales de estas oscilaciones son:

1. Tamaño de dataset de validación: Si el conjunto de validación es pequeño, pequeñas variaciones en los datos pueden causar fluctuaciones notables en la pérdida.

2. Aleatoriedad en el entrenamiento: El orden de los datos, el inicializado de pesos y el uso de técnicas como dropout pueden hacer que el modelo tenga resultados variables en cada época.

3. Modelo poco entrenado: En las primeras épocas, el modelo aún está aprendiendo y puede tener dificultades para generalizar, lo que se refleja en una pérdida de validación inestable.

4. Overfitting parcial: Si el modelo empieza a memorizar el conjunto de entrenamiento, la pérdida de validación puede aumentar, pero si luego aprende patrones útiles, puede volver a bajar.

5. Batching: Si la validación se hace por lotes, la composición de cada batch puede afectar la pérdida calculada.

En resumen, estas subidas y bajadas son normales, especialmente en las primeras épocas o con datasets pequeños. Si la tendencia general es descendente y no hay aumentos bruscos sostenidos, el entrenamiento va bien. Si quieres menos oscilación, puedes aumentar el tamaño del conjunto de validación o usar técnicas como el promedio móvil para suavizar la curva.