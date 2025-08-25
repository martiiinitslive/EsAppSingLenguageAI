**mejoras en v3.71 -> v3.72**

- Parametrización de hiperparámetros
- Uso de Dropout
- Activación LeakyReLU parametrizada
- Skip connections tipo U-Net

**mejoras en v3.72 -> v3.73**

- Skip connections tipo U-Net - detalles espaciales
- Bloques de Self-Attention añadidos en encoder y decoder - enfocar regiones relevantes
- Pérdida perceptual (VGG) integrada junto con la pérdida pixel a pixel - normalizada 
- mas Parametrización 
- creación del discriinador 
- Entrenamiento GAN alternando generador y discriminador con Adam
- Guardado de generador y discriminador 
- Mejor organización y control en el script de trian_ttd  
- Ajuste automático de canales en encoder/decoder según  hiperparámetros de config_ttd.py
- Validación y normalización de imágenes para la pérdida perceptual en todo el flujo
