# app-back

Backend para el proyecto de intérprete de lengua de signos en español.

## Estructura del proyecto

- **main.py**: API principal con FastAPI. Endpoint `/procesar_video/` para procesar vídeos y devolver texto o vídeo en dictadología.
- **data/**: Datasets de imágenes de dictadología y transiciones para entrenamiento de modelos.
- **models/**:
	- `generator_model.py`: Modelo generador de imágenes de dictadología (letras).
	- `gan_morph.py`: GAN para generar transiciones (morphing) entre posiciones de la mano.
	- `train/`: Scripts de entrenamiento (`train_gen_model.py`, `train_gan_morph.py`).
	- `validate/`: Scripts de validación de modelos.
- **src/components/**:
	- `audio_extractor.py`: Extrae audio de vídeos.
	- `speech_to_text.py`: Convierte audio a texto.
	- `text_to_images.py`: Convierte texto en imágenes de dictadología.
	- `images_to_video.py`: Une imágenes en un vídeo.
	- `video_to_text.py`: (Futuro) Reconoce texto a partir de vídeo de dictadología.
- **notebooks/**: Experimentos y pruebas en Jupyter.
- **utils/**: Funciones auxiliares.

## Flujo principal
1. El usuario sube un vídeo.
2. Se extrae el audio y se convierte a texto.
3. El texto se transforma en imágenes de dictadología (modelo generador).
4. Se generan transiciones entre letras (GAN morphing).
5. Las imágenes y transiciones se unen en un vídeo final.

## Entrenamiento de modelos
- Ejecuta los scripts en `models/train/` para entrenar y validar los modelos generadores y de morphing.

## Requisitos
- Python 3.8+
- Instalar dependencias:
	```
	pip install torch torchvision fastapi moviepy matplotlib pillow
	```

## Ejecución de la API
Lanza el backend con:
```
uvicorn main:app --reload
```

## Contacto
Para dudas o mejoras, contacta con el autor del repositorio.
