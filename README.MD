# EsAppSingLenguageAI

Repositorio para el proyecto de interpretación de lengua de signos en español mediante IA.

## Descripción general

Este proyecto permite convertir vídeos hablados en texto y en dictadología (letras del alfabeto en lengua de signos española), utilizando modelos de inteligencia artificial y procesamiento multimedia.

Incluye dos aplicaciones principales:

- **Backend (app-back):** API con FastAPI que procesa vídeos, extrae audio, convierte audio a texto y genera imágenes/vídeos de dictadología. Contiene módulos para entrenamiento de modelos generativos y de morphing (transiciones entre letras).
- **Frontend (app-front):** Aplicación web en React para interactuar con el sistema, subir vídeos y visualizar resultados.

## Flujo principal

1. El usuario sube un vídeo desde el frontend.
2. El backend extrae el audio y lo convierte a texto.
3. El texto se transforma en imágenes de dictadología usando modelos generativos.
4. Se generan transiciones suaves entre letras (GAN morphing).
5. Las imágenes y transiciones se unen en un vídeo final.

## Estructura del repositorio

- **app-back/**: Backend con FastAPI, módulos de procesamiento y modelos de IA.
- **app-front/**: Frontend en React para la interfaz de usuario.
- **data/**: Datasets de dictadología y morphing para entrenamiento.
- **models/**: Modelos generativos y scripts de entrenamiento/validación.
- **notebooks/**: Guías y experimentos en Jupyter.
- **utils/**: Funciones auxiliares.

## Requisitos

- Python 3.8+ (para backend)
- Node.js y npm (para frontend)
- Instalar dependencias según los README de cada carpeta.

## Ejecución rápida

1. **Backend:**  
    ```bash
    cd app-back
    python run_api.py
    ```
2. **Frontend:**  
    ```bash
    cd app-front
    npm install
    npm start
    ```

## Contacto

Para dudas o sugerencias, consulta los README específicos o contacta con el autor del repositorio.