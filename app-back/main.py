from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de interpretación de lengua de signos en español"}

# Para lanzar la API, ejecuta en terminal:
# uvicorn main:app --reloadcd