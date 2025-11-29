import uvicorn
import os

if __name__ == "__main__":
    # You can configure host and port via environment variables
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=False  # Set reload=False to avoid reloader subprocess issues in certain environments
    )
