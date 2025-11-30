import os
# Workaround for OpenMP duplicate-runtime initialization errors (libiomp5md.dll)
# This allows libraries that embed different OpenMP runtimes to run together
# in this process. It's a pragmatic workaround; see notes below.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import uvicorn

if __name__ == "__main__":
    # You can configure host and port via environment variables
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=False  # Set reload=False to avoid reloader subprocess issues in certain environments
    )
