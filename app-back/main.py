from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import importlib.util
import time
from pathlib import Path
from src.components.audio_extractor import extract_audio_from_video
from src.components.speech_to_text import speech_to_text
from src.components import format_text_for_renderer
from src.components.downloader import download_youtube
# Future: model integration (TextToDictaModel)
# Future: images_to_video integration

app = FastAPI()

# Allow CORS for local development (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

@app.get("/")
def read_root():
    return {"message": "Welcome to the Spanish sign-language interpretation API"}

def _render_from_text(text: str):
    """Helper: normalize text, import renderer and run it. Returns (video_path, download_url)."""
    BASE_DIR_LOCAL = os.path.abspath(os.path.dirname(__file__))
    mp_script = Path(BASE_DIR_LOCAL) / "mp" / "run_pose_to_video_mediapipe.py"
    json_path = Path(BASE_DIR_LOCAL) / "mp" / "poses_mediapipe_video.json"

    if not mp_script.exists() or not json_path.exists():
        raise RuntimeError("Renderer or poses JSON missing")

    # Dynamic import
    spec = importlib.util.spec_from_file_location("mp_renderer", str(mp_script))
    mp_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mp_mod)

    # Normalize text for renderer
    safe_text = format_text_for_renderer.normalize_text_for_renderer(text)
    seq = mp_mod.text_to_pose_sequence(safe_text)
    if not seq:
        raise RuntimeError("Could not convert text to pose sequence")

    out_dir = Path(BASE_DIR_LOCAL) / "mp" / "output_mp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"render_{int(time.time())}.mp4")

    video_path = mp_mod.render_sequence_from_json(str(json_path), seq, out_path=out_path, show=False, save=True)
    filename = Path(video_path).name
    download_url = f"/download_video/{filename}"
    return str(video_path), download_url


@app.post("/procesar_video/")
@app.post("/process_video/")
async def process_video(file: UploadFile = File(...)):
    temp_dir = os.path.join(BASE_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    video_path = os.path.join(temp_dir, f"temp_{file.filename}")
    # Save the uploaded video to a temporary file
    with open(video_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        # Extract audio
        audio_path = video_path + ".wav"
        extract_audio_from_video(video_path, audio_path)

        # Convert audio to text
        text = speech_to_text(audio_path)
    finally:
        # Remove uploaded video (do not keep uploaded video)
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception:
            pass

    if not text:
        raise HTTPException(status_code=500, detail="Could not transcribe uploaded video")

    # Clean up audio file as well
    try:
        if os.path.exists(audio_path):
            os.remove(audio_path)
    except Exception:
        pass

    # Render the output video from the transcribed text
    try:
        video_path_out, download_url = _render_from_text(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rendering failed: {e}")

    return {"video_path": video_path_out, "download_url": download_url}


@app.post("/transcribe_youtube/")
async def transcribe_youtube(payload: dict):
    url = payload.get("url") or payload.get("link")
    if not url:
        raise HTTPException(status_code=400, detail="Missing 'url' in payload")

    temp_dir = os.path.join(BASE_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Download video
        downloaded = download_youtube(url, temp_dir)
        # Extract audio
        audio_path = str(Path(downloaded).with_suffix('.wav'))
        extract_audio_from_video(downloaded, audio_path)
        # Transcribe
        text = speech_to_text(audio_path)
    except Exception as e:
        # Cleanup any partial files
        try:
            if 'downloaded' in locals() and os.path.exists(downloaded):
                os.remove(downloaded)
        except Exception:
            pass
        try:
            if 'audio_path' in locals() and os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Failed processing YouTube URL: {e}")
    finally:
        # Always remove downloaded video to avoid storage accumulation
        try:
            if 'downloaded' in locals() and os.path.exists(downloaded):
                os.remove(downloaded)
        except Exception:
            pass

    if not text:
        raise HTTPException(status_code=500, detail="Could not transcribe YouTube video")

    # Remove audio file
    try:
        if os.path.exists(audio_path):
            os.remove(audio_path)
    except Exception:
        pass

    try:
        video_path_out, download_url = _render_from_text(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rendering failed: {e}")

    return {"video_path": video_path_out, "download_url": download_url}


# Endpoint: generate a pose-video from input text and return path/download URL
@app.post("/generate_from_text/")
async def generate_from_text(payload: dict):
    text = payload.get("text") or payload.get("texto")
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' in payload")

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    mp_script = Path(BASE_DIR) / "mp" / "run_pose_to_video_mediapipe.py"
    json_path = Path(BASE_DIR) / "mp" / "poses_mediapipe_video.json"

    if not mp_script.exists():
        raise HTTPException(status_code=500, detail=f"Renderer script not found: {mp_script}")
    if not json_path.exists():
        raise HTTPException(status_code=500, detail=f"Poses JSON not found: {json_path}")

    # Dynamically import the renderer module so we can call its functions
    try:
        spec = importlib.util.spec_from_file_location("mp_renderer", str(mp_script))
        mp_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mp_mod)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load renderer: {e}")

    # Build pose sequence and output path
    try:
        seq = mp_mod.text_to_pose_sequence(text)
    except Exception:
        seq = None

    if not seq:
        raise HTTPException(status_code=400, detail="Could not convert text to a pose sequence")

    out_dir = Path(BASE_DIR) / "mp" / "output_mp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"render_{int(time.time())}.mp4")

    try:
        video_path = mp_mod.render_sequence_from_json(str(json_path), seq, out_path=out_path, show=False, save=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rendering failed: {e}")

    # Provide a download endpoint URL for the front-end to fetch the generated video
    filename = Path(video_path).name
    download_url = f"/download_video/{filename}"
    return {"video_path": str(video_path), "download_url": download_url}


@app.get("/download_video/{filename}")
def download_video(filename: str):
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    p = Path(BASE_DIR) / "mp" / "output_mp" / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(p), media_type="video/mp4", filename=filename)