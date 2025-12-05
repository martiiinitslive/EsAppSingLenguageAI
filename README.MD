# EsAppSingLenguageAI
│
├── launcher.py                # Quick-start launcher script

**EsAppSingLenguageAI** is a full-stack application built as a **Feasibility Study** (TFG - Trabajo de Fin de Grado) that combines:

- 🎤 **Speech Recognition** (audio-to-text conversion)
- 📝 **Text Normalization** (format text for sign language)
- 🤖 **Pose Sequence Generation** (text-to-sign language poses)
- 🎬 **MediaPipe Rendering** (poses-to-video with realistic hand animations)
- 🌐 **Web Interface** (React frontend + FastAPI backend)
- 📊 **Benchmarking & Metrics** (performance profiling)

The system outputs high-quality sign language videos with synchronized subtitles, designed for **Spanish Sign Language (LSE)**.

---

## 🏗️ Architecture

```
EsAppSingLenguageAI/
├── app-back/                  # FastAPI backend (Python 3.10+)
│   ├── src/components/        # Core processing modules
│   │   ├── audio_extractor.py     # Extract audio from video/YouTube
│   │   ├── speech_to_text.py      # ASR (speech-to-text)
│   │   ├── format_text_for_renderer.py  # Normalize text
│   │   └── downloader.py      # YouTube video download
│   ├── mp/                    # MediaPipe pose rendering
│   │   ├── run_pose_to_video_mediapipe.py  # Main rendering engine
│   │   ├── mediapipe/         # MediaPipe utilities
│   │   └── pose_cache/        # Cached rendered pose frames
│   ├── main.py                # FastAPI app definition
│   ├── run_api.py             # API server launcher (Uvicorn)
│   ├── bench_runner.py        # Benchmarking utility
│   ├── metrics_logger.py      # Metrics collection schema
│   ├── data/                  # Datasets (ASL poses, videos)
│   └── logs/                  # CSV output files, runtime metrics
│
├── app-front/                 # React frontend
│   ├── src/
│   │   ├── App.js             # Main app component
│   │   └── components/
│   │       └── MainTranslator.js  # UI form (text/audio/video/YouTube)
│   └── package.json
│
├── launcher.py                # Quick-start launcher script
├── requirements.txt           # Python dependencies
└── pytest.ini / test/         # Unit tests
```

---

## 🎯 Key Features

### 1. **Multi-Modal Input Support**
- ✅ Direct text input
- ✅ Audio file upload
- ✅ Video file upload
- ✅ YouTube video URL (automatic download)

### 2. **Text-to-Sign Language Pipeline**
```
Input → Audio Extract → Speech-to-Text → Text Normalization → 
Pose Sequence Generation → MediaPipe Rendering → Output Video
```

### 3. **Advanced Rendering**
- 📹 **Realistic Hand Animation** using MediaPipe landmarks (21 points per hand)
- 🎨 **Skin tone rendering** with depth-based shadowing
- 🖼️ **Picture-in-Picture** overlay (reference video in corner)
- 📝 **Dynamic Subtitles** (ASS format with ffmpeg burning or SRT soft subtitles)
- ⚡ **Frame Caching** (reuse rendered pose frames across requests)

### 4. **Performance Monitoring**
- 📊 **Runtime Metrics** collection (20-column CSV schema)
- 🔍 **Process-Level Monitoring** (CPU/memory usage)
- 🆔 **Request Correlation** (UUID-based request tracking)
- 📈 **Benchmarking Tool** for local/API testing

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+** (with venv)
- **Node.js 16+** (for frontend)
- **FFmpeg** (for video processing and subtitle burning)
- **CUDA 12.1** (optional, for GPU acceleration)

### Backend Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Navigate to backend
cd app-back

# 4. Start the API server
python run_api.py
# Server runs on http://0.0.0.0:8000
```

### Frontend Setup

```bash
# 1. Navigate to frontend
cd app-front

# 2. Install dependencies
npm install

# 3. Start development server
npm start
# Frontend runs on http://localhost:3000
```

---

## 📡 API Endpoints

### `POST /generate_from_text/`
Convert text directly to sign language video.

**Request:**
```json
{
  "text": "Hello world",
  "request_id": "optional-uuid"
}
```

**Response:**
```json
{
  "video_path": "/mp/output_mp/render_*.mp4",
  "duration_seconds": 45.2,
  "text_rendered": "Hello world"
}
```

### `POST /transcribe_youtube/`
Download YouTube video, extract audio, transcribe, and render to sign language.

**Request:**
```json
{
  "url": "https://www.youtube.com/watch?v=...",
  "request_id": "optional-uuid"
}
```

**Response:**
```json
{
  "video_path": "/mp/output_mp/render_*.mp4",
  "transcribed_text": "...",
  "duration_seconds": 120.5
}
```

---

## 🔧 Core Components

### Audio Extractor
- Extracts audio from MP4/video files or YouTube streams
- Returns WAV file for speech recognition

### Speech-to-Text (ASR)
- Converts audio to text using a speech recognition model
- Returns normalized text ready for pose generation

### Text Normalizer
- Removes diacritics (accents)
- Normalizes special characters
- Preserves spaces and punctuation

### Pose Sequence Generator
- Maps text characters to sign language poses
- Builds cumulative text representation
- Handles capitalization rules (first letter, after periods)

### MediaPipe Renderer
- Renders poses as video frames using MediaPipe landmarks
- Supports both **realistic** and **wire** hand styles
- Caches frames for performance (~1s per pose)
- Burns ASS subtitles or embeds SRT

---

## 📊 Benchmarking & Metrics

The project includes comprehensive benchmarking tools to measure performance:

### Benchmark Runner
```bash
# Test with random text (API mode)
python .\app-back\bench_runner.py \
  --mode api \
  --input-mode text \
  --api-base "http://127.0.0.1:8000" \
  --repetitions 10 \
  --min-length 20 \
  --max-length 100 \
  --metrics \
  --timeout 600
```

### Output Files
- `logs/benchmarks.csv` - Aggregated benchmark results
- `logs/benchmarks_summary.csv` - Statistical summary
- `logs/runtime_metrics.csv` - Detailed runtime metrics (20 columns)

### Metrics Schema (20 columns)
```
timestamp_utc, request_id, endpoint, input_type, input_length_seconds,
text_length_chars, n_letters_rendered, success, error_stage,
error_message_short, output_video_filename, output_video_duration_seconds,
n_poses_rendered, t_total, t_download, t_extract_audio, t_asr,
t_text_normalisation, t_pose_sequence, t_render
```

---

## ⚙️ Configuration

### Environment Variables
```bash
# API Server
API_HOST=0.0.0.0          # Server host
API_PORT=8000             # Server port

# Processing
KMP_DUPLICATE_LIB_OK=TRUE # OpenMP compatibility
```

### Render Settings (in `mp/run_pose_to_video_mediapipe.py`)
```python
SPEED_FACTOR = 1.0                    # Video playback speed (1.0 = normal)
FRAMES_PER_POSE = 60                  # Frames per character (higher = slower)
PUNCT_FRAMES_PER_POSE = 15            # Frames for punctuation
SUBTITLE_FONT_SIZE = 74               # ASS subtitle font size
DRAW_SUBTITLES_INLINE = False         # Draw subtitles on frames (False = ffmpeg)
HAND_RENDER_STYLE = "realistic"       # "realistic" or "wire"
```

---

## 🧪 Testing

### Unit Tests
```bash
pytest test/ -v
```

### Manual Testing
```bash
# Local command mode (test helper)
python .\app-back\bench_runner.py \
  --mode local \
  --input-mode text \
  --repetitions 1 \
  --min-length 5 \
  --max-length 5
```

---

## 📈 Performance Characteristics

- **Text Processing:** ~0.5s (ASR + normalization)
- **Pose Rendering:** ~1s per character (with caching)
- **Video Encoding:** ~2-3s per minute of video
- **Total Pipeline:** 50-100 char text = ~2-3 minutes

**Memory Usage:**
- Backend server: ~500 MB baseline
- Per-request: +100-300 MB (depending on video length)
- Cache overhead: ~50 MB per unique pose

---

## 📦 Dependencies

### Key Python Libraries
- **FastAPI** - Web framework
- **Uvicorn** - ASGI server
- **OpenCV (cv2)** - Video processing
- **MediaPipe** - Hand landmark detection
- **NumPy** - Numerical computations
- **librosa** - Audio processing
- **moviepy** - Video manipulation
- **yt-dlp** - YouTube video download
- **PyTorch** - Deep learning (if using advanced ASR)

### Frontend
- **React 18+** - UI framework
- **Node.js/npm** - Package management

---

## 🎬 Workflow Example

### Text-to-Sign Video (5-step process)

1. **User Input**
   ```
   Input: "Hola mundo"
   ```

2. **Text Normalization**
   ```
   Output: "Hola mundo" → ['H', 'o', 'l', 'a', 'SPACE', 'm', 'u', 'n', 'd', 'o']
   ```

3. **Pose Lookup**
   ```
   Each character maps to a sign language pose (cached or from dataset)
   ```

4. **Frame Rendering**
   ```
   Per pose: extract landmarks → draw on canvas → add subtitles → write frame
   ~60 frames per character = 600 frames total
   ```

5. **Video Output**
   ```
   Output: render_1764963007.mp4 (10s duration, 30fps, 1920x1080)
   With subtitles and picture-in-picture overlay
   ```

---

## 🔍 Project Structure Details

### `app-back/src/components/`
| File | Purpose |
|------|---------|
| `audio_extractor.py` | Extract WAV from MP4/YouTube |
| `speech_to_text.py` | Convert audio to text (ASR) |
| `format_text_for_renderer.py` | Normalize text for poses |
| `downloader.py` | Download YouTube videos |

### `app-back/mp/`
| Directory | Purpose |
|-----------|---------|
| `mediapipe/` | MediaPipe utilities & JSON conversion |
| `pose_cache/` | Pre-rendered frame cache |
| `output_mp/` | Generated video output |

### `app-back/data/`
- **ASL Dataset** - MediaPipe poses for Spanish Sign Language letters
- **Videos** - Reference videos for each sign

---

## 🐛 Troubleshooting

### "ffmpeg not found"
```bash
# Install ffmpeg
# Windows: Download from https://ffmpeg.org/download.html
# Or use: choco install ffmpeg
# Or use: winget install FFmpeg
```

### "OpenMP duplicate runtime" Error
✅ Already handled via `KMP_DUPLICATE_LIB_OK=TRUE` in `run_api.py`

### API Returns 500 Error
Check backend logs for:
- Missing dataset files in `data/dataset-en-bruto/`
- Video encoding issues (ffmpeg path)
- Insufficient memory for pose rendering

### Benchmark Timeout
Increase `--timeout` value:
```bash
# For YouTube videos (90+ characters)
--timeout 1800  # 30 minutes
```

---

## 📝 License & Attribution

This project is part of a **Feasibility Study (TFG)** for **Spanish Sign Language AI Translation**.

**Built with:**
- MediaPipe (Google) - Pose estimation
- FFmpeg - Video processing
- FastAPI - Backend framework
- React - Frontend UI

---

## 📞 Contact & Support

For issues, questions, or contributions:
- Check existing GitHub issues
- Run benchmarks to identify performance bottlenecks

---

## 🎓 Research & References

This project implements a **feasibility study** for automated sign language video generation. Key research areas:

- Pose estimation accuracy (MediaPipe vs alternatives)
- Hand animation realism
- Text-to-pose sequence mapping
- Real-time rendering performance
- ASR accuracy for Spanish

---

**Last Updated:** December 2025  
**Version:** 1.0.0  
**Status:** Active Development (Feasibility Study)