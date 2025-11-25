import os
import sys
import json
import argparse
import subprocess
import shutil
from pathlib import Path
try:
    import cv2
except Exception:
    cv2 = None  # type: ignore
    # cv2 (OpenCV) is required at runtime; if missing the script will exit with an instructive message.
import time
import numpy as np

# Simple visualization pipeline that mirrors "visualize_mediapipe_json_video.py"
# but accepts an ordered `--poses` string (e.g. "A,B,C") and renders a single
# video composed by those poses. It draws landmarks (from the JSON) on the
# main canvas and overlays a reference image/video frame in the bottom-right
# corner (Picture-in-Picture). Text subtitles are applied via ffmpeg in post-production.


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_NAME = "run_pose_to_video_mediapipe.mp4"

# ==========  SPEED_FACTOR ==========
# SPEED_FACTOR: Controla la velocidad GLOBAL del vídeo (fps de salida)
#   - Factor aplicado a los fps base (por defecto 30)
#   - SPEED_FACTOR = 3.0 → fps_out = 30 * 3.0 = 90 fps (vídeo 3x más rápido)
#   - SPEED_FACTOR = 0.5 → fps_out = 30 * 0.5 = 15 fps (vídeo 0.5x más lento)
#   - Usado principalmente para acelerar/ralentizar todo el vídeo
#   - Uso: python script.py --speed 2.0

SPEED_FACTOR = 1.0

#   - SPEED_FACTOR es un MULTIPLICADOR de fps (global)
#   - FRAMES_PER_POSE es la DURACIÓN en frames de cada pose (local)

# ========== PARAMETRIZACIÓN DE ENTRADA ==========
# Variable parametrizable para el texto de entrada.
# Formatos soportados:
#   - Letras (A-Z): se convierten a MAYÚSCULAS → ["H", "E", "L", "L", "O"]
#   - Números (0-9): se mantienen → ["1", "2", "3"]
#   - Espacios " ": se convierten a pose "SPACE"
#   - Puntos ".": se convierten a pose "PERIOD"
#   - Comas ",": se convierten a pose "COMMA"
# Ejemplos:
#   INPUT_TEXT = "HELLO"          # → ["H", "E", "L", "L", "O"]
#   INPUT_TEXT = "HELLO WORLD"    # → ["H", "E", "L", "L", "O", "SPACE", "W", "O", "R", "L", "D"]
#   INPUT_TEXT = "HI."            # → ["H", "I", "PERIOD"]
#   INPUT_TEXT = None             # → Usa --text o --poses de línea de comandos
INPUT_TEXT = "hOla. como estas? yo muy bien primo, juan"  # Cambia a "HELLO", "HI THERE", etc. para usar texto fijo

# ========== PARAMETRIZACIÓN DE VELOCIDAD ==========
# Controla cuántos frames dura cada letra en el vídeo
# Valores más altos = vídeo más lento (más tiempo en cada letra)
# Valores más bajos = vídeo más rápido
# Ejemplos:
#   FRAMES_PER_POSE = 10   # Muy rápido
#   FRAMES_PER_POSE = 30   # Normal (por defecto)
#   FRAMES_PER_POSE = 60   # Lento (2x más tiempo)
#   FRAMES_PER_POSE = 120  # Muy lento (4x más tiempo)
FRAMES_PER_POSE = 60  # Ajusta este valor para ralentizar/acelerar
# ===============================================

# ========== PUNCTUATION SPEED OVERRIDE ==========
# Número de frames a usar para tokens de puntuación/espacio (más rápido que
# una pose normal). Puedes ajustar este valor al inicio del script.
# Por defecto usamos un fractionario de FRAMES_PER_POSE para mantener proporción
# si el usuario cambia FRAMES_PER_POSE más abajo.
PUNCT_FRAMES_PER_POSE = max(1, int(FRAMES_PER_POSE * 0.25))  # por defecto 25% de FRAMES_PER_POSE
# ===============================================

# ========== SUBTITLES CONFIGURATION ==========
# Ajustes accesibles para controlar comportamiento de subtítulos
SUBTITLE_MAX_LINE_CHARS = 35  # número máximo de caracteres por línea de subtítulo
SUBTITLE_FONT_NAME = "Arial"
SUBTITLE_FONT_SIZE = 74  # tamaño de fuente para ASS
SUBTITLE_BG_ALPHA = 0.45  # alfa para fondo semi-opaco en dibujo inline
SUBTITLE_ALIGN = 2  # 2 = Bottom-Center en ASS (subtítulos centrados abajo)
SUBTITLE_MARGIN = 10  # márgenes en ASS
# ==============================================

# Si True, dibuja subtítulos directamente en cada frame (inline) en lugar de
# generar .ass y quemarlos con ffmpeg. Esto evita problemas de formatos y
# sincronización con ffmpeg y reproductores.
DRAW_SUBTITLES_INLINE = False

# Conexiones estándar de MediaPipe para manos (21 landmarks)
# Muestra la estructura ósea de la mano
HAND_CONNECTIONS = [
    # Palma
    (0, 1), (1, 2), (2, 3), (3, 4),      # Pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),      # Índice
    (5, 9), (9, 10), (10, 11), (11, 12), # Dedo medio
    (9, 13), (13, 14), (14, 15), (15, 16), # Anular
    (13, 17), (17, 18), (18, 19), (19, 20), # Meñique
    (0, 17), (0, 13), (0, 9), (0, 5)     # Conexiones desde muñeca a base de dedos
]

# Grosor de las líneas que conectan landmarks (ajustable)
CONNECTION_LINE_THICKNESS = 10

def progress(s):
    print(f"[{time.strftime('%H:%M:%S')}][MP_VIS] {s}")

def ensure_dir(p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)


def find_ffmpeg():
    """Locate ffmpeg executable in PATH or common Windows locations.
    Returns the path as string or raises RuntimeError if not found.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
    common_paths = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
    ]
    for path in common_paths:
        if Path(path).exists():
            return str(path)
    raise RuntimeError("ffmpeg no encontrado")


def cleanup_sidecars(ass_path, keep_sidecars=False):
    """Remove ASS and SRT sidecar files unless `keep_sidecars` is True."""
    try:
        ass_p = Path(ass_path)
        srt_p = ass_p.with_suffix('.srt')
        if not keep_sidecars:
            if ass_p.exists():
                ass_p.unlink()
            if srt_p.exists():
                srt_p.unlink()
        return True
    except Exception:
        return False

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _get_size(meta, args):
    w = args.width or None
    h = args.height or None
    if not w and meta:
        den = meta.get("denormalization", {}) or {}
        w = int(den.get("image_width") or den.get("width") or 1920)
        h = int(den.get("image_height") or den.get("height") or 1080)
    if not w: w = 1920
    if not h: h = 1080
    return int(w), int(h)


def _draw_landmarks_on_img(img, landmarks, color=(0,180,255), connections=None, radius=12):
    """
    Dibuja landmarks en la imagen con alta visibilidad estilo mano humana.
    - Se dibuja un círculo interior de color brillante con borde oscuro
    - Las líneas de conexión son visibles para mostrar estructura ósea
    - Usa colores naturales: carne + puntos azul-verdosos como tendones
    """
    h, w = img.shape[:2]
    pts = []
    for lm in landmarks:
        if isinstance(lm, dict):
            x = lm.get("x", lm.get("X", None))
            y = lm.get("y", lm.get("Y", None))
            z = lm.get("z", lm.get("Z", None))
        elif isinstance(lm, (list, tuple)):
            if len(lm) >= 2:
                x, y = lm[0], lm[1]
                z = lm[2] if len(lm) > 2 else 0
            else:
                continue
        else:
            continue
        if x is None or y is None:
            continue
        px = int(round(float(x) * (w - 1)))
        py = int(round(float(y) * (h - 1)))
        pts.append((px, py, float(z) if z is not None else 0.0))
    
    # Dibujar conexiones (tendones/huesos) primero
    if connections:
        for a, b in connections:
            if a < len(pts) and b < len(pts):
                # Líneas oscuras para representar tendones
                cv2.line(img, (pts[a][0], pts[a][1]), (pts[b][0], pts[b][1]), (100, 120, 140), CONNECTION_LINE_THICKNESS, cv2.LINE_AA)
    
    # Dibujar puntos de articulación (landmarks)
    for (px, py, z) in pts:
        # Borde oscuro exterior (sombra)
        cv2.circle(img, (px, py), radius + 1, (20, 20, 20), -1, cv2.LINE_AA)
        # Color principal (azul-verdoso como venas/tendones)
        cv2.circle(img, (px, py), radius - 1, color, -1, cv2.LINE_AA)
        # Brillo interior (punto blanco para efecto 3D)
        cv2.circle(img, (px, py), max(2, radius // 3), (255, 255, 255), -1, cv2.LINE_AA)


def extract_frame_landmarks(frame_obj):
    if frame_obj is None:
        return None
    if isinstance(frame_obj, dict):
        for k in ("multi_hand_landmarks", "hands", "hand_landmarks", "multi_handedness"):
            if k in frame_obj and frame_obj[k]:
                return frame_obj[k]
        for k in ("landmarks", "landmark", "pose_landmarks", "pose_world_landmarks", "world_landmarks"):
            if k in frame_obj and frame_obj[k]:
                return frame_obj[k]
        if "bones" in frame_obj and isinstance(frame_obj["bones"], dict):
            pts = []
            for k, v in frame_obj["bones"].items():
                if isinstance(v, dict) and "loc" in v:
                    loc = v.get("loc")
                    if loc and len(loc) >= 2:
                        pts.append({"x": loc[0], "y": loc[1], "z": loc[2] if len(loc) > 2 else 0.0})
            if pts:
                return pts
    return None

def _load_ref_image(path, target_w, target_h):
    canvas = 255 * np.ones((target_h, target_w, 3), dtype="uint8")
    if not path:
        return canvas
    p = Path(path)
    if not p.exists():
        return canvas
    im = cv2.imread(str(p))
    if im is None:
        return canvas
    ih, iw = im.shape[:2]
    scale = min(target_w / iw, target_h / ih)
    nw = max(1, int(iw * scale))
    nh = max(1, int(ih * scale))
    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
    x = (target_w - nw) // 2
    y = (target_h - nh) // 2
    canvas[y:y+nh, x:x+nw] = im_resized
    return canvas

def _extract_frame_from_video(video_path, target_w, target_h):
    canvas = 255 * np.ones((target_h, target_w, 3), dtype="uint8")
    if not video_path:
        return canvas
    v = Path(video_path)
    if not v.exists():
        return canvas
    try:
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            return canvas
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            ret, frame = cap.read()
        else:
            mid = max(0, total // 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return canvas
        ih, iw = frame.shape[:2]
        scale = min(target_w / iw, target_h / ih)
        nw = max(1, int(iw * scale))
        nh = max(1, int(ih * scale))
        frame_resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        x = (target_w - nw) // 2
        y = (target_h - nh) // 2
        canvas[y:y+nh, x:x+nw] = frame_resized
        return canvas
    except Exception as e:
        progress(f"    warning: failed to extract from video {video_path}: {e}")
        return canvas

def _seq_from_poses_string(poses_str):
    return [s.strip() for s in poses_str.split(",") if s.strip()]


def pose_to_char(pose_name: str) -> str:
    """Map pose identifiers to their display character.
    - 'SPACE' -> ' '
    - 'PERIOD' -> '.'
    - 'COMMA' -> ','
    - otherwise: if single-character string, keep it; else use first character
    """
    if pose_name is None:
        return ""
    if isinstance(pose_name, str):
        if pose_name.upper() == "SPACE":
            return " "
        if pose_name.upper() == "PERIOD":
            return "."
        if pose_name.upper() == "COMMA":
            return ","
        # if it's a single char like 'A' or '?' keep it
        if len(pose_name) == 1:
            return pose_name
        # otherwise try to take first visible char
        return pose_name[0]
    return str(pose_name)

def _format_ass_time(sec):
    """
    Convierte segundos a formato ASS (h:mm:ss.cs donde cs = centisegundos).
    Ejemplo: 123.456 -> "0:02:03.45"
    """
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _draw_inline_subtitle(img, text, font_scale=None, thickness=None, margin_v=40):
    """Draw a centered subtitle with outline and semi-transparent background.

    This function now respects `SUBTITLE_FONT_SIZE` to compute an OpenCV
    `font_scale` and `thickness` so inline rendering visually matches the
    ASS font size configured by the user.
    """
    if not text:
        return
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Map ASS font size to OpenCV font_scale using the historic default
    # mapping: SUBTITLE_FONT_SIZE == 24 -> font_scale ~= 1.6
    try:
        base_scale = 1.6
        base_ass = 24.0
        computed_scale = (SUBTITLE_FONT_SIZE / base_ass) * base_scale
        font_scale = font_scale if font_scale is not None else max(0.1, float(computed_scale))
    except Exception:
        font_scale = font_scale if font_scale is not None else 1.6

    # Thickness proportional to font size for consistent stroke
    thickness = thickness if thickness is not None else max(1, int(round(SUBTITLE_FONT_SIZE * 0.12)))

    # Measure text size and center horizontally; position vertically near bottom
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = (w - text_w) // 2
    y = h - margin_v
    # Draw a thin semi-opaque bar under the text (underline effect)
    pad_x = max(8, int(SUBTITLE_FONT_SIZE * 0.5))
    rect_w = text_w + pad_x * 2
    rect_h = max(8, int(text_h * 0.35))
    # position the underline slightly below baseline
    rect_x1 = x - pad_x
    rect_y1 = y + 6
    rect_x2 = rect_x1 + rect_w
    rect_y2 = rect_y1 + rect_h
    overlay = img.copy()
    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
    alpha = SUBTITLE_BG_ALPHA
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    # Draw outline and text on top of the underline
    cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _estimate_text_width(text: str, font_scale=1.6, thickness=3) -> int:
    """Estimate pixel width of `text` using OpenCV if available, otherwise fallback to heuristic."""
    # Prefer Pillow measurement with Arial 48 to match ASS rendering
    try:
        from PIL import ImageFont
        # try common Windows Arial path and use SUBTITLE_FONT_SIZE as basis
        arial_paths = [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\Arial.ttf",
        ]
        font = None
        # Map ASS font size to a reasonable pixel size for PIL. Historically
        # we used 48px for SUBTITLE_FONT_SIZE 24, so use factor 2x.
        pil_px_size = max(8, int(round(SUBTITLE_FONT_SIZE * 2)))
        for p in arial_paths:
            try:
                font = ImageFont.truetype(p, pil_px_size)
                break
            except Exception:
                font = None
        if font is None:
            # fallback to a default PIL font
            font = ImageFont.load_default()
        size = font.getsize(text)
        return int(size[0])
    except Exception:
        pass

    try:
        if cv2 is None:
            raise RuntimeError
        (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        return int(w)
    except Exception:
        # fallback: assume average 18 px per character at default sizes
        return max(1, int(len(text) * 18))


def generate_ass_subtitles(sequence, fps_out, frames_per_pose, ass_path, start_frame=1, write_ass=True):
    """
    Genera archivo ASS con subtítulos que se actualizan con cada pose.
    Replica exactamente la lógica de generate_ass_for_cumulative_letters de run_pose_to_video_from_static.py
    """
    # Include PlayRes resolution if provided via hints so ASS font sizing matches video
    play_res_x = getattr(generate_ass_subtitles, "img_width", None) or 1920
    play_res_y = getattr(generate_ass_subtitles, "img_height", None) or 1080
    header = f"[Script Info]\nScriptType: v4.00+\nPlayResX: {int(play_res_x)}\nPlayResY: {int(play_res_y)}\n\n[V4+ Styles]\n"
    header += f"Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, " \
              f"Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, " \
              f"Alignment, MarginL, MarginR, MarginV, Encoding\n"
    # Use subtitle configuration constants for style. Set outline+shadow for legibility
    # BackColour uses alpha to create a semi-opaque box behind text (ASS BackColour).
    header += f"Style: Default,{SUBTITLE_FONT_NAME},{SUBTITLE_FONT_SIZE},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,2,2,1,{SUBTITLE_ALIGN},{SUBTITLE_MARGIN},{SUBTITLE_MARGIN},{SUBTITLE_MARGIN},1\n\n"
    header += "Events:\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"

    # Enforce a maximum number of characters per subtitle line and avoid splitting words.
    MAX_LINE_CHARS = getattr(generate_ass_subtitles, "max_line_chars", SUBTITLE_MAX_LINE_CHARS)
    lines = []
    events = []  # list of (start_sec, end_sec, text) for SRT output

    # Convert sequence of pose identifiers into display characters
    chars = [pose_to_char(p) for p in sequence]
    # normalize frames_per_pose into a list matching length
    if isinstance(frames_per_pose, (list, tuple)):
        frames_list = [int(max(0, int(x))) for x in frames_per_pose]
        if len(frames_list) < len(chars):
            frames_list += [FRAMES_PER_POSE] * (len(chars) - len(frames_list))
    else:
        frames_list = [int(max(0, int(frames_per_pose)))] * len(chars)

    n = len(chars)

    # Build tokens: words (sequences of non-space characters) and space tokens
    tokens = []  # each token: {'text': str, 'indices': [pose_indices]}
    i = 0
    while i < n:
        ch = chars[i]
        if ch == " ":
            tokens.append({"text": " ", "indices": [i]})
            i += 1
        else:
            start = i
            tchars = []
            while i < n and chars[i] != " ":
                tchars.append(chars[i])
                i += 1
            token_text = "".join(tchars)
            indices = list(range(start, start + len(tchars)))
            tokens.append({"text": token_text, "indices": indices})

    # Map each pose index to the display text at that pose
    display_for_index = [""] * n
    current_line = ""

    for token in tokens:
        tok_text = token["text"]
        idxs = token["indices"]

        # compute candidate if we append this token
        if tok_text == " ":
            candidate = current_line + " "
        else:
            candidate = current_line + ("" if current_line == "" else " ") + tok_text

        # if candidate would overflow, start a new line before this token
        if len(candidate) > MAX_LINE_CHARS:
            # start new line: if token is a space, skip leading space
            if tok_text == " ":
                current_line = ""
                for idx in idxs:
                    display_for_index[idx] = current_line
            else:
                # new line starts with this token progressively
                current_line = ""
                for k, idx in enumerate(idxs):
                    # reveal token progressively
                    part = tok_text[: k + 1]
                    display_for_index[idx] = part
                # after finishing token, set current_line to full token
                current_line = tok_text
        else:
            # append token to current line; reveal progressively for token indices
            if tok_text == " ":
                # the space token: append single space and assign display
                current_line = candidate
                for idx in idxs:
                    display_for_index[idx] = current_line
            else:
                # token is a word; reveal progressively
                # determine prefix part for partial reveals
                prefix = current_line + ("" if current_line == "" else " ")
                for k, idx in enumerate(idxs):
                    part = prefix + tok_text[: k + 1]
                    display_for_index[idx] = part
                # after finishing token, update current_line
                current_line = candidate

    # Now write dialogues using display_for_index and frames_list
    # Precompute frame start indices for timing
    frame_starts = []
    f = start_frame
    for fr in frames_list:
        frame_starts.append(f)
        f += int(fr)

    for idx in range(n):
        frames_for_this = frames_list[idx]
        start_sec = (frame_starts[idx] - 1) / float(fps_out)
        end_sec = (frame_starts[idx] + frames_for_this - 1) / float(fps_out)
        raw = (display_for_index[idx] or "")
        # For ASS we must escape backslashes first, then commas as field separators.
        # This avoids malformed escape sequences being interpreted or displayed.
        # Escape backslashes, commas and braces for ASS
        text_for_ass = raw.replace("\\", "\\\\").replace("\n", " ").replace(",",",")
        text_for_ass = text_for_ass.replace("{", r"\{").replace("}", r"\}")
        lines.append(f"Dialogue: 0,{_format_ass_time(start_sec)},{_format_ass_time(end_sec)},Default,,0,0,0,,{text_for_ass}\n")
        # SRT is plain text, keep commas unchanged
        events.append((start_sec, end_sec, raw.replace("\n", " ")))

    # Write output files: ASS (if requested) and always SRT
    srt_path = Path(ass_path).with_suffix(".srt")
    try:
        if write_ass:
            ensure_dir(ass_path)
            with open(ass_path, "w", encoding="utf-8") as f:
                f.write(header)
                for line in lines:
                    f.write(line)
            progress(f"Generated ASS subtitles: {ass_path}")
        # Always write SRT (used for soft subtitles)
        ensure_dir(srt_path)
        with open(srt_path, "w", encoding="utf-8") as srtf:
            for idx, (ssec, esec, txt) in enumerate(events, start=1):
                # formatear a 00:00:00,000
                def _format_srt_time(sec):
                    h = int(sec // 3600)
                    m = int((sec % 3600) // 60)
                    s = int(sec % 60)
                    ms = int(round((sec - int(sec)) * 1000))
                    return f"{h:d}:{m:02d}:{s:02d},{ms:03d}"
                srtf.write(f"{idx}\n")
                srtf.write(f"{_format_srt_time(ssec)} --> {_format_srt_time(esec)}\n")
                srtf.write(f"{txt}\n\n")
        progress(f"Generated SRT subtitles: {srt_path}")
    except Exception as e:
        progress(f"[WARN] Could not write subtitle files: {e}")

def apply_ass_subtitles_with_ffmpeg(input_video, ass_path, output_video):
    """
    Usa ffmpeg para quemar los subtítulos ASS en el vídeo.
    Replica la lógica de burn_ass_with_ffmpeg de run_pose_to_video_from_static.py
    """
    # Locate ffmpeg
    try:
        ffmpeg_path = find_ffmpeg()
    except Exception:
        progress("[ERROR] ffmpeg no encontrado.")
        raise
    
    # Validar archivos
    ass_p = Path(ass_path)
    if not ass_p.exists():
        progress(f"[ERROR] Archivo ASS no encontrado: {ass_path}")
        raise RuntimeError(f"ASS no encontrado: {ass_path}")

    input_p = Path(input_video)
    if not input_p.exists():
        progress(f"[ERROR] Archivo de entrada no encontrado: {input_video}")
        raise RuntimeError(f"Input no encontrado: {input_video}")
    
    # Usar forward slashes y escapar ':' y comillas simples
    ass_posix = ass_p.as_posix()
    # escape colon and single quote for ffmpeg subtitles filter
    escaped = ass_posix.replace(":", r"\:").replace("'", r"\'")
    # Use the subtitles filter string
    vf_arg = f"subtitles='{escaped}'"
    
    progress(f"[DEBUG] ffmpeg_path: {ffmpeg_path}")
    progress(f"[DEBUG] ass_path (posix): {ass_posix}")
    progress(f"[DEBUG] vf_arg: {vf_arg}")
    
    cmd = [ffmpeg_path, "-y", "-i", str(input_p), "-vf", vf_arg, "-c:a", "copy", str(output_video)]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    
    if result.stdout:
        progress(f"[DEBUG] ffmpeg output:\n{result.stdout[-500:]}")  # Últimos 500 chars
    
    if result.returncode != 0:
        progress(f"[ERROR] ffmpeg falló con código {result.returncode}")
        raise RuntimeError(f"ffmpeg falló: {result.stdout}")
    
    output_p = Path(output_video)
    if not output_p.exists() or output_p.stat().st_size == 0:
        progress(f"[ERROR] ffmpeg no produjo salida válida: {output_video}")
        raise RuntimeError(f"ffmpeg no produjo salida válida")
    
    return str(output_video)


def embed_srt_with_ffmpeg(input_video, srt_path, output_video, language="eng"):
    """
    Embed an SRT file as a soft subtitle track (mov_text) into the MP4 container
    without re-encoding the video/audio streams.

    Example ffmpeg command used:
      ffmpeg -i input.mp4 -i subtitle.en.srt -c copy -c:s mov_text -metadata:s:s:0 language=eng output.mp4
    """
    # Locate ffmpeg
    try:
        ffmpeg_path = find_ffmpeg()
    except Exception:
        progress("[ERROR] ffmpeg no encontrado (para embed SRT).")
        raise

    srt_p = Path(srt_path)
    if not srt_p.exists():
        progress(f"[ERROR] Archivo SRT no encontrado: {srt_path}")
        raise RuntimeError(f"SRT no encontrado: {srt_path}")

    input_p = Path(input_video)
    if not input_p.exists():
        progress(f"[ERROR] Archivo de entrada no encontrado: {input_video}")
        raise RuntimeError(f"Input no encontrado: {input_video}")

    # Build command: copy video/audio, convert subtitle to mov_text and set language
    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(input_p),
        "-i",
        str(srt_p),
        "-c",
        "copy",
        "-c:s",
        "mov_text",
        "-metadata:s:s:0",
        f"language={language}",
        str(output_video),
    ]

    progress(f"[DEBUG] Ejecutando embed SRT: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    progress(f"[DEBUG] ffmpeg returncode: {result.returncode}")
    if result.stdout:
        progress(f"[DEBUG] ffmpeg output (tail):\n{result.stdout[-500:]}")

    if result.returncode != 0:
        progress(f"[ERROR] ffmpeg embed SRT falló con código {result.returncode}")
        raise RuntimeError(f"ffmpeg falló: {result.stdout}")

    output_p = Path(output_video)
    if not output_p.exists() or output_p.stat().st_size == 0:
        progress(f"[ERROR] ffmpeg no produjo salida válida al embed SRT: {output_video}")
        raise RuntimeError("ffmpeg no produjo salida válida")

    progress(f"[OK] SRT embedded correctamente: {output_video}")
    return str(output_video)

def text_to_pose_sequence(text_input):
    """
    Convierte un texto en una secuencia de poses (caracteres).
    Formato esperado del input text:
      - Cada carácter es una pose individual
      - Espacios " " → pose "SPACE" (opcional, se puede ignorar)
      - Puntos "." → pose "PERIOD" (opcional, al final marca fin de palabra)
      - Mayúsculas: se tratarán como minúsculas excepto la primera letra
    
    Ejemplos:
      - "HELLO" → ["H", "E", "L", "L", "O"]
      - "Hola mundo" → ["H", "O", "L", "A", "SPACE", "M", "U", "N", "D", "O"]
      - "Hi." → ["H", "I", "PERIOD"]
    """
    sequence = []
    prev_non_space = None
    for char in text_input:
        if char == " ":
            sequence.append("SPACE")
            # do not change prev_non_space
            continue
        if char == ".":
            sequence.append("PERIOD")
            prev_non_space = "."
            continue
        if char == ",":
            sequence.append("COMMA")
            prev_non_space = ","
            continue

        # For alphabetic characters apply capitalization rules based on previous
        # meaningful character: first meaningful alpha -> uppercase; after '.' -> uppercase;
        # otherwise lowercase. Non-alpha characters are preserved as-is.
        if isinstance(char, str) and char.isalpha():
            if prev_non_space is None or prev_non_space == ".":
                pose_char = char.upper()
            else:
                pose_char = char.lower()
            sequence.append(pose_char)
            prev_non_space = pose_char
        else:
            # preserve other characters (digits, symbols)
            sequence.append(char)
            prev_non_space = char

    return sequence

def render_sequence_from_json(json_path, sequence, out_path=None, show=True, save=True, fps=None, width=None, height=None, speed_factor=None, apply_subtitles=True, soft_subtitles=False, subtitle_lang="eng", keep_sidecars=False):
    progress(f"Loading JSON: {json_path}")
    data = load_json(json_path)
    meta = data.get("meta", {}) if isinstance(data, dict) else {}
    poses = data.get("poses", {}) if isinstance(data, dict) else {}

    w, h = _get_size(meta, argparse.Namespace(width=width, height=height))
    fps_use = fps or meta.get("denormalization", {}).get("fps") or 30
    try:
        fps_use = int(fps_use)
    except Exception:
        fps_use = 30
    speed_factor = float(speed_factor) if speed_factor is not None else float(SPEED_FACTOR)
    fps_out = max(1, int(round(fps_use * float(speed_factor))))
    progress(f"Resolved size {w}x{h}, source fps {fps_use}, speed_factor {speed_factor} -> writer fps {fps_out}")

    if out_path is None:
        out_path = str(OUTPUT_DIR / OUTPUT_NAME)
    ensure_dir(out_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Tamaño del output: solo ancho original (no concatenado)
    out_w = w
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_out, (out_w, h))
    if not writer.isOpened():
        raise RuntimeError("No se pudo abrir VideoWriter (comprueba codecs)")
    
    # Configuración de Picture-in-Picture (esquina inferior derecha)
    pip_width = int(w * 0.25)  # 25% del ancho original
    pip_height = int(h * 0.25)  # 25% del alto original
    pip_margin = 15  # Margen desde las esquinas

    base_app_back = Path(__file__).resolve().parent.parent.parent
    default_ref_dir = base_app_back / "data" / "dataset-en-bruto" / "asl_dataset" / "mediapipe_poses"
    default_videos_dir = base_app_back / "data" / "dataset-en-bruto" / "asl_dataset" / "videos" / "videos_de_las_letras"
    video_exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")

    total_frames = 0
    # collect actual frames written per pose to generate accurate subtitle timing
    frames_counts = []
    cumulative = ""
    for idx, pose_name in enumerate(sequence, start=1):
        progress(f"[{idx}/{len(sequence)}] Rendering pose '{pose_name}'")
        pose_entry = poses.get(pose_name) or poses.get(pose_name.lower()) or poses.get(pose_name.upper())

        # find reference image or video
        video_candidate = None
        possible_folder_names = [pose_name, pose_name.lower(), pose_name.upper()]
        for fname in possible_folder_names:
            folder_candidate = default_videos_dir / fname
            if folder_candidate.exists() and folder_candidate.is_dir():
                for file in sorted(folder_candidate.iterdir()):
                    if file.suffix.lower() in video_exts:
                        video_candidate = str(file); break
            if video_candidate: break

        if video_candidate:
            ref_img = _extract_frame_from_video(video_candidate, w, h)
        else:
            src_img_path = None
            if isinstance(pose_entry, dict):
                src_img_path = pose_entry.get("source_image")
            if not src_img_path:
                for name in possible_folder_names:
                    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
                        candidate = default_ref_dir / (name + ext)
                        if candidate.exists():
                            src_img_path = str(candidate); break
                    if src_img_path: break
            ref_img = _load_ref_image(src_img_path, w, h)

        frames_list = pose_entry.get("frames") if isinstance(pose_entry, dict) else None

        # update cumulative text using pose->char mapping and same capitalization rules
        c = pose_to_char(pose_name)
        # Capitalization rule: first letter of the whole text or letter after a
        # period should be uppercase; other alphabetic letters should be lowercase.
        # Non-alphabetic characters are preserved as-is.
        # find last non-space char in cumulative (if any)
        prev_non_space = None
        for ch in reversed(cumulative):
            if ch and ch != " ":
                prev_non_space = ch
                break

        # Determine if we're at the start of the meaningful text (ignore leading spaces)
        at_start = (cumulative.strip() == "")

        # Capitalization rule:
        # - If this is the first meaningful alphabetic character (at_start) -> UPPER
        # - If previous meaningful character was a period -> UPPER
        # - Otherwise alphabetic -> lower
        if c.isalpha():
            if at_start or prev_non_space == ".":
                dc = c.upper()
            else:
                dc = c.lower()
        else:
            dc = c

        cumulative = cumulative + dc

        pose_frame_count = 0
        if frames_list and isinstance(frames_list, list) and len(frames_list) > 0:
            for fi, frame_obj in enumerate(frames_list):
                canvas = 255 * np.ones((h, w, 3), dtype="uint8")

                ldata = extract_frame_landmarks(frame_obj)
                if ldata:
                    if isinstance(ldata, list) and len(ldata) > 0 and isinstance(ldata[0], list):
                        for hand in ldata:
                            _draw_landmarks_on_img(canvas, hand, color=(0,180,255), connections=HAND_CONNECTIONS, radius=12)
                    else:
                        _draw_landmarks_on_img(canvas, ldata, color=(0,180,255), connections=HAND_CONNECTIONS, radius=12)
                else:
                    cv2.putText(canvas, "no landmarks", (40, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30,30,30), 2, cv2.LINE_AA)

                # Agregar PiP en esquina inferior derecha
                ref_img_resized = cv2.resize(ref_img, (pip_width, pip_height), interpolation=cv2.INTER_AREA)
                x_pos = w - pip_width - pip_margin
                y_pos = h - pip_height - pip_margin
                canvas[y_pos:y_pos+pip_height, x_pos:x_pos+pip_width] = ref_img_resized
                # Dibujar borde alrededor del PiP
                cv2.rectangle(canvas, (x_pos-2, y_pos-2), (x_pos+pip_width+2, y_pos+pip_height+2), (200,200,200), 2)
                # draw inline subtitle if requested
                if DRAW_SUBTITLES_INLINE:
                    _draw_inline_subtitle(canvas, cumulative)
                writer.write(canvas)
                total_frames += 1
                pose_frame_count += 1
        else:
            # Use a shorter duration for punctuation and special-character tokens
            # so they appear faster than a normal pose. Criteria:
            #  - explicit tokens 'SPACE','PERIOD','COMMA' -> fast
            #  - if there is no pose entry (we generated visualization from a char)
            #    and the character is a single non-alphanumeric (special char),
            #    treat it as fast. Note: str.isalpha() returns True for accented
            #    letters, so accented letters are treated as normal (NOT fast).
            if isinstance(pose_name, str):
                if pose_name.upper() in ("SPACE", "PERIOD", "COMMA"):
                    duration_iter = PUNCT_FRAMES_PER_POSE
                else:
                    # If the renderer didn't find a pose entry in the JSON
                    # (pose_entry is None) and this is a single character that
                    # is NOT an alphabetic (including accented) nor a digit,
                    # treat it as a special char and use the punct duration.
                    if (pose_entry is None) and len(pose_name) == 1 and not (pose_name.isalpha() or pose_name.isdigit()):
                        duration_iter = PUNCT_FRAMES_PER_POSE
                    else:
                        duration_iter = FRAMES_PER_POSE
            else:
                duration_iter = FRAMES_PER_POSE

            for _ in range(duration_iter):
                canvas = 255 * np.ones((h, w, 3), dtype="uint8")
                cv2.putText(canvas, f"{pose_name} (no frames)", (40, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (10,10,10), 3, cv2.LINE_AA)
                # Agregar PiP en esquina inferior derecha
                ref_img_resized = cv2.resize(ref_img, (pip_width, pip_height), interpolation=cv2.INTER_AREA)
                x_pos = w - pip_width - pip_margin
                y_pos = h - pip_height - pip_margin
                canvas[y_pos:y_pos+pip_height, x_pos:x_pos+pip_width] = ref_img_resized
                # Dibujar borde alrededor del PiP
                cv2.rectangle(canvas, (x_pos-2, y_pos-2), (x_pos+pip_width+2, y_pos+pip_height+2), (200,200,200), 2)
                if DRAW_SUBTITLES_INLINE:
                    _draw_inline_subtitle(canvas, cumulative)
                writer.write(canvas)
                total_frames += 1
                pose_frame_count += 1
        # record how many frames this pose produced (append for both branches)
        frames_counts.append(pose_frame_count)
    writer.release()
    progress(f"Writer released. Total frames written: {total_frames}")
    
    # Generar subtítulos ASS y aplicarlos con ffmpeg (si no dibujamos inline)
    if apply_subtitles:
        try:
            if DRAW_SUBTITLES_INLINE:
                progress("[SUBTITLES] DRAW_SUBTITLES_INLINE enabled -> subtitles already drawn on frames. Skipping ASS/ffmpeg.")
            else:
                out_path_obj = Path(out_path)
                ass_path = out_path_obj.with_suffix(".ass")
                progress(f"[SUBTITLES] Generando archivo ASS: {ass_path}")
                progress(f"[SUBTITLES] frames_counts: {frames_counts} (total_frames={total_frames})")
                # Generar ASS usando las duraciones reales por pose
                # pass image width and pip_x to generate_ass_subtitles so it can avoid PiP overlap
                try:
                    setattr(generate_ass_subtitles, "img_width", w)
                    pip_left = w - pip_width - pip_margin
                    setattr(generate_ass_subtitles, "pip_x", pip_left)
                except Exception:
                    pass

                # Generate sidecars: if we will embed soft subtitles, avoid creating ASS
                write_ass = not soft_subtitles
                generated_path = generate_ass_subtitles(sequence, fps_out, frames_counts, str(ass_path), start_frame=1, write_ass=write_ass)
                progress(f"[SUBTITLES] Sidecars generados: {generated_path}")

                # Decide embedding method: soft subtitles (mov_text) or burn ASS
                if soft_subtitles:
                    progress(f"[SUBTITLES] Embedding soft subtitles (SRT) into container (language={subtitle_lang})...")
                    temp_output = out_path_obj.with_stem(out_path_obj.stem + "_temp")
                    srt_path = ass_path.with_suffix('.srt')
                    try:
                        result = embed_srt_with_ffmpeg(out_path, str(srt_path), str(temp_output), language=subtitle_lang)
                    except Exception as e:
                        progress(f"[SUBTITLES] ✗ Embed SRT failed: {e}")
                        result = None

                    if result and temp_output.exists():
                        out_path_obj.unlink()
                        temp_output.rename(out_path)
                        # remove sidecars unless user asked to keep them
                        cleanup_sidecars(ass_path, keep_sidecars)
                        progress(f"[SUBTITLES] ✓ Video final con subtítulos (soft): {out_path}")
                    else:
                        progress(f"[SUBTITLES] ✗ No se pudieron embeber subtítulos; se mantiene el video sin cambios")
                else:
                    # Hard burn ASS into the video (existing path)
                    progress(f"[SUBTITLES] Quemando subtítulos ASS en el video...")
                    temp_output = out_path_obj.with_stem(out_path_obj.stem + "_temp")
                    try:
                        result = apply_ass_subtitles_with_ffmpeg(out_path, str(ass_path), str(temp_output))
                    except Exception as e:
                        progress(f"[SUBTITLES] ✗ Burning ASS failed: {e}")
                        result = None

                    if result and temp_output.exists():
                        out_path_obj.unlink()  # remove original
                        temp_output.rename(out_path)
                        # remove sidecars unless user asked to keep them
                        cleanup_sidecars(ass_path, keep_sidecars)
                        progress(f"[SUBTITLES] - [OK] Video final con subtítulos: {out_path}")
                    else:
                        progress(f"[SUBTITLES] ✗ No se pudieron aplicar subtítulos, pero se mantiene el video sin ellos")
        except Exception as e:
            progress(f"[SUBTITLES] Excepción: {e}")
            import traceback
            progress(traceback.format_exc())
    
    return out_path

def main():
    default_json = str(Path(__file__).resolve().parent / "poses_mediapipe_video.json")
    default_out = str(OUTPUT_DIR / OUTPUT_NAME)
    ap = argparse.ArgumentParser(description="Render poses sequence into annotated video using MediaPipe-style JSON (no Blender)")
    ap.add_argument("--json", "-j", default=default_json, help="ruta al JSON de poses")
    ap.add_argument("--poses", "-p", default=None, help="cadena de poses separadas por comas (ej: A,B,C). Si no se proporciona, usar --text")
    ap.add_argument("--text", "-t", default=None, help="texto a convertir en poses (ej: 'HELLO' → ['H','E','L','L','O']). Soporta espacios y puntos")
    ap.add_argument("--out", "-o", default=default_out, help="ruta de salida mp4")
    ap.add_argument("--fps", type=int, default=None, help="fps entrada/base (se multiplica por speed)")
    ap.add_argument("--width", type=int, default=None, help="ancho salida (por pose)")
    ap.add_argument("--height", type=int, default=None, help="alto salida")
    ap.add_argument("--speed", "-s", type=float, default=SPEED_FACTOR, help="factor de velocidad de reproducción (>1 acelera)")
    ap.add_argument("--soft-subtitles", dest="soft_subtitles", action="store_true", help="Embed generated SRT as soft subtitles (mov_text) instead of burning ASS")
    ap.add_argument("--subtitle-lang", dest="subtitle_lang", default="eng", help="ISO-639-2 language code for embedded subtitle track (default: eng)")
    ap.add_argument("--keep-sidecars", dest="keep_sidecars", action="store_true", help="Keep generated .ass/.srt sidecar files after embedding/burning")
    args = ap.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # determinar la secuencia de poses: desde INPUT_TEXT, luego --text, luego --poses
    if INPUT_TEXT is not None:
        progress(f"Using INPUT_TEXT variable: '{INPUT_TEXT}'")
        seq = text_to_pose_sequence(INPUT_TEXT)
        progress(f"Sequence from INPUT_TEXT: {seq}")
    elif args.text:
        progress(f"Converting text to poses: '{args.text}'")
        seq = text_to_pose_sequence(args.text)
        progress(f"Sequence from text: {seq}")
    elif args.poses:
        seq = _seq_from_poses_string(args.poses)
    else:
        # valor por defecto si ninguno se proporciona
        seq = _seq_from_poses_string("A")
    
    if not seq:
        print("No hay poses en la secuencia. Usar --text 'HELLO' o --poses 'A,B' por ejemplo.")
        sys.exit(1)
    # Llamar al renderer con los parámetros recibidos
    progress("Starting render...")
    out = render_sequence_from_json(args.json, seq, out_path=args.out, show=False, save=True,
                                   fps=args.fps, width=args.width, height=args.height, speed_factor=args.speed,
                                   apply_subtitles=True, soft_subtitles=args.soft_subtitles, subtitle_lang=args.subtitle_lang, keep_sidecars=args.keep_sidecars)
    progress(f"Render finished. Output: {out}")
    return out

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        progress(f"[FATAL] Unhandled exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)