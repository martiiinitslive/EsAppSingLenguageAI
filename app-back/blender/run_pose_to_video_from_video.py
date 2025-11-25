import os
import sys
import shlex
import shutil
import subprocess
import re
from pathlib import Path

# ========== CONFIG (ajusta según tu entorno) ==========

BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
BASE_DIR = Path(__file__).resolve().parent

BLEND_FILE   = str(BASE_DIR / "cuerpo_humano_rigged.blend")
SCRIPT_PATH  = str(BASE_DIR / "text_to_video_from_poses_video.py")
POSES_JSON   = str(BASE_DIR / "poses_converted_video.json")
CAMERA_JSON  = str(BASE_DIR / "camera_library.json")
ARMATURE     = "Human.rig"
POSES        = "A"            # formato 'A,B,C' o cadena vacía para usar --text
FPS          = 60
FRAME_STEP   = 1              # procesar cada N frames del vídeo (1 = todos)
ENGINE       = "EEVEE"
OUT_PATH     = str(BASE_DIR / "output" / "out_video_from_video.mp4")
WIDTH        = 1920
HEIGHT       = 1080
CAMERA_NAME  = "Cam_01"
FFMPEG_PATH  = r"C:\ffmpeg\bin\ffmpeg.exe"
ADD_SUBTITLES = True
SUB_FONT = "Arial"
SUB_FONT_SIZE = 48
SUB_MARGIN_V = 40
SKIP_DEFOCUS = False
POSE_DURATION = 0   # segundos por pose (None o 0 para no comprimir)
# ======================================================

LOG_PREFIX = "[PIPELINE]"

def log(msg):
    print(f"{LOG_PREFIX} {msg}")

def require_file(path, description, exit_on_missing=True):
    p = Path(path)
    if not p.exists():
        msg = f"❌ No se encontró {description}: {path}"
        if exit_on_missing:
            log(msg); sys.exit(1)
        else:
            log(msg); return False
    return True

def _atempo_chain_factors(speed):
    target = 1.0 / float(speed)
    factors = []
    t = target
    while t > 2.0 + 1e-9:
        factors.append(2.0); t /= 2.0
    while t < 0.5 - 1e-9:
        factors.append(0.5); t /= 0.5
    factors.append(max(1e-6, t))
    return factors

def adjust_video_speed_ffmpeg(src_path, speed, ffmpeg_exec=FFMPEG_PATH):
    src = Path(src_path)
    if not src.exists():
        raise RuntimeError(f"Archivo de entrada no encontrado: {src}")
    exec_path = ffmpeg_exec
    if ffmpeg_exec == "ffmpeg":
        exec_path = shutil.which("ffmpeg") or ""
    if not exec_path or not Path(exec_path).exists():
        raise RuntimeError(f"ffmpeg no encontrado: {ffmpeg_exec}")

    suffix = str(speed).replace(".", "p")
    out = src.with_name(f"{src.stem}_speed{suffix}{src.suffix}")
    setpts = f"{speed}*PTS"
    atempo_chain = ",".join(f"atempo={f:.8f}" for f in _atempo_chain_factors(speed))

    cmd = [
        str(exec_path), "-y", "-i", str(src),
        "-filter_complex", f"[0:v]setpts={setpts}[v];[0:a]{atempo_chain}[a]",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        str(out)
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg falló (rc={proc.returncode}):\n{proc.stdout}")
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg no produjo salida válida: {out}")
    return str(out)

def run_blender_pipeline(out_path=None):
    args = [
        BLENDER_PATH,
        BLEND_FILE,
        "--background",
        "--python", SCRIPT_PATH,
        "--",
        "--library", POSES_JSON,
        "--poses", POSES,
        "--armature", ARMATURE,
        "--fps", str(FPS),
        "--engine", ENGINE,
        "--width", str(WIDTH),
        "--height", str(HEIGHT),
        "--camera_lib", CAMERA_JSON,
        "--camera_name", CAMERA_NAME,
        "--frame_step", str(FRAME_STEP)
    ]
    if SKIP_DEFOCUS:
        args.append("--skip_defocus")
    if POSE_DURATION and POSE_DURATION > 0:
        args += ["--pose_duration", str(POSE_DURATION)]
    if out_path:
        args += ["--out", out_path]
    log("Running Blender: " + " ".join(shlex.quote(a) for a in args))
    subprocess.run(args, check=True)

def _seq_from_poses_string(poses_str):
    return [s.strip() for s in poses_str.split(",") if s.strip()]

def _format_ass_time(sec):
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"

def generate_ass_for_cumulative_letters(sequence, fps, hold_frames, transition_frames, ass_path, start_frame=1, per_pose_frame_counts=None):
    """
    Si per_pose_frame_counts se proporciona (lista de frames por pose) usa esos valores
    para calcular la duración de cada pose; en caso contrario se usa hold/transition
    como antes (pero evitar duraciones cero).
    """
    header = "[Script Info]\nScriptType: v4.00+\n\n[V4+ Styles]\n"
    header += f"Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, " \
              f"Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, " \
              f"Alignment, MarginL, MarginR, MarginV, Encoding\n"
    header += f"Style: Default,{SUB_FONT},{SUB_FONT_SIZE},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,0,0,{SUB_MARGIN_V},1\n\n"
    header += "Events:\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"

    lines = []
    frame = start_frame
    cumulative = ""
    for idx, pose_name in enumerate(sequence):
        c = pose_name[0] if pose_name else ""
        c = str(c)
        if cumulative == "":
            cumulative = c.upper()
        else:
            stripped = cumulative.rstrip()
            if stripped.endswith("."):
                cumulative = cumulative + " " + c.upper()
            else:
                cumulative = cumulative + c.lower()

        # duración en frames para esta pose:
        if per_pose_frame_counts and idx < len(per_pose_frame_counts):
            dur_frames = max(1, int(per_pose_frame_counts[idx]))
            start_sec = (frame - 1) / float(fps)
            end_sec = (frame + dur_frames - 1) / float(fps)
            next_frame = frame + dur_frames
        else:
            # fallback: usar hold+transition (asegurar al menos 1 frame)
            frame_hold_end = frame + max(0, hold_frames)
            next_frame = frame_hold_end + max(0, transition_frames)
            if next_frame <= frame:
                next_frame = frame + 1
            start_sec = (frame - 1) / float(fps)
            end_sec = (next_frame - 1) / float(fps)

        text = cumulative.replace("\n", " ").replace(",", "\\,")
        lines.append(f"Dialogue: 0,{_format_ass_time(start_sec)},{_format_ass_time(end_sec)},Default,,0,0,0,,{text}\n")
        frame = next_frame

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header)
        for l in lines:
            f.write(l)
    return ass_path

def burn_ass_with_ffmpeg(input_video, ass_path, output_video, ffmpeg_exec):
    from pathlib import Path
    ass_p = Path(ass_path)
    if not ass_p.exists():
        raise RuntimeError(f"ASS no encontrado: {ass_path}")
    ass_posix = ass_p.as_posix()
    escaped = ass_posix.replace(":", r"\:").replace("'", r"\'")
    vf_arg = f"subtitles='{escaped}'"
    cmd = [ffmpeg_exec, "-y", "-i", str(input_video), "-vf", vf_arg, "-c:a", "copy", str(output_video)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg ass burn falló (rc={proc.returncode}):\n{proc.stdout}")
    return str(output_video)

def main(out_path=None):
    log("Iniciando pipeline (video-based poses)...")
    require_file(BLENDER_PATH, "Blender")
    require_file(BLEND_FILE, ".blend")
    require_file(SCRIPT_PATH, "script interno de Blender")
    if not require_file(POSES_JSON, "poses_converted_video.json", exit_on_missing=False):
        log("Continuando sin poses JSON completo (posibles fallbacks).")
    if Path(CAMERA_JSON).exists():
        log(f"Usando camera lib: {CAMERA_JSON}")
    else:
        log("Aviso: no se encontró camera_library.json (se usará la cámara de la escena).")

    # si no se pasó out_path, generar nombre basado en la cadena procesada (POSES)
    if out_path is None:
        seq_str = POSES or "out"
        # normalizar: reemplazar comas/espacios por guión bajo y quitar caracteres inválidos
        seq_norm = seq_str.replace(",", "_").replace(" ", "_")
        safe_name = re.sub(r'[^A-Za-z0-9_\-\.]', '_', seq_norm).strip("_")
        out_path = str(Path(BASE_DIR) / "output" / f"{safe_name}.mp4")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    run_blender_pipeline(out_path)

    out_file = Path(out_path)
    if not out_file.exists() or out_file.stat().st_size == 0:
        log(f"Error: Blender finalizó sin generar un vídeo válido en: {out_path}")
        sys.exit(1)
    log(f"✅ Vídeo generado: {out_path}")

    if ADD_SUBTITLES:
        try:
            ff = FFMPEG_PATH if FFMPEG_PATH != "ffmpeg" else shutil.which("ffmpeg")
            if not ff or not Path(ff).exists():
                log("⚠️ ffmpeg no encontrado. Se omite generación de subtítulos quemados.")
            else:
                seq = _seq_from_poses_string(POSES)
                # intentar cargar duraciones reales desde el JSON (total_frames / aplicar frame_step)
                per_pose_frames = []
                try:
                    import json as _json
                    pj = _json.load(open(POSES_JSON, "r", encoding="utf-8"))
                    for p in seq:
                        ent = pj.get("poses", {}).get(p, {})
                        tf = ent.get("total_frames") or ent.get("video_info", {}).get("total_frames") or 0
                        if FRAME_STEP and FRAME_STEP > 1:
                            tf = int((tf + (FRAME_STEP - 1)) // FRAME_STEP)
                        per_pose_frames.append(int(tf) if tf else 1)
                except Exception as e:
                    log(f"⚠️ No se pudieron leer duraciones desde JSON para subtítulos: {e}")
                    per_pose_frames = None

                ass_path = Path(out_file.parent) / (out_file.stem + ".ass")
                generate_ass_for_cumulative_letters(seq, FPS, 0, 0, str(ass_path), start_frame=1, per_pose_frame_counts=per_pose_frames)
                temp_out = out_file.with_name(out_file.stem + "_subs" + out_file.suffix)
                log(f"Quemando subtítulos ASS -> {ass_path} en {out_path}")
                burn_ass_with_ffmpeg(out_file, str(ass_path), temp_out, ff)
                temp_out.replace(out_file)
                log(f"Subtítulos aplicados: {out_file}")
        except Exception as e:
            log(f"Error aplicando subtítulos: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"{LOG_PREFIX} ⚠️ Error inesperado: {e}")
        raise