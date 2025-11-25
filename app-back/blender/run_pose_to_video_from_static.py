# run_pose_video.py
'''
para ejecutar desde terminal/PowerShell y generar un vídeo automáticamente:

& C:/Users/marti/AppData/Local/Programs/Python/Python313/python.exe c:/Users/marti/Desktop/tfg_teleco/proyectos/EsAppSingLenguageAI/pruebas_blender/run_pose_video.py  

'''
import os
import sys
import shlex
import shutil
import subprocess
from pathlib import Path

# ========== CONFIG ==========
BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
BASE_DIR = Path(__file__).resolve().parent

BLEND_FILE   = str(BASE_DIR / "cuerpo_humano_rigged.blend")
SCRIPT_PATH  = str(BASE_DIR / "text_to_video_from_poses.py")
POSES_JSON   = str(BASE_DIR / "poses_library.json")
CAMERA_JSON  = str(BASE_DIR / "camera_library.json")
ARMATURE     = "Human.rig"
POSES        = "H,O,L,A" #formato ha de ser 'H,O,L,A'
FPS          = 24
HOLD         = 12
TRANSITION   = 12
ENGINE       = "EEVEE"
OUT_PATH     = str(BASE_DIR / "output" / "HOLA.mp4")
WIDTH        = 1920   # ancho en px (cambiar para hacer más rectangular)
HEIGHT       = 1080   # alto en px
CAMERA_NAME  = "Cam_01"
FFMPEG_PATH  = r"C:\ffmpeg\bin\ffmpeg.exe"
SPEED        = 2.5  # factor final de velocidad (ej. 2.5)
ADD_SUBTITLES = True     # True = generar subtítulos; False = no
SUB_FONT = "Arial"
SUB_FONT_SIZE = 48       # ajustar para resolución (48 es razonable para 1920x1080)
SUB_MARGIN_V = 40        # margen inferior en píxeles
# ============================

LOG_PREFIX = "[PIPELINE]"

def log(msg):
    print(f"{LOG_PREFIX} {msg}")

def require_file(path, description, exit_on_missing=True):
    p = Path(path)
    if not p.exists():
        msg = f"❌ No se encontró {description}: {path}"
        if exit_on_missing:
            log(msg)
            sys.exit(1)
        else:
            log(msg)
            return False
    return True

# Helper to break speed factor into atempo chain accepted by ffmpeg
def _atempo_chain_factors(speed):
    # we will change audio speed by 1/speed (to match video setpts)
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

    # si tiene audio, usamos filter_complex para mantener vídeo+audio sincronizados
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
        "--hold", str(HOLD),
        "--transition", str(TRANSITION),
        "--engine", ENGINE,
        "--width", str(WIDTH),
        "--height", str(HEIGHT),
        "--camera_lib", CAMERA_JSON,
        "--camera_name", CAMERA_NAME
    ]
    if out_path:
        args += ["--out", out_path]
    log("Running Blender: " + " ".join(shlex.quote(a) for a in args))
    subprocess.run(args, check=True)

def _seq_from_poses_string(poses_str):
    return [s.strip() for s in poses_str.split(",") if s.strip()]

def _format_ass_time(sec):
    # ASS times: H:MM:SS.cs  (centiseconds)
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"

def generate_ass_for_cumulative_letters(sequence, fps, hold_frames, transition_frames, ass_path, start_frame=1):
    """
    Crea un archivo .ass con líneas que muestran el texto acumulado por cada pose,
    centrado abajo. Cada entrada dura desde la aplicación de la pose hasta el inicio
    de la siguiente (simula la lógica del script de poses).
    """
    # header + estilo (alineación bottom-center -> Alignment=2)
    header = "[Script Info]\nScriptType: v4.00+\n\n[V4+ Styles]\n"
    header += f"Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, " \
              f"Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, " \
              f"Alignment, MarginL, MarginR, MarginV, Encoding\n"
    header += f"Style: Default,{SUB_FONT},{SUB_FONT_SIZE},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,0,0,{SUB_MARGIN_V},1\n\n"
    header += "Events:\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"

    lines = []
    frame = start_frame
    cumulative = ""
    # simulate same frame stepping as in Blender script
    for idx, pose_name in enumerate(sequence):
        # derive a single-char representation:
        c = pose_name[0] if pose_name else ""
        c = str(c)
        # apply capitalization rules when appending:
        if cumulative == "":
            cumulative = c.upper()
        else:
            # if last non-space char is '.' then ensure space + capital
            stripped = cumulative.rstrip()
            if stripped.endswith("."):
                cumulative = cumulative + " " + c.upper()
            else:
                cumulative = cumulative + c.lower()
        # compute start and end times
        start_sec = (frame - 1) / float(fps)
        frame_hold_end = frame + max(0, hold_frames)
        next_frame = frame_hold_end + max(0, transition_frames)
        # for last element we extend until next_frame (or +hold) — consistent with Blender script
        end_sec = (next_frame - 1) / float(fps)
        # ASS requires escaped commas and special characters; keep plain letters safe
        text = cumulative.replace("\n", " ").replace(",", "\\,")
        lines.append(f"Dialogue: 0,{_format_ass_time(start_sec)},{_format_ass_time(end_sec)},Default,,0,0,0,,{text}\n")
        frame = next_frame

    # write file
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header)
        for l in lines:
            f.write(l)
    return ass_path

def burn_ass_with_ffmpeg(input_video, ass_path, output_video, ffmpeg_exec):
    """
    Quema subtítulos ASS sobre input_video con ffmpeg.
    Escapa ':' y comillas para que ffmpeg no interprete la ruta como opciones.
    """
    from pathlib import Path
    ass_p = Path(ass_path)
    if not ass_p.exists():
        raise RuntimeError(f"ASS no encontrado: {ass_path}")

    # Representación con forward slashes (más segura) y escape de ':' y "'"
    ass_posix = ass_p.as_posix()
    escaped = ass_posix.replace(":", r"\:").replace("'", r"\'")

    # usar el filtro 'subtitles' (libass) y envolver en comillas simples
    vf_arg = f"subtitles='{escaped}'"

    cmd = [ffmpeg_exec, "-y", "-i", str(input_video), "-vf", vf_arg, "-c:a", "copy", str(output_video)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg ass burn falló (rc={proc.returncode}):\n{proc.stdout}")
    return str(output_video)

def main(out_path=None):
    log("Iniciando pipeline automatizado para generación de vídeo...")
    require_file(BLENDER_PATH, "Blender")
    require_file(BLEND_FILE, ".blend")
    require_file(SCRIPT_PATH, "script interno de Blender")
    if not require_file(POSES_JSON, "poses_library.json", exit_on_missing=False):
        log("Continuando sin poses_library.json (se usará lo que haya en el .blend).")
    if Path(CAMERA_JSON).exists():
        log(f"Usando camera lib: {CAMERA_JSON}")
    else:
        log("Aviso: no se encontró camera_library.json (se usará la cámara de la escena).")

    # usar el out_path recibido o la constante OUT_PATH
    out_path = out_path if out_path is not None else OUT_PATH
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    run_blender_pipeline(out_path)

    out_file = Path(out_path)
    if not out_file.exists() or out_file.stat().st_size == 0:
        log(f"Error: Blender finalizó sin generar un vídeo válido en: {out_path}")
        sys.exit(1)
    log(f"✅ Vídeo generado: {out_path}")

    # subtítulos opcional: generarlos y quemarlos con ffmpeg
    if ADD_SUBTITLES:
        try:
            ff = FFMPEG_PATH if FFMPEG_PATH != "ffmpeg" else shutil.which("ffmpeg")
            if not ff or not Path(ff).exists():
                log("⚠️ ffmpeg no encontrado. Se omite generación de subtítulos quemados.")
            else:
                seq = _seq_from_poses_string(POSES)
                ass_path = Path(out_file.parent) / (out_file.stem + ".ass")
                generate_ass_for_cumulative_letters(seq, FPS, HOLD, TRANSITION, str(ass_path), start_frame=1)
                temp_out = out_file.with_name(out_file.stem + "_subs" + out_file.suffix)
                log(f"Quemando subtítulos ASS -> {ass_path} en {out_path}")
                burn_ass_with_ffmpeg(out_file, str(ass_path), temp_out, ff)
                # replace original file with subtitled version
                temp_out.replace(out_file)
                log(f"Subtítulos aplicados: {out_file}")
        except Exception as e:
            log(f"Error aplicando subtítulos: {e}")

    # seguir con ajuste de velocidad si procede...
    # ajustar velocidad si se solicita
    if float(SPEED) != 1.0:
        try:
            ff = FFMPEG_PATH if FFMPEG_PATH != "ffmpeg" else shutil.which("ffmpeg")
            if not ff or not Path(ff).exists():
                log("⚠️ ffmpeg no encontrado. Se omite ajuste de velocidad.")
            else:
                log(f"Ajustando velocidad con ffmpeg: {ff}")
                adjusted = adjust_video_speed_ffmpeg(out_path, SPEED, ffmpeg_exec=ff)
                log(f"Archivo ajustado: {adjusted}")
        except Exception as e:
            log(f"Error ajustando velocidad con ffmpeg: {e}")
    else:
        log("SPEED=1.0 -> no se aplica ajuste de velocidad.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"{LOG_PREFIX} ⚠️ Error inesperado: {e}")
        raise
