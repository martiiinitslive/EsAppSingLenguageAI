"""
PIPELINE COMPLETO: Video → MediaPipe → Poses → Blender → Video Renderizado

Este script ejecuta toda la cadena:
1. Convierte MediaPipe JSON a poses para Blender
2. Aplica poses en Blender y renderiza

Ajusta las rutas y configuración al inicio.
"""

import os
import sys
import shlex
import shutil
import subprocess
import re
import json
from pathlib import Path

# ============================================================================
# CONFIGURACIÓN - AJUSTA ESTOS VALORES
# ============================================================================

# Ruta a ejecutable de Blender
BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
# En Linux: "/usr/bin/blender"
# En Mac: "/Applications/Blender.app/Contents/MacOS/Blender"

# Carpeta base (donde están los scripts)
BASE_DIR = Path(__file__).resolve().parent

# Archivos
BLEND_FILE = str(BASE_DIR / "cuerpo_humano_rigged.blend")
# Script de conversión (está en la subcarpeta `mediapipe`)
SCRIPT_CONVERT = str(BASE_DIR / "mediapipe" / "convert_mediapipe_to_blender.py")
# Script de renderizado (en la carpeta principal `blender`)
SCRIPT_RENDER = str(BASE_DIR / "text_to_video_from_poses_video.py")

# JSON de entrada (MediaPipe) -- normalmente en `mediapipe/`
POSES_MEDIAPIPE_JSON = str(BASE_DIR / "mediapipe" / "poses_mediapipe_video.json")

# JSON intermedio (convertido) -- lo genera el script de conversión en `mediapipe/`
POSES_CONVERTED_JSON = str(BASE_DIR / "mediapipe" / "poses_converted_video.json")

# Configuración de renderizado
ARMATURE_NAME = "Human.rig"
POSES_TO_RENDER = "L,U,C,I,A"  # Lista de poses separadas por comas
FPS = 60
WIDTH = 1920
HEIGHT = 1080
ENGINE = "EEVEE"  # o "CYCLES" para más calidad

# Ruta de salida
OUTPUT_DIR = str(BASE_DIR / "output")
OUTPUT_VIDEO = str(Path(OUTPUT_DIR) / "alphabet_video.mp4")

# FFmpeg (para post-procesamiento)
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"
# En Linux/Mac: "ffmpeg" (si está en PATH)

# ============================================================================
# FUNCIONES
# ============================================================================

LOG_PREFIX = "[PIPELINE]"

def log(msg):
    """Imprime log con prefijo."""
    print(f"{LOG_PREFIX} {msg}")

def require_file(path, description, exit_on_missing=True):
    """Verifica que un archivo exista."""
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

def step_convert_mediapipe():
    """
    PASO 1: Convertir MediaPipe JSON a formato Blender.
    """
    log("\n" + "="*70)
    log("PASO 1: CONVERTIR MediaPipe → Formato Blender")
    log("="*70)

    require_file(POSES_MEDIAPIPE_JSON, "poses_mediapipe_video.json")
    require_file(SCRIPT_CONVERT, "script de conversión")

    log(f"Entrada:  {POSES_MEDIAPIPE_JSON}")
    log(f"Salida:   {POSES_CONVERTED_JSON}")


    # Si el JSON convertido ya existe, saltar la conversión para ahorrar tiempo
    out_p = Path(POSES_CONVERTED_JSON)
    if out_p.exists():
        regenerate = False
        try:
            with out_p.open("r", encoding="utf-8") as f:
                existing = json.load(f)
            poses = existing.get("poses", {}) if isinstance(existing, dict) else {}
            # inspeccionar primer frame para ver si ya tiene landmarks_blender
            missing_landmarks = True
            for pose in poses.values():
                frames = pose.get("frames", []) if isinstance(pose, dict) else []
                for fr in frames:
                    if isinstance(fr, dict) and fr.get("landmarks_blender"):
                        missing_landmarks = False
                        break
                if not missing_landmarks:
                    break
            if not missing_landmarks:
                log(f"✓ Saltando conversión: '{POSES_CONVERTED_JSON}' ya contiene landmarks")
                return
            regenerate = True
        except Exception:
            regenerate = True

        if not regenerate:
            return
        log(f"ℹ️  Regenerando conversión: falta 'landmarks_blender' en {POSES_CONVERTED_JSON}")

    cmd = [sys.executable, SCRIPT_CONVERT]

    log(f"Ejecutando: {' '.join(shlex.quote(a) for a in cmd)}")

    result = subprocess.run(cmd, cwd=str(BASE_DIR))

    if result.returncode != 0:
        log(f"❌ Conversión falló (rc={result.returncode})")
        sys.exit(1)

    if not require_file(POSES_CONVERTED_JSON, "poses convertidas", exit_on_missing=False):
        log("❌ Conversión no generó salida")
        sys.exit(1)

    log("✓ Conversión completada")

def step_render_blender():
    """
    PASO 2: Aplicar poses en Blender y renderizar.
    """
    log("\n" + "="*70)
    log("PASO 2: APLICAR POSES EN BLENDER Y RENDERIZAR")
    log("="*70)

    require_file(BLENDER_PATH, "Blender")
    require_file(BLEND_FILE, "archivo .blend")
    require_file(SCRIPT_RENDER, "script de renderizado")
    require_file(POSES_CONVERTED_JSON, "poses convertidas")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    log(f"Blender:   {BLENDER_PATH}")
    log(f"Escena:    {BLEND_FILE}")
    log(f"Poses:     {POSES_CONVERTED_JSON}")
    log(f"Armature:  {ARMATURE_NAME}")
    log(f"Salida:    {OUTPUT_VIDEO}")

    args = [
        BLENDER_PATH,
        BLEND_FILE,
        "--background",
        "--python", SCRIPT_RENDER,
        "--",
        "--library", POSES_CONVERTED_JSON,
        "--poses", POSES_TO_RENDER,
        "--armature", ARMATURE_NAME,
        "--fps", str(FPS),
        "--width", str(WIDTH),
        "--height", str(HEIGHT),
        "--engine", ENGINE,
        "--out", OUTPUT_VIDEO
    ]

    log(f"\nEjecutando Blender...")
    log(f"Comando: {' '.join(shlex.quote(a) for a in args)}")

    result = subprocess.run(args)

    if result.returncode != 0:
        log(f"❌ Blender falló (rc={result.returncode})")
        sys.exit(1)

    log("✓ Blender completado")

def main():
    """Ejecuta el pipeline completo."""
    log("="*70)
    log("PIPELINE COMPLETO: MediaPipe → Blender → Video")
    log("="*70)

    # PASO 1: Conversión
    try:
        step_convert_mediapipe()
    except Exception as e:
        log(f"❌ Error en conversión: {e}")
        sys.exit(1)

    # PASO 2: Renderizado
    try:
        step_render_blender()
    except Exception as e:
        log(f"❌ Error en renderizado: {e}")
        sys.exit(1)

    # Verificar salida
    if Path(OUTPUT_VIDEO).exists():
        size_mb = Path(OUTPUT_VIDEO).stat().st_size / (1024*1024)
        log(f"\n✅ VIDEO COMPLETADO: {OUTPUT_VIDEO} ({size_mb:.1f} MB)")
    else:
        log(f"\n⚠️ No se encontró video de salida: {OUTPUT_VIDEO}")

    log("\n" + "="*70)
    log("PIPELINE COMPLETADO")
    log("="*70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
