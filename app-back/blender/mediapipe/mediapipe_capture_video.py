# ============================================
# PASO 1: Capturar datos de MediaPipe desde VÍDEOS
# ============================================
import cv2
import json
import mediapipe as mp
from pathlib import Path
import argparse
import datetime
import numpy as np

# base app-back directory (relative pathing)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT_DIR = BASE_DIR / "data" / "dataset-en-bruto" / "asl_dataset" / "videos" / "videos_de_las_letras"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_hand_landmarks_from_video(video_path, gesture_name=None, convert_to_blender=False, 
                                      sample_every_n_frames=1, max_frames=None):
    """
    Procesa un vídeo y extrae landmarks de MediaPipe por frame.
    
    Args:
        video_path: ruta al archivo de vídeo
        gesture_name: nombre del gesto (ej. "A", "B", "C")
        convert_to_blender: si True, convierte world_landmarks a ejes Blender
        sample_every_n_frames: procesar cada N frames (para acelerar)
        max_frames: máximo número de frames a procesar (None = todo)
    
    Returns:
        dict con estructura de vídeo procesado
    """
    video_p = Path(video_path)
    if not video_p.exists():
        raise FileNotFoundError(f"Vídeo no encontrado: {video_path}")
    
    cap = cv2.VideoCapture(str(video_p))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV no pudo abrir el vídeo: {video_path}")
    
    # Obtener información del vídeo
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"  Procesando: {video_p.name}")
    print(f"    FPS: {fps}, Total frames: {total_frames}, Resolución: {width}x{height}")
    
    frame_landmarks = []
    frames_processed = 0
    frame_count = 0
    hand_detected_count = 0
    
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                       min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Saltar frames según sample_every_n_frames
            if frame_count % sample_every_n_frames != 0:
                continue
            
            if max_frames and frames_processed >= max_frames:
                break
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            frame_data = {
                "frame_number": frame_count,
                "timestamp_sec": frame_count / fps if fps > 0 else 0,
                "hand_detected": False,
                "handedness": None,
                "landmarks": None,
                "world_landmarks": None
            }
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                handedness = None
                if results.multi_handedness:
                    handedness = results.multi_handedness[0].classification[0].label
                
                # Landmarks normalizados (0-1)
                norm_lms = [
                    {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} 
                    for lm in hand_landmarks.landmark
                ]
                
                # World landmarks (coordenadas reales en metros)
                world_lms = None
                if hasattr(results, "multi_hand_world_landmarks") and results.multi_hand_world_landmarks:
                    try:
                        w = results.multi_hand_world_landmarks[0]
                        world_lms = [
                            {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} 
                            for lm in w.landmark
                        ]
                    except Exception:
                        world_lms = None
                
                # Convertir a Blender si se requiere
                if convert_to_blender and world_lms:
                    world_lms_blender = []
                    for p in world_lms:
                        world_lms_blender.append({
                            "x": float(p["x"]), 
                            "y": float(-p["z"]), 
                            "z": float(p["y"])
                        })
                    world_lms = world_lms_blender
                
                frame_data.update({
                    "hand_detected": True,
                    "handedness": handedness,
                    "landmarks": norm_lms,
                    "world_landmarks": world_lms
                })
                
                hand_detected_count += 1
            
            frame_landmarks.append(frame_data)
            frames_processed += 1
            
            if frames_processed % 30 == 0:
                print(f"    Procesados {frames_processed} frames...")
    
    cap.release()
    
    print(f"    ✓ Completado: {frames_processed} frames procesados, {hand_detected_count} con mano detectada")
    
    return {
        "source_video": str(video_p),
        "gesture_name": gesture_name,
        "video_info": {
            "fps": float(fps),
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration_sec": total_frames / fps if fps > 0 else 0
        },
        "frames_processed": frames_processed,
        "hand_detected_frames": hand_detected_count,
        "hand_detection_ratio": hand_detected_count / frames_processed if frames_processed > 0 else 0,
        "frames": frame_landmarks
    }

def _iter_video_dirs(base_dir):
    """
    Itera sobre directorios de letras, cada uno con sus vídeos.
    Estructura esperada:
    base_dir/
      A/
        video1.mp4
        video2.mp4
      B/
        video1.mp4
      ...
    """
    base_p = Path(base_dir)
    if not base_p.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {base_dir}")
    
    for letter_dir in sorted(base_p.iterdir()):
        if not letter_dir.is_dir():
            continue
        
        letter = letter_dir.name.strip().upper()
        video_files = []
        
        # Buscar archivos de vídeo
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.flv", "*.webm"):
            video_files.extend(sorted(letter_dir.glob(ext)))
        
        if video_files:
            yield letter, video_files

def main():
    ap = argparse.ArgumentParser(
        description="Capturar landmarks MediaPipe desde vídeos -> JSON único"
    )
    ap.add_argument(
        "--input_dir", 
        default=str(DEFAULT_INPUT_DIR),
        help=f"Carpeta raíz con subcarpetas por letra (cada una con vídeos). "
             f"Por defecto: {DEFAULT_INPUT_DIR}"
    )
    ap.add_argument(
        "--single_output", 
        default=r"C:\Users\marti\Desktop\tfg_teleco\proyectos\EsAppSingLenguageAI\app-back\blender\mediapipe\poses_mediapipe_video.json",
        help="Archivo JSON único donde se guardarán todas las poses de vídeos"
    )
    ap.add_argument(
        "--blender", 
        action="store_true", 
        help="Convertir world_landmarks a coordenadas tipo Blender (x,-z,y)"
    )
    ap.add_argument(
        "--sample_frames", 
        type=int, 
        default=1,
        help="Procesar cada N frames (ej. 2 = cada 2 frames, acelera procesamiento)"
    )
    ap.add_argument(
        "--max_frames_per_video", 
        type=int, 
        default=None,
        help="Máximo número de frames a procesar por vídeo (None = todo)"
    )
    args = ap.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Directorio de entrada no encontrado: {input_dir}")
    
    print("=" * 70)
    print("CAPTURAR LANDMARKS MediaPipe DESDE VÍDEOS")
    print("=" * 70)
    print(f"Directorio entrada: {input_dir}")
    print(f"Blender conversion: {args.blender}")
    print(f"Sample every N frames: {args.sample_frames}")
    print(f"Max frames por vídeo: {args.max_frames_per_video}")
    print()
    
    all_poses = {}
    total_videos = 0
    
    meta = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "source": "MediaPipe hand landmarks from videos",
        "converted_to_blender": bool(args.blender),
        "sample_every_n_frames": args.sample_frames,
        "max_frames_per_video": args.max_frames_per_video
    }
    
    try:
        for letter, video_files in _iter_video_dirs(input_dir):
            print(f"\nProcesando letra: {letter}")
            print(f"  Vídeos encontrados: {len(video_files)}")
            
            letter_videos = []
            
            for video_file in video_files:
                try:
                    result = extract_hand_landmarks_from_video(
                        str(video_file),
                        gesture_name=letter,
                        convert_to_blender=args.blender,
                        sample_every_n_frames=args.sample_frames,
                        max_frames=args.max_frames_per_video
                    )
                    
                    letter_videos.append(result)
                    total_videos += 1
                    
                except Exception as e:
                    print(f"  ❌ Error procesando {video_file.name}: {e}")
            
            if letter_videos:
                # Si hay un único vídeo, guardar directamente; si hay múltiples, guardar como lista
                if len(letter_videos) == 1:
                    all_poses[letter] = letter_videos[0]
                else:
                    all_poses[letter] = letter_videos
                
                print(f"  ✓ {letter}: {len(letter_videos)} vídeo(s) procesado(s)")
    
    except Exception as e:
        print(f"❌ Error durante procesamiento: {e}")
        return
    
    if not all_poses:
        print("\n❌ No se procesaron vídeos. Verifica la estructura de directorios.")
        return
    
    # Escribir JSON único
    output_file = Path(args.single_output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    meta["total_videos_processed"] = total_videos
    meta["total_letters"] = len(all_poses)
    
    final = {
        "meta": meta,
        "poses": all_poses
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print(f"✅ COMPLETADO")
    print(f"   Total vídeos procesados: {total_videos}")
    print(f"   Total letras: {len(all_poses)}")
    print(f"   Salida: {output_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
