"""
CONVERSIÓN: MediaPipe Video → Poses Convertidas para Blender
=============================================================

Convierte un JSON de MediaPipe (video-based) a un formato que Blender puede aplicar
directamente a los huesos de MakeHuman usando quaternions y jerarquía parent-child.

ENTRADA: poses_mediapipe_video.json (landmarks de MediaPipe frame-by-frame)
SALIDA: poses_converted_video.json (rotaciones de huesos preparadas para Blender)

Mapeando 21 landmarks de MediaPipe a 15 huesos de MakeHuman.
"""

import json
from pathlib import Path
import math
import numpy as np

# ============================================================================
# MAPEO: MediaPipe 21 Landmarks → MakeHuman Hand Bones
# ============================================================================
# 
# MediaPipe proporciona 21 puntos (0-20). Mapeamos cada par de landmarks
# consecutivos a un hueso de MakeHuman.
#
# Formato: 'bone_name': (start_landmark_idx, end_landmark_idx)

MEDIAPIPE_TO_MAKEHUMAN_BONES = {
    # Pulgar (Thumb: landmarks 1-4)
    'finger1-1.L': (0, 1),   # WRIST → THUMB_CMC
    'finger1-2.L': (1, 2),   # THUMB_CMC → THUMB_MCP
    'finger1-3.L': (2, 3),   # THUMB_MCP → THUMB_IP

    # Índice (Index: landmarks 5-8)
    'finger2-1.L': (0, 5),   # WRIST → INDEX_MCP
    'finger2-2.L': (5, 6),   # INDEX_MCP → INDEX_PIP
    'finger2-3.L': (6, 7),   # INDEX_PIP → INDEX_DIP

    # Medio (Middle: landmarks 9-12)
    'finger3-1.L': (0, 9),   # WRIST → MIDDLE_MCP
    'finger3-2.L': (9, 10),  # MIDDLE_MCP → MIDDLE_PIP
    'finger3-3.L': (10, 11), # MIDDLE_PIP → MIDDLE_DIP

    # Anular (Ring: landmarks 13-16)
    'finger4-1.L': (0, 13),  # WRIST → RING_MCP
    'finger4-2.L': (13, 14), # RING_MCP → RING_PIP
    'finger4-3.L': (14, 15), # RING_PIP → RING_DIP

    # Meñique (Pinky: landmarks 17-20)
    'finger5-1.L': (0, 17),  # WRIST → PINKY_MCP
    'finger5-2.L': (17, 18), # PINKY_MCP → PINKY_PIP
    'finger5-3.L': (18, 19), # PINKY_PIP → PINKY_DIP
}

# ============================================================================
# RUTAS (Ajusta según tu estructura de carpetas)
# ============================================================================

IN_PATH = Path(__file__).resolve().parent / "poses_mediapipe_video.json"
OUT_PATH = Path(__file__).resolve().parent / "poses_converted_video.json"
# Información del rig exportada con utils/info_pose_base.py (bone tails/heads en reposo)
INFO_POSE_BASE = Path(__file__).resolve().parent.parent / "info_pose_base.json"

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def load_json(p):
    """Carga un JSON."""
    return json.loads(Path(p).read_text(encoding="utf-8"))

def save_json(p, data):
    """Guarda un JSON formateado."""
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(data, indent=2), encoding="utf-8")

def to_np(lm):
    """
    Convierte un landmark de cualquier formato a numpy array [x, y, z].
    Soporta: dict {x, y, z}, list/tuple.
    """
    if lm is None:
        return None

    if isinstance(lm, dict):
        if "x" in lm and "y" in lm:
            x = float(lm.get("x", 0.0))
            y = float(lm.get("y", 0.0))
            z = float(lm.get("z", 0.0))
            return np.array([x, y, z], dtype=float)

    if isinstance(lm, (list, tuple)) and len(lm) >= 2:
        x = float(lm[0])
        y = float(lm[1])
        z = float(lm[2]) if len(lm) > 2 else 0.0
        return np.array([x, y, z], dtype=float)

    return None

def mediapipe_to_blender_coords(p):
    """
    Convierte coordenadas de MediaPipe a sistema de Blender.

    MediaPipe: X (derecha), Y (abajo), Z (hacia atrás/cámara)
    Blender:   X (derecha), Y (adelante), Z (arriba)
    """
    return np.array([
        p[0],       # X se mantiene
        -p[2],      # Z de MP → -Y de Blender
        p[1]        # Y de MP → Z de Blender
    ], dtype=float)

def quat_from_vectors(u, v):
    """
    Calcula el quaternion [w, x, y, z] que rota el vector u al vector v.

    Usa la fórmula de Rodrigues para rotaciones.
    """
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)

    dot = np.dot(u, v)

    # Casos especiales: vectores paralelos o antiparalelos
    if dot > 0.999999:
        return [1.0, 0.0, 0.0, 0.0]

    if dot < -0.999999:
        # 180 grados: elegir eje ortogonal
        axis = np.cross(u, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(u, np.array([0.0, 1.0, 0.0]))
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return [0.0, float(axis[0]), float(axis[1]), float(axis[2])]

    # Caso general: usar fórmula de Rodrigues
    axis = np.cross(u, v)
    s = math.sqrt((1.0 + dot) * 2.0)
    invs = 1.0 / s

    qx = axis[0] * invs
    qy = axis[1] * invs
    qz = axis[2] * invs
    qw = 0.5 * s

    return [float(qw), float(qx), float(qy), float(qz)]


def load_rest_vectors(info_path):
    """
    Lee info_pose_base.json y devuelve un dict nombre_hueso -> vector_tail-head normalizado (numpy).
    Si hay cualquier problema, devuelve dict vacío.
    """
    try:
        info = json.loads(Path(info_path).read_text(encoding="utf-8"))
    except Exception:
        return {}

    rest = {}
    for arm in info.get("armatures", []):
        # priorizar Human.rig si existe; si no, se toma el primero
        if arm.get("object_name") not in ("Human.rig", None):
            continue
        for b in arm.get("bones", []):
            head = np.array(b.get("head_local") or b.get("head") or [0.0, 0.0, 0.0], dtype=float)
            tail = np.array(b.get("tail_local") or b.get("tail") or [0.0, 1.0, 0.0], dtype=float)
            vec = tail - head
            n = np.linalg.norm(vec)
            if n > 1e-9:
                rest[b.get("name")] = vec / n
        # solo necesitamos un armature
        break
    return rest

def convert_frame_to_bones(frame_data, landmarks, denorm_config, rest_vectors):
    """
    Convierte un frame con 21 landmarks a un diccionario de huesos con rotaciones.

    Args:
        frame_data: datos del frame (frame_number, timestamp, etc.)
        landmarks: lista de 21 puntos de MediaPipe
        denorm_config: configuración de desnormalización (ancho, alto, escala)

    Returns:
        dict con estructura {'frame_number': N, 'timestamp_sec': T, 'bones': {...}}
    """
    frame_out = {
        'frame_number': frame_data.get('frame_number'),
        'timestamp_sec': frame_data.get('timestamp_sec'),
        'hand_detected': frame_data.get('hand_detected', False),
        'handedness': frame_data.get('handedness')
    }

    # Convertir landmarks a numpy arrays
    pts = []
    for lm in landmarks:
        p = to_np(lm)
        if p is None:
            continue
        pts.append(p)

    if len(pts) == 0:
        frame_out['bones'] = {}
        return frame_out

    pts = np.array(pts)

    # Detectar si vienen world_landmarks (metros)
    is_world = bool(frame_data.get('world_landmarks'))

    if is_world:
        pts_m = pts.copy()
    else:
        # Si los puntos parecen normalizados (~0..1), desnormalizar a metros
        if np.nanmax(np.abs(pts)) <= 1.5:
            # Heurística: si los valores están en [0, 1], son normalizados
            hand_width_mm = float(denorm_config.get('hand_width_mm', 200))
            scale_m = hand_width_mm / 1000.0  # convertir mm a metros

            # Centrar y voltear Y
            px = (pts[:, 0] - 0.5) * scale_m
            py = (0.5 - pts[:, 1]) * scale_m
            pz = pts[:, 2] * scale_m

            pts_m = np.column_stack([px, py, pz])
        else:
            # Ya están en metros (world_landmarks)
            pts_m = pts.copy()

    # Convertir a sistema de coordenadas de Blender
    pts_blender = np.array([mediapipe_to_blender_coords(p) for p in pts_m])

    # Generar rotaciones para cada hueso
    bones = {}

    for bone_name, (start_idx, end_idx) in MEDIAPIPE_TO_MAKEHUMAN_BONES.items():
        if start_idx >= len(pts_blender) or end_idx >= len(pts_blender):
            continue

        # Posición inicial y final del hueso en Blender
        pos_start = pts_blender[start_idx]
        pos_end = pts_blender[end_idx]

        # Vector de dirección del hueso
        v_target = pos_end - pos_start

        if np.linalg.norm(v_target) < 1e-9:
            continue

        # Vector de referencia: usar vector de reposo real si existe, sino +Y
        v_from = np.array(rest_vectors.get(bone_name, [0.0, 1.0, 0.0]), dtype=float)

        # Calcular quaternion
        quat = quat_from_vectors(v_from, v_target)

        # Solo rotación: no mover huesos en espacio mundo para evitar desplazar la malla
        bones[bone_name] = {'rotation_quat': quat}  # [w, x, y, z]

    frame_out['bones'] = bones
    frame_out['landmarks_blender'] = pts_blender.tolist()
    return frame_out

def convert_pose_entry(pose_entry, denorm_config, rest_vectors):
    """
    Convierte una entrada de pose (puede tener múltiples frames) a formato Blender.

    Args:
        pose_entry: entrada del JSON con 'frames' (lista de frames)
        denorm_config: configuración global de desnormalización

    Returns:
        lista de frames convertidos
    """
    frames_in = pose_entry.get('frames', [])
    frames_out = []

    for frame_in in frames_in:
        landmarks = frame_in.get('world_landmarks') or frame_in.get('landmarks', [])

        if len(landmarks) != 21:
            # Frame inválido, saltar
            frame_out = {
                'frame_number': frame_in.get('frame_number'),
                'timestamp_sec': frame_in.get('timestamp_sec'),
                'hand_detected': False,
                'bones': {}
            }
        else:
            frame_out = convert_frame_to_bones(frame_in, landmarks, denorm_config, rest_vectors)

        frames_out.append(frame_out)

    return frames_out

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("CONVERSIÓN: MediaPipe Video → Poses para Blender")
    print("="*70)

    # Cargar entrada
    input_file = IN_PATH
    if not input_file.exists():
        # Buscar relativa al script
        input_file = Path(__file__).resolve().parent / "poses_mediapipe_video.json"

    if not input_file.exists():
        raise SystemExit(f"❌ Archivo de entrada no encontrado: {IN_PATH}")

    print(f"✓ Cargando: {input_file}")
    data_in = load_json(input_file)

    # Metadatos de salida
    meta_out = {
        'armature': 'Human.rig',
        'rotation_format': 'QUATERNION',
        'source': 'MediaPipe hand landmarks (VIDEO)',
        'conversion_note': 'Mapeado a huesos de mano MakeHuman con jerarquía parent-child',
        'denormalization': {
            'image_width': 1920,
            'image_height': 1080,
            'hand_width_mm': 200,
            'camera_fov_deg': 65
        }
    }

    denorm_config = meta_out['denormalization']

    # Cargar vectores de reposo del rig (si info_pose_base.json existe)
    rest_vectors = load_rest_vectors(INFO_POSE_BASE)

    # Convertir poses
    out = {'meta': meta_out, 'poses': {}}

    poses_in = data_in.get('poses', {})

    for pose_name, pose_entry in poses_in.items():
        print(f"\n  Procesando pose: {pose_name}")

        frames_converted = convert_pose_entry(pose_entry, denorm_config, rest_vectors)

        out['poses'][pose_name] = {
            'gesture_name': pose_entry.get('gesture_name', pose_name),
            'video_info': pose_entry.get('video_info', {}),
            'total_frames': len(frames_converted),
            'frames': frames_converted
        }

        print(f"    ✓ {len(frames_converted)} frames convertidos")

    # Guardar salida
    save_json(OUT_PATH, out)
    print(f"\n✓ Guardado: {OUT_PATH}")
    print(f"\nResumen:")
    print(f"  - Poses procesadas: {len(out['poses'])}")
    print(f"  - Mapeo: 21 landmarks MediaPipe → 15 huesos MakeHuman")
    print(f"  - Rotaciones: Quaternions [w, x, y, z]")
    print(f"  - Armature destino: {meta_out['armature']}")

if __name__ == "__main__":
    main()
