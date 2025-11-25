import json
import math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

BASE_DIR = Path(__file__).parent

MEDIAPIPE_JSON = BASE_DIR / "poses_mediapipe_video.json"
#MEDIAPIPE_JSON = BASE_DIR / "poses_mediapipe.json"

OUTPUT_JSON = BASE_DIR.parent / "poses_converted_video.json"
#OUTPUT_JSON = BASE_DIR.parent / "poses_converted.json"

# ========== PARÁMETROS DE DESNORMALIZACIÓN ==========
IMAGE_WIDTH = 640           # píxeles (ajusta a tu resolución)
IMAGE_HEIGHT = 480          # píxeles
HAND_WIDTH_MM = 200         # milímetros (~20cm)
CAMERA_FOV_DEG = 65         # grados (típico en móviles)

HAND_WIDTH_M = HAND_WIDTH_MM / 1000.0  # convertir a metros

# ========== MAPEO MEDIAPIPE -> BLENDER ==========
MEDIAPIPE_TO_BLENDER = {
    0: "wrist.L",
    1: "finger1-1.L",
    2: "finger1-2.L",
    3: "finger1-3.L",
    5: "finger2-1.L",
    6: "finger2-2.L",
    7: "finger2-3.L",
    9: "finger3-1.L",
    10: "finger3-2.L",
    11: "finger3-3.L",
    13: "finger4-1.L",
    14: "finger4-2.L",
    15: "finger4-3.L",
    17: "finger5-1.L",
    18: "finger5-2.L",
    19: "finger5-3.L",
}

def denormalize_mediapipe(mp_point: dict, image_width: int = IMAGE_WIDTH, 
                          image_height: int = IMAGE_HEIGHT) -> dict:
    """
    Convierte un punto normalizado MediaPipe (0-1) a coordenadas Blender (metros).
    
    Args:
        mp_point: {"x": 0-1, "y": 0-1, "z": 0-1}
        image_width: ancho de imagen en píxeles
        image_height: alto de imagen en píxeles
    
    Returns:
        {"x": metros, "y": metros, "z": metros} en coordenadas Blender
    """
    # Paso 1: Pasar de (0-1) a píxeles
    x_px = mp_point["x"] * image_width
    y_px = mp_point["y"] * image_height
    z_norm = mp_point["z"]
    
    # Paso 2: Centrar en (0, 0)
    x_centered_px = x_px - image_width / 2
    y_centered_px = y_px - image_height / 2
    
    # Paso 3: Convertir píxeles a metros
    # Scale: metros por píxel
    scale = HAND_WIDTH_M / image_width
    
    x_blender = x_centered_px * scale
    y_blender = -y_centered_px * scale  # ← Invertir Y (MediaPipe vs Blender)
    z_blender = z_norm * HAND_WIDTH_M   # Escalar profundidad
    
    return {
        "x": float(x_blender),
        "y": float(y_blender),
        "z": float(z_blender)
    }

def compute_rotation(vec_from: np.ndarray, vec_to: np.ndarray) -> dict:
    """Calcula rotación que lleva vec_from a vec_to."""
    
    def safe_normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        if n < 1e-8:
            return None
        return v / n
    
    v1 = safe_normalize(np.array(vec_from, dtype=float))
    v2 = safe_normalize(np.array(vec_to, dtype=float))
    
    if v1 is None or v2 is None:
        rot = R.from_matrix(np.eye(3))
    else:
        dot = np.dot(v1, v2)
        
        if np.isclose(dot, 1.0, atol=1e-8):
            rot = R.from_matrix(np.eye(3))
        elif np.isclose(dot, -1.0, atol=1e-8):
            axis = np.array([1.0, 0.0, 0.0])
            if abs(v1[0]) > 0.9:
                axis = np.array([0.0, 1.0, 0.0])
            ortho = np.cross(v1, axis)
            ortho = ortho / np.linalg.norm(ortho)
            rot = R.from_rotvec(math.pi * ortho)
        else:
            cross = np.cross(v1, v2)
            s = np.linalg.norm(cross)
            K = np.array([[0, -cross[2], cross[1]],
                          [cross[2], 0, -cross[0]],
                          [-cross[1], cross[0], 0]])
            rot_matrix = np.eye(3) + K + K @ K * ((1 - dot) / (s ** 2))
            rot = R.from_matrix(rot_matrix)
    
    # scipy devuelve [x,y,z,w]; convertir a [w,x,y,z]
    q_xyzw = rot.as_quat()
    q_xyzw = q_xyzw / np.linalg.norm(q_xyzw)
    q_wxyz = [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]
    
    # Euler en radianes
    euler_rad = rot.as_euler('XYZ', degrees=False)
    
    return {
        "rot_quat": q_wxyz,  # [w,x,y,z]
        "rot_euler": [float(e) for e in euler_rad]  # radianes
    }

# ========== CARGAR Y PROCESAR ==========

if not MEDIAPIPE_JSON.exists():
    raise FileNotFoundError(f"No encontrado: {MEDIAPIPE_JSON}")

with open(MEDIAPIPE_JSON) as f:
    mediapipe_data = json.load(f)

output_data = {
    "meta": {
        "armature": "Human.rig",
        "rotation": "QUATERNION",
        "source": "MediaPipe hand landmarks",
        "denormalization": {
            "image_width": IMAGE_WIDTH,
            "image_height": IMAGE_HEIGHT,
            "hand_width_mm": HAND_WIDTH_MM,
            "camera_fov_deg": CAMERA_FOV_DEG
        }
    },
    "poses": {}
}

print(f"Procesando {len(mediapipe_data.get('poses', {}))} gestos...")
print(f"Config desnormalización:")
print(f"  - Resolución imagen: {IMAGE_WIDTH}x{IMAGE_HEIGHT} px")
print(f"  - Ancho mano: {HAND_WIDTH_MM} mm = {HAND_WIDTH_M} m")
print(f"  - FOV cámara: {CAMERA_FOV_DEG}°\n")

for gesture_name, gesture_info in mediapipe_data.get("poses", {}).items():
    if not gesture_info.get("hand_detected", False):
        print(f"⊘ {gesture_name}: hand_detected=False (saltado)")
        continue
    
    world_landmarks = gesture_info.get("world_landmarks", [])
    if len(world_landmarks) != 21:
        print(f"⊘ {gesture_name}: {len(world_landmarks)} landmarks (esperado 21, saltado)")
        continue
    
    # Desnormalizar landmarks
    points_denorm = []
    for lm in world_landmarks:
        pt = denormalize_mediapipe(lm)
        points_denorm.append(pt)
    
    points_array = np.array([[p["x"], p["y"], p["z"]] for p in points_denorm])
    
    # Calcular rotaciones
    bones_rotations = {}
    
    # Mapeo: para cada dedo, calcula rotaciones entre segmentos
    fingers = {
        "Thumb": [0, 1, 2, 3, 4],
        "Index": [0, 5, 6, 7, 8],
        "Middle": [0, 9, 10, 11, 12],
        "Ring": [0, 13, 14, 15, 16],
        "Pinky": [0, 17, 18, 19, 20]
    }
    
    for finger_name, indices in fingers.items():
        for i in range(len(indices) - 1):
            start_idx = indices[i]
            end_idx = indices[i + 1]
            
            # Nombre hueso Blender
            bone_name = MEDIAPIPE_TO_BLENDER.get(end_idx)
            if not bone_name:
                continue
            
            # Vector del hueso (desde start a end)
            vec_from = np.array([1.0, 0.0, 0.0])  # dirección por defecto
            vec_to = points_array[end_idx] - points_array[start_idx]
            
            # Calcular rotación
            rot_data = compute_rotation(vec_from, vec_to)
            bones_rotations[bone_name] = {
                "rot_quat": rot_data["rot_quat"],
                "rot_euler": rot_data["rot_euler"]
            }
    
    output_data["poses"][gesture_name] = {"bones": bones_rotations}
    print(f"✓ {gesture_name}: {len(bones_rotations)} huesos procesados")

# Escribir salida
with open(OUTPUT_JSON, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"\n✓ Conversión completada.")
print(f"  Salida: {OUTPUT_JSON}")
print(f"  Gestos: {len(output_data['poses'])}")