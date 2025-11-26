"""
APLICAR POSES EN BLENDER: Convierte poses_converted_video.json a animación renderizada

Ejecutar dentro de Blender:
blender archivo.blend --python text_to_video_from_poses_video_UPDATED.py -- \
    --library poses_converted_video.json \
    --poses "A,B,C" \
    --armature "Human.rig" \
    --fps 60

Este script APLICARÁ las rotaciones quaternion directamente a los huesos de mano.
"""

import bpy
import os
import sys
import json
import argparse
try:
    import numpy as np
except ImportError:  # pragma: no cover - Blender env should provide numpy
    np = None  # type: ignore[assignment]
from mathutils import Vector, Quaternion, Matrix  # type: ignore[import]

# ============================================================================
# MAPEO: MediaPipe 21 Landmarks → MakeHuman Hand Bones
# ============================================================================

MEDIAPIPE_TO_MAKEHUMAN = {
    'finger1-1.L': (0, 1),
    'finger1-2.L': (1, 2),
    'finger1-3.L': (2, 3),
    'finger2-1.L': (0, 5),
    'finger2-2.L': (5, 6),
    'finger2-3.L': (6, 7),
    'finger3-1.L': (0, 9),
    'finger3-2.L': (9, 10),
    'finger3-3.L': (10, 11),
    'finger4-1.L': (0, 13),
    'finger4-2.L': (13, 14),
    'finger4-3.L': (14, 15),
    'finger5-1.L': (0, 17),
    'finger5-2.L': (17, 18),
    'finger5-3.L': (18, 19),
}

REFERENCE_PAIRS = [
    ((0, 5), 'finger2-1.L'),
    ((0, 9), 'finger3-1.L'),
    ((0, 13), 'finger4-1.L'),
    ((0, 17), 'finger5-1.L'),
]

# ============================================================================
# UTILIDADES
# ============================================================================

def _script_dir():
    """Obtiene el directorio del script."""
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except Exception:
        pass

    try:
        for t in getattr(bpy, "data", {}).texts:
            fp = getattr(t, "filepath", "")
            if fp:
                return os.path.dirname(bpy.path.abspath(fp))
    except Exception:
        pass

    try:
        bd = bpy.path.abspath("//")
        if bd:
            return bd
    except Exception:
        pass

    return os.getcwd()

SCRIPT_DIR = _script_dir()

def log(msg):
    """Imprime log."""
    try:
        print(f"[BLENDER] {msg}")
    except:
        pass

def abspath(p):
    """Convierte ruta relativa a absoluta."""
    if not p:
        return None

    try:
        if isinstance(p, str) and p.startswith("//"):
            return bpy.path.abspath(p)
    except Exception:
        pass

    return p if os.path.isabs(p) else os.path.abspath(os.path.join(SCRIPT_DIR, p))

# ============================================================================
# ARGUMENTOS
# ============================================================================

argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

ap = argparse.ArgumentParser()
ap.add_argument("--library", default=None, help="Ruta a poses_converted_video.json")
ap.add_argument("--poses", default="", help="Poses: 'A,B,C'")
ap.add_argument("--armature", default="Human.rig", help="Nombre del armature")
ap.add_argument("--fps", type=int, default=24, help="FPS de salida")
ap.add_argument("--engine", default="EEVEE", help="Motor: EEVEE o CYCLES")
ap.add_argument("--width", type=int, default=1080, help="Ancho")
ap.add_argument("--height", type=int, default=1080, help="Alto")
ap.add_argument("--out", default=None, help="Ruta de salida")
ap.add_argument("--camera_name", default="Camera", help="Nombre de cámara")
ap.add_argument("--frame_step", type=int, default=1, help="Procesar cada N frames")
ap.add_argument("--pose_duration", type=float, default=None, help="Segundos por pose")

args = ap.parse_args(argv)

# ============================================================================
# FUNCIONES DE CARGA Y APLICACIÓN
# ============================================================================

def load_json(path):
    """Carga JSON."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"❌ No se pudo leer JSON {path}: {e}")
        return None

def get_sequence():
    """Obtiene secuencia de poses."""
    if args.poses:
        return [s.strip() for s in args.poses.split(",") if s.strip()]
    raise RuntimeError("❌ Proporciona --poses")

def find_armature():
    """Encuentra el armature."""
    if args.armature:
        ob = bpy.data.objects.get(args.armature)
        if ob and ob.type == 'ARMATURE':
            return ob

    for o in bpy.data.objects:
        if o.type == 'ARMATURE':
            return o

    raise RuntimeError(f"❌ No se encontró Armature. Proporciona --armature")

def ensure_pose_mode(arm):
    """Asegura modo pose."""
    bpy.context.view_layer.objects.active = arm
    if bpy.context.mode != 'POSE':
        bpy.ops.object.mode_set(mode='POSE')

def reset_pose(arm):
    """Resetea la pose."""
    for pb in arm.pose.bones:
        try:
            pb.matrix_basis.identity()
        except:
            pass

def calculate_bone_rotation(pose_bone, start_world, end_world, armature):
    """Calcula la rotación local necesaria para alinear el hueso con la dirección target."""
    target_direction_world = (end_world - start_world)

    if target_direction_world.length < 1e-6:
        return Quaternion()

    target_direction_world.normalize()

    if pose_bone.parent:
        parent_matrix_world = armature.matrix_world @ pose_bone.parent.matrix
        parent_rotation_world = parent_matrix_world.to_quaternion()
    else:
        parent_rotation_world = armature.matrix_world.to_quaternion()

    target_direction_local = parent_rotation_world.inverted() @ target_direction_world

    bone = pose_bone.bone
    bone_rest_vector = bone.tail_local - bone.head_local

    if bone_rest_vector.length < 1e-6:
        return Quaternion()

    bone_rest_direction = bone_rest_vector.normalized()
    rotation_local = bone_rest_direction.rotation_difference(target_direction_local)
    return rotation_local

def apply_landmarks_pose(arm, landmarks):
    """Aplica rotaciones calculadas desde landmarks Blender-space (lista de 21 vectores)."""
    if len(landmarks) != 21:
        return []

    applied = []
    vectors = [Vector(lm) for lm in landmarks]
    transform, _scale = _transform_mp_to_rig(vectors, arm)

    for bone_name, (start_idx, end_idx) in MEDIAPIPE_TO_MAKEHUMAN.items():
        pose_bone = arm.pose.bones.get(bone_name)
        if not pose_bone:
            log(f"⚠️ Hueso no encontrado para landmarks: {bone_name}")
            continue

        start_vec = vectors[end_idx] - vectors[start_idx]
        if start_vec.length < 1e-6:
            continue
        target_vec = transform @ start_vec
        if target_vec.length < 1e-6:
            continue
        target_vec.normalize()

        bone_rest_vector = pose_bone.bone.tail_local - pose_bone.bone.head_local
        if bone_rest_vector.length < 1e-6:
            continue
        rest_dir = bone_rest_vector.normalized()
        rotation_local = rest_dir.rotation_difference(target_vec)

        pose_bone.rotation_mode = 'QUATERNION'
        pose_bone.rotation_quaternion = rotation_local
        applied.append(bone_name)

    bpy.context.view_layer.update()
    return applied

def _transform_mp_to_rig(landmarks, arm):
    if np is None:
        return Matrix.Identity(3), 1.0

    mp_vectors = []
    rig_vectors = []

    for (start_idx, end_idx), bone_name in REFERENCE_PAIRS:
        if start_idx >= len(landmarks) or end_idx >= len(landmarks):
            continue
        mp_vec = Vector(landmarks[end_idx]) - Vector(landmarks[start_idx])
        if mp_vec.length < 1e-6:
            continue
        bone_data = arm.data.bones.get(bone_name)
        if not bone_data:
            continue
        rig_vec = bone_data.tail_local - bone_data.head_local
        if rig_vec.length < 1e-6:
            continue
        mp_vectors.append(np.array(mp_vec, dtype=float))
        rig_vectors.append(np.array(rig_vec, dtype=float))

    if len(mp_vectors) < 2:
        return Matrix.Identity(3), 1.0

    A = np.stack(mp_vectors, axis=0)
    B = np.stack(rig_vectors, axis=0)

    cov = A.T @ B
    U, S, Vt = np.linalg.svd(cov)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    scale_den = np.sum(A * A)
    scale = float(np.sum(S) / scale_den) if scale_den > 1e-8 else 1.0

    matrix_rows = tuple(tuple(float(val) for val in row) for row in R)
    return Matrix(matrix_rows), scale

def apply_quat_to_bone(pose_bone, quat_w_x_y_z):
    """Aplica un quaternion a un hueso."""
    try:
        quat = Quaternion(quat_w_x_y_z)
        quat.normalize()
        pose_bone.rotation_mode = 'QUATERNION'
        pose_bone.rotation_quaternion = quat
        return True
    except Exception as e:
        log(f"⚠️ Error aplicando quat a {pose_bone.name}: {e}")
        return False

def apply_pose_dict(arm, bones_dict):
    """
    Aplica rotaciones desde un diccionario de huesos.

    Devuelve lista de huesos a los que se aplicó transformación.
    """
    applied = []

    for bone_name, bone_data in bones_dict.items():
        if bone_name not in arm.pose.bones:
            log(f"⚠️ Hueso no encontrado: {bone_name}")
            continue

        pb = arm.pose.bones[bone_name]

        # Aplicar rotación quaternion
        if 'rotation_quat' in bone_data:
            quat_data = bone_data['rotation_quat']
            if isinstance(quat_data, (list, tuple)) and len(quat_data) == 4:
                if apply_quat_to_bone(pb, quat_data):
                    applied.append(bone_name)

        # Aplicar posición (opcional)
        if 'position' in bone_data:
            try:
                pos_data = bone_data['position']
                pb.location = Vector([float(x) for x in pos_data])
                if bone_name not in applied:
                    applied.append(bone_name)
            except Exception:
                pass

    return applied

def insert_keyframes(arm, applied_bones, frame):
    """Inserta keyframes para los huesos aplicados."""
    if not applied_bones:
        return

    for bone_name in applied_bones:
        pb = arm.pose.bones.get(bone_name)
        if not pb:
            continue

        try:
            if pb.rotation_mode == 'QUATERNION':
                pb.keyframe_insert(data_path="rotation_quaternion", frame=frame)
            else:
                pb.keyframe_insert(data_path="rotation_euler", frame=frame)

            # Insertar keyframe de posición si tiene
            try:
                pb.keyframe_insert(data_path="location", frame=frame)
            except:
                pass
        except Exception as e:
            log(f"⚠️ Error con keyframe en {bone_name}: {e}")

def set_smooth_interpolation(action, mode="BEZIER"):
    """Establece interpolación suave."""
    if not action:
        return

    for fc in action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = mode

# ============================================================================
# MAIN
# ============================================================================

def main():
    log("="*70)
    log("RENDERIZAR VIDEO DESDE POSES_CONVERTED_VIDEO.JSON")
    log("="*70)

    lib_path = abspath(args.library) or os.path.join(SCRIPT_DIR, "poses_converted_video.json")

    poses_lib = load_json(lib_path) or {}

    if not poses_lib:
        raise RuntimeError(f"❌ No se pudo cargar: {lib_path}")

    log(f"✓ Biblioteca cargada desde: {lib_path}")
    log(f"✓ Frame step: {args.frame_step}")

    seq = get_sequence()
    log(f"✓ Secuencia de poses: {seq}")

    arm = find_armature()
    ensure_pose_mode(arm)
    log(f"✓ Armature: {arm.name}")

    scn = bpy.context.scene
    # Seleccionar un engine compatible con la versión de Blender en tiempo de ejecución.
    def _select_render_engine(preferred):
        p = (preferred or "").upper()
        candidates = []
        if p.startswith("EEVEE"):
            candidates = ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "EEVEE", "BLENDER_WORKBENCH"]
        elif "CYCLE" in p:
            candidates = ["CYCLES", "CYCLE"]
        else:
            candidates = [preferred]

        for c in candidates:
            try:
                scn.render.engine = c
                log(f"✓ Engine: {c}")
                return c
            except Exception:
                continue

        # último intento: dejar lo que tenga por defecto y notificar
        log(f"⚠️  No se pudo aplicar el engine preferido '{preferred}'; usando el engine por defecto de la escena: {scn.render.engine}")
        return scn.render.engine

    _select_render_engine(args.engine)
    scn.render.resolution_x = args.width
    scn.render.resolution_y = args.height
    scn.render.fps = args.fps

    log(f"✓ Renderizado: {args.engine} {args.width}x{args.height} @{args.fps}fps")

    # Configurar cámara
    if args.camera_name:
        cam = bpy.data.objects.get(args.camera_name)
        if cam:
            scn.camera = cam
            log(f"✓ Cámara: {args.camera_name}")

    # Procesar cada pose
    frame = 1
    scn.frame_start = frame

    for idx, pose_name in enumerate(seq):
        log(f"\n[{idx+1}/{len(seq)}] Procesando pose: {pose_name}")

        pose_entry = poses_lib.get("poses", {}).get(pose_name)

        if not pose_entry:
            log(f"  ⚠️ Pose no encontrada")
            continue

        frames_list = pose_entry.get("frames", [])

        if args.frame_step > 1:
            original_count = len(frames_list)
            frames_list = frames_list[::args.frame_step]
            log(f"  Frames: {original_count} → {len(frames_list)} (step={args.frame_step})")
        else:
            log(f"  Frames: {len(frames_list)}")

        # Aplicar frame-by-frame
        for frame_idx, frame_data in enumerate(frames_list):
            landmarks_blender = frame_data.get("landmarks_blender")
            bones_dict = frame_data.get("bones", {})

            if not landmarks_blender and not bones_dict:
                continue

            reset_pose(arm)

            if landmarks_blender and len(landmarks_blender) == 21:
                applied = apply_landmarks_pose(arm, landmarks_blender)
            else:
                applied = apply_pose_dict(arm, bones_dict)

            insert_keyframes(arm, applied, frame)

            frame += 1

        # Actualizar
        bpy.context.view_layer.update()

    scn.frame_end = max(1, frame - 1)

    log(f"\n✓ Animación lista: frames {scn.frame_start} → {scn.frame_end}")

    # Establecer interpolación suave
    if arm.animation_data and arm.animation_data.action:
        set_smooth_interpolation(arm.animation_data.action, "BEZIER")
        log(f"✓ Interpolación suavizada")

    # Renderizar
    out_path = None
    if args.out:
        out_path = abspath(args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    else:
        out_path = os.path.join(SCRIPT_DIR, "output", "out_video.mp4")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    scn.render.filepath = out_path

    # Configurar formato de salida según la extensión
    ext = os.path.splitext(out_path)[1].lower()
    if ext in {".mp4", ".mov", ".avi", ".mkv", ".mpg", ".mpeg"}:
        scn.render.image_settings.file_format = "FFMPEG"
        scn.render.image_settings.color_mode = "RGB"
        try:
            ff = scn.render.ffmpeg
            ff.format = "MPEG4"
            ff.codec = "H264"
            if hasattr(ff, "audio_codec"):
                ff.audio_codec = "AAC"
            if hasattr(ff, "gopsize") and scn.render.fps:
                ff.gopsize = max(1, int(scn.render.fps))
            for attr in ("pix_fmt", "pixel_format", "pixfmt"):
                if hasattr(ff, attr):
                    setattr(ff, attr, "YUV420P")
        except Exception as e:
            log(f"⚠️  Aviso FFmpeg: {e}")
    else:
        # fallback a secuencia de imágenes
        scn.render.image_settings.file_format = "PNG"
        scn.render.image_settings.color_mode = "RGB"

    log(f"\nRenderizando animación a: {out_path}")

    bpy.ops.render.render(animation=True)
    log(f"✓ Renderizado completado")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise
