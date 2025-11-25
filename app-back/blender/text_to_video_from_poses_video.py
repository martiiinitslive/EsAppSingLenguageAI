"""
Script para renderizar vídeo a partir de poses_converted_video.json (con información frame-by-frame).
Ejecutar dentro de Blender:
  blender archivo.blend --python text_to_video_from_poses_video.py -- --library poses_converted_video.json --poses "A,B" --frame_step 2
"""

import bpy
import os
import sys
import json
import argparse
from mathutils import Vector, Euler, Quaternion, Matrix

# ========== UTILIDADES ==========
def _script_dir():
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
    try: print(f"[BLENDER] {msg}")
    except: pass

def abspath(p):
    if not p:
        return None
    try:
        if isinstance(p, str) and p.startswith("//"):
            return bpy.path.abspath(p)
    except Exception:
        pass
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(SCRIPT_DIR, p))

# ========== ARGUMENTOS ==========
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

ap = argparse.ArgumentParser()
ap.add_argument("--library", default=None, help="Ruta a poses_converted_video.json")
ap.add_argument("--poses", default="", help="Poses: 'A,B,C'")
ap.add_argument("--text", default="", help="Texto: 'ABC'")
ap.add_argument("--armature", default="", help="Nombre del armature")
ap.add_argument("--fps", type=int, default=24, help="FPS de salida")
ap.add_argument("--engine", default="EEVEE", help="Motor: EEVEE o CYCLES")
ap.add_argument("--width", type=int, default=1080, help="Ancho en píxeles")
ap.add_argument("--height", type=int, default=1080, help="Alto en píxeles")
ap.add_argument("--out", default=None, help="Ruta de salida")
ap.add_argument("--camera_lib", default=None, help="Ruta a camera_library.json")
ap.add_argument("--camera_name", default="Cam_01", help="Nombre de cámara")
ap.add_argument("--frame_step", type=int, default=1, help="Procesar cada N frames")
ap.add_argument("--skip_defocus", action="store_true", help="Desactivar desenfoque (más rápido)")
ap.add_argument("--pose_duration", type=float, default=None, help="Si se indica, comprime cada pose a N segundos (float)")
args = ap.parse_args(argv)

# ========== FUNCIONES DE POSES ==========
def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"❌ No se pudo leer JSON {path}: {e}")
        return None

def get_sequence():
    if args.poses:
        return [s.strip() for s in args.poses.split(",") if s.strip()]
    if args.text:
        return [ch for ch in args.text if not ch.isspace()]
    raise RuntimeError("❌ Proporciona --poses o --text")

def find_armature():
    if args.armature:
        ob = bpy.data.objects.get(args.armature)
        if ob and ob.type == 'ARMATURE':
            return ob
    ob = bpy.context.object
    if ob and ob.type == 'ARMATURE':
        return ob
    for o in bpy.context.selected_objects:
        if o.type == 'ARMATURE':
            return o
    for o in bpy.data.objects:
        if o.type == 'ARMATURE':
            return o
    raise RuntimeError("❌ No se encontró Armature. Proporciona --armature o selecciona uno.")

def ensure_pose_mode(arm):
    bpy.context.view_layer.objects.active = arm
    if bpy.context.mode != 'POSE':
        bpy.ops.object.mode_set(mode='POSE')

def reset_pose(arm):
    for pb in arm.pose.bones:
        try:
            pb.matrix_basis.identity()
        except:
            pass

BONE_MAP = {}  # mapping json_name -> rig_bone_name (cached)

def build_bone_map(arm, sample_keys):
    """
    Construye un mapeo entre nombres de huesos del JSON y nombres del armature.
    Estrategia: exact match -> lowercase exact -> substring -> prefix/suffix.
    """
    if not sample_keys:
        return {}
    rig_names = [b.name for b in arm.pose.bones]
    map_out = {}
    rig_lower = {r.lower(): r for r in rig_names}
    for k in sample_keys:
        if k in rig_names:
            map_out[k] = k
            continue
        kl = k.lower()
        if kl in rig_lower:
            map_out[k] = rig_lower[kl]
            continue
        # substring match
        found = None
        for r in rig_names:
            rl = r.lower()
            if kl in rl or rl in kl:
                found = r
                break
        if found:
            map_out[k] = found
            continue
        # try replacing common separators
        k2 = kl.replace("-", "_").replace(" ", "_")
        for r in rig_names:
            if k2 in r.lower():
                found = r
                break
        if found:
            map_out[k] = found
            continue
        # no match
        map_out[k] = None
    return map_out

# Replace apply_pose_dict + insert_keys_for with mapping-aware versions
def apply_pose_dict(arm, bones_dict):
    """
    Aplica rotaciones/loc usando BONE_MAP. Devuelve lista de rig bone names a las que se aplicó algo.
    """
    global BONE_MAP
    applied = []

    if not BONE_MAP:
        # construir mapa usando las keys actuales
        BONE_MAP = build_bone_map(arm, list(bones_dict.keys()))
        log(f"✓ Bone map creado (ejemplo {min(10, len(BONE_MAP))}): {dict(list(BONE_MAP.items())[:10])}")

    def _to_blender_quat(q):
        try:
            qf = [float(x) for x in q]
        except Exception:
            return None
        # heurística: preferir (w,x,y,z) pero si parece (x,y,z,w) convertir
        # si q length !=4 return None
        if len(qf) != 4:
            return None
        # if last component bigger than first, maybe (x,y,z,w)
        if abs(qf[3]) > abs(qf[0]):
            quat = Quaternion((qf[3], qf[0], qf[1], qf[2]))
        else:
            quat = Quaternion((qf[0], qf[1], qf[2], qf[3]))
        try:
            quat.normalize()
        except Exception:
            pass
        return quat

    for json_name, data in bones_dict.items():
        rig_name = BONE_MAP.get(json_name)
        if not rig_name:
            # reportar para debugging pero no spam
            log(f"⚠️  Sin mapeo para JSON-bone '{json_name}'")
            continue
        pb = arm.pose.bones.get(rig_name)
        if not pb:
            log(f"⚠️  Rig bone no encontrado: '{rig_name}' (mapeado desde '{json_name}')")
            continue

        rq = data.get("rot_quat")
        re = data.get("rot_euler")
        if rq:
            pb.rotation_mode = 'QUATERNION'
            quat = _to_blender_quat(rq)
            if quat is None:
                log(f"⚠️  Cuaternión inválido para {json_name}: {rq}")
            else:
                try:
                    pb.rotation_quaternion = quat
                    applied.append(rig_name)
                except Exception as e:
                    log(f"⚠️  Error aplicando rot_quat a {rig_name}: {e}")
        elif re:
            try:
                pb.rotation_mode = 'XYZ'
                pb.rotation_euler = Euler([float(re[0]), float(re[1]), float(re[2])], 'XYZ')
                applied.append(rig_name)
            except Exception as e:
                log(f"⚠️  Error aplicando rot_euler a {rig_name}: {e}")

        if "loc" in data and data["loc"] is not None:
            try:
                pb.location = Vector([float(x) for x in data["loc"]])
                if rig_name not in applied:
                    applied.append(rig_name)
            except Exception:
                pass

    return applied

def insert_keys_for(arm, applied_rig_names, frame):
    """
    Inserta keyframes sólo en la lista de rig bone names aplicada.
    """
    if not applied_rig_names:
        return
    for rig_name in applied_rig_names:
        pb = arm.pose.bones.get(rig_name)
        if not pb:
            continue
        try:
            if pb.rotation_mode == 'QUATERNION':
                pb.keyframe_insert(data_path="rotation_quaternion", frame=frame, group="POSE")
            else:
                pb.keyframe_insert(data_path="rotation_euler", frame=frame, group="POSE")
            # location key
            try:
                pb.keyframe_insert(data_path="location", frame=frame, group="POSE")
            except Exception:
                pass
        except Exception as e:
            log(f"⚠️  Error insertando key en {rig_name} frame {frame}: {e}")

def set_interpolation(action, mode="BEZIER"):
    if not action:
        return
    for fc in action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = mode

def apply_camera_from_lib_if_present(camera_lib_path, camera_name):
    camera_lib_path = abspath(camera_lib_path) if camera_lib_path else os.path.join(SCRIPT_DIR, "camera_library.json")
    try:
        lib = load_json(camera_lib_path)
        if lib:
            entry = lib.get("cameras", {}).get(camera_name)
            if entry:
                obj_name = entry.get("object_name", "Camera")
                cam_obj = bpy.data.objects.get(obj_name) or bpy.context.scene.camera
                if cam_obj is None:
                    cam_data = bpy.data.cameras.new(obj_name + "_data")
                    cam_obj = bpy.data.objects.new(obj_name, cam_data)
                    bpy.context.collection.objects.link(cam_obj)

                tr = entry.get("transform", {}) or {}
                loc = tr.get("location")
                try:
                    if loc and len(loc) >= 3:
                        loc_v = Vector((float(loc[0]), float(loc[1]), float(loc[2])))
                    else:
                        loc_v = cam_obj.matrix_world.to_translation()

                    if "rotation_quaternion" in tr:
                        q = tr["rotation_quaternion"]
                        quat = Quaternion((float(q[0]), float(q[1]), float(q[2]), float(q[3])))
                        mat_rot = quat.to_matrix().to_4x4()
                    elif "rotation_euler" in tr:
                        e = tr["rotation_euler"]
                        mat_rot = Euler((float(e[0]), float(e[1]), float(e[2])), 'XYZ').to_matrix().to_4x4()
                    else:
                        mat_rot = cam_obj.matrix_world.to_3x3().to_4x4()

                    cam_obj.parent = None
                    cam_obj.matrix_world = Matrix.Translation(loc_v) @ mat_rot
                    bpy.context.scene.camera = cam_obj
                except Exception as e:
                    log(f"⚠️  Error aplicando transform de cámara: {e}")
                return True
    except Exception as e:
        log(f"⚠️  Error cargando camera library: {e}")
    return False

def mute_defocus_nodes():
    if not args.skip_defocus:
        return
    try:
        scn = bpy.context.scene
        if getattr(scn, "use_nodes", False) and scn.node_tree:
            for n in scn.node_tree.nodes:
                if getattr(n, "type", "") in ("DEFOCUS", "BLUR"):
                    try:
                        n.mute = True
                    except:
                        pass
    except Exception:
        pass

# ========== MAIN ==========
def main():
    log("=" * 70)
    log("RENDERIZAR VIDEO DESDE POSES_CONVERTED_VIDEO.JSON (frame-by-frame)")
    log("=" * 70)

    lib_path = abspath(args.library) or os.path.join(SCRIPT_DIR, "poses_converted_video.json")
    poses_lib = load_json(lib_path) or {}

    if not poses_lib:
        raise RuntimeError(f"❌ No se pudo cargar: {lib_path}")

    log(f"✓ Biblioteca cargada desde: {lib_path}")
    log(f"✓ Frame step: {args.frame_step} (procesar cada {args.frame_step} frames)")

    if args.skip_defocus:
        log(f"✓ Desenfoque desactivado")

    if args.camera_lib:
        if apply_camera_from_lib_if_present(args.camera_lib, args.camera_name):
            log(f"✓ Cámara aplicada desde {args.camera_lib}")
            mute_defocus_nodes()
        else:
            log("⚠️  No se pudo aplicar camera_lib")

    seq = get_sequence()
    log(f"✓ Secuencia: {seq}")

    arm = find_armature()
    ensure_pose_mode(arm)
    log(f"✓ Armature: {arm.name}")

    scn = bpy.context.scene
    frame = 1
    scn.frame_start = frame

    total_frames = 0

    for idx, pose_name in enumerate(seq):
        log(f"\nProcesando pose [{idx+1}/{len(seq)}]: {pose_name}")

        pose_entry = poses_lib.get("poses", {}).get(pose_name)

        if not pose_entry:
            log(f"  ⚠️  Pose '{pose_name}' no encontrada en biblioteca")
            continue

        if "frames" in pose_entry:
            frames_list = pose_entry.get("frames", [])
            video_info = pose_entry.get("video_info", {})
            if args.frame_step > 1:
                original_count = len(frames_list)
                frames_list = frames_list[::args.frame_step]
                log(f"  ℹ  Vídeo: {original_count} frames originales → {len(frames_list)} frames procesados (step={args.frame_step})")
            else:
                log(f"  ℹ  Vídeo: {len(frames_list)} frames, {video_info.get('fps', 30)} FPS")

            # si se pidió comprimir cada pose a N segundos, muestrear frames equiespaciados
            if args.pose_duration and args.pose_duration > 0:
                target_frames = max(1, int(round(args.pose_duration * args.fps)))
                frames_list = _sample_frames(frames_list, target_frames)
                log(f"  ℹ  Comprimido a {args.pose_duration}s → {len(frames_list)} frames (fps salida={args.fps})")

            # DEBUG: mostrar estructura del primer frame para inspección
            if len(frames_list) > 0:
                try:
                    first = frames_list[0]
                    log(f"  DEBUG primer frame keys: {list(first.keys())}")
                    if isinstance(first.get("bones"), dict):
                        sample_bones = dict(list(first.get("bones").items())[:3])
                        log(f"  DEBUG ejemplo bones (hasta 3): {sample_bones}")
                    else:
                        log(f"  DEBUG bones no es dict, tipo: {type(first.get('bones'))}")
                except Exception as e:
                    log(f"  DEBUG fallo leyendo primer frame: {e}")

            for frame_data in frames_list:
                # usar clave 'hand_detected' si existe, si no asumir True
                hand_ok = frame_data.get("hand_detected", True)
                if not hand_ok:
                    reset_pose(arm)
                    insert_keys_for(arm, {}, frame)
                    frame += 1
                    continue

                bones_dict = frame_data.get("bones", {}) or {}
                if not bones_dict:
                    reset_pose(arm)
                    insert_keys_for(arm, {}, frame)
                    frame += 1
                    continue

                reset_pose(arm)
                apply_pose_dict(arm, bones_dict)
                insert_keys_for(arm, bones_dict, frame)

                frame += 1

            total_frames += len(frames_list)

        elif "bones" in pose_entry:
            bones_dict = pose_entry.get("bones", {})
            reset_pose(arm)
            apply_pose_dict(arm, bones_dict)
            insert_keys_for(arm, bones_dict, frame)
            frame += 1
            total_frames += 1

    scn.frame_end = max(scn.frame_start + 1, frame)

    log(f"\n✓ Total frames aplicados (estimado): {total_frames}")
    log(f"✓ Rango de frames: {scn.frame_start}..{scn.frame_end}")

    if arm.animation_data and arm.animation_data.action:
        set_interpolation(arm.animation_data.action, "BEZIER")
        log("✓ Interpolación configurada a BEZIER")

    engine = str(args.engine).upper()
    for eng in (["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "EEVEE"] if engine.startswith("EEVEE") else ["CYCLES", "CYCLE"]):
        try:
            scn.render.engine = eng
            log(f"✓ Engine: {eng}")
            break
        except Exception:
            continue

    scn.render.resolution_x = args.width
    scn.render.resolution_y = args.height
    scn.render.resolution_percentage = 100
    scn.render.pixel_aspect_x = 1.0
    scn.render.pixel_aspect_y = 1.0
    scn.render.fps = args.fps

    out_path = abspath(args.out) or os.path.join(SCRIPT_DIR, "output", "out_video.mp4")
    out_dir = os.path.dirname(out_path) or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    scn.render.filepath = out_path

    scn.render.image_settings.file_format = 'FFMPEG'
    scn.render.image_settings.color_mode = 'RGB'
    try:
        ff = scn.render.ffmpeg
        ff.format = 'MPEG4'
        ff.codec = 'H264'
        if hasattr(ff, "audio_codec"):
            ff.audio_codec = 'AAC'
        if hasattr(ff, "gopsize"):
            ff.gopsize = max(1, args.fps)
        for pix_attr in ("pix_fmt", "pixfmt", "pixel_format"):
            if hasattr(ff, pix_attr):
                try:
                    setattr(ff, pix_attr, 'YUV420P')
                    break
                except:
                    pass
    except Exception as e:
        log(f"⚠️  Aviso FFmpeg: {e}")

    log(f"\n✓ Configurado para render")
    log(f"  Resolución: {args.width}x{args.height}")
    log(f"  FPS: {args.fps}")
    log(f"  Frame step: {args.frame_step}")
    log(f"  Salida: {out_path}")
    log(f"\nIniciando render...\n")

    try:
        bpy.ops.render.render(animation=True)
        log(f"\n✅ Render completado")
        log(f"   Archivo: {out_path}")
    except Exception as e:
        log(f"\n❌ Error durante render: {e}")
        raise

def _sample_frames(frames_list, k):
    if not frames_list:
        return []
    n = len(frames_list)
    if k >= n:
        return list(frames_list)
    if k == 1:
        return [frames_list[n//2]]
    out = []
    for i in range(k):
        idx = int(round(i * (n - 1) / float(k - 1)))
        out.append(frames_list[idx])
    return out

if __name__ == "__main__":
    main()