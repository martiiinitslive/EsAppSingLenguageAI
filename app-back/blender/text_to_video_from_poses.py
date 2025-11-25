# text_to_video_from_poses.py
import bpy, os, sys, json, argparse
from mathutils import Vector, Euler, Quaternion, Matrix
import math

# ========== utilidades ==========
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
    if not p: return None
    try:
        if isinstance(p, str) and p.startswith("//"):
            return bpy.path.abspath(p)
    except Exception:
        pass
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(SCRIPT_DIR, p))

# ========== CLI (cuando se ejecuta desde blender --python) ==========
argv = sys.argv
argv = argv[argv.index("--")+1:] if "--" in argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--library", default=None)        # poses library path
ap.add_argument("--poses", default="")            # "A,B"
ap.add_argument("--text", default="")             # "ABC"
ap.add_argument("--armature", default="")         # nombre armature
ap.add_argument("--fps", type=int, default=24)
ap.add_argument("--hold", type=int, default=12)
ap.add_argument("--transition", type=int, default=12)
ap.add_argument("--engine", default="EEVEE")
ap.add_argument("--width", type=int, default=1080)
ap.add_argument("--height", type=int, default=1080)
ap.add_argument("--out", default=None)
ap.add_argument("--camera_lib", default=None)
ap.add_argument("--camera_name", default="Cam_01")
args = ap.parse_args(argv)

# ========== helpers para poses ==========
def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"no se pudo leer JSON {path}: {e}")
        return None

def get_sequence():
    if args.poses:
        return [s.strip() for s in args.poses.split(",") if s.strip()]
    if args.text:
        return [ch for ch in args.text if not ch.isspace()]
    raise RuntimeError("Indica --poses \"A,B\" o --text \"AB\"")

def find_armature():
    if args.armature:
        ob = bpy.data.objects.get(args.armature)
        if ob and ob.type == 'ARMATURE': return ob
    ob = bpy.context.object
    if ob and ob.type == 'ARMATURE': return ob
    for o in bpy.context.selected_objects:
        if o.type == 'ARMATURE': return o
    for o in bpy.data.objects:
        if o.type == 'ARMATURE': return o
    raise RuntimeError("No se encontró un Armature. Pasa --armature o selecciónalo activo.")

def ensure_pose_mode(arm):
    bpy.context.view_layer.objects.active = arm
    if bpy.context.mode != 'POSE':
        bpy.ops.object.mode_set(mode='POSE')

def reset_pose(arm):
    for pb in arm.pose.bones:
        try: pb.matrix_basis.identity()
        except: pass

def apply_pose_dict(arm, bones_dict):
    for name, data in bones_dict.items():
        pb = arm.pose.bones.get(name)
        if not pb: continue
        rq = data.get("rot_quat"); re = data.get("rot_euler")
        if rq:
            pb.rotation_mode = 'QUATERNION'
            pb.rotation_quaternion = rq
        elif re:
            pb.rotation_mode = 'XYZ'
            pb.rotation_euler = re
        if "loc" in data:
            pb.location = data["loc"]

def insert_keys_for(arm, bones_dict, frame):
    for name in bones_dict.keys():
        pb = arm.pose.bones.get(name)
        if not pb: continue
        try:
            if pb.rotation_mode == 'QUATERNION':
                pb.keyframe_insert(data_path="rotation_quaternion", frame=frame, group="POSE")
            else:
                pb.keyframe_insert(data_path="rotation_euler", frame=frame, group="POSE")
            if "loc" in bones_dict[name]:
                pb.keyframe_insert(data_path="location", frame=frame, group="POSE")
        except Exception:
            pass

def set_interpolation(action, mode="BEZIER"):
    if not action: return
    for fc in action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = mode

# ========== cámara: intentar delegar a camera_library si existe ==========
def apply_camera_from_lib_if_present(camera_lib_path, camera_name):
    camera_lib_path = abspath(camera_lib_path) if camera_lib_path else os.path.join(SCRIPT_DIR, "camera_library.json")
    try:
        # preferir importar el módulo local camera_library si está disponible
        if os.path.exists(os.path.join(SCRIPT_DIR, "camera_library.py")):
            import importlib.util
            spec = importlib.util.spec_from_file_location("camera_library", os.path.join(SCRIPT_DIR, "camera_library.py"))
            cammod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cammod)
            # el módulo puede exponer apply_from_lib o apply_camera_from_lib o apply_camera_dict
            if hasattr(cammod, "apply_from_lib"):
                return cammod.apply_from_lib(camera_lib_path, cam_name=camera_name, lock_focus=True)
            if hasattr(cammod, "apply_camera_from_lib"):
                return cammod.apply_camera_from_lib(camera_lib_path, cam_name=camera_name, lock_focus=True)
            if hasattr(cammod, "apply_camera_from_lib"):
                return cammod.apply_camera_from_lib(camera_lib_path, cam_name=camera_name)
        # fallback: intentar leer JSON y aplicar la entrada mínima
        lib = load_json(camera_lib_path)
        if lib:
            entry = lib.get("cameras", {}).get(camera_name)
            if entry:
                # aplicar transform de forma simple (similar a camera module)
                obj_name = entry.get("object_name", "Camera")
                cam_obj = bpy.data.objects.get(obj_name) or bpy.context.scene.camera
                if cam_obj is None:
                    cam_data = bpy.data.cameras.new(obj_name + "_data")
                    cam_obj = bpy.data.objects.new(obj_name, cam_data)
                    bpy.context.collection.objects.link(cam_obj)
                # aplicar transform (world-space) -- simple operación
                tr = entry.get("transform", {}) or {}
                loc = tr.get("location")
                rot_mode = tr.get("rotation_mode", "XYZ")
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
                        mat_rot = Euler((float(e[0]), float(e[1]), float(e[2])), rot_mode).to_matrix().to_4x4()
                    else:
                        mat_rot = cam_obj.matrix_world.to_3x3().to_4x4()
                    cam_obj.parent = None
                    cam_obj.matrix_world = Matrix.Translation(loc_v) @ mat_rot
                    bpy.context.scene.camera = cam_obj
                except Exception as e:
                    log(f"fallback apply camera failed: {e}")
                return True
    except Exception as e:
        log(f"apply camera module fail: {e}")
    return False

def mute_defocus_nodes():
    try:
        scn = bpy.context.scene
        if getattr(scn, "use_nodes", False) and scn.node_tree:
            for n in scn.node_tree.nodes:
                if getattr(n, "type", "") in ("DEFOCUS","BLUR"):
                    try: n.mute = True
                    except: pass
    except Exception:
        pass

# ========== workflow principal ==========
def main():
    # cargar poses lib
    lib_path = abspath(args.library) or os.path.join(SCRIPT_DIR, "poses_library.json")
    poses_lib = load_json(lib_path) or {}
    log(f"Cargando biblioteca de poses desde: {lib_path}")

    # aplicar cámara si existe lib
    if args.camera_lib:
        if apply_camera_from_lib_if_present(args.camera_lib, args.camera_name):
            log(f"Camera aplicada desde {args.camera_lib}")
            mute_defocus_nodes()
        else:
            log("No se pudo aplicar camera_lib: se usará la cámara existente en la escena.")

    seq = get_sequence()
    log(f"Secuencia: {seq}")

    arm = find_armature()
    ensure_pose_mode(arm)

    # preparar frames
    scn = bpy.context.scene
    frame = 1
    scn.frame_start = frame

    for idx, pose_name in enumerate(seq):
        log(f"Aplicando pose {pose_name} ({idx+1}/{len(seq)})")
        if "poses" in poses_lib:
            entry = poses_lib["poses"].get(pose_name)
            if not entry:
                raise KeyError(f"Pose '{pose_name}' no existe en {lib_path}")
            bones_dict = entry.get("bones", {})
        elif "bones" in poses_lib:
            bones_dict = poses_lib["bones"]
        else:
            raise RuntimeError(f"Biblioteca de poses inválida o vacía: {lib_path}")

        reset_pose(arm)
        apply_pose_dict(arm, bones_dict)
        insert_keys_for(arm, bones_dict, frame)

        frame_hold_end = frame + max(0, args.hold)
        if args.hold > 0:
            apply_pose_dict(arm, bones_dict)
            insert_keys_for(arm, bones_dict, frame_hold_end)

        frame = frame_hold_end + max(0, args.transition)

    scn.frame_end = max(scn.frame_start + 1, frame)

    if arm.animation_data and arm.animation_data.action:
        set_interpolation(arm.animation_data.action, "BEZIER")

    # render config
    engine = str(args.engine).upper()
    for eng in (["BLENDER_EEVEE_NEXT","BLENDER_EEVEE","EEVEE"] if engine.startswith("EEVEE") else ["CYCLES","CYCLE"]):
        try:
            scn.render.engine = eng
            log(f"Engine establecido: {eng}")
            break
        except Exception:
            continue

    scn.render.resolution_x = args.width
    scn.render.resolution_y = args.height
    scn.render.resolution_percentage = 100
    # opcional: forzar pixel aspect
    scn.render.pixel_aspect_x = 1.0
    scn.render.pixel_aspect_y = 1.0
    scn.render.fps = args.fps

    out_path = abspath(args.out) or os.path.join(SCRIPT_DIR, "output", "out.mp4")
    out_dir = os.path.dirname(out_path) or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    scn.render.filepath = out_path

    scn.render.image_settings.file_format = 'FFMPEG'
    scn.render.image_settings.color_mode = 'RGB'
    try:
        ff = scn.render.ffmpeg
        ff.format = 'MPEG4'
        ff.codec = 'H264'
        if hasattr(ff, "audio_codec"): ff.audio_codec = 'AAC'
        if hasattr(ff, "gopsize"): ff.gopsize = max(1, args.fps)
        for pix_attr in ("pix_fmt","pixfmt","pixel_format"):
            if hasattr(ff, pix_attr):
                try: setattr(ff, pix_attr, 'YUV420P'); break
                except: pass
    except Exception as e:
        log(f"Aviso: no se pudo configurar FFmpeg: {e}")

    log(f"Preparado para render. Rango: {scn.frame_start}..{scn.frame_end}  Salida: {out_path}")
    try:
        bpy.ops.render.render(animation=True)
    except Exception as e:
        log(f"Error durante el render: {e}")
        raise

if __name__ == "__main__":
    main()
