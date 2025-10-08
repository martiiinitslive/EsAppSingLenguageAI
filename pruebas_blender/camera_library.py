# Guarda la configuración (ubicación + enfoque) de la cámara activa en un JSON junto al .blend.
# para ponerlo en script de blender y ejecutarlo
# guarda la pose actual del armature activo en una biblioteca JSON
# PARA EJECUTAR DESDE POWERSHELL:
# & "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe" -b "C:\Users\marti\Desktop\tfg_teleco\proyectos\EsAppSingLenguageAI\pruebas_blender\cuerpo_humano_rigged.blend" --python "C:\Users\marti\Desktop\tfg_teleco\proyectos\EsAppSingLenguageAI\pruebas_blender\camera_library.py"
import bpy, json, os, sys
from mathutils import Vector, Euler, Quaternion, Matrix

# ========== CONFIG ==========
CAM_NAME = "Cam_01"               # nombre bajo el que se guarda esta cámara en la biblioteca
TARGET_NAME = "Human.rig"         # nombre exacto del objeto que debe quedar centrado (armature/malla)
DEFAULT_APERTURE = 4.0            # f-stop por defecto si no hay valor en la cámara
# ============================

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
LIB_PATH = os.path.join(SCRIPT_DIR, "camera_library.json")

def _log(msg):
    print(f"[camera_library] {msg}")

def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _save_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_scene_camera():
    scn = bpy.context.scene
    return getattr(scn, "camera", None) or next((o for o in bpy.data.objects if o.type == "CAMERA"), None)

def find_target(name=TARGET_NAME):
    if name and name in bpy.data.objects:
        return bpy.data.objects[name]
    for o in bpy.data.objects:
        n = (o.name or "").lower()
        if ("human" in n or "rig" in n or "cuerpo" in n) and o.type in ("ARMATURE", "MESH"):
            return o
    return next((o for o in bpy.data.objects if o.type in ("ARMATURE","MESH")), None)

def serialize_camera(cam):
    mw = cam.matrix_world
    loc = mw.to_translation()
    try:
        e = mw.to_euler(cam.rotation_mode)
        rot = {"rotation_mode": cam.rotation_mode, "rotation_euler": [float(e.x), float(e.y), float(e.z)]}
    except Exception:
        q = mw.to_quaternion()
        rot = {"rotation_mode": "QUATERNION", "rotation_quaternion": [float(q.w), float(q.x), float(q.y), float(q.z)]}
    data = cam.data
    dof = {}
    try:
        dof["dof_focus_object"] = getattr(data.dof, "focus_object", None).name if getattr(data.dof, "focus_object", None) else None
        dof["dof_focus_distance"] = float(getattr(data.dof, "focus_distance", 0.0))
        dof["aperture_fstop"] = float(getattr(data.dof, "aperture_fstop", DEFAULT_APERTURE))
    except Exception:
        dof = {"dof_focus_object": None, "dof_focus_distance": 0.0, "aperture_fstop": DEFAULT_APERTURE}
    cam_info = {
        "object_name": cam.name,
        "transform": {"location": [float(loc.x), float(loc.y), float(loc.z)], **rot},
        "data": {
            "lens_mm": float(getattr(data, "lens", 50.0)),
            "sensor_width": float(getattr(data, "sensor_width", 36.0)),
            "sensor_height": float(getattr(data, "sensor_height", 24.0)),
            "sensor_fit": str(getattr(data, "sensor_fit", "AUTO")),
            "shift_x": float(getattr(data, "shift_x", 0.0)),
            "shift_y": float(getattr(data, "shift_y", 0.0)),
            **dof
        },
        "scene_camera": getattr(bpy.context.scene, "camera").name if getattr(bpy.context.scene, "camera", None) else None,
        "frame": int(getattr(bpy.context.scene, "frame_current", 0))
    }
    return cam_info

def _apply_transform_from_dict(cam_obj, tdict):
    tr = tdict or {}
    loc = tr.get("location")
    rot_mode = tr.get("rotation_mode", None)
    try:
        loc_v = Vector((float(loc[0]), float(loc[1]), float(loc[2]))) if loc and len(loc) >= 3 else cam_obj.matrix_world.to_translation()
        if "rotation_quaternion" in tr:
            q = tr["rotation_quaternion"]
            mat_rot = Quaternion((float(q[0]), float(q[1]), float(q[2]), float(q[3]))).to_matrix().to_4x4()
        elif "rotation_euler" in tr:
            e = tr["rotation_euler"]
            rm = rot_mode or cam_obj.rotation_mode or "XYZ"
            mat_rot = Euler((float(e[0]), float(e[1]), float(e[2])), rm).to_matrix().to_4x4()
        else:
            mat_rot = cam_obj.matrix_world.to_3x3().to_4x4()
        target_mw = Matrix.Translation(loc_v) @ mat_rot
    except Exception:
        target_mw = cam_obj.matrix_world.copy()

    # mute constraints, detach parent, apply matrix_world, restore parent/constraints
    cons = [(c, getattr(c, "mute", False)) for c in getattr(cam_obj, "constraints", [])]
    for c, _ in cons:
        try: c.mute = True
        except Exception: pass
    old_parent = cam_obj.parent
    try:
        cam_obj.parent = None
        cam_obj.matrix_world = target_mw
        if old_parent:
            cam_obj.parent = old_parent
            try:
                cam_obj.matrix_parent_inverse = old_parent.matrix_world.inverted() @ target_mw
            except Exception:
                pass
    finally:
        for c, prev in cons:
            try: c.mute = prev
            except Exception: pass

def _apply_camera_entry(entry, cam_obj=None):
    if not entry:
        return False
    cam_obj = cam_obj or get_scene_camera()
    if cam_obj is None:
        return False
    _apply_transform_from_dict(cam_obj, entry.get("transform"))
    # camera data
    cinfo = entry.get("data", {}) or {}
    try:
        cam_obj.data.lens = cinfo.get("lens_mm", cam_obj.data.lens)
        cam_obj.data.sensor_width = cinfo.get("sensor_width", cam_obj.data.sensor_width)
        cam_obj.data.sensor_height = cinfo.get("sensor_height", cam_obj.data.sensor_height)
        cam_obj.data.sensor_fit = cinfo.get("sensor_fit", cam_obj.data.sensor_fit)
        cam_obj.data.shift_x = cinfo.get("shift_x", cam_obj.data.shift_x)
        cam_obj.data.shift_y = cinfo.get("shift_y", cam_obj.data.shift_y)
    except Exception:
        pass
    # DOF
    if hasattr(cam_obj.data, "dof"):
        dof = cam_obj.data.dof
        dof.use_dof = True
        target_name = cinfo.get("dof_focus_object")
        if target_name and target_name in bpy.data.objects:
            tgt = bpy.data.objects[target_name]
            dof.focus_object = tgt
            try:
                dof.focus_distance = (cam_obj.matrix_world.to_translation() - tgt.matrix_world.to_translation()).length
            except Exception:
                pass
        else:
            dof.focus_object = None
            if cinfo.get("dof_focus_distance") is not None:
                try: dof.focus_distance = float(cinfo.get("dof_focus_distance"))
                except Exception: pass
        if cinfo.get("aperture_fstop") is not None:
            try: dof.aperture_fstop = float(cinfo.get("aperture_fstop"))
            except Exception: pass
    try:
        bpy.context.scene.camera = cam_obj
    except Exception:
        pass
    return True

def lock_focus_frames(cam_obj, target_obj, keyframe=True):
    if not cam_obj or not target_obj:
        return
    scn = bpy.context.scene
    start = int(getattr(scn, "frame_start", 1) or 1)
    end = int(getattr(scn, "frame_end", start) or start)
    cur = scn.frame_current
    try:
        cam_obj.data.dof.use_dof = True
    except Exception:
        pass
    for f in range(start, end + 1):
        try:
            scn.frame_set(f)
            dist = (cam_obj.matrix_world.to_translation() - target_obj.matrix_world.to_translation()).length
            cam_obj.data.dof.focus_distance = float(dist)
            if keyframe:
                try:
                    cam_obj.data.keyframe_insert(data_path="dof.focus_distance", frame=f)
                except Exception:
                    pass
        except Exception:
            pass
    try: scn.frame_set(cur)
    except Exception: pass

def apply_from_lib(lib_path=LIB_PATH, cam_name=CAM_NAME, target_name=TARGET_NAME, lock_focus=True):
    lib_path = os.path.abspath(lib_path)
    lib = _load_json(lib_path)
    if not lib:
        _log(f"library not found: {lib_path}")
        return False
    entry = lib.get("cameras", {}).get(cam_name)
    if not entry:
        _log(f"camera entry '{cam_name}' missing in {lib_path}")
        return False
    ok = _apply_camera_entry(entry)
    if ok:
        # silence common compositor DOF/BLUR nodes
        try:
            scn = bpy.context.scene
            if getattr(scn, "use_nodes", False) and scn.node_tree:
                for n in scn.node_tree.nodes:
                    if getattr(n, "type", "") in ("DEFOCUS", "BLUR"):
                        try: n.mute = True
                        except Exception: pass
        except Exception:
            pass
        if lock_focus:
            cam = get_scene_camera()
            tgt = find_target(target_name)
            if cam and tgt:
                lock_focus_frames(cam, tgt, keyframe=True)
        _log("camera applied from lib")
        return True
    return False

def save_current(cam_name=CAM_NAME, lib_path=LIB_PATH):
    cam = get_scene_camera()
    if not cam:
        _log("no camera found to save")
        return False
    lib = _load_json(lib_path) or {"meta": {"created_by": "camera_library.py"}, "cameras": {}}
    lib.setdefault("cameras", {})[cam_name] = serialize_camera(cam)
    _save_json(lib_path, lib)
    _log(f"camera saved -> {lib_path}")
    return True

# CLI: default behavior: save; with --apply apply and exit
if __name__ == "__main__":
    if "--apply" in sys.argv:
        ok = apply_from_lib()
        sys.exit(0 if ok else 1)
    else:
        # when called from Blender UI run main save
        save_current()
