# ============================================
# 
# & "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe" `
#   "C:\Users\marti\Desktop\tfg_teleco\proyectos\EsAppSingLenguageAI\app-back\blender\cuerpo_humano_rigged.blend" `
#   --background --python "C:\Users\marti\Desktop\tfg_teleco\proyectos\EsAppSingLenguageAI\app-back\blender\mediapipe\blender_poses_traduction.py" -- `
#   --input "C:\Users\marti\Desktop\tfg_teleco\proyectos\EsAppSingLenguageAI\app-back\blender\mediapipe\poses_mediapipe.json" `
#   --output "C:\Users\marti\Desktop\tfg_teleco\proyectos\EsAppSingLenguageAI\app-back\blender\poses_library.json" `
#   --armature "Human.rig"
#
# ============================================
import bpy
import json
from pathlib import Path
from mathutils import Vector, Quaternion
import argparse

HERE = Path(__file__).resolve().parent
MP_POSES = HERE / "poses_mediapipe.json"
OUT_LIB = HERE / "poses_library.json"
BM_SUGGESTED = HERE / "bone_mapping_suggested.json"

# Default mapping: bone_name -> (landmark_start_idx, landmark_end_idx)
DEFAULT_BONE_LM_MAP = {
    "thumb.01.L": (0, 1), "thumb.02.L": (1, 2), "thumb.03.L": (2, 3),
    "f_index.01.L": (0, 5), "f_index.02.L": (5, 6), "f_index.03.L": (6, 7),
    "f_middle.01.L": (0, 9), "f_middle.02.L": (9,10), "f_middle.03.L": (10,11),
    "f_ring.01.L": (0,13), "f_ring.02.L": (13,14), "f_ring.03.L": (14,15),
    "f_pinky.01.L": (0,17), "f_pinky.02.L": (17,18), "f_pinky.03.L": (18,19),
    "wrist": None
}

def mp_to_blender(p):
    # MediaPipe world_landmarks -> Blender approx: (x, -z, y)
    return Vector((p["x"], -p["z"], p["y"]))

def safe_normalize(v: Vector):
    if v.length <= 1e-9:
        return None
    return v.normalized()

def quat_between(v0: Vector, v1: Vector):
    v0n = safe_normalize(v0)
    v1n = safe_normalize(v1)
    if v0n is None or v1n is None:
        return None
    # use rotation_difference
    return v0n.rotation_difference(v1n)

def load_bone_mapping():
    if BM_SUGGESTED.exists():
        try:
            j = json.loads(BM_SUGGESTED.read_text(encoding="utf-8"))
            # mapping may be in j["mapping"] or j["mapping_suggested"]
            m = j.get("mapping") or j.get("mapping_suggested") or j.get("mapping_suggested", {})
            # normalize values to tuples
            out = {}
            for k, v in m.items():
                if v is None:
                    out[k] = None
                elif isinstance(v, (list, tuple)) and len(v) >= 2:
                    out[k] = (int(v[0]), int(v[1]))
            if out:
                return out
        except Exception:
            pass
    return DEFAULT_BONE_LM_MAP

def find_rest_vector_for_bone(arm_obj, bone_name):
    """
    Try to get the rest (edit/data) bone tail-head vector in armature local space.
    Returns Vector or None.
    """
    try:
        b = arm_obj.data.bones.get(bone_name)
        if not b:
            return None
        # head_local -> tail_local in armature local coords
        return (Vector(b.tail_local) - Vector(b.head_local))
    except Exception:
        return None

def convert_pose_entry(pose_entry, bone_map, arm_obj=None, default_axis=Vector((0.0,1.0,0.0))):
    # pose_entry may be dict or list -> use first if list
    if isinstance(pose_entry, list):
        pose_entry = pose_entry[0]
    wl = pose_entry.get("world_landmarks") or pose_entry.get("landmarks")
    if not wl:
        return None
    pts = [mp_to_blender(p) for p in wl]
    bones_out = {}
    for bone_name, pair in bone_map.items():
        if pair is None:
            bones_out[bone_name] = {"rot_quat": [1.0, 0.0, 0.0, 0.0], "rot_euler": [0.0,0.0,0.0]}
            continue
        a, b = pair
        if a >= len(pts) or b >= len(pts):
            continue
        target_vec = pts[b] - pts[a]
        tnorm = safe_normalize(target_vec)
        if tnorm is None:
            continue
        # rest vector: try armature bone, else default_axis
        rest_vec = default_axis
        if arm_obj:
            rv = find_rest_vector_for_bone(arm_obj, bone_name)
            if rv and rv.length > 1e-9:
                rest_vec = rv
        q = quat_between(rest_vec, target_vec)
        if q is None:
            continue
        # ensure quaternion normalized
        q.normalize()
        e = q.to_euler('XYZ')
        bones_out[bone_name] = {
            "rot_quat": [float(q.w), float(q.x), float(q.y), float(q.z)],
            "rot_euler": [float(e.x), float(e.y), float(e.z)]
        }
    return {"bones": bones_out}

def main(input_path: Path = None, output_path: Path = None, armature_name: str = None):
    input_path = Path(input_path or MP_POSES)
    output_path = Path(output_path or OUT_LIB)
    bone_map = load_bone_mapping()

    # attempt to find armature in current .blend if requested
    arm_obj = None
    if armature_name:
        arm_obj = bpy.data.objects.get(armature_name)
    else:
        # try to auto-find first armature object
        for o in bpy.data.objects:
            if o.type == 'ARMATURE':
                arm_obj = o
                break

    data = json.loads(input_path.read_text(encoding="utf-8"))
    poses = data.get("poses", {})
    out = {"meta": {"armature": arm_obj.name if arm_obj else (armature_name or "Unknown"), "rotation": "QUATERNION"}, "poses": {}}

    for key, value in poses.items():
        processed = convert_pose_entry(value, bone_map, arm_obj=arm_obj)
        if processed:
            out["poses"][key] = processed

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Converted {len(out['poses'])} poses -> {output_path}")

def apply_poses_library(armature_name: str = None, poses_path: Path = None, start_frame: int = 1, frame_step: int = 10):
    """
    Apply the poses from poses_library.json to the armature and insert keyframes.
    Each pose will be placed in the timeline starting at start_frame, stepping by frame_step.
    """
    poses_path = Path(poses_path or OUT_LIB)
    if not poses_path.exists():
        print(f"File not found: {poses_path}")
        return

    # find armature object
    arm_obj = None
    if armature_name:
        arm_obj = bpy.data.objects.get(armature_name)
    if not arm_obj:
        arm_obj = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)
    if not arm_obj:
        print("Armature not found.")
        return

    data = json.loads(poses_path.read_text(encoding="utf-8"))
    poses = data.get("poses", {})
    if not poses:
        print("No poses found in library.")
        return

    # set active and enter pose mode
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)
    bpy.ops.object.mode_set(mode='POSE')

    frame = start_frame
    applied = 0
    for pose_name, pose_data in sorted(poses.items()):
        bones = pose_data.get("bones", {})
        if not bones:
            frame += frame_step
            continue
        bpy.context.scene.frame_set(frame)
        for bone_name, bone_data in bones.items():
            pb = arm_obj.pose.bones.get(bone_name)
            if pb is None:
                print(f"[MISSING BONE] '{bone_name}' not in armature '{arm_obj.name}'")
                continue
            # apply quaternion if present (expecting [w,x,y,z])
            rq = bone_data.get("rot_quat")
            if rq and len(rq) == 4:
                try:
                    pb.rotation_mode = 'QUATERNION'
                    q = Quaternion((float(rq[0]), float(rq[1]), float(rq[2]), float(rq[3])))
                    q.normalize()
                    pb.rotation_quaternion = q
                    pb.keyframe_insert(data_path="rotation_quaternion", frame=frame)
                except Exception as e:
                    print(f"[ERROR] applying quat to {bone_name}: {e}")
                    continue
            else:
                reul = bone_data.get("rot_euler")
                if reul and len(reul) == 3:
                    try:
                        pb.rotation_mode = 'XYZ'
                        pb.rotation_euler = (float(reul[0]), float(reul[1]), float(reul[2]))
                        pb.keyframe_insert(data_path="rotation_euler", frame=frame)
                    except Exception as e:
                        print(f"[ERROR] applying euler to {bone_name}: {e}")
                        continue
        bpy.context.view_layer.update()
        print(f"[APPLIED] pose '{pose_name}' -> frame {frame}")
        applied += 1
        frame += frame_step

    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"Done. Applied {applied} poses to armature '{arm_obj.name}'.")

if __name__ == "__main__":
    import sys
    ap = argparse.ArgumentParser(description="Convert poses_mediapipe.json -> poses_library.json (run inside Blender)")
    ap.add_argument("--input", default=str(MP_POSES))
    ap.add_argument("--output", default=str(OUT_LIB))
    ap.add_argument("--armature", default=None, help="Armature object name in the .blend (optional)")

    # tomar solo los args que vienen después de "--" (Blender convention)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        # fallback cuando se ejecuta desde Text Editor dentro de Blender
        argv = argv[1:]

    args = ap.parse_args(argv)
    main(Path(args.input), Path(args.output), armature_name=args.armature)
