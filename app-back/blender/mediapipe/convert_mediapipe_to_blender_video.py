import json
from pathlib import Path
import math
import numpy as np

IN_PATH = Path(__file__).resolve().parent / ".." / "mediapipe" / "poses_mediapipe_video.json"
OUT_PATH = Path(__file__).resolve().parent.parent / "poses_converted_video.json"

# mapping MP hand landmark index -> bone name (simple, one-to-one)
MP_HAND_BONES = {
    0: "wrist",
    1: "thumb_cmc", 2: "thumb_mcp", 3: "thumb_ip", 4: "thumb_tip",
    5: "index_mcp", 6: "index_pip", 7: "index_dip", 8: "index_tip",
    9: "middle_mcp", 10: "middle_pip", 11: "middle_dip", 12: "middle_tip",
    13: "ring_mcp", 14: "ring_pip", 15: "ring_dip", 16: "ring_tip",
    17: "pinky_mcp", 18: "pinky_pip", 19: "pinky_dip", 20: "pinky_tip"
}

def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def save_json(p, data):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(data, indent=2), encoding="utf-8")

def to_np(lm):
    # supports dict {x,y,z} or list/tuple
    if lm is None:
        return None
    if isinstance(lm, dict):
        if "x" in lm and "y" in lm:
            x = float(lm.get("x", 0.0)); y = float(lm.get("y", 0.0)); z = float(lm.get("z", 0.0))
            return np.array([x, y, z], dtype=float)
    if isinstance(lm, (list, tuple)) and len(lm) >= 2:
        x = float(lm[0]); y = float(lm[1]); z = float(lm[2]) if len(lm) > 2 else 0.0
        return np.array([x, y, z], dtype=float)
    return None

def quat_from_vectors(u, v):
    # return quaternion [w,x,y,z] rotating u->v
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)
    dot = np.dot(u, v)
    if dot > 0.999999:
        return [1.0, 0.0, 0.0, 0.0]
    if dot < -0.999999:
        # 180 deg: pick orthogonal axis
        axis = np.cross(u, np.array([1.0,0.0,0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(u, np.array([0.0,1.0,0.0]))
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return [0.0, float(axis[0]), float(axis[1]), float(axis[2])]
    axis = np.cross(u, v)
    s = math.sqrt((1.0 + dot) * 2.0)
    invs = 1.0 / s
    qx = axis[0] * invs
    qy = axis[1] * invs
    qz = axis[2] * invs
    qw = 0.5 * s
    return [float(qw), float(qx), float(qy), float(qz)]

def convert_pose_entry(pose_entry, fallback_denorm):
    video_info = pose_entry.get("video_info", {}) or {}
    W = int(video_info.get("width") or fallback_denorm.get("image_width") or 1920)
    H = int(video_info.get("height") or fallback_denorm.get("image_height") or 1080)
    hand_mm = float(fallback_denorm.get("hand_width_mm", 200)) # mm
    scale_m = hand_mm / 1000.0  # use as rough scale for normalized coords -> meters

    frames_out = []
    frames = pose_entry.get("frames") or []
    for fr in frames:
        fo = {
            "frame_number": fr.get("frame_number"),
            "timestamp_sec": fr.get("timestamp_sec"),
            "hand_detected": fr.get("hand_detected", False),
            "handedness": fr.get("handedness")
        }
        # pick landmarks: prefer world_landmarks (3D in meters when available), else landmarks (normalized)
        lms = fr.get("world_landmarks") or fr.get("landmarks") or []
        if not lms:
            fo["bones"] = {}
            frames_out.append(fo)
            continue

        # convert to numpy points list
        pts = []
        for lm in lms:
            p = to_np(lm)
            if p is None:
                continue
            pts.append(p)
        if len(pts) == 0:
            fo["bones"] = {}
            frames_out.append(fo)
            continue
        pts = np.array(pts)

        # If points look normalized (values ~0..1) and not world coords, denormalize to meters using scale_m
        if np.nanmax(np.abs(pts)) <= 1.5:
            # center x,y around 0; y flip so that +Y is up (heuristic)
            px = (pts[:,0] - 0.5) * scale_m
            py = (0.5 - pts[:,1]) * scale_m
            pz = pts[:,2] * scale_m  # relative depth estimate
            pts_m = np.column_stack([px, py, pz])
        else:
            # assume already meters (world_landmarks)
            pts_m = pts.copy()

        bones = {}
        # produce bone entries for each known MP index if available
        for idx, name in MP_HAND_BONES.items():
            if idx >= len(pts_m):
                continue
            pos = pts_m[idx].tolist()
            bone = {"position": [float(pos[0]), float(pos[1]), float(pos[2])]}
            # compute a rotation quaternion by using vector to next joint if exists
            next_idx = None
            # choose neighbor joint to define bone direction (tip uses parent, others use next)
            if idx in (4,8,12,16,20):  # tips: use previous
                next_idx = idx - 1
            else:
                next_idx = idx + 1 if (idx + 1) < len(pts_m) else None
            if next_idx is not None and next_idx < len(pts_m):
                v_from = np.array([0.0, 1.0, 0.0])  # model bone default points +Y (heuristic)
                v_to = pts_m[next_idx] - pts_m[idx]
                if np.linalg.norm(v_to) > 1e-9:
                    q = quat_from_vectors(v_from, v_to)
                    bone["rotation_quat"] = q  # [w,x,y,z]
            bones[name] = bone

        fo["bones"] = bones
        frames_out.append(fo)
    return frames_out

def main():
    inp = Path(IN_PATH)
    if not inp.exists():
        # try relative location adjacent to this script (user-provided)
        inp = Path(__file__).resolve().parent / "poses_mediapipe_video.json"
    if not inp.exists():
        raise SystemExit(f"Input JSON not found: {IN_PATH}")

    data = load_json(inp)
    meta_out = {
        "armature": "Human.rig",
        "rotation": "QUATERNION",
        "source": "MediaPipe hand landmarks (VIDEO)",
        "denormalization": {
            "image_width": 1920,
            "image_height": 1080,
            "hand_width_mm": 200,
            "camera_fov_deg": 65
        }
    }
    fallback_denorm = meta_out["denormalization"]

    out = {"meta": meta_out, "poses": {}}
    poses = data.get("poses", {}) or {}
    for name, entry in poses.items():
        converted = {
            "gesture_name": entry.get("gesture_name") or name,
            "video_info": entry.get("video_info", {}),
            "total_frames": entry.get("frames_processed") or entry.get("video_info", {}).get("total_frames"),
            "frames": convert_pose_entry(entry, fallback_denorm)
        }
        out["poses"][name] = converted

    save_json(OUT_PATH, out)
    print(f"Wrote converted poses to: {OUT_PATH}")

if __name__ == "__main__":
    main()