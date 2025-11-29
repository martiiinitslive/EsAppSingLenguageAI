"""
Compute angles between connected landmarks for each frame in poses_mediapipe_video.json.

Outputs: mediapipe/info_angles_video.json (same folder as this script)

Usage:
    python obtain_angles_video.py
or with args:
    python obtain_angles_video.py --input poses_mediapipe_video.json --output info_angles_video.json

Notes:
- This script expects frames to contain landmarks as lists of dicts or lists (x,y[,z]).
- For each joint (landmark index) we compute angles between every unordered pair of its neighbors
  as defined in HAND_CONNECTIONS. The angle is computed at the joint between segments neighbor->joint->neighbor.
- If a frame lacks landmarks or vectors are degenerate (zero length), angle is recorded as null.
"""

from pathlib import Path
import json
import math
import argparse
from collections import defaultdict

# Local copy of typical MediaPipe hand connections used by the other scripts.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17), (0, 13), (0, 9), (0, 5)     # Wrist to bases
]


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def to_vec(pt):
    """Normalize different landmark representations to a 3-tuple (x,y,z).
    Accepts dicts with keys 'x','y','z' or lists/tuples with 2 or 3 elements.
    If missing or invalid, returns None.
    """
    if pt is None:
        return None
    try:
        if isinstance(pt, dict):
            x = pt.get('x', pt.get('X'))
            y = pt.get('y', pt.get('Y'))
            z = pt.get('z', pt.get('Z', 0.0))
            if x is None or y is None:
                return None
            return (float(x), float(y), float(z if z is not None else 0.0))
        elif isinstance(pt, (list, tuple)):
            if len(pt) >= 2:
                x = float(pt[0]); y = float(pt[1]); z = float(pt[2]) if len(pt) > 2 else 0.0
                return (x, y, z)
            else:
                return None
        else:
            return None
    except Exception:
        return None


def sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def norm(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])


def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def angle_between(a, b, c):
    """Angle at point b between a-b and c-b in degrees. Returns None on degenerate vectors."""
    va = sub(a, b)
    vc = sub(c, b)
    na = norm(va)
    nc = norm(vc)
    if na < 1e-8 or nc < 1e-8:
        return None
    # use stable acos with clamping
    d = dot(va, vc) / (na * nc)
    d = max(-1.0, min(1.0, d))
    ang = math.degrees(math.acos(d))
    return ang


def build_neighbor_map(connections):
    nbrs = defaultdict(set)
    for a, b in connections:
        nbrs[a].add(b)
        nbrs[b].add(a)
    return {k: sorted(list(v)) for k, v in nbrs.items()}


def compute_angles_for_frame(landmarks, neighbor_map):
    """Given landmarks (list) and neighbor_map, compute angles per joint.
    Returns dict: joint_index -> { "pairs": [ {"neighbors":[i,j], "angle": float|null}, ... ] }
    """
    if not landmarks:
        return None
    pts = [to_vec(p) for p in landmarks]
    # if many None, treat as missing
    if all(p is None for p in pts):
        return None
    angles = {}
    L = len(pts)
    for j, neighbors in neighbor_map.items():
        if j >= L:
            continue
        if pts[j] is None:
            continue
        pairs = []
        # compute all unordered neighbor pairs (i,k) with i<k
        for i_idx in range(len(neighbors)):
            for k_idx in range(i_idx+1, len(neighbors)):
                i = neighbors[i_idx]
                k = neighbors[k_idx]
                # ensure indices exist
                if i >= L or k >= L:
                    continue
                if pts[i] is None or pts[k] is None:
                    pairs.append({"neighbors": [i, k], "angle": None})
                    continue
                ang = angle_between(pts[i], pts[j], pts[k])
                pairs.append({"neighbors": [i, k], "angle": (None if ang is None else round(ang, 3))})
        if pairs:
            angles[str(j)] = {"pairs": pairs}
    return angles


def main():
    ap = argparse.ArgumentParser(description="Obtain angles between connected landmarks from poses_mediapipe_video.json")
    ap.add_argument('--input', '-i', default=str(Path(__file__).parent / 'poses_mediapipe_video.json'), help='Input poses JSON (relative to mediapipe folder)')
    ap.add_argument('--output', '-o', default=str(Path(__file__).parent / 'info_angles_video.json'), help='Output JSON with angles')
    ap.add_argument('--limit-poses', type=int, default=None, help='Optional: limit number of poses processed (for speed/testing)')
    args = ap.parse_args()

    input_p = Path(args.input)
    output_p = Path(args.output)

    data = load_json(input_p)
    meta = data.get('meta', {})
    poses = data.get('poses', {})

    neighbor_map = build_neighbor_map(HAND_CONNECTIONS)

    out = {
        'meta': {
            'source': str(input_p),
            'generated_at': None,
            'note': 'Angles computed at joints between neighbor pairs (degrees)'
        },
        'poses': {}
    }

    import datetime
    out['meta']['generated_at'] = datetime.datetime.utcnow().isoformat() + 'Z'

    pose_items = list(poses.items())
    if args.limit_poses is not None:
        pose_items = pose_items[:args.limit_poses]

    for pname, pinfo in pose_items:
        frames = pinfo.get('frames', []) or []
        pose_out_frames = []
        for fi, frame in enumerate(frames):
            # Attempt to extract landmarks in common shapes
            landmarks = None
            # frame might directly be list of landmarks
            if isinstance(frame, list):
                landmarks = frame
            elif isinstance(frame, dict):
                # common keys
                for key in ('landmarks', 'multi_hand_landmarks', 'hand_landmarks', 'pose_landmarks'):
                    if key in frame and frame[key]:
                        landmarks = frame[key]
                        break
                # some stored as nested dicts like {'0': {...}, '1': {...}}
                if landmarks is None:
                    # try to detect numeric-keyed dict
                    if all(k.isdigit() for k in frame.keys() if isinstance(k, str)):
                        ordered = [frame[k] for k in sorted(frame.keys(), key=lambda x: int(x))]
                        landmarks = ordered
            # compute angles
            angles = compute_angles_for_frame(landmarks, neighbor_map)
            pose_out_frames.append({'frame_index': fi, 'angles': angles})
        out['poses'][pname] = {
            'gesture_name': pinfo.get('gesture_name'),
            'frames': pose_out_frames,
            'frames_count': len(pose_out_frames)
        }

    # ensure output dir exists
    output_p.parent.mkdir(parents=True, exist_ok=True)
    save_json(out, output_p)
    print(f"Wrote angles JSON: {output_p}")

if __name__ == '__main__':
    main()
