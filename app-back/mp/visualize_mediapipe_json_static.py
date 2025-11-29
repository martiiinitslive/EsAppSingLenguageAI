import os
import sys
import json
import argparse
from pathlib import Path
import cv2
import numpy as np

# ========== CONFIGURACIÓN GLOBAL (ajusta aquí) ==========
SPEED_FACTOR = 1   # factor de velocidad por defecto (>1 acelera, <1 ralentiza)
#las letreas con movimeinto, no estan, ya que esto es estatico
# ========================================================
try:
    import mediapipe as mp
    HAND_CONNECTIONS = getattr(mp.solutions.hands, "HAND_CONNECTIONS", None)
    POSE_CONNECTIONS = getattr(mp.solutions.pose, "POSE_CONNECTIONS", None)
except Exception:
    mp = None
    HAND_CONNECTIONS = None
    POSE_CONNECTIONS = None

def log(s):
    print(f"[VIS_STATIC] {s}")

# seguimiento/estado más detallado
def progress(s):
    print(f"[VIS_STATIC][PROG] {s}")

def ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _get_size(meta, width, height):
    if width and height:
        return int(width), int(height)
    den = (meta or {}).get("denormalization", {}) or {}
    w = int(den.get("image_width") or den.get("width") or 1920)
    h = int(den.get("image_height") or den.get("height") or 1080)
    return w, h

def _draw_landmarks_on_img(img, landmarks, connections=None, color=(0,180,255), radius=5):
    h,w = img.shape[:2]
    pts = []
    for lm in landmarks:
        if isinstance(lm, dict):
            x = lm.get("x", lm.get("X", None))
            y = lm.get("y", lm.get("Y", None))
            z = lm.get("z", lm.get("Z", 0.0))
        elif isinstance(lm, (list, tuple)):
            if len(lm) >= 2:
                x,y = lm[0], lm[1]
                z = lm[2] if len(lm) > 2 else 0.0
            else:
                continue
        else:
            continue
        if x is None or y is None:
            continue
        px = int(round(float(x) * (w-1)))
        py = int(round(float(y) * (h-1)))
        pts.append((px,py,float(z)))
    if connections:
        for a,b in connections:
            if a < len(pts) and b < len(pts):
                cv2.line(img, (pts[a][0], pts[a][1]), (pts[b][0], pts[b][1]), (40,40,40), 2, cv2.LINE_AA)
    for (px,py,z) in pts:
        depth_col = max(0, min(255, int((1.0 - (z if isinstance(z,float) else 0)) * 200)))
        col = (int(color[0]*0.8), int(color[1]*0.8), depth_col)
        cv2.circle(img, (px,py), radius, col, -1, cv2.LINE_AA)

def extract_landmarks_from_pose_entry(entry):
    # prefer 'landmarks' then 'pose_landmarks' then 'world_landmarks'
    for k in ("landmarks", "pose_landmarks", "world_landmarks"):
        if k in entry and entry[k]:
            return entry[k]
    # fallback: if bones dict present, extract locs
    if "bones" in entry and isinstance(entry["bones"], dict):
        pts = []
        for v in entry["bones"].values():
            if isinstance(v, dict) and "loc" in v:
                loc = v["loc"]
                if loc and len(loc) >= 2:
                    pts.append({"x": loc[0], "y": loc[1], "z": (loc[2] if len(loc)>2 else 0.0)})
        if pts:
            return pts
    return None

def _load_ref_image(path, target_w, target_h):
    """
    Carga la imagen de referencia (source_image) y la redimensiona a target_w x target_h.
    Si no existe devuelve una imagen de placeholder.
    """
    if not path:
        return 127 * np.ones((target_h, target_w, 3), dtype=np.uint8)
    p = Path(path)
    if not p.exists():
        return 127 * np.ones((target_h, target_w, 3), dtype=np.uint8)
    im = cv2.imread(str(p))
    if im is None:
        return 127 * np.ones((target_h, target_w, 3), dtype=np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_h, im_w = im.shape[:2]
    # mantener aspecto y luego centrar en target canvas
    scale = min(target_w / im_w, target_h / im_h)
    nw = max(1, int(im_w * scale))
    nh = max(1, int(im_h * scale))
    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = 127 * np.ones((target_h, target_w, 3), dtype=np.uint8)
    x = (target_w - nw) // 2
    y = (target_h - nh) // 2
    canvas[y:y+nh, x:x+nw] = im_resized
    return canvas

def visualize_static(json_path, out_path, fps=30, hold_seconds=0.6, width=None, height=None, show=False, speed_factor=None):
    progress(f"Loading JSON: {json_path}")
    data = load_json(json_path)
    meta = data.get("meta", {}) if isinstance(data, dict) else {}
    poses = data.get("poses", {}) if isinstance(data, dict) else {}

    w,h = _get_size(meta, width, height)
    fps = int(fps) if fps else 30

    # aplicar factor de velocidad al fps de salida
    sf = float(speed_factor) if speed_factor is not None else float(SPEED_FACTOR)
    fps_out = max(1, int(round(fps * sf)))

    frames_per_pose = max(1, int(round(hold_seconds * fps_out)))

    out_path = out_path or (Path(json_path).with_suffix("").name + "_static.mp4")
    ensure_dir(out_path)
    progress(f"Preparing writer -> {out_path}  ({w}x{h} @ {fps_out} fps)  speed_factor={sf}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # ahora la salida mostrará lado-a-lado: ancho = w * 2
    out_w = w * 2
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_out, (out_w, h))
    if not writer.isOpened():
        raise RuntimeError("No se pudo abrir VideoWriter (cambia fourcc o instala codecs)")

    hand_cons = None
    pose_cons = None
    if HAND_CONNECTIONS:
        hand_cons = [(int(a),int(b)) for a,b in HAND_CONNECTIONS]
    if POSE_CONNECTIONS:
        pose_cons = [(int(a),int(b)) for a,b in POSE_CONNECTIONS]

    total = 0
    pose_count = len(poses)
    progress(f"Found {pose_count} poses. frames_per_pose={frames_per_pose}")

    try:
        for idx, (pose_name, entry) in enumerate(poses.items(), start=1):
            progress(f"[{idx}/{pose_count}] Processing pose '{pose_name}'")
            # cargar imagen de referencia si existe
            src_img_path = entry.get("source_image") if isinstance(entry, dict) else None
            ref_img = _load_ref_image(src_img_path, w, h)

            # título / header frames (mostrar título y la referencia a la derecha)
            title_left = (255 * np.ones((h, w, 3), dtype="uint8")) // 2
            cv2.putText(title_left, f"{pose_name}", (40, h//2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (30,30,30), 6, cv2.LINE_AA)
            # componer canvas lado-a-lado (left: title, right: referencia)
            header_frames = max(3, int(round(0.15*fps)))
            for _ in range(header_frames):
                canvas = 127 * np.ones((h, out_w, 3), dtype=np.uint8)
                canvas[:, :w] = title_left
                canvas[:, w:] = ref_img
                writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)); total += 1

            landmarks = extract_landmarks_from_pose_entry(entry)
            if landmarks:
                progress(f"  -> landmarks found (type={type(landmarks).__name__}). Rendering {frames_per_pose} frames")
                for fidx in range(frames_per_pose):
                    # generar imagen anotada (izquierda)
                    left_img = (255 * np.ones((h, w, 3), dtype="uint8")) // 2

                    # --- persistent pose title on left ---
                    rect_top = 18
                    rect_h = 72
                    overlay = left_img.copy()
                    cv2.rectangle(overlay, (20, rect_top), (w-20, rect_top + rect_h), (0,0,0), -1)
                    alpha = 0.45
                    cv2.addWeighted(overlay, alpha, left_img, 1 - alpha, 0, left_img)
                    cv2.putText(left_img, pose_name, (40, rect_top + int(rect_h * 0.7)),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255,255,255), 4, cv2.LINE_AA)

                    # dibujar landmarks sobre left_img
                    if isinstance(landmarks, list) and len(landmarks)>0 and isinstance(landmarks[0], list):
                        for hand in landmarks:
                            _draw_landmarks_on_img(left_img, hand, connections=hand_cons, radius=6)
                    else:
                        _draw_landmarks_on_img(left_img, landmarks, connections=hand_cons or pose_cons, radius=6)

                    # componer canvas: izquierda = annotated, derecha = ref_img
                    canvas = 127 * np.ones((h, out_w, 3), dtype=np.uint8)
                    canvas[:, :w] = left_img
                    canvas[:, w:] = ref_img
                    writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)); total += 1

                    if (fidx + 1) % 20 == 0:
                        progress(f"    wrote {fidx+1}/{frames_per_pose} frames for pose '{pose_name}'")
            else:
                progress(f"  -> NO landmarks for pose '{pose_name}'. Writing placeholder for {frames_per_pose} frames")
                for _ in range(frames_per_pose):
                    left_img = (255 * np.ones((h, w, 3), dtype="uint8")) // 2
                    rect_top = 18
                    rect_h = 72
                    overlay = left_img.copy()
                    cv2.rectangle(overlay, (20, rect_top), (w-20, rect_top + rect_h), (0,0,0), -1)
                    alpha = 0.45
                    cv2.addWeighted(overlay, alpha, left_img, 1 - alpha, 0, left_img)
                    cv2.putText(left_img, pose_name, (40, rect_top + int(rect_h * 0.7)),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255,255,255), 4, cv2.LINE_AA)

                    canvas = 127 * np.ones((h, out_w, 3), dtype=np.uint8)
                    canvas[:, :w] = left_img
                    canvas[:, w:] = ref_img
                    writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)); total += 1

            progress(f"[{idx}/{pose_count}] Finished pose '{pose_name}' (total frames so far: {total})")
    finally:
        writer.release()
        progress(f"Writer released. Total frames written: {total}")

    log(f"Saved static visualization (side-by-side): {out_path}  (frames {total})")
    if show:
        log("Open the file to view.")
    return out_path

def main():
    default_json = Path(__file__).resolve().parent.parent / "mediapipe" / "poses_mediapipe.json"
    default_out = Path(__file__).resolve().parent / "output" / "visualize_mediapipe_json_static.mp4"

    ap = argparse.ArgumentParser(description="Visualiza JSON de MediaPipe con imágenes estáticas (produce MP4)")
    ap.add_argument("--json", "-j", default=str(default_json), help="ruta al JSON (por defecto el dataset static)")
    ap.add_argument("--out", "-o", default=str(default_out), help="ruta de salida mp4")
    ap.add_argument("--fps", type=int, default=30, help="fps base de entrada (se multiplicará por speed)")
    ap.add_argument("--hold", type=float, default=0.6, help="segundos que se muestra cada pose (en tiempo real ajustado por speed)")
    ap.add_argument("--width", type=int, default=None, help="ancho salida (opcional)")
    ap.add_argument("--height", type=int, default=None, help="alto salida (opcional)")
    ap.add_argument("--speed", type=float, default=SPEED_FACTOR, help="factor de velocidad de reproducción (>1 acelera, <1 ralentiza)")
    ap.add_argument("--no-show", action="store_true", help="no mostrar mensaje de reproducción")
    args = ap.parse_args()

    out = visualize_static(args.json, args.out, fps=args.fps, hold_seconds=args.hold, width=args.width, height=args.height, show=(not args.no_show), speed_factor=args.speed)
    print(out)

if __name__ == "__main__":
    try:
        import numpy as np
    except Exception:
        log("ERROR: requiere numpy y opencv-python. Instálalos con: pip install numpy opencv-python")
        sys.exit(1)
    main()