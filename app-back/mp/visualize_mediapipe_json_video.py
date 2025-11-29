import os
import sys
import json
import argparse
from pathlib import Path
import cv2
import math
import numpy as np
import time

# ========== CONFIGURACIÓN GLOBAL (ajusta aquí) ==========
SPEED_FACTOR = 3.0   # factor de velocidad por defecto ( >1 acelera, <1 ralentiza )
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_NAME = "visualize_mediapipe_json_video.mp4"
# =======================================================

# Opcional: usar conexiones oficiales si mediapipe está instalado
try:
    import mediapipe as mp
    HAND_CONNECTIONS = getattr(mp.solutions.hands, "HAND_CONNECTIONS", None)
    POSE_CONNECTIONS = getattr(mp.solutions.pose, "POSE_CONNECTIONS", None)
except Exception:
    mp = None
    HAND_CONNECTIONS = None
    POSE_CONNECTIONS = None

def log(s):
    print(f"[VIS] {s}")

# reemplazamos progress por una versión con timestamp
def progress(s):
    print(f"[{time.strftime('%H:%M:%S')}][VIS][PROG] {s}")

def ensure_dir(p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _get_size(meta, args):
    w = args.width or None
    h = args.height or None
    if not w and meta:
        den = meta.get("denormalization", {}) or {}
        w = int(den.get("image_width") or den.get("width") or 1920)
        h = int(den.get("image_height") or den.get("height") or 1080)
    if not w: w = 1920
    if not h: h = 1080
    return int(w), int(h)

def _frame_bg(w,h):
    return (200,200,200)  # gray background color

def _draw_landmarks_on_img(img, landmarks, color=(0,180,255), connections=None, radius=4):
    h,w = img.shape[:2]
    pts = []
    for lm in landmarks:
        if isinstance(lm, dict):
            x = lm.get("x", lm.get("X", None))
            y = lm.get("y", lm.get("Y", None))
            z = lm.get("z", lm.get("Z", None))
        elif isinstance(lm, (list, tuple)):
            if len(lm) >= 2:
                x,y = lm[0], lm[1]
                z = lm[2] if len(lm) > 2 else 0
            else:
                continue
        else:
            continue
        if x is None or y is None:
            continue
        px = int(round(float(x) * (w-1)))
        py = int(round(float(y) * (h-1)))
        pts.append((px,py,float(z) if z is not None else 0.0))
    # draw connections
    if connections:
        for a,b in connections:
            if a < len(pts) and b < len(pts):
                cv2.line(img, (pts[a][0], pts[a][1]), (pts[b][0], pts[b][1]), (60,60,60), 2, cv2.LINE_AA)
    # draw points (color can vary with depth)
    for (px,py,z) in pts:
        zc = max(0, min(255, int((1.0 - (z if isinstance(z,float) else 0)) * 200)))
        col = (int(color[0]*0.8), int(color[1]*0.8), zc)
        cv2.circle(img, (px,py), radius, col, -1, cv2.LINE_AA)

def extract_frame_landmarks(frame_obj):
    """
    Intentos flexibles de extraer landmarks de un frame json entry.
    Devuelve lista(s) de landmarks o None.
    Soporta:
      - frame['landmarks'] -> list
      - frame['multi_hand_landmarks'] -> list(list)
      - frame['hands'] -> list(list)
      - frame['pose_landmarks'] -> list
      - frame['bones'] -> dictionary with positions (returns list of positions)
    """
    # hands (multi)
    if frame_obj is None:
        return None
    if isinstance(frame_obj, dict):
        # multi-hand common keys
        for k in ("multi_hand_landmarks", "hands", "hand_landmarks", "multi_handedness"):
            if k in frame_obj and frame_obj[k]:
                return frame_obj[k]
        # single list under 'landmarks' or 'landmark'
        for k in ("landmarks", "landmark", "pose_landmarks", "pose_world_landmarks", "world_landmarks"):
            if k in frame_obj and frame_obj[k]:
                return frame_obj[k]
        # if bones mapping present -> try to convert to list of positions
        if "bones" in frame_obj and isinstance(frame_obj["bones"], dict):
            pts = []
            for k,v in frame_obj["bones"].items():
                if isinstance(v, dict) and "loc" in v:
                    loc = v.get("loc")
                    if loc and len(loc) >= 2:
                        pts.append({"x": loc[0], "y": loc[1], "z": loc[2] if len(loc) > 2 else 0.0})
            if pts:
                return pts
    return None

def _load_ref_image(path, target_w, target_h):
    """
    Mantiene compatibilidad con imágenes; si 'path' es None o no existe devuelve placeholder.
    Devuelve BGR image scaled to (target_w, target_h).
    """
    canvas = 127 * np.ones((target_h, target_w, 3), dtype="uint8")
    if not path:
        return canvas
    p = Path(path)
    if not p.exists():
        return canvas
    im = cv2.imread(str(p))  # BGR
    if im is None:
        return canvas
    ih, iw = im.shape[:2]
    scale = min(target_w / iw, target_h / ih)
    nw = max(1, int(iw * scale))
    nh = max(1, int(ih * scale))
    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
    x = (target_w - nw) // 2
    y = (target_h - nh) // 2
    canvas[y:y+nh, x:x+nw] = im_resized
    return canvas

def _extract_frame_from_video(video_path, target_w, target_h):
    """
    Extrae un frame representativo (frame medio) del vídeo y lo escala a target_w x target_h.
    Devuelve BGR image o placeholder si falla.
    """
    canvas = 127 * np.ones((target_h, target_w, 3), dtype="uint8")
    if not video_path:
        return canvas
    v = Path(video_path)
    if not v.exists():
        return canvas
    try:
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            return canvas
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            # intentar leer primer frame
            ret, frame = cap.read()
        else:
            mid = max(0, total // 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return canvas
        ih, iw = frame.shape[:2]
        scale = min(target_w / iw, target_h / ih)
        nw = max(1, int(iw * scale))
        nh = max(1, int(ih * scale))
        frame_resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        x = (target_w - nw) // 2
        y = (target_h - nh) // 2
        canvas[y:y+nh, x:x+nw] = frame_resized
        return canvas
    except Exception as e:
        progress(f"    warning: failed to extract from video {video_path}: {e}")
        return canvas

def visualize_json(json_path, out_path=None, show=True, save=True, fps=None, width=None, height=None, speed_factor=None):
    progress("Starting visualization script")
    progress(f"Loading JSON: {json_path}")
    data = load_json(json_path)
    meta = data.get("meta", {}) if isinstance(data, dict) else {}
    poses = data.get("poses", {}) if isinstance(data, dict) else {}

    w,h = _get_size(meta, argparse.Namespace(width=width, height=height))
    fps_use = fps or meta.get("denormalization",{}).get("fps") or 30
    try:
        fps_use = int(fps_use)
    except Exception:
        fps_use = 30
    # apply speed factor: output fps will be input_fps * speed_factor
    speed_factor = float(speed_factor) if speed_factor is not None else float(SPEED_FACTOR)
    fps_out = max(1, int(round(fps_use * float(speed_factor))))
    progress(f"Resolved size {w}x{h}, source fps {fps_use}, speed_factor {speed_factor} -> writer fps {fps_out}")

    # prepare connections heuristics
    hand_cons = None
    pose_cons = None
    if HAND_CONNECTIONS:
        hand_cons = [(int(a),int(b)) for a,b in HAND_CONNECTIONS]
    if POSE_CONNECTIONS:
        pose_cons = [(int(a),int(b)) for a,b in POSE_CONNECTIONS]

    # build combined sequence frames and write video streaming to disk
    if out_path is None:
        out_path = str(OUTPUT_DIR / OUTPUT_NAME)
    ensure_dir(out_path)
    progress(f"Preparing VideoWriter -> {out_path}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_w = w * 2
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_out, (out_w, h))
    if not writer.isOpened():
        progress("ERROR: VideoWriter could not be opened")
        raise RuntimeError("No se pudo abrir VideoWriter (comprueba codecs)")
    progress("VideoWriter opened successfully")

    total_frames = 0
    pose_count = len(poses)
    progress(f"Found {pose_count} poses")

    # default folders (FIXED: use app-back as base)
    base_app_back = Path(__file__).resolve().parent.parent.parent  # app-back
    default_ref_dir = base_app_back / "data" / "dataset-en-bruto" / "asl_dataset" / "mediapipe_poses"
    default_videos_dir = base_app_back / "data" / "dataset-en-bruto" / "asl_dataset" / "videos" / "videos_de_las_letras"
    video_exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")

    for idx, (pose_name, pose_entry) in enumerate(poses.items(), start=1):
        progress(f"[{idx}/{pose_count}] Start pose '{pose_name}'")
        # 1) Preferir vídeo de la carpeta videos_de_las_letras/<pose_name>/*
        video_candidate = None

        # try multiple casings for folder name
        possible_folder_names = [pose_name, pose_name.lower(), pose_name.upper()]
        for fname in possible_folder_names:
            folder_candidate = default_videos_dir / fname
            if folder_candidate.exists() and folder_candidate.is_dir():
                progress(f"  found video folder: {folder_candidate}")
                # pick first video file found
                for file in sorted(folder_candidate.iterdir()):
                    if file.suffix.lower() in video_exts:
                        video_candidate = str(file)
                        break
            if video_candidate:
                break

        if video_candidate:
            progress(f"  found video reference: {video_candidate}")
            ref_img = _extract_frame_from_video(video_candidate, w, h)
        else:
            # 2) si JSON tiene source_image usarla
            src_img_path = None
            if isinstance(pose_entry, dict):
                src_img_path = pose_entry.get("source_image")
            # 3) fallback: buscar imagen en default_ref_dir/<pose_name>.* con distintos casos
            if not src_img_path:
                for name in possible_folder_names:
                    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
                        candidate = default_ref_dir / (name + ext)
                        if candidate.exists():
                            src_img_path = str(candidate)
                            break
                    if src_img_path:
                        break

            if src_img_path:
                progress(f"  using image reference: {src_img_path}")
            else:
                progress("  no reference (video/image) found, using placeholder")
            ref_img = _load_ref_image(src_img_path, w, h)

        # header/title frames (left: title, right: reference)
        title_left = (255 * np.ones((h, w, 3), dtype="uint8")) // 2
        cv2.putText(title_left, f"{pose_name}", (40, h//2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (30,30,30), 6, cv2.LINE_AA)
        header_frames = max(3, int(round(0.15*fps_use)))
        for _ in range(header_frames):
            canvas = np.concatenate([title_left, ref_img], axis=1)
            writer.write(canvas); total_frames += 1
        progress(f"  wrote {header_frames} header frames for '{pose_name}'")

        frames_list = pose_entry.get("frames") if isinstance(pose_entry, dict) else None
        if frames_list and isinstance(frames_list, list) and len(frames_list) > 0:
            progress(f"  -> rendering {len(frames_list)} frames (will appear faster due to speed factor)")
            for fi, frame_obj in enumerate(frames_list):
                left = (255 * np.ones((h, w, 3), dtype="uint8")) // 2

                # persistent title bar on left
                rect_top = 18
                rect_h = 72
                overlay = left.copy()
                cv2.rectangle(overlay, (20, rect_top), (w-20, rect_top + rect_h), (0,0,0), -1)
                alpha = 0.45
                cv2.addWeighted(overlay, alpha, left, 1 - alpha, 0, left)
                cv2.putText(left, pose_name, (40, rect_top + int(rect_h * 0.7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255,255,255), 4, cv2.LINE_AA)

                ldata = extract_frame_landmarks(frame_obj)
                if ldata:
                    if isinstance(ldata, list) and len(ldata) > 0 and isinstance(ldata[0], list):
                        for hand in ldata:
                            _draw_landmarks_on_img(left, hand, color=(0,180,255), connections=hand_cons, radius=4)
                    else:
                        _draw_landmarks_on_img(left, ldata, color=(0,180,255), connections=hand_cons or pose_cons, radius=4)
                else:
                    cv2.putText(left, "no landmarks", (40, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30,30,30), 2, cv2.LINE_AA)

                # combine left annotated (BGR) and right reference (BGR)
                canvas = np.concatenate([left, ref_img], axis=1)
                writer.write(canvas)
                total_frames += 1

                if (fi + 1) % 50 == 0:
                    progress(f"    wrote {fi+1}/{len(frames_list)} frames for pose '{pose_name}'")
            progress(f"  finished frames for '{pose_name}'")
        else:
            progress(f"  -> no frames for pose '{pose_name}', writing placeholder")
            for _ in range( max(1, int(round(0.6 * fps_use))) ):
                left = (255 * np.ones((h, w, 3), dtype="uint8")) // 2
                cv2.putText(left, f"{pose_name} (no frames)", (40, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (10,10,10), 3, cv2.LINE_AA)
                canvas = np.concatenate([left, ref_img], axis=1)
                writer.write(canvas); total_frames += 1
            progress(f"  wrote placeholder frames for '{pose_name}'")

        progress(f"[{idx}/{pose_count}] finished '{pose_name}' (total frames so far: {total_frames})")

    writer.release()
    progress(f"Writer released. Total frames written: {total_frames}")
    log(f"Annotated video saved: {out_path}  (frames written: {total_frames})")
    if show:
        log("Play the output file in your preferred player.")
    return out_path

def main():
    progress("Script entry -> building args and launching visualization")
    default_json = r"C:\Users\marti\Desktop\tfg_teleco\proyectos\EsAppSingLenguageAI\app-back\blender\mediapipe\poses_mediapipe_video.json"
    default_out = str(OUTPUT_DIR / OUTPUT_NAME)

    ap = argparse.ArgumentParser(description="Visualiza y genera vídeo anotado del JSON de MediaPipe (con referencia lado-a-lado)")
    ap.add_argument("--json", "-j", required=False, default=default_json, help="ruta al JSON (poses_mediapipe_video.json)")
    ap.add_argument("--out", "-o", default=default_out, help="ruta de salida mp4")
    ap.add_argument("--no-show", action="store_true", help="no mostrar info (solo guardar)")
    ap.add_argument("--fps", type=int, default=None, help="fps entrada/base (se multiplica por speed)")
    ap.add_argument("--width", type=int, default=None, help="ancho salida (por pose, el vídeo final será ancho*2)")
    ap.add_argument("--height", type=int, default=None, help="alto salida")
    ap.add_argument("--speed", "-s", type=float, default=SPEED_FACTOR, help="factor de velocidad de reproducción (>1 acelera)")
    args = ap.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress(f"Calling visualize_json with json={args.json} out={args.out} speed={args.speed}")
    out = visualize_json(args.json, out_path=args.out, show=not args.no_show, fps=args.fps, width=args.width, height=args.height, speed_factor=args.speed)
    progress(f"Visualization finished. Output: {out}")

if __name__ == "__main__":
    try:
        import numpy as np  # ensure numpy available
    except Exception:
        log("ERROR: requiere numpy y opencv-python. Instálalos con: pip install numpy opencv-python")
        sys.exit(1)
    main()