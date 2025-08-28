import cv2
import time
import numpy as np
from ultralytics import YOLO

WINDOW_NAME = "Jarvis Navigation HUD (Local YOLO)"
MODEL_PATH = "yolov8n.pt"          # swap to your custom trained weights later, e.g., "runs/detect/train/weights/best.pt"

# Tunables
CENTER_FRAC = 0.34                 # middle band width (0.34 ≈ 1/3 of frame)
NEAR_Y_FRAC = 0.80                 # if box bottom y is below this fraction of height -> considered near
NEAR_H_FRAC = 0.35                 # if bbox height > 35% of frame height -> near
NEAR_AREA_FRAC = 0.20              # if bbox area > 20% of frame area -> near
CONF_THRESH = 0.35                 # detection confidence threshold
TARGET_CLASSES = {
    # Names as in COCO; your custom model can use same names or your own
    "door", "stairs", "person", "chair", "table", "bench", "sofa", "couch", "tv", "potted plant", "bicycle", "bottle", "backpack", "suitcase"
}
NAV_OBJECTS = {"door", "stairs"}
OBSTACLES = TARGET_CLASSES - NAV_OBJECTS

# Colors (BGR)
COLOR_GO = (0, 255, 0)
COLOR_TURN = (255, 0, 0)
COLOR_STOP = (0, 0, 255)
COLOR_INFO = (255, 255, 255)
COLOR_BOX = (0, 255, 255)
COLOR_BAND = (50, 50, 50)

def set_fullscreen(on: bool):
    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL)

def draw_hud_text(frame, text, color):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 64), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    cv2.putText(frame, text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)

def draw_status(frame, cam_index, flipped, fullscreen, fps, model_name):
    lines = [
        f"CAM:{cam_index}  FLIP:{'ON' if flipped else 'OFF'}  FULL:{'ON' if fullscreen else 'OFF'}  FPS:{fps:.1f}",
        f"MODEL:{model_name}  CONF>={CONF_THRESH}"
    ]
    x, y = 16, frame.shape[0] - 48
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + 22*i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_INFO, 1, cv2.LINE_AA)

def object_position(x1, x2, frame_w):
    center = (x1 + x2) / 2.0
    left_band = (1 - CENTER_FRAC) * 0.5 * frame_w
    right_band = frame_w - left_band
    if center < left_band:
        return "left"
    elif center > right_band:
        return "right"
    else:
        return "center"

def is_near(x1, y1, x2, y2, frame_w, frame_h):
    w = x2 - x1
    h = y2 - y1
    area = w * h
    frame_area = frame_w * frame_h
    bottom_frac = y2 / float(frame_h)
    h_frac = h / float(frame_h)
    area_frac = area / float(frame_area)
    return (bottom_frac >= NEAR_Y_FRAC) or (h_frac >= NEAR_H_FRAC) or (area_frac >= NEAR_AREA_FRAC)

def choose_main_door(doors):
    # Pick largest area door (better target)
    if not doors:
        return None
    return max(doors, key=lambda d: (d["area"]))

def navigation_decision(frame_w, frame_h, dets):
    """
    dets: list of dicts: 
      {"label","conf","x1","y1","x2","y2","pos","near","area"}
    Returns (text, color).
    """
    # Split
    doors = [d for d in dets if d["label"] == "door"]
    stairs = [d for d in dets if d["label"] == "stairs"]
    obstacles = [d for d in dets if d["label"] in OBSTACLES]

    left_band = int((1 - CENTER_FRAC) * 0.5 * frame_w)
    right_band = frame_w - left_band

    # Any near obstacle inside center band?
    center_obstacles_near = [
        o for o in obstacles 
        if o["near"] and not (o["x2"] < left_band or o["x1"] > right_band)
    ]

    # Prefer stairs first (explicit instruction)
    if stairs:
        # Where are the stairs?
        # Pick the most central stairs (min distance to frame center)
        center_x = frame_w / 2.0
        st = min(stairs, key=lambda s: abs(((s["x1"] + s["x2"]) / 2.0) - center_x))
        if st["pos"] == "left":
            return "TAKE THE STAIRS — GO LEFT", COLOR_TURN
        elif st["pos"] == "right":
            return "TAKE THE STAIRS — GO RIGHT", COLOR_TURN
        else:
            return "TAKE THE STAIRS — FORWARD", COLOR_GO

    # If door(s) and center is clear → walk forward
    if doors and len(center_obstacles_near) == 0:
        main_door = choose_main_door(doors)
        if main_door:
            if main_door["pos"] == "left":
                return "TURN LEFT TOWARD DOOR", COLOR_TURN
            elif main_door["pos"] == "right":
                return "TURN RIGHT TOWARD DOOR", COLOR_TURN
            else:
                return "WALK FORWARD — DOOR AHEAD", COLOR_GO

    # If center near obstacle → stop, suggest side if one is clearer
    if len(center_obstacles_near) > 0:
        # Count near obstacles left vs right bands
        left_blocked = any(o["x2"] <= frame_w * 0.5 for o in center_obstacles_near)
        right_blocked = any(o["x1"] >= frame_w * 0.5 for o in center_obstacles_near)
        if left_blocked and not right_blocked:
            return "STOP — OBSTACLE AHEAD • TRY RIGHT", COLOR_STOP
        elif right_blocked and not left_blocked:
            return "STOP — OBSTACLE AHEAD • TRY LEFT", COLOR_STOP
        else:
            return "STOP — PATH BLOCKED", COLOR_STOP

    # Default
    return "WALK FORWARD", COLOR_GO

def main():
    # Load model
    model = YOLO(MODEL_PATH)
    model_name = MODEL_PATH

    # Camera
    def open_camera(idx):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap

    cam_index = 0
    cap = open_camera(cam_index)
    if cap is None:
        # Try a few indices
        for i in range(1, 4):
            cap = open_camera(i)
            if cap:
                cam_index = i
                break
    if cap is None:
        raise RuntimeError("No camera found.")

    fullscreen = True
    flipped = True
    set_fullscreen(fullscreen)

    prev_time = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            cap = open_camera(cam_index)
            if not cap:
                break
            continue

        if flipped:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        left_band = int((1 - CENTER_FRAC) * 0.5 * w)
        right_band = w - left_band

        # Detection (Conf & classes filtered)
        results = model.predict(frame, conf=CONF_THRESH, verbose=False)
        dets = []
        if results and len(results):
            r = results[0]
            for box in r.boxes:
                cls_id = int(box.cls)
                label = r.names.get(cls_id, str(cls_id))
                if label not in TARGET_CLASSES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy()) if box.conf is not None else 0.0
                pos = object_position(x1, x2, w)
                near = is_near(x1, y1, x2, y2, w, h)
                area = (x2 - x1) * (y2 - y1)

                # Draw fancy box + label
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)
                tag = f"{label.upper()} {conf:.2f} ({pos}){' NEAR' if near else ''}"
                cv2.putText(frame, tag, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                dets.append({
                    "label": label, "conf": conf,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "pos": pos, "near": near, "area": area
                })

        # Decision
        main_text, color = navigation_decision(w, h, dets)

        # HUD top bar
        draw_hud_text(frame, f"Instruction: {main_text}", color)

        # Center band guides
        cv2.line(frame, (left_band, 0), (left_band, h), COLOR_BAND, 1, cv2.LINE_AA)
        cv2.line(frame, (right_band, 0), (right_band, h), COLOR_BAND, 1, cv2.LINE_AA)

        # FPS
        now = time.time()
        dt = now - prev_time
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        prev_time = now

        draw_status(frame, cam_index, flipped, fullscreen, fps, model_name)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            fullscreen = not fullscreen
            set_fullscreen(fullscreen)
        elif key == ord('c'):
            flipped = not flipped
        elif key == ord('b'):
            new_idx = (cam_index + 1) % 4
            new_cap = cv2.VideoCapture(new_idx, cv2.CAP_DSHOW)
            if new_cap.isOpened():
                cap.release()
                cap = new_cap
                cam_index = new_idx

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
