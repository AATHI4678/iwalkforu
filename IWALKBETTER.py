import os
import time
import base64
import json
import threading
from typing import Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# Settings you can tweak
# =========================
MODEL_NAME = "gpt-4o-mini"
SEND_INTERVAL_SEC = 1.0          # send 1 frame per second to the API
JPEG_QUALITY = 80                # lower => smaller upload size
SEND_WIDTH = 640                 # downscale before sending (keeps UI smooth)
WINDOW_NAME = "Jarvis Navigation HUD"

# Keyboard:
#   Q = quit
#   F = toggle fullscreen
#   C = toggle mirror/flip
#   B = switch to next camera (front/back/etc)
# =========================

load_dotenv()  # load OPENAI_API_KEY from .env if present

# OpenAI client
client = OpenAI(api_key=os.getenv("proj-xodfWF1qkfeHCWBV4mczT3BlbkFJWGGVqLzbLgmKEwHsV76G"))

# HUD colors (BGR)
COLOR_GO = (0, 255, 0)
COLOR_TURN = (255, 0, 0)
COLOR_STOP = (0, 0, 255)
COLOR_INFO = (255, 255, 255)
COLOR_BOX = (0, 255, 255)

def draw_hud_text(frame, text, color):
    """Top translucent bar with big instruction text."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 64), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    cv2.putText(frame, text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)

def draw_status(frame, cam_index, flipped, fullscreen, latency_ms, last_update_str):
    """Small status row bottom-left."""
    lines = [
        f"CAM:{cam_index}  FLIP:{'ON' if flipped else 'OFF'}  FULL:{'ON' if fullscreen else 'OFF'}",
        f"API:{latency_ms:.0f} ms  LAST:{last_update_str}"
    ]
    x, y = 16, frame.shape[0] - 48
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + 22*i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_INFO, 1, cv2.LINE_AA)

def encode_frame_for_api(frame) -> str:
    """Downscale and JPEG-encode frame, then base64 for data URL."""
    h, w = frame.shape[:2]
    scale = SEND_WIDTH / float(w)
    if scale < 1.0:
        frame = cv2.resize(frame, (SEND_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode frame.")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

PROMPT = (
    "You are a navigation safety checker for a forward-facing camera. "
    "Look only for obstacles blocking a straight walking path within the CENTER third of the frame. "
    "Return STRICT JSON (no extra text) with keys:\n"
    '{'
    '"action": "WALK" or "STOP", '
    '"direction": "forward" or "left" or "right", '
    '"reason": "short explanation"'
    "}\n"
    "Rules:\n"
    "- If the center path looks clear for ~2–3 meters, action=WALK, direction=forward.\n"
    "- If an obstacle blocks center, action=STOP. Then suggest direction left/right if one side looks clearer.\n"
    "- Only output JSON. No prose.\n"
)

def call_openai(image_data_url: str) -> Tuple[str, str, str]:
    """
    Send one frame to OpenAI Vision and parse JSON.
    Returns (action, direction, reason).
    """
    # Using Chat Completions w/ image per current SDK patterns
    start = time.time()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": image_data_url}}
            ]
        }]
    )
    latency_ms = (time.time() - start) * 1000.0
    text = resp.choices[0].message.content.strip()

    # Be robust: extract first {...} JSON in the output
    try:
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1:
            text = text[first:last+1]
        data = json.loads(text)
        action = str(data.get("action", "WALK")).upper()
        direction = str(data.get("direction", "forward")).lower()
        reason = str(data.get("reason", ""))
    except Exception:
        action, direction, reason = "WALK", "forward", "vision parse fallback"

    return action, direction, reason, latency_ms

class VisionWorker:
    """Background thread that sends frames every SEND_INTERVAL_SEC and stores the latest result."""
    def __init__(self):
        self.lock = threading.Lock()
        self.last_action = "WALK"
        self.last_direction = "forward"
        self.last_reason = ""
        self.last_latency_ms = 0.0
        self.last_update_ts = 0.0
        self.next_due = 0.0
        self.running = True
        self.pending = False

    def stop(self):
        self.running = False

    def maybe_send(self, frame_bgr):
        now = time.time()
        if self.pending or now < self.next_due:
            return
        self.pending = True

        # Copy & encode in a thread
        frame_copy = frame_bgr.copy()
        def _job():
            try:
                data_url = encode_frame_for_api(frame_copy)
                action, direction, reason, latency_ms = call_openai(data_url)
                with self.lock:
                    self.last_action = action
                    self.last_direction = direction
                    self.last_reason = reason
                    self.last_latency_ms = latency_ms
                    self.last_update_ts = time.time()
            finally:
                self.next_due = time.time() + SEND_INTERVAL_SEC
                self.pending = False

        threading.Thread(target=_job, daemon=True).start()

    def read(self):
        with self.lock:
            return (self.last_action, self.last_direction, self.last_reason,
                    self.last_latency_ms, self.last_update_ts)

def open_camera(index: int) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # CAP_DSHOW helps on Windows
    if not cap.isOpened():
        return None
    # Try to set a reasonable resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap

def set_fullscreen(on: bool):
    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL)

def main():
    cam_index = 0
    cap = open_camera(cam_index)
    if cap is None:
        # fallback to 0..3 search
        for i in range(3):
            cap = open_camera(i)
            if cap: 
                cam_index = i
                break
    if cap is None:
        raise RuntimeError("No camera found.")

    fullscreen = True
    flipped = True  # start mirrored
    set_fullscreen(fullscreen)

    worker = VisionWorker()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # try reopen
                cap.release()
                cap = open_camera(cam_index)
                if not cap:
                    break
                continue

            if flipped:
                frame = cv2.flip(frame, 1)

            # Ask background thread to send frame if due
            worker.maybe_send(frame)

            # Read latest decision
            action, direction, reason, latency_ms, last_ts = worker.read()

            # Decide HUD color + main line
            if action == "STOP":
                main_text = "STOP — Obstacle Ahead"
                color = COLOR_STOP
                if direction in ("left", "right"):
                    main_text += f" • Try {direction.upper()}"
            else:
                main_text = "WALK FORWARD"
                color = COLOR_GO

            draw_hud_text(frame, main_text, color)

            # Optional: crosshair for center third
            h, w = frame.shape[:2]
            left_band = w // 3
            right_band = (w * 2) // 3
            cv2.line(frame, (left_band, 0), (left_band, h), (50, 50, 50), 1, cv2.LINE_AA)
            cv2.line(frame, (right_band, 0), (right_band, h), (50, 50, 50), 1, cv2.LINE_AA)

            # Status footer
            last_update_str = time.strftime("%H:%M:%S", time.localtime(last_ts)) if last_ts > 0 else "--:--:--"
            draw_status(frame, cam_index, flipped, fullscreen, latency_ms, last_update_str)

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
                # next camera
                new_index = cam_index + 1
                if new_index > 3:
                    new_index = 0
                new_cap = open_camera(new_index)
                if new_cap:
                    cap.release()
                    cap = new_cap
                    cam_index = new_index
    finally:
        worker.stop()
        if cap:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
