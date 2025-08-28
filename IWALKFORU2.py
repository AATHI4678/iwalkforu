import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

NAV_OBJECTS = ["door", "stairs"]
OBSTACLES = ["chair", "person", "table", "bench"]

def get_object_position(box, frame_width):
    x1, _, x2, _ = map(int, box.xyxy[0])
    obj_center = (x1 + x2) // 2
    if obj_center < frame_width / 3:
        return "left"
    elif obj_center > (frame_width / 3) * 2:
        return "right"
    else:
        return "center"

def navigation_logic(nav_objs, obstacle_objs):
    if "stairs" in nav_objs:
        return "Take the stairs", (0, 255, 255)  # Yellow
    if "door" in nav_objs:
        door_pos = nav_objs["door"]
        if door_pos == "left":
            return "Turn left towards the door", (0, 128, 255)  # Orange
        elif door_pos == "right":
            return "Turn right towards the door", (0, 128, 255)
        else:
            return "Walk forward through the door", (0, 255, 0)  # Green
    if obstacle_objs:
        if "center" in obstacle_objs:
            if "left" not in obstacle_objs:
                return "Turn left to avoid obstacle", (255, 0, 0)  # Blue
            elif "right" not in obstacle_objs:
                return "Turn right to avoid obstacle", (255, 0, 0)
            else:
                return "STOP — path blocked", (0, 0, 255)  # Red
        else:
            return "Walk forward", (0, 255, 0)
    return "Walk forward", (0, 255, 0)

def draw_hud_text(frame, text, color):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

# Open webcam
cap = cv2.VideoCapture(0)
flip_cam = True  # default flipped

# Fullscreen window
cv2.namedWindow("Jarvis Navigation HUD", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Jarvis Navigation HUD", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip if toggle is on
    if flip_cam:
        frame = cv2.flip(frame, 1)

    frame_height, frame_width = frame.shape[:2]
    results = model(frame)
    names = model.names

    detected_nav_objs = {}
    detected_obstacles = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = names[cls_id]
            pos = get_object_position(box, frame_width)

            # Fancy bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"{label.upper()} ({pos})", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            if label in NAV_OBJECTS:
                detected_nav_objs[label] = pos
            elif label in OBSTACLES:
                detected_obstacles.append(pos)

    instruction, color = navigation_logic(detected_nav_objs, detected_obstacles)
    draw_hud_text(frame, f"Instruction: {instruction}", color)

    cv2.imshow("Jarvis Navigation HUD", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('f'):  # Toggle flip
        flip_cam = not flip_cam

cap.release()
cv2.destroyAllWindows()

