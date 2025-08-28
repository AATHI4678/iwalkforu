import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Objects we care about
NAV_OBJECTS = ["door", "stairs"]
OBSTACLES = ["chair", "person", "table", "bench"]

def get_object_position(box, frame_width):
    """Return object position: left, center, or right."""
    x1, _, x2, _ = map(int, box.xyxy[0])
    obj_center = (x1 + x2) // 2

    if obj_center < frame_width / 3:
        return "left"
    elif obj_center > (frame_width / 3) * 2:
        return "right"
    else:
        return "center"

def navigation_logic(nav_objs, obstacle_objs):
    """Decide navigation instruction based on object positions."""
    if "stairs" in nav_objs:
        return "Take the stairs"

    if "door" in nav_objs:
        # Check if door is left, center, or right
        door_pos = nav_objs["door"]
        if door_pos == "left":
            return "Turn left towards the door"
        elif door_pos == "right":
            return "Turn right towards the door"
        else:
            return "Walk forward through the door"

    if obstacle_objs:
        # If obstacle in front, suggest turning
        if "center" in obstacle_objs:
            if "left" not in obstacle_objs:
                return "Turn left to avoid obstacle"
            elif "right" not in obstacle_objs:
                return "Turn right to avoid obstacle"
            else:
                return "Stop — path blocked"
        else:
            return "Walk forward"

    return "Walk forward"

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

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

            # Draw detection box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({pos})", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if label in NAV_OBJECTS:
                detected_nav_objs[label] = pos
            elif label in OBSTACLES:
                detected_obstacles.append(pos)

    # Get navigation instruction
    instruction = navigation_logic(detected_nav_objs, detected_obstacles)

    # Show navigation text on screen
    cv2.putText(frame, f"Instruction: {instruction}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Advanced Navigation AI", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
