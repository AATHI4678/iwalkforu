import cv2
import torch

from ultralytics import solutions


def main(inputSource=0):
    cap = cv2.VideoCapture(inputSource)
    assert cap.isOpened(), "Error reading video file"

    # # Set resolution
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Width
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Height

    # Video writer
    w, h, fps = (
        int(cap.get(x))
        for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
    )
    # video_writer = cv2.VideoWriter(
    #     "region_counting.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    # )

    # Pass region as list
    region_points = {
        "bottom": [(0, h), (w, h), (w, h // 2), (0, h // 2)],
        "top": [(0, 0), (w, 0), (w, h // 2), (0, h // 2)],
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)
    # Initialize region counter object
    regioncounter = solutions.RegionCounter(
        show=True,  # display the frame
        region=region_points,  # pass region points
        model="yolo11n.pt",  # model for counting in regions i.e yolo11s.pt
        device=device,
        classes=[0],
    )

    # Process video
    while cap.isOpened():
        success, im0 = cap.read()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if not success:
            print("Video frame is empty or processing is complete.")
            break

        results = regioncounter(im0)

        print(f"Device is {device} and FPS is {fps}.")

        print(results.region_counts)  # access the output

        # video_writer.write(results.plot_im)

    cap.release()
    
    cv2.destroyAllWindows()  # destroy all opened windows


main("rtsp://localhost:8554/mystream")
