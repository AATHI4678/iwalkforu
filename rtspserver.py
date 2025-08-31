import cv2
import threading
import subprocess
import re


class WebcamRTSPStreamer:
    def __init__(self, webcam_name, rtsp_url='rtsp://localhost:8554/mystream'):
        self.webcam_name = webcam_name
        self.rtsp_url = rtsp_url
        self.stream_process = None
# ffmpeg -f dshow -i - -c:v libx264 -preset ultrafast -b:v 1000k -f rtsp rtsp://localhost:8554/mystream


def list_webcams():
    try:
        print("Available Webcams:")
        # List DirectShow video devices
        result = subprocess.run(
            ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy", "-hide_banner"],
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        availableCameras = [str(i) for i in result.stderr.split("\n")]

        for i in availableCameras:
            if "Alternative" not in i or "error" in i:
                    print(re.findall(r'"([^"]+)"', i)[0])
    except Exception as e:
        print(f"Error detecting webcams: {e}")


list_webcams()

# def openCamera():

#     pass

# ffmpeg -f dshow -i video="EMEET SmartCam Nova 4K" -vf "scale=1920:1080" -c:v libx264 -preset ultrafast  -tune zerolatency -b:v 1000k -f rtsp rtsp://localhost:8554/mystream -hide_banner

# ffmpeg -f dshow -i video="EMEET SmartCam Nova 4K" -vf "scale=1920:1080" -c:v libx264 -preset slower -f rtsp rtsp://localhost:8554/mystream -hide_banner
