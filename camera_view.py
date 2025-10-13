# camera_view.py
import cv2
import numpy as np

class CameraView:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.opened = self.cap.isOpened()
        self.human_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        if not self.opened:
            print(f"[ERROR] Cannot open camera at index {camera_index}")

    def get_frame(self):
        if not self.opened:
            return None
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def detect_human(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        humans = self.human_cascade.detectMultiScale(gray, 1.1, 3)
        if len(humans) > 0:
            print("[CAMERA] Human detected!")

    def detect_color(self, frame):
        avg_color = frame.mean(axis=(0,1))
        print(f"[CAMERA] Average BGR color: {avg_color}")

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
