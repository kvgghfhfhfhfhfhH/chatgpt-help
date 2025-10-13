# main.py
import os
from dotenv import load_dotenv
import time
import numpy as np

from camera_view import CameraView
from audio_stream import AudioStream

# Load OpenAI key
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

import openai
openai.api_key = OPENAI_KEY

def get_default_camera_index():
    import cv2
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.release()
            return index
        index += 1
        if index > 10:
            raise RuntimeError("No camera detected!")

def get_default_mic_device():
    import sounddevice as sd
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            return i
    raise RuntimeError("No microphone detected!")

def recognize_audio(clip):
    # For demo, just convert to string shape info
    return f"Audio clip shape: {clip.shape}"

def chatgpt_response(prompt):
    # If greeting, reply "sir"
    greetings = ["hello", "hi", "hey", "greetings"]
    if any(g.lower() in prompt.lower() for g in greetings):
        return "... sir"
    
    # Use OpenAI API for other prompts
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user", "content": prompt}]
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[ERROR] OpenAI API failed: {e}"

def main():
    cam_index = get_default_camera_index()
    print(f"[INFO] Using camera index: {cam_index}")
    camera = CameraView(camera_index=cam_index)

    mic_device = get_default_mic_device()
    print(f"[INFO] Using microphone device: {mic_device}")
    audio = AudioStream(device=mic_device, duration=3)

    try:
        print("[INFO] Starting camera and audio test. Press Ctrl+C to quit.")
        while True:
            frame = camera.get_frame()
            if frame is not None:
                # Placeholder for object detection: just print frame shape
                print(f"[INFO] Frame captured: {frame.shape}")

            clip = audio.record_short_clip()
            if clip is not None:
                recognized = recognize_audio(clip)
                print(f"[INFO] {recognized}")

                # Use ChatGPT to respond
                response = chatgpt_response(recognized)
                print(f"[ChatGPT] {response}")

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("[INFO] Exiting...")

    finally:
        camera.release()
        audio.close()

if __name__ == "__main__":
    main()
