# main.py
import os
import time
import cv2
import sounddevice as sd
import numpy as np
import speech_recognition as sr
from dotenv import load_dotenv
from openai import OpenAI
from camera_view import CameraView
from audio_stream import AudioStream

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_default_camera_index():
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
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            return i
    raise RuntimeError("No microphone detected!")

def recognize_speech(mic_index):
    recognizer = sr.Recognizer()
    with sr.Microphone(device_index=mic_index) as source:
        print("[LISTENING] Waiting for speech...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source, phrase_time_limit=3)

    try:
        text = recognizer.recognize_google(audio).lower()
        return text
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        return None

def respond_to_speech(text):
    if text is None:
        return

    print(f"You said: {text}")

    # Trigger GPT only for greetings
    if any(greet in text for greet in ["hello", "hi", "hey"]):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a polite assistant."},
                {"role": "user", "content": "Say 'Yes, sir.'"}
            ]
        )
        print("GPT:", response.choices[0].message.content)
    else:
        print("[INFO] No greeting detected.")

def main():
    cam_index = get_default_camera_index()
    print(f"[INFO] Using camera index: {cam_index}")
    camera = CameraView(camera_index=cam_index)

    mic_device = get_default_mic_device()
    print(f"[INFO] Using microphone device: {mic_device}")

    print("[INFO] Starting system. Press Ctrl+C to quit.")
    recognizer = sr.Recognizer()

    try:
        while True:
            frame = camera.get_frame()
            if frame is not None:
                print("[INFO] Frame captured.")

            text = recognize_speech(mic_device)
            respond_to_speech(text)

            time.sleep(1)
    except KeyboardInterrupt:
        print("[INFO] Exiting...")
    finally:
        camera.release()

if __name__ == "__main__":
    main()
