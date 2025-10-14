# main.py
import os
import threading
import time
import queue
from dotenv import load_dotenv
import sounddevice as sd
import soundfile as sf
import numpy as np
import pyttsx3
import openai
import cv2

# Load OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("[ERROR] OpenAI API key not found in .env!")

# TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

# Audio & Camera
SAMPLERATE = 16000
MIC_DEVICE = None
CAMERA_INDEX = 0
FRAME_INTERVAL = 1.0
camera_enabled = True

# Queues
audio_queue = queue.Queue()
frame_queue = queue.Queue()

# Detect default mic
def get_default_mic():
    global MIC_DEVICE
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            MIC_DEVICE = i
            print(f"[INFO] Using microphone device {i}: {d['name']}")
            return
    raise RuntimeError("No microphone detected!")

# Detect camera
def get_default_camera():
    global CAMERA_INDEX
    index = 0
    while index < 10:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            CAMERA_INDEX = index
            cap.release()
            print(f"[INFO] Using camera index {index}")
            return
        index += 1
    print("[WARNING] No camera detected. Camera features disabled.")

# Headless camera listener
def camera_listener():
    global camera_enabled
    cap = cv2.VideoCapture(CAMERA_INDEX)
    while True:
        if camera_enabled:
            ret, frame = cap.read()
            if ret:
                frame_queue.put(frame)
                print("[INFO] Frame captured")
        time.sleep(FRAME_INTERVAL)

# Audio listener with voice activity detection (VAD)
def audio_listener():
    def callback(indata, frames, time_info, status):
        audio_queue.put(indata.copy())
    with sd.InputStream(samplerate=SAMPLERATE, device=MIC_DEVICE, channels=1, callback=callback):
        while True:
            time.sleep(0.1)

# Wait until user stops speaking
def process_audio_clip(clip):
    filename = "temp.wav"
    sf.write(filename, clip, SAMPLERATE)
    with open(filename, "rb") as f:
        result = openai.Audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    return result.text.strip()

# Ask GPT
def ask_gpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

# Main loop
def main():
    global camera_enabled
    get_default_mic()
    get_default_camera()

    print("[INFO] System ready. Press Ctrl+C to quit.")

    threading.Thread(target=camera_listener, daemon=True).start()
    threading.Thread(target=audio_listener, daemon=True).start()

    accumulated_clip = []
    speaking = False

    while True:
        # Handle audio
        while not audio_queue.empty():
            clip = audio_queue.get()
            accumulated_clip.append(clip)
        
        if accumulated_clip and not speaking:
            audio_data = np.concatenate(accumulated_clip, axis=0)
            accumulated_clip = []
            if np.abs(audio_data).mean() > 0.01:  # simple threshold for speech
                try:
                    text = process_audio_clip(audio_data)
                    if text:
                        print(f"You said: {text}")

                        # Camera control commands
                        if "hide camera" in text.lower():
                            camera_enabled = False
                            print("[INFO] Camera disabled.")
                            tts_engine.say("Camera disabled.")
                            tts_engine.runAndWait()
                            continue
                        elif "enable camera" in text.lower():
                            camera_enabled = True
                            print("[INFO] Camera enabled.")
                            tts_engine.say("Camera enabled.")
                            tts_engine.runAndWait()
                            continue

                        speaking = True
                        answer = ask_gpt(text)
                        print(f"GPT: {answer}")
                        tts_engine.say(answer)
                        tts_engine.runAndWait()
                        speaking = False

                except Exception as e:
                    print(f"[ERROR] {e}")

        # Handle camera frames (object detection can be added here)
        while not frame_queue.empty():
            frame = frame_queue.get()
            # Example: save latest frame
            # cv2.imwrite("latest_frame.jpg", frame)
            if camera_enabled:
                print("[INFO] Frame captured")

        time.sleep(0.1)

if __name__ == "__main__":
    main()
