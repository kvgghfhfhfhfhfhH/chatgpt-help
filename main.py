# main.py
import os
import time
import cv2
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from camera_view import CameraView
from audio_stream import AudioStream
import openai
import queue
import threading

# Load OpenAI key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ----------------------------
# Configuration
# ----------------------------
CAMERA_INDEX = 0          # USB webcam
MIC_DEVICE = 1            # Replace with your mic device number (from `arecord -l`)
SPEAKER_DEVICE = 2        # Replace with your speaker device number (from `aplay -l`)
SAMPLERATE = 16000
CLIP_DURATION = 1.0       # seconds
THRESHOLD_VOLUME = 0.01   # Minimum volume to trigger recognition

# Queue to communicate audio between threads
audio_queue = queue.Queue()

# ----------------------------
# Helper functions
# ----------------------------
def detect_greeting(text: str):
    greetings = ["hello", "hi", "hey", "greetings"]
    for g in greetings:
        if g in text.lower():
            return True
    return False

def ask_gpt(prompt: str):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are polite and call the user 'sir' for greetings."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

# ----------------------------
# Audio thread function
# ----------------------------
def audio_listener(audio_stream: AudioStream):
    while True:
        clip = audio_stream.record_clip()
        if clip is not None:
            # Check volume to avoid false triggers
            volume = np.abs(clip).mean()
            if volume > THRESHOLD_VOLUME:
                audio_queue.put(clip)
        time.sleep(0.1)

# ----------------------------
# Main loop
# ----------------------------
def main():
    print("[INFO] Using camera index:", CAMERA_INDEX)
    print("[INFO] Using microphone device:", MIC_DEVICE)
    print("[INFO] System ready. Listening and observing...")

    # Initialize camera (we won't show GUI to avoid XCB issues)
    camera = CameraView(CAMERA_INDEX)

    # Initialize audio
    audio = AudioStream(input_device=MIC_DEVICE, output_device=SPEAKER_DEVICE,
                        samplerate=SAMPLERATE, duration=CLIP_DURATION)

    # Start audio listener thread
    t = threading.Thread(target=audio_listener, args=(audio,), daemon=True)
    t.start()

    try:
        while True:
            frame = camera.get_frame()
            if frame is not None:
                # Process frame if needed (e.g., object detection)
                print("[INFO] Frame captured.")

            # Process audio clips in queue
            while not audio_queue.empty():
                clip = audio_queue.get()
                print("[INFO] Audio clip recorded.")
                # Convert audio clip to text here if you have a STT model
                # For demo, let's assume it always "hears" hello
                recognized_text = "hello"
                if detect_greeting(recognized_text):
                    print("You said:", recognized_text)
                    gpt_response = "Yes, sir."
                    print("GPT:", gpt_response)
                    # Optional: use sounddevice to speak GPT response
                    # You could integrate pyttsx3 or gTTS for TTS here

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("[INFO] Exiting...")
    finally:
        camera.release()
        audio.close()

if __name__ == "__main__":
    main()
