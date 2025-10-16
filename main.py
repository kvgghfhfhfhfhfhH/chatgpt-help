import cv2
import sounddevice as sd
import numpy as np
import queue
import threading
import openai
import os
from dotenv import load_dotenv
import time

# Load OpenAI key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("[ERROR] OpenAI API key not found in .env!")
openai.api_key = OPENAI_API_KEY

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CLIP_DURATION = 3  # seconds
AUDIO_QUEUE = queue.Queue()

# Camera settings
CAM_INDEX = 0
FRAME_QUEUE = queue.Queue()

# Gender detection placeholders
def detect_gender(text):
    text = text.lower()
    if any(x in text for x in ["he", "him", "sir", "male"]):
        return "male"
    elif any(x in text for x in ["she", "her", "maam", "female"]):
        return "female"
    else:
        return "unknown"

# Thread: capture audio continuously
def audio_listener(device=None):
    def callback(indata, frames, time_, status):
        if status:
            print(f"[ERROR] {status}")
        AUDIO_QUEUE.put(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback, device=device):
        while True:
            sd.sleep(1000)

# Thread: capture frames continuously
def camera_listener():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera at index {CAM_INDEX}")
        return
    while True:
        ret, frame = cap.read()
        if ret:
            FRAME_QUEUE.put(frame)
        time.sleep(0.1)

# Process audio and respond with GPT
def process_audio():
    while True:
        if AUDIO_QUEUE.empty():
            time.sleep(0.1)
            continue
        clip = AUDIO_QUEUE.get()
        # Convert audio to raw bytes for OpenAI transcription
        audio_bytes = clip.tobytes()
        try:
            # Use OpenAI transcription API (Whisper)
            transcript = openai.audio.transcriptions.create(
                file=openai.File.create(file=audio_bytes, filename="clip.wav"),
                model="whisper-1"
            )
            text = transcript["text"]
            if text.lower().startswith("hey nova"):
                # Only trigger on "Hey Nova"
                query = text[len("hey nova"):].strip()
                gender = detect_gender(query)
                print(f"You said: {query}")
                print(f"Detected speaker gender: {gender}")

                # GPT-4 response
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are Nova, helpful assistant."},
                        {"role": "user", "content": query}
                    ]
                )
                answer = response.choices[0].message.content
                print(f"Nova: {answer}")

                # Show the latest frame when answering
                if not FRAME_QUEUE.empty():
                    frame = FRAME_QUEUE.get()
                    cv2.imshow("Nova Camera View", frame)
                    cv2.waitKey(1)

        except Exception as e:
            print(f"[ERROR] Transcription failed: {e}")

def main():
    print("[INFO] Detecting camera and microphone...")
    print("[INFO] System ready. Say 'Hey Nova' to interact.")

    # Start threads
    threading.Thread(target=audio_listener, daemon=True).start()
    threading.Thread(target=camera_listener, daemon=True).start()
    threading.Thread(target=process_audio, daemon=True).start()

    try:
        while True:
            time.sleep(1)  # Keep main thread alive
    except KeyboardInterrupt:
        print("[INFO] Exiting...")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
