# main.py
import os
import time
import threading
from dotenv import load_dotenv
from camera_view import CameraView
from audio_stream import AudioStream
import openai
import pyttsx3

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize TTS
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Speech speed

def ask_gpt(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"[ERROR] GPT request failed: {e}")
        return "I am having trouble thinking right now."

def audio_listener(audio_stream):
    while True:
        clip = audio_stream.record_clip()
        if clip is not None:
            # Basic detection placeholder: convert audio to text using GPT
            prompt = "The user said: [audio clip placeholder]"
            gpt_response = ask_gpt(prompt)
            if any(greet in gpt_response.lower() for greet in ["hello", "hi", "hey"]):
                gpt_response = "Yes, sir."
            print(f"You said: {gpt_response}")
            tts_engine.say(gpt_response)
            tts_engine.runAndWait()

def main():
    print("[INFO] Detecting camera...")
    camera_index = 0  # Default USB camera index
    camera = CameraView(camera_index)

    print("[INFO] Detecting microphone...")
    audio = AudioStream(duration=3)  # Continuous listening

    # Start audio thread
    listener_thread = threading.Thread(target=audio_listener, args=(audio,), daemon=True)
    listener_thread.start()

    print("[INFO] System ready. Press 'q' in camera window to quit.")

    try:
        while True:
            frame = camera.get_frame()
            if frame is not None:
                camera.show_frame(frame)  # Mirrored feed
            time.sleep(1)  # Slow processing to mimic original
    except KeyboardInterrupt:
        print("[INFO] Exiting...")
    finally:
        camera.release()
        audio.close()

if __name__ == "__main__":
    main()
