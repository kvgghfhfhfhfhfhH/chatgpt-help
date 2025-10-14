# audio_stream.py
import sounddevice as sd
import numpy as np

class AudioStream:
    def __init__(self, samplerate=16000, duration=3):
        self.samplerate = samplerate
        self.duration = duration

    def record_clip(self):
        try:
            data = sd.rec(int(self.duration * self.samplerate), samplerate=self.samplerate, channels=1)
            sd.wait()
            return data
        except Exception as e:
            print(f"[ERROR] Audio recording failed: {e}")
            return None

    def close(self):
        pass  # Cleanup if needed
