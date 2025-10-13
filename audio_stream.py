# audio_stream.py
import sounddevice as sd
import numpy as np

class AudioStream:
    def __init__(self, device=None, samplerate=16000, duration=1):
        self.device = device
        self.samplerate = samplerate
        self.duration = duration

    def record_short_clip(self):
        try:
            data = sd.rec(int(self.duration * self.samplerate), samplerate=self.samplerate, channels=1, device=self.device)
            sd.wait()
            return data
        except Exception as e:
            print(f"[ERROR] Audio recording failed: {e}")
            return None

    def close(self):
        pass
