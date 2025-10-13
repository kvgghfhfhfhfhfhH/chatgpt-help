# audio_stream.py
import sounddevice as sd
import numpy as np

class AudioStream:
    def __init__(self, device=None, samplerate=16000, duration=3):
        self.device = device
        self.samplerate = samplerate
        self.duration = duration

    def record_short_clip(self):
        """Record a short audio clip from the microphone."""
        try:
            print("[DEBUG] Listening for sound...")
            audio = sd.rec(
                int(self.duration * self.samplerate),
                samplerate=self.samplerate,
                channels=1,
                dtype="float32",
                device=self.device,
            )
            sd.wait()

            # Return audio only if non-silent
            volume = np.max(np.abs(audio))
            if volume < 0.01:
                # Silence threshold
                print("[DEBUG] Silence detected, skipping.")
                return None
            return audio
        except Exception as e:
            print(f"[ERROR] Audio recording failed: {e}")
            return None

    def close(self):
        """Nothing to close for now; for future expansion."""
        pass
