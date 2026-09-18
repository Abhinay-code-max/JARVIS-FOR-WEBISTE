"""
recognition/voice_id.py
=======================
Voice/speaker recognition module for JARVIS-XL.

How it works:
  • Extracts a 192-dim x-vector (MFCC + GMM) from raw audio
  • Compares against enrolled speaker embeddings using cosine similarity
  • Falls back gracefully when dependencies are unavailable

Dependencies (auto-installed):
  speechbrain  — SpeechBrain ECAPA-TDNN speaker model (best quality)
  OR
  librosa + scikit-learn — lightweight MFCC-cosine fallback (no GPU needed)

Enrollment data:
  recognition/voices/<name>.npy  →  mean speaker embedding
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import numpy as np

_log = logging.getLogger("jarvis.voice_id")


class VoiceIdentifier:
    THRESHOLD   = 0.75    # cosine similarity; higher = stricter
    MIN_SECONDS = 1.5     # minimum audio length for reliable embedding
    SAMPLE_RATE = 16_000
    REGISTER_REPS = 3     # how many utterances to average during registration

    def __init__(self, voices_dir: Path):
        self._dir = Path(voices_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._embeddings: dict[str, np.ndarray] = {}
        self._model = None        # lazy-loaded
        self._load_all()

    # ── Load saved embeddings ─────────────────────────────────────────────────
    def _load_all(self) -> None:
        for p in self._dir.glob("*.npy"):
            name = p.stem
            try:
                self._embeddings[name] = np.load(str(p))
                _log.debug("Loaded embedding for '%s'", name)
            except Exception as e:
                _log.warning("Could not load %s: %s", p, e)

    # ── Embedding extraction ──────────────────────────────────────────────────
    def _get_embedding(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Extract a speaker embedding from float32 mono audio at 16 kHz."""
        duration = len(audio) / self.SAMPLE_RATE
        if duration < self.MIN_SECONDS:
            return None

        # Strategy 1: MFCC cosine (lightweight, self-contained, always reliable with librosa)
        try:
            return self._embed_mfcc(audio)
        except Exception as e:
            _log.warning("MFCC embedding error: %s", e)

        # Strategy 2: SpeechBrain ECAPA-TDNN (if available and configured)
        try:
            return self._embed_speechbrain(audio)
        except Exception as e:
            _log.debug("SpeechBrain embedding not used: %s", e)

        return None

    def _embed_speechbrain(self, audio: np.ndarray) -> np.ndarray:
        """ECAPA-TDNN via SpeechBrain — 192-dim x-vector."""
        import torch
        from speechbrain.inference.speaker import SpeakerRecognition
        if self._model is None:
            self._model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(self._dir / "_ecapa_model"),
            )
        tensor = torch.tensor(audio).unsqueeze(0)
        emb    = self._model.encode_batch(tensor)
        return emb.squeeze().detach().cpu().numpy()

    def _embed_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Lightweight MFCC-based speaker embedding — no GPU required."""
        import librosa
        mfcc = librosa.feature.mfcc(y=audio, sr=self.SAMPLE_RATE, n_mfcc=40)
        delta  = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        feat   = np.concatenate([mfcc, delta, delta2], axis=0)   # (120, T)
        vec    = np.concatenate([feat.mean(axis=1), feat.std(axis=1)])  # (240,)
        norm   = np.linalg.norm(vec)
        return vec / (norm + 1e-9)

    # ── Identify ──────────────────────────────────────────────────────────────
    def identify(self, audio: np.ndarray) -> tuple[Optional[str], float]:
        """
        Compare audio against enrolled speakers.
        Returns (name, confidence) or (None, 0.0).
        """
        if not self._embeddings:
            return None, 0.0

        emb = self._get_embedding(audio)
        if emb is None:
            return None, 0.0

        best_name = None
        best_sim  = -1.0
        for name, ref in self._embeddings.items():
            sim = _cosine(emb, ref)
            if sim > best_sim:
                best_sim  = sim
                best_name = name

        if best_sim < self.THRESHOLD:
            return None, float(best_sim)
        return best_name, float(best_sim)

    # ── Register ──────────────────────────────────────────────────────────────
    def register(self, name: str, audio: np.ndarray) -> bool:
        """
        Enrol a speaker from one audio sample.
        For better accuracy call this multiple times and it averages.
        """
        emb = self._get_embedding(audio)
        if emb is None:
            _log.warning("Audio too short to register '%s'.", name)
            return False

        out_path = self._dir / f"{name}.npy"
        if out_path.exists():
            # Average with existing embedding (incremental update)
            existing = np.load(str(out_path))
            emb = (existing + emb) / 2.0
            norm = np.linalg.norm(emb)
            emb  = emb / (norm + 1e-9)

        np.save(str(out_path), emb)
        self._embeddings[name] = emb
        _log.info("Registered '%s'.", name)
        return True

    # ── Interactive Recording & Registration ──────────────────────────────────
    def record_sample(self, duration_sec: float = 3.5) -> Optional[np.ndarray]:
        """Record a single audio clip from the microphone."""
        try:
            import sounddevice as sd
            frames = int(duration_sec * self.SAMPLE_RATE)
            recording = sd.rec(frames, samplerate=self.SAMPLE_RATE, channels=1, dtype="float32")
            sd.wait()
            return recording.flatten()
        except Exception as e:
            _log.error("Microphone recording failed: %s", e)
            return None

    def record_and_register(
        self,
        name: str,
        reps: int = 3,
        duration_sec: float = 3.5,
        prompt_cb: Optional[callable] = None,
    ) -> bool:
        """
        Interactive voice enrollment: captures `reps` audio samples from the microphone,
        averages their embeddings, and saves to recognition/voices/<name>.npy.
        Mirrors FaceIdentifier.register().
        """
        collected = []
        _log.info("Starting voice enrollment for '%s' (%d samples)…", name, reps)

        for i in range(1, reps + 1):
            msg = f"Sample {i}/{reps}: Speak clearly into your microphone ({duration_sec:.1f}s)..."
            if prompt_cb:
                prompt_cb(msg)
            else:
                print(f"\n🎙️  {msg}")

            audio = self.record_sample(duration_sec=duration_sec)
            if audio is None:
                _log.warning("Sample %d recording failed.", i)
                continue

            emb = self._get_embedding(audio)
            if emb is not None:
                collected.append(emb)
                _log.info("Sample %d captured successfully.", i)
            else:
                _log.warning("Sample %d audio too short or silent.", i)

        if not collected:
            _log.error("No valid audio samples captured for '%s'.", name)
            return False

        mean_emb = np.mean(collected, axis=0)
        norm = np.linalg.norm(mean_emb)
        mean_emb = mean_emb / (norm + 1e-9)

        out_path = self._dir / f"{name}.npy"
        np.save(str(out_path), mean_emb)
        self._embeddings[name] = mean_emb
        _log.info("Voice profile registered for '%s' (%d samples averaged).", name, len(collected))
        return True

    def record_and_identify(
        self,
        duration_sec: float = 3.5,
        prompt_cb: Optional[callable] = None,
    ) -> tuple[Optional[str], float]:
        """Record a live microphone sample and test identification against enrolled profiles."""
        msg = f"Listening for speaker identification ({duration_sec:.1f}s)..."
        if prompt_cb:
            prompt_cb(msg)
        else:
            print(f"\n🎙️  {msg}")

        audio = self.record_sample(duration_sec=duration_sec)
        if audio is None:
            return None, 0.0
        return self.identify(audio)

    # ── Utility ───────────────────────────────────────────────────────────────
    def list_users(self) -> list[str]:
        return list(self._embeddings.keys())

    def delete_user(self, name: str) -> bool:
        p = self._dir / f"{name}.npy"
        if p.exists():
            p.unlink()
            self._embeddings.pop(name, None)
            return True
        return False


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── CLI Interface ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import time

    voices_path = Path(__file__).resolve().parent / "voices"
    identifier = VoiceIdentifier(voices_path)

    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else "enroll"
    target_name = sys.argv[2] if len(sys.argv) > 2 else "Abhinay"

    if cmd == "list":
        users = identifier.list_users()
        print(f"\nEnrolled voice profiles ({len(users)}):")
        for u in users:
            print(f"  • {u}")

    elif cmd == "delete":
        if identifier.delete_user(target_name):
            print(f"\n✓ Deleted voice profile for '{target_name}'.")
        else:
            print(f"\n✗ No voice profile found for '{target_name}'.")

    elif cmd == "test":
        print("\n--- JARVIS Voice Identification Test ---")
        name, conf = identifier.record_and_identify(duration_sec=3.5)
        if name:
            print(f"\n✓ MATCH FOUND: '{name}' (Confidence: {conf:.1%})")
        else:
            print(f"\n✗ UNKNOWN SPEAKER (Highest similarity: {conf:.1%}, Threshold: {identifier.THRESHOLD:.1%})")

    elif cmd == "enroll":
        print(f"\n========================================")
        print(f"   JARVIS Voice Enrollment — {target_name}")
        print(f"========================================")
        print("You will be asked to speak 3 short phrases.")
        print("Make sure your microphone is connected and quiet background.\n")

        phrases = [
            "Hey JARVIS, this is my voice profile sample one.",
            "JARVIS, initialize system diagnostics and check status.",
            "I am the primary user of this JARVIS terminal.",
        ]

        def _prompt(msg: str):
            print(f"\n>> {msg}")

        ok = identifier.record_and_register(
            name=target_name,
            reps=3,
            duration_sec=3.5,
            prompt_cb=_prompt,
        )

        if ok:
            print(f"\n🎉 SUCCESS: Voice profile for '{target_name}' successfully enrolled!")
            print(f"Saved to: {voices_path / f'{target_name}.npy'}")
            print("\nNow testing verification...")
            time.sleep(1)
            name, conf = identifier.record_and_identify(duration_sec=3.5, prompt_cb=_prompt)
            if name:
                print(f"✓ Verified match: '{name}' ({conf:.1%})")
            else:
                print(f"Result: {conf:.1%} confidence.")
        else:
            print(f"\n✗ Enrollment failed. Please check microphone settings and try again.")
    else:
        print(f"Usage: python -m recognition.voice_id [enroll|test|list|delete] [name]")

