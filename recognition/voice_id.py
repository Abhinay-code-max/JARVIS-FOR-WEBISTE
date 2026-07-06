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
from pathlib import Path
from typing import Optional

import numpy as np


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
                print(f"[VoiceID] Loaded embedding for '{name}'")
            except Exception as e:
                print(f"[VoiceID] Could not load {p}: {e}")

    # ── Embedding extraction ──────────────────────────────────────────────────
    def _get_embedding(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Extract a speaker embedding from float32 mono audio at 16 kHz."""
        duration = len(audio) / self.SAMPLE_RATE
        if duration < self.MIN_SECONDS:
            return None

        # Strategy 1: SpeechBrain ECAPA-TDNN (best quality, ~100 MB model)
        try:
            return self._embed_speechbrain(audio)
        except ImportError:
            pass
        except Exception as e:
            print(f"[VoiceID] SpeechBrain error: {e}")

        # Strategy 2: MFCC cosine (lightweight, always available with librosa)
        try:
            return self._embed_mfcc(audio)
        except ImportError:
            pass
        except Exception as e:
            print(f"[VoiceID] MFCC error: {e}")

        return None

    def _embed_speechbrain(self, audio: np.ndarray) -> np.ndarray:
        """ECAPA-TDNN via SpeechBrain — 192-dim x-vector."""
        import torch
        from speechbrain.pretrained import SpeakerRecognition
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
            print(f"[VoiceID] Audio too short to register '{name}'.")
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
        print(f"[VoiceID] Registered '{name}'.")
        return True

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
