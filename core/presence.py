"""
core/presence.py
================
Dual-Factor Biometric Presence Gate (Voice + Face) for sensitive actions.

When an action evaluated as ASK_AND_WAIT is confirmed by the user,
this presence gate verifies that the active human at the terminal is Abhinay
by requiring BOTH:
  1. Live Face verification: Captures webcam frames (~1.5s), extracts face encodings,
     and verifies face match confidence >= FACE_CONFIDENCE_THRESHOLD against Abhinay.npy.
  2. Live Voice verification: Prompts the user to speak a dedicated verification phrase (~2.5-3.0s),
     extracts the speaker embedding, and verifies cosine similarity >= VOICE_THRESHOLD against Abhinay.npy.

Fail-Closed:
  - If face check fails (mismatch, low confidence, no face detected, camera unavailable) -> REJECTED.
  - If voice check fails (mismatch, low confidence, silent/too short, mic error) -> REJECTED.
  - If neither matches -> REJECTED.
  - If no enrolled profile exists for Abhinay -> REJECTED.

Transparency:
  - Writes HUD log: "🔒 Verifying biometric presence (Voice + Face)..."
  - Logs results explicitly: "🔒 Biometric presence verified: Abhinay" or failure reason.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from config import BASE_DIR
from recognition.face_id import FaceIdentifier
from recognition.voice_id import VoiceIdentifier

_log = logging.getLogger("jarvis.presence")

TARGET_USER: str = "Abhinay"
VOICE_THRESHOLD: float = 0.75
FACE_CONFIDENCE_THRESHOLD: float = 0.45
VOICE_SAMPLE_DURATION: float = 3.0

FACES_DIR  = BASE_DIR / "recognition" / "faces"
VOICES_DIR = BASE_DIR / "recognition" / "voices"

# Feature toggle (enabled by default for interactive desktop execution)
_presence_check_enabled: bool = True

# Lazy-loaded singletons
_face_identifier: Optional[FaceIdentifier] = None
_voice_identifier: Optional[VoiceIdentifier] = None


def is_presence_check_enabled() -> bool:
    return _presence_check_enabled


def set_presence_check_enabled(enabled: bool) -> None:
    global _presence_check_enabled
    _presence_check_enabled = enabled


def get_face_identifier() -> FaceIdentifier:
    global _face_identifier
    if _face_identifier is None:
        _face_identifier = FaceIdentifier(FACES_DIR)
    return _face_identifier


def get_voice_identifier() -> VoiceIdentifier:
    global _voice_identifier
    if _voice_identifier is None:
        _voice_identifier = VoiceIdentifier(VOICES_DIR)
    return _voice_identifier


def verify_biometric_presence(
    player=None,
    speak=None,
    face_id: Optional[FaceIdentifier] = None,
    voice_id: Optional[VoiceIdentifier] = None,
    voice_audio: Optional[Any] = None,
    target_user: str = TARGET_USER,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Executes dual-factor presence verification (Face + Voice) for target_user.
    Returns (success: bool, reason: str, details: dict).
    Fail-closed: returns False if either or both biometrics fail to match.
    """
    if not is_presence_check_enabled():
        return True, "Presence checking disabled", {
            "face": {"name": target_user, "confidence": 1.0, "matched": True},
            "voice": {"name": target_user, "confidence": 1.0, "matched": True},
        }

    f_id = face_id or get_face_identifier()
    v_id = voice_id or get_voice_identifier()

    details: Dict[str, Any] = {
        "face": {"name": None, "confidence": 0.0, "matched": False},
        "voice": {"name": None, "confidence": 0.0, "matched": False},
    }

    if player is not None:
        player.write_log("🔒 Verifying biometric presence (Voice + Face)...")

    # ── 1. Face Verification ──────────────────────────────────────────────────
    try:
        face_name, face_conf = f_id.identify(timeout=2.5)
        details["face"]["name"] = face_name
        details["face"]["confidence"] = face_conf
        if face_name == target_user and face_conf >= FACE_CONFIDENCE_THRESHOLD:
            details["face"]["matched"] = True
            _log.info("Face presence match: %s (%.1f%%)", face_name, face_conf * 100)
        else:
            _log.warning(
                "Face presence mismatch: got %r (%.1f%%), expected %r",
                face_name, face_conf * 100, target_user,
            )
    except Exception as e:
        _log.error("Face verification encountered an error: %s", e)
        face_name, face_conf = None, 0.0

    if not details["face"]["matched"]:
        msg = f"Face not recognized (detected: {face_name or 'None'}, conf: {face_conf:.1%})"
        if player is not None:
            player.write_log(f"❌ Biometric presence failed: {msg}")
        return False, msg, details

    # ── 2. Voice Verification ─────────────────────────────────────────────────
    try:
        if voice_audio is not None:
            audio = voice_audio
        else:
            prompt_msg = "Please speak to verify voice presence (3s)..."
            if player is not None:
                player.write_log(f"🎙️ {prompt_msg}")
            if speak:
                speak("Please speak to verify your identity.")
            audio = v_id.record_sample(duration_sec=VOICE_SAMPLE_DURATION)

        if audio is None or len(audio) == 0:
            voice_name, voice_conf = None, 0.0
        else:
            voice_name, voice_conf = v_id.identify(audio)

        details["voice"]["name"] = voice_name
        details["voice"]["confidence"] = voice_conf

        if voice_name == target_user and voice_conf >= VOICE_THRESHOLD:
            details["voice"]["matched"] = True
            _log.info("Voice presence match: %s (%.1f%%)", voice_name, voice_conf * 100)
        else:
            _log.warning(
                "Voice presence mismatch: got %r (%.1f%%), expected %r",
                voice_name, voice_conf * 100, target_user,
            )
    except Exception as e:
        _log.error("Voice verification encountered an error: %s", e)
        voice_name, voice_conf = None, 0.0

    if not details["voice"]["matched"]:
        msg = f"Voice not recognized (detected: {voice_name or 'None'}, conf: {voice_conf:.1%})"
        if player is not None:
            player.write_log(f"❌ Biometric presence failed: {msg}")
        return False, msg, details

    # ── Both Matched ──────────────────────────────────────────────────────────
    success_msg = f"Abhinay verified (Face: {face_conf:.0%}, Voice: {voice_conf:.0%})"
    if player is not None:
        player.write_log(f"🔒 Biometric presence verified: {success_msg}")
    return True, success_msg, details
