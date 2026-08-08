"""
core/stt_deepgram.py
====================
Deepgram Nova-2 STT — real-time streaming for JARVIS-XL.
Fixed to match the verified working pattern (ABNF.OPCODE_BINARY).
"""

from __future__ import annotations
import io
import json
import threading
import time

import numpy as np
import sounddevice as sd

from config import load_config

SAMPLE_RATE = 16_000
CHANNELS    = 1
BLOCK_SIZE  = 4_000


def _get_api_key() -> str:
    return load_config().get("deepgram_api_key", "")


class DeepgramSTT:
    """Batch transcription fallback — not used in normal operation."""

    def __init__(self, api_key: str = "", language: str = "en"):
        self._api_key  = api_key or _get_api_key()
        self._language = language if language != "auto" else "en"
        if not self._api_key:
            raise ValueError("Deepgram API key not found in config/api_keys.json")
        print(f"[DeepgramSTT] Ready. Language={self._language}")

    def transcribe(self, audio: np.ndarray) -> str:
        return ""  # Not used — streaming mode handles everything


class DeepgramStreamingSTT:
    """
    Live streaming STT using Deepgram WebSocket.
    Uses the verified working pattern: websocket-client with ABNF.OPCODE_BINARY.
    """

    def __init__(self, api_key: str = "", language: str = "en",
                 on_transcript=None,
                 is_speaking=None, is_muted=None,
                 on_barge_in=None, on_silence_nudge=None,
                 speech_thresh: float = 0.008,
                 barge_in_speech_sec: float = 0.18,
                 silence_nudge_sec: float = 2.0):
        self._api_key       = api_key or _get_api_key()
        self._language      = language if language != "auto" else "en"
        self._on_transcript = on_transcript
        self._running       = False
        self._ws            = None

        # Deepgram decides utterance boundaries server-side (see the
        # `endpointing` param below) — there's no local VAD buffer to feed
        # like whisper/vosk have. Barge-in and the "still listening" nudge
        # are instead driven off a small local RMS watch on the same raw
        # audio that's already being streamed to the WebSocket, so they
        # don't depend on a network round-trip.
        self._is_speaking     = is_speaking      # () -> bool
        self._is_muted        = is_muted         # () -> bool
        self._on_barge_in     = on_barge_in      # () -> None
        self._on_silence_nudge = on_silence_nudge  # () -> None
        self._speech_thresh   = speech_thresh
        self._barge_in_n      = int(barge_in_speech_sec * SAMPLE_RATE)
        self._nudge_n         = int(silence_nudge_sec * SAMPLE_RATE)
        self._barge_hot_n     = 0
        self._in_speech       = False
        self._silence_n       = 0
        self._nudge_shown     = False

        if not self._api_key:
            raise ValueError("Deepgram API key not found.")

    def start(self):
        self._running = True
        threading.Thread(target=self._stream_loop, daemon=True).start()

    def stop(self):
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def _stream_loop(self):
        while self._running:
            try:
                self._connect_and_stream()
            except Exception as e:
                print(f"[DeepgramStreaming] Connection error: {e}")
            if self._running:
                print("[DeepgramStreaming] Reconnecting in 2s...")
                time.sleep(2)

    def _connect_and_stream(self):
        import websocket

        url = (
            f"wss://api.deepgram.com/v1/listen"
            f"?model=nova-2"
            f"&language={self._language}"
            f"&encoding=linear16"
            f"&sample_rate={SAMPLE_RATE}"
            f"&channels=1"
            f"&punctuate=true"
            f"&smart_format=true"
            f"&interim_results=true"
            # Wait ~5s of silence before finalizing a segment (was left at
            # Deepgram's short default), matching the local VAD's
            # silence_sec=5.0 so a normal mid-sentence pause doesn't get
            # read as the end of the utterance.
            f"&endpointing=5000"
        )
        headers = [f"Authorization: Token {self._api_key}"]

        conn_open = threading.Event()

        def on_message(ws, message):
            try:
                data = json.loads(message)
                transcript = (
                    data.get("channel", {})
                    .get("alternatives", [{}])[0]
                    .get("transcript", "")
                )
                is_final = data.get("is_final", False)
                if is_final:
                    # Utterance boundary from Deepgram's perspective — reset
                    # the local nudge watch so it doesn't carry over into
                    # the next utterance.
                    self._in_speech   = False
                    self._silence_n   = 0
                    self._nudge_shown = False
                if transcript and is_final:
                    print(f"[DeepgramStreaming] FINAL: {transcript}")
                    if self._on_transcript:
                        self._on_transcript(transcript)
            except Exception as e:
                print(f"[DeepgramStreaming] parse error: {e}")

        def on_error(ws, error):
            print(f"[DeepgramStreaming] WS error: {error}")

        def on_close(ws, code, msg):
            print(f"[DeepgramStreaming] Closed. code={code} msg={msg}")
            conn_open.clear()

        def on_open(ws):
            print("[DeepgramStreaming] Connected to Deepgram Nova-2.")
            conn_open.set()

            def audio_thread():
                def callback(indata, frames, time_info, status):
                    if not self._running:
                        return
                    audio = indata[:, 0]
                    pcm = (audio * 32767).astype(np.int16).tobytes()
                    try:
                        ws.send(pcm, opcode=websocket.ABNF.OPCODE_BINARY)
                    except Exception as e:
                        print(f"[DeepgramStreaming] send error: {e}")

                    if self._is_muted and self._is_muted():
                        return

                    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))

                    if self._is_speaking and self._is_speaking():
                        # JARVIS is talking — watch for sustained speech-level
                        # RMS (not a single loud chunk) to confirm a real
                        # interruption before cutting playback.
                        self._in_speech   = False
                        self._silence_n   = 0
                        self._nudge_shown = False
                        if rms > self._speech_thresh:
                            self._barge_hot_n += len(audio)
                            if self._barge_hot_n >= self._barge_in_n:
                                self._barge_hot_n = 0
                                if self._on_barge_in:
                                    self._on_barge_in()
                        else:
                            self._barge_hot_n = 0
                        return

                    self._barge_hot_n = 0

                    # UI-only "still listening" nudge partway through a
                    # mid-utterance pause. Deepgram's own endpointing (set
                    # to 5s above) still decides the real utterance
                    # boundary — this is purely a local cue for the UI.
                    if rms > self._speech_thresh:
                        self._in_speech   = True
                        self._silence_n   = 0
                        self._nudge_shown = False
                    elif self._in_speech:
                        self._silence_n += len(audio)
                        if (self._silence_n >= self._nudge_n
                                and not self._nudge_shown
                                and self._on_silence_nudge):
                            self._nudge_shown = True
                            self._on_silence_nudge()

                try:
                    with sd.InputStream(
                        samplerate=SAMPLE_RATE,
                        channels=CHANNELS,
                        blocksize=BLOCK_SIZE,
                        dtype="float32",
                        callback=callback,
                    ):
                        while self._running and conn_open.is_set():
                            time.sleep(0.1)
                except Exception as e:
                    print(f"[DeepgramStreaming] mic stream error: {e}")

                try:
                    ws.send(json.dumps({"type": "CloseStream"}))
                except Exception:
                    pass

            threading.Thread(target=audio_thread, daemon=True).start()

        self._ws = websocket.WebSocketApp(
            url,
            header=headers,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        self._ws.run_forever()
