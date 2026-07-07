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
                 on_transcript=None):
        self._api_key       = api_key or _get_api_key()
        self._language      = language if language != "auto" else "en"
        self._on_transcript = on_transcript
        self._running       = False
        self._ws            = None

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
                    pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
                    try:
                        ws.send(pcm, opcode=websocket.ABNF.OPCODE_BINARY)
                    except Exception as e:
                        print(f"[DeepgramStreaming] send error: {e}")

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
