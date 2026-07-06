"""
Standalone Deepgram connection test.
Run this directly to verify your API key and mic work with Deepgram,
independent of JARVIS.
"""
import json
import time
import sys

import numpy as np
import sounddevice as sd

API_KEY = input("Paste your Deepgram API key: ").strip()

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 4000

def main():
    try:
        import websocket
    except ImportError:
        print("Installing websocket-client...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "websocket-client"])
        import websocket

    url = (
        f"wss://api.deepgram.com/v1/listen"
        f"?model=nova-2"
        f"&language=en"
        f"&encoding=linear16"
        f"&sample_rate={SAMPLE_RATE}"
        f"&channels=1"
        f"&punctuate=true"
        f"&smart_format=true"
        f"&interim_results=true"
    )
    headers = [f"Authorization: Token {API_KEY}"]

    running = [True]

    def on_message(ws, message):
        data = json.loads(message)
        try:
            transcript = data["channel"]["alternatives"][0]["transcript"]
            is_final = data.get("is_final", False)
            if transcript:
                tag = "FINAL" if is_final else "interim"
                print(f"[{tag}] {transcript}")
        except Exception as e:
            print(f"[parse error] {e} | raw: {data}")

    def on_error(ws, error):
        print(f"[ERROR] {error}")

    def on_close(ws, code, msg):
        print(f"[CLOSED] code={code} msg={msg}")
        running[0] = False

    def on_open(ws):
        print("[CONNECTED] Speak now! Listening for 20 seconds...")

        def audio_thread():
            def callback(indata, frames, time_info, status):
                if status:
                    print(f"[mic status] {status}")
                pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
                try:
                    ws.send(pcm, opcode=websocket.ABNF.OPCODE_BINARY)
                except Exception as e:
                    print(f"[send error] {e}")

            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                blocksize=BLOCK_SIZE,
                dtype="float32",
                callback=callback,
            ):
                start = time.time()
                while running[0] and time.time() - start < 20:
                    time.sleep(0.1)
            try:
                ws.send(json.dumps({"type": "CloseStream"}))
            except Exception:
                pass
            ws.close()

        import threading
        threading.Thread(target=audio_thread, daemon=True).start()

    print("Checking microphone devices...")
    print(sd.query_devices())
    print(f"\nDefault input device: {sd.default.device}")

    ws = websocket.WebSocketApp(
        url,
        header=headers,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.run_forever()

if __name__ == "__main__":
    main()
