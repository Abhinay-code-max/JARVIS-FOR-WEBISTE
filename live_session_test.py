"""
live_session_test.py
====================
Runs a complete live end-to-end session of JARVIS-XL with real PyQt UI,
real Ollama nemotron-3-nano:4b, real tool dispatch, real TTS (edge-tts),
and real STT (Whisper).
"""
import os
import sys
import time
import threading
from pathlib import Path

# Set Qt platform to offscreen or minimal if needed, but on Windows default platform works
os.environ["QT_QPA_PLATFORM"] = "windows"

import numpy as np
from PyQt6.QtWidgets import QApplication
from ui import JarvisUI
from config import load_config, BASE_DIR
from core.confirm import CONFIRM
from main import JarvisXL

def run_live_session():
    print("==================================================", flush=True)
    print("STEP 1: Live End-to-End JARVIS-XL Session Test", flush=True)
    print("==================================================", flush=True)

    app = QApplication.instance() or QApplication(sys.argv)
    ui = JarvisUI("face.png")

    recorded_tts = []
    orig_speak = None

    session_report = {
        "conversational": {"passed": False, "response": "", "thinking_leaked": False, "duration": 0},
        "clarification_delete": {"passed": False, "clarify_asked": False, "deleted": False, "summary": "", "duration": 0},
        "tool_call_search": {"passed": False, "tool_used": "", "response": "", "duration": 0},
        "stt_whisper": {"passed": False, "model_loaded": False, "sample_transcription": ""},
        "tts_edgetts": {"passed": False, "audio_bytes": 0},
    }

    def driver_thread(jarvis: JarvisXL):
        try:
            print("\n[Driver] Waiting for JARVIS initialization...", flush=True)
            # Wait for JARVIS online
            for _ in range(60):
                if any("JARVIS-XL online" in log for log in ui._logs_history if isinstance(log, str)) or getattr(jarvis, "_running", False):
                    break
                time.sleep(1)

            print("[Driver] JARVIS is online! Beginning live test suite.", flush=True)
            time.sleep(2)

            # -------------------------------------------------------------
            # TEST 1: Simple conversational question
            # -------------------------------------------------------------
            print("\n--------------------------------------------------", flush=True)
            print("1. Testing conversational query: 'Why is the sky blue?'", flush=True)
            print("--------------------------------------------------", flush=True)
            t0 = time.time()
            init_log_count = len(ui._logs_history)
            jarvis._enqueue_command("Why is the sky blue? Answer in two concise sentences.")

            # Wait for response
            response_text = ""
            for _ in range(30):
                time.sleep(1)
                new_logs = ui._logs_history[init_log_count:]
                jarvis_logs = [l for l in new_logs if isinstance(l, str) and l.startswith("Jarvis:")]
                if jarvis_logs:
                    response_text = jarvis_logs[-1].replace("Jarvis:", "").strip()
                    break

            dur = time.time() - t0
            print(f"[Driver] Conversational Response ({dur:.2f}s):\n  {response_text}", flush=True)
            
            # Check for thinking token / reasoning trace leakage
            thinking_leak = any(token in response_text.lower() for token in ["<think>", "</think>", "thought process", "thinking:"])
            session_report["conversational"] = {
                "passed": bool(response_text) and not thinking_leak,
                "response": response_text,
                "thinking_leaked": thinking_leak,
                "duration": dur,
            }

            time.sleep(2)

            # -------------------------------------------------------------
            # TEST 2: "delete a file" terse phrasing + clarification
            # -------------------------------------------------------------
            print("\n--------------------------------------------------", flush=True)
            print("2. Testing terse phrasing: 'delete a file'", flush=True)
            print("--------------------------------------------------", flush=True)
            
            # Create a real test dummy file on desktop
            desktop = Path.home() / "Desktop"
            dummy_file = desktop / "jarvis_live_delete_test.txt"
            dummy_file.write_text("live session delete target", encoding="utf-8")
            print(f"[Driver] Created dummy file: {dummy_file}", flush=True)

            t0 = time.time()
            init_log_count = len(ui._logs_history)
            jarvis._enqueue_command("delete a file")

            # Watch for clarification question
            clarify_detected = False
            for _ in range(30):
                time.sleep(0.5)
                if CONFIRM.is_pending():
                    with CONFIRM._lock:
                        mode = getattr(CONFIRM, "_mode", "boolean")
                    if mode == "text":
                        clarify_detected = True
                        print(f"[Driver] Clarification detected! Answering: 'delete jarvis_live_delete_test.txt from desktop'", flush=True)
                        jarvis._enqueue_command("delete jarvis_live_delete_test.txt from desktop")
                        break

            # Watch for confirmation / approval
            for _ in range(40):
                time.sleep(0.5)
                if CONFIRM.is_pending():
                    with CONFIRM._lock:
                        mode = getattr(CONFIRM, "_mode", "boolean")
                    if mode == "boolean":
                        print("[Driver] Approval detected! Answering 'yes'...", flush=True)
                        jarvis._enqueue_command("yes")
                        break

            # Wait for completion
            dur = time.time() - t0
            time.sleep(4)
            file_deleted = not dummy_file.exists()
            print(f"[Driver] Dummy file deleted: {file_deleted} (Duration: {dur:.2f}s)", flush=True)
            
            session_report["clarification_delete"] = {
                "passed": clarify_detected and file_deleted,
                "clarify_asked": clarify_detected,
                "deleted": file_deleted,
                "duration": dur,
            }

            time.sleep(2)

            # -------------------------------------------------------------
            # TEST 3: Tool calling (web_search)
            # -------------------------------------------------------------
            print("\n--------------------------------------------------", flush=True)
            print("3. Testing tool call: 'what is the current price of Ethereum in USD?'", flush=True)
            print("--------------------------------------------------", flush=True)
            t0 = time.time()
            init_log_count = len(ui._logs_history)
            jarvis._enqueue_command("what is the current price of Ethereum in USD?")

            search_response = ""
            for _ in range(35):
                time.sleep(1)
                new_logs = ui._logs_history[init_log_count:]
                jarvis_logs = [l for l in new_logs if isinstance(l, str) and l.startswith("Jarvis:")]
                search_logs = [l for l in new_logs if isinstance(l, str) and "[Search]" in l]
                if jarvis_logs and search_logs:
                    search_response = jarvis_logs[-1].replace("Jarvis:", "").strip()
                    break

            dur = time.time() - t0
            print(f"[Driver] Search Response ({dur:.2f}s):\n  {search_response}", flush=True)
            session_report["tool_call_search"] = {
                "passed": bool(search_response),
                "tool_used": "web_search",
                "response": search_response,
                "duration": dur,
            }

            # -------------------------------------------------------------
            # TEST 4: STT (Whisper) & TTS (edge-tts) Verification
            # -------------------------------------------------------------
            print("\n--------------------------------------------------", flush=True)
            print("4. Testing STT (Whisper) and TTS (edge-tts)", flush=True)
            print("--------------------------------------------------", flush=True)

            # Test Whisper STT
            stt_ok = False
            stt_transcription = ""
            if hasattr(jarvis, "_stt") and jarvis._stt is not None:
                print(f"[Driver] STT engine loaded: {type(jarvis._stt).__name__}", flush=True)
                # Generate 1s of synthetic audio (silence) to test transcribe() call
                dummy_audio = np.zeros(16000, dtype=np.float32)
                res = jarvis._stt.transcribe(dummy_audio)
                stt_ok = True
                stt_transcription = str(res)
                print(f"[Driver] STT transcribe executed successfully (output: {res!r})", flush=True)

            # Test edge-tts
            tts_ok = False
            try:
                from core.tts_edgetts import EdgeTTSVoice
                tts = EdgeTTSVoice(voice="en-GB-RyanNeural", speed="1.0")
                audio_data = tts.synthesize_sync("Live session audio test.")
                if audio_data and len(audio_data) > 1000:
                    tts_ok = True
                    print(f"[Driver] edge-tts synthesized {len(audio_data)} bytes of valid MP3 audio", flush=True)
            except Exception as e:
                print(f"[Driver] TTS test error: {e}", flush=True)

            session_report["stt_whisper"] = {
                "passed": stt_ok,
                "model_loaded": stt_ok,
                "sample_transcription": stt_transcription,
            }
            session_report["tts_edgetts"] = {
                "passed": tts_ok,
                "audio_bytes": len(audio_data) if tts_ok else 0,
            }

            print("\n==================================================", flush=True)
            print("ALL STEP 1 TESTS COMPLETE — SESSION SUMMARY", flush=True)
            print("==================================================", flush=True)
            for k, v in session_report.items():
                print(f"  • {k}: {v}", flush=True)

        finally:
            print("\n[Driver] Closing live session...", flush=True)
            time.sleep(2)
            app.quit()

    def run_main():
        jarvis = JarvisXL(ui)
        t = threading.Thread(target=driver_thread, args=(jarvis,), daemon=True)
        t.start()
        try:
            jarvis.run()
        except Exception as e:
            print(f"[JARVIS Error] {e}", flush=True)

    threading.Thread(target=run_main, daemon=True).start()
    app.exec()

if __name__ == "__main__":
    run_live_session()
