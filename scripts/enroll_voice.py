#!/usr/bin/env python3
"""
scripts/enroll_voice.py
=======================
Interactive Voice Profile Enrollment Tool for JARVIS-XL.

This script records multiple voice samples from your real microphone,
computes your voice embedding, saves it to recognition/voices/<name>.npy,
and verifies speaker identification.

Usage:
    python scripts/enroll_voice.py [name]
    python scripts/enroll_voice.py Abhinay
"""

import sys
import time
from pathlib import Path

# Set up project root in path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from recognition.voice_id import VoiceIdentifier

VOICES_DIR = ROOT_DIR / "recognition" / "voices"

def main():
    target_name = sys.argv[1] if len(sys.argv) > 1 else "Abhinay"
    
    print("\n" + "=" * 60)
    print(f"      JARVIS-XL VOICE PROFILE ENROLLMENT")
    print("=" * 60)
    print(f"Target User Profile : {target_name}")
    print(f"Destination Path    : {VOICES_DIR / f'{target_name}.npy'}")
    print("=" * 60)
    print("\nInstructions:")
    print(" 1. Ensure your microphone is connected and background noise is minimal.")
    print(" 2. You will be prompted to speak 3 different sentences (~3.5 seconds each).")
    print(" 3. Speak naturally in your normal speaking volume and tone.")
    print("=" * 60)
    
    input("\nPress [ENTER] when you are ready to begin enrollment...")
    
    identifier = VoiceIdentifier(VOICES_DIR)
    
    def prompt_callback(msg: str):
        print(f"\n🎙️  {msg}")
    
    print("\n--- Starting Audio Capture ---")
    
    success = identifier.record_and_register(
        name=target_name,
        reps=3,
        duration_sec=3.5,
        prompt_cb=prompt_callback,
    )
    
    if not success:
        print("\n❌ Enrollment failed. Please check your microphone permissions and try again.")
        sys.exit(1)
        
    print("\n" + "=" * 60)
    print(f"✅ SUCCESS: Voice profile for '{target_name}' is enrolled and saved!")
    print(f"Profile path: {VOICES_DIR / f'{target_name}.npy'}")
    print("=" * 60)
    
    print("\n--- Verification Test ---")
    print("Let's verify that JARVIS identifies you correctly.")
    print("When prompted, speak any sentence for 3.5 seconds (e.g. 'Hey JARVIS, do you recognize me?').")
    input("\nPress [ENTER] to start verification test...")
    
    name, confidence = identifier.record_and_identify(duration_sec=3.5, prompt_cb=prompt_callback)
    
    print("\n" + "-" * 40)
    if name == target_name:
        print(f"🎉 VERIFIED: Identified as '{name}' with {confidence:.1%} confidence!")
        print(f"JARVIS Voice ID is now fully active for {target_name}.")
    elif name:
        print(f"⚠️ Identified as '{name}' with {confidence:.1%} confidence.")
    else:
        print(f"⚠️ Confidence score: {confidence:.1%} (Threshold is {identifier.THRESHOLD:.0%}).")
        print("You can run this enrollment script again at any time to re-calibrate.")
    print("-" * 40 + "\n")

if __name__ == "__main__":
    main()
