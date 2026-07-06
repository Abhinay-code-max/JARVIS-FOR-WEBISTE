"""
Auto-patcher: 
1. Adds "vision_model": "qwen2.5vl:7b" to config/api_keys.json
2. Improves the fallback logic in screen_processor.py
"""
import json

CONFIG_PATH = r"C:\Users\Abhinay Kandrika\Desktop\JARVIS-XL\config\api_keys.json"
SCREEN_PROC = r"C:\Users\Abhinay Kandrika\Desktop\JARVIS-XL\actions\screen_processor.py"

# ── 1. Add vision_model to config ────────────────────────────────────────
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

if cfg.get("vision_model") != "qwen2.5vl:7b":
    cfg["vision_model"] = "qwen2.5vl:7b"
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)
    print("[1/2] Added 'vision_model': 'qwen2.5vl:7b' to config/api_keys.json")
else:
    print("[1/2] vision_model already set correctly — skipping.")

# ── 2. Fix the fallback in screen_processor.py ──────────────────────────
with open(SCREEN_PROC, "r", encoding="utf-8") as f:
    content = f.read()

old_line = 'vision_model = cfg.get("vision_model") or cfg.get("llm_model", "llava")'
new_line = 'vision_model = cfg.get("vision_model") or "qwen2.5vl:7b"'

if old_line in content:
    content = content.replace(old_line, new_line, 1)
    with open(SCREEN_PROC, "w", encoding="utf-8") as f:
        f.write(content)
    print("[2/2] Fixed fallback logic in screen_processor.py — now defaults to qwen2.5vl:7b instead of the text-only chat model.")
else:
    print("[2/2] WARNING: Could not find the exact fallback line — may already be patched or different.")

print("\n✅ Vision model fix applied!")
