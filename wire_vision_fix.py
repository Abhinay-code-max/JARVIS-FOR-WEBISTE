"""
Auto-patcher: adds the vision_fix_code tool to main.py.
Simple, automatic: 'fix this code' -> looks at screen, finds file, fixes it.
"""

MAIN_PY = r"C:\Users\Abhinay Kandrika\Desktop\JARVIS-XL\main.py"

with open(MAIN_PY, "r", encoding="utf-8") as f:
    content = f.read()

changed = False

# ── 1. Add import ─────────────────────────────────────────────────────────
candidates = [
    "from actions.auto_fix_code     import auto_fix_code",
    "from actions.daily_briefing    import daily_briefing",
    "from actions.weather_report    import weather_action",
]

if "from actions.vision_fix_code" not in content:
    for old_import in candidates:
        if old_import in content:
            new_import = old_import + "\nfrom actions.vision_fix_code   import vision_fix_code"
            content = content.replace(old_import, new_import, 1)
            changed = True
            print(f"[1/3] Import added after: {old_import.strip()}")
            break
    else:
        print("[1/3] ERROR: Could not find any anchor import line.")
else:
    print("[1/3] Import already present.")

# ── 2. Add tool declaration ──────────────────────────────────────────────
marker = '''    {
        "name": "save_memory",'''

tool_block = '''    {
        "name": "vision_fix_code",
        "description": (
            "Looks at whatever code is currently visible on the screen (in an editor "
            "or terminal), automatically identifies the file and the bug, and directly "
            "fixes the file on disk. Call this when the user says things like "
            "'fix this', 'fix this code', 'fix the error', 'correct it', 'fix it for me', "
            "without needing them to specify a file path — you figure it out from the screen."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {}
        }
    },
    {
        "name": "save_memory",'''

if '"name": "vision_fix_code"' not in content:
    if marker in content:
        content = content.replace(marker, tool_block, 1)
        changed = True
        print("[2/3] Tool declaration added.")
    else:
        print("[2/3] WARNING: Could not find save_memory tool block.")
else:
    print("[2/3] Tool declaration already present.")

# ── 3. Add execution branch ──────────────────────────────────────────────
exec_marker = '''            elif name == "shutdown_jarvis":'''

exec_block = '''            elif name == "vision_fix_code":
                r = vision_fix_code(parameters=args, player=self.ui, speak=self.speak)
                result = r or "Done looking at the code."

            elif name == "shutdown_jarvis":'''

if content.count('name == "vision_fix_code"') == 0:
    if exec_marker in content:
        content = content.replace(exec_marker, exec_block, 1)
        changed = True
        print("[3/3] Execution branch added.")
    else:
        print("[3/3] WARNING: Could not find shutdown_jarvis execution branch.")
else:
    print("[3/3] Execution branch already present.")

if changed:
    with open(MAIN_PY, "w", encoding="utf-8") as f:
        f.write(content)
    print("\n✅ vision_fix_code tool wired into main.py successfully!")
else:
    print("\n⚠️ No changes made — check warnings above.")
