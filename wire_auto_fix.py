"""
Auto-patcher: adds the auto_fix_code tool to main.py.
"""

MAIN_PY = r"C:\Users\Abhinay Kandrika\Desktop\JARVIS-XL\main.py"

with open(MAIN_PY, "r", encoding="utf-8") as f:
    content = f.read()

changed = False

# ── 1. Add import ─────────────────────────────────────────────────────────
old_import = "from actions.daily_briefing    import daily_briefing"
new_import = "from actions.daily_briefing    import daily_briefing\nfrom actions.auto_fix_code     import auto_fix_code"

if "from actions.auto_fix_code" not in content:
    if old_import in content:
        content = content.replace(old_import, new_import, 1)
        changed = True
        print("[1/3] Import added.")
    else:
        print("[1/3] WARNING: Could not find daily_briefing import — trying weather_report instead.")
        old_import2 = "from actions.weather_report    import weather_action"
        new_import2 = "from actions.weather_report    import weather_action\nfrom actions.auto_fix_code     import auto_fix_code"
        if old_import2 in content:
            content = content.replace(old_import2, new_import2, 1)
            changed = True
            print("[1/3] Import added (fallback location).")
        else:
            print("[1/3] ERROR: Could not add import anywhere.")
else:
    print("[1/3] Import already present.")

# ── 2. Add tool declaration ──────────────────────────────────────────────
marker = '''    {
        "name": "save_memory",'''

tool_block = '''    {
        "name": "auto_fix_code",
        "description": (
            "Runs a Python script, captures the REAL error if it crashes, "
            "and automatically fixes the bug by editing the file directly. "
            "Call this when the user says 'fix this code', 'fix the error', "
            "'run and fix', 'debug this file', or wants JARVIS to directly "
            "correct a script rather than just explain the error."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "file_path": {"type": "STRING", "description": "Full path to the Python file to run and fix"},
                "max_attempts": {"type": "INTEGER", "description": "Max fix attempts, default 2"}
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "save_memory",'''

if '"name": "auto_fix_code"' not in content:
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

exec_block = '''            elif name == "auto_fix_code":
                r = auto_fix_code(parameters=args, player=self.ui, speak=self.speak)
                result = r or "Done attempting to fix the code."

            elif name == "shutdown_jarvis":'''

if 'name == "auto_fix_code"' not in content or content.count('name == "auto_fix_code"') == 0:
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
    print("\n✅ auto_fix_code tool wired into main.py successfully!")
else:
    print("\n⚠️ No changes made — check warnings above.")
