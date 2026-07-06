"""
Auto-patcher: adds the daily_briefing tool to main.py automatically.
Run this once: python apply_briefing_patch.py
"""
import re

MAIN_PY = r"C:\Users\Abhinay Kandrika\Desktop\JARVIS-XL\main.py"

with open(MAIN_PY, "r", encoding="utf-8") as f:
    content = f.read()

changed = False

# ── 1. Add import ─────────────────────────────────────────────────────────
import_line = "from actions.weather_report    import weather_action\n"
new_import  = "from actions.weather_report    import weather_action\nfrom actions.daily_briefing    import daily_briefing\n"

if "from actions.daily_briefing" not in content:
    if import_line in content:
        content = content.replace(import_line, new_import, 1)
        changed = True
        print("[1/3] Import added.")
    else:
        print("[1/3] WARNING: Could not find weather_report import line — skipping.")
else:
    print("[1/3] Import already present — skipping.")

# ── 2. Add tool declaration ──────────────────────────────────────────────
weather_tool_block = '''    {
        "name": "weather_report",
        "description": "Gets weather for any city.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"city": {"type": "STRING"}},
            "required": ["city"]
        }
    },'''

briefing_tool_block = '''    {
        "name": "weather_report",
        "description": "Gets weather for any city.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"city": {"type": "STRING"}},
            "required": ["city"]
        }
    },
    {
        "name": "daily_briefing",
        "description": (
            "Delivers a complete daily briefing: greeting, weather, today's reminders, "
            "and top news headlines. Call this when the user says 'good morning', "
            "'give me my briefing', 'what's happening today', 'catch me up', or similar."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "city": {"type": "STRING", "description": "Optional city override for weather"}
            },
            "required": []
        }
    },'''

if '"name": "daily_briefing"' not in content:
    if weather_tool_block in content:
        content = content.replace(weather_tool_block, briefing_tool_block, 1)
        changed = True
        print("[2/3] Tool declaration added.")
    else:
        print("[2/3] WARNING: Could not find weather_report tool block — skipping.")
else:
    print("[2/3] Tool declaration already present — skipping.")

# ── 3. Add execution branch ──────────────────────────────────────────────
weather_exec_block = '''            elif name == "weather_report":
                r = weather_action(parameters=args, player=self.ui)
                result = r or "Weather delivered."'''

briefing_exec_block = '''            elif name == "weather_report":
                r = weather_action(parameters=args, player=self.ui)
                result = r or "Weather delivered."

            elif name == "daily_briefing":
                r = daily_briefing(parameters=args, player=self.ui, speak=self.speak)
                result = r or "Briefing delivered."'''

if 'name == "daily_briefing"' not in content or content.count('name == "daily_briefing"') == 0:
    if weather_exec_block in content:
        content = content.replace(weather_exec_block, briefing_exec_block, 1)
        changed = True
        print("[3/3] Execution branch added.")
    else:
        print("[3/3] WARNING: Could not find weather_report execution block — skipping.")
else:
    print("[3/3] Execution branch already present — skipping.")

# ── Save ──────────────────────────────────────────────────────────────────
if changed:
    with open(MAIN_PY, "w", encoding="utf-8") as f:
        f.write(content)
    print("\n✅ main.py successfully patched with daily_briefing tool!")
else:
    print("\n⚠️ No changes were made. Check the warnings above.")
