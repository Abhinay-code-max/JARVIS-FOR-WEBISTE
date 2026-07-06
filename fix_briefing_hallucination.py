"""
Auto-patcher: makes daily_briefing speak its real result directly
and prevents the LLM from re-summarizing (and hallucinating placeholders).
"""

MAIN_PY = r"C:\Users\Abhinay Kandrika\Desktop\JARVIS-XL\main.py"

with open(MAIN_PY, "r", encoding="utf-8") as f:
    content = f.read()

changed = False

# Replace the daily_briefing execution branch to speak directly and return __SILENT__
old_block = '''            elif name == "daily_briefing":
                r = daily_briefing(parameters=args, player=self.ui, speak=self.speak)
                result = r or "Briefing delivered."'''

new_block = '''            elif name == "daily_briefing":
                r = daily_briefing(parameters=args, player=self.ui, speak=self.speak)
                briefing_text = r or "I don't have anything to brief you on right now."
                self.speak(briefing_text)
                self.ui.write_log(f"Jarvis: {briefing_text}")
                if not self.ui.muted:
                    self.ui.set_state("LISTENING")
                return "__SILENT__"'''

if old_block in content:
    content = content.replace(old_block, new_block, 1)
    changed = True
    print("Patched daily_briefing to speak directly (bypassing LLM hallucination).")
else:
    print("WARNING: Could not find the exact daily_briefing block. Checking for variations...")
    # Try a looser match
    import re
    pattern = re.compile(
        r'elif name == "daily_briefing":\s*\n\s*r = daily_briefing\(parameters=args, player=self\.ui, speak=self\.speak\)\s*\n\s*result = r or "Briefing delivered\."'
    )
    if pattern.search(content):
        content = pattern.sub(new_block.strip(), content)
        changed = True
        print("Patched using regex fallback.")
    else:
        print("ERROR: Could not locate daily_briefing block at all. Manual fix needed.")

if changed:
    with open(MAIN_PY, "w", encoding="utf-8") as f:
        f.write(content)
    print("\n✅ main.py patched successfully!")
else:
    print("\n⚠️ No changes made.")
