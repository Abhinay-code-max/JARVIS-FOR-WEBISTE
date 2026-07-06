"""
Auto-patcher: improves the vision prompt in vision_fix_code.py to be more
thorough and accurate at spotting the REAL bug (especially off-by-one errors,
index errors, and loop boundary issues) rather than guessing the first
suspicious-looking thing.
"""

FPATH = r"C:\Users\Abhinay Kandrika\Desktop\JARVIS-XL\actions\vision_fix_code.py"

with open(FPATH, "r", encoding="utf-8") as f:
    content = f.read()

old_prompt = '''    vision_prompt = (
        "Look at this screenshot of code (in an editor or terminal). "
        "Identify: 1) the filename if visible, 2) the exact bug/error if any. "
        "Respond in this exact format:\\n"
        "FILENAME: <filename or 'unknown'>\\n"
        "BUG: <one clear sentence describing the exact bug, or 'none found'>"
    )'''

new_prompt = '''    vision_prompt = (
        "Look at this screenshot of code in an editor. Carefully read EVERY line, "
        "top to bottom, before answering. Pay special attention to: "
        "loop ranges and boundaries (off-by-one errors like range(len(x)+1)), "
        "list/array indexing that might go out of bounds, "
        "uninitialized or undefined variables, "
        "incorrect comparisons or logic. "
        "Trace through the code mentally with the sample data shown (if visible) "
        "to find which line would actually crash or misbehave first. "
        "Identify: 1) the exact filename shown in the tab or breadcrumb, "
        "2) the SINGLE most likely bug that would cause a runtime error, "
        "including the specific line number. "
        "Respond in this exact format:\\n"
        "FILENAME: <filename or 'unknown'>\\n"
        "LINE: <line number or 'unknown'>\\n"
        "BUG: <one precise sentence describing the exact bug and why it fails, or 'none found'>"
    )'''

if old_prompt in content:
    content = content.replace(old_prompt, new_prompt, 1)
    with open(FPATH, "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ Vision prompt upgraded for more accurate bug detection!")
else:
    print("⚠️ Could not find the exact old prompt block — may already be modified.")
