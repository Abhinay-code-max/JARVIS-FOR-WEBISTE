"""
Shows the actual Ollama API call section in screen_processor.py
so we can patch the model name and request format correctly.
"""

FPATH = r"C:\Users\Abhinay Kandrika\Desktop\JARVIS-XL\actions\screen_processor.py"

with open(FPATH, "r", encoding="utf-8") as f:
    content = f.read()

print(f"Total file length: {len(content)} characters\n")

# Find the requests.post call and show surrounding context
import re
idx = content.find("requests.post")
if idx == -1:
    idx = content.find("requests.request")
if idx == -1:
    print("Could not find requests.post call directly. Searching for 'api/chat' or 'api/generate'...")
    idx = content.find("api/chat")
    if idx == -1:
        idx = content.find("api/generate")

if idx != -1:
    start = max(0, idx - 800)
    end = min(len(content), idx + 1500)
    print("=== Context around API call ===")
    print(content[start:end])
else:
    print("Could not locate the API call section at all.")
    print("\nShowing full file instead:\n")
    print(content)
