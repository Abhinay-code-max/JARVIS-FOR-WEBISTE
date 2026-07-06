"""
Auto-patcher: speeds up TTS by reducing the minimum chunk size before speaking,
so JARVIS starts talking sooner instead of waiting for full sentences.
"""

LLM_CLIENT_PY = r"C:\Users\Abhinay Kandrika\Desktop\JARVIS-XL\core\llm_client.py"

with open(LLM_CLIENT_PY, "r", encoding="utf-8") as f:
    content = f.read()

print("Searching for sentence-splitting logic in llm_client.py...")
print("File length:", len(content), "characters")

# Look for common sentence-boundary patterns
import re
matches = re.findall(r'.{0,80}(sentence|\\.|\\!|\\?).{0,80}', content)
print(f"\nFound {len(matches)} potential matches for sentence splitting logic.")
print("\nShowing lines containing 'sentence':")
for line in content.split("\n"):
    if "sentence" in line.lower():
        print(f"  {line.strip()}")
