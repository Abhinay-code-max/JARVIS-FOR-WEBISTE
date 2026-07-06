"""
Shows the exact sentence-chunking section so we can patch it precisely.
"""

LLM_CLIENT_PY = r"C:\Users\Abhinay Kandrika\Desktop\JARVIS-XL\core\llm_client.py"

with open(LLM_CLIENT_PY, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find the line with the regex match pattern and print surrounding context
for i, line in enumerate(lines):
    if "sentence boundary" in line.lower() or "[.!?]" in line:
        start = max(0, i - 3)
        end = min(len(lines), i + 25)
        print(f"--- Context around line {i+1} ---")
        for j in range(start, end):
            print(f"{j+1:4d}: {lines[j]}", end="")
        print("\n" + "="*60 + "\n")
        break
