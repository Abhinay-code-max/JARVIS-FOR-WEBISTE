"""
Diagnostic: tests the file search logic directly to see why it's failing
to find bugg.py on the Desktop.
"""
from pathlib import Path

filename = "bugg.py"
search_roots = [
    str(Path.home() / "Desktop"),
    str(Path.home() / "Documents"),
    str(Path.home() / "Downloads"),
]

print(f"Looking for: {filename}")
print(f"Home directory: {Path.home()}")
print()

for root in search_roots:
    root_path = Path(root)
    print(f"Checking root: {root_path}")
    print(f"  Exists: {root_path.exists()}")
    if root_path.exists():
        candidate = root_path / filename
        print(f"  Direct match path: {candidate}")
        print(f"  Direct match exists: {candidate.exists()}")

        # List what's actually in Desktop
        if "Desktop" in str(root_path):
            print(f"  Files in Desktop:")
            try:
                for item in root_path.iterdir():
                    if item.suffix == ".py":
                        print(f"    - {item.name}")
            except Exception as e:
                print(f"    Error listing: {e}")
    print()
