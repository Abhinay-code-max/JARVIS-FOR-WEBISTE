import sqlite3
import datetime

conn = sqlite3.connect('data/jarvis.db')
conn.row_factory = sqlite3.Row

print("=== ALL RECENT LOG EVENTS ===")
rows = conn.execute("SELECT * FROM log_events ORDER BY rowid DESC LIMIT 30").fetchall()
for r in reversed(rows):
    dt = datetime.datetime.fromtimestamp(r['created_at']).strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{dt}] [{r['source']}] {r['message']} | detail={r['detail']} | dur={r['duration_ms']}ms")

print("\n=== ALL RECENT TASK EVENTS ===")
rows = conn.execute("SELECT * FROM task_events ORDER BY rowid DESC LIMIT 30").fetchall()
for r in reversed(rows):
    dt = datetime.datetime.fromtimestamp(r['created_at']).strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{dt}] [{r['tool']}] {r['description']} | status={r['status']} | detail={r['detail']} | dur={r['duration_ms']}ms")
