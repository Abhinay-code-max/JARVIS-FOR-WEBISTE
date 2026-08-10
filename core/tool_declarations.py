"""
core/tool_declarations.py
===========================
Canonical tool-schema source of truth — moved out of main.py so it can be
shared without importing main.py itself (main.py is the entry-point
script: importing it runs logging setup, dependency bootstrap, and pulls
in heavy GUI/audio deps like PyQt6/sounddevice as a side effect, which a
lightweight core module must not trigger).

Two consumers:
  - main.py: OLLAMA_TOOLS is the tool-calling schema actually sent to the
    interactive LLM (via core.llm_client.call_llm_stream).
  - core/tool_contracts.py: TOOL_DECLARATIONS is the input_schema source
    for every ToolContract, so there is exactly one hand-maintained
    parameter schema per tool, not two independently-drifting ones.
    (agent/planner.py's PLANNER_PROMPT is generated from this same list
    too — see build_planner_tool_reference() — closing the third,
    previously-separate copy that had already gone stale.)
"""

TOOL_DECLARATIONS = [
    # ── Identity ──────────────────────────────────────────────────────────────
    {
        "name": "identify_user",
        "description": (
            "Identifies who is currently in front of the camera or speaking. "
            "Call this when: the user's identity is uncertain, someone new appears, "
            "or when asked 'do you know who I am'. Returns the recognised user profile."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "method": {
                    "type": "STRING",
                    "description": "'face' to use camera, 'voice' to use recent audio, 'both' for combined"
                }
            },
            "required": []
        }
    },
    {
        "name": "register_user",
        "description": (
            "Registers a new user by capturing their face and/or voice sample. "
            "Call when the user says 'remember me', 'learn my face', 'add me', or similar."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "name":   {"type": "STRING", "description": "The person's name"},
                "method": {"type": "STRING", "description": "'face', 'voice', or 'both'"}
            },
            "required": ["name"]
        }
    },
    # ── Apps ──────────────────────────────────────────────────────────────────
    {
        "name": "open_app",
        "description": (
            "Opens or launches any application, website, or program. "
            "Use when the user says: open, launch, start, run, pull up. "
            "Do NOT use send_message just because the app is a messaging app."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "app_name": {"type": "STRING", "description": "Name of the application or website"}
            },
            "required": ["app_name"]
        }
    },
    {
        "name": "web_search",
        "description": "Searches the web for any information.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query":  {"type": "STRING"},
                "mode":   {"type": "STRING", "description": "search (default) or compare"},
                "items":  {"type": "ARRAY",  "items": {"type": "STRING"}},
                "aspect": {"type": "STRING", "description": "price | specs | reviews"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "weather_report",
        "description": "Gets weather for any city.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"city": {"type": "STRING"}},
            "required": ["city"]
        }
    },
    {
        "name": "send_message",
        "description": (
            "Sends a message via WhatsApp, Telegram, etc. "
            "Only call when BOTH a recipient AND message content are given."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "receiver":     {"type": "STRING"},
                "message_text": {"type": "STRING"},
                "platform":     {"type": "STRING"}
            },
            "required": ["receiver", "message_text", "platform"]
        }
    },
    {
        "name": "reminder",
        "description": "Sets a timed reminder.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "date":    {"type": "STRING", "description": "YYYY-MM-DD"},
                "time":    {"type": "STRING", "description": "HH:MM (24h)"},
                "message": {"type": "STRING"}
            },
            "required": ["date", "time", "message"]
        }
    },
    {
        "name": "youtube_video",
        "description": "Plays, summarizes, or gets info about YouTube videos.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "play | summarize | get_info | trending"},
                "query":  {"type": "STRING"},
                "save":   {"type": "BOOLEAN"},
                "region": {"type": "STRING"},
                "url":    {"type": "STRING"}
            },
            "required": []
        }
    },
    {
        "name": "screen_process",
        "description": (
            "Captures a NEW live screenshot or webcam photo right now and analyzes it. "
            "MUST be called when user asks what is on screen, what you see, analyze screen, etc. "
            "Do NOT use this if the user has an uploaded/dropped file (see [UPLOADED FILE] "
            "in the system context) — use file_processor for that instead."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "angle": {"type": "STRING", "description": "screen or camera"},
                "text":  {"type": "STRING", "description": "Question about the captured image"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "computer_settings",
        "description": "Controls volume, brightness, windows, shortcuts, typing, WiFi, restart, etc.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING"},
                "description": {"type": "STRING"},
                "value":       {"type": "STRING"}
            },
            "required": []
        }
    },
    {
        "name": "browser_control",
        "description": "Controls any web browser: open sites, click, type, scroll, fill forms.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING"},
                "browser":     {"type": "STRING"},
                "url":         {"type": "STRING"},
                "query":       {"type": "STRING"},
                "selector":    {"type": "STRING"},
                "text":        {"type": "STRING"},
                "description": {"type": "STRING"},
                "direction":   {"type": "STRING"},
                "amount":      {"type": "INTEGER"},
                "key":         {"type": "STRING"},
                "incognito":   {"type": "BOOLEAN"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "file_controller",
        "description": "Manages files: list, create, delete, move, copy, rename, read, write, find.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING"},
                "path":        {"type": "STRING"},
                "destination": {"type": "STRING"},
                "new_name":    {"type": "STRING"},
                "content":     {"type": "STRING"},
                "name":        {"type": "STRING"},
                "extension":   {"type": "STRING"},
                "count":       {"type": "INTEGER"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "desktop_control",
        "description": "Controls the desktop: wallpaper, organize, clean, stats.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING"},
                "path":   {"type": "STRING"},
                "url":    {"type": "STRING"},
                "mode":   {"type": "STRING"},
                "task":   {"type": "STRING"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "code_helper",
        "description": "Writes, edits, explains, or runs code.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING"},
                "description": {"type": "STRING"},
                "language":    {"type": "STRING"},
                "output_path": {"type": "STRING"},
                "file_path":   {"type": "STRING"},
                "code":        {"type": "STRING"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "dev_agent",
        "description": "Builds complete multi-file projects from scratch.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "description":  {"type": "STRING"},
                "language":     {"type": "STRING"},
                "project_name": {"type": "STRING"},
            },
            "required": ["description"]
        }
    },
    {
        "name": "agent_task",
        "description": (
            "Run a complex multi-step goal autonomously in the background. "
            "Use ONLY when the goal needs 3 or more different tools chained "
            "together. Returns immediately; the result is announced when the "
            "task finishes. Do NOT use this if a single tool can do the job."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "goal": {
                    "type": "STRING",
                    "description": "The full multi-step goal, stated in one sentence"
                },
                "priority": {
                    "type": "STRING",
                    "description": "One of: high, normal, low. Default normal."
                }
            },
            "required": ["goal"]
        }
    },
    {
        "name": "list_pending_approvals",
        "description": (
            "List background tasks that are paused waiting on a yes/no "
            "approval — use when the user asks 'what's pending', 'anything "
            "waiting on me', or similar."
        ),
        "parameters": {"type": "OBJECT", "properties": {}, "required": []}
    },
    {
        "name": "approve_task",
        "description": (
            "Approve or deny a paused background-task step by its approval "
            "ID (from list_pending_approvals). Use when the user says "
            "'approve #5', 'yes, let task X do that', 'deny approval 3', etc."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "approval_id": {"type": "INTEGER", "description": "The approval ID to resolve."},
                "approve":     {"type": "BOOLEAN", "description": "true to approve, false to deny. Default true."},
            },
            "required": ["approval_id"]
        }
    },
    {
        "name": "computer_control",
        "description": "Direct mouse/keyboard control: click, type, scroll, hotkeys.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING"},
                "text":        {"type": "STRING"},
                "x":           {"type": "INTEGER"},
                "y":           {"type": "INTEGER"},
                "keys":        {"type": "STRING"},
                "direction":   {"type": "STRING"},
                "amount":      {"type": "INTEGER"},
                "seconds":     {"type": "NUMBER"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "game_updater",
        "description": "Steam / Epic Games install, update, and management.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":    {"type": "STRING"},
                "platform":  {"type": "STRING"},
                "game_name": {"type": "STRING"},
                "app_id":    {"type": "STRING"},
                "hour":      {"type": "INTEGER"},
                "minute":    {"type": "INTEGER"},
                "shutdown_when_done": {"type": "BOOLEAN"},
            },
            "required": []
        }
    },
    {
        "name": "flight_finder",
        "description": "Searches Google Flights and speaks the best options.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "origin":      {"type": "STRING"},
                "destination": {"type": "STRING"},
                "date":        {"type": "STRING"},
                "return_date": {"type": "STRING"},
                "passengers":  {"type": "INTEGER"},
                "cabin":       {"type": "STRING"},
            },
            "required": ["origin", "destination", "date"]
        }
    },
    {
        "name": "file_processor",
        "description": (
            "Processes an already-uploaded/dropped file on disk: images, PDFs, CSV, "
            "audio, video, etc. Use this whenever the user refers to a file they "
            "uploaded/dropped, or asks \"what is this\"/\"describe this\" while a file "
            "is loaded (see [UPLOADED FILE] in the system context) — NOT screen_process, "
            "which captures a brand-new live screenshot/webcam image instead."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "file_path":   {"type": "STRING"},
                "action":      {"type": "STRING"},
                "instruction": {"type": "STRING"},
                "format":      {"type": "STRING"},
            },
            "required": []
        }
    },
    {
        "name": "save_memory",
        "description": (
            "Save a personal fact about the user. "
            "Call IMMEDIATELY when user states name, age, city, job, preference, or relationship. "
            "Call SILENTLY — never announce you are saving."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "category": {"type": "STRING",
                             "description": "identity | preferences | projects | relationships | wishes | notes"},
                "key":      {"type": "STRING"},
                "value":    {"type": "STRING"},
            },
            "required": ["category", "key", "value"]
        }
    },
    {
        "name": "shutdown_jarvis",
        "description": "Shuts down JARVIS when the user says goodbye or stop.",
        "parameters": {"type": "OBJECT", "properties": {}}
    },
    {
        "name": "daily_briefing",
        "description": (
            "Give the user a spoken morning/afternoon/evening briefing: "
            "greeting, today's date, current weather in their city, "
            "today's pending reminders, and top news headlines. Use when "
            "the user asks for a briefing, summary of their day, or says "
            "something like 'what's my day look like' or 'give me a briefing'."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "city": {
                    "type": "STRING",
                    "description": (
                        "Optional. City for the weather portion. If omitted, "
                        "pulled automatically from saved memory."
                    )
                }
            },
            "required": []
        }
    },
    {
        "name": "vision_fix_code",
        "description": (
            "Look at the user's screen, identify a bug in the visible code, "
            "and fix it directly on disk after confirming with the user. "
            "Use ONLY when the user explicitly asks to look at/fix code on "
            "their screen — e.g. 'look at my screen and fix this bug', "
            "'what's wrong with this code'. Requires the file to be visibly "
            "open in an editor."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
            "required": []
        }
    },
]


# ── Type conversion (Gemini → Ollama format) ─────────────────────────────────
_TYPE_MAP = {
    "OBJECT": "object", "STRING": "string", "ARRAY": "array",
    "INTEGER": "integer", "BOOLEAN": "boolean", "NUMBER": "number",
}

def _convert_type(t): return _TYPE_MAP.get(t, t.lower()) if isinstance(t, str) else t
def _convert_props(props):
    out = {}
    for k, v in props.items():
        nv = dict(v)
        if "type" in nv: nv["type"] = _convert_type(nv["type"])
        if "items" in nv and isinstance(nv["items"], dict):
            nv["items"] = {"type": _convert_type(nv["items"].get("type", "string"))}
        out[k] = nv
    return out

def _to_ollama_tools(decls):
    tools = []
    for d in decls:
        params = d.get("parameters", {})
        new_params = {
            "type": "object",
            "properties": _convert_props(params.get("properties", {})),
        }
        req = params.get("required")
        if req: new_params["required"] = req
        tools.append({
            "type": "function",
            "function": {
                "name":       d["name"],
                "description": d["description"],
                "parameters": new_params,
            },
        })
    return tools

OLLAMA_TOOLS = _to_ollama_tools(TOOL_DECLARATIONS)


def get_declaration(tool_name: str) -> dict | None:
    """Returns the raw TOOL_DECLARATIONS entry for `tool_name`, or None."""
    for d in TOOL_DECLARATIONS:
        if d["name"] == tool_name:
            return d
    return None


def _render_param_line(name: str, spec: dict, required: bool) -> str:
    raw_type = spec.get("type", "STRING")
    if raw_type == "ARRAY":
        item_type = spec.get("items", {}).get("type", "STRING").lower()
        ptype = f"list of {item_type}"
    else:
        ptype = raw_type.lower()
    req_str = "required" if required else "optional"
    desc    = spec.get("description", "")
    suffix  = f" — {desc}" if desc else ""
    return f"  {name}: {ptype} ({req_str}){suffix}"


def build_planner_tool_reference(tool_names: set[str] | None = None) -> str:
    """Renders TOOL_DECLARATIONS as the prose parameter reference
    agent/planner.py's PLANNER_PROMPT embeds — generated from the same
    canonical schema every tool call is validated against
    (core/tool_contracts.py), instead of being a second, separately
    hand-maintained copy that drifts (the previous hand-written version
    was missing file_processor/daily_briefing/vision_fix_code entirely
    and had a stale action list for code_helper).

    `tool_names`, if given, restricts the reference to that subset (e.g.
    agent/planner.py passes TOOL_DISPATCH's keys, since the planner
    shouldn't be told about main.py-only tools like save_memory or
    agent_task itself)."""
    lines: list[str] = []
    for decl in TOOL_DECLARATIONS:
        name = decl["name"]
        if tool_names is not None and name not in tool_names:
            continue
        params   = decl.get("parameters", {})
        props    = params.get("properties", {})
        required = set(params.get("required", []))
        lines.append(name)
        if not props:
            lines.append("  (no parameters)")
        else:
            for pname, pspec in props.items():
                lines.append(_render_param_line(pname, pspec, pname in required))
        lines.append("")
    return "\n".join(lines).rstrip()
