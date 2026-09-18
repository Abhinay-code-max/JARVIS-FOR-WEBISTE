"""
core/intent_classifier.py
=========================
Fast, conservative intent classifier that decides whether an incoming user utterance
is a pure conversational turn (small talk, greeting, small question) or an action/tool request.

Design principles:
  - High recall for action requests: When in doubt, always route through the planner pipeline.
  - Zero latency penalty for pure small talk: Unambiguous greetings and conversational turns
    are handled via lightweight direct LLM streaming with tools=[] (zero tool schema overhead).
"""
import re

_ACTION_KEYWORDS = re.compile(
    r"\b("
    # Process & App Control
    r"open|launch|start|run|close|exit|kill|quit|shutdown|restart|reboot|"
    # Files & Code
    r"delete|remove|erase|create|make|write|build|generate|code|script|python|fix|debug|refactor|"
    # Info & Weather
    r"weather|forecast|rain|temperature|news|briefing|headline|"
    # Search & Retrieval
    r"search|google|find|look\s*up|check|browse|"
    # Filesystem & OS
    r"file|folder|directory|desktop|document|screenshot|camera|screen|"
    # Messaging & Social
    r"send|message|whatsapp|telegram|email|text|chat|"
    # Reminders & Clocks
    r"remind|reminder|timer|alarm|schedule|event|calendar|"
    # Settings & Hardware
    r"volume|brightness|wifi|bluetooth|mute|unmute|battery|power|"
    # Memory & Identity
    r"save|remember|learn|who\s*am\s*i|register"
    r")\b",
    re.IGNORECASE,
)

_PURE_CHAT_PATTERNS = [
    re.compile(r"^(hi|hello|hey|hey\s+jarvis|good\s+(morning|afternoon|evening|night))[\.!\?]*$", re.IGNORECASE),
    re.compile(r"^(how\s+are\s+you|how's\s+it\s+going|how\s+do\s+you\s+do)[\.!\?]*$", re.IGNORECASE),
    re.compile(r"^(who\s+are\s+you|what\s+is\s+your\s+name|what\s+are\s+you)[\.!\?]*$", re.IGNORECASE),
    re.compile(r"^(what\s+can\s+you\s+do|help|tell\s+me\s+about\s+yourself)[\.!\?]*$", re.IGNORECASE),
    re.compile(r"^(thanks|thank\s+you|thank\s+you\s+so\s+much|great|cool|awesome|ok|okay|nice|perfect)[\.!\?]*$", re.IGNORECASE),
    re.compile(r"^(tell\s+me\s+a\s+joke|make\s+me\s+laugh)[\.!\?]*$", re.IGNORECASE),
]


def is_pure_conversational(text: str) -> bool:
    """
    Returns True ONLY if the utterance is unambiguously pure conversation/small-talk
    with no action/tool requirement. Conservative default: returns False when uncertain.
    """
    if not text:
        return True

    cleaned = text.strip()

    # 1. Any action keyword forces planner/tool routing
    if _ACTION_KEYWORDS.search(cleaned):
        return False

    # 2. Match exact pure chat patterns
    for pat in _PURE_CHAT_PATTERNS:
        if pat.match(cleaned):
            return True

    # 3. Simple factual or conceptual questions without action verbs (e.g. "what is photosynthesis", "why is the sky blue")
    if re.match(r"^(what|why|how|explain)\s+(is|are|do|does)\b", cleaned, re.IGNORECASE) and not _ACTION_KEYWORDS.search(cleaned):
        return True

    # Conservative default: route to planner
    return False
