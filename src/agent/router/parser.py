"""
Request Parser — Lightweight pre-processor that runs BEFORE Laya.
Uses regex and keyword rules to extract clean intent and arguments from raw user text.
This eliminates the need for another model while giving Laya clean, structured input.
"""
import re
from typing import TypedDict

class ParsedRequest(TypedDict):
    raw: str           # Original user request
    cleaned: str       # Request stripped of action verbs (for Laya to route)
    target: str        # Extracted target entity (app name, URL, search query)
    likely_category: str  # Hint: 'browser', 'application', 'interaction', or 'unknown'

# File Explorer patterns
FILE_EXPLORER_PATTERNS = [
    r"^(please\s+)?(can you\s+)?organi[sz]e\s+(the\s+)?files?\s+(in\s+)?",
    r"^(please\s+)?(can you\s+)?group\s+(the\s+)?files?\s+(in\s+)?",
    r"^(please\s+)?(can you\s+)?open\s+(the\s+)?folder\s+(at\s+)?",
    r"^(please\s+)?(can you\s+)?list\s+(the\s+)?files?\s+(in\s+)?",
    r"^(please\s+)?(can you\s+)?create\s+(a\s+)?folder\s+",
    r"^(please\s+)?(can you\s+)?move\s+(the\s+)?file\s+",
    r"^(please\s+)?(can you\s+)?search\s+(for\s+)?files?\s+(in\s+)?",
]

# Action verb prefixes to strip from requests
OPEN_PATTERNS = [
    r"^(please\s+)?(can you\s+)?open\s+",
    r"^(please\s+)?(can you\s+)?launch\s+",
    r"^(please\s+)?(can you\s+)?start\s+",
    r"^(please\s+)?(can you\s+)?run\s+",
]

SEARCH_PATTERNS = [
    r"^(please\s+)?(can you\s+)?search\s+(for\s+)?",
    r"^(please\s+)?(can you\s+)?find\s+",
    r"^(please\s+)?(can you\s+)?(look up|lookup)\s+",
    r"^(please\s+)?(can you\s+)?(google|bing|search)\s+(for\s+)?",
    r"^(please\s+)?(can you\s+)?tell me (about|where is|what is|who is|how to)\s+",
    r"^(where is|what is|who is|how to|why is|when is|how does|how many|how much)\s+",
]

# Interaction patterns: keyboard and mouse control commands
INTERACTION_PATTERNS = [
    r"^(please\s+)?(can you\s+)?take\s+(a\s+)?screenshot",
    r"^(please\s+)?(can you\s+)?press\s+",
    r"^(please\s+)?(can you\s+)?click\s+",
    r"^(please\s+)?(can you\s+)?double.?click\s+",
    r"^(please\s+)?(can you\s+)?right.?click\s+",
    r"^(please\s+)?(can you\s+)?type\s+",
    r"^(please\s+)?(can you\s+)?move (the\s+)?mouse",
    r"^(please\s+)?(can you\s+)?drag\s+",
    r"^(please\s+)?(can you\s+)?scroll\s+",
    r"^(please\s+)?(can you\s+)?(switch|alt.?tab|tab through)",
    r"^(please\s+)?(can you\s+)?use (the\s+)?keyboard",
    r"^(please\s+)?(can you\s+)?hold\s+",
]

URL_PATTERN = re.compile(
    r"^(https?://)?"  # optional http(s)
    r"(www\.)?"        # optional www
    r"[a-zA-Z0-9.-]+" # domain
    r"\.[a-zA-Z]{2,}" # TLD
    r"(/\S*)?\s*$"     # optional path
)

def _strip_verbs(text: str, patterns: list[str]) -> str | None:
    """Strip verb patterns from the beginning of a string. Returns the remainder or None if no match."""
    for pattern in patterns:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            return text[match.end():].strip()
    return None

def parse_request(raw: str) -> ParsedRequest:
    """
    Parses a raw user request into a structured ParsedRequest.
    """
    text = raw.strip()
    lower = text.lower()
    
    # --- Try to detect FILE EXPLORER ---
    stripped_file = _strip_verbs(text, FILE_EXPLORER_PATTERNS)
    if stripped_file or re.search(r"[a-zA-Z]:\\", text) and ("folder" in lower or "organi" in lower or "file" in lower):
        target = stripped_file if stripped_file else text
        return ParsedRequest(
            raw=raw,
            cleaned=f"file management {target}",
            target=target,
            likely_category="file_explorer"
        )
    
    # --- Try to detect OPEN / APPLICATION ---
    stripped = _strip_verbs(text, OPEN_PATTERNS)
    if stripped:
        return ParsedRequest(
            raw=raw,
            cleaned=f"launch {stripped}",
            target=stripped,
            likely_category="application"
        )
    
    # --- Try to detect INTERACTION (keyboard/mouse) ---
    stripped = _strip_verbs(text, INTERACTION_PATTERNS)
    if stripped is not None:  # stripped can be empty string for 'move the mouse'
        return ParsedRequest(
            raw=raw,
            cleaned=text,      # keep full text so Laya has the action verb context
            target=text,
            likely_category="interaction"
        )
    
    # --- Try to detect SEARCH ---
    stripped = _strip_verbs(text, SEARCH_PATTERNS)
    if stripped:
        return ParsedRequest(
            raw=raw,
            cleaned=f"search for {stripped}",
            target=stripped,
            likely_category="browser"
        )
    
    # --- Check if the raw text itself looks like a URL ---
    if URL_PATTERN.match(lower):
        url = text if text.startswith("http") else f"https://{text}"
        return ParsedRequest(
            raw=raw,
            cleaned=f"open url {url}",
            target=url,
            likely_category="browser"
        )
    
    # --- Fallback: unknown — let Laya decide without a biased hint ---
    return ParsedRequest(
        raw=raw,
        cleaned=text,
        target=text,
        likely_category="unknown"
    )
