"""
Brain — Routing engine for the autonomous agent.
Architecture:
  Laya (System-1, fast) → confidence check → SLM fallback (qwen2.5:1.5b via Ollama) if low confidence
"""

# --- Confidence threshold ---
# Below this value, Laya's decision is considered unreliable and the SLM fallback is used.
CONFIDENCE_THRESHOLD = 0.72

try:
    from laya import Router
    laya_router = Router(preload=True)
except ImportError:
    laya_router = None
    print("Warning: 'laya' package not found. Routing will use SLM fallback only.")

from src.tools.registry import get_top_level_categories, get_category_tools
from src.agent.router.slm_fallback import slm_categorize, slm_select_tool

# ---------------------------------------------------------------------------
# Rich Laya criteria (guide Section 8: explicit descriptions + examples + contrasts)
# ---------------------------------------------------------------------------

CATEGORY_CRITERIA = {
    "browser": (
        "Requests that need the internet — searching for information, looking something up, "
        "finding facts, opening a website, visiting a URL, or asking questions about the world. "
        "Examples: 'what is X', 'how many Y', 'search for Z', 'where is X', 'open youtube.com', "
        "'find information about X'. NOT for opening desktop apps or controlling the mouse/keyboard."
    ),
    "application": (
        "Requests to launch or open a locally installed desktop application on this computer. "
        "Examples: 'open WhatsApp', 'launch Spotify', 'start Notepad', 'open VS Code', 'run Chrome'. "
        "The key signal is an app name following a launch verb. "
        "NOT for internet searches or keyboard/mouse actions."
    ),
    "interaction": (
        "Requests to physically control the keyboard or mouse — pressing keys, clicking, typing text, "
        "moving the cursor, scrolling, using key combinations, or switching windows. "
        "Examples: 'press enter', 'press alt tab', 'click here', 'move the mouse', 'type hello', "
        "'press ctrl+c', 'scroll down', 'switch the app', 'hold shift'. "
        "NOT for opening apps or searching the internet."
    )
}

TOOL_CRITERIA = {
    "browser": {
        "open_url": (
            "Use when the target is an exact website address or URL to navigate to directly. "
            "Examples: 'go to github.com', 'open https://google.com', 'visit reddit.com'. "
            "The input must be a URL, not a question or phrase."
        ),
        "search_browser": (
            "Use when the target is a question, a topic, or any phrase that needs to be searched online. "
            "Examples: 'what is machine learning', 'weather in Mumbai', 'best restaurants near me'. "
            "Use this for anything that is not already a full URL."
        )
    },
    "application": {
        "open_app": (
            "The primary and preferred method to open any application. Uses Windows Search to find and launch it. "
            "Works for all apps including WhatsApp, Spotify, Notepad, VS Code, Chrome, etc."
        ),
        "open_application_gui": (
            "Same as open_app but explicitly simulates Windows key press and typing. "
            "Use only if open_app is unavailable."
        ),
        "open_application_cmd": (
            "Opens an application through the command line. Only use for apps with known CLI commands "
            "like 'code' for VS Code or 'notepad'. NOT for mobile/store apps like WhatsApp."
        )
    },
    "interaction": {
        "type_text": (
            "Types out a string of text character by character as if typed on a keyboard. "
            "Examples: 'type hello world', 'write this message'."
        ),
        "press_key": (
            "Presses a single keyboard key. "
            "Examples: 'press enter', 'press escape', 'press tab', 'press space', 'press backspace'."
        ),
        "press_hotkey": (
            "Presses a combination of keys held simultaneously. "
            "Examples: 'press ctrl+c', 'press alt+tab', 'press ctrl+alt+delete', 'switch app'. "
            "Use for any multi-key shortcut."
        ),
        "move_mouse": (
            "Moves the mouse cursor to a specific (x, y) position on screen. "
            "Examples: 'move the mouse', 'move cursor to position'."
        ),
        "click": (
            "Performs a mouse click. Can be left or right click at a coordinate. "
            "Examples: 'click here', 'left click', 'click on the button'."
        ),
        "double_click": (
            "Performs a double-click at a position. "
            "Examples: 'double click', 'open file by double clicking'."
        ),
        "right_click": (
            "Performs a right-click to open context menus. "
            "Examples: 'right click', 'open context menu'."
        ),
        "get_mouse_position": (
            "Returns the current coordinates of the mouse cursor. "
            "Examples: 'where is the mouse', 'get cursor position'."
        )
    }
}


# ---------------------------------------------------------------------------
# Public routing functions
# ---------------------------------------------------------------------------

def categorize_with_laya(request: str) -> str:
    """
    Routes a user request to one of: 'browser', 'application', 'interaction'.
    
    Fast path: Laya (non-autoregressive, single forward pass).
    Fallback:  qwen2.5:1.5b via Ollama if Laya's confidence < CONFIDENCE_THRESHOLD.
    """
    if not laya_router:
        # No Laya — go straight to SLM
        print("[Brain] Laya unavailable → SLM fallback")
        return slm_categorize(request, CATEGORY_CRITERIA)

    questions = {
        "category": {
            "type": "choice",
            "instructions": (
                "You are routing a user request to the correct system. "
                "Choose the single best category for this request."
            ),
            "criteria": CATEGORY_CRITERIA
        }
    }

    result = laya_router.predict(request, questions)
    answer = result["answers"]["category"]
    choice = answer["choice"]
    confidence = answer.get("confidence", 1.0)

    print(f"[Laya] category='{choice}' confidence={confidence:.3f}")

    if confidence >= CONFIDENCE_THRESHOLD:
        return choice

    # Low confidence — hand off to SLM
    print(f"[Brain] Low confidence ({confidence:.3f} < {CONFIDENCE_THRESHOLD}) → SLM fallback")
    slm_choice = slm_categorize(request, CATEGORY_CRITERIA)
    print(f"[SLM] category='{slm_choice}'")
    return slm_choice


def select_tool_with_laya(request: str, category: str) -> str:
    """
    Selects the exact tool within a category.
    
    Fast path: Laya.
    Fallback:  qwen2.5:1.5b via Ollama if confidence < CONFIDENCE_THRESHOLD.
    """
    tools = TOOL_CRITERIA.get(category, {})
    if not tools:
        tools = get_category_tools(category)

    if not laya_router:
        print("[Brain] Laya unavailable → SLM fallback for tool selection")
        return slm_select_tool(request, category, tools)

    questions = {
        "tool": {
            "type": "choice",
            "instructions": (
                f"Choose the single best tool from the '{category}' category to handle this request."
            ),
            "criteria": tools
        }
    }

    result = laya_router.predict(request, questions)
    answer = result["answers"]["tool"]
    choice = answer["choice"]
    confidence = answer.get("confidence", 1.0)

    print(f"[Laya] tool='{choice}' confidence={confidence:.3f}")

    if confidence >= CONFIDENCE_THRESHOLD:
        return choice

    print(f"[Brain] Low confidence ({confidence:.3f} < {CONFIDENCE_THRESHOLD}) → SLM fallback for tool")
    slm_choice = slm_select_tool(request, category, tools)
    print(f"[SLM] tool='{slm_choice}'")
    return slm_choice
