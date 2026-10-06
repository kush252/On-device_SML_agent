"""
SLM Fallback — qwen2.5:1.5b via Ollama.
Called when Laya's routing confidence falls below a threshold.
Uses a tightly structured prompt to force a single-token response matching one of the valid options.
"""
import json
import re

try:
    import ollama
    _ollama_available = True
except ImportError:
    _ollama_available = False
    print("Warning: 'ollama' package not found. Install with: pip install ollama")

OLLAMA_MODEL = "hf.co/keisuke-miyako/Hammer2.1-1.5b-gguf-q4_k_m:Q4_K_M"

def _call_slm(prompt: str) -> str:
    """Raw call to the Ollama model. Returns the response text."""
    if not _ollama_available:
        raise RuntimeError("ollama package not installed.")
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0}  # deterministic output
    )
    return response["message"]["content"].strip()

def slm_categorize(request: str, categories: dict) -> str:
    """
    Asks the SLM to pick the correct category for a request.
    Formats a strict prompt that forces a single-word answer from the valid options.

    Args:
        request: the raw user request
        categories: dict of {category_name: description}
    Returns:
        The chosen category name string.
    """
    options_text = "\n".join(
        f'- "{k}": {v}' for k, v in categories.items()
    )
    valid_keys = list(categories.keys())

    prompt = f"""You are a routing classifier. Your ONLY job is to output exactly one word from the list below.

User request: "{request}"

Categories:
{options_text}

Rules:
- Output ONLY the category name, nothing else.
- Do not explain, do not add punctuation.
- Valid outputs: {valid_keys}

Category:"""

    raw = _call_slm(prompt).lower().strip().strip('"').strip("'")

    # Match to a valid key (handle partial or quoted responses)
    for key in valid_keys:
        if key in raw:
            return key

    # Fallback if model hallucinated
    print(f"[SLM] Unexpected response '{raw}', defaulting to 'browser'")
    return "browser"


def slm_select_tool(request: str, category: str, tools: dict) -> str:
    """
    Asks the SLM to pick the exact tool within a category.

    Args:
        request: the raw user request
        category: the already-confirmed category
        tools: dict of {tool_name: description}
    Returns:
        The chosen tool name string.
    """
    options_text = "\n".join(
        f'- "{k}": {v}' for k, v in tools.items()
    )
    valid_keys = list(tools.keys())

    prompt = f"""You are a tool selector. Your ONLY job is to output exactly one tool name from the list below.

User request: "{request}"
Category: {category}

Available tools:
{options_text}

Rules:
- Output ONLY the tool name, nothing else.
- Do not explain, do not add punctuation.
- Valid outputs: {valid_keys}

Tool:"""

    raw = _call_slm(prompt).lower().strip().strip('"').strip("'")

    for key in valid_keys:
        if key in raw:
            return key

    # Fallback defaults per category
    defaults = {"browser": "search_browser", "application": "open_app", "interaction": "press_key"}
    fallback = defaults.get(category, valid_keys[0])
    print(f"[SLM] Unexpected response '{raw}', defaulting to '{fallback}'")
    return fallback
