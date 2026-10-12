"""
Planner — uses Hammer2.1 (via Ollama) to decide the ONE immediate next step.
The planner receives: original goal + verified completed steps (from disk).
It leverages Hammer's native tool-calling capabilities to output precise actions,
which are then converted back to natural language for the existing parser/router.
"""
import ast
import json
import ollama
from src.agent.memory.storage import get_recent_steps, format_steps_simple
from src.tools.registry import get_slm_tool_schemas

OLLAMA_MODEL = "hf.co/keisuke-miyako/Hammer2.1-1.5b-gguf-q4_k_m:Q4_K_M"
GOAL_COMPLETE_SIGNAL = "GOAL_COMPLETE"

# Hammer requires explicit OpenAI-compatible tool definitions for ALL user-defined tools
# We now load these dynamically from the central registry
HAMMER_TOOLS = get_slm_tool_schemas()


SPECIAL_STEP_PHRASES = {
    "goal_complete": lambda args: GOAL_COMPLETE_SIGNAL,
    "search_browser": lambda args: f"search for {args.get('query', '')}",
    "open_url": lambda args: f"open url {args.get('url', '')}",
    "type_text": lambda args: f"type {args.get('text', '')}",
    "press_key": lambda args: f"press {args.get('key', '')}",
    "press_hotkey": lambda args: f"press hotkey {args.get('hotkey', '')}",
    "move_mouse": lambda args: f"move mouse to {args.get('x', 0)}, {args.get('y', 0)}",
    "get_unread_emails": lambda args: f"get {args.get('max_results', 5)} unread emails",
    "send_email": lambda args: f"send email to {args.get('to', '')}",
    "create_event": lambda args: f"create event '{args.get('summary', '')}'",
    "search_files": lambda args: f"search for file {args.get('name', '')} in {args.get('directory', '')}",
    "move_file": lambda args: f"move file {args.get('source', '')} to {args.get('destination', '')}",
    "copy_file": lambda args: f"copy file {args.get('source', '')} to {args.get('destination', '')}",
    "rename_file": lambda args: f"rename file {args.get('old_path', '')} to {args.get('new_path', '')}",
    "create_folder": lambda args: f"create folder {args.get('folder_name', '')} in {args.get('directory', '')}",
    "delete_item": lambda args: f"delete item {args.get('path', '')}",
}


def _map_to_step(f_name: str, f_args: dict = None) -> str:
    """
    Option 1: Generic Auto-Formatter + Smart Overrides.
    Maps Hammer function calls to natural language steps without brittle if/elif chains.
    """
    if not f_name:
        return GOAL_COMPLETE_SIGNAL

    f_args = f_args or {}
    if f_name in SPECIAL_STEP_PHRASES:
        return SPECIAL_STEP_PHRASES[f_name](f_args)

    # Normalize action name (e.g. 'open_application_cmd' -> 'open', 'open_folder' -> 'open folder')
    action = f_name
    for prefix in ("open_application_cmd", "open_application_gui", "open_app"):
        if action == prefix:
            action = "open"
            break
    action = action.replace("_", " ")

    if not f_args:
        return action

    # Single argument: format naturally (e.g. "open Word", "organize files in D:\temp")
    if len(f_args) == 1:
        val = next(iter(f_args.values()))
        if "organize" in action:
            return f"{action} in {val}".strip()
        return f"{action} {val}".strip()

    # Multi-argument clean fallback: "action (k1='v1', k2='v2')"
    args_str = ", ".join(f"{k}='{v}'" for k, v in f_args.items())
    return f"{action} ({args_str})".strip()


def generate_full_plan(goal: str, run_folder: str = "") -> list:
    """
    Given the original goal, ask Hammer for the full sequence of function calls.
    Returns a list of natural language strings.
    """
    messages = [
        {"role": "system", "content": "You are a helpful robotic assistant that solves goals by calling tools."}
    ]
    
    if run_folder:
        recent_steps = get_recent_steps(run_folder, n=10)
        if recent_steps:
            history_lines = []
            for s in recent_steps:
                status = "SUCCESS" if s.get("verified") else "FAILED"
                history_lines.append(f"Step {s['step_num']}: {s['step']}\nStatus: {status}\nResult: {s['result']}")
            history_text = "\n\n".join(history_lines)
            
            messages.append({
                "role": "user", 
                "content": f"Goal: {goal}\n\nYou have already executed the following steps:\n{history_text}\n\nCRITICAL INSTRUCTION: Do NOT repeat the steps you have already executed! Based on the results above, output only the NEXT new step(s) required to complete the goal."
            })
            return _call_ollama(messages)

    messages.append({"role": "user", "content": f"Please complete this goal: {goal}"})
    return _call_ollama(messages)


def _call_ollama(messages):
    """Helper to actually call Ollama and parse."""
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            tools=HAMMER_TOOLS,
            options={"temperature": 0.0, "num_predict": 1024}
        )
    except Exception as e:
        print(f"[Planner] Error calling Ollama: {e}")
        return []

    msg = response.get("message", {})
    generated_actions = []

    tool_calls = msg.get("tool_calls")
    if tool_calls and len(tool_calls) > 0:
        for call in tool_calls:
            func = call.get("function", {})
            f_name = func.get("name")
            f_args = func.get("arguments", {})
            act = _map_to_step(f_name, f_args).strip()
            if act and act != GOAL_COMPLETE_SIGNAL:
                generated_actions.append({"desc": act, "tool_name": f_name, "tool_args": f_args})
    else:
        content = msg.get("content", "")
        if "[{" in content and "}]" in content:
            start = content.find("[{")
            end = content.rfind("}]") + 2
            list_str = content[start:end]
            try:
                parsed = ast.literal_eval(list_str)
                for item in parsed:
                    f = item.get("function", item)
                    f_name = f.get("name")
                    f_args = f.get("arguments", {})
                    act = _map_to_step(f_name, f_args).strip()
                    if act and act != GOAL_COMPLETE_SIGNAL:
                        generated_actions.append({"desc": act, "tool_name": f_name, "tool_args": f_args})
            except Exception as e:
                print(f"[Planner] Failed to parse AST: {e}")
                pass

    if not generated_actions:
        print("[Planner] Could not find any valid function calls in Hammer's output.")
        
    return generated_actions
