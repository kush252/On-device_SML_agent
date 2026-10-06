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


from src.agent.memory.storage import get_recent_steps

def generate_full_plan(goal: str, run_folder: str = "") -> list:
    """
    Given the original goal, ask Hammer for the full sequence of function calls.
    Returns a list of natural language strings.
    """
    messages = [
        {"role": "system", "content": "You are a helpful robotic assistant that solves goals by calling tools."}
    ]
    
    if run_folder:
        from src.agent.memory.storage import get_recent_steps
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
    generated_actions = []
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

    # Map a single Hammer function call to a natural language step
    def _map_to_step(f_name, f_args):
        if f_name == "goal_complete":
            return GOAL_COMPLETE_SIGNAL
        elif f_name == "open_url":
            return f"open url {f_args.get('url', '')}"
        elif f_name == "search_browser":
            return f"search for {f_args.get('query', '')}"
        elif f_name in ("open_app", "open_application_gui", "open_application_cmd"):
            return f"open {f_args.get('app_name', '')}"
        elif f_name == "type_text":
            return f"type {f_args.get('text', '')}"
        elif f_name == "press_key":
            return f"press {f_args.get('key', '')}"
        elif f_name == "press_hotkey":
            return f"press hotkey {f_args.get('hotkey', '')}"
        elif f_name == "move_mouse":
            return f"move mouse to {f_args.get('x', 0)}, {f_args.get('y', 0)}"
        elif f_name == "click":
            return "click"
        elif f_name == "double_click":
            return "double click"
        elif f_name == "right_click":
            return "right click"
        elif f_name == "get_mouse_position":
            return "get mouse position"
        elif f_name == "take_screenshot":
            return "take screenshot"
        elif f_name == "get_unread_emails":
            return f"get {f_args.get('max_results', 5)} unread emails"
        elif f_name == "send_email":
            return f"send email to {f_args.get('to', '')}"
        elif f_name == "wait_for_new_email":
            return "wait for new email"
        elif f_name == "get_upcoming_events":
            return f"get {f_args.get('max_results', 10)} upcoming events"
        elif f_name == "create_event":
            return f"create event '{f_args.get('summary', '')}'"
        return GOAL_COMPLETE_SIGNAL

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
