import re
from .state import AgentState
from .router.brain import categorize_with_laya, select_tool_with_laya
from .router.parser import parse_request
from .memory.storage import create_run, append_step, update_step_verified
from .planner import generate_full_plan, GOAL_COMPLETE_SIGNAL
from .verifier import verify_step

MAX_STEPS = 12  # safety limit to prevent runaway loops


# ─────────────────────────────────────────────────────────────────────────────
# Multi-step nodes
# ─────────────────────────────────────────────────────────────────────────────

def init_run_node(state: AgentState) -> AgentState:
    """
    Node 0A: Initialise a new run.
    Creates the run folder on disk and seeds the state with run metadata.
    """
    goal = state["request"]
    run_id, run_folder = create_run(goal)
    print(f"\n[Run] Started run '{run_id}' → {run_folder}")
    return {
        "goal": goal,
        "run_id": run_id,
        "run_folder": run_folder,
        "step_count": 0,
        "goal_satisfied": False,
        "step_verified": None,
        "full_plan": []
    }


def plan_node(state: AgentState) -> AgentState:
    """
    Node 0B: Sequential Planner.
    Generates the entire plan once on step 0, then feeds the steps sequentially.
    """
    goal = state["goal"]
    full_plan = state.get("full_plan", [])
    step_count = state.get("step_count", 0)

    # 1. Generate plan or re-plan if we ran out of steps
    if not full_plan or step_count >= len(full_plan):
        run_folder = state.get("run_folder", "")
        if full_plan:
            print(f"\n[Planner] Re-evaluating plan with updated context...")
        else:
            print(f"\n[Planner] Generating full execution plan for goal: {goal}")
            
        new_plan = generate_full_plan(goal, run_folder)
        
        if not new_plan or (len(new_plan) == 1 and new_plan[0].get("tool_name") == "goal_complete"):
            print("[Planner] Planner declared goal complete.")
            return {"goal_satisfied": True, "current_step": GOAL_COMPLETE_SIGNAL, "full_plan": full_plan}
            
        # Anti-loop check: if the planner suggests the exact same tool as the last step
        if full_plan and len(new_plan) > 0 and full_plan[-1].get("tool_name") == new_plan[0].get("tool_name"):
            print(f"[Planner] Loop detected! Planner suggested '{new_plan[0].get('tool_name')}' again. Forcing GOAL_COMPLETE.")
            return {"goal_satisfied": True, "current_step": GOAL_COMPLETE_SIGNAL, "full_plan": full_plan}
            
        full_plan.extend(new_plan)
        print(f"[Planner] Generated Plan ({len(new_plan)} new steps): {[n['desc'] for n in new_plan]}")
        
    # 3. Fetch the next step sequentially
    next_step = full_plan[step_count]
    print(f"\n[Planner] Executing Step {step_count + 1}/{len(full_plan)}: \"{next_step['desc']}\"")

    return {
        "full_plan": full_plan,
        "current_step": next_step["desc"],
        "request": next_step["desc"],
        # Fast-track execution properties:
        "selected_tool": next_step.get("tool_name"),
        "tool_arguments": next_step.get("tool_args"),
        # Reset routing fields
        "parsed_target": None,
        "likely_category": None,
        "category": None,
        "tool_result": None
    }


def save_step_node(state: AgentState) -> AgentState:
    """
    Node between execute and verify.
    Writes the completed step to disk so verifier + future planners can load it.
    """
    run_folder = state.get("run_folder", "")
    step_count = state.get("step_count", 0) + 1
    step_desc = state.get("current_step", "")
    tool = state.get("selected_tool", "unknown")
    args = state.get("tool_arguments", {})
    result = str(state.get("tool_result", ""))

    append_step(
        run_folder=run_folder,
        step_num=step_count,
        step_desc=step_desc,
        tool=tool,
        args=args,
        result=result,
        verified=False  # will be updated by verify node
    )

    print(f"[Storage] Step {step_count} saved.")
    return {"step_count": step_count}


def verify_node(state: AgentState) -> AgentState:
    """
    Node after save_step.
    Asks Laya (+ SLM fallback) one question:
      1. Was this step successful?
    The goal_satisfied check is removed because the 1.5b SLM fallback 
    isn't reliable enough to assess overall completion. We let the planner
    decide when the goal is done via the GOAL_COMPLETE_SIGNAL.
    Updates 'step_verified' in state.
    """
    goal = state.get("goal", state["request"])
    run_folder = state.get("run_folder", "")
    current_step = state.get("current_step", "")
    tool_result = str(state.get("tool_result", ""))
    step_count = state.get("step_count", 0)

    step_ok = verify_step(
        goal=goal,
        run_folder=run_folder,
        current_step=current_step,
        tool_result=tool_result
    )

    # ── KEY FIX: write verified=True back to steps.json ──────────────────────
    # This is what makes format_steps_simple show the step as 'done' to the
    # planner. Without this, all steps stay verified=False and the planner
    # thinks every step failed, causing an infinite loop.
    if step_ok and run_folder:
        update_step_verified(run_folder, step_count, verified=True)
        print(f"[Storage] Step {step_count} marked verified=True")

    goal_done = False
    # If we've hit the safety cap, force exit
    if step_count >= MAX_STEPS:
        print(f"[Verifier] Reached max steps ({MAX_STEPS}). Stopping.")
        goal_done = True

    return {
        "step_verified": step_ok,
        "goal_satisfied": goal_done
    }


# ─────────────────────────────────────────────────────────────────────────────
# Existing single-step nodes (unchanged logic, but now feed off current_step)
# ─────────────────────────────────────────────────────────────────────────────

def parse_request_node(state: AgentState) -> AgentState:
    """
    Node 1: Rule-based parser. Uses state['request'] which is set to
    current_step during multi-step runs (by plan_node).
    """
    raw = state["request"]
    parsed = parse_request(raw)
    print(f"[Parser] Target: '{parsed['target']}' | Hint: '{parsed['likely_category']}'")
    return {
        "parsed_target": parsed["target"],
        "likely_category": parsed["likely_category"]
    }


def categorize_request_node(state: AgentState) -> AgentState:
    """
    Node 2: Trust the parser if confident; otherwise ask Laya.
    """
    raw = state["request"]
    parser_hint = state.get("likely_category", "unknown")

    if parser_hint != "unknown":
        print(f"[Router] Parser confident: '{parser_hint}' (Laya skipped)")
        return {"category": parser_hint}

    print(f"[Router] Ambiguous → Laya: '{raw}'")
    category = categorize_with_laya(raw)
    print(f"[Laya] category='{category}'")
    return {"category": category}


def browser_tool_node(state: AgentState) -> AgentState:
    """Node 3A: Deterministic URL and search detection."""
    target = state.get("parsed_target", state["request"])
    lower = target.lower()
    print("[Node] Routing browser tool…")
    
    is_url = target.startswith("http") or ("." in target and " " not in target.strip())
    if is_url:
        url = target if target.startswith("http") else f"https://{target}"
        tool_name, args = "open_url", {"url": url}
    else:
        tool_name, args = "search_browser", {"query": target}
    print(f"[Browser] tool='{tool_name}' | args={args}")
    return {"selected_tool": tool_name, "tool_arguments": args}


def application_tool_node(state: AgentState) -> AgentState:
    """Node 3B: Laya selects exact app tool; always passes clean app name."""
    target = state.get("parsed_target", state["request"])
    print("[Node] Routing application tool…")
    tool_name = select_tool_with_laya(target, "application")
    print(f"[App] tool='{tool_name}'")
    return {"selected_tool": tool_name, "tool_arguments": {"app_name": target}}


def _resolve_interaction(raw: str):
    """Deterministically resolve interaction command → (tool_name, args)."""
    lower = raw.lower().strip()

    KEY_NAMES = {
        "ctrl", "control", "alt", "shift", "win", "windows", "super", "cmd",
        "enter", "return", "esc", "escape", "tab", "space", "backspace", "delete",
        "up", "down", "left", "right", "home", "end", "pageup", "pagedown",
        "f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12",
        "capslock", "numlock", "scrolllock", "printscreen", "insert"
    }

    def extract_keys(text: str) -> list[str]:
        text = re.sub(r"[\+\-]", " ", text)
        tokens = re.findall(r"[a-z0-9]+", text)
        return [t for t in tokens if t in KEY_NAMES or re.match(r"^[a-z]$", t)]

    if re.search(r"double.?click", lower): return "double_click", {}
    if re.search(r"right.?click", lower):  return "right_click", {}
    if "screenshot" in lower: return "take_screenshot", {}
    if re.search(r"move.*(mouse|cursor)", lower): return "move_mouse", {"x": 500, "y": 500}
    if "scroll" in lower:
        return "press_key", {"key": "down" if "down" in lower else "up"}
    if "click" in lower: return "click", {}
    if re.match(r"(type|write|input)\s+", lower):
        text_to_type = re.sub(r"^(type|write|input)\s+", "", lower).strip()
        return "type_text", {"text": text_to_type}
    if re.search(r"(press|hold|use|hit)", lower):
        cleaned = re.sub(r"^(please\s+)?(can you\s+)?(press|hold|use|hit)\s+", "", lower).strip()
        keys = extract_keys(cleaned)
        if len(keys) > 1: return "press_hotkey", {"keys": keys}
        if len(keys) == 1: return "press_key", {"key": keys[0]}
    if re.search(r"(switch|alt.?tab|tab through)", lower):
        return "press_hotkey", {"keys": ["alt", "tab"]}
    keys = extract_keys(lower)
    if len(keys) > 1: return "press_hotkey", {"keys": keys}
    if len(keys) == 1: return "press_key", {"key": keys[0]}
    return "move_mouse", {"x": 500, "y": 500}


def interaction_tool_node(state: AgentState) -> AgentState:
    """Node 3C: Deterministic interaction resolver."""
    raw = state["request"]
    print("[Node] Resolving interaction tool…")
    tool_name, args = _resolve_interaction(raw)
    print(f"[Interaction] tool='{tool_name}' | args={args}")
    return {"selected_tool": tool_name, "tool_arguments": args}


def file_explorer_tool_node(state: AgentState) -> AgentState:
    """Node 3D: Deterministic and Laya routing for file explorer tools."""
    raw = state["request"]
    target = state.get("parsed_target", raw)
    lower = raw.lower().strip()
    print("[Node] Routing file explorer tool…")

    import os
    path_match = re.search(r'([a-zA-Z]:\\[^\s"\']+|"[^"]+"|\'[^\']+\')', raw)
    extracted_path = path_match.group(1).strip('"\'') if path_match else target

    if "organi" in lower or "group" in lower:
        return {"selected_tool": "organize_files", "tool_arguments": {"directory": extracted_path}}
    elif "open" in lower and ("folder" in lower or "dir" in lower):
        return {"selected_tool": "open_folder", "tool_arguments": {"directory": extracted_path}}
    elif "search" in lower or "find" in lower:
        return {"selected_tool": "search_files", "tool_arguments": {"directory": extracted_path, "name": target}}
    elif "list" in lower:
        return {"selected_tool": "list_directory", "tool_arguments": {"directory": extracted_path}}
    elif "create" in lower and "folder" in lower:
        return {"selected_tool": "create_folder", "tool_arguments": {"directory": extracted_path, "folder_name": "New Folder"}}

    if os.path.exists(extracted_path) and os.path.isdir(extracted_path):
        return {"selected_tool": "organize_files", "tool_arguments": {"directory": extracted_path}}

    tool_name = select_tool_with_laya(raw, "file_explorer")
    return {"selected_tool": tool_name, "tool_arguments": {"directory": extracted_path}}


def execute_tool_node(state: AgentState) -> AgentState:
    """Node 4: Execute the chosen tool with the resolved arguments."""
    from src.tools.browser import open_url, search_browser
    from src.tools.browser.email_tools import get_unread_emails, send_email, wait_for_new_email, get_upcoming_events, create_event
    from src.tools.application import open_app, open_application_cmd, open_application_gui
    from src.tools.interaction import (
        type_text, press_key, press_hotkey, move_mouse,
        click, double_click, right_click, get_mouse_position, take_screenshot
    )
    from src.tools.file_explorer import (
        open_folder, organize_files, search_files, move_file, copy_file,
        rename_file, list_directory, get_all_file_names, create_folder,
        get_file_metadata, delete_item
    )

    def _press_hotkey_wrapper(**kwargs):
        if "keys" in kwargs:
            return press_hotkey(keys=kwargs["keys"])
        hotkey_str = kwargs.get("hotkey", "")
        keys = re.split(r'[\s\+]+', hotkey_str)
        keys = [k for k in keys if k]
        return press_hotkey(keys=keys)

    TOOL_MAPPING = {
        "open_url": open_url, "search_browser": search_browser,
        "get_unread_emails": get_unread_emails, "send_email": send_email,
        "wait_for_new_email": wait_for_new_email, "get_upcoming_events": get_upcoming_events,
        "create_event": create_event,
        "open_app": open_app, "open_application_gui": open_application_gui,
        "open_application_cmd": open_application_cmd,
        "type_text": type_text, "press_key": press_key, "press_hotkey": _press_hotkey_wrapper,
        "move_mouse": move_mouse, "click": click, "double_click": double_click,
        "right_click": right_click, "get_mouse_position": get_mouse_position,
        "take_screenshot": take_screenshot,
        "open_folder": open_folder, "organize_files": organize_files,
        "search_files": search_files, "move_file": move_file,
        "copy_file": copy_file, "rename_file": rename_file,
        "list_directory": list_directory, "get_all_file_names": get_all_file_names,
        "create_folder": create_folder, "get_file_metadata": get_file_metadata,
        "delete_item": delete_item
    }

    tool_name = state.get("selected_tool")
    args = state.get("tool_arguments")
    if args is None:
        args = {}
    print(f"[Execute] '{tool_name}' args={args}")

    if tool_name in TOOL_MAPPING:
        try:
            result = TOOL_MAPPING[tool_name](**args)
            msg = f"OK: {result}"
            
            # Dynamic wait times based on the action performed
            import time
            if tool_name in ["open_url", "search_browser"]:
                print(f"[Execute] Waiting for browser to load (4s)...")
                time.sleep(4.0)
            elif tool_name in ["open_app", "open_application_gui", "open_application_cmd", "open_folder"]:
                print(f"[Execute] Waiting for application to open (5s)...")
                time.sleep(5.0)
            elif tool_name in ["organize_files", "move_file", "copy_file", "create_folder"]:
                print(f"[Execute] Short wait after file operation (1s)...")
                time.sleep(1.0)
            elif tool_name in ["type_text", "press_key", "press_hotkey"]:
                print(f"[Execute] Short wait after keyboard action (0.5s)...")
                time.sleep(0.5)
            elif tool_name in ["move_mouse", "click", "double_click", "right_click"]:
                print(f"[Execute] Short wait after mouse action (0.3s)...")
                time.sleep(0.3)
                
        except Exception as e:
            msg = f"ERROR: {e}"
    else:
        msg = f"Tool '{tool_name}' not found."

    print(f"[Execute] Result: {msg}")
    return {"tool_result": msg}

def email_triage_node(state: AgentState) -> AgentState:
    """Specialized node to evaluate email results and automatically reply or schedule."""
    email_text = state.get("tool_result", "")
    if not isinstance(email_text, str) or not email_text:
        return state

    print(f"\n[Email Triage] Analyzing email for action...")
    
    import ollama
    import json
    import re
    from src.agent.planner import OLLAMA_MODEL
    from src.agent.router.brain import laya_router, CONFIDENCE_THRESHOLD
    
    # 1. Decision using Laya
    decision = "IGNORE"
    if laya_router:
        questions = {
            "triage": {
                "type": "choice",
                "instructions": "Determine the action required for this email.",
                "criteria": {
                    "SCHEDULE": "The email provides a specific time and date for a meeting or event that should be scheduled. (Do not use this if they are just asking if you are free).",
                    "REPLY": "The email asks a question, proposes an idea, or explicitly requests a response. Use this if they ask if you are free.",
                    "IGNORE": "The email is an FYI, newsletter, spam, or does not require any response."
                }
            }
        }
        try:
            result = laya_router.predict(email_text, questions)
            laya_decision = result["answers"]["triage"]["choice"].upper()
            confidence = result["answers"]["triage"].get("confidence", 1.0)
            print(f"[Email Triage/Laya] Decision: {laya_decision} (confidence: {confidence:.2f})")
            
            if confidence >= CONFIDENCE_THRESHOLD:
                decision = laya_decision
            else:
                print(f"[Email Triage] Low confidence ({confidence:.2f} < {CONFIDENCE_THRESHOLD}) → SLM fallback for triage")
                laya_decision = None # trigger fallback
        except Exception as e:
            print(f"[Email Triage/Laya] Error: {e}")
            laya_decision = None
            
    if not laya_router or not laya_decision:
        # Fallback if Laya isn't loaded or confidence is too low
        prompt = f"Email content:\n{email_text}\n\nDoes this email provide a concrete time to schedule an event (SCHEDULE), ask a question that needs an answer (REPLY), or require no action (IGNORE)? Reply with exactly one of these words: SCHEDULE, REPLY, IGNORE."
        try:
            response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}], options={"temperature": 0.0})
            decision = response.get("message", {}).get("content", "").strip().upper()
            print(f"[Email Triage/Ollama] Decision: {decision}")
        except Exception as e:
            pass

    def _extract_json(text):
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m: return json.loads(m.group(0))
        return {}

    if "SCHEDULE" in decision:
        print("[Email Triage] Extracting event details with SLM...")
        import datetime
        now = datetime.datetime.now().isoformat()
        prompt = f"Extract event details from this email:\n{email_text}\nReturn ONLY a JSON object with keys: summary, description, start_time_iso, end_time_iso. The current date and time is {now}. Try to calculate the correct future date in format YYYY-MM-DDTHH:MM:SSZ based on the current date."
        try:
            res = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}], options={"temperature": 0.0})
            content = res.get("message", {}).get("content", "{}")
            print(f"[Email Triage] SLM Output: {content.strip()}")
            data = _extract_json(content)
            from src.tools.browser.email_tools import create_event
            summary = data.get("summary", "New Event")
            print(f"[Email Triage] Executing create_event for '{summary}'...")
            result = create_event(
                summary=summary,
                description=data.get("description", ""),
                start_time_iso=data.get("start_time_iso", "2026-10-06T09:00:00Z"),
                end_time_iso=data.get("end_time_iso", "2026-10-06T10:00:00Z")
            )
            action_msg = f"\n\n[Triage Action] Scheduled Event: {result}"
            print(action_msg.strip())
            state["tool_result"] += action_msg
        except Exception as e:
            action_msg = f"\n\n[Triage Action] Failed to schedule: {e}"
            print(action_msg.strip())
            state["tool_result"] += action_msg

    elif "REPLY" in decision:
        print("[Email Triage] Extracting reply details with SLM...")
        prompt = f"Extract the sender's email address and generate a short polite reply for this email:\n{email_text}\nReturn ONLY a JSON object with keys: to, subject, body."
        try:
            res = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}], options={"temperature": 0.0})
            content = res.get("message", {}).get("content", "{}")
            print(f"[Email Triage] SLM Output: {content.strip()}")
            data = _extract_json(content)
            from src.tools.browser.email_tools import send_email
            to = data.get("to", "unknown@example.com")
            print(f"[Email Triage] Executing send_email to '{to}'...")
            result = send_email(
                to=to,
                subject=data.get("subject", "Re:"),
                body=data.get("body", "Received.")
            )
            action_msg = f"\n\n[Triage Action] Sent Reply: {result}"
            print(action_msg.strip())
            state["tool_result"] += action_msg
        except Exception as e:
            action_msg = f"\n\n[Triage Action] Failed to reply: {e}"
            print(action_msg.strip())
            state["tool_result"] += action_msg

    return {"tool_result": state.get("tool_result", "")}
