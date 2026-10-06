"""
Run storage — persists task context per run.
Each run gets a timestamped folder under runs/.
Context sent to planners/verifiers is always pulled from disk
so the in-memory state stays lean.
"""
import os
import json
from datetime import datetime
from typing import List, Dict, Any

RUNS_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "runs")

def create_run(goal: str) -> tuple[str, str]:
    """Create a new run folder, save the goal. Returns (run_id, run_folder)."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.abspath(os.path.join(RUNS_BASE_DIR, run_id))
    os.makedirs(run_folder, exist_ok=True)

    with open(os.path.join(run_folder, "goal.txt"), "w", encoding="utf-8") as f:
        f.write(goal)

    with open(os.path.join(run_folder, "steps.json"), "w", encoding="utf-8") as f:
        json.dump([], f, indent=2)

    return run_id, run_folder

def append_step(run_folder: str, step_num: int, step_desc: str,
                tool: str, args: dict, result: str, verified: bool) -> None:
    """Append a completed step record to steps.json."""
    steps_file = os.path.join(run_folder, "steps.json")
    with open(steps_file, "r", encoding="utf-8") as f:
        steps = json.load(f)

    steps.append({
        "step_num": step_num,
        "step": step_desc,
        "tool": tool,
        "args": args,
        "result": result,
        "verified": verified,
        "timestamp": datetime.now().isoformat()
    })

    with open(steps_file, "w", encoding="utf-8") as f:
        json.dump(steps, f, indent=2, ensure_ascii=False)

def update_step_verified(run_folder: str, step_num: int, verified: bool) -> None:
    """
    Update the verified flag for a specific step number in steps.json.
    Called by verify_node after checking step success so the planner
    sees completed steps correctly instead of always seeing them as failed.
    """
    steps_file = os.path.join(run_folder, "steps.json")
    if not os.path.exists(steps_file):
        return
    with open(steps_file, "r", encoding="utf-8") as f:
        steps = json.load(f)
    for step in steps:
        if step["step_num"] == step_num:
            step["verified"] = verified
            break
    with open(steps_file, "w", encoding="utf-8") as f:
        json.dump(steps, f, indent=2, ensure_ascii=False)

def get_recent_steps(run_folder: str, n: int = 4) -> List[Dict[str, Any]]:
    """
    Load the last N steps from disk.
    Keeps planner/verifier context tight — we never dump the full history into a prompt.
    """
    steps_file = os.path.join(run_folder, "steps.json")
    if not os.path.exists(steps_file):
        return []
    with open(steps_file, "r", encoding="utf-8") as f:
        steps = json.load(f)
    return steps[-n:]

def get_all_steps(run_folder: str) -> List[Dict[str, Any]]:
    """Load all steps (used by verifier for goal-satisfaction check)."""
    steps_file = os.path.join(run_folder, "steps.json")
    if not os.path.exists(steps_file):
        return []
    with open(steps_file, "r", encoding="utf-8") as f:
        return json.load(f)

def format_steps_for_prompt(steps: List[Dict[str, Any]]) -> str:
    """Rich format for verifier/display. Shows verified status and result."""
    if not steps:
        return "  (none yet)"
    lines = []
    for s in steps:
        status = "✓" if s.get("verified") else "✗"
        lines.append(f"  {s['step_num']}. [{status}] {s['step']} → {s['result'][:80]}")
    return "\n".join(lines)

def format_steps_simple(steps: List[Dict[str, Any]]) -> str:
    """
    Simple numbered list for the planner prompt.
    Only shows VERIFIED (successful) steps so the model knows exactly what's done.
    Format matches the planner's few-shot examples exactly: '1. open notepad'
    """
    if not steps:
        return "(none)"
    done = [s for s in steps if s.get("verified")]
    if not done:
        return "(none)"
    return "\n".join(f"{i+1}. {s['step']}" for i, s in enumerate(done))
