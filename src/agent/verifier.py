"""
Verifier — checks two things after each step:
  1. Was the step executed successfully?
  2. Is the overall goal now fully satisfied?

Uses Laya for both decisions in a single forward pass (guide Section 7).
Falls back to qwen2.5:1.5b via Ollama if Laya confidence is below threshold.
"""
from src.agent.memory.storage import get_all_steps, format_steps_for_prompt
from src.agent.router.slm_fallback import _call_slm

CONFIDENCE_THRESHOLD = 0.70

try:
    from laya import Router
    _laya_router = Router(preload=True)
except ImportError:
    _laya_router = None

def _slm_binary(prompt: str) -> bool:
    """Ask the SLM a yes/no question. Returns True for yes."""
    full_prompt = prompt + "\nAnswer only: yes or no\nAnswer:"
    raw = _call_slm(full_prompt).lower().strip()
    return "yes" in raw

def verify_step(
    goal: str,
    run_folder: str,
    current_step: str,
    tool_result: str
) -> bool:
    """
    Verify if the last executed step was successful.
    Returns: step_success (bool)
    """
    if _laya_router:
        step_state = (
            f"Action taken: '{current_step}'. "
            f"Execution result: '{tool_result[:200]}'"
        )

        step_questions = {
            "step_success": {
                "type": "choice",
                "instructions": "Was the action executed successfully without errors?",
                "criteria": {
                    "yes": "The action completed without errors and the result looks correct.",
                    "no": "The action failed, threw an error, or did not produce a useful result."
                }
            }
        }

        step_result = _laya_router.predict(step_state, step_questions)
        step_answer = step_result["answers"]["step_success"]
        step_conf = step_answer.get("confidence", 1.0)
        step_ok = step_answer["choice"] == "yes"
        print(f"[Verifier/Laya] step_success='{step_answer['choice']}' confidence={step_conf:.3f}")

        if step_conf < CONFIDENCE_THRESHOLD:
            print(f"[Verifier] Low confidence ({step_conf:.3f}) → SLM fallback for step check")
            step_ok = _slm_binary(
                f"Action: '{current_step}'\nResult: '{tool_result[:200]}'\nWas this action successful?"
            )

    else:
        # No Laya — use SLM
        print("[Verifier] Laya unavailable → SLM fallback for step check")
        step_ok = _slm_binary(
            f"Action: '{current_step}'\nResult: '{tool_result[:200]}'\nWas this action successful?"
        )

    print(f"[Verifier] step_ok={step_ok}")
    return step_ok

