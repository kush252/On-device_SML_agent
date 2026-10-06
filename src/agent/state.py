from typing import TypedDict, Optional, Dict, Any, List

class AgentState(TypedDict):
    """
    State passed through the agent graph for both single-step and multi-step tasks.
    """
    # --- Original request (the user's raw input, never changes) ---
    request: str

    # --- Multi-step: per-run identity ---
    goal: Optional[str]            # same as request — kept separately for clarity
    run_id: Optional[str]          # e.g. "20241004_012345"
    run_folder: Optional[str]      # absolute path to this run's storage folder

    # --- Multi-step: loop control ---
    current_step: Optional[str]    # the planner's output for this iteration
    step_count: int                # how many steps have been executed
    goal_satisfied: bool           # True → exit the loop
    step_verified: Optional[bool]  # True if the last step was verified OK
    full_plan: Optional[List[dict]] # The complete list of planned steps (each step is a dict)

    # --- Single-step routing (set by parse / categorize nodes) ---
    parsed_target: Optional[str]
    likely_category: Optional[str]
    category: Optional[str]

    # --- Tool execution ---
    selected_tool: Optional[str]
    tool_arguments: Optional[Dict[str, Any]]
    tool_result: Optional[Any]
