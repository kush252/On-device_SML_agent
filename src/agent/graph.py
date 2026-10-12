from langgraph.graph import StateGraph, START, END
from .state import AgentState
from .nodes import (
    init_run_node,
    plan_node,
    parse_request_node,
    categorize_request_node,
    browser_tool_node,
    application_tool_node,
    interaction_tool_node,
    file_explorer_tool_node,
    execute_tool_node,
    save_step_node,
    verify_node,
    email_triage_node
)


# ─── Conditional edge: categorize → browser / application / interaction / file_explorer ───

def route_to_category(state: AgentState) -> str:
    category = state.get("category", "browser")
    if category == "browser":        return "browser_node"
    if category == "application":    return "application_node"
    if category == "interaction":    return "interaction_node"
    if category == "file_explorer":  return "file_explorer_node"
    return "browser_node"  # safe default


# ─── Conditional edge: verify → plan (loop) or END ────────────────────────────

def route_after_verify(state: AgentState) -> str:
    if state.get("goal_satisfied"):
        return END
    return "plan"


# ─── Conditional edge: plan → parse OR end (if planner declared GOAL_COMPLETE) ─

def route_after_plan(state: AgentState) -> str:
    if state.get("goal_satisfied"):
        return END
    # If planner provided an exact tool and arguments, skip straight to execution!
    if state.get("selected_tool"):
        return "execute_tool"
    return "parse"


def build_agent_graph() -> StateGraph:
    """
    Constructs the multi-step LangGraph pipeline.

    Flow:
        START → init_run → plan → parse → categorize
                  ↑                           ↓
                  │        browser / app / interaction / file_explorer node
                  │                           ↓
                  │                      execute_tool
                  │                           ↓
                  │                       save_step
                  │                           ↓
                  └──── (loop) ──────────── verify
                                              ↓
                                          goal done? → END
    """
    workflow = StateGraph(AgentState)

    # ── Nodes ──────────────────────────────────────────────────────────────
    workflow.add_node("init_run",            init_run_node)
    workflow.add_node("plan",                plan_node)
    workflow.add_node("parse",               parse_request_node)
    workflow.add_node("categorize",          categorize_request_node)
    workflow.add_node("browser_node",        browser_tool_node)
    workflow.add_node("application_node",    application_tool_node)
    workflow.add_node("interaction_node",    interaction_tool_node)
    workflow.add_node("file_explorer_node",  file_explorer_tool_node)
    workflow.add_node("execute_tool",        execute_tool_node)
    workflow.add_node("save_step",           save_step_node)
    workflow.add_node("verify",              verify_node)

    # ── Edges ───────────────────────────────────────────────────────────────
    workflow.add_edge(START, "init_run")
    workflow.add_edge("init_run", "plan")

    # After planning: exit if planner says GOAL_COMPLETE, else go to parse or execute
    workflow.add_conditional_edges(
        "plan", route_after_plan, {"parse": "parse", "execute_tool": "execute_tool", END: END}
    )

    workflow.add_edge("parse", "categorize")

    # Route to correct tool family
    workflow.add_conditional_edges(
        "categorize",
        route_to_category,
        {
            "browser_node":        "browser_node",
            "application_node":    "application_node",
            "interaction_node":    "interaction_node",
            "file_explorer_node":  "file_explorer_node"
        }
    )

    # All tool families → execute
    workflow.add_edge("browser_node",       "execute_tool")
    workflow.add_edge("application_node",   "execute_tool")
    workflow.add_edge("interaction_node",   "execute_tool")
    workflow.add_edge("file_explorer_node", "execute_tool")

    def route_after_execute(state: AgentState) -> str:
        if state.get("selected_tool") == "get_unread_emails":
            return "email_triage_node"
        return "save_step"

    # execute → triage OR save
    workflow.add_conditional_edges(
        "execute_tool",
        route_after_execute,
        {"email_triage_node": "email_triage_node", "save_step": "save_step"}
    )
    
    workflow.add_node("email_triage_node", email_triage_node)
    workflow.add_edge("email_triage_node", "save_step")
    workflow.add_edge("save_step",    "verify")

    # verify → loop back to plan OR exit
    workflow.add_conditional_edges(
        "verify",
        route_after_verify,
        {"plan": "plan", END: END}
    )

    return workflow.compile()
