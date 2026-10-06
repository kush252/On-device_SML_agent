from src.agent.graph import build_agent_graph

def main():
    print("Initializing Robot Brain (Loading Laya Router + SLM)…")
    graph = build_agent_graph()

    print("\n==============================================")
    print("🤖 Autonomous Daily Life Helper Robot Ready!")
    print("   Multi-step task execution enabled.")
    print("   Type 'exit' or 'quit' to stop.")
    print("==============================================\n")

    while True:
        try:
            request = input("You: ").strip()

            if request.lower() in ("exit", "quit"):
                print("Robot: Shutting down. Goodbye!")
                break
            if not request:
                continue

            print("Robot: Starting task…\n")
            initial_state = {
                "request": request,
                "step_count": 0,
                "goal_satisfied": False,
            }

            last_step_count = 0
            for output in graph.stream(initial_state):
                for node_name, state_update in output.items():

                    if node_name == "init_run":
                        run_id = state_update.get("run_id", "")
                        print(f"  📁 Run ID: {run_id}")

                    elif node_name == "plan":
                        step = state_update.get("current_step", "")
                        if step:
                            print(f"\n  🧠 Step planned: \"{step}\"")

                    elif node_name == "categorize":
                        cat = state_update.get("category", "")
                        if cat:
                            print(f"  ↳ Category: {cat}")

                    elif node_name == "execute_tool":
                        result = state_update.get("tool_result", "")
                        tool = state_update.get("selected_tool", "")
                        print(f"  ⚙️  Executed: {tool} → {result}")

                    elif node_name == "save_step":
                        count = state_update.get("step_count", 0)
                        last_step_count = count
                        print(f"  💾 Step {count} saved.")

                    elif node_name == "verify":
                        ok = state_update.get("step_verified")
                        done = state_update.get("goal_satisfied")
                        print(f"  ✅ Step verified: {ok} | Goal satisfied: {done}")

            print(f"\nRobot: Task complete after {last_step_count} step(s). What's next?\n")

        except KeyboardInterrupt:
            print("\nRobot: Shutting down. Goodbye!")
            break
        except Exception as e:
            print(f"\nRobot Error: {e}\n")

if __name__ == "__main__":
    main()
