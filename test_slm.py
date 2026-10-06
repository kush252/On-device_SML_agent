import ollama

prompt = """You are an AI assistant controlling a computer to complete a task step by step.

Here are examples of how to plan:

Example 1:
ORIGINAL GOAL: open notepad and type hello world
COMPLETED STEPS SO FAR:
  (none yet)
NEXT ACTION: open notepad

Example 2:
ORIGINAL GOAL: open notepad and type hello world
COMPLETED STEPS SO FAR:
  1. [✓] open notepad → OK: True
NEXT ACTION: type hello world

Example 3:
ORIGINAL GOAL: open notepad and type hello world
COMPLETED STEPS SO FAR:
  1. [✓] open notepad → OK: True
  2. [✓] type hello world → OK: True
NEXT ACTION: GOAL_COMPLETE

Now, plan the next step for the current task:

ORIGINAL GOAL: search for the weather in Mumbai and then open youtube

COMPLETED STEPS SO FAR:
  (none yet)

YOUR JOB:
- Output the ONE single next action the robot should take RIGHT NOW.
- Write it as a short, simple robot command.
- Valid command formats:
    open <app name>
    search for <query>
    press <key or combo>
    type <text>
    click / double click
    move the mouse
    open <url>
- Do NOT repeat an action if it is already completed successfully.
- If the goal is already fully satisfied by the completed steps, output exactly: GOAL_COMPLETE
- Output ONLY the command or GOAL_COMPLETE. Nothing else. No explanation.

NEXT ACTION:"""

res = ollama.chat(model="qwen2.5:1.5b", messages=[{"role": "user", "content": prompt}], options={"temperature": 0.1, "num_predict": 30})
print("TEST 1:", res["message"]["content"])
