import ollama

prompt = """You are a robotic task planner. Your job is to read the original goal and the completed steps, and output ONLY the next step.

Valid command formats:
- open <app name>
- search for <query>
- press <key or combo>
- type <text>
- click
- move the mouse
- open <url>

ORIGINAL GOAL: open calculator and press enter
COMPLETED STEPS: (none)
NEXT STEP: open calculator

ORIGINAL GOAL: open calculator and press enter
COMPLETED STEPS:
1. open calculator
NEXT STEP: press enter

ORIGINAL GOAL: open notepad and type hello world
COMPLETED STEPS: (none)
NEXT STEP:"""

response = ollama.generate(
    model="qwen2.5:1.5b",
    prompt=prompt,
    options={"temperature": 0.0, "num_predict": 20}
)
print("TEST SCRIPT OUTPUT:", response["response"].strip())


