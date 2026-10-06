import os
import json
try:
    import ollama
except ImportError:
    print("Please install ollama: pip install ollama")
    exit(1)

OUTPUT_DIR = "test_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# The models to test (make sure you pull or create these in Ollama first!)
# e.g., ollama run hammer2.1, ollama run manthan, ollama run arch-agent
MODELS = {
    "hammer": "hf.co/keisuke-miyako/Hammer2.1-1.5b-gguf-q4_k_m:Q4_K_M",
    "manthan": "hf.co/Shahansha/Manthan-1.5B:Q4_K_M",
    "arch": "hf.co/DevQuasar/katanemo.Arch-Agent-1.5B-GGUF:Q4_K_M"
}

REQUEST = "Open Notepad and type hello world."

# ---------------------------------------------------------------------------
# 1. Hammer 2.1 (Native Tool Calling via Ollama Chat)
# Hammer prefers the official tool-calling interface (OpenAI compatible).
# ---------------------------------------------------------------------------
def test_hammer():
    print(f"\n--- Testing Hammer ({MODELS['hammer']}) ---")
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "open_app",
                "description": "Open a desktop application.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "app_name": {"type": "string", "description": "Name of the application."}
                    },
                    "required": ["app_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "type_text",
                "description": "Type text into the currently focused application.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to type."}
                    },
                    "required": ["text"]
                }
            }
        }
    ]
    
    messages = [{"role": "user", "content": REQUEST}]
    
    try:
        response = ollama.chat(
            model=MODELS["hammer"],
            messages=messages,
            tools=tools,
            options={"temperature": 0.0}
        )
        
        output_msg = response.get("message", {})
        # Depending on ollama version, message might be a dict or a Message object
        if hasattr(output_msg, 'model_dump_json'):
            output = output_msg.model_dump_json(indent=2)
        elif hasattr(output_msg, 'dict'):
            output = json.dumps(output_msg.dict(), indent=2)
        else:
            output = str(output_msg)
            
        save_output("hammer", output)
        print("Success! Output saved.")
        
    except Exception as e:
        print(f"Hammer failed: {e}")
        save_output("hammer", f"ERROR:\n{e}")

# ---------------------------------------------------------------------------
# 2. Manthan (Genesis Manthan System Prompt)
# Manthan is explicitly trained with a specific system prompt and uses <tool_call> tags.
# ---------------------------------------------------------------------------
def test_manthan():
    print(f"\n--- Testing Manthan ({MODELS['manthan']}) ---")
    
    sys_prompt = (
        "You are Genesis Manthan, an AI agent that solves problems by calling tools. "
        "Never reason verbally - always reason through tool execution.\n\n"
        "Available Tools:\n"
        "- open_app(app_name: str)\n"
        "- type_text(text: str)\n"
    )
    
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": REQUEST}
    ]
    
    try:
        response = ollama.chat(
            model=MODELS["manthan"],
            messages=messages,
            options={"temperature": 0.0}
        )
        
        output = response["message"]["content"]
        save_output("manthan", output)
        print("Success! Output saved.")
        
    except Exception as e:
        print(f"Manthan failed: {e}")
        save_output("manthan", f"ERROR:\n{e}")

# ---------------------------------------------------------------------------
# 3. Arch-Agent (XML <tools> Prompt)
# Arch explicitly recommends providing tools in an XML block.
# ---------------------------------------------------------------------------
def test_arch():
    print(f"\n--- Testing Arch ({MODELS['arch']}) ---")
    
    prompt = f"""# Tools

<tools>
[
  {{
    "name": "open_app",
    "description": "Open a desktop application.",
    "parameters": {{"app_name": "string"}}
  }},
  {{
    "name": "type_text",
    "description": "Type text into the currently focused application.",
    "parameters": {{"text": "string"}}
  }}
]
</tools>

User: {REQUEST}
"""
    
    try:
        response = ollama.generate(
            model=MODELS["arch"],
            prompt=prompt,
            options={"temperature": 0.0}
        )
        
        output = response["response"]
        save_output("arch", output)
        print("Success! Output saved.")
        
    except Exception as e:
        print(f"Arch failed: {e}")
        save_output("arch", f"ERROR:\n{e}")


def save_output(model_name: str, content: str):
    filepath = os.path.join(OUTPUT_DIR, f"{model_name}_output.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    print(f"Starting SLM Scorer against task: '{REQUEST}'\n")
    test_hammer()
    test_manthan()
    test_arch()
    print(f"\nAll tests complete. Check the '{OUTPUT_DIR}' folder for results.")
