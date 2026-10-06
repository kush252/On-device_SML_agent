# SLM Agent Model Evaluation Guide

This document defines how to install and run the three candidate 1.5B agent/tool-use models for the Laya fallback layer:

1. **Hammer2.1-1.5B**
2. **Manthan-1.5B**
3. **Arch-Agent-1.5B**

The models should **not** be forced into one common prompt format. Each model was trained with a different objective and tool-call format, so the evaluation harness should give each model its native/recommended prompting and inference interface.

The purpose of this experiment is to determine which model is best for the project's fallback agent, while allowing every candidate to operate in the way for which it was trained.

---

## 1. Target Architecture

The intended production architecture is:

```text
User
  |
  v
Parser
  |
  v
Laya
  |
  +---- high-confidence route ----> deterministic tool/domain execution
  |
  +---- low-confidence/ambiguous --> selected SLM agent
                                      |
                                      v
                                  Tool call
                                      |
                                      v
                                  Executor
                                      |
                                      v
                                  Observation
                                      |
                                      v
                                  SLM agent
                                      |
                                      +---- next tool call
                                      |
                                      +---- DONE
```

The SLM is therefore not simply a text generator.

Its useful capability is:

```text
Goal
  -> choose an action/tool
  -> receive observation
  -> decide what to do next
  -> repeat until the goal is satisfied
```

The benchmark should preserve this behavior.

---

# 2. Why the Three Models Are Different

## 2.1 Hammer2.1-1.5B

Model:

`MadeAgents/Hammer2.1-1.5b`

Hammer is a lightweight Large Action Model family designed specifically around function calling and on-device agentic applications.

Hammer2.1 supports function calling through its Hugging Face Transformers chat template and through vLLM. Its repository also provides a Hammer client and an OpenAI-compatible function-calling interface.

The Hammer project describes its core approach as **function masking**, intended to improve function selection and reduce inappropriate tool selection. Hammer models are based on the Qwen 2.5 coder family.

The Hammer repository provides training/evaluation infrastructure using function-calling data including:

- `Salesforce/xlam-function-calling-60k`
- `MadeAgents/xlam-irrelevance-7.5k`

The repository also evaluates function calling on datasets such as BFCL, API-Bank, Tool-Alpaca, Nexus Raven and Seal-Tools.

Important implication:

**Hammer should be treated primarily as a function-calling/action model.**

Do not force it into a generic "tell me the next step in natural language" interface.

Its native interface is:

```text
user request
+
available tools
+
conversation/tool observations
        |
        v
function call
```

For multi-step tasks, continue the conversation by feeding the tool result back as a `tool` message.

Example conceptual interaction:

```text
USER:
Open Notepad and type hello world.

TOOLS:
open_app(...)
type_text(...)

HAMMER:
open_app({"app_name": "notepad"})

TOOL:
Notepad opened successfully.

HAMMER:
type_text({"text": "hello world"})

TOOL:
Text entered successfully.
```

This is the preferred Hammer evaluation mode.

Source:
https://huggingface.co/MadeAgents/Hammer2.1-1.5b

Repository:
https://github.com/MadeAgents/Hammer

---

# 3. Hammer2.1-1.5B Installation

## Option A — Transformers

Recommended when benchmarking Hammer's native Hugging Face tool-calling behavior.

Install:

```bash
pip install -U "transformers>=4.47.0" torch
```

Basic loading:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "MadeAgents/Hammer2.1-1.5b"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

Hammer's chat template supports tools.

Conceptual usage:

```python
messages = [
    {
        "role": "user",
        "content": "Open Notepad and type hello world."
    }
]

tools = [
    {
        "name": "open_app",
        "description": "Open a desktop application.",
        "parameters": {
            "type": "object",
            "properties": {
                "app_name": {
                    "type": "string",
                    "description": "Name of the application."
                }
            },
            "required": ["app_name"]
        }
    },
    {
        "name": "type_text",
        "description": "Type text into the currently focused application.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to type."
                }
            },
            "required": ["text"]
        }
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    tools=tools,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)

inputs = {
    key: value.to(model.device)
    for key, value in inputs.items()
}

output = model.generate(
    **inputs,
    max_new_tokens=128
)

print(
    tokenizer.decode(
        output[0][len(inputs["input_ids"][0]):],
        skip_special_tokens=True
    )
)
```

Do not replace the Hammer tool-calling template with an arbitrary `NEXT STEP:` prompt during the primary benchmark.

---

## Option B — vLLM

Hammer's official repository recommends vLLM for efficient serving.

Install:

```bash
pip install vllm
```

Serve:

```bash
vllm serve MadeAgents/Hammer2.1-1.5b \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

Then use an OpenAI-compatible client.

This is preferable if the benchmark needs repeated calls and latency measurements.

---

## Option C — Ollama

Ollama can be used for GGUF versions, but the Hammer repository has documented issues around tool support in Ollama.

Therefore:

**Do not make Ollama the reference implementation for Hammer if the goal is to evaluate Hammer's actual function-calling capability.**

For Hammer, use Transformers or vLLM first.

If an Ollama/GGUF build is tested, record it as a separate deployment variant rather than treating it as identical to the official Transformers/vLLM implementation.

---

# 4. Hammer Prompting Rules

Use the native tool interface.

Good:

```text
USER:
Open Notepad and type hello world.

TOOLS:
open_app(...)
type_text(...)
```

Then execute the generated tool call and provide the result.

Bad:

```text
You are a planner.
Always output:

NEXT STEP:
...
```

The second format discards the function-calling interface that Hammer was designed around.

For Hammer:

```text
Goal
+
tools
+
previous messages
+
tool observations
```

should form the model context.

---

# 5. Manthan-1.5B

Model:

`Shahansha/Manthan-1.5B`

Manthan is substantially different from Hammer.

The model is based on:

```text
Qwen2.5-1.5B-Instruct
```

and its stated training recipe is:

```text
QLoRA SFT
    ->
GRPO with tool-execution rewards
    ->
budget forcing during inference
```

The training data includes:

- `Shahansha/manthan-tool-reasoning-v1`
- Glaive function-calling traces
- Hermes function-calling traces

The model is explicitly designed around **tool-mediated reasoning**.

Its intended behavior is:

```text
problem
  ->
tool call
  ->
tool result
  ->
next action
  ->
...
  ->
final answer
```

The model card explicitly recommends the system instruction:

```text
You are Genesis Manthan, an AI agent that solves problems by calling tools.
Never reason verbally - always reason through tool execution.
```

Manthan uses structured `<tool_call>` blocks and ChatML/custom tool roles.

Therefore, Manthan should be tested as an **action-first agent**, not as a generic chat model.

Source:

https://huggingface.co/Shahansha/Manthan-1.5B

Dataset:

https://huggingface.co/datasets/Shahansha/manthan-tool-reasoning-v1

---

# 6. Manthan Installation

## Option A — Transformers

Install:

```bash
pip install -U transformers torch
```

Load:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "Shahansha/Manthan-1.5B"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="auto"
)
```

Recommended system message:

```text
You are Genesis Manthan, an AI agent that solves problems by calling tools.
Never reason verbally - always reason through tool execution.
```

Example:

```python
messages = [
    {
        "role": "system",
        "content": (
            "You are Genesis Manthan, an AI agent that solves problems "
            "by calling tools. Never reason verbally - always reason "
            "through tool execution."
        )
    },
    {
        "role": "user",
        "content": "Open Notepad and type hello world."
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.2
)

result = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=False
)

print(result)
```

The expected output should contain a structured tool call.

---

## Manthan's important inference characteristic

The model card explicitly states that Manthan performs best when the system prompt tells it to solve the task through tools.

Therefore retain the system instruction.

Do not reduce the prompt to:

```text
What is the next step?
```

Instead expose the tool environment and make tool execution the mechanism through which the agent progresses.

---

# 7. Manthan Tool Loop

The preferred loop is:

```text
SYSTEM:
You are Genesis Manthan...

USER:
Open Notepad and type hello world.

MANthan:
<tool_call>
...
</tool_call>

EXECUTOR:
Notepad opened successfully.

MANthan:
<tool_call>
...
</tool_call>

EXECUTOR:
Text entered successfully.

MANthan:
final answer / completion
```

The executor should append each tool result to the conversation.

Do not call Manthan independently for every step with only the original user request.

It needs the interaction history.

---

# 8. Manthan Quantized Local Inference

The model card provides GGUF/llama.cpp support.

For llama.cpp:

```bash
llama serve -hf Shahansha/Manthan-1.5B:Q4_K_M
```

or:

```bash
llama cli -hf Shahansha/Manthan-1.5B:Q4_K_M
```

For Windows, the model card also documents installation through WinGet:

```bash
winget install llama.cpp
```

The model card additionally documents Docker, vLLM and other local deployment methods.

For this project, the recommended initial reference implementation is still Transformers because it gives direct access to the model's intended chat/tool behavior.

---

# 9. Arch-Agent-1.5B

Model:

`katanemo/Arch-Agent-1.5B`

Arch-Agent is designed around function calling and multi-step/multi-turn agent interactions.

Its most important characteristic for this benchmark is that the authors explicitly say:

**The model works best with their provided prompt format.**

The prompt uses:

```text
# Tools

<tools>
...
</tools>

<tool_call>
{
    "name": <function-name>,
    "arguments": <args-json-object>
}
</tool_call>
```

Therefore Arch-Agent must be evaluated using this prompt format.

Do not replace it with the Hammer or Manthan format.

Source:

https://huggingface.co/katanemo/Arch-Agent-1.5B

GGUF:

https://huggingface.co/katanemo/Arch-Agent-1.5B.gguf

---

# 10. Arch-Agent Installation

Install:

```bash
pip install -U "transformers>=4.51.0" torch
```

Load:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "katanemo/Arch-Agent-1.5B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
```

---

# 11. Arch-Agent Prompt Format

Use the model's documented structure.

The core system/task prompt should follow this pattern:

```text
You are a helpful assistant designed to assist with the user query by making one or more function calls if needed.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:

<tools>
{tool definitions}
</tools>

For each function call, return a JSON object with function name and arguments within <tool_call></tool_call> XML tags:

<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
```

Then insert the actual tool definitions.

For example:

```text
<tools>

{
  "name": "open_app",
  "description": "Open a desktop application.",
  "parameters": {
    "type": "object",
    "properties": {
      "app_name": {
        "type": "string"
      }
    },
    "required": ["app_name"]
  }
}

{
  "name": "type_text",
  "description": "Type text into the focused application.",
  "parameters": {
    "type": "object",
    "properties": {
      "text": {
        "type": "string"
      }
    },
    "required": ["text"]
  }
}

</tools>
```

Then:

```text
User:
Open Notepad and type hello world.
```

The model should produce:

```text
<tool_call>
{"name": "open_app", "arguments": {"app_name": "notepad"}}
</tool_call>
```

After execution, provide the tool result and continue the conversation.

---

# 12. Arch-Agent Multi-Step Usage

The correct interaction is:

```text
USER:
Open Notepad and type hello world.

ARCH:
<tool_call>
{"name":"open_app","arguments":{"app_name":"notepad"}}
</tool_call>

TOOL:
Notepad opened successfully.

ARCH:
<tool_call>
{"name":"type_text","arguments":{"text":"hello world"}}
</tool_call>

TOOL:
Text entered successfully.

ARCH:
...
```

The important part is that the second model call contains the updated conversation/tool result.

Do not repeatedly ask:

```text
Open Notepad and type hello world.
```

with a fresh context.

That removes the multi-turn state that an agent model needs.

---

# 13. Arch-Agent GGUF

The official GGUF repository is:

```text
katanemo/Arch-Agent-1.5B.gguf
```

It can be used with local GGUF runtimes such as llama.cpp.

For the benchmark, record whether the model is being tested through:

```text
Transformers
```

or:

```text
GGUF / llama.cpp
```

because deployment format can affect behavior.

Do not assume that a GGUF runtime automatically reproduces every feature of the Transformers implementation.

---

# 14. Common Tool Schema

The three models should ultimately receive equivalent semantic capabilities, but the serialization should remain model-specific.

Use a common internal representation:

```python
TOOLS = {
    "open_app": {
        "description": "Open a desktop application.",
        "parameters": {
            "app_name": "string"
        }
    },

    "close_app": {
        "description": "Close a desktop application.",
        "parameters": {
            "app_name": "string"
        }
    },

    "type_text": {
        "description": "Type text into the currently focused application.",
        "parameters": {
            "text": "string"
        }
    },

    "left_click": {
        "description": "Perform a left mouse click.",
        "parameters": {}
    },

    "right_click": {
        "description": "Perform a right mouse click.",
        "parameters": {}
    },

    "move_mouse": {
        "description": "Move the mouse.",
        "parameters": {
            "direction": "string"
        }
    },

    "hotkey": {
        "description": "Press a keyboard shortcut.",
        "parameters": {
            "keys": "string"
        }
    },

    "search_browser": {
        "description": "Search the browser.",
        "parameters": {
            "query": "string"
        }
    }
}
```

Then create adapters:

```text
internal tool schema
        |
        +--> Hammer formatter
        |
        +--> Manthan formatter
        |
        +--> Arch-Agent formatter
```

This is the correct way to compare them.

The semantic tool environment stays constant.

The model-specific representation changes.

---

# 15. What the Evaluation Harness Should Do

The harness should NOT force a common prompt.

Instead:

```text
                SAME TASK
                    |
        ┌───────────┼───────────┐
        v           v           v
     Hammer      Manthan      Arch
        |           |           |
   native API   native API   native API
   + prompt     + prompt     + prompt
        |           |           |
        v           v           v
     output      output       output
        |           |           |
        └───────────┼───────────┘
                    v
             common evaluator
```

The evaluator only compares the resulting action.

For example:

```text
Task:
Open Notepad and type hello world.

Hammer:
open_app("notepad")
type_text("hello world")

Manthan:
open_app("notepad")
type_text("hello world")

Arch:
open_app("notepad")
type_text("hello world")
```

All three are correct despite using different internal prompts/output formats.

---

# 16. Evaluation Dimensions

Evaluate the models on the capabilities relevant to the actual agent.

## A. Tool selection

Can it identify:

```text
left click
```

as:

```text
left_click()
```

rather than:

```text
open_app()
```

---

## B. Argument extraction

Can it transform:

```text
move the mouse to the left
```

into:

```json
{
    "direction": "left"
}
```

---

## C. Multi-step execution

Can it correctly handle:

```text
Open Notepad and type hello world.
```

as:

```text
open_app("notepad")
    ->
type_text("hello world")
```

---

## D. Observation-driven continuation

After:

```text
open_app("notepad")
```

and:

```text
SUCCESS: Notepad is now open and focused.
```

does it correctly choose:

```text
type_text("hello world")
```

rather than repeating `open_app`?

---

## E. Error recovery

Given:

```text
open_app("notepad")
```

and:

```text
ERROR: application was not found.
```

does it adapt instead of blindly repeating the same action?

---

## F. Goal completion

After:

```text
type_text("hello world")
```

and:

```text
SUCCESS: text entered.
```

does it stop instead of generating another unnecessary action?

---

# 17. Important Difference Between "Tool Success" and "Goal Success"

The executor returning:

```text
OK: True
```

does NOT mean that the agent made the correct decision.

Example:

```text
Goal:
Open Notepad and type hello.

Agent:
type_text("hello")
```

The computer may successfully execute the typing operation.

But the agent failed because Notepad was not opened first.

Therefore the benchmark needs:

```text
tool execution success
```

and:

```text
goal progress / goal satisfaction
```

as separate concepts.

---

# 18. Recommended Model-Specific Adapters

Create:

```text
models/
    hammer.py
    manthan.py
    arch.py
```

Each adapter should expose the same internal interface:

```python
class AgentModel:

    def generate_action(
        self,
        goal,
        conversation,
        tools
    ):
        ...
```

Internally:

```text
HammerAdapter
    -> Hammer-native tool calling

ManthanAdapter
    -> Manthan ChatML + <tool_call>

ArchAdapter
    -> Arch-Agent <tools> + <tool_call>
```

The rest of the benchmark does not care how the model was prompted.

---

# 19. Do Not Normalize the Prompts

Avoid this:

```text
All models receive:

You are an autonomous agent.
Return exactly:

ACTION:
tool
arguments
```

That would test:

```text
"Which model obeys our artificial format?"
```

instead of:

```text
"Which model is best at the capability it was trained for?"
```

The correct benchmark is:

```text
Hammer:
native Hammer function calling

Manthan:
native Manthan tool reasoning

Arch:
native Arch-Agent function calling
```

Then evaluate their actual actions.

---

# 20. Final Selection Criteria

The winning model should not simply be the one with the highest raw function-call accuracy.

For the Laya fallback layer, prioritize:

```text
1. Correct first action
2. Correct tool selection
3. Correct arguments
4. Multi-step continuation
5. Observation handling
6. Recovery from failed tools
7. Correct termination
8. Latency
9. Memory usage
```

A model that is slightly worse at single-turn function calling but substantially better at:

```text
goal
 -> tool
 -> observation
 -> next tool
 -> observation
 -> done
```

may be the better model for this project.

---

# 21. Recommended Testing Strategy

Run three separate native evaluations.

### Hammer evaluation

```text
Hammer native tool calling
+
multi-turn tool observations
+
function-call output
```

### Manthan evaluation

```text
Manthan system instruction
+
tool-first behavior
+
<tool_call>
+
tool observations
```

### Arch-Agent evaluation

```text
Arch-Agent official prompt
+
<tools>
+
<tool_call>
+
multi-turn tool observations
```

Then compare their resulting action traces.

---

# 22. Expected Benchmark Output

The benchmark should produce something similar to:

```text
Model: Hammer2.1-1.5B

Single-turn:
    Tool selection:       92%
    Arguments:            94%

Multi-step:
    First action:         95%
    Sequence completion:  86%
    Observation handling: 83%
    Recovery:              79%

Average latency:          420 ms
```

```text
Model: Manthan-1.5B

Single-turn:
    Tool selection:       ...
    Arguments:            ...

Multi-step:
    First action:         ...
    Sequence completion:  ...
    Observation handling: ...
    Recovery:              ...

Average latency:           ...
```

```text
Model: Arch-Agent-1.5B

Single-turn:
    Tool selection:       ...
    Arguments:            ...

Multi-step:
    First action:         ...
    Sequence completion:  ...
    Observation handling: ...
    Recovery:              ...

Average latency:           ...
```

The numbers above are only an example of the reporting format.

---

# 23. Important Implementation Rule

Do not let the benchmark execute real desktop actions.

Initially use:

```text
model
  ->
tool call
  ->
simulated tool result
  ->
model
```

Only after selecting the best model should it be connected to:

```text
PyAutoGUI
keyboard
mouse
application launcher
browser
filesystem
etc.
```

This makes experimentation safe and reproducible.

---

# 24. Sources

Hammer2.1:

https://huggingface.co/MadeAgents/Hammer2.1-1.5b

Hammer repository:

https://github.com/MadeAgents/Hammer

Manthan:

https://huggingface.co/Shahansha/Manthan-1.5B

Manthan dataset:

https://huggingface.co/datasets/Shahansha/manthan-tool-reasoning-v1

Arch-Agent:

https://huggingface.co/katanemo/Arch-Agent-1.5B

Arch-Agent GGUF:

https://huggingface.co/katanemo/Arch-Agent-1.5B.gguf

---

# 25. Bottom Line

The models should be tested according to their training assumptions:

```text
Hammer2.1
    -> function-calling/action model
    -> native tool schema
    -> multi-turn tool results
    -> function-call output

Manthan
    -> tool-mediated reasoning model
    -> explicit tool-first system instruction
    -> structured <tool_call>
    -> tool-result feedback loop

Arch-Agent
    -> function-calling agent
    -> official <tools> prompt
    -> <tool_call> JSON
    -> multi-turn function interaction
```

The benchmark's job is therefore **not** to make the three models behave identically.

Its job is to give each model the interface it was trained for, then compare the actual agent behavior produced by those interfaces.

Only after this evaluation should one model be selected as the SLM fallback behind Laya.
