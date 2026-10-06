# Laya for Agentic Routing — General Guide

## 1. What Laya Is

Laya is a multilingual, non-autoregressive "System 1" decision model. Instead of generating a textual response, it receives:

1. A **state** — the information it should inspect.
2. One or more **typed questions** — the decisions you want it to make.

It returns structured answers and probabilities in a single forward pass.

This makes Laya useful as a fast decision layer in agentic systems:

```text
User / Environment
        |
        v
      Laya
        |
        +---- route
        +---- classify
        +---- score
        +---- gate
        |
        v
   Agent / Tool Graph
        |
        v
     Execution
```

Laya should generally be treated as a **decision and routing component**, not as the general-purpose reasoning model of an agent.

Official model:
https://huggingface.co/convaiinnovations/laya

---

## 2. Core Mental Model

The most important distinction is:

```text
LLM / SLM:
"What should I do and how should I do it?"

Laya:
"Given this state and these predefined alternatives, which decision is appropriate?"

Tools / Graph:
"Actually perform the selected operation."
```

For example, an agent may have many possible branches:

```text
request
   |
   v
 Laya
   |
   +--> branch A
   +--> branch B
   +--> branch C
   +--> branch D
```

The branch definitions are supplied at inference time. You do not need to retrain Laya every time you create a new set of choices.

---

# 3. Installation

Python 3.10+ is required.

Basic installation:

```bash
pip install laya
```

Optional extras:

```bash
pip install "laya[serve]"
pip install "laya[mcp]"
pip install "laya[langchain]"
pip install "laya[onnx]"
pip install "laya[fast]"
```

Use the extras only when the corresponding deployment/integration is needed.

---

# 4. The Basic API

The simplest workflow is:

```python
from laya import Router

router = Router()

state = "Your input/state goes here"

questions = {
    "decision": {
        "type": "choice",
        "instructions": "Which category does this belong to?",
        "criteria": {
            "option_a": "Description of option A",
            "option_b": "Description of option B",
            "option_c": "Description of option C",
        },
    }
}

result = router.predict(state, questions)
```

Then inspect:

```python
print(result)
```

The selected answer is available under:

```python
result["answers"]["decision"]["choice"]
```

The model also returns routing metadata under:

```python
result["routing"]
```

---

# 5. State: What You Give Laya

The `state` is the information Laya should use to make its decision.

It can be plain text:

```python
state = """
The application starts successfully but crashes when the user opens settings.
"""
```

Or structured data:

```python
state = {
    "user_request": "Schedule a meeting with Alex",
    "current_time": "10:30",
    "calendar_context": "...",
}
```

Or a document-like object:

```python
state = {
    "subject": "...",
    "body": "...",
    "metadata": {
        "source": "email"
    }
}
```

The important principle is:

> Put the evidence needed for the decision into the state.

Do not assume Laya will independently retrieve missing information from external tools.

If information is not available in the state, either:

1. obtain it before calling Laya, or
2. route to a component that can gather the missing information.

---

# 6. Typed Decisions

Laya supports different decision primitives.

The main ones relevant to routing are:

- `choice`
- `score`
- `noul`

## 6.1 Choice

Use `choice` when exactly one option should be selected.

```python
questions = {
    "route": {
        "type": "choice",
        "instructions": "Which route should handle this request?",
        "criteria": {
            "route_a": "Requests belonging to category A",
            "route_b": "Requests belonging to category B",
            "route_c": "Requests belonging to category C",
        },
    }
}
```

Read the result:

```python
route = result["answers"]["route"]["choice"]
```

Conceptually:

```text
state
  |
  v
choice question
  |
  +--> route_a
  +--> route_b
  +--> route_c
```

This is the primary primitive for agentic routing.

---

## 6.2 Score

Use `score` when the decision represents an ordered scale.

Example:

```python
questions = {
    "priority": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": [
            "low",
            "medium",
            "high",
        ],
    }
}
```

Read:

```python
priority = result["answers"]["priority"]["score"]
```

Use this for things such as:

- urgency
- severity
- risk
- priority
- confidence-like ordinal judgments

Do not use `score` merely because you want probabilities. If the actual requirement is selecting one route, use `choice`.

The model documentation notes that ordinal scoring is currently weaker than the other primitives, so validate score-based decisions on your own data.

---

## 6.3 Noul

`noul` is intended for binary yes/no-style decisions.

Example:

```python
questions = {
    "needs_confirmation": {
        "type": "noul",
        "instructions": "Does this action require confirmation?"
    }
}
```

Read:

```python
probability = result["answers"]["needs_confirmation"]["noul"]
```

The value represents the model's probability for the positive answer.

However, the model documentation identifies a limitation with the current English checkpoint: in some cases the `noul` option labels themselves can influence the result.

For critical binary decisions, validate `noul` on your own data. If it behaves poorly, use a two-option `choice` with neutral labels:

```python
questions = {
    "needs_confirmation": {
        "type": "choice",
        "instructions": "Does this action require confirmation?",
        "criteria": {
            "A": "yes, confirmation is required",
            "B": "no, confirmation is not required",
        },
    }
}
```

---

# 7. Multiple Decisions in One Call

One of Laya's important properties is that multiple questions can be evaluated in a single forward pass.

For example:

```python
questions = {
    "route": {
        "type": "choice",
        "instructions": "Which subsystem should handle this?",
        "criteria": {
            "system_a": "Description of subsystem A",
            "system_b": "Description of subsystem B",
            "system_c": "Description of subsystem C",
        },
    },

    "priority": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": [
            "low",
            "medium",
            "high",
        ],
    },

    "needs_confirmation": {
        "type": "noul",
        "instructions": "Does this action require confirmation?",
    },
}
```

Then:

```python
result = router.predict(state, questions)

route = result["answers"]["route"]["choice"]
priority = result["answers"]["priority"]["score"]
confirmation_probability = result["answers"]["needs_confirmation"]["noul"]
```

This allows one Laya invocation to produce several decisions.

A useful pattern is:

```text
                   STATE
                     |
                     v
                  LAYA
                     |
        +------------+------------+
        |            |            |
      route        priority    confirmation
        |            |            |
        v            v            v
      graph        policy       safety gate
```

---

# 8. Designing Good Routing Questions

Routing quality depends heavily on how the options are defined.

Bad:

```python
"criteria": {
    "a": "email",
    "b": "calendar",
    "c": "other"
}
```

Better:

```python
"criteria": {
    "email": "Requests involving reading, searching, composing, replying to, or managing email",
    "calendar": "Requests involving events, meetings, schedules, reminders, or appointments",
    "other": "Requests that do not belong to email or calendar operations",
}
```

The descriptions should explain the semantic meaning of each option.

A useful pattern is:

```text
option name
    +
clear description of what belongs there
    +
clear distinction from competing options
```

Do not write descriptions as implementation details unless implementation details are actually what determine the route.

---

# 9. Keep the Number of Options Reasonable

Do not expose a huge flat list of options to one `choice` question.

For example, this is usually a poor architecture:

```text
Laya
 |
 +-- tool_001
 +-- tool_002
 +-- tool_003
 +-- ...
 +-- tool_057
```

Prefer hierarchical routing:

```text
                 Laya
                  |
       +----------+----------+
       |          |          |
    domain A   domain B   domain C
       |          |          |
      Laya       Laya       Laya
       |          |          |
    actions     actions    actions
```

For example:

```text
Level 1:
Which domain?

    communication
    productivity
    knowledge
    device
    files

Level 2:
Which operation inside that domain?

    search
    read
    create
    update
    delete
```

This coarse-to-fine structure is also useful because the model documentation reports a significant degradation for very large option spaces.

For 50+ choices, consider increasing the option-token budget or, preferably, splitting the decision hierarchically.

---

# 10. Confidence

Laya returns confidence information with its decisions.

Use the confidence as a routing signal rather than blindly trusting every argmax result.

Conceptually:

```python
answer = result["answers"]["route"]

route = answer["choice"]
confidence = answer["confidence"]

if confidence >= THRESHOLD:
    execute_route(route)
else:
    fallback_to_stronger_reasoner()
```

A common architecture is:

```text
                    Laya
                      |
             confidence check
                 /       \
              high       low
               |          |
               v          v
          execute       SLM/LLM
          directly      planner
```

This gives Laya a fast path while preserving a slower reasoning fallback.

Important:

> Do not assume the raw confidence is perfectly calibrated.

The model documentation explicitly reports that calibration improves after temperature fitting on domain data. Therefore, for high-stakes or autonomous routing, measure calibration on your own workload before using a confidence threshold operationally.

---

# 11. Laya Router vs Direct Model Loading

There are two main ways to use Laya.

## Router mode

Recommended when you want automatic checkpoint selection.

```python
from laya import Router

router = Router(preload=True)

result = router.predict(state, questions)
```

The router can select the appropriate checkpoint based on language/script.

This is the recommended general production approach when supporting multiple languages.

---

## Direct model mode

Use a specific checkpoint directly when you already know which model you want.

```python
import laya

agent = laya.load("convaiinnovations/laya")

result = agent.predict(state, questions)
```

For the multilingual checkpoint:

```python
agent = laya.load(
    "convaiinnovations/laya",
    subfolder="multilingual"
)
```

For the typed-decisions checkpoint:

```python
agent = laya.load(
    "convaiinnovations/laya",
    subfolder="typed-decisions"
)
```

Use direct loading when the application has a fixed language/domain/model requirement.

---

# 12. Multilingual Routing

The repository contains multiple checkpoints.

The important distinction is:

```text
laya
    -> English-oriented checkpoint

laya-multilingual
    -> 100+ languages

laya-typed-decisions
    -> checkpoint fine-tuned for typed decision workflows
```

The `Router` can automatically select between checkpoints.

```python
from laya import Router

router = Router(preload=True)

result = router.predict(state, questions)

print(result["routing"])
```

If your application already knows the language, you can provide that information using the supported language override mechanisms.

Do not use the English-only checkpoint for arbitrary non-English input and assume confidence will detect the problem. The model documentation specifically notes that the English checkpoint can remain highly confident while performing poorly on unsupported scripts.

---

# 13. Production Preloading

The first model construction can take substantially longer than normal inference.

For latency-sensitive applications:

```python
router = Router(preload=True)
```

You can also explicitly preload selected checkpoints.

```python
router.preload(["english", "multilingual"])
```

The general principle is:

```text
Startup:
    load models

Runtime:
    reuse models

Request:
    Laya inference
```

rather than:

```text
Request:
    load model
    run inference
    unload model
```

Repeated model loading is particularly undesirable in an agent that performs many decisions.

---

# 14. Memory-Constrained Deployment

If memory is constrained, Laya supports limiting the number of resident checkpoints.

Conceptually:

```python
router = Router(max_loaded=1)
```

However, switching between checkpoints can then require model reloads, which significantly increases latency.

For latency-sensitive applications, keeping the required checkpoints resident is generally preferable if hardware permits it.

---

# 15. Laya in an Agent Graph

Laya fits naturally before graph execution.

Generic architecture:

```text
                    INPUT
                      |
                      v
                 State Builder
                      |
                      v
                    LAYA
                      |
              +-------+-------+
              |       |       |
              v       v       v
            Route   Gate    Priority
              |
              v
          Graph Router
              |
      +-------+-------+
      |       |       |
      v       v       v
    Node A  Node B  Node C
      |       |       |
      +-------+-------+
              |
              v
          Observation
              |
              v
          Next Decision
```

Laya does not need to replace the graph.

It makes decisions that determine which graph branch should execute.

---

# 16. Laya + Planner Architecture

For more capable agents, separate fast routing from reasoning.

```text
                         USER
                           |
                           v
                         LAYA
                           |
                 +---------+---------+
                 |                   |
              simple              complex
                 |                   |
                 v                   v
             Tool Graph             SLM
                                     |
                                  Planning
                                     |
                                     v
                                  Laya
                                     |
                                     v
                                Tool Graph
```

The responsibilities become:

### Laya

- classify
- route
- gate
- select among known alternatives
- estimate ordinal priority
- provide confidence

### SLM/LLM

- interpret complicated goals
- create multi-step plans
- resolve ambiguity
- reason over observations
- generate arguments for tools
- re-plan when execution fails

### Graph

- enforce deterministic workflow
- manage state
- execute nodes
- control loops
- enforce constraints

### Tools

- perform actual external actions

This separation is usually easier to control than asking one SLM to do everything.

---

# 17. State Design for Agentic Systems

For an agent, do not necessarily send the entire global state to every Laya decision.

Instead create a decision-specific state.

Example:

```python
state = {
    "user_request": "...",
    "current_goal": "...",
    "recent_observation": "...",
    "available_context": "...",
}
```

Then construct the state needed by a particular router:

```python
routing_state = {
    "user_request": state["user_request"],
    "current_goal": state["current_goal"],
}
```

For a later execution decision:

```python
execution_state = {
    "goal": state["current_goal"],
    "last_action": state["last_action"],
    "observation": state["recent_observation"],
}
```

This reduces unnecessary context and makes each decision easier to evaluate.

---

# 18. Routing After Tool Execution

Laya can also be used after a tool produces an observation.

For example:

```text
User request
     |
     v
   Laya
     |
     v
 Tool A
     |
     v
Observation
     |
     v
   Laya
     |
     +---- continue
     +---- retry
     +---- use another tool
     +---- ask user
     +---- finish
```

The second Laya call is a different decision from the initial route.

Do not make the first router responsible for every future decision.

Instead, use small decision points throughout the graph.

---

# 19. Confidence-Based Fallback

A robust pattern is:

```python
result = router.predict(state, questions)

answer = result["answers"]["route"]

if answer["confidence"] >= threshold:
    route = answer["choice"]
else:
    route = "planner"
```

Architecture:

```text
                 Laya
                   |
            confidence?
              /       \
            high       low
             |          |
             v          v
        deterministic   SLM
           graph       planner
             |          |
             +----+-----+
                  |
                  v
               execute
```

This gives you a fast path without forcing Laya to solve cases where it is uncertain.

---

# 20. Designing a Good Fallback

A fallback should not simply be:

```text
confidence < threshold
    -> error
```

For an agent, useful fallbacks include:

```text
low confidence
      |
      +--> larger SLM
      |
      +--> hierarchical router
      |
      +--> retrieve additional context
      |
      +--> ask user
      |
      +--> safe/default branch
```

Which fallback is appropriate depends on the cost of making the wrong decision.

---

# 21. Hierarchical Routing

When the option space is large, use multiple smaller choices.

Instead of:

```text
60 possible actions
       |
       v
    one Laya
```

use:

```text
              Laya
               |
       +-------+-------+
       |       |       |
    domain1 domain2 domain3
       |       |       |
      Laya    Laya    Laya
       |       |       |
     5 opts   4 opts  6 opts
```

This has several advantages:

- fewer alternatives per decision
- clearer option descriptions
- easier evaluation
- easier thresholding
- easier graph organization
- easier replacement of one domain router

It also addresses Laya's documented weakness on very high-cardinality choices.

---

# 22. Combining Multiple Laya Decisions

A useful design is to ask independent questions together.

For example:

```python
questions = {
    "domain": {
        "type": "choice",
        "instructions": "Which domain does this belong to?",
        "criteria": {
            "domain_a": "...",
            "domain_b": "...",
            "domain_c": "...",
        },
    },

    "requires_reasoning": {
        "type": "choice",
        "instructions": "Does this require multi-step reasoning?",
        "criteria": {
            "yes": "The task requires planning or reasoning across multiple steps",
            "no": "The task can be handled by a direct operation",
        },
    },

    "requires_confirmation": {
        "type": "choice",
        "instructions": "Does execution require explicit confirmation?",
        "criteria": {
            "yes": "The action should be confirmed before execution",
            "no": "The action can proceed without confirmation",
        },
    },
}
```

One call can therefore produce a routing vector:

```text
domain = domain_b
reasoning = yes
confirmation = no
```

The graph can use those fields to determine execution.

---

# 23. Do Not Use Laya for Everything

Laya is not a replacement for a general LLM.

Good uses:

```text
classification
routing
binary gating
priority estimation
workflow branching
policy selection
state classification
tool-domain selection
```

Poor uses:

```text
long-form generation
complex open-ended reasoning
writing arbitrary tool arguments
multi-step mathematical reasoning
creative generation
large-scale synthesis
```

For those tasks, use a generative model.

---

# 24. Long Inputs

The multilingual checkpoint supports longer inputs than the default configuration, but longer context can substantially increase inference time and may reduce accuracy depending on the task.

For long multilingual input, the model documentation supports:

```python
result = router.predict(
    long_document,
    questions,
    model="multilingual",
    max_len=8192,
)
```

Do not automatically pass huge documents to Laya.

A better architecture is often:

```text
large document
      |
      v
retrieve/reduce relevant evidence
      |
      v
small decision state
      |
      v
Laya
```

This keeps the routing decision focused.

---

# 25. Fine-Tuning

The base checkpoints are intended to work zero-shot, but the model documentation shows that domain-specific fine-tuning can significantly improve typed-decision performance.

The general workflow is:

```text
Domain examples
      |
      v
Decision dataset
      |
      v
Fine-tune Laya
      |
      v
Calibration
      |
      v
Evaluation
      |
      v
Production
```

A training example should capture:

```text
state
+
question
+
available options
+
correct decision
```

For example, conceptually:

```json
{
  "state": "...",
  "question": "...",
  "options": {
    "A": "...",
    "B": "...",
    "C": "..."
  },
  "answer": "B"
}
```

For an agent, collect examples from actual routing decisions rather than artificially generated examples alone.

The official model card provides a fine-tuning notebook and reports substantially higher typed-decision benchmark performance for its fine-tuned checkpoint than the base checkpoint.

---

# 26. Calibration

If your agent makes autonomous decisions based on confidence, calibration matters.

Do not assume:

```text
confidence = 0.90
```

means:

```text
90% empirical correctness
```

without measuring it.

A practical calibration workflow:

```text
1. Collect labelled production-like examples.
2. Run Laya.
3. Record prediction + confidence.
4. Compare confidence with actual correctness.
5. Fit temperature scaling or another calibration method.
6. Re-evaluate.
7. Choose operating thresholds.
```

Evaluate separately for:

- question type
- number of options
- domain
- language
- model checkpoint

---

# 27. Threshold Selection

Do not choose a threshold arbitrarily.

For example:

```python
if confidence > 0.8:
    execute()
else:
    fallback()
```

is only a starting point.

Instead measure:

```text
threshold
   |
   +--> automatic execution accuracy
   +--> fallback rate
   +--> false-route rate
   +--> latency
```

Then choose the threshold according to the cost of errors in your application.

A wrong route that only causes a slightly longer workflow is very different from a wrong route that triggers an irreversible action.

---

# 28. Safety for Autonomous Actions

Laya confidence should not be treated as the only safety mechanism.

For actions with external side effects, combine:

```text
Laya decision
      +
policy rules
      +
permissions
      +
confirmation requirements
      +
tool validation
```

Example:

```text
Laya
 |
 v
"send_message"
 |
 v
Policy check
 |
 +---- not allowed --> stop
 |
 +---- confirmation required --> ask user
 |
 +---- allowed --> execute
```

The routing model should not be the final authority for security-sensitive operations.

---

# 29. Using Laya Behind an HTTP Service

Laya provides a Jev-compatible HTTP server.

Install:

```bash
pip install "laya[serve]"
```

Run:

```bash
LAYA_DEVICE=cuda LAYA_PRELOAD=1 laya-serve
```

The server exposes:

```text
POST /v1/systemone
```

A request conceptually looks like:

```json
{
  "state": {
    "document": "..."
  },
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Which route applies?",
      "criteria": {
        "A": "...",
        "B": "..."
      }
    }
  }
}
```

This can be useful when:

```text
Mobile Agent
     |
     v
Local/Private API
     |
     v
Laya Server
     |
     v
Decision
```

instead of embedding the model directly into the application process.

Authentication can be configured using the server's API-key mechanism.

---

# 30. LangChain / LangGraph Integration

Laya provides an optional LangChain/LangGraph integration.

Install:

```bash
pip install "laya[langchain]"
```

The general architecture becomes:

```text
LangGraph State
      |
      v
   Laya Router
      |
      v
Conditional Edge
      |
      +--> Node A
      +--> Node B
      +--> Node C
```

This is particularly useful when Laya is being used as a deterministic decision component inside a larger state machine.

Keep the responsibilities separate:

```text
LangGraph
    = workflow/state/orchestration

Laya
    = fast decisions

LLM/SLM
    = reasoning/generation

Tools
    = external actions
```

---

# 31. A Generic Router Wrapper

For larger applications, hide Laya behind a small abstraction.

```python
from laya import Router


class DecisionRouter:
    def __init__(self):
        self.router = Router(preload=True)

    def predict(self, state, questions):
        result = self.router.predict(state, questions)
        return result

    def choice(self, state, name, instructions, criteria):
        questions = {
            name: {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            }
        }

        result = self.router.predict(state, questions)

        return result["answers"][name]
```

Then application code can use:

```python
decision = router.choice(
    state,
    "route",
    "Which route should handle this request?",
    {
        "A": "Description of route A",
        "B": "Description of route B",
        "C": "Description of route C",
    },
)

if decision["confidence"] > threshold:
    execute(decision["choice"])
else:
    fallback()
```

This prevents Laya-specific implementation details from spreading throughout the agent codebase.

---

# 32. Recommended Generic Agent Architecture

A strong general architecture is:

```text
                         INPUT
                           |
                           v
                    Context Builder
                           |
                           v
                    +-------------+
                    |    LAYA     |
                    | Fast Router |
                    +------+------+
                           |
                     confidence?
                      /          \
                    high         low
                     |            |
                     v            v
              Deterministic      SLM
                  graph          planner
                     |            |
                     +-----+------+
                           |
                           v
                     Tool Selection
                           |
                           v
                       Executor
                           |
                           v
                      Observation
                           |
                           v
                    +-------------+
                    |    LAYA     |
                    | Next-State  |
                    | Decision    |
                    +------+------+
                           |
             +-------------+-------------+
             |             |             |
          continue       retry        finish
             |             |             |
             +-------------+-------------+
                           |
                           v
                          END
```

This gives you a layered agent:

```text
Laya       -> fast System 1 decisions
SLM/LLM    -> System 2 reasoning
Graph      -> deterministic control
Tools      -> actions
State      -> memory/context
Policies   -> safety constraints
```

---

# 33. Practical Development Workflow

Build a Laya router in this order.

### Step 1 — Define the decision

Write the question in plain language:

```text
What decision does the system actually need to make?
```

### Step 2 — Define mutually meaningful options

Each option should represent a real branch.

### Step 3 — Create representative states

Collect real examples covering:

- obvious cases
- ambiguous cases
- edge cases
- adversarial cases
- missing-information cases

### Step 4 — Run zero-shot evaluation

Measure:

```text
accuracy
confidence
latency
fallback rate
```

### Step 5 — Add confidence gating

Do not automatically execute low-confidence decisions.

### Step 6 — Add hierarchical routing if necessary

Split large choice spaces into smaller decisions.

### Step 7 — Calibrate

If confidence controls autonomous behavior, calibrate it using domain data.

### Step 8 — Fine-tune if zero-shot performance is insufficient

Use labelled decisions from the actual application domain.

### Step 9 — Integrate into the graph

Connect the decision output to deterministic graph edges.

### Step 10 — Monitor production decisions

Log:

```text
input/state
question
options
selected answer
confidence
actual outcome
fallback
latency
```

This gives you data for future calibration and fine-tuning.

---

# 34. What to Log

For every production decision, maintain something like:

```json
{
  "decision_id": "...",
  "state": "...",
  "question": "...",
  "selected_option": "...",
  "confidence": 0.91,
  "model": "english",
  "latency_ms": 34,
  "fallback_used": false,
  "actual_outcome": "..."
}
```

Do not necessarily store sensitive raw state. Apply the privacy requirements of your application.

The important point is to preserve enough information to determine whether the router made the correct decision.

---

# 35. Common Mistakes

## Mistake 1: Treating Laya like an LLM

Do not ask it to generate arbitrary plans or text.

Use typed decisions.

## Mistake 2: Giving it too many choices

Use hierarchical routing.

## Mistake 3: Trusting confidence blindly

Calibrate on your own domain.

## Mistake 4: Sending irrelevant context

Build a compact decision-specific state.

## Mistake 5: Using one router for the entire agent

Use multiple decision points where the workflow actually branches.

## Mistake 6: Using Laya as the security layer

Use explicit policy and permission checks for sensitive actions.

## Mistake 7: Assuming zero-shot domain accuracy

The base model can perform poorly on specialized typed decisions. Benchmark your actual routing tasks.

## Mistake 8: Using `noul` without validation

Check binary decisions on your data; use neutral two-option `choice` if necessary.

## Mistake 9: Using huge option sets

Large option spaces can reduce performance because option descriptions share a limited head-token budget.

---

# 36. Minimal Reference Implementation

```python
from laya import Router


router = Router(preload=True)


state = {
    "request": "Example user request",
    "context": "Relevant context"
}


questions = {
    "route": {
        "type": "choice",
        "instructions": "Which route should handle this request?",
        "criteria": {
            "route_a": "Requests belonging to route A",
            "route_b": "Requests belonging to route B",
            "route_c": "Requests belonging to route C",
        },
    },

    "needs_reasoning": {
        "type": "choice",
        "instructions": "Does this require multi-step reasoning?",
        "criteria": {
            "yes": "Requires planning or multi-step reasoning",
            "no": "Can be handled directly",
        },
    },
}


result = router.predict(state, questions)

route = result["answers"]["route"]["choice"]
route_confidence = result["answers"]["route"]["confidence"]

needs_reasoning = result["answers"]["needs_reasoning"]["choice"]

print("Route:", route)
print("Confidence:", route_confidence)
print("Needs reasoning:", needs_reasoning)


if route_confidence < 0.80:
    # Send to a stronger reasoning/fallback component.
    pass
else:
    # Continue through the selected deterministic graph branch.
    pass
```

---

# 37. Key Takeaway

The most useful way to think about Laya is:

```text
             GENERATIVE MODEL
             "What should I do?"
                    |
                    v
                 LAYA
             "Which branch?"
                    |
                    v
               GRAPH
             "Where next?"
                    |
                    v
                TOOLS
             "Do the action."
```

Laya is most valuable when the agent has many known branches but does not need a large generative model to choose among them.

The recommended design is therefore:

```text
small decision spaces
        +
typed questions
        +
multiple decisions per forward pass
        +
confidence gating
        +
hierarchical routing
        +
deterministic graph execution
        +
SLM/LLM fallback for genuinely ambiguous or reasoning-heavy cases
```

This turns Laya into a fast, structured routing layer rather than trying to make it the entire agent.

---

## Official References

- Model card: https://huggingface.co/convaiinnovations/laya
- Laya GitHub: https://github.com/NandhaKishorM/laya
- PyPI: https://pypi.org/project/laya/
- Documentation: https://nandhakishorm.github.io/laya/

Always check the current repository/model card for API changes before implementing production code.
