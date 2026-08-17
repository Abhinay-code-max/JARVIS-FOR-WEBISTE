# EYV / JARVIS — PROJECT MEMORY

> This file is the authoritative project memory for Claude Code working on the EYV/JARVIS agent system (repo: `Abhinay-code-max/JARVIS-FOR-WEBISTE`, internally "MARK XL").
>
> IMPORTANT: The existing EYV architecture is already established and working. Do NOT redesign the entire system. The primary objective is to migrate the coordinator brain from the local chat model currently configured to NVIDIA Nemotron-3-Nano 4B, and to add Google Antigravity as the engineering sub-agent, while preserving the rest of the architecture and existing functionality.
>
> This file supersedes any earlier draft of this memory. Where earlier drafts said "Nemotron Q3" or implied Claude Code was already wired into this repo, those were wrong — see VERIFIED FACTS below.

---

# 0. VERIFIED FACTS — READ THIS FIRST

These are confirmed by direct repo inspection, not assumptions. Any instruction elsewhere in this file that conflicts with these facts is the thing that's wrong, not this section.

**Coding sub-agent — current state:**
There is no Claude Code / Anthropic dependency anywhere in this repo today — confirmed by grep across the codebase. The "coding sub-agent" today is `actions/dev_agent.py` + `actions/code_helper.py`, which call the *same* local LLM (`core.llm_client.call_llm_text` / `call_llm_vision`) to generate code, run it via `subprocess`, parse tracebacks, and retry (up to `MAX_FIX_ATTEMPTS` / `MAX_BUILD_ATTEMPTS`). Anywhere this file talks about "replacing Claude Code with Antigravity," read that as *giving this self-contained LLM loop a real external agent* — it is not un-plugging an existing Claude Code integration, because none exists.

**Model naming and target:**
"Nemotron Q3" is not a real model identifier and must not appear anywhere in this repo's code, comments, or config. Q3/Q4 are GGUF quantization levels, not model names.

Confirmed target:
| Field | Value |
|---|---|
| Model | NVIDIA Nemotron-3-Nano 4B |
| Ollama tag | `nemotron-3-nano:4b` |
| Quantization | Q4_K_M |
| Model size | ~2.8 GB |
| Parameters | ~3.97B |
| Context | 256K listed max — do NOT configure the full 256K on this laptop; pick a practical size based on measured VRAM/RAM headroom |
| Runtime | Ollama |
| GPU | RTX 4060 Laptop GPU, 8GB VRAM |

The 30B-A3B Nemotron-3-Nano variant is explicitly **not** the target — it needs ~24GB VRAM and is not viable on this hardware.

Before implementing anything based on these numbers, re-confirm them live with `ollama show nemotron-3-nano:4b` — Ollama's model listings change over time, so treat this table as "best known at time of writing," not frozen truth.

Naming rule going forward: a future experiment with Q3 quantization must be described as "a Q3 quantization of Nemotron-3-Nano," never as if Q3 were a model name.

**Repo structure, confirmed:**
`main.py`, `ui.py` (PyQt6 HUD) at root; `core/` (llm_client.py, tool_dispatch.py, tool_declarations.py, tool_gate.py, tool_contracts.py, policy.py, db.py, proactive.py, stt.py, tts.py, confirm.py, task_approval.py, postconditions.py, error handling under `agent/error_handler.py`); `agent/` (planner.py, executor.py, task_queue.py, step_references.py); `actions/` (18 tools including dev_agent.py, code_helper.py, browser_control.py, file_controller.py, web_search.py, weather_report.py, etc.); `memory/` (memory_manager.py); `recognition/` (face_id.py, voice_id.py, wake_word.py); `tests/`; `config/` (api_keys.json is the single source of truth for provider/model/URL, read via `config/__init__.py`).

`core/llm_client.py` already implements a provider-agnostic abstraction (`"ollama"` or `"openai"`-compatible backend, selected via `llm_provider` in `config/api_keys.json`), with normalized tool-call handling across both. This means the model swap in Phase 1 is primarily configuration + verification, not a rewrite of the LLM layer.

There is no Denver / Bob / Sara / EYV-backend code in this repo. Those sub-agents live in the separate EYV website repo. This repo is the laptop-side JARVIS coordinator only.

---

# 1. PROJECT IDENTITY

Project: **EYV Agent System**

Core coordinator: **JARVIS**

JARVIS is the central AI coordinator/orchestrator for the EYV system. The system is not intended to be a simple chatbot.

The main architecture is:

```text
                    USER
                      │
                      ▼
                  JARVIS
          (Nemotron-3-Nano 4B)
                      │
             ┌────────┼────────┐
             │        │        │
             ▼        ▼        ▼
          Denver     Bob      Sara
          Support   Marketing Analytics
             │        │        │
             └────────┼────────┘
                      │
                      ▼
                Coding Agent
                Antigravity
                      │
                      ▼
               EYV / GitHub / Code
```

JARVIS is the coordinator. Nemotron-3-Nano 4B is the brain. The existing sub-agents remain responsible for their specialized domains. Antigravity becomes the engineering/coding sub-agent.

---

# 2. CORE MOTTO

## JARVIS IS THE COORDINATOR, NOT THE WORKER.

> **Nemotron thinks. JARVIS coordinates. Sub-agents specialize. Tools execute.**

JARVIS should:

1. Understand the user's request.
2. Determine the user's intent.
3. Decide whether the request is simple or requires delegation.
4. Select the appropriate sub-agent.
5. Provide the sub-agent with the required context.
6. Monitor the task.
7. Receive the result.
8. Evaluate the result.
9. Delegate additional work if required.
10. Produce the final response.

JARVIS should NOT unnecessarily perform specialized work itself.

```text
User asks for customer support analysis → JARVIS → Denver → Result → JARVIS → Final response
Marketing task                          → JARVIS → Bob    → Result → JARVIS → Final response
Analytics task                          → JARVIS → Sara   → Result → JARVIS → Final response
Engineering task                        → JARVIS → Antigravity → Code/test/debug → Result → JARVIS → Final response
```

For complex tasks, JARVIS may coordinate multiple agents.

---

# 3. PRIMARY BRAIN — NEMOTRON-3-NANO 4B

The prior architecture used a generic local chat model (llama3.2/qwen2.5-class) via Ollama as the reasoning/orchestration layer. This is being replaced.

## NEW PRIMARY BRAIN

**NVIDIA Nemotron-3-Nano 4B**, run locally via Ollama as `nemotron-3-nano:4b`, Q4_K_M quantization. See Section 0 for the full confirmed spec table — do not use any other naming for this model.

Claude (Anthropic) must NOT be used as the runtime brain. Claude Code is only the development environment used to modify this repository — it is not, and never has been, a runtime dependency of JARVIS. See Section 0 for confirmation that no Claude/Anthropic dependency exists in this repo today.

The final deployed JARVIS system must not require a Claude/Anthropic API key for its primary reasoning.

---

# 4. NEMOTRON RUNTIME

The existing repository uses Ollama for AI inference via `core/llm_client.py`, which already abstracts over "ollama" and OpenAI-compatible providers selected through `config/api_keys.json`.

Do NOT remove Ollama. Ollama remains the runtime layer for `nemotron-3-nano:4b`.

Target architecture:

```text
JARVIS Coordinator
        ↓
core/llm_client.py (existing abstraction)
        ↓
Ollama Provider
        ↓
nemotron-3-nano:4b
```

The rest of the application communicates with this abstraction rather than hard-coding Ollama-specific implementation throughout the project. This is important because the model/runtime may change in the future.

---

# 5. HARDWARE TARGET

The local Nemotron runtime runs on Abhinay's laptop.

Hardware:

* GPU: NVIDIA RTX 4060 Laptop GPU, 8 GB VRAM
* RAM: 16 GB
* CPU: AMD Ryzen 9 HS
* Storage: 1 TB SSD
* OS: Windows

The implementation must account for the 8 GB VRAM limitation, which is **shared** — Whisper STT and any local TTS (e.g. Kokoro) also consume VRAM/RAM concurrently with the LLM, so headroom must be measured under a realistic concurrent session, not the LLM in isolation.

Confirmed model choice for this hardware: `nemotron-3-nano:4b` (see Section 0). Before implementation, re-verify live via `ollama show nemotron-3-nano:4b`: exact quantization, chat template, context requirements, GPU offloading behavior (confirm via `ollama ps` that inference is actually on GPU, not silently falling back to CPU), RAM requirements, VRAM requirements. Do not invent model names or parameters — prefer official NVIDIA documentation/model information and reputable model repositories, and prefer live verification over anything written in this file, since listings change.

---

# 6. LLM ABSTRACTION

`core/llm_client.py` already provides this abstraction — do not rebuild it. It supports (confirm exact function names on inspection, expected to include):

```text
call_llm()          — chat with tool-call normalization
call_llm_text()      — plain text completion
call_llm_vision()    — vision-capable completion
call_llm_stream()    — streaming
```

The coordinator calls this abstraction rather than calling Ollama directly. Reuse this existing code — do not introduce a new framework.

Known risk to verify in Phase 1: Nemotron-3-Nano 4B is a hybrid reasoning model that may emit a reasoning trace before its final answer. Confirm this doesn't break the sentence-boundary splitting used for streaming TTS (`_SENT_END` regex in `llm_client.py`) or the code-fence stripping used in `dev_agent.py`/`code_helper.py`. If Nemotron's Ollama template exposes a way to suppress the trace for tool-calling/coordinator turns, use it; otherwise add minimal post-processing rather than rewriting the abstraction.

---

# 7. JARVIS COORDINATOR

JARVIS remains the central coordinator (`agent/planner.py`, `agent/executor.py`, `agent/task_queue.py`, `core/tool_dispatch.py`). The only major change is the underlying brain:

### BEFORE
```text
JARVIS → previous local chat model (e.g. llama3.2/qwen2.5)
```

### AFTER
```text
JARVIS → nemotron-3-nano:4b
```

The coordinator's responsibilities remain: reasoning, intent detection, planning, delegation, agent selection, tool selection, task monitoring, result evaluation, final response generation. Do not remove existing coordinator functionality simply because the LLM changes.

---

# 8. SUB-AGENTS

The existing sub-agents remain in place. Do NOT redesign or remove them unless there is a concrete technical requirement. Note: none of Denver, Bob, or Sara's code lives in this repo — they're on the EYV backend side (see Section 0).

| Sub-agent   | Role                        | Location                          |
| ----------- | --------------------------- | ---------------------------------- |
| Denver      | Customer support & feedback | EYV backend / Railway              |
| Bob         | Marketing                   | EYV backend / Railway              |
| Sara        | Analytics / BI              | EYV backend / Railway              |
| Antigravity | Engineering / bug-fix       | Local laptop / this repo           |

---

# 9. DENVER — CUSTOMER SUPPORT

Denver remains the customer support and feedback sub-agent (ticket classification, question/bug/feature/other routing, registry access, deterministic Q&A). Implementation: `support_agent_service.py` (EYV backend, not this repo). Denver currently uses Gemini-based ticket classification — do NOT replace Denver's internal model merely because JARVIS is moving to Nemotron. Nemotron is the central coordinator; Denver's implementation remains intact unless a concrete integration issue requires modification.

---

# 10. BOB — MARKETING

Bob remains the marketing sub-agent (Instagram, WhatsApp, Buffer). Keep the existing implementation. Do not redesign Bob as part of the Nemotron migration. JARVIS/Nemotron delegates marketing tasks to Bob.

---

# 11. SARA — ANALYTICS / BI

Sara remains the analytics/BI sub-agent (RevenueCat, internal analytics, BI). Keep the existing implementation. Do not redesign Sara as part of the Nemotron migration. JARVIS/Nemotron delegates analytics tasks to Sara.

---

# 12. CODING SUB-AGENT — ANTIGRAVITY

This is the second major architecture change. Per Section 0: there is currently no external coding agent at all — `actions/dev_agent.py`/`actions/code_helper.py` are a self-contained loop of the same local LLM writing, running, and fixing its own code via subprocess. This section is about adding Google Antigravity as a real external engineering sub-agent alongside (or eventually in place of) that loop.

Target:

```text
JARVIS
   ↓
Coding / Engineering task
   ↓
Antigravity (via `agy` headless CLI)
   ↓
Repository
   ↓
Code changes
   ↓
Tests
   ↓
Result
   ↓
JARVIS
```

Antigravity is the specialized engineering worker. Nemotron remains the coordinator.

Practical integration note: Google ships a headless CLI, `agy`, supporting `agy -p "<prompt>" --output-format json` and `--agent <name> --effort high`. It respects a permission policy — file read/write in the working directory is auto-allowed; shell commands default to "Ask" and are soft-denied (exit 0, warning to stderr) in headless mode unless explicitly granted. Authentication is via Google Sign-In (cached in the OS keyring) or Application Default Credentials for headless use — this is a one-time manual setup step, not something to script.

---

# 13. ANTIGRAVITY RESPONSIBILITIES

Antigravity should handle: repository inspection, coding, bug fixing, implementation, refactoring, testing, debugging, code review, Git operations where appropriate, GitHub-related development workflows, running development commands, investigating technical issues.

JARVIS decides when an engineering task should be delegated to Antigravity, and provides: objective, relevant context, requirements, constraints, affected files, expected behavior, acceptance criteria.

Antigravity should return: status, changes made, tests performed, failures, remaining work, relevant artifacts — normalized into whatever result shape `core/tool_contracts.py` already defines for agent results, not a new ad hoc shape.

Keep `actions/dev_agent.py`'s existing local-LLM loop available as a fallback for when Antigravity is unavailable or unauthenticated (see Section 31, error handling).

---

# 14. ANTIGRAVITY ABSTRACTION

Do not hard-code Antigravity throughout JARVIS. Prefer a conceptual interface:

```text
CodingAgent
    │
    └── AntigravityProvider (shells out to `agy`)
```

This keeps the architecture replaceable — if another coding agent becomes preferable later, it should be swappable without rebuilding the coordinator.

---

# 15. IMPORTANT DISTINCTION

Claude Code and JARVIS are NOT the same thing, and Claude Code has never been a runtime component of this system (see Section 0 — verified by grep, no matches). Claude Code is used only during development, by a human, to modify this repository's source files. It is not invoked by JARVIS at runtime, is not a dependency of `main.py`, and should never become one.

The production architecture must be:

```text
nemotron-3-nano:4b
    ↓
JARVIS Coordinator
    ↓
Sub-agents
    ↓
Tools
```

---

# 16. MULTI-AGENT COORDINATION

JARVIS must support multi-agent workflows, e.g.:

```text
"Analyze why sales dropped and prepare a marketing response."
USER → JARVIS/Nemotron → Sara → Analytics results → JARVIS/Nemotron → Bob → Marketing response → JARVIS/Nemotron → Final response
```

```text
USER → JARVIS → Denver + Sara → Combined results → JARVIS → Decision → Bob → Final response
```

The coordinator should determine whether multiple agents are actually necessary. Do not invoke agents unnecessarily.

---

# 17. AGENT CONTEXT

Each agent receives only the context relevant to its task. Do NOT send the entire conversation or entire project state to every agent by default. Conceptually:

```json
{
  "task_id": "...",
  "agent": "...",
  "objective": "...",
  "context": "...",
  "inputs": "...",
  "constraints": "...",
  "expected_output": "..."
}
```

Follow whatever task/message system already exists in this repo (`agent/task_queue.py`) rather than creating a new framework.

---

# 18. AGENT RESULTS

Agents return structured results where practical:

```json
{
  "task_id": "...",
  "status": "success",
  "result": "...",
  "artifacts": [],
  "errors": [],
  "next_steps": []
}
```

JARVIS evaluates agent results rather than blindly forwarding them. If a result is incomplete, JARVIS may: ask the same agent for additional work, delegate to another agent, perform a permitted tool call, or ask the user for clarification if genuinely necessary.

---

# 19. TOOL ARCHITECTURE

The existing tool architecture (`core/tool_declarations.py`, `core/tool_dispatch.py`, `core/tool_gate.py`) remains intact. Tools are controlled capabilities — web search, browser, file operations, system info, application launching, internal APIs, analytics, customer support operations, GitHub operations, etc.

The model should NOT receive unrestricted system access. Tool calls go through the application's controlled tool layer:

```text
Nemotron → Tool decision → Validation → Tool execution → Tool result → Nemotron
```

---

# 20. SECURITY

Do not weaken existing security. JARVIS must not blindly execute arbitrary commands generated by Nemotron. Potentially destructive operations keep appropriate confirmation/security controls (`core/tool_gate.py`, `core/confirm.py`, `core/task_approval.py`) — deleting files, destructive Git operations, modifying production infrastructure, changing security settings, executing unknown binaries, sending external communications, making purchases, destructive database operations.

Preserve existing authentication, authorization, rate limiting, and audit logging (`core/db.py`, `core/logging_setup.py`, `core/policy.py`).

---

# 21. LAPTOP COMMUNICATION

The existing architecture uses laptop-initiated polling only: no inbound connection to the laptop, no unnecessary port forwarding, no unnecessary public tunnel, no exposed local development port. Preserve this. JARVIS communicates with EYV through the existing secure polling mechanism — do not replace with an inbound connection unless explicitly required and approved.

---

# 22. EYV BACKEND

The existing backend runs on Railway. Keep existing backend architecture. Do not move existing backend sub-agents to the laptop unnecessarily.

```text
EYV Backend / Railway
    │
    ├── Denver
    ├── Bob
    ├── Sara
    ├── Ticket APIs
    ├── Notification services
    └── Analytics APIs
```

JARVIS runs locally. Antigravity runs locally as the engineering agent. Nemotron-3-Nano 4B runs locally as the JARVIS brain, via Ollama.

---

# 23. EXISTING BACKEND INFRASTRUCTURE

Preserve: internal ticket API, ticket deduplication service, notification service, EmailClient/Resend integration, in-app notifications, frontend support widget, notification bell, analytics models, analytics event recording, internal analytics API. Do not rewrite these as part of the LLM migration.

---

# 24. ANALYTICS

Existing analytics functionality must remain intact: AnalyticsEventDoc, PromotionDoc, Mongo persistence, event recording, plan_generated, plan_to_booking, booking_completed, booking_abandoned, internal analytics API. Do not change unless required for integration.

---

# 25. HERMES

Hermes is a separate project. Hermes is NOT the JARVIS coordinator. Do not merge Hermes into this architecture, make Hermes the coordinator, or introduce Hermes dependencies into JARVIS unless explicitly requested.

---

# 26. PROJECT BOUNDARIES

This project is EYV/JARVIS. Do not silently absorb unrelated architecture from the separate Hermes project. If something from Hermes appears useful, propose it separately instead of importing it automatically.

---

# 27. MEMORY

`memory/memory_manager.py` handles this already. JARVIS should eventually maintain:

* **Short-term memory** — current conversation.
* **Working memory** — information required for the active task.
* **Long-term memory** — useful persistent information across sessions.

Memory should be retrieved selectively — do not inject all memory into every prompt. Do not store sensitive information unnecessarily.

---

# 28. VOICE / USER INTERFACE

Preserve the existing JARVIS interface and communication architecture (`ui.py`, `core/stt.py`, `core/stt_deepgram.py`, `core/tts.py`, `recognition/`). Do not remove voice functionality during the Nemotron migration.

```text
Voice / Text → JARVIS → Nemotron → Agents / Tools → Result → Voice / Text
```

The model layer remains independent from the input/output layer.

---

# 29. MODEL PERSONALITY

JARVIS should remain: intelligent, concise, calm, confident, helpful, natural, slightly futuristic. Personality must not interfere with task execution. For complex technical tasks, prioritize correctness over personality.

---

# 30. PERFORMANCE

Target hardware: RTX 4060 Laptop GPU, 8GB VRAM, 16GB RAM (shared with STT/TTS — see Section 5). Priorities, in order: stability, correctness, tool/agent coordination, reasoning, reasonable latency, memory efficiency. Avoid unnecessarily huge context windows (see Section 0 — don't configure the full 256K). Avoid loading every agent's context into every request. Do not run multiple large models locally unless necessary.

---

# 31. ERROR HANDLING

Handle: Ollama unavailable, Nemotron model unavailable, CUDA out-of-memory, insufficient RAM, connection failures, malformed model output, invalid tool calls, agent failures, timeouts, backend unavailable, Antigravity unavailable (including "not authenticated" / "agy soft-denied a permission" as a distinct case from a generic failure), incomplete agent results.

Follow the existing convention in `agent/error_handler.py` rather than inventing a new one. JARVIS should fail gracefully — do not crash the entire system because one sub-agent is unavailable.

---

# 32. LOGGING / OBSERVABILITY

Maintain logs for: request ID, task ID, selected agent, tool calls, model timing, agent timing, errors, task status, retries (`core/logging_setup.py`, `core/db.py`). Do not unnecessarily log private/sensitive user data.

---

# 33. MIGRATION OBJECTIVE

Current architecture:
```text
JARVIS → previous local chat model → Denver / Bob / Sara (EYV backend) / self-contained code-fix loop (dev_agent.py, code_helper.py)
```

Target architecture:
```text
                         JARVIS
                  (Nemotron-3-Nano 4B)
                          │
             ┌────────────┼────────────┐
             │            │            │
             ▼            ▼            ▼
          Denver         Bob          Sara
         Support      Marketing     Analytics
             │            │            │
             └────────────┼────────────┘
                          │
                          ▼
                     Antigravity
                    Engineering
```

Two major changes:

**CHANGE 1** — LLM: previous local chat model → `nemotron-3-nano:4b`.
**CHANGE 2** — Coding: self-contained LLM code-fix loop (`dev_agent.py`/`code_helper.py`) → Antigravity as a real external engineering sub-agent (with the existing loop retained as fallback).

Everything else remains as-is unless a compatibility change is necessary.

---

# 34. DO NOT DO THIS

Do NOT:

* rewrite the entire backend
* replace Denver, Bob, or Sara
* redesign EYV
* merge Hermes
* remove existing analytics
* remove existing ticket infrastructure
* remove existing notification infrastructure
* change the laptop polling model
* introduce Claude API as a runtime dependency (it was never one — keep it that way)
* unnecessarily replace Ollama
* rebuild working components (especially `core/llm_client.py`'s existing provider abstraction)
* create unnecessary frameworks
* make JARVIS directly execute arbitrary shell commands
* expose the laptop to inbound internet connections
* write "Nemotron Q3" anywhere, ever, as if it were a model name

---

# 35. DEVELOPMENT PROCESS

## STEP 1 — INSPECT
Before major changes, inspect: project structure, JARVIS coordinator implementation (`agent/`), current LLM client and model configuration (`core/llm_client.py`, `config/api_keys.json`), tool system (`core/tool_declarations.py`, `core/tool_dispatch.py`, `core/tool_gate.py`), current dev_agent/code_helper implementation, memory (`memory/memory_manager.py`), configuration, authentication, tests, deployment configuration.

## STEP 2 — MAP
Produce a concise migration map: Current component | Current implementation | Required change | Risk | Files affected.

## STEP 3 — PLAN
Provide: Nemotron migration plan, Antigravity integration plan, files to modify, files to leave untouched, dependencies to add/remove, testing strategy, rollback strategy.

DO NOT perform a large rewrite before this inspection.

---

# 36. PHASED IMPLEMENTATION

## PHASE 1 — NEMOTRON
Verify `nemotron-3-nano:4b` compatibility live (`ollama show`), configure Ollama, update `config/api_keys.json` / `llm_client.py` defaults, preserve existing coordinator behavior, confirm GPU offload is actually engaged (not CPU fallback), test basic reasoning, tool-calling, multi-turn conversation, structured output, and record real VRAM/RAM usage under a concurrent STT+TTS+LLM session.

## PHASE 2 — COORDINATOR
Verify intent detection, task planning, delegation, result processing, multi-step workflows all still work against the new model. Do not redesign the coordinator unnecessarily. If Nemotron's tool-call reliability is measurably worse than the previous model, report it rather than silently patching prompts to compensate.

## PHASE 3 — ANTIGRAVITY
Confirm `agy` is installed and authenticated (manual step, not scripted). Build a `CodingAgent`/`AntigravityProvider` module that shells out to `agy -p "..." --output-format json`, normalizes its output into the existing agent-result contract, and surfaces soft-denied-permission failures distinctly from generic errors. Keep the existing local-LLM fix loop as fallback. Verify: coding task delegation, repository access, task context, code modification, testing, result reporting, error handling.

## PHASE 4 — MULTI-AGENT
Verify Denver, Bob, Sara, and Antigravity delegation; sequential tasks; multi-agent tasks; result aggregation.

## PHASE 5 — MEMORY
Verify that changing the LLM does not break conversation memory, working memory, long-term memory, or context retrieval.

## PHASE 6 — END-TO-END
```text
User → JARVIS → Nemotron → Agent → Tool → Agent → JARVIS → Nemotron → User
```

---

# 37. TEST CASES

1. **Simple question** — User → JARVIS → Nemotron → response.
2. **Customer support** — User → JARVIS → Denver → result → JARVIS.
3. **Marketing** — User → JARVIS → Bob → result → JARVIS.
4. **Analytics** — User → JARVIS → Sara → result → JARVIS.
5. **Coding** — User → JARVIS → Antigravity → code → tests → result → JARVIS.
6. **Multi-agent** — User → JARVIS → Sara → Bob → JARVIS → Final response.
7. **Agent failure** — Simulate an unavailable agent (including Antigravity unauthenticated/unavailable) and verify graceful handling.

---

# 38. ACCEPTANCE CRITERIA

The migration is successful when, and only when, all of the following are demonstrated (stop-and-report on any failure — do not silently work around it):

1. Ollama is running `nemotron-3-nano:4b`.
2. JARVIS's reasoning requests go to this model (confirmed in logs, not assumed).
3. No Claude/Anthropic API is required for runtime reasoning — production JARVIS starts and runs fully with no `ANTHROPIC_API_KEY` set.
4. The RTX 4060 is doing the inference (GPU offload confirmed, not CPU fallback).
5. JARVIS maintains a multi-turn conversation correctly.
6. JARVIS can call at least one existing tool successfully.
7. JARVIS can delegate to at least one existing sub-agent (Denver/Bob/Sara).
8. JARVIS can delegate a coding task to Antigravity and get a usable result back.
9. The existing EYV architecture (backend, polling, analytics, notifications, auth) is unchanged and still functional.
10. Actual VRAM and RAM usage are recorded for a realistic concurrent session (LLM + STT + TTS together).
11. Existing tests continue passing.
12. No unnecessary architecture has been removed.

---

# 39. FINAL ARCHITECTURE

```text
                         ┌───────────────┐
                         │     USER      │
                         └───────┬───────┘
                                 │
                                 ▼
                     ┌─────────────────────┐
                     │       JARVIS        │
                     │     COORDINATOR     │
                     │                     │
                     │ Nemotron-3-Nano 4B  │
                     │  (nemotron-3-nano   │
                     │      :4b, Ollama)   │
                     └──────────┬──────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
        ┌───────────┐     ┌───────────┐    ┌───────────┐
        │  Denver   │     │    Bob    │    │   Sara    │
        │  Support  │     │ Marketing │    │ Analytics │
        └───────────┘     └───────────┘    └───────────┘
                                │
                                ▼
                       ┌────────────────┐
                       │  Antigravity   │
                       │   Engineering  │
                       │  (agy headless │
                       │      CLI)      │
                       └───────┬────────┘
                               │
                               ▼
                         EYV Repository
                         Code / GitHub
```

Supporting architecture:

```text
EYV Backend (Railway)
    │
    ├── Denver
    ├── Bob
    ├── Sara
    ├── Ticket APIs
    ├── Deduplication
    ├── Notifications
    ├── Analytics
    └── Frontend

Local Laptop (RTX 4060 8GB / 16GB RAM / Ryzen 9 HS / Windows)
    │
    ├── JARVIS (this repo)
    ├── nemotron-3-nano:4b (via Ollama)
    ├── Antigravity (agy CLI)
    └── STT/TTS (sharing the 8GB VRAM budget)

Communication
    │
    └── Laptop-initiated polling only — no inbound connections
```

---

# 40. GOLDEN RULE

## NEMOTRON-3-NANO 4B IS THE BRAIN.
## JARVIS IS THE COORDINATOR.
## DENVER, BOB AND SARA ARE SPECIALISTS.
## ANTIGRAVITY IS THE CODING SPECIALIST.
## TOOLS ARE THE HANDS.
## MEMORY IS THE KNOWLEDGE.
## THE USER IS IN CONTROL.

The goal is NOT to build another chatbot. The goal is to build a modular, intelligent, multi-agent AI coordination system where Nemotron-3-Nano 4B provides the reasoning capability and JARVIS coordinates specialized agents to accomplish real tasks.

Preserve the existing EYV system. Change the brain. Add the coding agent. Keep the rest.
