# EYV JARVIS — Project Memory

## Coding sub-agent — current state

There is no Claude Code / Anthropic dependency anywhere in this repo today.

The "coding sub-agent" today is `actions/dev_agent.py` + `actions/code_helper.py`,
which call the same local LLM (`core.llm_client.call_llm_text` /
`call_llm_vision`) to generate code, run it via subprocess, parse tracebacks,
and retry (up to `MAX_FIX_ATTEMPTS` / `MAX_BUILD_ATTEMPTS`).

Any "replace Claude Code with Antigravity" language elsewhere in this project
is describing giving that self-contained LLM loop a real external agent — it
is **not** un-plugging an existing Claude Code integration, because none
exists.

## Local model — naming and target

"Nemotron Q3" is not a real model identifier and must not appear anywhere in
this file, in code, in comments, or in config going forward. Q3/Q4 are GGUF
quantization levels, not model names.

**Confirmed target:**

| Field | Value |
|---|---|
| Model | NVIDIA Nemotron-3-Nano 4B |
| Ollama tag | `nemotron-3-nano:4b` |
| Quantization | Q4_K_M |
| Model size | ~2.8 GB |
| Parameters | ~3.97B |
| Context | 256K listed max — do NOT configure the full 256K on this laptop; pick a practical size based on actual measured VRAM/RAM headroom, not the listed max |
| Runtime | Ollama |
| GPU | RTX 4060 Laptop GPU, 8GB VRAM |

The 30B-A3B Nemotron-3-Nano variant is explicitly **NOT** the target — it
needs ~24GB VRAM and is not viable on this hardware.

Before implementing anything based on these numbers, re-confirm them live
with `ollama show nemotron-3-nano:4b`, since Ollama's model listings change
over time — treat the numbers above as "best known at time of writing," not
as frozen truth.

**Naming rule going forward:** any future experiment with Q3 quantization
must be described as "a Q3 quantization of Nemotron-3-Nano," never as if Q3
were a model name.
