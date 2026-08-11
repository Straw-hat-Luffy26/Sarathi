# Hugging Face PEFT and Sarathi's LoRA Switching System

**Status:** Analysis only — no production code was modified.
**Date:** 2026-08-09
**Scope:** Adapter management, switching, hotswapping, caching, routing. The model
engine, model selection, and hardware sizing are explicitly out of scope and are
not redesigned here.

**Epistemic key used throughout:**

| Marker | Meaning |
|---|---|
| **[CODE]** | Verified by reading this repository at the cited file and line. |
| **[DOC]** | Verified against official Hugging Face / vLLM / SGLang documentation, fetched during this analysis. |
| **[EXT]** | External research, papers, or third-party sources. |
| **[REC]** | This document's own recommendation or design reasoning — not a claim of fact. |

---

## 1. Executive Summary

**Sarathi cannot use Hugging Face PEFT as its runtime LoRA switching layer, and
should not try to.** The blocker is not a design gap or a missing feature — it is
that the two systems operate on incompatible runtimes.

Three verified facts settle it:

1. **Sarathi's inference engine is llama.cpp, in-process, through the Rust crate
   `llama-cpp-2`.** There is no Python interpreter, no PyTorch, and no
   `torch.nn.Module` anywhere in the inference path. **[CODE]**
   `src-tauri/Cargo.toml:69`, `src-tauri/src/ai_engine/runtime.rs`
2. **PEFT operates exclusively on PyTorch `nn.Module` graphs.** `load_adapter()`
   and `set_adapter()` work by replacing `nn.Linear` layers with `lora.Linear`
   wrappers and flipping an `active_adapter` pointer on those Python objects.
   **[DOC]** PEFT v0.20.0 `peft_model` reference
3. **The adapter file formats differ, and Sarathi already knows this.** llama.cpp
   loads GGUF adapters; PEFT produces and consumes
   `adapter_model.safetensors` + `adapter_config.json`. Sarathi's own validator
   classifies PEFT safetensors as `RequiresConversion` and its resolver
   deliberately refuses to bind them. **[CODE]**
   `src-tauri/src/lora/validator.rs:179`, `src-tauri/src/capability/resolver.rs:137-143`

Handing PEFT a `LlamaModel` is not a matter of writing an adapter shim. There is
no Python object graph for PEFT to perform its module surgery on.

**However, the analysis is not simply "no."** Two useful conclusions follow:

- **Sarathi has already independently built the architecture that PEFT-based
  designs are told to build on top of PEFT.** The data-driven adapter registry,
  the compare-before-switch check, the load-once/cache-handle pattern, the
  pre-bind structural verification, the strict serialization of model access, and
  the never-fatal degradation path all exist in the Rust code today. On
  concurrency safety Sarathi is *stronger* than raw PEFT, which ships no locking
  at all. See §4 and §6.
- **PEFT has a real, non-runtime role: offline adapter ingestion.** The single
  largest functional gap in Sarathi's LoRA system today is that most published
  adapters are PEFT safetensors and are therefore unusable. A conversion step —
  llama.cpp's `convert_lora_to_gguf.py`, with PEFT/`safetensors` as supporting
  libraries — turns those into GGUF that Sarathi's *existing, unchanged* binding
  path can load. This touches no part of the model engine. **[REC]** See §7 and §13.

**Verdict: Incompatible as a runtime switching layer. Partially compatible as an
offline, build-time adapter ingestion tool.**

This document also records five verified defects and gaps found in Sarathi's
current LoRA path during the analysis (§4, §15) — including one dead struct field
that silently prevents capability routing from ever reaching gateway traffic.

---

## 2. Current Sarathi Architecture

### 2.1 Process and language topology **[CODE]**

```
┌─────────────────────────────────────────────────────────────────┐
│  React 19 + TypeScript (src/)          — Tauri webview          │
├─────────────────────────────────────────────────────────────────┤
│  Rust (src-tauri/src/)                 — Tauri 2 host process   │
│    ├── ai_engine/     llama.cpp runtime, scheduler, LoRA cache  │
│    ├── capability/    intent → capability → backend resolution  │
│    ├── adapter_manager/ package manifest, GGUF verification     │
│    ├── gateway/       local OpenAI + Anthropic HTTP server      │
│    └── launcher/      spawns external coding tools              │
├─────────────────────────────────────────────────────────────────┤
│  llama.cpp (via llama-cpp-2 0.1.x FFI)  — in-process, native    │
├─────────────────────────────────────────────────────────────────┤
│  Python sidecars (sidecars/)  — memory engine + MCP research    │
│    NOT in the inference path. No torch, no transformers.        │
└─────────────────────────────────────────────────────────────────┘
```

Verified inference dependency set — `src-tauri/Cargo.toml` contains
`llama-cpp-2 = { version = "0.1" }` with `cuda` and `vulkan` feature flags, and
**no** `candle`, `pyo3`, `tch`, `ort`, or `onnx` dependency. **[CODE]**
`src-tauri/Cargo.toml:45,46,69`

### 2.2 Model loading **[CODE]**

`InferenceManager` (`src-tauri/src/ai_engine/manager.rs`) owns an
`Arc<Mutex<LlamaCppRuntime>>`. Loading resolves a GGUF path from the package
directory, plans VRAM, selects a GPU, and calls into `llama-cpp-2`. Exactly one
model is resident at a time; `ActivePackage` records the package directory and
its `manifest.json` so per-turn capability resolution has context.
`src-tauri/src/ai_engine/manager.rs:105-260`

**This system is out of scope and is not modified by any recommendation here.**

### 2.3 The LoRA layer: three distinct components **[CODE]**

| Component | File | Responsibility |
|---|---|---|
| `IntentClassifier` | `capability/classifier.rs` | Keyword-signal scoring of the prompt → intent + confidence |
| `CapabilityTracker` / `SwitchPolicy` | `capability/policy.rs` | Hysteresis — decides whether to *actually* switch |
| `CapabilityResolver` | `capability/resolver.rs` | Binds a capability to a concrete backend (`Base`, `PromptProfile`, or `LoraAdapter`) |
| `LoraAdapterCache` | `ai_engine/lora_binding.rs` | Owns initialised `LlamaLoraAdapter` handles, keyed by absolute path |
| `AdapterRegistry` | `adapter_manager/mod.rs` | `manifest.json` read/write, startup scan, self-healing |

The header comment in `capability/mod.rs` states the design intent directly: this
module *is* "the working replacement for what the build plan called the 'Dynamic
LoRA Switching Engine'." **[CODE]** `src-tauri/src/capability/mod.rs:1-6`

### 2.4 The three-tier capability backend **[CODE]**

```rust
pub enum CapabilityBackend {
    Base,                                    // unmodified base model
    PromptProfile,                           // system directive + sampling overrides
    LoraAdapter { path: PathBuf, scale: f32 },  // GGUF adapter bound to live context
}
```
`src-tauri/src/capability/profile.rs:66-75`

This is the key architectural decision. A capability **always** resolves to
something usable: if no loadable GGUF adapter exists, the same capability is
delivered through a prompt profile instead of being dropped. The inference path
does not branch on which. `src-tauri/src/capability/resolver.rs:1-12`

### 2.5 On-disk layout **[CODE]**

```
%APPDATA%/com.sarathi.app/models/<provider>/<sanitized-model-id>/
├── manifest.json                      # ModelPackageManifest
├── base/<model>.gguf
└── adapters/
    ├── coding/                        # capability slot (discovery-populated)
    │   ├── adapter_config.json
    │   └── adapter_model.safetensors  # ← often PEFT; NOT loadable
    ├── reasoning/ | mathematics/ | tool-calling/ | research/
    └── <sanitized-repo-id>/           # user-installed slot
        ├── adapter.gguf               # ← GGUF only, enforced
        └── source.txt
```
`src-tauri/src/adapter_manager/mod.rs:63-67`, `src-tauri/src/adapter_manager/store.rs:1-19`

---

## 3. Current Request/Response Flow

Sarathi has **two entry paths**, and they behave differently with respect to LoRA.

### 3.1 Path A — Desktop UI (capability routing active) **[CODE]**

```
User types in Sarathi chat window
   │
   ▼
src/services/ai.service.ts → Tauri invoke("send_chat_message")
   │
   ▼
commands/inference.rs:102  send_chat_message
   │  ├─ memory_engine: extract facts from user turn
   │  └─ memory_engine: inject recalled context into system message
   ▼
ai_engine/manager.rs:293   InferenceManager::send_chat_message
   │
   ▼
ai_engine/manager.rs:360   prepare_capability_turn
   │  ├─ classify latest USER message only
   │  ├─ CapabilityTracker::decide  (hysteresis)
   │  ├─ CapabilityResolver::resolve → CapabilityBackend
   │  ├─ apply_directive   → system prompt
   │  ├─ apply_sampling    → temperature / top_p / …
   │  └─ emit "capability:changed" to the UI badge
   ▼
ai_engine/runtime.rs:483   generate_with_capability(msgs, params, Some(backend), cb)
   │  ├─ render prompt with the GGUF's own chat template
   │  ├─ create fresh LlamaContext
   │  ├─ LoraAdapterCache::get_or_init(model, path)   ← load if not cached
   │  ├─ lora_binding::bind_adapter(&mut ctx, adapter, scale)  ← llama_set_adapters_lora
   │  ├─ chunked prefill (cancellable)
   │  └─ token loop → callback
   ▼
emit "inference:token" per token → React state → rendered in chat
```

### 3.2 Path B — External providers via the local gateway (capability routing **NOT** active) **[CODE]**

```
Claude Code / opencode / openclaw / hermes-agent
   │  (launched by launcher/mod.rs with the gateway address in its env)
   ▼
HTTP → gateway/server.rs   (axum)
   │    POST /v1/chat/completions   (OpenAI protocol)
   │    POST /v1/messages           (Anthropic protocol)
   ▼
gateway/server.rs:211  submit()
   │    GenerationJob { messages, params, capability: None, origin: Gateway{..} }
   │                                      ^^^^^^^^^^^^^^^^^ always None
   ▼
ai_engine/scheduler.rs:172  GenerationScheduler::submit  → queue
   ▼
ai_engine/scheduler.rs:200  run_job (dedicated OS thread, strictly serialized)
   │    manager.generate_direct(&job.messages, &job.params, cb)
   │                            ^^^^^^^^^^^^^^^^^^^^^^^^^^ job.capability is NOT passed
   ▼
ai_engine/manager.rs:422  generate_direct → runtime.generate(...)
   ▼
ai_engine/runtime.rs:462  generate → generate_with_capability(.., None, ..)
   │                                                          ^^^^ base model
   ▼
SSE / streaming JSON back through gateway → provider → user's terminal
```

**Registered providers** (`launcher/spec.rs:363` `builtin_tools()`): **[CODE]**

| id | Protocol | Line |
|---|---|---|
| `claude-code` | Anthropic | `launcher/spec.rs:366` |
| `opencode` | OpenAI | `launcher/spec.rs:435` |
| `hermes-agent` | OpenAI | `launcher/spec.rs:510` |
| `openclaw` | OpenAI | `launcher/spec.rs:578` |

> **"FlowrdCode" does not exist in this repository.** A case-insensitive search
> across `src/`, `src-tauri/src/`, and `docs/` returns no matches. If this refers
> to a planned provider, it is not yet present. **[CODE]**

### 3.3 Component responsibility map

| Question | Answer | Location |
|---|---|---|
| **Who selects the adapter?** | `IntentClassifier` proposes; `CapabilityTracker` decides (hysteresis); `CapabilityResolver` binds it to a real file. **Desktop path only.** | `capability/classifier.rs`, `policy.rs`, `resolver.rs` |
| **Who loads/switches it?** | `LoraAdapterCache::get_or_init` loads (llama.cpp `llama_adapter_lora_init`); `lora_binding::bind_adapter` binds it to the fresh context (`llama_set_adapters_lora`). | `ai_engine/lora_binding.rs:91,128` |
| **Who generates?** | `LlamaCppRuntime::generate_with_capability`, on the scheduler's single worker thread. | `ai_engine/runtime.rs:483` |
| **Who returns/displays it?** | Desktop: `inference:token` Tauri events → React. Gateway: SSE or JSON via `gateway/openai.rs` / `gateway/anthropic.rs` → the provider's own UI. | `commands/inference.rs`, `gateway/` |

---

## 4. Current LoRA Switching Flow

### 4.1 The switching mechanism, precisely **[CODE]**

Sarathi does **not** restart `llama-server` with a new `--lora` flag. The
`lora_binding.rs` header records why that plan was abandoned: `llama-cpp-2`
exposes `llama_adapter_lora_init` and `llama_set_adapters_lora`, so an adapter
binds to an existing context with no model reload, no process restart, and no
queueing. `src-tauri/src/ai_engine/lora_binding.rs:1-8`

The actual per-turn sequence:

1. A **fresh `LlamaContext` is created for every generation**
   (`runtime.rs:624`). This is important: there is never a stale binding to
   clear, because the context that held it no longer exists.
2. `LoraAdapterCache::get_or_init(model, path)` returns a cached
   `LlamaLoraAdapter`, or initialises it from disk on first use
   (`lora_binding.rs:91`).
3. `bind_adapter(&mut ctx, adapter, scale)` binds it **before the prefill decode**
   so the prompt is processed against adapted weights (`runtime.rs:628-655`).
4. Binding latency is logged in **microseconds**; adapter *initialisation* is
   logged in milliseconds (`lora_binding.rs:109-142`).

### 4.2 Why the cache exists — and why it cannot evict **[CODE]**

`LlamaLoraAdapter` has no `Drop` implementation in `llama-cpp-2` 0.1.153 and its
inner pointer is `pub(crate)`, so `llama_adapter_lora_free` is unreachable from
outside the crate. Initialising an adapter per generation would leak its full
50–150 MB footprint *every turn*. Caching by path bounds the leak to one
allocation per distinct adapter for the model's lifetime.
`src-tauri/src/ai_engine/lora_binding.rs:10-21`

`LoraAdapterCache::clear()` is called on model unload; it drops Sarathi's handles
but cannot free the underlying allocation. **There is no eviction policy and no
resident-count bound.** `lora_binding.rs:78-86`, `runtime.rs:443-446`

### 4.3 Hysteresis — the anti-thrash policy **[CODE]**

`CapabilityTracker::decide` implements a confidence band with sustained-signal
requirements, so a borderline turn does not flip the adapter. The test names in
`capability/policy.rs` state the contract: `borderline_turns_do_not_thrash_the_adapter`,
`one_offhand_turn_does_not_release_the_capability`,
`alternating_conversation_switches_far_less_than_it_classifies`.
`src-tauri/src/capability/policy.rs:141,277,317,369`

### 4.4 Verified defects and gaps in the current LoRA path

These were found while tracing the code and are stated as findings, not
speculation.

| # | Finding | Evidence |
|---|---|---|
| **G1** | **`GenerationJob.capability` is a dead field.** `run_job` destructures the envelope and calls `manager.generate_direct(&job.messages, &job.params, cb)` — `job.capability` is never read. Even if the gateway populated it, no capability or LoRA adapter would be applied. | `ai_engine/scheduler.rs:53` (field), `:257` (ignored) |
| **G2** | **Gateway traffic never gets LoRA routing.** `capability: None` is set deliberately, with a documented rationale ("external tools send their own system prompts and sampling, and overriding those silently can break output they depend on"). This is an intentional decision, but combined with G1 it means the opt-in has no mechanism behind it. | `gateway/server.rs:222-225` |
| **G3** | **Two parallel routers exist; one is effectively dead.** `model_intelligence::AdapterRouter::select_adapter_for_prompt` is reachable only from the Tauri command `route_prompt_capability`, which **has no frontend caller** (no match in `src/`). The live router is `capability::IntentClassifier`. | `model_intelligence/adapter_router.rs:27`, `commands/intelligence.rs:74-103` |
| **G4** | **User-installed GGUF adapters are invisible to capability routing.** `download_adapter` writes to `adapters/<sanitized-repo-id>/adapter.gguf` and **does not update `manifest.adapters`**. `CapabilityResolver::try_bind_adapter` reads only `manifest.adapters.get(capability)`, and `perform_startup_scan` scans only the five fixed capability keys. So the one adapter type Sarathi *can* actually load is the one the router cannot reach. | `commands/adapters.rs:97-163` (no manifest write), `capability/resolver.rs:126-129`, `adapter_manager/mod.rs:295-297` |
| **G5** | **No revision pinning.** Every Hub fetch is hardcoded to `main`: discovery uses `/raw/main/adapter_config.json`, download uses `/resolve/main/{file}`. Adapter identity is `repo_id` alone, so a repository update silently changes behaviour with no code change. | `model_providers/huggingface/adapter_provider.rs:252,318`, `commands/adapters.rs:103` |
| **G6** | **The frontend "LoRA" surface is a stub.** `src/pages/LoRA.tsx` renders the literal text "LoRA Orchestration — Coming in Phase 6", and every function in `src/services/lora.service.ts` (`getLoRAs`, `loadAdapter`, `switchAdapter`, `composeAdapters`) returns empty or does nothing. The working LoRA UI is the capability badge plus the adapter cards on the Models page — not this page. Cosmetic, but it misrepresents a subsystem that does work. | `src/pages/LoRA.tsx:1-2`, `src/services/lora.service.ts:1-4` |

---

## 5. PEFT Analysis

All API facts in this section were verified against PEFT **v0.20.0** and
Transformers **v5.14.1** documentation fetched during this analysis. **[DOC]**

### 5.1 Core APIs **[DOC]**

| API | Verified signature / behaviour |
|---|---|
| `load_adapter` | `load_adapter(model_id, adapter_name, is_trainable=False, torch_device=None, autocast_adapter_dtype=True, ephemeral_gpu_offload=False, low_cpu_mem_usage=False, key_mapping=None, **kwargs)`. Docs state explicitly: "The new adapter is not automatically set as the active adapter." |
| `set_adapter` | `set_adapter(adapter_name: str, inference_mode: bool = False)`. Activates the named adapter(s); others remain resident but inert. (The `inference_mode` parameter was absent from the supplied research.) |
| `delete_adapter` | `delete_adapter(adapter_name: str)`. Removes weights and config entry, freeing memory. |
| `disable_adapter` | Context manager on `PeftModel` (`with model.disable_adapter():`). Bulk `disable_adapters()`/`enable_adapters()` on the Transformers mixin. |
| `add_adapter` | Attaches a new, **untrained** adapter. Training-time; not relevant to inference integration. |
| `get_layer_status` | `PeftModel.get_layer_status()` and module-level `peft.get_layer_status(model)` → `list[TunerLayerStatus]`. |
| `get_model_status` | `PeftModel.get_model_status()` and module-level `peft.get_model_status(model)` → `TunerModelStatus`. |

### 5.2 The `"irregular"` state is real **[DOC]**

Confirmed verbatim in the v0.20.0 reference. `TunerModelStatus` fields
`enabled`, `active_adapters`, `merged_adapters`, `requires_grad`, and
`quantization_backend` are each typed as their normal type **or**
`Literal["irregular"]`, documented as: *"If some are enabled and some are not,
this will be `\"irregular\"`… which means that your model is in an inconsistent
state and might not work as expected."*

This is the single most valuable idea in the supplied research, and — notably —
Sarathi does not need it, because its equivalent state cannot go irregular (§6.3).

### 5.3 Transformers integration **[DOC]**

The `PeftAdapterMixin` on every `PreTrainedModel` requires **`peft >= 0.19.1`**
(the supplied research's version claim is correct). It adds `add_adapter`,
`load_adapter`, `set_adapter`, `enable_adapters`, `disable_adapters`,
`active_adapters`, `delete_adapter`, and `enable_peft_hotswap`.

### 5.4 Multiple adapters, mixed ranks **[DOC]**

Confirmed. Each adapter's `lora_A`/`lora_B` live in their own `ModuleDict` entry,
so `adapter_1` at `r=8` and `adapter_2` at `r=16` coexist on one model with no
special configuration. The docs demonstrate `model.set_adapter(["adapter1",
"adapter2"])` to activate several simultaneously.

### 5.5 Loaded / selected / enabled / disabled / base-only **[DOC]**

| State | Achieved by | Verified by |
|---|---|---|
| Loaded | `load_adapter()` | name in `available_adapters` |
| Selected | `set_adapter(name)` | `active_adapters == [name]` |
| Enabled | default; `enable_adapters()` | `enabled is True` |
| Disabled | `disable_adapters()` / `with model.disable_adapter():` | `enabled is False` |
| Base-only | disabled, or never activated | `active_adapters == []` or `enabled is False` |

The genuinely subtle case is **loaded + selected + disabled**: checking
`active_adapters` alone will report an adapter that is contributing nothing.

### 5.6 Hotswapping **[DOC]**

Verified against the PEFT hotswap reference and the Transformers PEFT guide:

- `peft.utils.hotswap.hotswap_adapter(model, model_name_or_path, adapter_name, torch_device=None, **kwargs)`
- `prepare_model_for_compiled_hotswap(model, target_rank=max_rank)` — must be
  called **before** the first adapter load and before `torch.compile()`.
- Transformers equivalent: `model.enable_peft_hotswap(target_rank=...)` then
  `model.load_adapter(path, hotswap=True, adapter_name="default")`.
  `target_rank` defaults to **128**. After `enable_peft_hotswap`, subsequent
  `load_adapter` calls hotswap by default; pass `hotswap=False` to opt out.

Documented caveats, verbatim in substance:
- Only LoRA is supported; no swapping between PEFT method types.
- The incoming adapter must target **the same layers or a subset** — never new
  layers. "If possible, start with the adapter that targets most layers."
- Incompatible adapters raise `RuntimeError` rather than corrupting state.
- `target_parameters`-based LoRA cannot avoid recompilation/graph breaks.

**Relevance to Sarathi: none.** Hotswapping's entire value proposition is
avoiding `torch.compile` recompilation. Sarathi does not use `torch.compile`,
because it does not use torch.

### 5.7 Hub integration **[DOC]**

PEFT delegates to `huggingface_hub`: repo-id or local path are interchangeable,
`revision=` pins a commit/tag/branch, downloads go through the shared cache
(`HF_HOME` / `HF_HUB_CACHE`), `HF_HUB_OFFLINE=1` or `local_files_only=True`
forces cache-only resolution, and `token=`/`HF_TOKEN` handles gated repos.
Required files: `adapter_config.json` + `adapter_model.safetensors` (or the
legacy `.bin`).

### 5.8 Compatibility checking and failure behaviour **[DOC] [EXT]**

- **Target-module mismatch** → `ValueError: Target modules [...] not found in the base model.`
- **Shape mismatch** → `RuntimeError: size mismatch for ... lora_B...` from `load_state_dict`.
- **`base_model_name_or_path` is metadata, not enforced.** PEFT will happily load
  an adapter onto a differently-named base model if the structural checks pass.
  An application-level check is a defensive addition PEFT does not make for you.
- `inject_adapter` validates before mutating, so a failed load of a *new* adapter
  does not disturb an already-active one. **[EXT]**

### 5.9 Concurrency **[EXT]**

`active_adapter` is **global mutable state on the shared model object**. PEFT
maintainers have characterised `set_adapter()` as equivalent to a global mode
setting (PEFT issue #804). Two threads calling `set_adapter` on the same model
instance can produce a forward pass that mixes both adapters' weights — a wrong
answer, not a crash. PEFT provides **no built-in locking**; the serving layer must
supply it.

---

## 6. Compatibility Assessment

### Verdict

| Role | Verdict | Basis |
|---|---|---|
| PEFT as Sarathi's **runtime adapter-switching layer** | **INCOMPATIBLE** | Runtime mismatch — no PyTorch module graph exists to attach to. **[CODE]** + **[DOC]** |
| PEFT as an **offline / build-time adapter ingestion tool** | **PARTIALLY COMPATIBLE — recommended** | Converts unusable PEFT safetensors into GGUF that the existing binding path already loads. **[REC]** |
| PEFT-**format** checkpoints as an interchange standard | **ALREADY COMPATIBLE** | Sarathi already parses `adapter_config.json` for `peft_type`, `base_model_name_or_path`, and `target_modules`. **[CODE]** |

### 6.1 Why "incompatible" is structural, not a gap to be closed

PEFT's mechanism is described precisely in its own source: `BaseTuner.inject_adapter()`
performs one-time in-place surgery on the model graph, replacing each targeted
`nn.Linear` with a `lora.Linear` holding per-adapter `ModuleDict`s; `set_adapter`
then flips the `active_adapter` attribute on every `BaseTunerLayer`. **[DOC] [EXT]**

Sarathi's model is a `llama_cpp_2::model::LlamaModel` — a `NonNull` pointer into
a C++ heap allocation, wrapped in Rust. **[CODE]** `ai_engine/lora_binding.rs:37-48`
There are no `nn.Linear` objects, no `ModuleDict`s, and no Python attribute to
flip. Making PEFT work would require:

1. Adding a Python runtime and PyTorch to a desktop app that currently ships a
   single native binary, **and**
2. Replacing llama.cpp with `transformers` for inference, **and**
3. Abandoning GGUF quantisation and the VRAM planner built around it.

That is a replacement of the model engine, which is out of scope by explicit
instruction — and would be a large regression for a local-first desktop app
regardless.

### 6.2 The format gap, restated as the real problem

Sarathi's own comments already frame this correctly:

> "public GGUF LoRA adapters for current base models are scarce, and PEFT
> safetensors adapters — which is what HuggingFace discovery actually finds —
> cannot be loaded by llama.cpp without conversion."
> **[CODE]** `capability/resolver.rs:8-10`

So the problem PEFT could plausibly solve for Sarathi is **not** "how do I switch
adapters" (solved) but **"how do I make the adapters people actually publish
loadable at all"** (unsolved). That is a conversion problem, not a switching
problem.

### 6.3 Feature-by-feature: what Sarathi already has

| PEFT capability | Sarathi equivalent | Assessment |
|---|---|---|
| `load_adapter` | `LoraAdapterCache::get_or_init` — lazy, cached by path | **Equivalent** |
| `set_adapter` | `bind_adapter` on a fresh context per turn (µs-scale) | **Equivalent**; no stale-binding class of bug by construction |
| Multiple resident adapters | `HashMap<PathBuf, SendAdapter>` | **Equivalent** |
| `delete_adapter` | `LoraAdapterCache::clear()` on model unload only | **Weaker** — no per-adapter eviction (upstream FFI limit, see §15) |
| `disable_adapters` | `CapabilityBackend::Base` / omitting the bind | **Equivalent** |
| `get_model_status` → `"irregular"` | Not needed — a fresh context per turn cannot desynchronise across layers | **N/A by construction** |
| Structural validation | `verify_gguf_magic`, `gguf::verify_is_lora_adapter`, `validate_adapter` | **Stronger** — Sarathi checks magic bytes *and* GGUF metadata, and enforces a 2 GB ceiling |
| Base-model compatibility check | `verify_candidate` matches `base_model_name_or_path` against model aliases | **Stronger** — PEFT explicitly does *not* enforce this |
| Thread safety | `GenerationScheduler`: one dedicated OS thread, all model access serialized | **Stronger** — PEFT ships no locking at all |
| Failure handling | Degrade to `PromptProfile`, then `Base`; never fatal | **Stronger** — PEFT raises |
| Revision pinning | Absent; hardcoded `main` | **Weaker** (gap G5) |
| Caching / eviction policy | Absent | **Equal** — PEFT has none either |
| Semantic adapter selection | `IntentClassifier` + hysteresis | **Stronger** — PEFT has none (§12.4) |

**Nine of thirteen capabilities are equivalent or stronger in Sarathi today.**
The two genuine weaknesses (eviction, revision pinning) are not solved by PEFT —
PEFT has no eviction policy either, and revision pinning is a two-line URL change
in Sarathi's existing `reqwest` calls.

---

## 7. Exact Integration Points

Given the verdict, "integration" means one narrow, additive seam. **[REC]**

### 7.1 The single recommended seam: an offline conversion step

```
model_providers/huggingface/adapter_provider.rs   discovery (unchanged)
        │  finds PEFT safetensors adapter
        ▼
download_manager/manager.rs                       download (unchanged)
        │  writes adapters/<capability>/adapter_model.safetensors
        ▼
lora/validator.rs::validate_adapter               (unchanged)
        │  → AdapterRuntimeStatus::RequiresConversion
        ▼
╔════════════════════════════════════════════════════════════╗
║  NEW, ADDITIVE:  lora/convert.rs                           ║
║  Invokes llama.cpp's convert_lora_to_gguf.py out-of-proc   ║
║  writes adapters/<capability>/adapter.gguf                 ║
╚════════════════════════════════════════════════════════════╝
        ▼
lora/validator.rs::validate_adapter               (unchanged, re-run)
        │  → AdapterRuntimeStatus::Compatible
        ▼
capability/resolver.rs::try_bind_adapter          (unchanged)
        │  GGUF magic verified → CapabilityBackend::LoraAdapter
        ▼
ai_engine/lora_binding.rs                         (unchanged)
        │  get_or_init → bind_adapter
        ▼
ai_engine/runtime.rs::generate_with_capability    (unchanged)
```

**Files that would change: one new module, one status transition. The model
engine, the runtime, the scheduler, the resolver, and the binding path are all
untouched.**

### 7.2 What PEFT itself contributes here

Honest accounting: **llama.cpp's `convert_lora_to_gguf.py` does not require the
`peft` package.** It reads `adapter_config.json` and the safetensors weights
directly using `torch`/`safetensors`/`gguf`. **[EXT]**

PEFT is therefore **optional** in this path. It earns its place only if Sarathi
wants:

- **Config normalisation/validation** — loading `adapter_config.json` through
  `peft.LoraConfig` to catch malformed or non-LoRA configs with a real error
  before conversion, rather than after.
- **Merge-and-export** — `merge_and_unload()` to produce a merged model for
  adapters that resist conversion. (One-way; see §15.)

**[REC]** Start without PEFT — the converter alone closes the gap. Add
`peft.LoraConfig` validation only if malformed configs prove to be a real source
of conversion failures in practice.

### 7.3 Integration points that are explicitly NOT recommended

| Idea | Why not |
|---|---|
| Embed a Python inference sidecar running `transformers` + PEFT | Duplicates the model in RAM, abandons GGUF quantisation, doubles VRAM planning complexity, and replaces the engine — out of scope |
| Call PEFT via `pyo3` from Rust | Same problem; PEFT still needs a PyTorch model, which Sarathi does not have |
| Reimplement `set_adapter` semantics in Rust to "match PEFT" | Sarathi's fresh-context-per-turn design is already safer; matching a weaker model would be a regression |

---

## 8. Adapter Download / Cache Integration

### 8.1 What Sarathi does today **[CODE]**

Sarathi does **not** use `huggingface_hub`. It issues raw `reqwest` calls:

| Purpose | URL |
|---|---|
| Search | `https://huggingface.co/api/models?filter=lora&search={q}&limit=10` |
| Config probe | `https://huggingface.co/{repo}/raw/main/adapter_config.json` |
| Size probe | `HEAD https://huggingface.co/{repo}/resolve/main/{file}` |
| Download | `GET https://huggingface.co/{repo}/resolve/main/{file}?download=true` |

`adapter_provider.rs:151,252,318`, `commands/adapters.rs:70,103`

Auth is a process-global token loaded from settings at startup
(`config/hf_token.rs`, wired in `lib.rs:80-98`). Cache is the package directory
itself — there is no separate blob cache, and no offline mode flag.

### 8.2 Comparison with PEFT/`huggingface_hub` **[DOC]**

| Concern | Sarathi today **[CODE]** | `huggingface_hub` **[DOC]** |
|---|---|---|
| Cache location | Package dir, keyed by capability or repo dir name | `HF_HOME`/`HF_HUB_CACHE`, keyed by repo + revision |
| Revision pinning | **None** — hardcoded `main` (G5) | `revision=` parameter |
| Offline mode | None | `HF_HUB_OFFLINE=1` / `local_files_only=True` |
| Auth | Process-global bearer token | `token=` / `HF_TOKEN` |
| Integrity | GGUF magic + metadata + 2 GB ceiling + ≥100 KB floor | ETag/hash-based cache validation |
| Resumable | No — `resp.bytes()` buffers whole file | Yes |

### 8.3 Recommendation **[REC]**

**Do not adopt `huggingface_hub`.** It is a Python dependency, and Sarathi's
download path is Rust. The two genuine gaps are cheap to close in the existing
code:

1. **Pin revisions (closes G5).** Extend `AdapterManifestInfo` with an optional
   `revision: Option<String>`, resolve the current commit SHA from the Hub API at
   download time, and record it. Treat adapter identity as `(repo_id, revision)`.
   The URLs already accept a SHA in place of `main` — this is a substitution, not
   a redesign.
2. **Consider `hf-hub` (the Rust crate)** if resumable downloads and ETag caching
   become worth the dependency. This is optional and orthogonal to PEFT.

---

## 9. Dynamic Switching Flow

The flow requested in the brief, mapped onto Sarathi's **actual** components.
Every step below already exists in code except the two marked **[GAP]**.

```
User request (desktop chat)
   │
   ▼
① Determine required adapter
   IntentClassifier::classify(prompt)  →  intent + confidence
   capability/classifier.rs:235                                        [CODE ✓]
   │
   ▼
② Compare with current adapter
   CapabilityTracker::decide(&classification, manual_override)
   Hysteresis band; manual override wins.
   capability/policy.rs:141                                            [CODE ✓]
   │
   ├── SAME  → SwitchDecision::Hold
   │            Adapter handle stays cached; a fresh context still
   │            re-binds it, at microsecond cost.                      [CODE ✓]
   │
   └── DIFFERENT → commit switch
         │
         ▼
   ③ Resolve to a concrete backend
      CapabilityResolver::resolve(capability, package_dir, manifest)
      capability/resolver.rs:72                                        [CODE ✓]
         │
         ├── manifest has no entry            → PromptProfile
         ├── status != "Installed"            → PromptProfile
         ├── runtime_status = requires_conversion → PromptProfile  ← PEFT safetensors
         ├── runtime_status = incompatible    → PromptProfile
         ├── file is not .gguf                → PromptProfile
         ├── file missing on disk             → PromptProfile
         └── GGUF magic verified              → LoraAdapter{path, scale}
         │
         ▼
   ④ If not loaded → load it
      LoraAdapterCache::get_or_init(model, path)
      Cache hit: return handle. Miss: llama_adapter_lora_init from disk.
      ai_engine/lora_binding.rs:91                                     [CODE ✓]
         │
         ▼
   ⑤ If not installed → resolve / download / cache
      HuggingFaceAdapterProvider::discover_adapters (async, 5 capabilities)
      → download_manager writes into adapters/<capability>/
      → AdapterRegistry::write_manifest (single-source-of-truth merge)
      adapter_provider.rs:218, adapter_manager/mod.rs:223              [CODE ✓]
      ⚠ Runs on model install / explicit refresh, NOT inline per request.
      A cold adapter does not block a turn; the turn degrades instead.  [REC: correct]
         │
         ▼
   ⑥ Verify actual state
      verify_gguf_magic(path)                capability/resolver.rs:180 [CODE ✓]
      gguf::verify_is_lora_adapter(path)     adapter_manager/gguf.rs    [CODE ✓]
      bind_adapter → Result; failure is logged, not fatal
      ai_engine/runtime.rs:636-654                                     [CODE ✓]
      ⚠ [GAP] No post-bind read-back. llama.cpp/llama-cpp-2 0.1.x
        exposes no "which adapters are bound to this context" query,
        so there is no equivalent of get_model_status(). The bind
        result code is the only available confirmation.
         │
         ▼
   ⑦ Generate
      LlamaCppRuntime::generate_with_capability
      Prefill (chunked, cancellable) → token loop → callback
      ai_engine/runtime.rs:483                                         [CODE ✓]
         │
         ▼
   ⑧ Keep current adapter active
      CapabilityTracker retains the capability across turns (stickiness).
      The adapter HANDLE stays cached. The BINDING does not persist —
      each turn creates a fresh context and re-binds.
      capability/policy.rs:219, ai_engine/runtime.rs:624               [CODE ✓]
```

### 9.1 Two honest differences from the PEFT-shaped flow in the brief

**Difference 1 — "keep current adapter active" means something different here.**
In PEFT, the model object persists and `active_adapter` persists with it, so a
repeat request costs literally zero. In Sarathi, the context is destroyed and
recreated each turn, so the adapter is re-bound every time. The cost is a cached
handle plus one `llama_set_adapters_lora` call, logged in **microseconds**
(`lora_binding.rs:138-142`). This is not a deficiency: it is what makes an entire
class of stale-binding and irregular-state bugs structurally impossible.

**Difference 2 — download is not inline.** Step ⑤ runs at install/refresh time,
not inside the request path. A capability whose adapter is not yet present
degrades to its prompt profile for that turn instead of stalling the user behind
a network fetch. **[REC]** This is the right trade-off for a desktop app and
should be preserved.

---

## 10. Verification and Failure Handling

### 10.1 What happens when loading fails **[CODE]**

Sarathi's failure model is *degradation*, applied at four layers:

| Layer | Failure | Result |
|---|---|---|
| Discovery | No verified adapter for the base model | `status: "Unavailable"`, capability handled natively — `adapter_provider.rs:205-213` |
| Download | Repo ships full model weights, or file > 2 GB | Refused **before** bytes are fetched — `store.rs:check_installable`, `commands/adapters.rs:117-126` |
| Post-download | File is not a LoRA adapter GGUF | Directory deleted, error returned — `commands/adapters.rs:139-142` |
| Resolution | Missing / non-GGUF / bad magic / `requires_conversion` | `CapabilityBackend::PromptProfile` with a human-readable reason — `capability/resolver.rs:100-107` |
| Binding | `lora_adapter_init` or `lora_adapter_set` fails | **Logged as a warning; generation proceeds on the base model** — `runtime.rs:645-654` |

The contract is stated in the runtime's own doc comment: *"A LoRA binding failure
is never fatal: it is logged and generation proceeds on the base model."*
`ai_engine/runtime.rs:480-482`

### 10.2 How the previous valid state is retained

**Sarathi does not need the "snapshot → attempt → verify → rollback" wrapper the
research recommends for PEFT, because it has no mutable global adapter state to
roll back.** **[REC]**

The reasoning is structural:

- PEFT needs rollback because `set_adapter` mutates a long-lived shared object.
  If the call throws midway, the model can be left in an ambiguous state.
- Sarathi binds to a **context created fresh for this generation**
  (`runtime.rs:624`). If `bind_adapter` fails, the only consequence is that this
  one context has no adapter — so this one turn runs on the base model. The next
  turn gets a brand-new context and retries from scratch. There is no persistent
  state to corrupt and nothing to restore.
- The one piece of state that *does* persist across turns is the
  `CapabilityTracker`'s active capability. It is guarded against lock poisoning
  taking inference down: `.unwrap_or_else(|poisoned| poisoned.into_inner())`
  (`capability/mod.rs:145-148`).
- The manifest has its own protection: `write_manifest` merges against the
  existing file and **refuses to downgrade** an `Installed`/`READY` adapter when
  its files still exist on disk, logging a `SingleSourceOfTruthProtection`
  transition. `adapter_manager/mod.rs:229-260`

### 10.3 The one real verification gap **[CODE]**

Sarathi verifies extensively **before** binding (magic bytes, GGUF metadata, file
presence, size bounds, `peft_type`, base-model alias match) but has **no
post-bind read-back**. PEFT's `get_model_status()` has no equivalent here because
`llama-cpp-2` 0.1.x exposes no query for "which adapters are bound to this
context."

The available signal is the `Result` from `ctx.lora_adapter_set(...)`, which is
checked (`lora_binding.rs:135-136`). **[REC]** Given a fresh context per turn and
a single serialized worker thread, the risk this gap represents is low — there is
no concurrent mutation that could desynchronise layers. Recording
`active_adapter_label` into `LoadedModelInfo.active_adapter` (the field already
exists at `ai_engine/traits.rs:197` but is only ever set to `None` at
`runtime.rs:351`) would close the observability side cheaply.

---

## 11. Concurrency and Performance

### 11.1 Concurrency: Sarathi is safer than raw PEFT **[CODE]** vs **[EXT]**

| | PEFT | Sarathi |
|---|---|---|
| Model access | Any thread, unguarded | One dedicated OS thread (`sarathi-generation`) |
| Adapter mutation | Global `active_adapter` on a shared object | Per-context bind, context is thread-local to the worker |
| Locking | **None provided** | Structural — the queue *is* the lock |
| Concurrent different-adapter requests | Can mix weights mid-forward-pass (issue #804) | Impossible; requests queue with a reported position |
| Cancellation | Not addressed | Cancel flag cloned before generation; `CancelOnDrop` guard catches client hangups |

`ai_engine/scheduler.rs:143-159` (worker thread), `:103-125` (`CancelOnDrop`),
`:236-253` (prefill-phase cancel watcher)

The scheduler's own header states the design: *"Only one model fits in VRAM, and
llama.cpp generation is blocking, so exactly one generation can run at a time."*
This is precisely the "serialize all adapter-affecting operations" pattern the
research recommends bolting onto PEFT — except here it is the architecture rather
than a wrapper.

**Trade-off, stated honestly:** this caps throughput at one generation at a time.
Sarathi cannot serve two different adapters concurrently. For a single-user
desktop app driving one local model on one GPU, that is the correct trade — and
it is the same limit raw PEFT has, minus PEFT's correctness hazard.

### 11.2 Performance characteristics **[CODE]**

| Operation | Cost | Evidence |
|---|---|---|
| Adapter switch (cached handle) | **Microseconds** — one `llama_set_adapters_lora` | `lora_binding.rs:138-142` logs `µs` |
| Adapter first load | **Milliseconds** — disk read + `llama_adapter_lora_init` | `lora_binding.rs:108-112` logs `ms` |
| Context creation (every turn) | Allocation proportional to `n_ctx` | `runtime.rs:619-626` |
| Prefill | **Dominant cost.** ~98 s measured for a coding agent's system prompt on a CPU-only build | `runtime.rs:659-664` |
| Per-token LoRA overhead | Small extra low-rank matmul per targeted layer | **[EXT]** |

**The adapter switch is not the bottleneck and will never be.** Prefill dominates
by three to five orders of magnitude. Any optimisation effort aimed at switching
latency is misdirected.

### 11.3 Memory **[CODE]**

- Base model VRAM is unaffected by adapter count.
- Each *distinct* adapter costs one 50–150 MB allocation for the model's lifetime.
- **The allocation cannot be freed** — `llama-cpp-2` 0.1.153 exposes no
  destructor. Bounded at "a few hundred MB across all capabilities," reclaimed
  only at process exit. `lora_binding.rs:10-21`
- With five capability slots, worst case is roughly 250–750 MB. Acceptable today;
  it becomes a real problem if the adapter catalogue grows (§15).

---

## 12. Open-Source Alternatives

Adapter management, switching, hotswapping, caching, and routing only.

### 12.1 Comparison **[DOC]** except where marked

| Capability | HF PEFT | vLLM Multi-LoRA | SGLang | LoRAX | **Sarathi (llama.cpp)** |
|---|---|---|---|---|---|
| Runtime | PyTorch | PyTorch/CUDA | PyTorch/CUDA | PyTorch/CUDA | **Native C++/Rust** |
| Adapter format | PEFT safetensors | PEFT | PEFT | PEFT | **GGUF** |
| Concurrent multi-adapter batching | No | Yes (`LoRARequest`) | Yes (`--max-loras-per-batch`, default 8) | Yes (SGMV) | No |
| Runtime load, no restart | Yes (your code) | `/v1/load_lora_adapter` | `/load_lora_adapter` | Just-in-time per request | Yes (`get_or_init`) **[CODE]** |
| Eviction policy | **None** | LRU over `--max-loras`/`--max-cpu-loras` | `--max-loaded-loras`, `--lora-eviction-policy` (`lru`\|`fifo`) | Tiered GPU/CPU/disk | **None** **[CODE]** |
| Mixed ranks | Yes | Yes (`--max-lora-rank` ceiling) | Yes | Yes | Yes (per-file) |
| Hotswap-in-place | `hotswap_adapter` | `load_inplace=True` | — | — | N/A |
| Batched-adapter kernel | None | Punica-derived | Triton / chunked SGMV (`csgmv`) | SGMV | None |
| HTTP serving surface | **None** | OpenAI-compatible | OpenAI-compatible | OpenAI-compatible | **OpenAI + Anthropic** **[CODE]** |
| Semantic adapter selection | **None** | None | None | None | **`IntentClassifier` + hysteresis** **[CODE]** |
| Desktop / CPU / consumer GPU viable | No | No | No | No | **Yes** |

### 12.2 The load-bearing observation **[EXT]**

vLLM, SGLang, and LoRAX all consume PEFT-format checkpoints as an *interchange
format* while replacing PEFT's runtime switching with kernel-level per-request
adapter batching. None uses `set_adapter()` for concurrent mixed-adapter serving.
Published benchmarks treat PEFT sequential serving as the **baseline these
systems are built to beat** — S-LoRA reports up to ~4×, Punica up to ~12× over
PEFT/Transformers baselines in the multi-adapter-per-batch regime.

**Implication for Sarathi:** even if the runtime barrier did not exist, adopting
PEFT would mean adopting the slow baseline. Sarathi's llama.cpp path is the only
one in this table that runs on a consumer laptop without CUDA.

### 12.3 Named systems from the brief

- **Shiftgate** — **not found.** No LoRA serving system by this name surfaced in
  search. If it exists it is not publicly indexed; it cannot be assessed here.
- **LORAUTER** — **real and relevant.** *"Effective LoRA Adapter Routing using
  Task Representations"* ([arXiv:2601.21795](https://arxiv.org/abs/2601.21795),
  January 2026). Routes queries via **task embeddings** derived from small
  validation sets rather than mapping queries directly to adapters, so routing
  cost scales with the number of *tasks* rather than the number of *adapters*.
  Requires no adapter training data, and the code is open-sourced. **[EXT]**

  **[REC]** This is the most interesting external result for Sarathi's *router*,
  which is currently keyword-signal scoring (`capability/classifier.rs:52-195`).
  It is entirely orthogonal to the PEFT question — LORAUTER decides *which*
  adapter, Sarathi's existing binding path handles *how*. Worth a separate
  evaluation; out of scope here.

- **NVIDIA Dynamo** — also offers runtime LoRA load/unload, caching, and
  adapter-aware request routing. Same PyTorch/datacenter constraint. **[EXT]**

### 12.4 Semantic selection: nobody else has it

PEFT, vLLM, SGLang, and LoRAX all require the caller to already know the adapter
name. PEFT's closest offering is **X-LoRA**, a trained gating architecture that
*blends* several loaded adapters' numerical contributions — not a
route-to-a-discrete-adapter decision system. **[DOC]**

Sarathi's `IntentClassifier` + `CapabilityTracker` occupies a layer that none of
these systems provide. It is a genuine differentiator and should be preserved.

---

## 13. Recommended Architecture

### 13.1 Keep everything; add one converter **[REC]**

```
                        UNCHANGED
   ┌──────────────────────────────────────────────────────────┐
   │  Intent classification → hysteresis → capability          │
   │  Manifest registry → resolver → GGUF verification         │
   │  LoraAdapterCache → bind_adapter → llama.cpp context      │
   │  GenerationScheduler → single worker thread               │
   │  Gateway (OpenAI + Anthropic) → providers                 │
   └──────────────────────────────────────────────────────────┘
                              ▲
                              │  feeds loadable GGUF adapters
                              │
   ┌──────────────────────────────────────────────────────────┐
   │  NEW: offline PEFT → GGUF conversion                      │
   │  llama.cpp convert_lora_to_gguf.py, invoked out-of-proc   │
   │  Triggered on RequiresConversion; never on the hot path   │
   └──────────────────────────────────────────────────────────┘
```

### 13.2 Why this is the right shape **[REC]**

1. **It uses an existing open-source mechanism** rather than a custom one, as the
   brief requires. `convert_lora_to_gguf.py` is maintained in llama.cpp — the same
   project supplying Sarathi's runtime, so converter and loader stay in lockstep.
2. **It touches no engine code.** The conversion writes a `.gguf` next to the
   safetensors; `validate_adapter` reclassifies it `Compatible` on the next scan;
   `try_bind_adapter` picks it up with no change. The path from resolver to
   `llama_set_adapters_lora` is already correct.
3. **It is the highest-leverage change available.** Today, discovery finds PEFT
   adapters and the resolver rejects every one of them — so the LoRA backend is
   largely unreachable in practice, and capabilities silently run as prompt
   profiles. Conversion is what turns the existing LoRA path from theoretical
   into used.
4. **It hardcodes nothing.** No model, GPU, adapter, or adapter count. The
   converter runs per adapter directory found in `RequiresConversion` state.

### 13.3 A caveat worth stating plainly **[REC]**

Conversion is not free of risk. It requires Python plus `torch`, `safetensors`,
and `gguf` on the user's machine — which conflicts with Sarathi shipping as a
self-contained desktop binary. Three options, in order of preference:

| Option | Trade-off |
|---|---|
| **(a) Opt-in, detected** | Detect a usable Python; offer conversion only when present. Users without it see today's behaviour exactly. Lowest risk, and Sarathi already runs Python sidecars, so an interpreter is often present. **Recommended.** |
| **(b) Server-side pre-conversion** | Convert adapters ahead of time and publish GGUF; the app only ever downloads GGUF. Zero client dependency, but needs infrastructure and curation. |
| **(c) Native Rust converter** | No Python dependency, full control — but reimplements a moving upstream target. Highest cost, highest maintenance. |

Per this project's Python policy (`CLAUDE.md`), option (a) must use the **system
interpreter** — no venv, virtualenv, or conda env.

### 13.4 Ordering note

Gaps **G1–G4** (§4.4) are cheaper to fix than the converter and independently
useful. **G4 in particular** — user-installed GGUF adapters being invisible to
the capability router — means Sarathi cannot currently route to the only adapter
format it can load. Fixing that unlocks the existing LoRA path with no new
dependency at all, and should come first.

---

## 14. Implementation Plan

No code is written by this document. Phases are ordered by value-per-risk. **[REC]**

### Phase 0 — Decide (no code)
Confirm the verdict in §6 and the ordering in §13.4. If PEFT-as-runtime is still
desired, that is a model-engine replacement and needs its own proposal.

### Phase 1 — Close the routing gaps (Rust only, no new dependencies)
1. **G4** — register user-installed GGUF adapters in `manifest.adapters` so the
   resolver can reach them. Either write the manifest entry in
   `commands/adapters.rs::download_adapter`, or extend `perform_startup_scan`
   beyond the five fixed capability keys.
2. **G1** — either wire `job.capability` through `run_job` into
   `generate_with_capability`, or delete the field. A struct field that silently
   does nothing is worse than either.
3. **G3** — remove `model_intelligence::AdapterRouter` and the unused
   `route_prompt_capability` command, or document them as a diagnostic surface.
4. **G5** — resolve and record a commit SHA per adapter; make identity
   `(repo_id, revision)`.
5. Populate `LoadedModelInfo.active_adapter` from `active_adapter_label`
   (`runtime.rs:633-643`) so the UI can show what is actually bound.

**Exit criterion:** an adapter installed through the Models UI is selectable by
the capability router and visibly bound during generation.

### Phase 2 — Conversion pipeline (the PEFT-adjacent work)
1. New `src-tauri/src/lora/convert.rs`: locate a system Python, verify
   `torch`/`safetensors`/`gguf` are importable, invoke
   `convert_lora_to_gguf.py` on an adapter directory, capture stderr.
2. Trigger it explicitly from the UI on `RequiresConversion` adapters — user
   action, never automatic, never on the request path.
3. Re-run `validate_adapter` afterwards; expect `Compatible`.
4. Existing verification (`verify_gguf_magic`, `verify_is_lora_adapter`) already
   guards the output. No new verification needed.
5. Surface progress and failure through the existing download/status event
   channels.

**Exit criterion:** a discovered PEFT safetensors adapter converts, validates as
`Compatible`, binds via the unchanged path, and shows `Code · lora` on the badge.

### Phase 3 — Bounded residency (optional; only if the catalogue grows)
`LoraAdapterCache` cannot free llama.cpp allocations (§4.2). If adapter count
grows enough to matter, the options are: track upstream `llama-cpp-2` for a
`Drop`/`free` API, contribute one, or bound distinct adapters per model session
and reload the model to reclaim. **Do not build an eviction policy that cannot
actually free memory** — it would report success while leaking.

### Phase 4 — Router quality (independent of everything above)
Evaluate LORAUTER-style task-representation routing (§12.3) against the current
keyword classifier. Separate proposal; separate decision.

### Explicitly out of scope, permanently
Model loading, model selection, GPU allocation, VRAM planning, the scheduler's
serialization model, and the gateway protocol handlers.

---

## 15. Risks and Limitations

### 15.1 Risks in what exists today **[CODE]**

| Risk | Severity | Detail |
|---|---|---|
| **Adapter memory cannot be freed** | Medium | No `Drop` in `llama-cpp-2` 0.1.153; `llama_adapter_lora_free` is unreachable. Bounded per model session, reclaimed only at exit. |
| **Unpinned revisions (G5)** | Medium | Every fetch is `main`. An upstream repo update changes behaviour with no code change and no audit trail. |
| **Untrusted adapter content** | Medium | GGUF magic + metadata + size bounds are checked, which is good. But an adapter is still third-party weights loaded from a Hub search result, and a shape-valid adapter can degrade output arbitrarily. There is no provenance model beyond `base_model_name_or_path` alias matching. |
| **Capability routing invisible to gateway users (G1/G2)** | Low–Medium | Deliberate for G2, but the mechanism behind the opt-in does not exist (G1). |
| **User-installed adapters unreachable by the router (G4)** | Medium | The only loadable format is the one the router cannot select. |
| **Silent degradation** | Low | Every failure path degrades to prompt profile or base with a logged reason — good. But a user seeing `Code · prompt-profile` may not realise the adapter never loaded. `backend_reason` is carried in the payload; ensure the UI surfaces it. |

### 15.2 Risks specific to a conversion pipeline **[REC]**

| Risk | Mitigation |
|---|---|
| Python/torch not present | Detect and offer conversion only when available; never regress the no-Python experience |
| Converter fails on an architecture | Treat as expected; leave status `RequiresConversion` and report why |
| Conversion output is wrong but structurally valid | Existing GGUF metadata verification catches format errors, not numerical ones. Consider a smoke generation after first conversion. |
| Upstream converter changes | Pin the llama.cpp revision the converter is taken from; it moves independently of `llama-cpp-2` |
| Disk growth | Converted GGUF sits alongside the safetensors. Offer to delete the source after successful conversion. |

### 15.3 Limitations inherited from PEFT that do **not** apply here

Recorded for completeness so future readers do not import worries that are not
Sarathi's: no concurrent multi-adapter batched inference (Sarathi is
single-generation by design), the `active_adapter` global-state race (no such
state exists), `torch.compile` recompilation (no torch), `merge_and_unload()`
destroying switchability (never called), and `"irregular"` layer desynchronisation
(impossible with a fresh context per turn).

---

## 16. Final Recommendation

> **"Can Sarathi use Hugging Face PEFT with its existing architecture? If yes,
> exactly how would an adapter be downloaded, loaded, switched, verified and used
> for generation, and how would the generated response travel back through
> Sarathi to the provider/user?"**

### 16.1 Direct answer

**No — not as a runtime adapter-switching layer, and the reason is structural
rather than a matter of effort.**

PEFT switches adapters by mutating a PyTorch `nn.Module` graph: it replaces
`nn.Linear` layers with `lora.Linear` wrappers holding per-adapter `ModuleDict`s,
then flips an `active_adapter` attribute on those Python objects. **[DOC]**
Sarathi's model is a `NonNull` pointer into a llama.cpp C++ heap allocation,
reached through `llama-cpp-2` FFI, in a process with no Python interpreter and no
PyTorch. **[CODE]** There is no object graph for PEFT to operate on. Bridging that
would mean replacing the model engine — explicitly out of scope, and a
significant regression for a local-first desktop application that depends on GGUF
quantisation to run on consumer hardware.

**Yes — as an offline adapter-ingestion tool, in a narrow and genuinely valuable
role.** Sarathi's discovery pipeline finds PEFT safetensors adapters and its
resolver correctly refuses every one of them, which means the LoRA backend is
mostly unreachable in practice today. Converting those adapters to GGUF — using
llama.cpp's own `convert_lora_to_gguf.py`, an existing open-source mechanism —
makes them loadable by the binding path that is **already written, already
tested, and already correct**. PEFT itself is optional even here; the converter
reads `adapter_config.json` and safetensors directly. **[EXT]**

### 16.2 The end-to-end flow, with the recommended addition

Steps marked **[NEW]** are the only additions. Everything else exists today.

```
DOWNLOAD
  HuggingFaceAdapterProvider::discover_adapters
    → HF API search, adapter_config.json fetched and verified
      (base_model_name_or_path alias match, peft_type == LORA,
       weight file ≥ 100 KB via HEAD)
    → download_manager writes adapters/<capability>/adapter_model.safetensors
    → AdapterRegistry::write_manifest  (SSoT merge; never downgrades a
      valid Installed entry)
  [NEW] revision SHA recorded in the manifest → identity = (repo_id, revision)

CONVERT  [NEW — offline, user-triggered, never on the request path]
  validate_adapter → RequiresConversion
    → convert_lora_to_gguf.py → adapters/<capability>/adapter.gguf
    → validate_adapter → Compatible

LOAD
  LoraAdapterCache::get_or_init(model, path)
    cache hit → return handle
    cache miss → llama_adapter_lora_init from disk (ms)

SWITCH
  IntentClassifier::classify(prompt) → intent + confidence
  CapabilityTracker::decide → hysteresis; Hold or Switch
  CapabilityResolver::resolve → Base | PromptProfile | LoraAdapter{path, scale}

VERIFY
  verify_gguf_magic(path)             — before llama.cpp sees the file
  gguf::verify_is_lora_adapter(path)  — GGUF metadata, not just magic
  bind_adapter(...) -> Result         — checked; failure is non-fatal
  [NEW] active_adapter recorded in LoadedModelInfo for UI/observability

GENERATE
  fresh LlamaContext (n_ctx, threads)
  bind_adapter BEFORE prefill, so the prompt runs against adapted weights
  chunked, cancellable prefill → token loop → per-token callback

RESPONSE PATH — DESKTOP
  token callback → app_handle.emit("inference:token", chunk)
    → React listener → chat message state → rendered in the Sarathi window
  capability:changed → badge, e.g. "Code · lora"

RESPONSE PATH — PROVIDER (opencode / claude-code / openclaw / hermes-agent)
  token callback → tokio unbounded channel
    → GenerationHandle.chunks
    → gateway/openai.rs   → SSE  data: {"choices":[{"delta":{...}}]}
      gateway/anthropic.rs → SSE  content_block_delta events
    → HTTP response → the provider's own renderer → user's terminal
  CancelOnDrop releases the model if the client hangs up mid-answer
```

**With the recommended change, the provider path is byte-for-byte unchanged.**
Conversion happens offline; by generation time the adapter is an ordinary GGUF
file that the existing resolver and binder handle exactly as they do today.

### 16.3 What should not be built

- No Python inference sidecar and no `transformers` runtime.
- No `pyo3` bridge to PEFT — PEFT would still need a PyTorch model.
- No reimplementation of PEFT's `set_adapter` semantics in Rust. Sarathi's
  fresh-context-per-turn model already eliminates the state-drift and
  `"irregular"` failure classes that PEFT's verification APIs exist to detect.
- No snapshot/rollback wrapper around adapter binding. There is no persistent
  mutable adapter state to restore; a failed bind costs exactly one turn, which
  runs on the base model, and the next turn retries from a clean context.
- No eviction policy while `llama-cpp-2` cannot free adapter allocations — it
  would report reclaiming memory it did not reclaim.

### 16.4 Closing assessment

The supplied research is technically accurate about PEFT — every API claim
checked against the official v0.20.0 and Transformers v5.14.1 documentation held
up, including the `peft >= 0.19.1` requirement, the `"irregular"` status literal,
and the hotswap constraints. Its one incorrect premise is the assumption in §26
and §31 that Sarathi's base model engine "almost certainly already depends on"
PEFT. It does not. It depends on llama.cpp.

Once that premise is corrected, the research's own recommended architecture
(its §27) turns out to describe, almost step for step, what Sarathi has already
built in Rust: a data-driven adapter registry, compare-before-switch, load-once
with a cached handle, structural verification before binding, serialized model
access, and graceful degradation. On concurrency safety and pre-load validation
Sarathi is meaningfully stronger than raw PEFT, and it has a semantic
adapter-selection layer that PEFT, vLLM, SGLang, and LoRAX all lack.

The productive question is therefore not "how do we add PEFT," but **"why is the
LoRA path Sarathi already built so rarely exercised?"** The answer is in the
code: the adapters discovery finds cannot be loaded (format), and the adapters
users install cannot be routed to (G4). Fix those two things and the existing
engine does the rest.

---

## References

**Official documentation** (fetched and verified during this analysis)

1. [PEFT — PeftModel API reference (v0.20.0)](https://huggingface.co/docs/peft/en/package_reference/peft_model)
2. [PEFT — Hotswapping adapters](https://huggingface.co/docs/peft/main/en/package_reference/hotswap)
3. [Transformers — Parameter-efficient fine-tuning integration](https://huggingface.co/docs/transformers/main/peft)
4. [vLLM — LoRA Adapters](https://docs.vllm.ai/en/latest/features/lora/)
5. [SGLang — LoRA Serving](https://docs.sglang.io/advanced_features/lora.html)

**External research and source**

6. [llama.cpp — `convert_lora_to_gguf.py`](https://github.com/ggml-org/llama.cpp/blob/master/convert_lora_to_gguf.py)
7. [Hugging Face blog — Introducing GGUF-my-LoRA](https://huggingface.co/blog/ngxson/gguf-my-lora)
8. [LoRAX (Predibase)](https://github.com/predibase/lorax)
9. [S-LoRA: Serving Thousands of Concurrent LoRA Adapters (arXiv:2311.03285)](https://arxiv.org/abs/2311.03285)
10. [Punica: Multi-Tenant LoRA Serving (arXiv:2310.18547)](https://arxiv.org/abs/2310.18547)
11. [LORAUTER — Effective LoRA Adapter Routing using Task Representations (arXiv:2601.21795)](https://arxiv.org/abs/2601.21795)
12. [LoRA: Low-Rank Adaptation of Large Language Models (arXiv:2106.09685)](https://arxiv.org/abs/2106.09685)
13. [NVIDIA Dynamo — LoRA Adapters](https://docs.nvidia.com/dynamo/v1.0.1/user-guides/lo-ra-adapters)
14. PEFT issue #804 — `set_adapter` as a global mode setting (concurrency)

**Primary source of truth for all [CODE] claims:** this repository's **working
tree** as of 2026-08-09 (base commit `9749d46`, with uncommitted modifications
present in `ai_engine/runtime.rs`, `ai_engine/traits.rs`, `capability/profile.rs`,
`gateway/*`, and `launcher/*`). Line numbers reflect the working tree, not the
commit. Files under `src-tauri/src/` and `src/` as cited inline.
