# Sarathi

Tauri 2 desktop app for running local LLMs. React 19 + TypeScript + Vite frontend
(`src/`), Rust backend (`src-tauri/`), Python sidecars (`sidecars/`).

Build variants: `npm run dev:cuda` (CUDA), `npm run dev:vulkan` (Vulkan), or
`npm start` / `npm run dev:auto` to let `scripts/select-backend.mjs` pick.
Prefer the selector — a CPU-only binary cannot use a GPU later, because backend
selection in llama.cpp is compile-time.

## Agent capability layer (MCP)

MCP servers are declared once in `%APPDATA%\com.sarathi.app\mcp.json` and handed
to every launched tool by `src-tauri/src/launcher/mcp.rs`. Research/search/crawl
sidecar lives in `sidecars/mcp/sarathi_research/`. Services:
`.\scripts\mcp-services.ps1 status`. Details in
[docs/agent-capabilities.md](docs/agent-capabilities.md).

Python here targets the **system interpreter**. Do not create a venv,
virtualenv, conda env, or any project-local Python environment — global `pip`,
or `uv tool install` for a package that needs isolation.

## Capability (LoRA) routing with Laya

Each chat turn's capability slot — and so its LoRA adapter — is chosen by the
Laya sidecar in `sidecars/laya_router/` (stdio JSON-RPC, no port), driven by
`src-tauri/src/capability/laya.rs`, falling back to the keyword classifier
when Laya is off (`ai_settings.laya_routing`), not installed, or slower than
1.5 s. Install with `.\scripts\laya-setup.ps1`: it adds only `laya==0.3.20`
(`--no-deps`) to the system interpreter and puts weights in
`%APPDATA%\com.sarathi.app\laya`. Laya does not choose models for hardware;
that stays in the deterministic `model_recommendation/` engine.

## Coexisting with ARJUN

ARJUN (`C:\Users\lenovo\Desktop\Arjun-1`) is a separate app grown from this
codebase. Never edit it. Sarathi must not reuse its ports — ARJUN owns gateway
11435 and Vite 1420/1421; Sarathi uses gateway **11535** and Vite **1430/1431**
— nor its `ARJUN_*` variables or `com.arjun.workbench` data. ARJUN *reads*
Sarathi's `%APPDATA%\com.sarathi.app` (`mcp-servers/`, `models/`), so do not
move or delete those. Both apps share `laya==0.3.20` in the system
interpreter; do not change that pin on one side only.

## Skill routing

- Changes to the Rust launcher/model logic (`src-tauri/src/launcher/`) → `/review` before merge
- UI or design changes (`src/`) → `/design-review`
- Verifying the app actually runs → `/qa`
- Model download / hardware sizing bugs → `/investigate`
- MCP server or capability-layer changes → `/review`, then re-run `.\scripts\mcp-services.ps1 status`

## gstack (REQUIRED — global install)

**Before doing ANY work, verify gstack is installed:**

```bash
test -d ~/.claude/skills/gstack/bin && echo "GSTACK_OK" || echo "GSTACK_MISSING"
```

If GSTACK_MISSING: STOP. Do not proceed. Tell the user:

> gstack is required for all AI-assisted work in this repo.
> Install it:
> ```bash
> git clone --depth 1 https://github.com/garrytan/gstack.git ~/.claude/skills/gstack
> cd ~/.claude/skills/gstack && ./setup --team
> ```
> Then restart your AI coding tool.

Do not skip skills, ignore gstack errors, or work around missing gstack.

Using gstack skills: After install, skills like /qa, /ship, /review, /investigate,
and /browse are available. Use /browse for all web browsing.
Use ~/.claude/skills/gstack/... for gstack file paths (the global path).
