# MCP usage and integration — how Sarathi should provide tools to AI clients

**Status:** analysis only. Nothing here has been implemented.
**Date:** 2026-08-08
**Companion:** [sarathi-mcp-tool-host-feasibility.md](sarathi-mcp-tool-host-feasibility.md),
which evaluates the centralized-aggregator proposal. This document answers the narrower,
more useful question: *given the code that exists today, what is the best practical way
to use MCP with Sarathi?*

---

## 1. The one-sentence answer

**Sarathi should be an MCP *provisioner*, not an MCP *proxy*.** It owns the definition,
credentials, readiness and distribution of MCP capabilities; the client owns the
connection. Sarathi sits on the configuration path, not the runtime data path.

This is not a compromise — it is strictly better than centralizing (§11).

---

## 2. A correction to the mental model

The proposed flow is usually drawn like this:

```
Claude Code → Sarathi/MCP → Web Search
```

That reads as a runtime data path, and it is not what should happen — nor what happens
today. The accurate picture separates two timelines:

```
CONFIG TIME  (when Sarathi launches a tool, or on demand)

   mcp.json ──► Sarathi ──► clients/claude-code/mcp.json
                        └─► clients/opencode/opencode.json

RUNTIME      (Sarathi is not in the path at all)

   Claude Code ──spawns stdio──► searxng MCP ──HTTP──► SearxNG :8888
   opencode    ──spawns stdio──► playwright  ──────► its own Chromium
```

Sarathi is the thing that *made the capability exist and be configured correctly*. It is
not a hop. Every property people actually want from "Sarathi/MCP" — install once,
configure once, works in every client — is a config-time property.

---

## 3. Current state — verified against the code

### What exists

| Piece | Location | What it does |
| --- | --- | --- |
| Registry file | `%APPDATA%\com.sarathi.app\mcp.json` | `mcpServers` map; five servers registered |
| Registry loader | `launcher/mcp.rs` — `load()` | Parses, validates (`command` non-empty), drops `disabled`, collects warnings |
| Dialect renderer | `launcher/mcp.rs` — `render(dialect)` | `Standard` (command/args/env) and `Opencode` (`type:"local"`, command+args in one array, `environment`) |
| Document renderer | `launcher/mcp.rs` — `render_document()` | Emits a full `{"mcpServers": …}` file — **currently never called** |
| Placeholder substitution | `launcher/spec.rs` — `PLACEHOLDER_MCP` | `{mcpServers}` substituted as *structure*, never escaped |
| Config write | `launcher/mod.rs` — `launch()` | Writes the rendered config into the tool's private `client_dir` |
| Arg substitution | `launcher/spec.rs` — `resolve_args` | Resolves `--mcp-config {clientDir}/mcp.json` |
| IPC | `commands/launcher.rs` — `user_mcp_file` | Returns the registry path |

### How MCPs work today, end to end

1. User hand-edits `mcp.json`. There is no UI for this.
2. User loads a model in Sarathi (**required** — see the coupling defect below).
3. User presses Launch on a tool card.
4. `launch_tool` (`commands/launcher.rs:219`) loads the registry *fresh on every launch*,
   so an edit reaches the next launch without restarting Sarathi.
5. `launch()` renders the registry in that tool's dialect and writes it into
   `<appdata>/clients/<tool-id>/`.
6. The client is spawned with a private config directory and, for Claude Code,
   `--mcp-config <clientDir>/mcp.json --strict-mcp-config` so Sarathi's set replaces the
   user's own rather than merging with it.
7. **The client spawns its own MCP server processes.** Sarathi never connects to them.

### Verified defects and limits in the current implementation

| # | Finding | Evidence |
| --- | --- | --- |
| D1 | **MCP config generation is gated on a loaded model.** `launch_tool` returns early with "Load a model first" before any MCP work happens. A user who wants Sarathi only to configure MCP for Claude Code on Anthropic's own models cannot get a config written at all. | `commands/launcher.rs:189-191` |
| D2 | **Only 2 of 4 shipped tools receive MCP servers.** `claude-code` and `opencode` carry `{mcpServers}`; `hermes-agent` and `openclaw` have an `mcp_dialect` set but **no placeholder in their templates**, so the field is inert and they get nothing. | `launcher/spec.rs`, per-tool audit |
| D3 | **The registry is stdio-only.** `McpServerSpec` has `command`/`args`/`env`/`disabled`/`description` and no `url` or `type`. A remote/HTTP MCP server cannot be expressed. `render()` hardcodes `"type":"local"` for opencode. | `launcher/mcp.rs:39-86` |
| D4 | **Nothing about MCP is visible in the UI.** Zero MCP references in `src/`. `user_mcp_file` is registered as an IPC command but no frontend code calls it. Registry warnings go to `log::warn!` only — unlike tool-registry warnings, they are not returned in `LaunchOverview`. | `src/` grep; `commands/launcher.rs:220-222` vs `:112` |
| D5 | **Clients Sarathi did not launch get nothing.** Distribution happens only inside `launch()`. `render_document()` — which exists and is tested for exactly this — is dead code. | `launcher/mcp.rs:176`; call-site grep |
| D6 | **No readiness signal.** Sarathi cannot tell whether a configured server actually starts or answers. `McpRegistry` never opens a transport. | `launcher/mcp.rs` |

D1 and D5 together are the important ones: **Sarathi is currently only an MCP configurator
for tools it launches, while a model is loaded.** That is a much narrower capability than
the value on offer.

---

## 4. Proposed usage model — three modes, mostly one

### Mode A — Provisioned (default, and what should stay default)

Sarathi writes client configs; clients spawn and own their servers.

- No runtime coupling: closing Sarathi does not break anyone's tools.
- No new protocol surface, no new auth surface, no SPOF.
- Native client semantics: Claude Code's `mcp__server__tool` names, its permission
  prompts, its `/mcp` UI, its OAuth flows all keep working untouched.
- This is what exists. It needs *extending* (§10), not replacing.

### Mode B — Supervised (recommended addition, small)

Sarathi additionally *verifies* each configured server: spawn it, complete the
`initialize` handshake, read back `tools/list`, record health, tear down. Still writes
configs; still not in the runtime path.

This is the highest value-per-line change available. It converts `mcp.json` from a file
that might be right into a set of capabilities Sarathi can assert are real — and it makes
D6 go away without any of the aggregator's costs.

### Mode C — Proxied (opt-in, narrow, later)

Sarathi exposes *selected* servers over an authenticated `/mcp` endpoint, for the two
cases Mode A genuinely cannot serve:

- clients that only speak remote MCP;
- servers whose credentials should never reach the client.

Everything else stays in Mode A. Mode C is a per-server opt-in, not a migration.

---

## 5. Answers to the specific questions

**1. How `mcp.json` should work.** As the single source of truth, extended with a
transport discriminator (`stdio` | `http`) and per-server operational metadata
(`stateful`, `shareable`, `exposeVia: provisioned|proxied`). Loaded fresh per
distribution so edits take effect without a restart — already true.

**2. How clients discover/use them.** Two paths. *Managed*: Sarathi writes the config into
the client's private directory at launch (works today for Claude Code and opencode).
*Unmanaged*: Sarathi exports a standard `mcpServers` document the user points any client
at — `render_document()` already produces exactly this and is unused (D5).

**3. Expose / spawn / aggregate / hybrid.** **Hybrid, weighted heavily to provisioning.**
Sarathi should *define and distribute* always, *supervise* for health, and *proxy* only
for the narrow cases in Mode C. It should **not** spawn servers for clients to share by
default — that is where the availability regression and the stateful-sharing hazards come
from.

**4. Reuse across clients.** The dialect renderer is the mechanism and it already works.
It needs a third dialect (`Remote`) for Mode C and the two missing placeholders (D2).

**5. Lifecycle.** See §7.

**6. Stateful vs stateless.** See §8.

**7. Auth and permissions.** See §9. In Mode A the answer is "no change" — the client
already owns permissions. Auth only becomes necessary if Mode C ships.

**8. Hermes components.** See §10.

**9. Practical use.** See §12.

**10. Why not centralize.** See §11.

---

## 6. Client → Sarathi → MCP flows

### Flow 1 — Managed client (today's path, extended)

```
User edits mcp.json
        │
        ▼
Sarathi ──validate──► render(dialect) ──write──► clients/claude-code/mcp.json
        │
        └─ launch: claude --mcp-config <dir>/mcp.json --strict-mcp-config
                                │
                                ▼
                        Claude Code spawns searxng, crawl4ai, research, git, playwright
                                │
                                ▼
                        tools/call ──► the server ──► SearxNG :8888
```

Sarathi's involvement ends at step 2.

### Flow 2 — Unmanaged client (needs D5 fixed)

```
Sarathi UI: "Export MCP config" ──► ~/mcp-sarathi.json   (render_document)
        │
        ▼
User: claude mcp add --transport stdio ...   or paste into any client's config
        │
        ▼
Client spawns its own servers. Sarathi never runs again for this to keep working.
```

### Flow 3 — Proxied server (Mode C, future)

```
Claude Code ──HTTP+token──► Sarathi /mcp ──stdio──► credentialed server
```

Used only where the credential must not leave Sarathi, or the client cannot spawn
processes.

---

## 7. Lifecycle model

The right lifecycle differs per mode, and conflating them is the mistake to avoid.

**Mode A — nothing to manage.** The client owns start, restart and shutdown. Sarathi's
only lifecycle duty is that the config it writes is current. Two consequences:

- Distribution must be **decoupled from model launch** (D1). Writing a config is not an
  inference operation and should not require a loaded model.
- Distribution should be **re-runnable on demand** — a "Sync MCP config" action per tool,
  not only a side effect of Launch.

**Mode B — probe, don't supervise.** Health checks are *transient*: spawn, `initialize`,
`tools/list`, terminate. Nothing stays resident, so there is nothing to keep alive and no
process to leak. Cache results with a TTL and a last-good grace window so a single slow
probe does not flip a healthy server to red (§10).

**Mode C — real supervision, and only then.** Retained stdio pipes, backoff restart,
per-call timeouts, ref-counted teardown on client disconnect. This is the expensive part
and it should not be paid until a server actually needs proxying.

**Failure recovery, per mode.** In A a broken server is the client's problem and the
client reports it — which is correct, because the client is who the user is looking at. In
B a failing probe marks the server unhealthy in Sarathi's UI and, ideally, is written into
the exported config as `disabled` with the reason, so a known-broken server does not get
handed out. In C a crash drops that server's tools and emits `tools/list_changed`.

---

## 8. Stateful vs stateless MCP strategy

The five registered servers do not have the same sharing properties:

| Server | State | Strategy |
| --- | --- | --- |
| `searxng` | none | Stateless. Safe to proxy if ever wanted. |
| `git` | none per call (`repo_path` is an argument) | Stateless *to MCP*; concurrent writes to one repo are the caller's problem either way. |
| `research` | shared SQLite index — **intended** shared state | Stateless-ish. Sharing is a feature here: two clients ingesting into one notebook is the point. Concurrent writes need SQLite WAL discipline. |
| `crawl4ai` | session-scoped features (`manage_session`) | Mixed. Stateless for `get_markdown`; stateful if sessions are used. |
| `playwright` | **browser context, cookies, navigation** | **Stateful. Must never be shared.** |

The rule this implies:

> **Stateless servers may be proxied. Stateful servers must be provisioned per client.**

Sharing one Playwright browser between two agents is not a performance trade-off, it is a
correctness bug — navigation, dialogs and cookies interleave with no error. Mode A gives
each client its own instance for free, because each client spawns its own. This is a case
where the "wasteful" architecture is the correct one, and it is a strong argument against
blanket centralization.

`mcp.json` should carry this explicitly (`stateful: true`) so the choice is data, not
folklore.

---

## 9. Security model

**Mode A: the current posture is already right, and centralizing would make it worse.**
Each client spawns servers as itself, under its own permission prompts. Sarathi adds no
attack surface because it is not listening for tool calls.

The one real weakness today is **credential distribution**: `mcp.json` holds the Crawl4AI
token in plaintext and that token is copied into every client config Sarathi writes. It is
a local-service token in a user-only directory, so the blast radius is small, but the
pattern does not scale to a server with a real API key.

Mitigations that fit Mode A:

- Keep secrets in a separate, gitignored `secrets.json`; store `${VAR}` references in
  `mcp.json` and resolve at render time. Claude Code already expands `${VAR}`; for
  dialects that do not, Sarathi resolves before writing.
- Never write a resolved secret into a config for a client that does not need that server.
- Mark secret-bearing servers as `exposeVia: proxied` when Mode C exists — that is
  precisely the case proxying is *for*.

**Mode C, if built, requires all of this before exposure**: per-client bearer tokens
issued through the existing config-generation path, a default-deny grant set for mutating
tools, and an audit log. Note the existing gateway is deliberately unauthenticated
(`gateway/guard.rs` — Origin/Host only, modelled on Ollama). That trade is defensible for
inference and not for tool execution; `/mcp` would need its own auth layer that `/v1/*`
does not have.

---

## 10. Hermes integration boundary

Provisioning needs far less of Hermes than either the agent or the aggregator does. The
useful subset:

| Hermes component | Fits here? | Use |
| --- | --- | --- |
| `check_fn` TTL cache + last-good grace (`tools/registry.py:145-206`) | **Yes** | Exactly Mode B's health caching. Its documented rationale — one slow probe must not strip a whole toolset — is the same failure this would otherwise have. |
| `sanitize_tool_schemas` **core subset** (bare `{"type":"object"}`, string-valued schema nodes, `$ref` siblings, array `type`) | **Yes, at probe time** | These are *malformed* schemas. Detecting them during a health probe lets Sarathi warn "this server emits schemas llama.cpp will reject" — a diagnostic, not a mutation. |
| `_normalize_name_filter` include/exclude (`tools/mcp_tool.py:4299`) | **Yes** | Per-server tool filtering in `mcp.json`, so a client can be given three tools from a server instead of forty. |
| `mcp_prefixed_tool_name`, `mcp-{server}` toolsets | **Only in Mode C** | In Mode A the client already namespaces. Applying it too would double-prefix. |
| `strip_pattern_and_format`, `strip_slash_enum` | **No** | Provider-targeted, and a provisioner does not know the client's model. Diagnose, do not rewrite. |
| `tool_guardrails`, `_should_parallelize_tool_batch`, `ProviderProfile`, `tool_executor`, `conversation_loop` | **No** | Agent-layer. No turn boundary and no provider exist here. |

**Boundary rule:** Sarathi may use Hermes' knowledge to *validate and describe* servers.
It should not use Hermes' policy to *rewrite* what a client sends, because in Mode A
Sarathi never sees a tool call.

---

## 11. Why this beats centralizing everything

| Property | Provisioned (recommended) | Fully centralized aggregator |
| --- | --- | --- |
| Tools available when Sarathi is closed | **Yes** | No — gateway dies with the window (`lib.rs:124-144`; no tray, no `CloseRequested` handler) |
| Single point of failure | None | One process removes all tools from all clients |
| New auth surface | None | Required, and non-trivial (§9) |
| Playwright correctness | Per-client instance, free | Must special-case, or silently corrupt |
| Client permission rules | Unchanged | Broken by re-namespacing (`mcp__sarathi__searxng_searxng_web_search`) |
| Client OAuth to third-party MCPs | Works natively | Must be proxied |
| Protocol code to maintain | Zero | Sessions, SSE, resumability, notification fan-out, sampling forwarding |
| Extra latency per call | None | One hop + re-encode |
| Delivers "configure once" | **Yes — already** | Yes |
| Credential containment | Partial (§9) | **Better** |
| Remote-only clients | No | **Yes** |

The last two rows are the entire honest case for centralization — and they are per-server
concerns, which is why Mode C is a per-server opt-in rather than an architecture.

Everything else favours provisioning, and the first row is close to decisive: an approach
whose tools stop working when a desktop app is closed is worse than the file it replaced.

---

## 12. Real-world workflows

Rendering the requested examples as what would actually happen:

**"Claude Code → Sarathi/MCP → Web Search."** In Sarathi the user adds a server to
`mcp.json` (or picks it from a catalog, §13). Sarathi validates it, probes it once, shows
green, writes it into `clients/claude-code/mcp.json`, launches Claude Code with
`--mcp-config … --strict-mcp-config`. Claude Code spawns `mcp-searxng`, which queries the
local SearxNG container. Sarathi is idle throughout.

**"OpenCode → Sarathi/MCP → Browser."** Same registry, rendered in opencode's dialect
(`type:"local"`, command+args merged, `environment`). opencode spawns **its own**
Playwright — separate browser, separate cookies. If Claude Code is also running, the two
never collide. Under a centralized host they would.

**"Claude Code → Sarathi/MCP → Git."** `git_log`/`git_diff` against a local checkout, no
GitHub token anywhere. The `repo_path` argument makes it stateless, so this one *could*
be proxied later — it just has no reason to be.

**"OpenCode → Sarathi/MCP → Research."** The one case where shared state is wanted: both
clients ingest into the same SQLite index, so a page Claude Code scraped is searchable
from opencode. That works in Mode A because the shared state lives in the *database*, not
in the server process.

**A client Sarathi never launched (needs D5).** User clicks "Export MCP config", gets a
standard `mcpServers` document, pastes it into Cursor or Windsurf. Works forever, with
Sarathi closed.

---

## 13. Required future changes

Described only. Ordered by value per unit of risk.

| # | Change | Fixes | Risk |
| --- | --- | --- | --- |
| C1 | **Decouple MCP distribution from model launch.** New command that writes a tool's MCP config without requiring a loaded model; a "Sync MCP config" action per tool card. | D1 | Low |
| C2 | **Export a standalone registry document.** Wire the existing, unused `render_document()` to an IPC command + UI button. | D5 | Very low — the function and its test already exist |
| C3 | **Add `{mcpServers}` to the hermes-agent and openclaw templates**, or remove the inert `mcp_dialect` field from them so the code stops implying support it does not have. Requires confirming each client's config key first. | D2 | Low |
| C4 | **Surface MCP in the UI.** Server list, per-server enable/disable, registry path, and registry warnings promoted from `log::warn!` into `LaunchOverview.warnings` alongside the tool-registry ones. | D4 | Low |
| C5 | **Transport discriminator on `McpServerSpec`** (`stdio` \| `http` + `url`/`headers`), with dialect renderers extended accordingly. | D3 | Medium — touches the render tests |
| C6 | **Health probe (Mode B).** Transient spawn → `initialize` → `tools/list` → terminate, with TTL + last-good grace. Show tool counts and schema warnings per server. | D6 | Medium — first real MCP client code in the crate; use `rmcp` |
| C7 | **Secret indirection.** `${VAR}` references in `mcp.json` resolved at render time from a gitignored store. | §9 | Medium |
| C8 | **Operational metadata** — `stateful`, `shareable`, `exposeVia`, per-server tool include/exclude. | §8 | Low |
| C9 | *(Only if Mode C is justified)* authenticated `/mcp` endpoint, `Remote` dialect, per-client tokens, grants, audit. | — | High — see the companion document |

C1 + C2 together are the highest-leverage work in this document: they turn Sarathi from
"an MCP configurator for tools it launches while a model is loaded" into "the place MCP
capabilities are defined and handed to anything."

---

## 14. Final recommendation

**Adopt a provisioning-first hybrid.**

1. **Default: Mode A (provisioned).** Sarathi defines, validates, credentials and
   distributes; clients spawn and own. Keep it the default permanently, not as a
   stepping stone.
2. **Next: C1, C2, C4** — decouple distribution from model launch, export a standalone
   document, make MCP visible. Small, low-risk, and they remove the two limits that make
   the current implementation feel narrower than it is.
3. **Then: C6 (Mode B)** — health probing. This is what people actually mean when they
   ask Sarathi to "manage" MCPs: not to run them, but to *vouch* for them.
4. **Only then, and only per server: Mode C.** Proxy a server when it holds a credential
   that must not reach clients, or when a client cannot spawn processes. Never proxy a
   stateful server.
5. **Hermes stays a validation library here**, not a policy layer (§10).

The reason this is better than centralizing is not that centralizing is hard — it is that
centralizing removes properties the current design gets for free: availability when
Sarathi is closed, per-client Playwright isolation, native client permissions, and zero
new attack surface. The genuine benefits of a hub — credential containment and remote
client reach — are per-server concerns and are better served by a per-server opt-in than
by routing everything through one process.

Sarathi's real advantage was never being in the middle of every tool call. It is being
the only place on the machine that knows what a working MCP setup looks like, and can
hand that to anything that asks.
