# Sarathi as a centralized MCP Tool Host — feasibility analysis

**Status:** analysis only. Nothing in this document has been implemented.
**Date:** 2026-08-08
**Scope:** the proposal to make Sarathi a provider/model-agnostic MCP aggregator that
Claude Code, opencode and other MCP clients connect to for a shared tool set.

---

## 0. Terminology — the proposal's name is wrong, and the confusion is load-bearing

In MCP's own vocabulary, a **Host** is the LLM application that *contains* clients —
Claude Desktop, Claude Code, opencode. A Host talks to a model and decides when tools
run. What the proposal describes is not that. It is an **aggregating MCP server**: a
process that is an MCP *server* to its callers and an MCP *client* to a set of
downstream servers. It never touches a model on the tool path.

The distinction is not pedantry. It separates two products that share almost no code:

| | **Aggregating MCP server** (this proposal) | **Sarathi-as-agent** |
| --- | --- | --- |
| Talks to a model for tools? | No | Yes |
| Owns the tool-calling loop? | No — the client does | Yes |
| Model/provider agnostic? | Yes, *by construction* — it never sees a model | Only via an abstraction layer |
| Needs Hermes' loop policy? | Mostly no (see §8) | Yes |
| Sarathi's current code supports it? | Partly (§2) | Not at all |

The rest of this document uses **MCP Aggregator** for the proposal and reserves
"host" for the spec's meaning. Where the two are conflated, the failure mode is
importing agent-layer policy (loop guardrails, parallel-batch planning, provider
schema targeting) into a component that has no turn boundary and no provider — see §8.

---

## 1. Can Sarathi technically act as this aggregator?

**Yes.** Nothing in the architecture forbids it, and the transport half is largely
already present. The blockers are operational, not technical (§10, §11).

Verified enablers:

- `src-tauri/Cargo.toml` — `axum = "0.8"`, `tokio` with `rt-multi-thread`/`net`/`io-util`,
  `tokio-stream`, `futures-util`, `serde_json`, `uuid` (v4). An HTTP server with SSE and
  async child-process I/O is already the shape of the crate.
- `gateway/server.rs` — a working axum `Router`, graceful shutdown via
  `GatewayHandle`, loopback-only bind with retry and port fallback, and **SSE already in
  production use** (`axum::response::sse::{Event, KeepAlive, Sse}`) for both protocol
  surfaces.
- `gateway/guard.rs` — `Origin`/`Host` validation middleware, which the MCP spec
  *mandates* for Streamable HTTP ("Servers **MUST** validate the `Origin` header").
- `rmcp`, the official Rust MCP SDK, provides server, client, `transport-streamable-http-server`
  and child-process transports. Sarathi would not implement the protocol by hand.

Verified gap: **there is no MCP crate in `Cargo.toml` today**, and no MCP protocol code
anywhere in `src-tauri/src/`.

---

## 2. What existing Sarathi components already support this

| Component | File | What it gives the aggregator | Reusable as-is? |
| --- | --- | --- | --- |
| axum router + graceful shutdown | `gateway/server.rs:141` | Somewhere to mount `/mcp` | Yes |
| SSE plumbing | `gateway/server.rs` | Streamable HTTP's server→client channel | Pattern, yes |
| Origin/Host guard | `gateway/guard.rs` | Spec-mandated DNS-rebinding defence | Yes |
| Loopback-only bind + port fallback | `gateway/server.rs:80-120` | Spec's "SHOULD bind localhost" | Yes |
| `McpRegistry` | `launcher/mcp.rs` | Server inventory, validation, disable flag | **Partly — see below** |
| Client config generation | `launcher/spec.rs`, `launcher/mod.rs` | Writes per-client MCP configs at launch | Yes, and it is the incumbent solution |
| Python sidecar IPC precedent | `sidecars/memory_engine_sidecar/main.py` | Newline-delimited JSON-RPC 2.0 over stdio, spawned from Rust | Pattern, yes |
| `GenerationScheduler` | `ai_engine/scheduler.rs` | A local model reachable from inside the process | Yes — enables §4's sampling win |

**Important correction about `McpRegistry`.** It is a *config renderer*, not a client.
`load()` parses `mcp.json`; `render(dialect)` emits per-client JSON; `render_document()`
emits a standalone file. It never spawns a process, never opens a transport, never
speaks JSON-RPC. Every consumer (`commands/launcher.rs:219`, `launcher/spec.rs`) uses it
for config generation only. The data model (`McpServerSpec { command, args, env,
disabled, description }`) is a good foundation for a supervisor, but zero connection
logic exists.

**What does not exist at all:**

- No MCP client or server implementation.
- No process supervision. `LaunchedProcesses` (`launcher/mod.rs:60-84`) is a
  `Mutex<HashMap<String, u32>>` of PIDs. `launch()` spawns detached and the module
  comments explicitly state Sarathi "does not kill the process" (`launcher/mod.rs:274`).
  There are no retained stdio pipes, no health checks, no restart.
- No authentication anywhere on the gateway.
- No background/tray mode — see §10.

---

## 3. What must be added or changed

Ordered by dependency.

1. **`rmcp` dependency** (server + client + streamable-http + child-process features).
2. **Downstream supervisor** — a real process manager for stdio MCP servers: spawn with
   retained stdin/stdout, initialize handshake, capability capture, health tracking,
   backoff restart, graceful shutdown. `LaunchedProcesses` does not do this and was
   never meant to.
3. **Session registry** — Streamable HTTP is stateful. Per-client `Mcp-Session-Id`,
   negotiated protocol version, declared client capabilities (`sampling`, `elicitation`,
   `roots`), and a per-session SSE stream for the GET channel.
4. **Aggregation/routing table** — namespaced tool index mapping exposed name →
   (downstream server, original name), rebuilt on `notifications/tools/list_changed`.
5. **Auth + permission model** — see §10. This is a prerequisite, not a nice-to-have.
6. **Lifecycle mode** — tray/background or a separate service process (§10, §12).
7. **Schema normalization pass** — the safe subset only (§8).
8. **UI** — server health, per-client grants, live call log. Sarathi is a desktop app;
   an aggregator with no visible state is a debugging trap.

---

## 4. Aggregation, discovery, namespacing, exposure

### Namespacing

Downstream tool names collide (`search`, `fetch`, `read`). Hermes' answer is directly
applicable: `mcp_prefixed_tool_name(server, tool)` plus per-server toolsets named
`mcp-{server}` and include/exclude filters (`tools/mcp_tool.py:4176-4460`).

**A flaw the proposal must confront: double prefixing.** Claude Code already exposes MCP
tools to the model as `mcp__<server>__<tool>`. If Sarathi aggregates five servers behind
one entry named `sarathi`, the model sees `mcp__sarathi__searxng_searxng_web_search`.
That is long, ugly, and — more importantly — **breaks every client-side allowlist,
permission rule and hook that matches on the current names**. Any migration silently
invalidates existing `settings.json` permission entries.

Mitigations, in order of preference:

- Expose **one aggregator entry per downstream server** (`sarathi-searxng`,
  `sarathi-crawl4ai`, …), each a distinct MCP endpoint path or a distinct session
  scope. Names stay one level deep; permissions map cleanly. Costs: N client entries
  again, which erodes the "one entry" selling point.
- Or accept a single entry and publish a rename map, documenting the permission break.

### Discovery

`tools/list` on the aggregator returns the union, filtered by that session's grants.
Downstream `notifications/tools/list_changed` must be fanned out to every connected
session over its GET SSE stream — which is exactly why §3.3 (session registry) is a hard
prerequisite and cannot be deferred.

### Health

Hermes' `check_fn` TTL cache transfers well: 30 s TTL with a 60 s last-good grace window
so a flapping probe does not strip an entire toolset mid-turn
(`tools/registry.py:145-206`). The rationale — "a single `docker version` that times out
under load returns False for one call, which would silently strip the entire terminal
toolset" — applies identically to a downstream MCP server that briefly stops answering.

### Sampling — the one genuinely novel capability

If a downstream server issues `sampling/createMessage`, a normal proxy must forward it
upstream and correlate across two sessions. **Sarathi has a loaded model and a scheduler
already wired for exactly this shape of request.** It can terminate sampling locally via
`GenerationScheduler::submit`, with the desktop UI providing the human-in-the-loop
approval the spec calls for. No other MCP aggregator can do this. This is the strongest
architectural argument for the proposal and it should be stated as such — it is a
capability, not just a consolidation.

`elicitation` similarly maps to a Tauri dialog. `roots` are client-scoped workspace
directories and **must** be forwarded per-session, which again requires §3.3.

---

## 5. How clients connect

Both target clients support remote MCP — verified against the installed binaries:

- **Claude Code**: `claude mcp add --transport http <name> <url>` with `--header` for
  the bearer token, plus `--client-id`/`--client-secret` for OAuth. stdio also supported.
- **opencode**: schema defines `McpLocalConfig` (`type: "local"`), `McpRemoteConfig`
  (`type: "remote"`) and `McpOAuthConfig`.

So the connection story is:

```
claude mcp add --transport http sarathi http://127.0.0.1:11435/mcp \
  --header "Authorization: Bearer <per-client token>"
```

and Sarathi's existing `launcher/mcp.rs` dialect renderer would emit the equivalent
`remote` entry for opencode automatically at launch — the generation machinery already
exists and only needs a third dialect.

**The uncomfortable comparison.** This is not a new capability. `mcp.json` +
`launcher/mcp.rs` already gives every launched client the same five servers, and that
path was verified working end-to-end (opencode reported all five connected). The
aggregator's incremental value is therefore **not** "configure once" — that is solved.
It is: centralized policy, credential isolation, sampling termination, and reach for
clients that only speak remote MCP.

---

## 6. Independence from model and serving provider

**Yes, and more cleanly than the agent architecture.** An aggregator never constructs a
model request, so there is nothing to be provider-specific about. `tools/list` and
`tools/call` are identical whether the caller's model is behind Sarathi's gateway,
Ollama, vLLM, SGLang, NIM or a cloud API.

**But this cuts both ways, and it invalidates part of the Hermes plan.** Because the
aggregator does not know which model or backend the client is using, it *cannot* apply
provider-targeted schema fixes. `strip_pattern_and_format` exists because llama.cpp's
grammar converter rejects `\d`; `strip_slash_enum` exists because xAI rejects `/` in
enums. Applying either unconditionally degrades prompting quality for every client that
did not need it; applying neither leaves llama.cpp clients broken. See §8 for the split.

---

## 7. Interaction with Sarathi's existing gateways

Three points, all verified:

1. **No contention with the model path.** `ai_engine/scheduler.rs` serializes *all model
   access behind a single worker thread*. Tool execution never touches it, so aggregator
   load is orthogonal to inference load. The exception is sampling termination (§4),
   which does enter the queue and can therefore be starved behind a long generation —
   sampling requests need a bounded timeout and a documented queue position.

2. **Port sharing is fine; auth sharing is not.** Mounting `/mcp` on the existing router
   is clean in axum. But `gateway/guard.rs` is deliberately unauthenticated, modelled on
   Ollama ("paste a URL into your tool and it works, no token to configure"). That trade
   is defensible for inference and **indefensible for tool execution** (§10). The `/mcp`
   route needs an additional auth layer that the `/v1/*` routes do not have.

3. **The two surfaces stay independent.** A client may use Sarathi for tools and a
   different gateway for its model, or vice versa. Nothing should couple them. In
   particular, do not infer a client's model from its gateway traffic to target schema
   fixes — there is no reliable correlation between an HTTP client on `/v1/messages` and
   a session on `/mcp`.

---

## 8. Hermes components: reuse, adapt, avoid

Hermes Agent is MIT (© 2025 Nous Research), so reuse is permitted with notice retained.
The important finding is that **most of Hermes' tool-calling value belongs to the agent
layer, not the host layer.** Sorting by which side of the boundary each piece lives on:

### Reuse — normalization and inventory (host-appropriate)

| Component | Why it transfers |
| --- | --- |
| `sanitize_tool_schemas` **core subset** — bare `{"type":"object"}` with no properties, string-valued schema nodes from malformed MCP output, `$ref` siblings, array-form `type` | These are *broken* schemas, not provider preferences. Fixing them is correct for every caller. |
| `mcp_prefixed_tool_name`, `mcp-{server}` toolsets, `_normalize_name_filter` include/exclude | Exactly the namespacing problem in §4. |
| `check_fn` TTL cache + last-good grace | Downstream health without flap-induced toolset stripping. |
| `ToolEntry` / `ToolRegistry` shape | Good data model for the routing table (design, not code — Sarathi's side is Rust). |

### Adapt — opt-in, per-session

| Component | Adaptation required |
| --- | --- |
| `strip_pattern_and_format`, `strip_slash_enum`, `strip_nullable_unions` | Provider-targeted. Must be opt-in per session (header or endpoint path, e.g. `/mcp?profile=llamacpp`), never global — see §6. |
| `tool_search` progressive disclosure | Depends on the client's context window, which the aggregator does not know. Opt-in only. |
| `coerce_tool_args` | Defensible as a last line of defence before a downstream call, but it **mutates caller intent**. Off by default. |

### Avoid — agent-layer, wrong side of the boundary

| Component | Why it does not belong here |
| --- | --- |
| `tool_guardrails` | Loop detection is per-turn state. An aggregator sees interleaved calls from N clients with no turn boundary; "repeated non-progressing call" is undefined. |
| `_should_parallelize_tool_batch` | The **client** decides parallelism and issues concurrent requests. The aggregator just serves them. |
| `ProviderProfile` | There are no providers in an aggregator. |
| `tool_executor.py`, `conversation_loop.py` | Agent-only, and unextractable regardless (50 `agent.*` attributes; one 3,900-line function). |

**Net:** of the ~3,200 reusable Hermes LOC identified previously, roughly **1,300**
(schema-sanitizer core, naming, health caching, registry shape) applies to the
aggregator. The remainder is agent-layer and should be deferred with the agent.

---

## 9. Required transport/protocol architecture

Target: **Streamable HTTP** (spec 2025-06-18 or later), single endpoint at `/mcp`
supporting POST and GET.

Concrete obligations, from the spec:

- One endpoint path, both `POST` and `GET`.
- POST of a request → either `application/json` (single response) or
  `text/event-stream` (SSE stream ending in the response).
- POST of a notification/response → `202 Accepted`, no body.
- GET → SSE stream for unsolicited server→client messages, or `405`.
- `Mcp-Session-Id` issued on `InitializeResult`, echoed by the client thereafter;
  `400` if missing on non-init requests; `404` after termination; `DELETE` to end.
- `MCP-Protocol-Version` header on all post-init requests; `400` on unsupported.
- SSE event `id`s + `Last-Event-ID` replay for resumability (MAY, but a desktop app
  that sleeps will need it).
- **MUST** validate `Origin`; **SHOULD** bind loopback; **SHOULD** authenticate.

Sarathi satisfies the Origin and loopback obligations today via `gateway/guard.rs`.
Everything else is new. Downstream connections use stdio (`rmcp` child-process
transport) for the servers currently in `mcp.json`, with streamable-HTTP support for
remote ones.

Supporting stdio *inbound* as well (a thin `sarathi-mcp-stdio` shim that clients spawn
and which proxies to the HTTP endpoint) is worth considering for maximum client
compatibility, and is cheap.

---

## 10. Security, lifecycle, concurrency, failure recovery

### Security — the largest change in threat model

Sarathi's gateway is intentionally unauthenticated. `gateway/guard.rs` documents the
reasoning: browsers cannot forge `Origin`, CLI tools send none, so an Origin/Host check
blocks drive-by pages "while costing legitimate clients nothing — no token, no
configuration."

That reasoning holds for inference. It **fails for tool execution**. The Origin guard
stops *browsers*; it does nothing about *local processes*. Today the worst a local
process can do through the gateway is spend tokens. Through an aggregator it can read
and write files, drive a headless browser, clone repositories, and reach whatever else
`mcp.json` lists — with Sarathi's privileges and Sarathi's stored credentials
(`mcp.json` currently holds the Crawl4AI token in plaintext). A malicious npm
post-install script becomes a filesystem-access primitive.

Required before exposure:

- Per-client bearer tokens, generated by Sarathi, delivered through the existing
  client-config generation path (which already writes per-tool config directories).
- A permission model: which sessions may see which servers/tools. Default-deny for
  anything mutating.
- Credential containment: downstream env vars stay in the aggregator process; clients
  never receive them. **This is a genuine security improvement over the status quo**,
  where every client config carries the token.
- Audit log of `tools/call` per session, surfaced in the UI.

### Process lifecycle — the largest practical blocker

Verified: the gateway is started inside Tauri's `setup` (`lib.rs:124-144`) and its
handle is stored with `app.manage()`, so it lives exactly as long as the desktop app.
There is **no tray-icon implementation** (the cargo feature is enabled but unused) and
**no `CloseRequested`/`prevent_close` handling**. Closing the window ends the process.

Consequence: **making Sarathi the tool host means closing Sarathi breaks every tool in
every client.** Today, with `mcp.json` distribution, clients spawn their own servers and
are unaffected by Sarathi's state. The proposal is therefore a strict availability
*regression* unless a background mode ships with it.

Options: implement tray + close-to-tray; or extract a separate supervised
`sarathi-mcpd` process with its own lifetime. The second is architecturally cleaner and
decouples tool availability from the GUI, at the cost of a second binary and IPC for the
UI to observe it.

### Concurrency — and a correctness trap

Good news: tool traffic bypasses the single-worker model scheduler entirely (§7).

Bad news: **stateful downstream servers cannot be safely shared between clients.**
Of the five servers currently registered:

| Server | Shareable across clients? |
| --- | --- |
| `searxng` | Yes — stateless request/response |
| `git` | Yes per call, but `repo_path` is caller-supplied; concurrent writes to one repo are the caller's problem |
| `research` | Mostly — SQLite index is shared state, but that is intended |
| `crawl4ai` | Session-scoped features (`manage_session`) would cross-contaminate |
| `playwright` | **No.** One browser context shared by two agents is a correctness bug: navigation, cookies and dialogs interleave silently |

So the aggregator needs a per-server **sharing policy** — `shared` vs `per-session` —
and must spawn per-session instances for the latter. That erases much of the
"fewer processes" benefit precisely for the heaviest servers.

Additionally, a shared stdio server is a single duplex pipe: Sarathi must multiplex
JSON-RPC ids across sessions and accept head-of-line blocking on slow calls.

### Failure recovery

- Downstream crash → mark unhealthy, drop its tools, emit `tools/list_changed`, restart
  with backoff. Never fail the whole `tools/list`.
- Downstream hang → per-call timeout returning an MCP error, not a stalled session.
- Aggregator restart → clients see `404` on their old `Mcp-Session-Id` and must
  re-initialize, which the spec already prescribes.
- Client disconnect → reference-count shared servers; tear down per-session ones.

---

## 11. Architectural conflicts and limitations

1. **Availability regression** (§10). The decisive one. Tool access becomes contingent
   on a GUI app being open.
2. **Single point of failure.** Five independent server processes become one process
   whose failure removes all tools from all clients.
3. **Security model inversion** (§10). The gateway's "no token" design is a deliberate
   trade that does not survive contact with tool execution.
4. **Double namespacing / permission breakage** (§4).
5. **Provider-blind normalization** (§6). The most valuable Hermes sanitizers cannot be
   applied automatically.
6. **Stateful-server sharing** (§10).
7. **Benefit overlap with the incumbent.** "Configure once, use everywhere" already
   works. The proposal must be justified on policy, credentials and sampling — not on
   the headline claim.
8. **Latency.** One extra hop and one extra JSON-RPC decode/encode per call. Negligible
   against network tools, measurable against fast local ones.

None of these is fatal. Items 1, 3 and 6 are prerequisites; 4 and 5 are design choices
that must be made explicitly rather than discovered later.

---

## 12. Target architecture

```
  Claude Code            opencode              other MCP client
       │                     │                        │
       │  Streamable HTTP + Bearer token (per client) │
       └──────────────┬──────┴────────────────────────┘
                      ▼
        ┌─────────────────────────────────────────────┐
        │ Sarathi  ::  axum router  (127.0.0.1:11435)  │
        │                                             │
        │  /v1/chat/completions  /v1/messages         │  ← unauthenticated (unchanged)
        │  ────────────────────────────────────────   │
        │  /mcp   POST · GET · DELETE                 │  ← token-authenticated (new)
        │            │                                │
        │     Session Registry                        │
        │     Mcp-Session-Id, protocol version,       │
        │     client capabilities, roots, grants      │
        │            │                                │
        │     Aggregator / Router                     │
        │     namespaced tool table                   │
        │     schema normalization (safe subset)      │
        │     permission check · audit                │
        │            │                                │
        │     Supervisor  (rmcp clients)              │
        │     spawn · handshake · health · restart    │
        │            │                                │
        │            │        sampling/createMessage  │
        │            │        ┌──────────────────────┐│
        │            │        ▼                      ││
        │            │   GenerationScheduler ────────┘│
        │            │   (local model answers, with   │
        │            │    UI approval)                │
        └────────────┼────────────────────────────────┘
                     │
     ┌───────────┬───┴────────┬────────────┬───────────┐
     ▼           ▼            ▼            ▼           ▼
  searxng    crawl4ai     research       git      playwright
  [shared]   [shared]     [shared]    [shared]  [per-session]
```

### End-to-end flow

**Connect.** Client POSTs `initialize` to `/mcp` with its bearer token. Guard validates
`Origin`/`Host`; auth layer validates the token and resolves it to a grant set. Session
registry mints an `Mcp-Session-Id`, records the negotiated protocol version and the
client's declared `sampling`/`elicitation`/`roots` capabilities, and returns
`InitializeResult`. Client POSTs `notifications/initialized` → `202`. Client optionally
opens the GET SSE stream for server-initiated messages.

**Discover.** `tools/list` → aggregator returns the union of healthy downstream tools,
namespaced `<server>_<tool>`, filtered by the session's grants, with the safe-subset
schema normalization applied.

**Call.** `tools/call` with `<server>_<tool>` → routing table resolves the target →
permission check → optional per-session profile fixes → forward to the downstream
supervisor (shared instance, or this session's private one) → downstream result →
truncate per policy → return. Errors come back as MCP tool errors, never as transport
failures.

**Change.** Downstream emits `tools/list_changed` → aggregator rebuilds the table and
fans the notification out to every session's GET stream.

**Sample.** Downstream emits `sampling/createMessage` → aggregator surfaces an approval
prompt in the Sarathi UI → on approval, submits to `GenerationScheduler` → returns the
completion downstream. The client is never involved.

**Disconnect.** `DELETE /mcp` with the session id → per-session servers torn down,
shared ones ref-count decremented.

---

## 13. Feasibility verdict

**Technically feasible. Architecturally sound. Not yet justified, and not yet safe to
build as specified.**

Splitting the verdict:

- **Feasible now:** the transport, the aggregation, the namespacing, the routing.
  `rmcp` plus the existing axum/SSE/guard infrastructure covers most of it.
- **Feasible but requires prerequisites:** exposure to real clients, which is gated on
  authentication (§10) and background lifecycle (§10).
- **Should be scoped down:** the Hermes import. Roughly 40% of the previously identified
  reusable surface applies here; the rest is agent-layer (§8).
- **Overstated in the proposal:** "install/configure once, expose to all clients." That
  is already true via `mcp.json` and the launcher's dialect rendering. The aggregator's
  real justification is credential containment, centralized policy, local sampling, and
  remote-only client reach.

---

## 14. Phased plan

Each phase is independently useful and independently abandonable.

| Phase | Work | Depends on | Risk |
| --- | --- | --- | --- |
| **0. Justify** | Decide whether credential containment + sampling + policy are worth a new SPOF, given `mcp.json` already solves distribution. Write down the answer. | — | Low. Skipping it is the main risk. |
| **1. Supervisor** | `rmcp` client-side only. Spawn, handshake, capability capture, health with TTL+grace, backoff restart. Surface status in the UI. **No inbound server yet.** | rmcp | Low. Useful alone: Sarathi can finally show whether a configured server actually works. |
| **2. Lifecycle** | Tray + close-to-tray, or extract `sarathi-mcpd`. Must land before anything depends on the aggregator. | — | Medium. Tauri tray on Windows; decide single-vs-two-binary early. |
| **3. Auth + permissions** | Per-client tokens issued through the existing config-generation path; grant sets; default-deny for mutating tools; audit log. | 1 | Medium. Get the model right before clients depend on it. |
| **4. Aggregator core** | Session registry, `/mcp` POST/GET/DELETE, routing table, namespacing, `tools/list_changed` fan-out. | 1, 2, 3 | **High.** Session semantics and SSE resumability are the fiddly parts. |
| **5. Normalization** | Port the Hermes safe-subset sanitizer to Rust. Opt-in profiles behind a query param. | 4 | Low. Pure functions with a known test corpus. |
| **6. Sampling/elicitation** | Terminate `sampling/createMessage` on the local model with UI approval; elicitation via Tauri dialog. | 4 | Medium. Queue starvation behind long generations needs a timeout policy. |
| **7. Sharing policy** | Per-server `shared`/`per-session`; per-session instances for playwright and session-scoped crawl4ai. | 4 | Medium. Silent cross-contamination if wrong. |
| **8. Client rollout** | Third dialect in `launcher/mcp.rs` emitting remote entries; hybrid mode keeping stateful servers local. | 4, 7 | Medium. Tool renames break existing permission rules — publish a map. |

**Hybrid is the recommended end state, not full centralization.** Stateless, credentialed,
policy-worthy servers go through the aggregator; stateful ones stay distributed via
`mcp.json`. This keeps the availability regression bounded and the sharing hazards away
from the servers that cannot tolerate them.

---

## 15. Prerequisites and blockers

**Hard blockers (must resolve before clients depend on the aggregator):**

1. **Background lifecycle.** No tray, no close-to-tray, gateway dies with the window.
2. **Authentication.** The gateway is unauthenticated by design; tool execution cannot be.

**Design decisions required before Phase 4:**

3. One aggregator entry or one per downstream server (§4 — permission breakage).
4. In-app or separate `sarathi-mcpd` process (§10).
5. Sharing policy per server (§10).
6. How per-session normalization profiles are selected (§6).

**Non-blockers, worth noting:** `rmcp` targets a newer spec revision than the
2025-06-18 baseline described here — confirm the negotiated version range against the
installed Claude Code and opencode before committing. Hermes' MIT notice must be
retained for any ported code.

---

## 16. Recommendation

Build **Phase 1 (supervisor)** next, and only then revisit whether the aggregator is
worth it. Phase 1 has no protocol risk, no security exposure and no lifecycle
dependency; it makes `mcp.json` observable — Sarathi could tell the user *which*
configured servers actually start and answer, which nothing does today. It is also a
strict prerequisite for every later phase.

Do not build the `/mcp` endpoint until Phases 2 and 3 exist. An unauthenticated tool
aggregator on loopback is a local privilege-escalation surface, and an aggregator that
dies with the window is worse than the file-distribution approach it replaces.
