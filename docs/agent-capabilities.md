# Agent capability layer

Sarathi launches agent CLIs against its local model. This layer gives those
agents something to *do* with it: search the web, read pages, clone and analyse
repositories, and answer questions grounded in both — all locally, with no
third-party API keys.

Everything is exposed as ordinary [MCP](https://modelcontextprotocol.io) servers
listed in one registry, so the same set is available whether an agent was
started by Sarathi or configured by hand against some other gateway.

## The registry

```
%APPDATA%\com.sarathi.app\mcp.json
```

One file, in the `mcpServers` shape most clients already read:

```json
{
  "mcpServers": {
    "searxng": {
      "command": "mcp-searxng.cmd",
      "args": [],
      "env": { "SEARXNG_URL": "http://127.0.0.1:8888" }
    }
  }
}
```

Sarathi reads it on every launch and writes it into each tool's generated config
in that tool's own dialect — opencode nests the command and its arguments in one
array and calls the environment `environment`; everything else uses
`command`/`args`/`env`. Adding a server once therefore reaches every tool, and
the file is a perfectly ordinary MCP config for any client Sarathi never
launched.

`mcp.json` carries local service tokens and lives outside the repository. Do not
commit it.

## What is registered

| Server | Capability | Why this one |
| --- | --- | --- |
| `searxng` | Web search | [SearxNG](https://github.com/searxng/searxng) self-hosted, aggregating public engines with no account or key. `mcp-searxng` is the maintained bridge. |
| `crawl4ai` | Crawling / scraping | [Crawl4AI](https://github.com/unclecode/crawl4ai) fetches statically first and only drives its headless browser when a page needs it. |
| `research` | Source-grounded research | Local: see below. |
| `git` | Repository analysis | The reference `mcp-server-git`. Operates on local checkouts, so no GitHub account or token is involved. |
| `playwright` | Headless browser | For pages that only exist after JavaScript runs, and for interaction a fetch cannot express. |

## The research server

`sidecars/mcp/sarathi_research/server.py` — the NotebookLM-shaped piece, with
nothing leaving the machine.

Web pages (via Crawl4AI) and git repositories (via plain `git clone`) are chunked,
embedded with a local ONNX model, and stored in one SQLite file indexed by
`sqlite-vec`. Putting both in the *same* index is the point: one query returns
the blog post and the source file that contradicts it, each carrying enough
provenance to cite — a URL for a page, a file and line range for a repository.

| Tool | Does |
| --- | --- |
| `research_ingest_url` | Fetch a page and index it |
| `research_ingest_repo` | Shallow-clone a repository and index its source |
| `research_ingest_text` | Index notes or a transcript |
| `research_search` | Retrieve passages with citations, across all source types |
| `research_ask` | Retrieve and answer, grounded in what was retrieved |
| `research_list_notebooks`, `research_sources`, `research_forget` | Manage the library |
| `research_health` | Report whether its dependencies are reachable |

Notebooks are independent indexes; pass `notebook` to keep unrelated research
apart.

`research_ask` returns the evidence with `[S#]` citation markers for the calling
agent's model to synthesise from. That is the portable path — it works from any
client against any backend. Setting `RESEARCH_LLM_BASE_URL` and
`RESEARCH_LLM_MODEL` makes the server write the answer itself instead; any
OpenAI-compatible endpoint will do, Sarathi's own included.

## Services

Two long-running containers. The MCP servers themselves are stdio processes
started on demand by whichever client is using them, so there is nothing to
start or stop for those.

```powershell
.\scripts\mcp-services.ps1 status     # what is up, and is it answering
.\scripts\mcp-services.ps1 start
.\scripts\mcp-services.ps1 stop
.\scripts\mcp-services.ps1 restart
```

`status` reports from whether each service *answers*, not from whether a
container exists — a running container that is not serving looks identical to a
healthy one in `docker ps`, and is the case worth catching.

## Tool calls through Sarathi's gateway

An MCP server is only useful if the model is told it exists. Sarathi's gateway
accepts `tools` on both protocol surfaces, passes them to the chat template, and
parses what comes back into `tool_calls` (OpenAI) or `tool_use` blocks
(Anthropic). Three emission formats are recognised — ChatML `<tool_call>` tags,
Llama 3.1 bare JSON, and Mistral `[TOOL_CALLS]` — because a GGUF has no channel
other than its own text to say it wants a tool run.

Two consequences worth knowing:

- The model must have a chat template that supports tools. A GGUF whose template
  ignores the `tools` variable will list them and never call one; the runtime
  logs a warning when it has to fall back to a path that cannot carry them.
- When a request carries tools, streaming is buffered — a tool call is only
  recognisable once all of it has arrived. Plain chat still streams token by
  token.

## Inspecting what a tool will receive

```bash
cargo run --example render_client_configs -- "%APPDATA%\com.sarathi.app" ./out
```

Writes the configs a launch would write, using the same code path, without
launching anything.
