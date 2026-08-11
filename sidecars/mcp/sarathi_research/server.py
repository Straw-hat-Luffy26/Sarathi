"""Sarathi Research — a local, source-grounded research index over MCP.

The NotebookLM-shaped capability, with nothing leaving the machine: pages are
fetched by the local Crawl4AI service, repositories by plain `git clone`, and
both land as chunks in one SQLite file with a `sqlite-vec` index over
locally-computed embeddings.

The point of putting web pages and repository files in the *same* index is
cross-source synthesis: one query returns the blog post and the source file that
contradicts it, each carrying enough provenance to cite.

Deliberately not in here: any judgement about what the passages mean. Retrieval
returns evidence with citations and the calling agent's own model writes the
answer, which is what keeps this usable from any client against any gateway.
`research_ask` can optionally do that synthesis itself against an
OpenAI-compatible endpoint named by the environment — Sarathi, Ollama, vLLM,
SGLang and NIM are all just a base URL here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import sqlite_vec
from mcp.server import MCPServer

# ─── Configuration ──────────────────────────────────────────────────────────
#
# Every path and endpoint is environment-driven so the same file serves
# Sarathi's launched agents and a bare `claude mcp add` on the same machine.

DATA_DIR = Path(
    os.environ.get("SARATHI_RESEARCH_DIR")
    or Path(os.environ.get("APPDATA", Path.home())) / "com.sarathi.app" / "research"
)
DB_PATH = DATA_DIR / "library.db"
REPO_DIR = DATA_DIR / "repos"
# fastembed defaults to the system temp folder, so a routine cleanup silently
# costs a re-download of the model on the next call.
MODEL_CACHE = DATA_DIR / "models"

CRAWL4AI_URL = os.environ.get("CRAWL4AI_BASE_URL", "http://127.0.0.1:11235").rstrip("/")
CRAWL4AI_TOKEN = os.environ.get("CRAWL4AI_API_KEY", "")

EMBED_MODEL = os.environ.get("SARATHI_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = int(os.environ.get("SARATHI_EMBED_DIM", "384"))

# Optional synthesis endpoint. Absent, `research_ask` returns evidence only.
LLM_BASE_URL = os.environ.get("RESEARCH_LLM_BASE_URL", "").rstrip("/")
LLM_MODEL = os.environ.get("RESEARCH_LLM_MODEL", "")
LLM_KEY = os.environ.get("RESEARCH_LLM_API_KEY", "sarathi-local")

CHUNK_CHARS = 1200
CHUNK_OVERLAP = 150

# Extensions worth indexing from a repository. A checkout is mostly bytes that
# would only dilute the index — lockfiles, minified bundles, images — so this is
# an allowlist rather than a denylist of the usual suspects.
CODE_SUFFIXES = {
    ".py", ".rs", ".ts", ".tsx", ".js", ".jsx", ".go", ".java", ".kt", ".rb",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".swift", ".scala", ".sh", ".ps1",
    ".sql", ".toml", ".yaml", ".yml", ".json", ".md", ".mdx", ".rst", ".txt",
    ".proto", ".graphql", ".tf", ".vue", ".svelte", ".lua", ".ex", ".exs",
}
SKIP_DIRS = {
    ".git", "node_modules", "target", "dist", "build", "vendor", "__pycache__",
    ".venv", "venv", ".next", ".nuxt", "coverage", ".pytest_cache", ".mypy_cache",
}
MAX_FILE_BYTES = 400_000


# ─── Embeddings ─────────────────────────────────────────────────────────────

_embedder = None


def embedder():
    """Loads the ONNX embedding model once, on first use.

    Import and construction are deferred because both are slow, and a client
    that only ever calls `research_list_notebooks` should not pay for them —
    an MCP server that takes ten seconds to answer `tools/list` looks hung.
    """
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding

        MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        _embedder = TextEmbedding(model_name=EMBED_MODEL, cache_dir=str(MODEL_CACHE))
    return _embedder


def embed_passages(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in embedder().embed(texts)]


def embed_query(text: str) -> list[float]:
    """Embeds a question.

    BGE models are trained with an asymmetric prefix on the query side; using
    the passage encoder for both halves measurably degrades recall, so this
    goes through `query_embed` rather than `embed`.
    """
    return list(embedder().query_embed([text]))[0].tolist()


# ─── Storage ────────────────────────────────────────────────────────────────


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    db.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS sources (
            id          INTEGER PRIMARY KEY,
            notebook    TEXT NOT NULL,
            source_type TEXT NOT NULL,      -- web | repo | text
            origin      TEXT NOT NULL,      -- URL, repo URL, or a label
            title       TEXT NOT NULL,
            path        TEXT,               -- file within a repo
            fetched_at  INTEGER NOT NULL,
            content_sha TEXT NOT NULL,
            UNIQUE (notebook, origin, path)
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id         INTEGER PRIMARY KEY,
            source_id  INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
            notebook   TEXT NOT NULL,
            ordinal    INTEGER NOT NULL,
            start_line INTEGER,
            end_line   INTEGER,
            text       TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS chunks_by_source ON chunks(source_id);
        CREATE INDEX IF NOT EXISTS chunks_by_notebook ON chunks(notebook);

        -- Cosine, not the vec0 default of L2. BGE embeddings are trained for
        -- cosine similarity, and scoring them by euclidean distance both ranks
        -- worse and makes the reported score meaningless.
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
            chunk_id  INTEGER PRIMARY KEY,
            notebook  TEXT PARTITION KEY,
            embedding FLOAT[{EMBED_DIM}] distance_metric=cosine
        );
        """
    )
    db.execute("PRAGMA foreign_keys = ON")
    _ensure_vector_metric(db)
    return db


def _ensure_vector_metric(db: sqlite3.Connection) -> None:
    """Rebuilds the vector index if it predates the current distance metric.

    `CREATE VIRTUAL TABLE IF NOT EXISTS` is a no-op against an existing table,
    so changing the metric in the schema above silently does nothing to an index
    already on disk — it keeps ranking by the old one while the code reports
    scores computed for the new. Detecting that here is what stops a metric
    change from being a change only in the source.
    """
    row = db.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunk_vectors'"
    ).fetchone()
    if row is None or "distance_metric=cosine" in (row["sql"] or ""):
        return

    chunks = db.execute("SELECT id, notebook, text FROM chunks ORDER BY id").fetchall()
    db.execute("DROP TABLE chunk_vectors")
    db.execute(
        f"CREATE VIRTUAL TABLE chunk_vectors USING vec0("
        f"chunk_id INTEGER PRIMARY KEY, notebook TEXT PARTITION KEY,"
        f" embedding FLOAT[{EMBED_DIM}] distance_metric=cosine)"
    )

    # Re-embedded in batches: the vectors lived only in the table just dropped,
    # so there is nothing to copy across.
    for i in range(0, len(chunks), 64):
        batch = chunks[i : i + 64]
        for row_, vec in zip(batch, embed_passages([r["text"] for r in batch])):
            db.execute(
                "INSERT INTO chunk_vectors (chunk_id, notebook, embedding) VALUES (?,?,?)",
                (row_["id"], row_["notebook"], json.dumps(vec)),
            )
    db.commit()
    print(
        f"[sarathi-research] rebuilt vector index for cosine ({len(chunks)} chunks)",
        file=sys.stderr,
    )


# ─── Chunking ───────────────────────────────────────────────────────────────


@dataclass
class Chunk:
    text: str
    start_line: int
    end_line: int


def chunk_text(body: str) -> list[Chunk]:
    """Splits on blank lines, packing paragraphs up to the target size.

    Line numbers are carried through because a repository citation that cannot
    say *where* in the file is not much of a citation.
    """
    lines = body.splitlines()
    chunks: list[Chunk] = []

    buf: list[str] = []
    buf_start = 1
    size = 0

    def flush(end_line: int) -> None:
        nonlocal buf, size, buf_start
        text = "\n".join(buf).strip()
        if text:
            chunks.append(Chunk(text=text, start_line=buf_start, end_line=end_line))
        buf, size = [], 0

    for i, line in enumerate(lines, start=1):
        if not buf:
            buf_start = i
        buf.append(line)
        size += len(line) + 1

        if size >= CHUNK_CHARS:
            flush(i)
            # Carry the tail forward so a fact split across the boundary is
            # still retrievable from one side of it.
            overlap: list[str] = []
            taken = 0
            for prev in reversed(chunks[-1].text.splitlines()):
                if taken + len(prev) > CHUNK_OVERLAP:
                    break
                overlap.insert(0, prev)
                taken += len(prev) + 1
            if overlap:
                buf = overlap
                buf_start = max(1, i - len(overlap) + 1)
                size = taken

    flush(len(lines) if lines else 1)
    return chunks


# ─── Ingestion ──────────────────────────────────────────────────────────────


def store_document(
    db: sqlite3.Connection,
    *,
    notebook: str,
    source_type: str,
    origin: str,
    title: str,
    path: str | None,
    body: str,
) -> dict[str, Any]:
    """Indexes one document, replacing any earlier copy of the same origin.

    Re-ingesting is how a page gets refreshed, so an unchanged document is
    detected by hash and skipped rather than re-embedded — embedding is the
    slow part, and a repo re-index otherwise pays for every unchanged file.
    """
    sha = hashlib.sha256(body.encode("utf-8", "replace")).hexdigest()

    existing = db.execute(
        "SELECT id, content_sha FROM sources WHERE notebook=? AND origin=? AND path IS ?",
        (notebook, origin, path),
    ).fetchone()

    if existing and existing["content_sha"] == sha:
        return {"status": "unchanged", "source_id": existing["id"], "chunks": 0}

    if existing:
        ids = [r["id"] for r in db.execute("SELECT id FROM chunks WHERE source_id=?", (existing["id"],))]
        for cid in ids:
            db.execute("DELETE FROM chunk_vectors WHERE chunk_id=?", (cid,))
        db.execute("DELETE FROM chunks WHERE source_id=?", (existing["id"],))
        db.execute("DELETE FROM sources WHERE id=?", (existing["id"],))

    cur = db.execute(
        "INSERT INTO sources (notebook, source_type, origin, title, path, fetched_at, content_sha)"
        " VALUES (?,?,?,?,?,?,?)",
        (notebook, source_type, origin, title, path, int(time.time()), sha),
    )
    source_id = cur.lastrowid

    pieces = chunk_text(body)
    if not pieces:
        db.commit()
        return {"status": "empty", "source_id": source_id, "chunks": 0}

    vectors = embed_passages([p.text for p in pieces])
    for ordinal, (piece, vec) in enumerate(zip(pieces, vectors)):
        c = db.execute(
            "INSERT INTO chunks (source_id, notebook, ordinal, start_line, end_line, text)"
            " VALUES (?,?,?,?,?,?)",
            (source_id, notebook, ordinal, piece.start_line, piece.end_line, piece.text),
        )
        db.execute(
            "INSERT INTO chunk_vectors (chunk_id, notebook, embedding) VALUES (?,?,?)",
            (c.lastrowid, notebook, json.dumps(vec)),
        )

    db.commit()
    return {"status": "indexed", "source_id": source_id, "chunks": len(pieces)}


def fetch_markdown(url: str, query: str | None = None) -> tuple[str, str]:
    """Fetches a page as markdown through the local Crawl4AI service.

    Crawl4AI is static-first and only drives its headless browser when a page
    needs it, which is the behaviour wanted here — spinning up Chromium to read
    a documentation page is pure latency.
    """
    payload: dict[str, Any] = {"url": url, "f": "bm25" if query else "fit"}
    if query:
        payload["q"] = query

    req = urllib.request.Request(
        f"{CRAWL4AI_URL}/md",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {CRAWL4AI_TOKEN}"} if CRAWL4AI_TOKEN else {}),
        },
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8", "replace"))

    body = data.get("markdown") or data.get("result") or ""
    if isinstance(body, dict):
        body = body.get("fit_markdown") or body.get("raw_markdown") or ""

    # First heading of any level: readability extraction often drops the H1, and
    # falling back to the raw URL makes every citation from a site look alike.
    title = ""
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            candidate = stripped.lstrip("#").strip()
            if candidate:
                title = candidate
                break
    return body, (title or url)


def run_git(args: list[str], cwd: Path | None = None, timeout: int = 600) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        # A checkout must never stop to ask for a password: without this a
        # private or mistyped URL hangs the tool call until it times out.
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "echo"},
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "git failed").strip()[:800])
    return proc.stdout


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:80] or "repo"


# ─── MCP surface ────────────────────────────────────────────────────────────

server = MCPServer(
    name="sarathi-research",
    instructions=(
        "Local source-grounded research. Ingest web pages and git repositories "
        "into a notebook, then search or ask across everything in it. Every "
        "passage returned carries a citation marker and its origin, so answers "
        "can be traced back to a URL or a file and line range."
    ),
)


@server.tool()
def research_ingest_url(url: str, notebook: str = "default", focus: str = "") -> dict:
    """Fetch a web page and index it into a research notebook.

    Args:
        url: Absolute http/https URL to ingest.
        notebook: Notebook to file it under. Notebooks are independent indexes.
        focus: Optional topic; ranks the extracted text toward it (BM25) rather
            than taking whole-page readable content.
    """
    body, title = fetch_markdown(url, focus or None)
    if not body.strip():
        return {"ok": False, "error": f"no readable content extracted from {url}"}

    db = connect()
    try:
        result = store_document(
            db, notebook=notebook, source_type="web", origin=url,
            title=title, path=None, body=body,
        )
    finally:
        db.close()
    return {"ok": True, "url": url, "title": title, "characters": len(body), **result}


@server.tool()
def research_ingest_repo(
    repo: str,
    notebook: str = "default",
    ref: str = "",
    include: str = "",
    max_files: int = 400,
) -> dict:
    """Clone a git repository and index its source files into a notebook.

    Uses plain `git clone` over https, so any public repository works without a
    GitHub account, token or API call.

    Args:
        repo: Repository URL (or an absolute path to a local checkout).
        notebook: Notebook to file it under.
        ref: Optional branch or tag; defaults to the repository's default branch.
        include: Optional comma-separated path prefixes to restrict indexing,
            e.g. "src,docs".
        max_files: Safety limit on how many files to index.
    """
    REPO_DIR.mkdir(parents=True, exist_ok=True)

    local = Path(repo)
    if local.is_dir() and (local / ".git").exists():
        checkout, origin = local, str(local)
    else:
        origin = repo
        checkout = REPO_DIR / slug(repo.rstrip("/").split("/")[-1].removesuffix(".git"))
        if checkout.exists():
            shutil.rmtree(checkout, ignore_errors=True)
        # Shallow and blobless: the history is not what gets indexed, and a full
        # clone of a large repository is minutes of waiting for nothing.
        args = ["clone", "--depth", "1", "--filter=blob:none", "--single-branch"]
        if ref:
            args += ["--branch", ref]
        run_git([*args, repo, str(checkout)])

    head = run_git(["rev-parse", "HEAD"], cwd=checkout).strip()[:12]
    prefixes = [p.strip() for p in include.split(",") if p.strip()]

    indexed, skipped, chunks = 0, 0, 0
    db = connect()
    try:
        for path in sorted(checkout.rglob("*")):
            if indexed >= max_files:
                break
            if not path.is_file() or path.suffix.lower() not in CODE_SUFFIXES:
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue

            rel = path.relative_to(checkout).as_posix()
            if prefixes and not any(rel.startswith(p) for p in prefixes):
                continue
            if path.stat().st_size > MAX_FILE_BYTES:
                skipped += 1
                continue

            try:
                body = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                skipped += 1
                continue
            if not body.strip():
                continue

            result = store_document(
                db, notebook=notebook, source_type="repo", origin=origin,
                title=rel, path=rel, body=body,
            )
            indexed += 1
            chunks += result["chunks"]
    finally:
        db.close()

    return {
        "ok": True, "repo": origin, "commit": head, "checkout": str(checkout),
        "files_indexed": indexed, "files_skipped": skipped, "chunks": chunks,
        "notebook": notebook,
    }


@server.tool()
def research_ingest_text(text: str, title: str, notebook: str = "default") -> dict:
    """Index a block of text (notes, a transcript, pasted content) into a notebook."""
    db = connect()
    try:
        result = store_document(
            db, notebook=notebook, source_type="text", origin=f"text:{title}",
            title=title, path=None, body=text,
        )
    finally:
        db.close()
    return {"ok": True, "title": title, **result}


def _retrieve(
    notebook: str, query: str, k: int, source_type: str | None
) -> list[dict[str, Any]]:
    vec = embed_query(query)
    db = connect()
    try:
        # Over-fetched because the type filter is applied after the vector
        # search: asking vec0 for exactly k and then discarding the web hits
        # would quietly return fewer repo results than requested.
        pool = k * 5 if source_type else k
        rows = db.execute(
            "SELECT chunk_id, distance FROM chunk_vectors"
            " WHERE embedding MATCH ? AND notebook = ? AND k = ?"
            " ORDER BY distance",
            (json.dumps(vec), notebook, pool),
        ).fetchall()

        hits: list[dict[str, Any]] = []
        for row in rows:
            meta = db.execute(
                "SELECT c.text, c.start_line, c.end_line, s.source_type, s.origin,"
                "       s.title, s.path, s.fetched_at"
                " FROM chunks c JOIN sources s ON s.id = c.source_id WHERE c.id = ?",
                (row["chunk_id"],),
            ).fetchone()
            if meta is None:
                continue
            if source_type and meta["source_type"] != source_type:
                continue

            location = meta["origin"]
            if meta["source_type"] == "repo":
                location = f"{meta['origin']} :: {meta['path']}#L{meta['start_line']}-{meta['end_line']}"

            hits.append({
                "citation": f"S{len(hits) + 1}",
                "score": round(1.0 - float(row["distance"]), 4),
                "source_type": meta["source_type"],
                "title": meta["title"],
                "origin": meta["origin"],
                "path": meta["path"],
                "lines": [meta["start_line"], meta["end_line"]] if meta["source_type"] == "repo" else None,
                "location": location,
                "text": meta["text"],
            })
            if len(hits) >= k:
                break
        return hits
    finally:
        db.close()


@server.tool()
def research_search(
    query: str,
    notebook: str = "default",
    k: int = 8,
    source_type: Literal["", "web", "repo", "text"] = "",
) -> dict:
    """Search a notebook and return the matching passages with their citations.

    Spans web pages and repository files together, so one query can surface both.

    Args:
        query: What to look for.
        notebook: Notebook to search.
        k: How many passages to return.
        source_type: Restrict to one kind of source; empty means all of them.
    """
    hits = _retrieve(notebook, query, k, source_type or None)
    return {
        "ok": True, "notebook": notebook, "query": query,
        "results": hits,
        "sources": [{"citation": h["citation"], "location": h["location"]} for h in hits],
    }


@server.tool()
def research_ask(question: str, notebook: str = "default", k: int = 10) -> dict:
    """Answer a question from a notebook, grounded in its indexed sources.

    Returns the retrieved evidence with citation markers, and — when a synthesis
    endpoint is configured (RESEARCH_LLM_BASE_URL) — an answer written against
    only that evidence. Without one, the evidence comes back for the calling
    agent's own model to synthesise, which is the portable path.

    Args:
        question: The question to answer.
        notebook: Notebook to answer from.
        k: How many passages to ground the answer in.
    """
    hits = _retrieve(notebook, question, k, None)
    if not hits:
        return {
            "ok": True, "answer": None, "evidence": [],
            "note": f"Notebook '{notebook}' has nothing matching. Ingest sources first.",
        }

    evidence = "\n\n".join(
        f"[{h['citation']}] ({h['source_type']}) {h['location']}\n{h['text']}" for h in hits
    )
    packet = {
        "ok": True,
        "notebook": notebook,
        "question": question,
        "evidence": hits,
        "citations": [{"citation": h["citation"], "location": h["location"]} for h in hits],
    }

    if not (LLM_BASE_URL and LLM_MODEL):
        packet["answer"] = None
        packet["instruction"] = (
            "Answer using only the evidence above. Cite every claim with its [S#] "
            "marker and do not assert anything the passages do not support."
        )
        return packet

    prompt = (
        "Answer the question using only the sources below. Cite every claim with "
        "its [S#] marker. If the sources do not answer it, say so.\n\n"
        f"SOURCES:\n{evidence}\n\nQUESTION: {question}"
    )
    body = json.dumps({
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{LLM_BASE_URL}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {LLM_KEY}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        packet["answer"] = data["choices"][0]["message"]["content"]
    except (urllib.error.URLError, KeyError, TimeoutError, json.JSONDecodeError) as e:
        # The evidence is still worth returning: a synthesis endpoint being down
        # should degrade to "here are the sources", not to a failed tool call.
        packet["answer"] = None
        packet["synthesis_error"] = f"{type(e).__name__}: {e}"
    return packet


@server.tool()
def research_list_notebooks() -> dict:
    """List the research notebooks and what each contains."""
    db = connect()
    try:
        rows = db.execute(
            "SELECT notebook, source_type, COUNT(*) n FROM sources"
            " GROUP BY notebook, source_type ORDER BY notebook"
        ).fetchall()
    finally:
        db.close()

    books: dict[str, dict[str, int]] = {}
    for r in rows:
        books.setdefault(r["notebook"], {})[r["source_type"]] = r["n"]
    return {"ok": True, "notebooks": books, "database": str(DB_PATH)}


@server.tool()
def research_sources(notebook: str = "default", limit: int = 100) -> dict:
    """List the sources indexed in a notebook, so citations can be resolved."""
    db = connect()
    try:
        rows = db.execute(
            "SELECT s.source_type, s.origin, s.title, s.path, s.fetched_at,"
            "       (SELECT COUNT(*) FROM chunks c WHERE c.source_id = s.id) chunks"
            " FROM sources s WHERE s.notebook = ? ORDER BY s.fetched_at DESC LIMIT ?",
            (notebook, limit),
        ).fetchall()
    finally:
        db.close()
    return {"ok": True, "notebook": notebook, "sources": [dict(r) for r in rows]}


@server.tool()
def research_forget(notebook: str, origin: str = "") -> dict:
    """Delete a notebook, or one source within it.

    Args:
        notebook: Notebook to remove from.
        origin: A specific URL or repository to drop; empty clears the notebook.
    """
    db = connect()
    try:
        if origin:
            ids = [r["id"] for r in db.execute(
                "SELECT id FROM sources WHERE notebook=? AND origin=?", (notebook, origin))]
        else:
            ids = [r["id"] for r in db.execute(
                "SELECT id FROM sources WHERE notebook=?", (notebook,))]

        removed = 0
        for sid in ids:
            for c in db.execute("SELECT id FROM chunks WHERE source_id=?", (sid,)).fetchall():
                db.execute("DELETE FROM chunk_vectors WHERE chunk_id=?", (c["id"],))
            db.execute("DELETE FROM chunks WHERE source_id=?", (sid,))
            db.execute("DELETE FROM sources WHERE id=?", (sid,))
            removed += 1
        db.commit()
    finally:
        db.close()
    return {"ok": True, "notebook": notebook, "sources_removed": removed}


@server.tool()
def research_health() -> dict:
    """Report whether the pieces this server depends on are actually reachable."""
    status: dict[str, Any] = {"database": str(DB_PATH), "embed_model": EMBED_MODEL}

    try:
        req = urllib.request.Request(f"{CRAWL4AI_URL}/health")
        with urllib.request.urlopen(req, timeout=10) as resp:
            status["crawl4ai"] = json.loads(resp.read().decode())
    except Exception as e:  # noqa: BLE001 — a health check reports, never raises
        status["crawl4ai"] = {"error": f"{type(e).__name__}: {e}"}

    try:
        run_git(["--version"], timeout=15)
        status["git"] = "available"
    except Exception as e:  # noqa: BLE001
        status["git"] = f"unavailable: {e}"

    try:
        db = connect()
        status["chunks"] = db.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
        db.close()
    except Exception as e:  # noqa: BLE001
        status["chunks"] = f"error: {e}"

    status["synthesis_endpoint"] = LLM_BASE_URL or "not configured (evidence-only mode)"
    return {"ok": True, **status}


if __name__ == "__main__":
    server.run("stdio")
