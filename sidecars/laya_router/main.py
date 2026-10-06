"""Sarathi's Laya capability router.

Reads JSON-RPC 2.0 frames on stdin and answers on stdout, one JSON object per
line -- the transport the memory engine sidecar already uses. No socket: nothing
to bind, no port to collide with (ARJUN runs its own Laya sidecar on this
machine), and nothing another process could reach. `laya-serve` was not used
because it binds 0.0.0.0 unless told otherwise.

Methods:

    laya.ping      -> {"ok": true}
    laya.load      -> loads the installed checkpoints; returns laya.status
    laya.status    -> device, threads, versions, question fingerprint
    laya.classify  {"prompt": str} -> choice, probabilities, answerConfidence
    laya.choose    {"text": str, "instructions": str, "labels": [str]} -> the same,
                   for ad-hoc questions such as an adapter's use case

Environment (all optional except the model directory):

    SARATHI_LAYA_DIR      the convaiinnovations/laya bundle, provisioned by
                          scripts/laya-setup.ps1
    SARATHI_LAYA_DEVICE   torch device, default "cpu"
    SARATHI_LAYA_THREADS  torch intra-op threads, default physical cores (max 8)
"""

import os

# Before anything can import huggingface_hub or transformers: Laya downloads a
# checkpoint whenever the path it is handed does not exist, and a chat turn must
# never turn into a silent multi-gigabyte download.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import json  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402

sidecar_dir = os.path.dirname(os.path.abspath(__file__))
if sidecar_dir not in sys.path:
    sys.path.insert(0, sidecar_dir)


def methods_for(engine):
    return {
        "laya.ping": lambda params: {"ok": True},
        "laya.load": lambda params: engine.load(),
        "laya.status": lambda params: engine.status(),
        "laya.classify": lambda params: engine.classify(str(params["prompt"])),
        "laya.choose": lambda params: engine.choose(
            str(params["text"]), str(params["instructions"]), list(params["labels"])
        ),
    }


def handle(line, methods):
    """One request line in, one response object out. Never raises."""
    try:
        req = json.loads(line)
    except Exception as err:
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error: %s" % err}}

    req_id = req.get("id") if isinstance(req, dict) else None
    try:
        if not isinstance(req, dict):
            raise ValueError("a request must be a JSON object")
        method = methods.get(req.get("method"))
        if method is None:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": "unknown method %r" % req.get("method")},
            }
        return {"jsonrpc": "2.0", "id": req_id, "result": method(req.get("params") or {})}
    except Exception as ex:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32603, "message": str(ex), "data": traceback.format_exc()},
        }


def serve(stdin, protocol, methods):
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        # `ensure_ascii` keeps every response valid whatever the console's code
        # page is; the parent decodes the escapes.
        protocol.write(json.dumps(handle(line, methods), ensure_ascii=True) + "\n")
        protocol.flush()


def main():
    # The protocol owns stdout. Laya and torch print warnings there, and one
    # stray line would be read by the parent as a malformed response, so
    # everything that is not a response goes to stderr.
    protocol = sys.stdout
    sys.stdout = sys.stderr
    # The parent writes UTF-8; Python would otherwise decode stdin with the
    # locale's code page and mangle a non-English prompt before Laya saw it.
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

    from engine import CapabilityEngine

    serve(sys.stdin, protocol, methods_for(CapabilityEngine()))


if __name__ == "__main__":
    main()
