"""Install Laya for Sarathi's capability (LoRA) routing.

    python scripts/laya_setup.py                 # English checkpoint (~0.85 GB)
    python scripts/laya_setup.py --multilingual  # plus 100+ languages (~0.68 GB more)
    python scripts/laya_setup.py --check         # verify only, download nothing
    python scripts/laya_setup.py --no-xet        # plain HTTP, if the download stalls

What it does, in order:

1. Checks the system interpreter already has torch >= 2.0 and transformers
   >= 4.48. It never installs or upgrades them: other tools on this machine --
   ARJUN's sidecars among them -- share this interpreter, and swapping its torch
   is not a side effect a routing feature gets to have.
2. Installs `laya==0.3.20` with `--no-deps`. That is the exact version ARJUN
   pins, so one install serves both applications and neither can move it under
   the other. A different laya version already present is left alone unless
   `--force-package` is given.
3. Downloads the convaiinnovations/laya bundle, pinned to one revision, into
   Sarathi's own app-data directory -- never ARJUN's.
4. Starts the sidecar the app will start and classifies three prompts, so a
   broken install is found here rather than on a chat turn.

No virtual environment is created: Sarathi's Python targets the system
interpreter (see CLAUDE.md).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

LAYA_VERSION = "0.3.20"
BUNDLE_REPO = "convaiinnovations/laya"
# The bundle revision this release was verified against. Laya 0.3.20 does not
# pin one itself, and checkpoints published later may expect newer code.
BUNDLE_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
CHECKPOINT_FILES = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]

REPO_ROOT = Path(__file__).resolve().parent.parent
SIDECAR = REPO_ROOT / "sidecars" / "laya_router" / "main.py"


def app_data_dir() -> Path:
    """Where Tauri puts `com.sarathi.app`'s data on this platform."""
    if sys.platform == "win32":
        return Path(os.environ["APPDATA"]) / "com.sarathi.app"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "com.sarathi.app"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "com.sarathi.app"


def version_tuple(v: str):
    out = []
    for part in v.split("+")[0].split("."):
        digits = "".join(c for c in part if c.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def installed_version(dist: str):
    try:
        from importlib.metadata import version

        return version(dist)
    except Exception:
        return None


def check_runtime() -> None:
    needs = {"torch": "2.0", "transformers": "4.48"}
    missing = []
    for dist, floor in needs.items():
        have = installed_version(dist)
        if have is None or version_tuple(have) < version_tuple(floor):
            missing.append("%s>=%s (found %s)" % (dist, floor, have or "nothing"))
        else:
            print("  ok  %s %s" % (dist, have))
    if missing:
        sys.exit(
            "Laya needs %s in this interpreter (%s). Install them yourself -- this script will "
            "not change torch or transformers, because other tools share this interpreter. "
            "A CPU build is enough:\n  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  pip install \"transformers>=4.48\"" % (", ".join(missing), sys.executable)
        )


def ensure_package(force: bool) -> None:
    have = installed_version("laya")
    if have == LAYA_VERSION:
        print("  ok  laya %s" % have)
        return
    if have and not force:
        sys.exit(
            "laya %s is installed, not %s. Another application may depend on it (ARJUN pins %s). "
            "Re-run with --force-package to replace it." % (have, LAYA_VERSION, LAYA_VERSION)
        )
    print("  installing laya==%s (no dependency changes)" % LAYA_VERSION)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "laya==%s" % LAYA_VERSION, "--no-deps"])


DOWNLOAD_ATTEMPTS = 8


def download(dest: Path, multilingual: bool) -> None:
    from huggingface_hub import snapshot_download

    patterns = list(CHECKPOINT_FILES)
    if multilingual:
        patterns += ["multilingual/" + p for p in CHECKPOINT_FILES]
    print("  downloading %s@%s -> %s" % (BUNDLE_REPO, BUNDLE_REVISION[:8], dest))

    # A dropped connection or a failed DNS lookup part-way through 0.85 GB is
    # ordinary on a slow link. snapshot_download resumes from the partial file
    # it keeps under `.cache`, so retrying costs little and giving up costs the
    # whole download.
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            snapshot_download(
                BUNDLE_REPO,
                revision=BUNDLE_REVISION,
                local_dir=str(dest),
                allow_patterns=patterns,
                token=os.environ.get("HF_TOKEN") or None,
            )
            break
        except Exception as err:  # network errors arrive as several unrelated types
            if attempt == DOWNLOAD_ATTEMPTS:
                raise
            wait = min(120, 10 * 2 ** (attempt - 1))
            print("  attempt %d/%d failed (%s); resuming in %ds" % (attempt, DOWNLOAD_ATTEMPTS, err, wait))
            time.sleep(wait)
    (dest / "SARATHI_LAYA_REVISION").write_text(BUNDLE_REVISION + "\n", encoding="ascii")


def verify(dest: Path) -> None:
    prompts = [
        "refactor this rust function so it stops cloning the vector",
        "solve for x: 3x + 7 = 22",
        "thanks, that was helpful!",
    ]
    lines = [{"jsonrpc": "2.0", "id": 1, "method": "laya.load", "params": {}}]
    lines += [
        {"jsonrpc": "2.0", "id": i + 2, "method": "laya.classify", "params": {"prompt": p}}
        for i, p in enumerate(prompts)
    ]
    env = dict(os.environ, SARATHI_LAYA_DIR=str(dest), PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [sys.executable, str(SIDECAR)],
        input="".join(json.dumps(l) + "\n" for l in lines),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=600,
    )
    answers = [json.loads(l) for l in proc.stdout.splitlines() if l.strip()]
    if not answers:
        sys.exit("The sidecar produced no answer.\n%s" % proc.stderr[-2000:])
    for answer in answers:
        if "error" in answer:
            sys.exit("The sidecar failed: %s\n%s" % (answer["error"]["message"], proc.stderr[-2000:]))
    load = answers[0]["result"]
    print("  ok  loaded %s on %s in %.0f ms" % (", ".join(load["installed"]), load["device"], load["loadMs"] or 0))
    for prompt, answer in zip(prompts, answers[1:]):
        r = answer["result"]
        print("  ok  %-12s p=%.2f  %5.0f ms  %r" % (r["choice"], r["answerConfidence"] or 0, r["latencyMs"], prompt))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--multilingual", action="store_true", help="also install the multilingual checkpoint")
    parser.add_argument("--check", action="store_true", help="verify the existing install; download nothing")
    parser.add_argument("--force-package", action="store_true", help="replace a different installed laya version")
    parser.add_argument(
        "--no-xet",
        action="store_true",
        help="download over plain HTTP instead of HuggingFace's Xet client, which can stall "
        "without progress on some connections",
    )
    parser.add_argument("--dest", type=Path, default=app_data_dir() / "laya" / "convaiinnovations_laya")
    args = parser.parse_args()
    if args.no_xet:
        # Read when huggingface_hub is imported, which `download` does lazily.
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    print("Python: %s" % sys.executable)
    check_runtime()
    if args.check:
        if installed_version("laya") != LAYA_VERSION:
            sys.exit("laya %s is not installed" % LAYA_VERSION)
    else:
        ensure_package(args.force_package)
        download(args.dest, args.multilingual)
    verify(args.dest)
    print("\nDone. Restart Sarathi; Laya will decide capability switches from the next launch.")


if __name__ == "__main__":
    main()
