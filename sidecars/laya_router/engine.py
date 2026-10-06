"""Laya `choice` over Sarathi's capability slots.

Answers one question per chat turn -- which kind of work is being asked for --
and returns Laya's calibrated probabilities. The Rust side feeds that verdict to
the capability switch policy, which decides whether to bind a different LoRA
adapter. Laya never names a model or an adapter itself; it only replaces the
keyword counting that used to feed that decision.

Decisions checked against Laya 0.3.20's source, not its README:

* **Local weights only.** `laya.Agent` calls `huggingface_hub.snapshot_download`
  whenever the path it is handed does not exist. `main.py` turns the Hub's
  offline switches on before anything imports it, and `load` refuses to build a
  Router unless the checkpoint is already in Sarathi's own directory. A missing
  model is an error the app can show, never a multi-gigabyte download in the
  middle of a chat turn. `scripts/laya-setup.ps1` is what provisions it.
* **CPU by default.** The English checkpoint is 421M parameters, about 1.7 GB
  at fp32. On an 8 GB card that is the margin deciding whether the chat model
  fits entirely in VRAM, which matters far more to the user than shaving a
  hundred milliseconds off routing. `SARATHI_LAYA_DEVICE=cuda` opts in.
* **No silently truncated option descriptions.** `laya.common.build_sequence`
  cuts every option to fit a 192-token head budget without saying so. A
  description the model never read is a routing rule that does not exist, so
  `load` measures the budget with the checkpoint's own tokenizer and refuses to
  start rather than run with cut descriptions.
"""

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional

QUESTION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "capability_question.json")

# The bundle repo keeps the English checkpoint at its root and the multilingual
# one in a subfolder. `typed-decisions` is tuned on unrelated workflows and is
# never offered here.
CHECKPOINTS = {
    "english": None,
    "multilingual": "multilingual",
}

# Mirrors `laya.common.build_sequence` in 0.3.20.
OPTION_TOKEN_CAP = 48
MIN_OPTION_ROOM = 16


def load_question(path: str = QUESTION_PATH) -> Dict[str, Any]:
    """The capability question and a fingerprint of its exact bytes.

    Changing one word of a description changes the probabilities Laya returns,
    so the fingerprint is reported with every status: a threshold tuned against
    one wording can be seen not to apply to another.
    """
    with open(path, "rb") as handle:
        raw = handle.read()
    spec = json.loads(raw.decode("utf-8"))
    return {
        "version": spec["version"],
        "fingerprint": hashlib.sha256(raw).hexdigest(),
        "question_id": spec["questionId"],
        "questions": {
            spec["questionId"]: {
                "type": "choice",
                "instructions": spec["instructions"],
                "criteria": spec["criteria"],
            }
        },
    }


def thread_count(raw: Optional[str] = None, cpu_count: Optional[int] = None) -> int:
    """Torch intra-op threads for the forward pass.

    Laya's own CPU benchmarks found the physical core count best, with SMT
    siblings contending. `os.cpu_count()` counts logical CPUs, so half of it
    approximates the physical cores. Capped at 8 so routing never takes the
    whole machine away from the chat model generating beside it.
    """
    raw = (raw if raw is not None else os.environ.get("SARATHI_LAYA_THREADS", "")).strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    cpus = cpu_count if cpu_count is not None else (os.cpu_count() or 2)
    return max(1, min(8, cpus // 2))


def check_option_budget(name: str, tokenize, instructions: str, criteria: Dict[str, str], head_max_len: int) -> None:
    """Raise when the checkpoint would cut the question short.

    `tokenize(text) -> list of token ids` is the checkpoint's tokenizer, so the
    measurement is the one Laya itself will make. Each option is rendered as
    `[MASK] label: text` and capped at 48 tokens; when the options leave fewer
    than 16 tokens of the head budget, or fewer than the instruction needs,
    every option and then the instruction are truncated in place.
    """
    head = tokenize("choice question: %s" % instructions)
    used = 0
    for label, text in criteria.items():
        rendered = label if text in (None, "") else "%s: %s" % (label, text)
        ids = tokenize(" " + rendered)
        if len(ids) > OPTION_TOKEN_CAP:
            raise ValueError(
                "the %s checkpoint would cut the option %r at %d tokens (it needs %d); "
                "shorten it in capability_question.json" % (name, label, OPTION_TOKEN_CAP, len(ids))
            )
        used += 1 + len(ids)  # the [MASK] marker
    room = head_max_len - used
    if room < max(MIN_OPTION_ROOM, len(head)):
        raise ValueError(
            "the %s checkpoint has a %d-token head budget; the options use %d and the "
            "instruction %d, so Laya would truncate them. Shorten capability_question.json."
            % (name, head_max_len, used, len(head))
        )


def installed_checkpoints(model_dir: str) -> List[str]:
    return [
        name
        for name, sub in CHECKPOINTS.items()
        if os.path.isfile(os.path.join(model_dir, sub or "", "rl_agent_config.json"))
    ]


class CapabilityEngine:
    """One Laya Router, built once, answering one question per turn."""

    def __init__(self) -> None:
        self.model_dir = os.environ.get("SARATHI_LAYA_DIR", "")
        self.device = (os.environ.get("SARATHI_LAYA_DEVICE") or "cpu").strip().lower()
        self.threads = thread_count()
        self.question = load_question()
        self.router = None
        self.installed: List[str] = []
        self.load_ms: Optional[float] = None

    def load(self) -> Dict[str, Any]:
        if self.router is not None:
            return self.status()
        if not self.model_dir or not os.path.isdir(self.model_dir):
            raise FileNotFoundError(
                "Laya's weights are not installed. Expected the convaiinnovations/laya bundle at %r; "
                "run scripts/laya-setup.ps1 to provision it." % self.model_dir
            )

        self.installed = installed_checkpoints(self.model_dir)
        if "english" not in self.installed:
            raise FileNotFoundError(
                "%r holds no Laya checkpoint (rl_agent_config.json is missing at its root). "
                "Run scripts/laya-setup.ps1." % self.model_dir
            )

        started = time.perf_counter()
        import torch

        # One forward pass per call leaves inter-op parallelism nothing to
        # overlap; Laya's benchmarks measured torch's default as many times
        # slower for exactly this shape of call.
        torch.set_num_threads(self.threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # Only settable before the first parallel region.

        from laya import Router

        models = {name: (self.model_dir, CHECKPOINTS[name]) for name in self.installed}
        router = Router(models=models, device=self.device, max_loaded=len(models))
        router.preload(self.installed)

        qid = self.question["question_id"]
        spec = self.question["questions"][qid]
        for name in self.installed:
            agent = router.load(name)
            check_option_budget(
                name,
                lambda text, tok=agent.tok: tok(text, add_special_tokens=False)["input_ids"],
                spec["instructions"],
                spec["criteria"],
                int(agent.cfg.get("head_max_len", 192)),
            )

        # The first forward pass pays for allocator warm-up and lazy init. Paid
        # here, it never lands on a user's turn -- where the app's deadline would
        # turn it into a keyword fallback for no reason.
        for name in self.installed:
            router.predict({"request": "hello"}, self.question["questions"], model=name)

        self.router = router
        self.load_ms = round((time.perf_counter() - started) * 1000.0, 1)
        return self.status()

    def status(self) -> Dict[str, Any]:
        try:
            from laya import __version__ as laya_version
        except ImportError:
            laya_version = None
        return {
            "loaded": self.router is not None,
            "installed": self.installed,
            "modelDir": self.model_dir,
            "device": self.device,
            "threads": self.threads,
            "loadMs": self.load_ms,
            "layaVersion": laya_version,
            "questionVersion": self.question["version"],
            "questionFingerprint": self.question["fingerprint"],
        }

    def choose(self, text: str, instructions: str, labels: List[str]) -> Dict[str, Any]:
        """One ad-hoc `choice` over `labels`, for questions other than routing.

        Used to name an adapter's use case from its model card. Labels carry no
        descriptions, so even a few dozen fit Laya's 192-token head budget;
        `instructions` must name the state field as `request`, in backticks.
        """
        if self.router is None:
            raise RuntimeError("laya.choose before laya.load")
        labels = [str(label) for label in labels if str(label).strip()]
        if not labels:
            raise ValueError("laya.choose needs at least one label")
        if "`request`" not in instructions:
            raise ValueError("instructions must name the state field as `request`")

        state = {"request": text}
        questions = {
            "choice": {
                "type": "choice",
                "instructions": instructions,
                "criteria": {label: None for label in labels},
            }
        }
        used = self.router.route(state, questions)["model"]
        if used not in self.installed:
            used = "english"
        answer = self.router.predict(state, questions, model=used)["answers"]["choice"]
        return {
            "choice": answer["choice"],
            "probabilities": answer["probabilities"],
            "answerConfidence": answer.get("answer_confidence"),
            "checkpoint": used,
        }

    def classify(self, prompt: str) -> Dict[str, Any]:
        if self.router is None:
            raise RuntimeError("laya.classify before laya.load")

        qid = self.question["question_id"]
        state = {"request": prompt}
        questions = self.question["questions"]
        started = time.perf_counter()

        # Laya's own script and language detection picks the checkpoint. A
        # checkpoint it picks but that is not installed falls back to English
        # rather than being downloaded.
        decision = self.router.route(state, questions)
        used = decision["model"]
        if used not in self.installed:
            used = "english"
        result = self.router.predict(state, questions, model=used)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        answer = result["answers"][qid]
        return {
            "choice": answer["choice"],
            "probabilities": answer["probabilities"],
            # max(p) after temperature scaling -- the quantity Laya calibrates.
            # `confidence` on a choice answer is normalised entropy, which Laya
            # documents as not comparable against the same threshold.
            "answerConfidence": answer.get("answer_confidence"),
            "checkpoint": used,
            "checkpointReason": decision.get("reason"),
            "latencyMs": round(elapsed_ms, 2),
        }
