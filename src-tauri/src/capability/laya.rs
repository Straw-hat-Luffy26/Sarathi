//! Laya as the decision layer of dynamic LoRA switching.
//!
//! [Laya](https://github.com/NandhaKishorM/laya) is a non-autoregressive
//! classifier: one forward pass answers a typed question over text with
//! *calibrated* probabilities. Sarathi asks it one question per turn — which
//! capability slot does this request need — and hands the verdict to the same
//! [`SwitchPolicy`](super::SwitchPolicy) hysteresis and
//! [`CapabilityResolver`](super::CapabilityResolver) that bind the LoRA
//! adapter. Laya does not load or merge adapters; llama.cpp's per-context
//! adapter binding still does that. What changes is the quality of the decision:
//! keyword weights could not tell "explain this stack trace" from "explain this
//! policy", and their "confidence" was a hand-tuned score rather than a
//! probability.
//!
//! ## Never in the way
//!
//! Laya is optional. Without Python, the `laya` package, or its weights, the
//! keyword [`IntentClassifier`](super::IntentClassifier) answers exactly as it
//! did before, and the reason is reported by [`LayaRouter::status`]. A turn
//! never waits longer than [`CLASSIFY_DEADLINE`] for Laya; a late answer is
//! discarded and that turn falls back to the keyword verdict.
//!
//! ## Not ARJUN's Laya
//!
//! ARJUN runs a Laya sidecar of its own on this machine. This one shares
//! nothing with it at runtime: it speaks JSON-RPC over stdio, so there is no
//! port to contend for; it reads `SARATHI_LAYA_*` variables, never `ARJUN_*`;
//! and its weights live under Sarathi's own app-data directory.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::Duration;

use serde::Serialize;
use serde_json::{json, Value};

use super::classifier::ClassificationResult;
use super::intent::PromptIntent;

/// Longest a chat turn waits for Laya before using the keyword verdict.
///
/// The English checkpoint answers in tens of milliseconds on a GPU and a few
/// hundred on a laptop CPU. This leaves room for a slow first call on a busy
/// machine without letting routing visibly delay the first token.
pub const CLASSIFY_DEADLINE: Duration = Duration::from_millis(1500);

/// Longest the startup load may take: reading ~1.7 GB of weights from disk and
/// one warm-up pass, on CPU.
const LOAD_DEADLINE: Duration = Duration::from_secs(240);

/// Prompt characters sent to Laya, split between the start and the end.
///
/// Laya reads at most a few hundred tokens of state, and tokenising a pasted
/// file only to discard most of it would spend the deadline on nothing. The
/// request is usually at one end of a long message — "fix this:" above a paste
/// or "what does this do?" below it — so both ends are kept.
const PROMPT_HEAD_CHARS: usize = 2000;
const PROMPT_TAIL_CHARS: usize = 2000;

/// Folder under Sarathi's app-data directory holding the
/// `convaiinnovations/laya` bundle. `scripts/laya-setup.ps1` provisions it.
pub const WEIGHTS_DIR: &str = "laya/convaiinnovations_laya";

/// What the router can do right now, for diagnostics and the settings screen.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "camelCase")]
pub enum LayaStatus {
    /// Turned off in settings.
    Disabled,
    /// Weights or the sidecar script are not installed. Nothing was started.
    NotInstalled { reason: String },
    /// The sidecar is loading its checkpoints.
    Starting,
    /// Answering turns.
    #[serde(rename_all = "camelCase")]
    Ready {
        device: String,
        checkpoints: Vec<String>,
        load_ms: Option<f64>,
        laya_version: Option<String>,
    },
    /// Started but failed, or exited. The keyword classifier is in use.
    Unavailable { reason: String },
}

type Pending = Arc<Mutex<HashMap<u64, mpsc::Sender<Result<Value, String>>>>>;

/// The running sidecar.
struct Link {
    child: Child,
    stdin: ChildStdin,
    pending: Pending,
    alive: Arc<AtomicBool>,
}

pub struct LayaRouter {
    status: RwLock<LayaStatus>,
    link: Mutex<Option<Link>>,
    next_id: AtomicU64,
}

static GLOBAL: OnceLock<Arc<LayaRouter>> = OnceLock::new();

/// The process-wide router, once [`start_global`] has run.
pub fn global() -> Option<&'static Arc<LayaRouter>> {
    GLOBAL.get()
}

/// Starts the router for this process, in the background.
///
/// Returns immediately: loading takes seconds and the app must not wait for
/// it. Until it is ready, every turn uses the keyword classifier.
pub fn start_global(app_data_dir: &Path, enabled: bool) {
    let router = GLOBAL.get_or_init(|| Arc::new(LayaRouter::new()));
    if !enabled {
        router.set_status(LayaStatus::Disabled);
        log::info!("[LAYA] Routing with Laya is turned off in settings; using keyword routing");
        return;
    }

    let weights = std::env::var_os("SARATHI_LAYA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| app_data_dir.join(WEIGHTS_DIR));
    let script = match locate_sidecar(app_data_dir) {
        Some(p) => p,
        None => {
            router.set_status(LayaStatus::NotInstalled {
                reason: "sidecars/laya_router/main.py was not found".into(),
            });
            return;
        }
    };
    if !weights.join("rl_agent_config.json").is_file() {
        let reason = format!(
            "Laya's weights are not at {}. Run scripts/laya-setup.ps1 to install them.",
            weights.display()
        );
        log::info!("[LAYA] {reason} Keyword routing stays in use.");
        router.set_status(LayaStatus::NotInstalled { reason });
        return;
    }

    router.set_status(LayaStatus::Starting);
    let router = router.clone();
    std::thread::Builder::new()
        .name("sarathi-laya-start".into())
        .spawn(move || router.boot(&script, &weights))
        .map(|_| ())
        .unwrap_or_else(|e| log::warn!("[LAYA] Could not start the loader thread: {e}"));
}

fn locate_sidecar(app_data_dir: &Path) -> Option<PathBuf> {
    let rel = Path::new("sidecars").join("laya_router").join("main.py");
    let cwd = std::env::current_dir().unwrap_or_default();
    let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf));
    [
        Some(app_data_dir.join(&rel)),
        Some(cwd.join(&rel)),
        cwd.parent().map(|p| p.join(&rel)),
        exe_dir.map(|d| d.join(&rel)),
    ]
    .into_iter()
    .flatten()
    .find(|p| p.is_file())
}

impl LayaRouter {
    fn new() -> Self {
        Self {
            status: RwLock::new(LayaStatus::Starting),
            link: Mutex::new(None),
            next_id: AtomicU64::new(1),
        }
    }

    pub fn status(&self) -> LayaStatus {
        self.status
            .read()
            .map(|s| s.clone())
            .unwrap_or(LayaStatus::Unavailable { reason: "status lock poisoned".into() })
    }

    fn set_status(&self, status: LayaStatus) {
        if let Ok(mut s) = self.status.write() {
            *s = status;
        }
    }

    fn boot(&self, script: &Path, weights: &Path) {
        if let Err(reason) = self.spawn(script, weights) {
            log::warn!("[LAYA] {reason}; keyword routing stays in use");
            self.set_status(LayaStatus::Unavailable { reason });
            return;
        }
        match self.call("laya.load", json!({}), LOAD_DEADLINE) {
            Ok(status) => {
                let ready = LayaStatus::Ready {
                    device: status["device"].as_str().unwrap_or("cpu").to_string(),
                    checkpoints: status["installed"]
                        .as_array()
                        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                        .unwrap_or_default(),
                    load_ms: status["loadMs"].as_f64(),
                    laya_version: status["layaVersion"].as_str().map(String::from),
                };
                log::info!("[LAYA] Ready: {ready:?}");
                self.set_status(ready);
            }
            Err(reason) => {
                let log_path = weights.parent().unwrap_or(weights).join("sidecar.log");
                let reason = format!("{reason} (details in {})", log_path.display());
                log::warn!("[LAYA] Load failed: {reason}; keyword routing stays in use");
                self.set_status(LayaStatus::Unavailable { reason });
                self.shutdown();
            }
        }
    }

    fn spawn(&self, script: &Path, weights: &Path) -> Result<(), String> {
        log::info!("[LAYA] Starting {}", script.display());

        // The GUI build has no console, so the sidecar's stderr — where Python
        // reports a missing package or a bad checkpoint — goes to a file beside
        // the weights. Without it, every such failure read as "the sidecar
        // exited" with nothing to say why.
        let log_path = weights.parent().unwrap_or(weights).join("sidecar.log");
        let stderr = std::fs::File::create(&log_path)
            .map(Stdio::from)
            .unwrap_or_else(|_| Stdio::null());

        let mut command = crate::system_analyzer::process_utils::create_hidden_command("python");
        command
            .arg(script)
            .env("SARATHI_LAYA_DIR", weights)
            .env("PYTHONIOENCODING", "utf-8")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(stderr);
        if let Some(dir) = script.parent() {
            command.current_dir(dir);
        }

        let mut child = command
            .spawn()
            .map_err(|e| format!("could not start Python for Laya ({e})"))?;
        let stdin = child.stdin.take().ok_or("no stdin on the Laya sidecar")?;
        let stdout = child.stdout.take().ok_or("no stdout on the Laya sidecar")?;

        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
        let alive = Arc::new(AtomicBool::new(true));
        {
            let pending = pending.clone();
            let alive = alive.clone();
            std::thread::Builder::new()
                .name("sarathi-laya-reader".into())
                .spawn(move || read_responses(BufReader::new(stdout), pending, alive))
                .map_err(|e| format!("could not start the Laya reader thread ({e})"))?;
        }

        if let Ok(mut link) = self.link.lock() {
            *link = Some(Link { child, stdin, pending, alive });
        }
        Ok(())
    }

    /// Sends one request and waits up to `deadline` for its answer.
    fn call(&self, method: &str, params: Value, deadline: Duration) -> Result<Value, String> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = mpsc::channel();
        {
            let mut guard = self.link.lock().map_err(|_| "Laya link lock poisoned")?;
            let link = guard.as_mut().ok_or("the Laya sidecar is not running")?;
            if !link.alive.load(Ordering::Relaxed) {
                return Err("the Laya sidecar has exited".into());
            }
            link.pending
                .lock()
                .map_err(|_| "Laya pending lock poisoned")?
                .insert(id, tx);
            let mut line = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})
                .to_string();
            line.push('\n');
            if let Err(e) = link.stdin.write_all(line.as_bytes()).and_then(|_| link.stdin.flush()) {
                link.pending.lock().ok().map(|mut p| p.remove(&id));
                return Err(format!("could not write to the Laya sidecar ({e})"));
            }
        }

        match rx.recv_timeout(deadline) {
            Ok(result) => result,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // Forget the request; the reader drops its answer when it lands.
                if let Ok(guard) = self.link.lock() {
                    if let Some(link) = guard.as_ref() {
                        link.pending.lock().ok().map(|mut p| p.remove(&id));
                    }
                }
                Err(format!("no answer within {} ms", deadline.as_millis()))
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => Err("the Laya sidecar has exited".into()),
        }
    }

    /// Classifies one prompt, or `None` when the keyword classifier should.
    pub fn classify(&self, prompt: &str) -> Option<ClassificationResult> {
        if !matches!(self.status(), LayaStatus::Ready { .. }) {
            return None;
        }
        let trimmed = clip_prompt(prompt);
        if trimmed.trim().is_empty() {
            return None;
        }
        match self.call("laya.classify", json!({ "prompt": trimmed }), CLASSIFY_DEADLINE) {
            Ok(answer) => {
                let parsed = to_classification(&answer);
                if parsed.is_none() {
                    log::warn!("[LAYA] Unrecognised answer, using keyword routing: {answer}");
                }
                parsed
            }
            Err(reason) => {
                log::warn!("[LAYA] Classification skipped ({reason}); using keyword routing");
                if self.link_dead() {
                    self.set_status(LayaStatus::Unavailable { reason });
                }
                None
            }
        }
    }

    /// Picks one of `labels` for `text`: a general Laya `choice`, used to name
    /// an adapter's use case when nothing in its metadata does.
    ///
    /// `instructions` must name the state field as `` `request` ``. Returns the
    /// label and Laya's calibrated probability for it, or `None` when Laya is
    /// not ready, did not answer in time, or answered with something unlisted.
    pub fn choose(&self, text: &str, instructions: &str, labels: &[&str]) -> Option<(String, f32)> {
        /// Not on a chat turn, so it can wait longer than a routing decision.
        const CHOOSE_DEADLINE: Duration = Duration::from_secs(5);

        if labels.is_empty() || !matches!(self.status(), LayaStatus::Ready { .. }) {
            return None;
        }
        let params = json!({
            "text": clip_prompt(text),
            "instructions": instructions,
            "labels": labels,
        });
        let answer = match self.call("laya.choose", params, CHOOSE_DEADLINE) {
            Ok(a) => a,
            Err(reason) => {
                log::debug!("[LAYA] choose skipped: {reason}");
                return None;
            }
        };
        let choice = answer.get("choice")?.as_str()?;
        if !labels.contains(&choice) {
            return None;
        }
        let confidence = answer
            .get("answerConfidence")
            .and_then(Value::as_f64)
            .or_else(|| answer.get("probabilities")?.get(choice)?.as_f64())?;
        Some((choice.to_string(), confidence.clamp(0.0, 1.0) as f32))
    }

    fn link_dead(&self) -> bool {
        self.link
            .lock()
            .ok()
            .and_then(|g| g.as_ref().map(|l| !l.alive.load(Ordering::Relaxed)))
            .unwrap_or(true)
    }

    fn shutdown(&self) {
        if let Ok(mut guard) = self.link.lock() {
            if let Some(mut link) = guard.take() {
                let _ = link.child.kill();
                let _ = link.child.wait();
            }
        }
    }
}

impl Drop for LayaRouter {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Routes each response line to the request waiting for it.
fn read_responses<R: BufRead>(reader: R, pending: Pending, alive: Arc<AtomicBool>) {
    for line in reader.lines() {
        let Ok(line) = line else { break };
        let Ok(frame) = serde_json::from_str::<Value>(&line) else {
            log::debug!("[LAYA] Ignoring a non-JSON line from the sidecar");
            continue;
        };
        let Some(id) = frame.get("id").and_then(Value::as_u64) else { continue };
        let result = match frame.get("error") {
            Some(err) => Err(err["message"].as_str().unwrap_or("Laya error").to_string()),
            None => Ok(frame.get("result").cloned().unwrap_or(Value::Null)),
        };
        if let Some(tx) = pending.lock().ok().and_then(|mut p| p.remove(&id)) {
            let _ = tx.send(result);
        }
    }
    alive.store(false, Ordering::Relaxed);
    // Dropping the senders wakes every waiter with `Disconnected`.
    if let Ok(mut p) = pending.lock() {
        p.clear();
    }
    log::warn!("[LAYA] The sidecar exited");
}

/// Keeps both ends of a long prompt, cut on character boundaries.
fn clip_prompt(prompt: &str) -> String {
    let count = prompt.chars().count();
    if count <= PROMPT_HEAD_CHARS + PROMPT_TAIL_CHARS {
        return prompt.to_string();
    }
    let head: String = prompt.chars().take(PROMPT_HEAD_CHARS).collect();
    let tail: String = prompt.chars().skip(count - PROMPT_TAIL_CHARS).collect();
    format!("{head}\n…\n{tail}")
}

fn intent_for(label: &str) -> Option<PromptIntent> {
    Some(match label {
        "coding" => PromptIntent::Coding,
        "mathematics" => PromptIntent::Mathematics,
        "reasoning" => PromptIntent::Reasoning,
        "tool-calling" => PromptIntent::ToolCalling,
        "research" => PromptIntent::Research,
        "general" => PromptIntent::GeneralChat,
        _ => return None,
    })
}

/// Turns a `laya.classify` answer into the shape the switch policy reads.
///
/// `confidence` is Laya's `answerConfidence` — the probability on the chosen
/// slot after temperature scaling, which is what Laya calibrates. Its entropy
/// `confidence` is a different quantity and is deliberately not used.
fn to_classification(answer: &Value) -> Option<ClassificationResult> {
    let choice = answer.get("choice")?.as_str()?;
    let intent = intent_for(choice)?;
    let probabilities = answer.get("probabilities")?.as_object()?;
    let p_choice = probabilities.get(choice).and_then(Value::as_f64);
    let confidence = answer
        .get("answerConfidence")
        .and_then(Value::as_f64)
        .or(p_choice)?
        .clamp(0.0, 1.0) as f32;

    let runner_up = probabilities
        .iter()
        .filter(|(label, _)| label.as_str() != choice)
        .filter_map(|(label, p)| Some((intent_for(label)?, p.as_f64()? as f32)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    Some(ClassificationResult {
        intent,
        confidence,
        raw_score: p_choice.unwrap_or(confidence as f64) as f32,
        runner_up,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn answer(choice: &str, conf: f64) -> Value {
        json!({
            "choice": choice,
            "probabilities": {"coding": 0.81, "reasoning": 0.12, "general": 0.07},
            "answerConfidence": conf,
            "checkpoint": "english",
        })
    }

    #[test]
    fn a_laya_answer_becomes_a_classification() {
        let c = to_classification(&answer("coding", 0.81)).expect("parses");
        assert_eq!(c.intent, PromptIntent::Coding);
        assert!((c.confidence - 0.81).abs() < 1e-6);
        assert_eq!(c.runner_up.as_ref().map(|r| r.0.clone()), Some(PromptIntent::Reasoning));
        assert_eq!(c.capability_name(), "coding", "the label must select the same manifest slot");
    }

    #[test]
    fn every_capability_label_maps_to_its_manifest_key() {
        for key in ["coding", "mathematics", "reasoning", "tool-calling", "research", "general"] {
            assert_eq!(intent_for(key).map(|i| i.to_capability_name()), Some(key));
        }
        assert!(intent_for("typed-decisions").is_none());
    }

    #[test]
    fn answer_confidence_falls_back_to_the_choices_probability() {
        let mut a = answer("coding", 0.0);
        a.as_object_mut().unwrap().remove("answerConfidence");
        let c = to_classification(&a).unwrap();
        assert!((c.confidence - 0.81).abs() < 1e-6);
    }

    #[test]
    fn an_unknown_label_is_not_guessed_at() {
        assert!(to_classification(&answer("astrology", 0.9)).is_none());
        assert!(to_classification(&json!({"probabilities": {}})).is_none());
    }

    #[test]
    fn long_prompts_keep_both_ends() {
        let long = format!("fix this:{}what is wrong?", "x".repeat(10_000));
        let clipped = clip_prompt(&long);
        assert!(clipped.starts_with("fix this:"));
        assert!(clipped.ends_with("what is wrong?"));
        assert!(clipped.chars().count() < 4_100);
        assert_eq!(clip_prompt("short"), "short");
        // Multi-byte text is cut on character boundaries, not bytes.
        let hindi = "क".repeat(5_000);
        assert!(clip_prompt(&hindi).chars().count() < 4_100);
    }

    #[test]
    fn responses_reach_the_request_that_asked_and_late_ones_are_dropped() {
        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
        let (tx, rx) = mpsc::channel();
        pending.lock().unwrap().insert(2, tx);
        let alive = Arc::new(AtomicBool::new(true));

        // id 1 was abandoned after its deadline; its answer must go nowhere.
        let input = "{\"id\":1,\"result\":{\"choice\":\"coding\"}}\nnot json\n{\"id\":2,\"result\":{\"ok\":true}}\n";
        read_responses(std::io::Cursor::new(input), pending.clone(), alive.clone());

        assert_eq!(rx.recv().unwrap().unwrap(), json!({"ok": true}));
        assert!(!alive.load(Ordering::Relaxed), "end of stream means the sidecar exited");
        assert!(pending.lock().unwrap().is_empty());
    }

    #[test]
    fn an_error_frame_is_an_error() {
        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
        let (tx, rx) = mpsc::channel();
        pending.lock().unwrap().insert(5, tx);
        let input = "{\"id\":5,\"error\":{\"code\":-32603,\"message\":\"weights missing\"}}\n";
        read_responses(std::io::Cursor::new(input), pending, Arc::new(AtomicBool::new(true)));
        assert_eq!(rx.recv().unwrap(), Err("weights missing".to_string()));
    }

    #[test]
    fn a_router_that_is_not_ready_defers_to_the_keyword_classifier() {
        let r = LayaRouter::new();
        r.set_status(LayaStatus::NotInstalled { reason: "no weights".into() });
        assert!(r.classify("refactor this rust function").is_none());
        r.set_status(LayaStatus::Disabled);
        assert!(r.classify("refactor this rust function").is_none());
    }

    /// The whole path through a real child process: spawn, load, answer, a
    /// turn that overruns its deadline, and recovery afterwards. The sidecar is
    /// a stand-in speaking the same protocol, so Laya need not be installed.
    #[test]
    fn a_real_sidecar_process_routes_turns_and_survives_a_slow_one() {
        let python_ok = std::process::Command::new("python")
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success());
        if !python_ok {
            return; // No interpreter on this machine; the plumbing cannot be exercised.
        }

        let dir = std::env::temp_dir().join(format!("sarathi-laya-fake-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("main.py");
        std::fs::write(
            &script,
            r#"
import json, sys, time
for line in sys.stdin:
    req = json.loads(line)
    if req["method"] == "laya.load":
        res = {"device": "cpu", "installed": ["english"], "loadMs": 1.0, "layaVersion": "0.3.20"}
    else:
        prompt = req["params"]["prompt"]
        if "slow" in prompt:
            time.sleep(2.5)
        coding = "rust" in prompt
        res = {
            "choice": "coding" if coding else "general",
            "probabilities": {"coding": 0.9, "general": 0.1} if coding else {"coding": 0.2, "general": 0.8},
            "answerConfidence": 0.9 if coding else 0.8,
        }
    print(json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": res}), flush=True)
"#,
        )
        .unwrap();

        let router = LayaRouter::new();
        // Nested so the sidecar log written beside the weights stays in `dir`.
        router.boot(&script, &dir.join("weights"));
        assert!(
            matches!(router.status(), LayaStatus::Ready { ref laya_version, .. } if laya_version.as_deref() == Some("0.3.20")),
            "got {:?}",
            router.status()
        );

        let c = router.classify("refactor this rust function").expect("answered");
        assert_eq!(c.intent, PromptIntent::Coding);
        assert!((c.confidence - 0.9).abs() < 1e-6);

        let started = std::time::Instant::now();
        assert!(router.classify("a slow one").is_none(), "a late answer must fall back");
        assert!(started.elapsed() < CLASSIFY_DEADLINE + Duration::from_millis(500));
        assert!(matches!(router.status(), LayaStatus::Ready { .. }), "a slow turn is not a dead sidecar");

        // Once the sidecar has worked through the slow request, routing resumes
        // and the abandoned answer is not mistaken for this one.
        std::thread::sleep(Duration::from_millis(1500));
        let next = router.classify("hello there").expect("answered after recovering");
        assert_eq!(next.intent, PromptIntent::GeneralChat);

        router.shutdown();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn status_serialises_for_the_ui() {
        let s = serde_json::to_value(LayaStatus::Ready {
            device: "cpu".into(),
            checkpoints: vec!["english".into()],
            load_ms: Some(4200.0),
            laya_version: Some("0.3.20".into()),
        })
        .unwrap();
        assert_eq!(s["state"], "ready");
        assert_eq!(s["loadMs"], 4200.0);
        assert_eq!(serde_json::to_value(LayaStatus::Disabled).unwrap()["state"], "disabled");
    }
}
