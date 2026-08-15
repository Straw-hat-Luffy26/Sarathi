//! End-to-end check: every installed GGUF is classified, loaded, and asked to
//! answer a question.
//!
//! Classification alone proves nothing about whether a model works. This runs
//! the real `LlamaCppRuntime` — the same code path the app uses — so a file that
//! the library calls loadable has to actually load *and* produce tokens, and a
//! file the library refuses has to be refused with a reason rather than a null.
//!
//! Built without GPU features this exercises the CPU path, which is the fallback
//! requirement. Build with `--features cuda` (or `vulkan`) to exercise GPU
//! offload and, on a MoE model too large for the card, partial expert offload.
//!
//! Run with:
//!   cargo run --release --example verify_pipeline
//!   cargo run --release --features cuda --example verify_pipeline

use std::path::{Path, PathBuf};

use sarathi_lib::ai_engine::gguf_meta::read_gguf_metadata;
use sarathi_lib::ai_engine::runtime::LlamaCppRuntime;
use sarathi_lib::ai_engine::traits::{ChatMessage, GenerationParams, ModelLoadConfig};

/// Short enough to be quick on CPU, long enough that an empty reply is a real
/// failure rather than a truncation.
const MAX_TOKENS: u32 = 24;
const PROMPT: &str = "Reply with exactly one short sentence: what is 2 + 2?";

fn gguf_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            gguf_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "gguf") {
            out.push(path);
        }
    }
}

#[derive(Debug)]
enum Outcome {
    /// Loaded and answered — the only result that counts as working.
    Generated { backend: String, reply: String },
    /// Refused before loading, with a reason. Correct for helper files.
    RefusedUpFront(String),
    /// Loaded but produced nothing, which must never be reported as ready.
    LoadedButSilent(String),
    Failed(String),
}

fn exercise(path: &Path, gpu_layers: u32) -> Outcome {
    let meta = match read_gguf_metadata(path) {
        Ok(m) => m,
        Err(e) => return Outcome::Failed(format!("header unreadable: {e:#}")),
    };

    if let Some(reason) = meta.role.refusal(&meta.architecture) {
        return Outcome::RefusedUpFront(reason);
    }

    let config = ModelLoadConfig {
        model_path: path.to_string_lossy().to_string(),
        model_id: path.file_stem().unwrap_or_default().to_string_lossy().to_string(),
        model_name: path.file_stem().unwrap_or_default().to_string_lossy().to_string(),
        quantization: String::new(),
        // Small on purpose: the KV cache at a full context would dominate the
        // run, and the question here is whether the path works at all.
        context_length: 512,
        gpu_layers,
        cpu_moe_layers: 0,
        threads: 4,
        // Left empty so the runtime uses the template embedded in the GGUF,
        // which is the one that matches the weights. A guessed family name has
        // been wrong in practice.
        chat_template: String::new(),
        stop_tokens: Vec::new(),
    };

    let mut runtime = LlamaCppRuntime::new();
    let loaded = match runtime.load_model(&config, |_| {}) {
        Ok(info) => info,
        Err(e) => return Outcome::Failed(format!("{e:#}")),
    };
    let backend = loaded.backend_used.clone();

    let messages = vec![ChatMessage::new("user", PROMPT)];
    let params = GenerationParams { max_tokens: MAX_TOKENS, ..Default::default() };

    let reply = match runtime.generate(&messages, &params, |_| {}) {
        Ok(text) => text,
        Err(e) => return Outcome::Failed(format!("loaded on {backend}, but generation failed: {e:#}")),
    };

    let _ = runtime.unload_model();

    if reply.trim().is_empty() {
        return Outcome::LoadedButSilent(backend);
    }
    Outcome::Generated { backend, reply: reply.trim().to_string() }
}

fn main() {
    let root = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(std::env::var("APPDATA").expect("APPDATA"))
            .join("com.sarathi.app")
            .join("models")
    });
    // 0 forces the CPU path; anything else asks for GPU offload, which a build
    // without a GPU feature silently ignores — so the label reflects the build.
    let gpu_layers: u32 = std::env::var("GPU_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(0);

    println!("Scanning {}", root.display());
    println!("Requesting gpu_layers={gpu_layers}\n");

    let mut files = Vec::new();
    gguf_files(&root, &mut files);
    files.sort();
    files.dedup();

    let (mut ok, mut refused, mut broken) = (0, 0, 0);

    for path in &files {
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        println!("{}", "-".repeat(88));
        println!("{name}");

        match exercise(path, gpu_layers) {
            Outcome::Generated { backend, reply } => {
                ok += 1;
                println!("  PASS      loaded via {backend}");
                println!("  reply:    {reply:?}");
            }
            Outcome::RefusedUpFront(reason) => {
                refused += 1;
                println!("  REFUSED   (correct for a helper or adapter file)");
                println!("  reason:   {reason}");
            }
            Outcome::LoadedButSilent(backend) => {
                broken += 1;
                println!("  FAIL      loaded via {backend} but generated nothing");
            }
            Outcome::Failed(e) => {
                broken += 1;
                println!("  FAIL      {e}");
            }
        }
    }

    println!("{}", "=".repeat(88));
    println!("{ok} generated, {refused} correctly refused, {broken} failed");
    if broken > 0 {
        std::process::exit(1);
    }
}
