//! Measures the cost of partial offload: throughput against layers on the GPU.
//!
//! The planner's job is to pick a layer count. This measures what that choice is
//! worth, by holding a real model and a real prompt fixed and moving only the
//! split — so the claim "keep as many layers resident as fit, because spilling
//! is expensive" stops being an assumption and becomes a number.
//!
//! Every other harness here runs at the count the planner chose, which on a card
//! that fits the model is always "all of them". That leaves the partial path —
//! the one that matters for a model larger than the card — measured only by unit
//! tests of the arithmetic. This runs it.
//!
//! Run with:
//!   cargo run --release --features cuda --example sweep_offload_layers -- <path.gguf>
//!
//! Layer counts may be given after the path; they default to a spread from
//! CPU-only to fully resident:
//!   cargo run --release --features cuda --example sweep_offload_layers -- model.gguf 0 8 16 24 99

use std::time::Instant;

use sarathi_lib::ai_engine::gguf_meta::read_gguf_metadata;
use sarathi_lib::ai_engine::runtime::LlamaCppRuntime;
use sarathi_lib::ai_engine::traits::{ChatMessage, GenerationParams, ModelLoadConfig};

/// Long enough that per-token cost dominates the fixed setup, short enough to
/// sweep several configurations without the run taking all afternoon.
const MAX_TOKENS: u32 = 96;
const PROMPT: &str = "Explain in three sentences how virtual memory works.";

/// A working context rather than 512: the KV cache is part of what competes for
/// the card, so measuring with a token-sized cache would flatter every split.
const CONTEXT: u32 = 4096;

fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}

fn vram_used_bytes() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits", "--id=0"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let mib: u64 = text.trim().lines().next()?.trim().parse().ok()?;
    Some(mib * 1024 * 1024)
}

fn show(bytes: Option<u64>) -> String {
    bytes.map_or("n/a".to_string(), |b| format!("{:.2} GB", gb(b)))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: sweep_offload_layers <path.gguf> [layers...]");
        std::process::exit(2);
    };
    let path = std::path::PathBuf::from(path);

    let requested: Vec<u32> = args.filter_map(|a| a.parse().ok()).collect();

    let meta = read_gguf_metadata(&path).expect("readable GGUF header");
    let total_layers = meta.block_count;
    let file_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

    println!("Model     : {}", path.display());
    println!("Size      : {:.2} GB", gb(file_bytes));
    println!("Layers    : {total_layers}");
    println!("MoE       : {}", meta.is_moe());
    println!("Context   : {CONTEXT} tokens");
    println!(
        "Build     : cuda={} vulkan={}",
        cfg!(feature = "cuda"),
        cfg!(feature = "vulkan")
    );
    println!("Baseline VRAM: {}", show(vram_used_bytes()));

    let counts: Vec<u32> = if requested.is_empty() {
        let q = total_layers.max(4) / 4;
        vec![0, q, q * 2, q * 3, 999]
    } else {
        requested
    };

    println!(
        "\n{:>7}  {:>9}  {:>10}  {:>9}  {:>10}  {}",
        "layers", "load s", "tok/s", "tokens", "VRAM", "result"
    );
    println!("{}", "-".repeat(78));

    for gpu_layers in counts {
        let config = ModelLoadConfig {
            model_path: path.to_string_lossy().to_string(),
            model_id: "sweep".into(),
            model_name: "sweep".into(),
            quantization: String::new(),
            context_length: CONTEXT,
            gpu_layers,
            cpu_moe_layers: 0,
            threads: 8,
            // Empty so the runtime uses the template embedded in the GGUF; a
            // guessed family name has been wrong in practice.
            chat_template: String::new(),
            stop_tokens: Vec::new(),
        };

        let mut runtime = LlamaCppRuntime::new();

        let t0 = Instant::now();
        let loaded = match runtime.load_model(&config, |_| {}) {
            Ok(i) => i,
            Err(e) => {
                // An out-of-memory refusal is a result, not an error to hide:
                // it is the upper end of the sweep and the thing the planner
                // exists to stay below.
                println!(
                    "{gpu_layers:>7}  {:>9}  {:>10}  {:>9}  {:>10}  LOAD FAILED: {e:#}",
                    "-", "-", "-", "-"
                );
                continue;
            }
        };
        let load_secs = t0.elapsed().as_secs_f64();
        let vram = vram_used_bytes();

        let messages = vec![ChatMessage::new("user", PROMPT)];
        let params = GenerationParams { max_tokens: MAX_TOKENS, ..Default::default() };

        let mut tokens = 0u32;
        let gen_start = Instant::now();
        let result = runtime.generate(&messages, &params, |chunk| {
            if !chunk.text.is_empty() {
                tokens += 1;
            }
        });
        let gen_secs = gen_start.elapsed().as_secs_f64();

        match result {
            Ok(text) => {
                let tps = if gen_secs > 0.0 { tokens as f64 / gen_secs } else { 0.0 };
                let preview: String = text.trim().chars().take(48).collect();
                println!(
                    "{gpu_layers:>7}  {load_secs:>9.2}  {tps:>10.2}  {tokens:>9}  {:>10}  {} [{preview}...]",
                    show(vram),
                    loaded.backend_used
                );
            }
            Err(e) => println!(
                "{gpu_layers:>7}  {load_secs:>9.2}  {:>10}  {:>9}  {:>10}  GENERATION FAILED: {e:#}",
                "-",
                "-",
                show(vram)
            ),
        }

        let _ = runtime.unload_model();
    }

    println!("\nDone.");
}
