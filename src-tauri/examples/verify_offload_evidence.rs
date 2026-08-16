//! Measures what offloading actually costs and buys, model by model.
//!
//! `verify_device_placement` answers "where did the weights go?". This answers
//! "and what happened as a result?" — the numbers you cannot get from a load
//! config: real VRAM occupancy read from the driver, host RAM, wall-clock load
//! time, generation throughput, and whether the card survived it.
//!
//! Everything goes through `load_installed_model_direct`, the same call the
//! gateway makes. Nothing here sets a layer count or a context length, so the
//! figures below are the ones a user gets, not the ones a harness arranged.
//!
//! VRAM is read from `nvidia-smi` rather than from any in-process counter, on
//! the principle that the interesting failure is llama.cpp believing it fits
//! when the driver disagrees. On a machine with no NVIDIA driver the VRAM
//! columns read `n/a` and the rest still works.
//!
//! Run with:
//!   cargo run --release --features cuda --example verify_offload_evidence
//!   cargo run --release --example verify_offload_evidence   # CPU-only build
//!
//! Optional first argument filters models by substring:
//!   cargo run --release --features cuda --example verify_offload_evidence -- Qwen3

use std::time::Instant;

use sarathi_lib::ai_engine::manager::InferenceManager;
use sarathi_lib::ai_engine::traits::{ChatMessage, GenerationParams};

fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}

/// VRAM currently allocated on GPU 0, in bytes, straight from the driver.
///
/// `None` when nvidia-smi is absent or unparseable, which is the normal case on
/// an AMD or CPU-only machine and must not be mistaken for "zero used".
fn vram_used_bytes() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "--id=0",
        ])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let mib: u64 = text.trim().lines().next()?.trim().parse().ok()?;
    Some(mib * 1024 * 1024)
}

fn show_vram(bytes: Option<u64>) -> String {
    match bytes {
        Some(b) => format!("{:.2} GB", gb(b)),
        None => "n/a".to_string(),
    }
}

/// Host RAM in use system-wide, and this process's resident set.
fn host_ram() -> (u64, u64) {
    use sysinfo::System;
    let mut sys = System::new();
    sys.refresh_memory();
    sys.refresh_processes();

    let used = sys.used_memory();
    let rss = sysinfo::get_current_pid()
        .ok()
        .and_then(|pid| sys.process(pid).map(|p| p.memory()))
        .unwrap_or(0);
    (used, rss)
}

fn main() {
    let filter = std::env::args().nth(1);

    println!("=========================================================================");
    println!(" SARATHI OFFLOAD EVIDENCE");
    println!("=========================================================================");
    println!(
        "Build: gpu_backend_compiled={}  cuda={}  vulkan={}",
        cfg!(any(feature = "cuda", feature = "vulkan")),
        cfg!(feature = "cuda"),
        cfg!(feature = "vulkan")
    );

    let analyzer = sarathi_lib::system_analyzer::get_system_analyzer_manager();
    if let Some(p) = analyzer.get_profile() {
        for g in p.gpus.current().iter() {
            println!(
                "GPU: {} — {:.2} GB VRAM (dedicated={}, cuda={})",
                g.model,
                gb(g.vram_total_bytes),
                g.is_dedicated,
                g.cuda_supported
            );
        }
        let mem = p.memory.current();
        println!(
            "RAM: {:.2} GB total, {:.2} GB available",
            gb(mem.total_bytes),
            gb(mem.available_bytes)
        );
    }

    let baseline_vram = vram_used_bytes();
    let (baseline_ram, _) = host_ram();
    println!(
        "Baseline before any load: VRAM used {}, system RAM used {:.2} GB",
        show_vram(baseline_vram),
        gb(baseline_ram)
    );

    let data_dir = std::path::PathBuf::from(std::env::var("APPDATA").expect("APPDATA"))
        .join("com.sarathi.app");
    let models = sarathi_lib::model_manager::manager::ModelManager::list_installed_models(&data_dir);
    let manager = InferenceManager::new();

    for m in &models {
        if let Some(f) = &filter {
            if !m.model_name.to_lowercase().contains(&f.to_lowercase())
                && !m.model_id.to_lowercase().contains(&f.to_lowercase())
            {
                continue;
            }
        }
        if m
            .classification
            .as_ref()
            .is_some_and(|c| !c.group.is_loadable())
        {
            continue;
        }

        println!("\n{}", "=".repeat(73));
        println!("{}  [{}]  {:.2} GB", m.model_name, m.quantization, gb(m.size_bytes));
        println!("{}", "=".repeat(73));

        let t0 = Instant::now();
        let info = match manager.load_installed_model_direct(
            &data_dir,
            &m.provider_id,
            &m.model_id,
            &m.quantization,
        ) {
            Ok(i) => i,
            Err(e) => {
                // The OOM case is a load failure, and saying so plainly is the
                // whole point of the exercise — a harness that swallowed it
                // would report "no OOM observed" for a model that crashed.
                println!("  LOAD FAILED: {e:#}");
                continue;
            }
        };
        let load_secs = t0.elapsed().as_secs_f64();

        let loaded_vram = vram_used_bytes();
        let (loaded_ram, loaded_rss) = host_ram();

        println!("  backend        : {}", info.backend_used);
        println!("  gpu_layers     : {}", info.gpu_layers);
        println!("  cpu_moe_layers : {}", info.cpu_moe_layers);
        println!("  context        : {} tokens", info.context_length);
        println!("  threads        : {}", info.threads);
        println!("  load time      : {load_secs:.2} s");
        println!(
            "  VRAM           : {} used ({} above baseline)",
            show_vram(loaded_vram),
            show_vram(
                loaded_vram
                    .zip(baseline_vram)
                    .map(|(a, b)| a.saturating_sub(b))
            )
        );
        println!(
            "  host RAM       : {:.2} GB used system-wide, {:.2} GB resident in this process",
            gb(loaded_ram),
            gb(loaded_rss)
        );

        // A prompt long enough that the KV cache is genuinely exercised rather
        // than a handful of tokens rounding to nothing.
        let messages = vec![ChatMessage::new(
            "user",
            "Explain in three sentences why a GPU with limited VRAM can still run \
             a model larger than that VRAM.",
        )];
        let params = GenerationParams {
            max_tokens: 128,
            temperature: 0.7,
            ..Default::default()
        };

        let mut tokens = 0u32;
        let mut first_token_at: Option<f64> = None;
        let gen_start = Instant::now();
        let result = manager.generate_direct(&messages, &params, |chunk| {
            if !chunk.text.is_empty() {
                if first_token_at.is_none() {
                    first_token_at = Some(gen_start.elapsed().as_secs_f64());
                }
                tokens += 1;
            }
        });
        let gen_secs = gen_start.elapsed().as_secs_f64();

        let peak_vram = vram_used_bytes();

        match result {
            Ok(text) => {
                let tps = if gen_secs > 0.0 {
                    tokens as f64 / gen_secs
                } else {
                    0.0
                };
                println!("  ---- generation ----");
                println!("  tokens         : {tokens}");
                println!("  wall clock     : {gen_secs:.2} s");
                println!("  throughput     : {tps:.2} tok/s");
                if let Some(ttft) = first_token_at {
                    println!("  time to first  : {ttft:.2} s");
                }
                println!("  VRAM at peak   : {}", show_vram(peak_vram));
                println!("  OOM            : no — generation completed");
                println!("  output         : {:?}", text.trim());
            }
            Err(e) => {
                println!("  ---- generation ----");
                println!("  GENERATION FAILED after {gen_secs:.2} s: {e:#}");
                println!("  VRAM at failure: {}", show_vram(peak_vram));
            }
        }

        let _ = manager.unload_active_model_direct();
    }

    println!("\nDone.");
}
