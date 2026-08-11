//! Loads every installed model through the *real* `InferenceManager` and reports
//! where its weights actually went.
//!
//! Deliberately different from `verify_pipeline`, which constructs a
//! `ModelLoadConfig` by hand and can therefore prove only that llama.cpp obeys a
//! layer count it is given. That leaves the interesting half untested: whether
//! hardware detection, GPU selection and the offload planner produce a sensible
//! count in the first place, and whether it survives the trip to the runtime.
//!
//! Nothing here sets a layer count. Everything comes from
//! `load_installed_model_direct`, which is the same call the gateway makes — so
//! whatever this reports is what Claude Code, opencode, Hermes and OpenClaw get.
//!
//! Run with:
//!   cargo run --features cuda --example verify_device_placement
//!   cargo run --example verify_device_placement          # CPU-only build

use sarathi_lib::ai_engine::manager::{select_inference_gpu, InferenceManager};
use sarathi_lib::ai_engine::traits::{ChatMessage, GenerationParams};

fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}

fn main() {
    println!(
        "Build: GPU backend compiled = {}  (cuda={}, vulkan={})",
        cfg!(any(feature = "cuda", feature = "vulkan")),
        cfg!(feature = "cuda"),
        cfg!(feature = "vulkan")
    );

    // ── What the hardware layer sees ────────────────────────────────────────
    let analyzer = sarathi_lib::system_analyzer::get_system_analyzer_manager();
    let profile = analyzer.get_profile();

    match &profile {
        Some(p) => {
            let gpus = p.gpus.current();
            println!("\nDetected {} GPU(s):", gpus.len());
            for g in gpus.iter() {
                println!(
                    "  {:<44} dedicated={:<5} vram total={:>6.2} GB free={:>6.2} GB  cuda={} vulkan={} rocm={}  [{}]",
                    g.model,
                    g.is_dedicated,
                    gb(g.vram_total_bytes),
                    gb(g.vram_free_bytes),
                    g.cuda_supported,
                    g.vulkan_supported,
                    g.rocm_supported,
                    g.detection_source
                );
            }

            match select_inference_gpu(gpus.as_slice()) {
                Some(g) => println!("\nSelected for inference: {} ({:.2} GB total)", g.model, gb(g.vram_total_bytes)),
                None => println!("\nSelected for inference: NONE — every model will load on CPU"),
            }

            let mem = p.memory.current();
            println!(
                "System RAM: {:.2} GB total, {:.2} GB available",
                gb(mem.total_bytes),
                gb(mem.available_bytes)
            );
        }
        None => println!("\nNo hardware profile — nothing can be planned against"),
    }

    // ── What each installed model actually gets ─────────────────────────────
    let data_dir = std::path::PathBuf::from(std::env::var("APPDATA").expect("APPDATA"))
        .join("com.sarathi.app");

    let models = sarathi_lib::model_manager::manager::ModelManager::list_installed_models(&data_dir);
    println!("\n{} installed model(s)\n", models.len());

    let manager = InferenceManager::new();

    for m in &models {
        let cls = m.classification.as_ref();
        let kind = cls
            .map(|c| {
                if c.is_moe {
                    format!("MoE ({} experts, {} active)", c.expert_count, c.expert_used_count)
                } else {
                    format!("{:?}", c.group)
                }
            })
            .unwrap_or_else(|| "unclassified".into());

        println!("{}", "-".repeat(84));
        println!("{}  [{}]  {:.2} GB  {}", m.model_name, m.quantization, gb(m.size_bytes), kind);

        if cls.is_some_and(|c| !c.group.is_loadable()) {
            println!("  skipped — not a standalone model");
            continue;
        }

        match manager.load_installed_model_direct(&data_dir, &m.provider_id, &m.model_id, &m.quantization) {
            Ok(info) => {
                println!("  backend      : {}", info.backend_used);
                println!("  gpu_layers   : {}", info.gpu_layers);
                println!("  cpu_moe_layers: {}", info.cpu_moe_layers);
                println!("  context      : {}", info.context_length);

                let messages = vec![ChatMessage {
                    role: "user".into(),
                    content: "Reply with exactly one short sentence: what is 2 + 2?".into(),
                    timestamp: None,
                }];
                let params = GenerationParams { max_tokens: 24, ..Default::default() };

                match manager.generate_direct(&messages, &params, |_| {}) {
                    Ok(text) => println!("  generated    : {:?}", text.trim()),
                    Err(e) => println!("  GENERATION FAILED: {e:#}"),
                }

                let _ = manager.unload_active_model_direct();
            }
            Err(e) => println!("  LOAD FAILED: {e:#}"),
        }
    }
}
