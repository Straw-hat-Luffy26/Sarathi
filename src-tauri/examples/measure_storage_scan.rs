//! Measures what the Storage screen asks the backend to do.
//!
//! `Reading what is on disk…` is the Storage screen waiting on three commands.
//! This times them against the real model store so the cost is a number rather
//! than an opinion, and breaks the scan into the stages that make it up.
//!
//! Run with:  cargo run --release --example measure_storage_scan

use std::path::{Path, PathBuf};
use std::time::Instant;

use sarathi_lib::adapter_manager::AdapterRegistry;
use sarathi_lib::model_manager::ModelManager;

fn app_data_dir() -> PathBuf {
    PathBuf::from(std::env::var("APPDATA").expect("APPDATA")).join("com.sarathi.app")
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// Times the pieces the listing is made of, per package.
fn stage_breakdown(app_data: &Path) {
    let models_dir = app_data.join("models");
    println!("\n--- per-package stages ---");
    println!(
        "{:<52} {:>10} {:>10} {:>10}",
        "package", "manifest", "gguf hdr", "total"
    );

    let Ok(providers) = std::fs::read_dir(&models_dir) else {
        println!("no models directory at {}", models_dir.display());
        return;
    };

    let mut manifest_total = 0.0;
    let mut gguf_total = 0.0;

    for provider in providers.flatten() {
        if !provider.path().is_dir() {
            continue;
        }
        let provider_id = provider.file_name().to_string_lossy().to_string();
        let Ok(packages) = std::fs::read_dir(provider.path()) else { continue };

        for pkg in packages.flatten() {
            let pkg_path = pkg.path();
            if !pkg_path.is_dir() {
                continue;
            }
            let folder = pkg_path.file_name().unwrap_or_default().to_string_lossy().to_string();
            let inferred = folder.replace('_', "/");

            let t = Instant::now();
            let manifest = AdapterRegistry::ensure_valid_manifest(&pkg_path, &provider_id, &inferred);
            let manifest_ms = ms(t);

            let mut gguf_ms = 0.0;
            if let Ok(m) = &manifest {
                let gguf = pkg_path.join(&m.base_model.file_path);
                if gguf.is_file() {
                    let t = Instant::now();
                    let _ = sarathi_lib::ai_engine::gguf_meta::read_gguf_metadata(&gguf);
                    gguf_ms = ms(t);
                }
            }

            manifest_total += manifest_ms;
            gguf_total += gguf_ms;
            println!(
                "{:<52} {:>9.1}ms {:>9.1}ms {:>9.1}ms",
                folder.chars().take(52).collect::<String>(),
                manifest_ms,
                gguf_ms,
                manifest_ms + gguf_ms
            );
        }
    }
    println!(
        "{:<52} {:>9.1}ms {:>9.1}ms {:>9.1}ms",
        "TOTAL",
        manifest_total,
        gguf_total,
        manifest_total + gguf_total
    );
}

fn main() {
    let app_data = app_data_dir();
    println!("store: {}", app_data.join("models").display());

    // Cold-ish first, then repeated: Windows caches directory metadata and file
    // pages aggressively, so one number would describe only one of the two
    // cases a user actually meets.
    println!("\n--- get_installed_models (5 runs) ---");
    for i in 1..=5 {
        let t = Instant::now();
        let models = ModelManager::list_installed_models(&app_data);
        println!("  run {i}: {:>8.1}ms  ({} models)", ms(t), models.len());
    }

    println!("\n--- get_storage_summary (5 runs) ---");
    for i in 1..=5 {
        let t = Instant::now();
        let s = ModelManager::get_storage_summary(&app_data);
        println!("  run {i}: {:>8.1}ms  ({} models)", ms(t), s.total_installed_models);
    }

    // What the screen actually costs: the Storage refresh issues both.
    println!("\n--- one Storage refresh (both commands, as the screen issues them) ---");
    for i in 1..=3 {
        let t = Instant::now();
        let _ = ModelManager::list_installed_models(&app_data);
        let _ = ModelManager::get_storage_summary(&app_data);
        println!("  refresh {i}: {:>8.1}ms", ms(t));
    }

    stage_breakdown(&app_data);
}
