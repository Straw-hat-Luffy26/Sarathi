//! Classifies every GGUF installed on this machine, using the same code the
//! loader and Storage use.
//!
//! Written to answer a specific question with evidence rather than reasoning:
//! which installed files are models, which are the side-cars that produced
//! "GPU error: NullResult, CPU error: NullResult", and which shelf each one
//! lands on in Storage. Synthetic headers prove the parser handles a shape;
//! only the real files prove it handles *these*.
//!
//! Run with:  cargo run --example classify_installed_models

use std::path::{Path, PathBuf};

use sarathi_lib::ai_engine::gguf_meta::read_gguf_metadata;
use sarathi_lib::model_manager::classify::classify;

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

fn main() {
    let root = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(std::env::var("APPDATA").expect("APPDATA"))
            .join("com.sarathi.app")
            .join("models")
    });

    println!("Scanning {}\n", root.display());
    let mut files = Vec::new();
    gguf_files(&root, &mut files);
    files.sort();

    let (mut loadable, mut inert) = (0, 0);

    for path in &files {
        let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
        let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

        match read_gguf_metadata(path) {
            Ok(meta) => {
                let c = classify(&meta, &name);
                if c.group.is_loadable() {
                    loadable += 1;
                } else {
                    inert += 1;
                }

                println!("{name}");
                println!(
                    "  shelf:      {}  ({})",
                    c.group.label(),
                    if c.group.is_loadable() { "loadable" } else { "NOT loadable" }
                );
                println!(
                    "  file:       {:.2} GB - arch '{}' - {} layers - quant {}",
                    size as f64 / 1e9,
                    c.architecture,
                    c.block_count,
                    c.quantization.as_deref().unwrap_or("unknown")
                );
                if c.is_moe {
                    println!(
                        "  experts:    {} total, {} consulted per token",
                        c.expert_count, c.expert_used_count
                    );
                }
                println!("  categories: {:?}", c.categories);
                if let Some(reason) = &c.not_loadable_reason {
                    println!("  reason:     {reason}");
                }
            }
            Err(e) => {
                inert += 1;
                println!("{name}\n  UNREADABLE: {e:#}");
            }
        }
        println!();
    }

    println!("{} file(s): {loadable} loadable, {inert} not", files.len());
}
