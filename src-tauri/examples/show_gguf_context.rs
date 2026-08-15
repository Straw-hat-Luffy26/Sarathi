//! What context a GGUF was actually trained for.
//!
//! The ceiling that matters when Sarathi decides how much room to give an
//! agentic client: asking llama.cpp for more than this does not fail, it makes
//! the model extrapolate RoPE past anything the weights saw, and the answer
//! degrades into fluent nonsense with nothing in the log to say why.
//!
//! Run with:  cargo run --release --example show_gguf_context

use std::path::PathBuf;

use sarathi_lib::ai_engine::gguf_meta::read_gguf_metadata;

fn main() {
    let root = PathBuf::from(std::env::var("APPDATA").expect("APPDATA"))
        .join("com.sarathi.app")
        .join("models");

    let mut found = 0;
    let mut stack = vec![root];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf")) {
                found += 1;
                match read_gguf_metadata(&path) {
                    Ok(meta) => println!(
                        "{:>9} trained tokens  {}",
                        meta.context_length,
                        path.file_name().unwrap_or_default().to_string_lossy()
                    ),
                    Err(e) => println!(
                        "  UNREADABLE ({e})  {}",
                        path.file_name().unwrap_or_default().to_string_lossy()
                    ),
                }
            }
        }
    }
    println!("\n{found} GGUF file(s) examined");
}
