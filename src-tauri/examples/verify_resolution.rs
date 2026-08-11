//! Resolves real HuggingFace repositories and reports which file was chosen.
//!
//! Read-only: it fetches public file listings and the first megabytes of GGUF
//! headers, and downloads nothing. The point is to prove, against the live Hub,
//! that a repository containing both a model and a helper module resolves to the
//! model — the failure that produced "GPU error: NullResult, CPU error:
//! NullResult" was a repository resolving to its EAGLE-3 draft.
//!
//! Run with:  cargo run --example verify_resolution

use sarathi_lib::model_providers::huggingface::{probe, resolver};

/// Repositories worth checking, and what each one is here to prove.
const CASES: &[(&str, &str, &str)] = &[
    // The failure itself. The repository holds the MXFP4 model and an EAGLE-3
    // draft; asking for BF16 used to match the draft.
    ("ggml-org/gpt-oss-20b-GGUF", "BF16", "MoE model beside a helper — must pick the model"),
    ("ggml-org/gpt-oss-20b-GGUF", "MXFP4", "the same repository asked for its real format"),
    // A dense model, to prove nothing that worked has been broken.
    ("Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", "Q4_K_M", "ordinary dense model"),
    ("bartowski/Llama-3.2-1B-Instruct-GGUF", "Q4_K_M", "small dense model"),
];

#[tokio::main]
async fn main() {
    let token = std::env::var("HF_TOKEN").ok();
    let token = token.as_deref().filter(|t| !t.trim().is_empty());
    println!("Token: {}\n", if token.is_some() { "present" } else { "anonymous" });

    for (repo, quant, why) in CASES {
        println!("{}", "=".repeat(88));
        println!("{repo}  (asked for {quant})");
        println!("  case: {why}");

        match resolver::resolve_artifact(repo, quant, token).await {
            Ok(a) => {
                println!("  -> file:         {}", a.file_name);
                println!("  -> quantization: {} (requested {quant})", a.quantization);
                println!("  -> size:         {:.2} GB", a.size_bytes as f64 / 1e9);
                println!(
                    "  -> architecture: {}",
                    a.architecture.as_deref().unwrap_or("<unverified: Hub unreachable>")
                );
                println!(
                    "  -> experts:      {}",
                    if a.expert_count > 0 {
                        format!("{} (MoE)", a.expert_count)
                    } else {
                        "0 (dense)".to_string()
                    }
                );

                let looks_auxiliary = a.file_name.to_lowercase().contains("eagle")
                    || a.file_name.to_lowercase().contains("mmproj")
                    || a.file_name.to_lowercase().contains("speculator");
                println!(
                    "  -> VERDICT:      {}",
                    if looks_auxiliary { "FAIL — resolved to a helper file" } else { "OK" }
                );
            }
            Err(e) => println!("  -> REFUSED: {e}"),
        }
        println!();
    }

    // Direct probes, so the classification is visible rather than inferred from
    // which file resolution happened to pick.
    println!("{}", "=".repeat(88));
    println!("Direct header probes\n");
    for (repo, file) in [
        ("ggml-org/gpt-oss-20b-GGUF", "gpt-oss-20b-MXFP4.gguf"),
        ("ggml-org/gpt-oss-20b-GGUF", "eagle3-gpt-oss-20b-BF16.gguf"),
    ] {
        match probe::probe_remote(repo, file, token).await {
            Ok(p) => println!(
                "  {file}\n    architecture '{}', role {:?}, standalone: {}",
                p.metadata.architecture,
                p.metadata.role,
                p.is_standalone_model()
            ),
            Err(e) => println!("  {file}\n    could not probe: {e:#}"),
        }
    }
}
