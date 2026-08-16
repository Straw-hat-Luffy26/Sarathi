//! Prints the sizes Discover would show for a repository, from the live Hub.
//!
//! `verify_resolution` proves the *downloader* picks the model rather than a
//! side-car. This proves the *listing* does — a different failure, and the one
//! that showed a 27.3B model as being available in 600 MB and 885 MB.
//!
//! Read-only: fetches the public file listing and downloads nothing.
//!
//! Run with:
//!   cargo run --release --example verify_catalog_sizes
//!   cargo run --release --example verify_catalog_sizes -- mradermacher/Qwen3.8-27B-GGUF

use sarathi_lib::model_providers::huggingface::live_catalog;

/// Repositories worth checking, and what each is here to prove.
const CASES: &[(&str, &str)] = &[
    ("mradermacher/Qwen3.8-27B-GGUF", "dot-named builds beside hyphen-named projectors"),
    ("mradermacher/Qwen3.8-27B-i1-GGUF", "the same publisher's imatrix repository"),
    ("Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", "hyphen-named — must be unchanged"),
    ("ggml-org/gpt-oss-20b-GGUF", "MoE beside an EAGLE-3 draft — must be unchanged"),
];

fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}

#[tokio::main]
async fn main() {
    let token = std::env::var("HF_TOKEN").ok();
    let token = token.as_deref().filter(|t| !t.trim().is_empty());

    let args: Vec<String> = std::env::args().skip(1).collect();
    let cases: Vec<(&str, &str)> = if args.is_empty() {
        CASES.to_vec()
    } else {
        args.iter().map(|a| (a.as_str(), "requested")).collect()
    };

    for (repo, why) in cases {
        println!("\n{}", "=".repeat(76));
        println!("{repo}  — {why}");
        println!("{}", "=".repeat(76));

        match live_catalog::fetch_repo(repo, token).await {
            Ok(r) => {
                let params = r.gguf.as_ref().map(|g| g.total_parameters).unwrap_or(0);
                println!("parameters : {:.1}B", params as f64 / 1e9);
                println!("sizes offered: {}", r.quantizations.len());

                for q in &r.quantizations {
                    // Bits per weight is the tell: no real quantization is under
                    // 1, so a row below that is a side-car wearing a quant name.
                    let bpw = if params > 0 {
                        format!("{:>5.2}", (q.size_bytes as f64 * 8.0) / params as f64)
                    } else {
                        "  n/a".to_string()
                    };
                    println!(
                        "  {:<10} {:>8.2} GB  {bpw} bits/weight  {}",
                        q.label, gb(q.size_bytes), q.filename
                    );
                }

                if r.quantizations.iter().any(|q| q.filename.contains("mmproj")) {
                    println!("  ** a projector is being offered as a size **");
                }
            }
            Err(e) => println!("FETCH FAILED: {e:#}"),
        }
    }
}
