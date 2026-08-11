//! HuggingFace GGUF Artifact Resolver
//!
//! Resolves exact GGUF file artifacts, download URLs, and sizes from HuggingFace Hub.

use serde::{Deserialize, Serialize};
use anyhow::{Result, anyhow};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedArtifact {
    pub repo_id: String,
    pub file_name: String,
    pub download_url: String,
    pub size_bytes: u64,
    /// The quantization actually chosen.
    ///
    /// May differ from what was asked for. A repository's files are matched on
    /// their names, and when the named match turns out to be a helper module the
    /// real model is used instead — its quantization is the truthful one to
    /// record, so Storage does not describe the download as something it is not.
    pub quantization: String,
    pub sha256: Option<String>,
    /// Architecture read from the file's own header, when it was checked before
    /// downloading. `None` only when the Hub could not be reached for the check.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
    /// Routed experts, from the header. Zero for a dense model.
    #[serde(default)]
    pub expert_count: u32,
}

#[derive(Debug, Clone, Deserialize)]
struct HfTreeItem {
    path: String,
    size: Option<u64>,
    lfs: Option<HfLfsInfo>,
}

#[derive(Debug, Clone, Deserialize)]
struct HfLfsInfo {
    oid: String,
    size: u64,
}

/// Orders the files to probe, most likely first.
///
/// Three rules, in order:
///
/// 1. Files matching the requested quantization come before the rest.
/// 2. A consolidated file comes before a split one — the shards of a split model
///    all carry the same header, but the first shard is the one to download.
/// 3. Larger before smaller. A helper module is small relative to the model it
///    serves, so within a tie this puts the model first and usually settles the
///    choice in a single request.
///
/// Ordering only decides what is checked first. Nothing here decides what is
/// *chosen* — the header does.
fn rank_candidates(matches: &[HfTreeItem], all_gguf: &[HfTreeItem]) -> Vec<HfTreeItem> {
    let size_of = |i: &HfTreeItem| i.lfs.as_ref().map(|l| l.size).or(i.size).unwrap_or(0);

    let mut ordered: Vec<HfTreeItem> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for group in [matches, all_gguf] {
        let mut tier: Vec<HfTreeItem> =
            group.iter().filter(|i| !seen.contains(&i.path)).cloned().collect();

        // A later shard is never the file to download: it has no header of its
        // own to check and the first shard stands for the whole model.
        tier.retain(|i| !i.path.contains("-of-") || i.path.contains("-00001-of-"));

        tier.sort_by(|a, b| {
            let split = a.path.contains("-of-").cmp(&b.path.contains("-of-"));
            split.then_with(|| size_of(b).cmp(&size_of(a)))
        });

        for item in tier {
            seen.insert(item.path.clone());
            ordered.push(item);
        }
    }

    ordered
}

#[cfg(test)]
mod ranking_tests {
    use super::*;

    fn item(path: &str, size: u64) -> HfTreeItem {
        HfTreeItem { path: path.into(), size: Some(size), lfs: None }
    }

    fn paths(items: &[HfTreeItem]) -> Vec<&str> {
        items.iter().map(|i| i.path.as_str()).collect()
    }

    /// The repository behind the reported failure. Asking for BF16 matches only
    /// the EAGLE-3 drafts by name, so the model has to be reachable through the
    /// second tier or the probe never gets the chance to choose it.
    #[test]
    fn the_real_model_is_always_reachable_even_when_only_helpers_match_the_name() {
        let matches = vec![item("eagle3-gpt-oss-20b-BF16.gguf", 1_722_588_800)];
        let all = vec![
            item("eagle3-gpt-oss-20b-BF16.gguf", 1_722_588_800),
            item("eagle3-gpt-oss-20b-Q8_0.gguf", 921_488_000),
            item("gpt-oss-20b-MXFP4.gguf", 12_109_566_624),
        ];

        let candidates = rank_candidates(&matches, &all);
        let ranked = paths(&candidates);

        assert_eq!(ranked[0], "eagle3-gpt-oss-20b-BF16.gguf", "the named match is tried first");
        assert!(
            ranked.contains(&"gpt-oss-20b-MXFP4.gguf"),
            "the model must be a candidate or it can never be chosen: {ranked:?}"
        );
    }

    /// Ordering must respect what was asked for. Sorting the whole repository by
    /// size would answer a request for Q4_K_M with whatever build is biggest.
    #[test]
    fn the_requested_quantization_is_tried_before_anything_else() {
        let matches = vec![item("model-Q4_K_M.gguf", 4_700_000_000)];
        let all = vec![
            item("model-Q4_K_M.gguf", 4_700_000_000),
            item("model-Q8_0.gguf", 8_100_000_000),
        ];

        let candidates = rank_candidates(&matches, &all);
        assert_eq!(paths(&candidates)[0], "model-Q4_K_M.gguf");
    }

    /// A later shard has no header of its own to check and is never the file to
    /// download — the first shard stands for the whole model.
    #[test]
    fn later_shards_are_never_candidates() {
        let all = vec![
            item("big-Q4_K_M-00001-of-00003.gguf", 4_000_000_000),
            item("big-Q4_K_M-00002-of-00003.gguf", 4_000_000_000),
            item("big-Q4_K_M-00003-of-00003.gguf", 2_000_000_000),
        ];

        let candidates = rank_candidates(&[], &all);
        let ranked = paths(&candidates);

        assert_eq!(ranked, vec!["big-Q4_K_M-00001-of-00003.gguf"]);
    }

    #[test]
    fn a_candidate_is_never_offered_twice() {
        let f = item("model-Q4_K_M.gguf", 4_000_000_000);
        let ranked = rank_candidates(std::slice::from_ref(&f), std::slice::from_ref(&f));
        assert_eq!(ranked.len(), 1);
    }

    #[test]
    fn an_empty_repository_ranks_to_nothing() {
        assert!(rank_candidates(&[], &[]).is_empty());
    }
}

/// Maps base model IDs from catalog to canonical Hugging Face GGUF repositories
pub fn resolve_gguf_repo(model_id: &str) -> String {
    if model_id.ends_with("-GGUF") || model_id.contains("/GGUF-") {
        return model_id.to_string();
    }

    match model_id {
        "meta-llama/Llama-3.2-1B" => "bartowski/Llama-3.2-1B-Instruct-GGUF".to_string(),
        "meta-llama/Llama-3.2-3B" => "bartowski/Llama-3.2-3B-Instruct-GGUF".to_string(),
        "meta-llama/Llama-3.1-8B" => "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF".to_string(),
        "Qwen/Qwen2.5-3B" => "Qwen/Qwen2.5-3B-Instruct-GGUF".to_string(),
        "Qwen/Qwen2.5-7B" => "Qwen/Qwen2.5-7B-Instruct-GGUF".to_string(),
        "Qwen/Qwen2.5-14B" => "Qwen/Qwen2.5-14B-Instruct-GGUF".to_string(),
        "Qwen/Qwen2.5-32B" => "Qwen/Qwen2.5-32B-Instruct-GGUF".to_string(),
        "Qwen/Qwen2.5-Coder-7B" => "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF".to_string(),
        "mistralai/Mistral-7B-v0.3" => "bartowski/Mistral-7B-Instruct-v0.3-GGUF".to_string(),
        "mistralai/Mixtral-8x7B-v0.1" => "bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF".to_string(),
        "microsoft/phi-4" => "bartowski/phi-4-GGUF".to_string(),
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" => "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF".to_string(),
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" => "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF".to_string(),
        "google/gemma-2-2b" => "bartowski/gemma-2-2b-it-GGUF".to_string(),
        "google/gemma-2-9b" => "bartowski/gemma-2-9b-it-GGUF".to_string(),
        "google/gemma-2-27b" => "bartowski/gemma-2-27b-it-GGUF".to_string(),
        "codellama/CodeLlama-7b-hf" => "TheBloke/CodeLlama-7B-GGUF".to_string(),
        "codellama/CodeLlama-13b-hf" => "TheBloke/CodeLlama-13B-GGUF".to_string(),
        other => {
            if other.contains('/') {
                format!("{}-GGUF", other)
            } else {
                format!("bartowski/{}-GGUF", other)
            }
        }
    }
}

/// Resolves exact downloadable GGUF artifact for a given model and quantization
pub async fn resolve_artifact(model_id: &str, quantization: &str, hf_token: Option<&str>) -> Result<ResolvedArtifact> {
    let repo_id = resolve_gguf_repo(model_id);
    let api_url = format!("https://huggingface.co/api/models/{}/tree/main", repo_id);

    let client = reqwest::Client::builder()
        .user_agent("Sarathi/0.1.0 (Windows; x64)")
        .build()?;

    let mut req = client.get(&api_url);
    if let Some(token) = hf_token {
        if !token.trim().is_empty() {
            req = req.header("Authorization", format!("Bearer {}", token.trim()));
        }
    }

    let quant_lower = quantization.to_lowercase();
    let quant_clean = quant_lower.replace('_', "");

    let resp = req.send().await;
    if let Ok(res) = resp {
        if res.status().is_success() {
            if let Ok(items) = res.json::<Vec<HfTreeItem>>().await {
                // Collect all matching GGUF files for the requested quantization
                let mut matches: Vec<HfTreeItem> = Vec::new();
                for item in &items {
                    let path_lower = item.path.to_lowercase();
                    if path_lower.ends_with(".gguf") {
                        let path_clean = path_lower.replace('_', "");
                        if path_clean.contains(&quant_clean) || path_lower.contains(&quant_lower) {
                            matches.push(item.clone());
                        }
                    }
                }

                // Everything in the repository, so a repository whose requested
                // quantization is only present as a helper file can still fall
                // back to the model it actually holds.
                let all_gguf: Vec<HfTreeItem> = items
                    .iter()
                    .filter(|i| i.path.to_lowercase().ends_with(".gguf"))
                    .cloned()
                    .collect();

                if !matches.is_empty() || !all_gguf.is_empty() {
                    // Order the shortlist by how likely each file is to be what
                    // was asked for: the requested quantization first, then
                    // anything else in the repository. The probe decides; this
                    // only decides what gets probed first, so a correct first
                    // guess costs one request.
                    let candidates = rank_candidates(&matches, &all_gguf);

                    let filenames: Vec<String> = candidates.iter().map(|c| c.path.clone()).collect();

                    // The authority. Names and sizes got us this far; the file's
                    // own header is what says whether it is a model.
                    match super::probe::select_model_file(&repo_id, &filenames, hf_token).await {
                        Ok(selection) => {
                            let (chosen_name, architecture, expert_count) = match &selection {
                                super::probe::Selection::Verified(p) => (
                                    p.filename.clone(),
                                    Some(p.metadata.architecture.clone()),
                                    p.metadata.expert_count,
                                ),
                                super::probe::Selection::Unverified { filename, reason } => {
                                    log::warn!(
                                        "[HF RESOLVER] {reason}; downloading '{filename}' \
                                         unverified. It will be checked against its own header \
                                         before it is loaded."
                                    );
                                    (filename.clone(), None, 0)
                                }
                            };

                            let selected = candidates
                                .iter()
                                .find(|c| c.path == chosen_name)
                                .cloned()
                                .unwrap_or_else(|| candidates[0].clone());

                            let size = selected.lfs.as_ref().map(|l| l.size).or(selected.size).unwrap_or(0);
                            let sha256 = selected.lfs.clone().map(|l| l.oid);
                            let download_url =
                                format!("https://huggingface.co/{}/resolve/main/{}", repo_id, selected.path);

                            // Read from the file, not from the request. Asking
                            // for BF16 and receiving the repository's MXFP4
                            // model must be recorded as MXFP4, or Storage
                            // describes a file that does not exist.
                            let actual_quant = super::discovery::quantization_label(&selected.path)
                                .unwrap_or_else(|| quantization.to_string());

                            if actual_quant != quantization {
                                log::info!(
                                    "[HF RESOLVER] '{}' was requested but the repository's \
                                     standalone model is '{}' ({}); using it",
                                    quantization,
                                    selected.path,
                                    actual_quant
                                );
                            }

                            log::info!(
                                "[HF RESOLVER] Resolved '{}' — architecture '{}', {} bytes",
                                selected.path,
                                architecture.as_deref().unwrap_or("unverified"),
                                size
                            );

                            return Ok(ResolvedArtifact {
                                repo_id,
                                file_name: selected.path,
                                download_url,
                                size_bytes: size,
                                quantization: actual_quant,
                                sha256,
                                architecture,
                                expert_count,
                            });
                        }
                        Err(e) => {
                            // Every file in the repository is a helper. Starting
                            // the download anyway is what produced a model that
                            // could never load; refusing here is the whole point
                            // of looking first.
                            return Err(anyhow!("{e}"));
                        }
                    }
                }
            }
        }
    }

    // Fallback: construct canonical Hugging Face URL if direct API tree search timed out or hit offline mode
    let file_name = format!("{}-{}.gguf", repo_id.split('/').last().unwrap_or("model"), quantization);
    let download_url = format!("https://huggingface.co/{}/resolve/main/{}", repo_id, file_name);

    // Ask the server directly for size and digest. A size of 0 used to flow
    // through to the downloader, which skips its integrity check when it does
    // not know how many bytes to expect — so a truncated transfer would be
    // renamed to `.gguf` and served as a working model.
    let (size_bytes, sha256) = probe_remote_artifact(&client, &download_url, hf_token).await;
    log::info!(
        "[HF RESOLVER] Tree API unavailable; fell back to '{}' (HEAD reported size: {} bytes)",
        file_name, size_bytes
    );

    Ok(ResolvedArtifact {
        repo_id,
        file_name,
        download_url,
        size_bytes,
        quantization: quantization.to_string(),
        sha256,
        // The tree listing was unreachable, so no header was read. The load-time
        // check in `ai_engine::gguf_meta` is the remaining guard, and it refuses
        // a helper file with the same explanation this would have given — after
        // the download rather than before it.
        architecture: None,
        expert_count: 0,
    })
}

/// Asks the server for an artifact's true size and digest via HEAD.
///
/// Returns `(0, None)` when the server cannot be reached or refuses the request;
/// callers must treat an unknown size as "verify some other way", never as
/// "no verification needed".
async fn probe_remote_artifact(
    client: &reqwest::Client,
    url: &str,
    hf_token: Option<&str>,
) -> (u64, Option<String>) {
    let mut req = client.head(url);
    if let Some(token) = hf_token {
        if !token.trim().is_empty() {
            req = req.header("Authorization", format!("Bearer {}", token.trim()));
        }
    }

    let Ok(resp) = req.send().await else {
        return (0, None);
    };
    if !resp.status().is_success() {
        return (0, None);
    }

    let header = |name: &str| {
        resp.headers()
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(|v| v.trim_matches('"').to_string())
    };

    // For LFS/Xet-backed files these headers describe the actual weights;
    // Content-Length on a redirect hop only describes the pointer file.
    let size = header("x-linked-size")
        .and_then(|v| v.parse::<u64>().ok())
        .or_else(|| resp.content_length())
        .unwrap_or(0);

    // `X-Linked-ETag` carries the LFS object id, which is the file's SHA-256.
    let sha256 = header("x-linked-etag").filter(|v| v.len() == 64 && v.chars().all(|c| c.is_ascii_hexdigit()));

    (size, sha256)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_gguf_repo_mapping() {
        assert_eq!(resolve_gguf_repo("meta-llama/Llama-3.2-1B"), "bartowski/Llama-3.2-1B-Instruct-GGUF");
        assert_eq!(resolve_gguf_repo("Qwen/Qwen2.5-Coder-7B"), "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF");
    }

    #[tokio::test]
    async fn test_real_hf_artifact_resolution() {
        let artifact = resolve_artifact("meta-llama/Llama-3.2-1B", "Q8_0", None).await;
        assert!(artifact.is_ok(), "HuggingFace API artifact resolution must succeed");
        let art = artifact.unwrap();
        println!("\n=== REAL HF ARTIFACT RESOLUTION RESULT ===");
        println!("Repo ID: {}", art.repo_id);
        println!("File Name: {}", art.file_name);
        println!("Download URL: {}", art.download_url);
        println!("Size: {:.2} MB ({} bytes)", art.size_bytes as f64 / 1_048_576.0, art.size_bytes);
        println!("==========================================\n");
        assert!(art.download_url.starts_with("https://huggingface.co/"));
        assert!(art.file_name.to_lowercase().contains("q8_0"));
        assert!(art.size_bytes > 0, "Artifact size must be > 0 bytes from HF API");
    }
}
