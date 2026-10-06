//! Live Hugging Face Model Discovery & Catalog Provider
//!
//! Queries Hugging Face Hub API dynamically for GGUF model repositories,
//! extracts architecture & parameter metadata, caches the catalog locally in AppData,
//! and normalizes models into Sarathi's ModelMetadata abstraction for local evaluation.

use std::fs;
use std::path::Path;
use serde::{Deserialize, Serialize};
use crate::model_recommendation::traits::{ModelMetadata, ModelArchitecture};
use crate::model_recommendation::catalog::bootstrap_models;
use crate::model_providers::huggingface::live_catalog;

#[derive(Debug, Serialize, Deserialize)]
pub struct CatalogCache {
    pub timestamp: u64,
    pub models: Vec<ModelMetadata>,
}

#[derive(Debug, Deserialize)]
struct HfModelSearchResult {
    id: String,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    likes: u64,
    #[serde(default)]
    pipeline_tag: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
}

pub struct HuggingFaceCatalogProvider;

impl HuggingFaceCatalogProvider {
    /// Serialises catalog sweeps so concurrent callers share one network fetch.
    ///
    /// Without this, two simultaneous requests both miss the cache — it is only
    /// written once a sweep finishes — and each runs a full 500-repository
    /// fetch. React's StrictMode double-invokes effects in development, so this
    /// happened on every launch: twice the API calls and twice the wait, for
    /// identical results.
    fn catalog_lock() -> &'static tokio::sync::Mutex<()> {
        static LOCK: std::sync::OnceLock<tokio::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
    }

    /// Reads the cache when it is present, parseable, non-empty, and fresh.
    ///
    /// `min_timestamp` rejects anything older than the given epoch second, which
    /// is how a caller ignores a stale entry it has already decided to refresh.
    fn read_cache(cache_file: &Path, min_timestamp: u64) -> Option<Vec<ModelMetadata>> {
        let content = fs::read_to_string(cache_file).ok()?;
        let cached = serde_json::from_str::<CatalogCache>(&content).ok()?;
        if cached.models.is_empty() || cached.timestamp < min_timestamp {
            return None;
        }
        // Before the empty-sweep check in `query_hf_api`, a rate-limited sweep
        // stored the bootstrap list alone as if it were live. A cache holding
        // no live record is that, and serving it for a day is the bug.
        if !cached.models.iter().any(|m| m.catalog_version.starts_with("live")) {
            return None;
        }
        Some(cached.models)
    }

    pub async fn fetch_catalog(app_data_dir: &Path, force_refresh: bool) -> Vec<ModelMetadata> {
        let cache_file = app_data_dir.join("hf_catalog_cache.json");
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Cache entries older than this are considered stale.
        const CACHE_TTL_SECS: u64 = 86_400;
        let freshness_floor = now.saturating_sub(CACHE_TTL_SECS);

        // 1. Fast path — a fresh cache satisfies the request with no lock held.
        if !force_refresh {
            if let Some(models) = Self::read_cache(&cache_file, freshness_floor) {
                log::info!(
                    "[HF_CATALOG] Loaded {} models from fresh local cache ({:?})",
                    models.len(),
                    cache_file
                );
                return models;
            }
        }

        // 2. Only one sweep at a time. Others queue here.
        let _guard = Self::catalog_lock().lock().await;

        // 3. Re-check after waiting: whoever held the lock may have just
        //    finished the exact sweep we were about to start. Accept only a
        //    result produced after we began waiting, so an explicit refresh is
        //    still honoured rather than served from the entry it meant to replace.
        if let Some(models) = Self::read_cache(&cache_file, now) {
            log::info!(
                "[HF_CATALOG] Reusing {} models from a concurrent refresh",
                models.len()
            );
            return models;
        }

        // 4. Reuse the library the model browser already swept.
        //
        // Both this and the browser used to run their own full sweep on
        // launch, concurrently and under separate locks. Together they asked
        // the Hub for roughly twice its five-minute allowance, so whichever ran
        // second was rate-limited into an empty result — and every search the
        // user made for the next five minutes failed the same way.
        if !force_refresh {
            if let Some(models) = Self::from_stored_library(app_data_dir) {
                log::info!(
                    "[HF_CATALOG] Using {} models from the stored model library",
                    models.len()
                );
                return models;
            }
        }

        // 5. Fetch live catalog from Hugging Face Hub API
        log::info!("[HF_CATALOG] Querying live Hugging Face Hub API for GGUF model repositories...");
        match Self::query_hf_api(Self::resolve_token().as_deref()).await {
            Ok(live_models) if !live_models.is_empty() => {
                log::info!("[HF_CATALOG] Successfully discovered {} models from Hugging Face API", live_models.len());
                let cache_data = CatalogCache {
                    timestamp: now,
                    models: live_models.clone(),
                };
                if let Ok(json) = serde_json::to_string_pretty(&cache_data) {
                    let _ = fs::create_dir_all(app_data_dir);
                    let _ = fs::write(&cache_file, json);
                    log::info!("[HF_CATALOG] Cached live catalog to {:?}", cache_file);
                }
                live_models
            }
            Ok(_empty_models) => {
                log::info!("[HF_CATALOG] API returned empty model list, using bootstrap fallback");
                bootstrap_models()
            }
            Err(err) => {
                log::warn!("[HF_CATALOG] Live API query failed: {}. Attempting cache or bootstrap fallback...", err);
                if cache_file.exists() {
                    if let Ok(content) = fs::read_to_string(&cache_file) {
                        if let Ok(cached) = serde_json::from_str::<CatalogCache>(&content) {
                            if !cached.models.is_empty() {
                                log::info!("[HF_CATALOG] Using cached catalog ({} models) as fallback", cached.models.len());
                                return cached.models;
                            }
                        }
                    }
                }
                log::info!("[HF_CATALOG] Using versioned bootstrap catalog ({} models) as fallback", bootstrap_models().len());
                bootstrap_models()
            }
        }
    }

    /// Engine records from the model browser's stored library, when it holds a
    /// usable one.
    ///
    /// The library is the same Hub data this provider would sweep for — every
    /// repository with its GGUF metadata — so converting it costs a file read
    /// instead of up to two thousand requests. Past `USABLE_FOR` it is refused
    /// for the same reason the browser refuses to show it.
    fn from_stored_library(app_data_dir: &Path) -> Option<Vec<ModelMetadata>> {
        use crate::model_providers::huggingface::catalog_cache;

        let stored = catalog_cache::load(app_data_dir)?;
        let usable = stored
            .age_at(chrono::Utc::now())
            .is_some_and(|age| age < catalog_cache::USABLE_FOR);
        if !usable {
            return None;
        }

        let mut models: Vec<ModelMetadata> = stored
            .repos
            .iter()
            .filter(|r| !r.is_lora_adapter)
            .filter_map(live_catalog::to_model_metadata)
            .collect();
        if models.is_empty() {
            return None;
        }

        for b_model in bootstrap_models() {
            if !models.iter().any(|m| m.id == b_model.id) {
                models.push(b_model);
            }
        }
        Some(models)
    }

    /// Resolves an optional HuggingFace token from the environment.
    ///
    /// Browsing needs no token — search and metadata reads work anonymously,
    /// including for gated repositories. A token only raises anonymous rate
    /// limits and unlocks *downloads* from gated repos (`meta-llama/*`,
    /// `google/gemma-*`), so it is never required to reach this code path.
    ///
    /// Reads the same variable names the official `huggingface_hub` tooling
    /// uses, so an existing setup is picked up automatically.
    fn resolve_token() -> Option<String> {
        crate::config::hf_token::get()
    }

    /// Discovers models from the live Hub.
    ///
    /// Previously this asked for 35 repositories and then passed each through a
    /// hand-written `if repo.contains("llama-3.2-1b")` chain, **discarding any
    /// result that did not match**. Only models typed into the source by hand
    /// could ever appear, so the catalog was pinned at 16 entries and no
    /// fine-tune or new release could ever surface.
    ///
    /// Metadata now comes from the API itself — parameter counts, architecture,
    /// context length, and exact per-quantization file sizes — so any GGUF
    /// repository on the Hub is understood without being known in advance.
    async fn query_hf_api(token: Option<&str>) -> Result<Vec<ModelMetadata>, anyhow::Error> {
        // Sweep breadth depends on whether a token is present: HuggingFace
        // rate-limits anonymous callers by IP, and each repository costs a
        // detail request on top of the search.
        let pages = live_catalog::pages_for(token);
        if token.is_none() {
            log::info!(
                "[HF_CATALOG] No HuggingFace token — sweeping {} page(s). Add a free token in \
                 Settings to browse the full library.",
                pages
            );
        }

        let mut discovered = live_catalog::discover(None, pages, token).await?;

        // A sweep that resolved nothing did not succeed, however it returned.
        // Padding it with the bootstrap list below and storing the result made
        // a rate-limited sweep look like a fresh live catalog of 16 models, and
        // served it for a day.
        if discovered.is_empty() {
            return Err(anyhow::anyhow!(
                "the live sweep resolved no models (most often a HuggingFace rate limit)"
            ));
        }

        // Keep the curated families as a floor. They carry hand-verified
        // architecture details, and guarantee a usable catalog if the Hub is
        // unreachable or a sweep comes back thin.
        for b_model in bootstrap_models() {
            if !discovered.iter().any(|m| m.id == b_model.id) {
                discovered.push(b_model);
            }
        }

        Ok(discovered)
    }

    /// Free-text search across the Hub, for the model browser.
    ///
    /// Bypasses the cache: the user is looking for something specific, and the
    /// cached listing only holds the popular sweep.
    pub async fn search(query: &str, token: Option<&str>) -> Vec<ModelMetadata> {
        match live_catalog::discover(Some(query), 1, token).await {
            Ok(models) => {
                log::info!("[HF_CATALOG] Search '{}' matched {} models", query, models.len());
                models
            }
            Err(e) => {
                log::warn!("[HF_CATALOG] Search '{}' failed: {}", query, e);
                Vec::new()
            }
        }
    }

    fn parse_hf_repo_to_metadata(repo_id: &str) -> Option<ModelMetadata> {
        let repo_lower = repo_id.to_lowercase();
        
        // Map GGUF repositories back to canonical base model families and architecture parameters
        if repo_lower.contains("llama-3.2-1b") {
            Some(ModelMetadata {
                id: "meta-llama/Llama-3.2-1B".into(),
                name: "Llama 3.2 1B".into(),
                family: "Llama 3.2".into(),
                architecture: ModelArchitecture::Dense,
                total_parameters: 1_235_814_400,
                active_parameters: None,
                num_layers: 16,
                num_attention_heads: 32,
                num_kv_heads: 8,
                head_dimension: 64,
                hidden_size: 2048,
                max_context_length: 131072,
                vocab_size: 128256,
                default_dtype: "bf16".into(),
                use_cases: vec!["chat".into(), "general".into()],
                catalog_version: "live_hf".into(),
            })
        } else if repo_lower.contains("llama-3.2-3b") {
            Some(ModelMetadata {
                id: "meta-llama/Llama-3.2-3B".into(),
                name: "Llama 3.2 3B".into(),
                family: "Llama 3.2".into(),
                architecture: ModelArchitecture::Dense,
                total_parameters: 3_212_749_824,
                active_parameters: None,
                num_layers: 28,
                num_attention_heads: 24,
                num_kv_heads: 8,
                head_dimension: 128,
                hidden_size: 3072,
                max_context_length: 131072,
                vocab_size: 128256,
                default_dtype: "bf16".into(),
                use_cases: vec!["chat".into(), "general".into()],
                catalog_version: "live_hf".into(),
            })
        } else if repo_lower.contains("llama-3.1-8b") || repo_lower.contains("llama-3-8b") {
            Some(ModelMetadata {
                id: "meta-llama/Llama-3.1-8B".into(),
                name: "Llama 3.1 8B".into(),
                family: "Llama 3.1".into(),
                architecture: ModelArchitecture::Dense,
                total_parameters: 8_030_261_248,
                active_parameters: None,
                num_layers: 32,
                num_attention_heads: 32,
                num_kv_heads: 8,
                head_dimension: 128,
                hidden_size: 4096,
                max_context_length: 131072,
                vocab_size: 128256,
                default_dtype: "bf16".into(),
                use_cases: vec!["chat".into(), "general".into(), "reasoning".into()],
                catalog_version: "live_hf".into(),
            })
        } else if repo_lower.contains("qwen2.5-coder-7b") {
            Some(ModelMetadata {
                id: "Qwen/Qwen2.5-Coder-7B".into(),
                name: "Qwen 2.5 Coder 7B".into(),
                family: "Qwen 2.5 Coder".into(),
                architecture: ModelArchitecture::Dense,
                total_parameters: 7_610_000_000,
                active_parameters: None,
                num_layers: 28,
                num_attention_heads: 28,
                num_kv_heads: 4,
                head_dimension: 128,
                hidden_size: 3584,
                max_context_length: 32768,
                vocab_size: 152064,
                default_dtype: "bf16".into(),
                use_cases: vec!["code".into()],
                catalog_version: "live_hf".into(),
            })
        } else if repo_lower.contains("qwen2.5-3b") {
            Some(ModelMetadata {
                id: "Qwen/Qwen2.5-3B".into(),
                name: "Qwen 2.5 3B".into(),
                family: "Qwen 2.5".into(),
                architecture: ModelArchitecture::Dense,
                total_parameters: 3_090_000_000,
                active_parameters: None,
                num_layers: 36,
                num_attention_heads: 16,
                num_kv_heads: 2,
                head_dimension: 128,
                hidden_size: 2048,
                max_context_length: 32768,
                vocab_size: 151936,
                default_dtype: "bf16".into(),
                use_cases: vec!["chat".into(), "general".into()],
                catalog_version: "live_hf".into(),
            })
        } else if repo_lower.contains("qwen2.5-7b") {
            Some(ModelMetadata {
                id: "Qwen/Qwen2.5-7B".into(),
                name: "Qwen 2.5 7B".into(),
                family: "Qwen 2.5".into(),
                architecture: ModelArchitecture::Dense,
                total_parameters: 7_610_000_000,
                active_parameters: None,
                num_layers: 28,
                num_attention_heads: 28,
                num_kv_heads: 4,
                head_dimension: 128,
                hidden_size: 3584,
                max_context_length: 131072,
                vocab_size: 152064,
                default_dtype: "bf16".into(),
                use_cases: vec!["chat".into(), "general".into(), "reasoning".into()],
                catalog_version: "live_hf".into(),
            })
        } else if repo_lower.contains("qwen2.5-14b") {
            Some(ModelMetadata {
                id: "Qwen/Qwen2.5-14B".into(),
                name: "Qwen 2.5 14B".into(),
                family: "Qwen 2.5".into(),
                architecture: ModelArchitecture::Dense,
                total_parameters: 14_770_000_000,
                active_parameters: None,
                num_layers: 48,
                num_attention_heads: 40,
                num_kv_heads: 8,
                head_dimension: 128,
                hidden_size: 5120,
                max_context_length: 131072,
                vocab_size: 152064,
                default_dtype: "bf16".into(),
                use_cases: vec!["chat".into(), "general".into(), "reasoning".into()],
                catalog_version: "live_hf".into(),
            })
        } else if repo_lower.contains("qwen2.5-32b") {
            Some(ModelMetadata {
                id: "Qwen/Qwen2.5-32B".into(),
                name: "Qwen 2.5 32B".into(),
                family: "Qwen 2.5".into(),
                architecture: ModelArchitecture::Dense,
                total_parameters: 32_760_000_000,
                active_parameters: None,
                num_layers: 64,
                num_attention_heads: 40,
                num_kv_heads: 8,
                head_dimension: 128,
                hidden_size: 5120,
                max_context_length: 131072,
                vocab_size: 152064,
                default_dtype: "bf16".into(),
                use_cases: vec!["chat".into(), "general".into(), "reasoning".into()],
                catalog_version: "live_hf".into(),
            })
        } else if repo_lower.contains("deepseek-r1-distill-qwen-7b") {
            Some(ModelMetadata {
                id: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B".into(),
                name: "DeepSeek R1 Distill Qwen 7B".into(),
                family: "DeepSeek R1".into(),
                architecture: ModelArchitecture::Dense,
                total_parameters: 7_610_000_000,
                active_parameters: None,
                num_layers: 28,
                num_attention_heads: 28,
                num_kv_heads: 4,
                head_dimension: 128,
                hidden_size: 3584,
                max_context_length: 131072,
                vocab_size: 152064,
                default_dtype: "bf16".into(),
                use_cases: vec!["reasoning".into(), "chat".into()],
                catalog_version: "live_hf".into(),
            })
        } else if repo_lower.contains("deepseek-r1-distill-qwen-14b") {
            Some(ModelMetadata {
                id: "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B".into(),
                name: "DeepSeek R1 Distill Qwen 14B".into(),
                family: "DeepSeek R1".into(),
                architecture: ModelArchitecture::Dense,
                total_parameters: 14_770_000_000,
                active_parameters: None,
                num_layers: 48,
                num_attention_heads: 40,
                num_kv_heads: 8,
                head_dimension: 128,
                hidden_size: 5120,
                max_context_length: 131072,
                vocab_size: 152064,
                default_dtype: "bf16".into(),
                use_cases: vec!["reasoning".into(), "chat".into()],
                catalog_version: "live_hf".into(),
            })
        } else if repo_lower.contains("gemma-2-2b") {
            Some(ModelMetadata {
                id: "google/gemma-2-2b".into(),
                name: "Gemma 2 2B".into(),
                family: "Gemma 2".into(),
                architecture: ModelArchitecture::Dense,
                total_parameters: 2_614_341_888,
                active_parameters: None,
                num_layers: 26,
                num_attention_heads: 8,
                num_kv_heads: 4,
                head_dimension: 256,
                hidden_size: 2304,
                max_context_length: 8192,
                vocab_size: 256000,
                default_dtype: "bf16".into(),
                use_cases: vec!["chat".into(), "general".into()],
                catalog_version: "live_hf".into(),
            })
        } else if repo_lower.contains("gemma-2-9b") {
            Some(ModelMetadata {
                id: "google/gemma-2-9b".into(),
                name: "Gemma 2 9B".into(),
                family: "Gemma 2".into(),
                architecture: ModelArchitecture::Dense,
                total_parameters: 9_241_705_984,
                active_parameters: None,
                num_layers: 42,
                num_attention_heads: 16,
                num_kv_heads: 8,
                head_dimension: 256,
                hidden_size: 3584,
                max_context_length: 8192,
                vocab_size: 256000,
                default_dtype: "bf16".into(),
                use_cases: vec!["chat".into(), "general".into()],
                catalog_version: "live_hf".into(),
            })
        } else if repo_lower.contains("gemma-2-27b") {
            Some(ModelMetadata {
                id: "google/gemma-2-27b".into(),
                name: "Gemma 2 27B".into(),
                family: "Gemma 2".into(),
                architecture: ModelArchitecture::Dense,
                total_parameters: 27_225_856_000,
                active_parameters: None,
                num_layers: 46,
                num_attention_heads: 32,
                num_kv_heads: 16,
                head_dimension: 128,
                hidden_size: 4608,
                max_context_length: 8192,
                vocab_size: 256000,
                default_dtype: "bf16".into(),
                use_cases: vec!["chat".into(), "general".into(), "reasoning".into()],
                catalog_version: "live_hf".into(),
            })
        } else if repo_lower.contains("phi-4") {
            Some(ModelMetadata {
                id: "microsoft/phi-4".into(),
                name: "Phi-4 14B".into(),
                family: "Phi".into(),
                architecture: ModelArchitecture::Dense,
                total_parameters: 14_700_000_000,
                active_parameters: None,
                num_layers: 40,
                num_attention_heads: 40,
                num_kv_heads: 10,
                head_dimension: 128,
                hidden_size: 5120,
                max_context_length: 16384,
                vocab_size: 100352,
                default_dtype: "bf16".into(),
                use_cases: vec!["reasoning".into(), "general".into()],
                catalog_version: "live_hf".into(),
            })
        } else if repo_lower.contains("mistral-7b") {
            Some(ModelMetadata {
                id: "mistralai/Mistral-7B-v0.3".into(),
                name: "Mistral 7B v0.3".into(),
                family: "Mistral".into(),
                architecture: ModelArchitecture::Dense,
                total_parameters: 7_248_020_480,
                active_parameters: None,
                num_layers: 32,
                num_attention_heads: 32,
                num_kv_heads: 8,
                head_dimension: 128,
                hidden_size: 4096,
                max_context_length: 32768,
                vocab_size: 32768,
                default_dtype: "bf16".into(),
                use_cases: vec!["chat".into(), "general".into()],
                catalog_version: "live_hf".into(),
            })
        } else if repo_lower.contains("mixtral-8x7b") {
            Some(ModelMetadata {
                id: "mistralai/Mixtral-8x7B-v0.1".into(),
                name: "Mixtral 8×7B".into(),
                family: "Mixtral".into(),
                architecture: ModelArchitecture::MixtureOfExperts {
                    num_experts: 8,
                    active_experts: 2,
                },
                total_parameters: 46_700_000_000,
                active_parameters: Some(12_900_000_000),
                num_layers: 32,
                num_attention_heads: 32,
                num_kv_heads: 8,
                head_dimension: 128,
                hidden_size: 4096,
                max_context_length: 32768,
                vocab_size: 32000,
                default_dtype: "bf16".into(),
                use_cases: vec!["chat".into(), "general".into()],
                catalog_version: "live_hf".into(),
            })
        } else if repo_lower.contains("nemotron-3-nano-4b") {
            Some(ModelMetadata {
                id: "nvidia/NVIDIA-Nemotron-3-Nano-4B".into(),
                name: "Nemotron 3 Nano 4B".into(),
                family: "Nemotron".into(),
                architecture: ModelArchitecture::Dense,
                total_parameters: 3_973_556_832,
                active_parameters: None,
                num_layers: 32,
                num_attention_heads: 32,
                num_kv_heads: 8,
                head_dimension: 128,
                hidden_size: 3072,
                max_context_length: 131072,
                vocab_size: 131072,
                default_dtype: "bf16".into(),
                use_cases: vec!["chat".into(), "general".into(), "reasoning".into()],
                catalog_version: "live_hf".into(),
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hf_live_catalog_fetching_and_caching() {
        let temp_dir = std::env::temp_dir().join(format!("sarathi_cat_test_{}", chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)));
        let models = HuggingFaceCatalogProvider::fetch_catalog(&temp_dir, true).await;
        assert!(!models.is_empty(), "Live catalog discovery or fallback must return model metadata");
        let cache_file = temp_dir.join("hf_catalog_cache.json");

        // Offline or rate-limited, the sweep resolves nothing and the bootstrap
        // list is served — and must *not* be stored as a live catalog, which is
        // what used to pin a machine to 16 models for a day.
        if !models.iter().any(|m| m.catalog_version.starts_with("live")) {
            assert!(!cache_file.exists(), "a bootstrap-only result must not be cached as live");
            let _ = fs::remove_dir_all(temp_dir);
            return;
        }
        assert!(cache_file.exists(), "Catalog metadata must be cached locally to disk");

        let cached_models = HuggingFaceCatalogProvider::fetch_catalog(&temp_dir, false).await;
        assert_eq!(models.len(), cached_models.len(), "Subsequent calls must return cached models");
        let _ = fs::remove_dir_all(temp_dir);
    }

    /// The cache found on a real machine: 16 bootstrap records stored as a
    /// fresh "live" catalog after a sweep that resolved nothing.
    #[test]
    fn a_cache_with_no_live_record_is_not_served() {
        let dir = std::env::temp_dir().join(format!("sarathi_cat_bootstrap_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("hf_catalog_cache.json");
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let bootstrap_only = CatalogCache { timestamp: now, models: bootstrap_models() };
        fs::write(&file, serde_json::to_string(&bootstrap_only).unwrap()).unwrap();
        assert!(HuggingFaceCatalogProvider::read_cache(&file, 0).is_none());

        let mut models = bootstrap_models();
        models[0].catalog_version = "live_hf_v2".into();
        fs::write(&file, serde_json::to_string(&CatalogCache { timestamp: now, models }).unwrap()).unwrap();
        assert!(HuggingFaceCatalogProvider::read_cache(&file, 0).is_some());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_parse_nemotron_repo() {
        let meta = HuggingFaceCatalogProvider::parse_hf_repo_to_metadata("nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF");
        assert!(meta.is_some());
        let m = meta.unwrap();
        assert_eq!(m.name, "Nemotron 3 Nano 4B");
        assert_eq!(m.family, "Nemotron");
        assert_eq!(m.total_parameters, 3_973_556_832);
    }
}

