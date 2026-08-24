//! Working out which model an installed GGUF was built from.
//!
//! ## Why this is not obvious
//!
//! A model package is identified by the repository its weights came from. For a
//! quantization that is the converter's repo — `bartowski/Qwen2.5-7B-Instruct-GGUF`
//! — not the model itself. Adapter authors declare the **original**
//! (`Qwen/Qwen2.5-7B-Instruct`) in their `base_model:adapter:` tag, so searching
//! adapters by the package id matches nothing at all.
//!
//! That failure is invisible: an empty adapter list looks exactly like a model
//! nobody has published adapters for. Storage could not offer "find adapters for
//! this model" until it knew the id to search by.
//!
//! ## Why it does not guess
//!
//! Trimming `-GGUF` off the repository name looks like it would work, and often
//! does. It also silently produces `TheBloke/Llama-2-7B-Chat` for
//! `TheBloke/Llama-2-7B-Chat-GGUF`, whose real parent is
//! `meta-llama/Llama-2-7b-chat-hf` — a different organisation entirely. A wrong
//! id here is worse than none: it returns a confident list of adapters for
//! *another model*, which install and then fail to bind.
//!
//! So every source below is one that *states* the parent. When none does, the
//! answer is `None` and the UI says so.

use std::path::Path;

use crate::adapter_manager::AdapterRegistry;

/// Where an upstream id came from, so the caller can explain itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpstreamSource {
    /// Already recorded in the package manifest.
    Manifest,
    /// Declared by the GGUF's own header.
    GgufHeader,
    /// Read from the repository's `base_model:` tag on HuggingFace.
    HubTags,
}

/// A resolved upstream model id and how it was found.
#[derive(Debug, Clone)]
pub struct UpstreamModel {
    pub model_id: String,
    pub source: UpstreamSource,
}

/// Resolves the upstream model for an installed package, persisting the answer.
///
/// Tried in order of cost: the manifest costs nothing, the GGUF header costs a
/// file read, and the Hub costs a request. The first source that *states* an
/// answer wins, and the result is written back so the later ones are consulted
/// at most once per model.
///
/// Returns `None` when no source declares a parent. Callers must treat that as
/// "unknown" and say so, never as an invitation to derive one from the name.
pub async fn resolve(package_dir: &Path, token: Option<&str>) -> Option<UpstreamModel> {
    let manifest = AdapterRegistry::read_manifest(package_dir).ok()?;

    // 1. Already known.
    if let Some(id) = manifest.base_model.upstream_model_id.clone() {
        if is_plausible_repo_id(&id) {
            return Some(UpstreamModel { model_id: id, source: UpstreamSource::Manifest });
        }
    }

    let package_id = manifest.base_model.model_id.clone();

    // 2. The file itself, when its converter recorded a parent. No network.
    if let Ok(gguf) = crate::lora::convert::arch::resolve_base_gguf(package_dir) {
        if let Ok(meta) = crate::ai_engine::gguf_meta::read_gguf_metadata(&gguf) {
            if let Some(id) = meta.base_model_repo.filter(|id| is_plausible_repo_id(id)) {
                log::info!(
                    "[UPSTREAM] '{package_id}' declares base model '{id}' in its GGUF header"
                );
                persist(package_dir, &id);
                return Some(UpstreamModel { model_id: id, source: UpstreamSource::GgufHeader });
            }
        }
    }

    // 3. The Hub's own tags for the repository the weights came from. One
    //    request, and only for a package that reached here without an answer —
    //    the result is persisted, so this does not repeat.
    if let Some(id) = fetch_base_model_tag(&package_id, token).await {
        log::info!("[UPSTREAM] '{package_id}' is tagged on HuggingFace as built from '{id}'");
        persist(package_dir, &id);
        return Some(UpstreamModel { model_id: id, source: UpstreamSource::HubTags });
    }

    log::info!(
        "[UPSTREAM] No source declares what '{package_id}' was built from; \
         adapter compatibility cannot be checked for it"
    );
    None
}

/// Records a resolved id so the cost is paid once.
///
/// A write failure is logged rather than propagated: the id is correct and
/// usable for this turn, and the caller's request should not fail because a
/// cache could not be updated.
fn persist(package_dir: &Path, upstream: &str) {
    let Ok(mut manifest) = AdapterRegistry::read_manifest(package_dir) else {
        return;
    };
    if manifest.base_model.upstream_model_id.as_deref() == Some(upstream) {
        return;
    }
    manifest.base_model.upstream_model_id = Some(upstream.to_string());
    manifest.updated_at = chrono::Utc::now().to_rfc3339();
    if let Err(e) = AdapterRegistry::write_manifest(package_dir, &manifest) {
        log::warn!("[UPSTREAM] Could not record the upstream model id: {e:#}");
    }
}

/// Asks HuggingFace what a repository declares as its base model.
async fn fetch_base_model_tag(repo_id: &str, token: Option<&str>) -> Option<String> {
    if !is_plausible_repo_id(repo_id) {
        return None;
    }

    let client = reqwest::Client::builder()
        .user_agent("Sarathi/0.1.0")
        .build()
        .ok()?;

    let mut req = client.get(format!("https://huggingface.co/api/models/{repo_id}"));
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }

    let info: serde_json::Value = req.send().await.ok()?.json().await.ok()?;
    let tags: Vec<String> = info
        .get("tags")?
        .as_array()?
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();

    // Reuses the tag reader the browse catalogue uses, so a model's parent is
    // read identically wherever the question is asked.
    super::discovery::base_model_from_tags(&tags).filter(|id| is_plausible_repo_id(id))
}

/// A Hub model id is `org/name` — exactly one slash, nothing empty.
///
/// Guards every path above, so a malformed id can never reach an adapter search
/// and come back confidently empty.
fn is_plausible_repo_id(id: &str) -> bool {
    let mut parts = id.split('/');
    match (parts.next(), parts.next(), parts.next()) {
        (Some(org), Some(name), None) => {
            !org.trim().is_empty() && !name.trim().is_empty() && !id.contains(char::is_whitespace)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hub_id_is_org_slash_name() {
        assert!(is_plausible_repo_id("Qwen/Qwen2.5-7B-Instruct"));
        assert!(is_plausible_repo_id("bartowski/Some-Model-GGUF"));
    }

    #[test]
    fn malformed_ids_are_refused_rather_than_searched_with() {
        // A bare name has no organisation, so it cannot address a repository.
        assert!(!is_plausible_repo_id("Qwen2.5-7B-Instruct"));
        // A path with a revision or file is not a model id.
        assert!(!is_plausible_repo_id("Qwen/Qwen2.5/blob/main"));
        assert!(!is_plausible_repo_id(""));
        assert!(!is_plausible_repo_id("/name"));
        assert!(!is_plausible_repo_id("org/"));
        // Whitespace means something has gone wrong upstream of here.
        assert!(!is_plausible_repo_id("org/na me"));
    }
}
