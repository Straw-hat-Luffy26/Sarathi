//! The one place adapters are discovered for a base model.
//!
//! Three screens ask the same question — the Discover model card, an installed
//! model in Storage, and a model's detail view — and they must get the same
//! answer. Before this existed only Discover could ask, because it was the only
//! screen holding a HuggingFace model card to read the parent id from; Storage
//! had to send the user back to the library to search for their own model again.
//!
//! So the question is asked here, once, and every screen calls it. What differs
//! between them is only how they *obtain* the base model id: Discover reads it
//! from the card it is already showing, Storage resolves it from the installed
//! package via [`super::resolve_upstream`].
//!
//! ## Caching
//!
//! Adapter listings change on the timescale of someone publishing an adapter,
//! and re-asking on every screen open cost a HuggingFace request per open. The
//! policy mirrors [`super::catalog_cache`]: fresh for an hour, usable for a
//! week, and a refresh the user can always force. Nothing here refreshes in the
//! background — a stale answer is shown with its age instead, which is honest
//! and needs no second writer.

use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::live_catalog::{self, AdapterListing};

/// Within this, a cached listing is served without comment.
pub const FRESH_FOR: chrono::Duration = chrono::Duration::hours(1);

/// Past this a listing is refetched rather than shown. A week-old answer is
/// still useful; a month-old one is misleading.
pub const USABLE_FOR: chrono::Duration = chrono::Duration::days(7);

/// How many adapters to request per lookup.
const LIMIT: u32 = 20;

/// Adapters published for a base model, and how usable they are here.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AdapterPage {
    /// The id these were found for. Surfaced so the UI can say *which* model it
    /// searched — the answer is only meaningful alongside the question.
    pub base_model_id: String,
    pub adapters: Vec<AdapterListing>,
    /// How many load as they are. The rest are PEFT safetensors, which Sarathi
    /// converts to GGUF during installation.
    pub ready_count: usize,
    /// Whole hours since this was fetched, when it came from the cache and is
    /// past its freshness window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub age_hours: Option<i64>,
    /// Shown when the result needs explaining rather than just listing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notice: Option<String>,
}

impl AdapterPage {
    /// The page shown when Sarathi cannot tell what a model was built from.
    ///
    /// Deliberately not an empty list: "no adapters exist" and "I do not know
    /// where to look" are different facts, and showing the first when the second
    /// is true is how a broken lookup passes for an honest one.
    pub fn unknown_base_model(package_id: &str) -> Self {
        Self {
            base_model_id: String::new(),
            adapters: Vec::new(),
            ready_count: 0,
            age_hours: None,
            notice: Some(format!(
                "Sarathi could not determine which model '{package_id}' was built from, so it \
                 cannot check adapter compatibility. Adapters are published against the original \
                 model rather than a quantization of it."
            )),
        }
    }

    fn describe(
        base_model_id: String,
        adapters: Vec<AdapterListing>,
        age_hours: Option<i64>,
    ) -> Self {
        let ready_count = adapters.iter().filter(|a| a.gguf_ready).count();

        let notice = if adapters.is_empty() {
            Some(format!("No LoRA adapters published for {base_model_id} yet."))
        } else if ready_count == 0 {
            Some(format!(
                "{} adapter(s) found. None ship GGUF, so Sarathi converts them during install — \
                 which needs this base model installed and its family supported.",
                adapters.len()
            ))
        } else {
            None
        };

        Self { base_model_id, adapters, ready_count, age_hours, notice }
    }
}

/// Finds adapters compatible with `base_model_id`.
///
/// `base_model_id` must be the **original** model an adapter author would
/// declare, not a quantization repository. Callers holding only an installed
/// package should get it from [`super::resolve_upstream::resolve`], which never
/// guesses one.
///
/// `force_refresh` bypasses the cache — this is what "Find more" does, and the
/// only way a user can make Sarathi ask again before the cache expires.
pub async fn compatible_adapters(
    app_data_dir: &Path,
    base_model_id: &str,
    base_architecture: Option<&str>,
    token: Option<&str>,
    force_refresh: bool,
) -> Result<AdapterPage> {
    let base = base_model_id.trim().to_string();
    if base.is_empty() {
        return Ok(AdapterPage::describe(base, Vec::new(), None));
    }

    if !force_refresh {
        if let Some((adapters, age)) = read_cached(app_data_dir, &base) {
            let mut adapters = adapters;
            judge_all(&mut adapters, base_architecture);
            let age_hours = (age >= FRESH_FOR).then(|| age.num_hours());
            log::debug!(
                "[ADAPTERS] Serving {} cached adapter(s) for '{base}' ({}h old)",
                adapters.len(),
                age.num_hours()
            );
            return Ok(AdapterPage::describe(base, adapters, age_hours));
        }
    }

    let mut adapters = fetch(&base, token).await?;
    enrich_use_cases(&mut adapters, token).await;
    // Cached before judging: what the Hub published is a fact about the adapter,
    // while installability is a fact about *this* base model. Storing the verdict
    // would make the cache wrong the moment it is read for a different one.
    write_cached(app_data_dir, &base, &adapters);
    judge_all(&mut adapters, base_architecture);
    Ok(AdapterPage::describe(base, adapters, None))
}

/// Queries the Hub, falling back to name aliases when the exact id finds nothing.
///
/// The exact `base_model:adapter:` lookup is authoritative and is always tried
/// first. The fallback exists because that tag is only as good as the adapter
/// author's spelling of it: an author writing `Qwen/Qwen2.5-7B` where the
/// canonical id is `Qwen/Qwen2.5-7B-Instruct` would otherwise be invisible. It
/// only ever runs when the exact lookup found nothing, so it can never displace
/// an authoritative match.
async fn fetch(base_model_id: &str, token: Option<&str>) -> Result<Vec<AdapterListing>> {
    let exact = live_catalog::find_adapters(base_model_id, LIMIT, token).await?;
    if !exact.is_empty() {
        return Ok(exact);
    }

    let aliases =
        super::adapter_provider::HuggingFaceAdapterProvider::extract_model_aliases(base_model_id);

    for alias in aliases {
        // Aliases without an organisation cannot address a repository, and the
        // Hub filter needs a full id.
        if !alias.contains('/') || alias == base_model_id.to_lowercase() {
            continue;
        }
        match live_catalog::find_adapters(&alias, LIMIT, token).await {
            Ok(found) if !found.is_empty() => {
                log::info!(
                    "[ADAPTERS] Exact lookup for '{base_model_id}' found nothing; \
                     '{alias}' matched {} adapter(s)",
                    found.len()
                );
                return Ok(found);
            }
            Ok(_) => {}
            Err(e) => log::debug!("[ADAPTERS] Alias lookup '{alias}' failed: {e:#}"),
        }
    }

    Ok(exact)
}

// ─── Use cases ──────────────────────────────────────────────────────────────

/// Reads each adapter's model card and settles what it is for.
///
/// Tags and the repository name already gave most adapters a use case; the
/// card supplies a one-line description for all of them and the use case for
/// names like `rinlekha` that say nothing. One small request per adapter — at
/// most [`LIMIT`] — made once per lookup and then cached with it for a week.
///
/// Laya reads what is still unsettled. It runs on the CPU beside chat routing,
/// so it is given a few adapters, not the whole list.
async fn enrich_use_cases(adapters: &mut [AdapterListing], token: Option<&str>) {
    use super::use_case;

    const CONCURRENCY: usize = 6;
    /// Laya calls per lookup. Each takes about half a second on a laptop CPU.
    const LAYA_BUDGET: usize = 8;

    let Ok(client) = reqwest::Client::builder()
        .user_agent("Sarathi/0.1.0")
        .timeout(std::time::Duration::from_secs(15))
        .build()
    else {
        return;
    };

    for chunk in adapters.chunks_mut(CONCURRENCY) {
        let cards = futures_util::future::join_all(
            chunk.iter().map(|a| fetch_card(&client, &a.repo_id, token)),
        )
        .await;
        for (adapter, card) in chunk.iter_mut().zip(cards) {
            let current = adapter
                .use_case
                .take()
                .unwrap_or_else(|| use_case::from_metadata(&adapter.repo_id, &[], None));
            adapter.use_case = Some(match card {
                Some(text) => use_case::with_card(&current, &adapter.repo_id, &text),
                None => current,
            });
        }
    }

    let unsettled: Vec<(usize, String, Option<String>)> = adapters
        .iter()
        .enumerate()
        .filter_map(|(i, a)| {
            let uc = a.use_case.as_ref()?;
            (uc.source == "none").then(|| (i, a.repo_id.clone(), uc.summary.clone()))
        })
        .take(LAYA_BUDGET)
        .collect();
    if unsettled.is_empty() {
        return;
    }

    // Laya blocks while it thinks, so it runs off the async runtime.
    let picks = tokio::task::spawn_blocking(move || {
        unsettled
            .into_iter()
            .filter_map(|(i, id, summary)| use_case::from_laya(&id, summary.as_deref()).map(|l| (i, l)))
            .collect::<Vec<_>>()
    })
    .await
    .unwrap_or_default();

    for (i, label) in picks {
        if let Some(uc) = adapters[i].use_case.as_mut() {
            log::debug!("[ADAPTERS] Laya named '{}' a {label} adapter", adapters[i].repo_id);
            uc.label = label;
            uc.source = "laya".to_string();
        }
    }
}

/// The head of an adapter's README: the purpose is in the first paragraph, and
/// some cards run to megabytes of benchmark tables.
async fn fetch_card(client: &reqwest::Client, repo_id: &str, token: Option<&str>) -> Option<String> {
    const CARD_BYTES: usize = 8 * 1024;

    let mut req = client
        .get(format!("https://huggingface.co/{repo_id}/raw/main/README.md"))
        .header(reqwest::header::RANGE, format!("bytes=0-{}", CARD_BYTES - 1));
    if let Some(t) = token.map(str::trim).filter(|t| !t.is_empty()) {
        req = req.bearer_auth(t);
    }
    let mut resp = req.send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }

    // Read only as far as needed, even from a server that ignored the range.
    let mut body: Vec<u8> = Vec::with_capacity(CARD_BYTES);
    while body.len() < CARD_BYTES {
        match resp.chunk().await {
            Ok(Some(bytes)) => body.extend_from_slice(&bytes),
            _ => break,
        }
    }
    body.truncate(CARD_BYTES);
    Some(String::from_utf8_lossy(&body).into_owned())
}

// ─── Cache ──────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
struct CacheEntry {
    fetched_at: String,
    adapters: Vec<AdapterListing>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct CacheFile {
    #[serde(default)]
    entries: std::collections::HashMap<String, CacheEntry>,
}

fn cache_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join("adapters").join("discovery-cache.json")
}

/// Returns a cached listing and its age, when one is still usable.
fn read_cached(
    app_data_dir: &Path,
    base_model_id: &str,
) -> Option<(Vec<AdapterListing>, chrono::Duration)> {
    let text = std::fs::read_to_string(cache_path(app_data_dir)).ok()?;
    let file: CacheFile = serde_json::from_str(&text).ok()?;
    let entry = file.entries.get(base_model_id)?;

    // Written before every adapter carried a use case. Served as it stands, the
    // list would show some adapters with one and some without for a week.
    if entry.adapters.iter().any(|a| a.use_case.is_none()) {
        return None;
    }

    let fetched = chrono::DateTime::parse_from_rfc3339(&entry.fetched_at).ok()?;
    // A clock that has gone backwards must not read as a negative age.
    let age =
        (chrono::Utc::now() - fetched.with_timezone(&chrono::Utc)).max(chrono::Duration::zero());

    (age <= USABLE_FOR).then(|| (entry.adapters.clone(), age))
}

/// Records a listing. Failures are logged, never propagated: the caller already
/// has the answer, and a request must not fail because a cache could not.
fn write_cached(app_data_dir: &Path, base_model_id: &str, adapters: &[AdapterListing]) {
    let path = cache_path(app_data_dir);
    if let Some(parent) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            log::warn!("[ADAPTERS] Could not create the adapter cache folder: {e:#}");
            return;
        }
    }

    let mut file: CacheFile = std::fs::read_to_string(&path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default();

    file.entries.insert(
        base_model_id.to_string(),
        CacheEntry {
            fetched_at: chrono::Utc::now().to_rfc3339(),
            adapters: adapters.to_vec(),
        },
    );

    match serde_json::to_string_pretty(&file) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&path, json) {
                log::warn!("[ADAPTERS] Could not write the adapter cache: {e:#}");
            }
        }
        Err(e) => log::warn!("[ADAPTERS] Could not serialise the adapter cache: {e:#}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("sarathi-adapter-cache-tests").join(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    pub(super) fn listing(repo: &str, ready: bool) -> AdapterListing {
        AdapterListing {
            size_bytes: 0,
            capability: None,
            capability_confidence: None,
            installable: true,
            blocked_reason: None,
            repo_id: repo.to_string(),
            name: repo.to_string(),
            author: "org".to_string(),
            downloads: 10,
            likes: 1,
            gguf_ready: ready,
            focus: "coding".to_string(),
            use_case: Some(crate::model_providers::huggingface::use_case::from_metadata(repo, &[], None)),
        }
    }

    #[test]
    fn a_listing_cached_before_use_cases_existed_is_refetched() {
        let dir = scratch("pre-use-case");
        let mut old = listing("org/a", true);
        old.use_case = None;
        write_cached(&dir, "Qwen/Qwen2.5-7B-Instruct", &[old]);
        assert!(
            read_cached(&dir, "Qwen/Qwen2.5-7B-Instruct").is_none(),
            "every adapter must show a use case, so an old listing is not served"
        );
    }

    #[test]
    fn a_written_listing_reads_back() {
        let dir = scratch("roundtrip");
        write_cached(&dir, "Qwen/Qwen2.5-7B-Instruct", &[listing("org/a", true)]);

        let (adapters, age) = read_cached(&dir, "Qwen/Qwen2.5-7B-Instruct").expect("cached");
        assert_eq!(adapters.len(), 1);
        assert_eq!(adapters[0].repo_id, "org/a");
        assert!(age < FRESH_FOR, "a listing just written is fresh");
    }

    #[test]
    fn entries_are_keyed_per_base_model() {
        let dir = scratch("per_model");
        write_cached(&dir, "Qwen/Qwen2.5-7B-Instruct", &[listing("org/qwen-lora", true)]);
        write_cached(&dir, "meta-llama/Llama-3.1-8B", &[listing("org/llama-lora", false)]);

        assert_eq!(
            read_cached(&dir, "Qwen/Qwen2.5-7B-Instruct").unwrap().0[0].repo_id,
            "org/qwen-lora"
        );
        assert_eq!(
            read_cached(&dir, "meta-llama/Llama-3.1-8B").unwrap().0[0].repo_id,
            "org/llama-lora"
        );
        // A model never looked up is a miss, not another model's answer.
        assert!(read_cached(&dir, "google/gemma-2-9b").is_none());
    }

    #[test]
    fn a_listing_past_its_usable_life_is_a_miss() {
        let dir = scratch("expired");
        let path = cache_path(&dir);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        let stale = chrono::Utc::now() - chrono::Duration::days(8);
        let mut entries = std::collections::HashMap::new();
        entries.insert(
            "Qwen/Qwen2.5-7B-Instruct".to_string(),
            CacheEntry { fetched_at: stale.to_rfc3339(), adapters: vec![listing("org/a", true)] },
        );
        std::fs::write(&path, serde_json::to_string(&CacheFile { entries }).unwrap()).unwrap();

        assert!(read_cached(&dir, "Qwen/Qwen2.5-7B-Instruct").is_none());
    }

    #[test]
    fn a_stale_but_usable_listing_is_served_with_its_age() {
        let dir = scratch("stale_usable");
        let path = cache_path(&dir);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        let stale = chrono::Utc::now() - chrono::Duration::hours(30);
        let mut entries = std::collections::HashMap::new();
        entries.insert(
            "Qwen/Qwen2.5-7B-Instruct".to_string(),
            CacheEntry { fetched_at: stale.to_rfc3339(), adapters: vec![listing("org/a", true)] },
        );
        std::fs::write(&path, serde_json::to_string(&CacheFile { entries }).unwrap()).unwrap();

        let (adapters, age) = read_cached(&dir, "Qwen/Qwen2.5-7B-Instruct").expect("still usable");
        assert_eq!(adapters.len(), 1);
        assert!(age >= FRESH_FOR, "30 hours is past the freshness window");
        assert!(age <= USABLE_FOR);
    }

    /// An unknown base model is a different fact from "no adapters exist", and
    /// the page has to say which one it is.
    #[test]
    fn an_unknown_base_model_explains_itself_rather_than_showing_an_empty_list() {
        let page = AdapterPage::unknown_base_model("bartowski/Some-Model-GGUF");
        assert!(page.adapters.is_empty());
        assert_eq!(page.ready_count, 0);
        let notice = page.notice.expect("must explain");
        assert!(notice.contains("could not determine"));
        assert!(notice.contains("bartowski/Some-Model-GGUF"));
    }

    #[test]
    fn a_page_counts_only_directly_loadable_adapters_as_ready() {
        let page = AdapterPage::describe(
            "Qwen/Qwen2.5-7B-Instruct".to_string(),
            vec![listing("org/a", true), listing("org/b", false), listing("org/c", false)],
            None,
        );
        assert_eq!(page.ready_count, 1);
        assert!(page.notice.is_none(), "at least one is ready, so nothing needs explaining");
    }

    #[test]
    fn a_page_with_nothing_loadable_says_conversion_is_needed() {
        let page = AdapterPage::describe(
            "Qwen/Qwen2.5-7B-Instruct".to_string(),
            vec![listing("org/b", false)],
            None,
        );
        assert_eq!(page.ready_count, 0);
        assert!(page.notice.unwrap().contains("converts them during install"));
    }
}

/// Decides whether each adapter can actually be installed against this base.
///
/// The rule is not invented here: an adapter that already ships GGUF loads as it
/// stands, and one that does not has to go through
/// [`crate::lora::convert_adapter`], which refuses architectures
/// [`tensor_map::supports_architecture`] does not cover. Asking the same question
/// at listing time is what stops the UI offering a download the installer is
/// going to reject.
///
/// An unknown architecture leaves the adapter installable. Being unable to prove
/// something incompatible is not evidence that it is.
fn judge_all(adapters: &mut [AdapterListing], base_architecture: Option<&str>) {
    use crate::lora::convert::tensor_map;

    for a in adapters.iter_mut() {
        // Ships GGUF: nothing to convert, so the conversion rule cannot bite.
        if a.gguf_ready {
            a.installable = true;
            a.blocked_reason = None;
            continue;
        }

        match base_architecture {
            Some(arch) if !tensor_map::supports_architecture(arch) => {
                a.installable = false;
                a.blocked_reason = Some(format!(
                    "Needs converting, and Sarathi cannot convert adapters for '{}' models yet. Supported families: {}.",
                    arch,
                    tensor_map::supported_architectures().join(", ")
                ));
            }
            _ => {
                a.installable = true;
                a.blocked_reason = None;
            }
        }
    }
}

#[cfg(test)]
mod compatibility_tests {
    use super::*;

    fn peft(repo: &str) -> AdapterListing {
        super::tests::listing(repo, false)
    }

    /// The bug this exists to prevent: an adapter needing conversion, offered
    /// against a base model whose family the converter does not handle. It used
    /// to show a Get button and fail only after the download.
    #[test]
    fn an_unconvertible_family_blocks_adapters_that_need_converting() {
        let mut list = vec![peft("org/a")];
        judge_all(&mut list, Some("lfm2moe"));

        assert!(!list[0].installable);
        let reason = list[0].blocked_reason.as_deref().unwrap();
        assert!(reason.contains("lfm2moe"), "names the family: {reason}");
        assert!(reason.contains("qwen2"), "lists what is supported: {reason}");
    }

    #[test]
    fn a_supported_family_leaves_them_installable() {
        let mut list = vec![peft("org/a")];
        judge_all(&mut list, Some("qwen2"));

        assert!(list[0].installable);
        assert!(list[0].blocked_reason.is_none());
    }

    /// A GGUF adapter is bound as it stands, so the conversion rule cannot apply
    /// to it however exotic the base model's family is.
    #[test]
    fn a_ready_adapter_is_never_blocked_by_the_conversion_rule() {
        let mut list = vec![super::tests::listing("org/a", true)];
        judge_all(&mut list, Some("lfm2moe"));

        assert!(list[0].installable);
    }

    /// Being unable to prove something incompatible is not evidence that it is.
    #[test]
    fn an_unknown_family_hides_nothing() {
        let mut list = vec![peft("org/a")];
        judge_all(&mut list, None);

        assert!(list[0].installable);
    }
}
