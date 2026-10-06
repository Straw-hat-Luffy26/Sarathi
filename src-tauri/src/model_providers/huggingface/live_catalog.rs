//! Live catalog fetching against the HuggingFace Hub.
//!
//! Turns [`discovery`] parsing into actual network calls. Two phases, because
//! bulk search omits the `gguf` metadata object:
//!
//! 1. One search request returns candidate repositories.
//! 2. Per-repo detail requests (run concurrently, in bounded batches) return
//!    parameter counts, architecture, context length, and exact file sizes.
//!
//! ## Authentication
//!
//! No token is required. Searching, reading model metadata, and reading the
//! metadata of *gated* repositories all work anonymously — verified against the
//! live API. A token is needed only to **download** files from a gated repo such
//! as `meta-llama/*` or `google/gemma-*`, and to raise anonymous rate limits.
//!
//! So a token is optional here and requested only when a download needs it,
//! rather than being demanded during setup.

use std::time::Duration;

use anyhow::{anyhow, Result};

use crate::model_providers::huggingface::brands;
use crate::model_providers::huggingface::discovery::{
    detail_url, estimate_architecture, family_name, org_search_url, search_url, GgufRepo,
    RawModelInfo,
};
use crate::model_recommendation::traits::{ModelArchitecture, ModelMetadata};

/// Concurrent detail requests. Kept modest so anonymous rate limits are not hit
/// and a large sweep does not saturate the user's connection.
const CONCURRENCY: usize = 8;

/// Search pages to sweep without a token.
///
/// HuggingFace rate-limits anonymous callers by IP, and every repository costs
/// one detail request on top of the search. A 5-page sweep is ~500 requests and
/// reliably trips the limit, which returns 429 and empties the catalog. One page
/// keeps anonymous use inside the allowance.
pub const ANONYMOUS_PAGES: u32 = 1;

/// Search pages to sweep with a token. Authenticated limits are far higher.
///
/// At 100 repositories per page this is 2,000 candidates, which is what fills
/// every category in the library rather than only the most-downloaded corner of
/// it. Three things make that affordable:
///
/// - The result is stored by
///   [`catalog_cache`](super::catalog_cache) and survives a restart, so the cost
///   is paid once and then refreshed behind an already-drawn listing rather than
///   per visit.
/// - Detail requests run at [`CONCURRENCY`], so the sweep is bounded work rather
///   than 2,000 serial round trips.
/// - A page that fails past the first is not fatal: `discover_repos` keeps
///   everything already collected and stops there. Reaching a rate limit
///   therefore shortens the catalog instead of emptying it, which is what makes
///   raising this safe rather than a gamble.
///
/// It is deliberately not unbounded. The Hub holds tens of thousands of GGUF
/// repositories, and sweeping all of them would cost an hour of requests to
/// surface models nobody scrolls to. The long tail is reachable through search,
/// which queries the Hub directly and is not limited by this.
pub const AUTHENTICATED_PAGES: u32 = 20;

/// Pages to sweep given whether a token is available.
pub fn pages_for(token: Option<&str>) -> u32 {
    if token.map(str::trim).is_some_and(|t| !t.is_empty()) {
        AUTHENTICATED_PAGES
    } else {
        ANONYMOUS_PAGES
    }
}

/// Raised when HuggingFace rejects a request for rate-limiting reasons.
///
/// Distinguished from other failures so the UI can tell the user the specific,
/// fixable cause — add a free token — rather than a generic network error.
#[derive(Debug, Clone, PartialEq)]
pub struct RateLimited {
    pub had_token: bool,
}

impl std::fmt::Display for RateLimited {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.had_token {
            write!(
                f,
                "HuggingFace rate limit reached even with a token. Wait a few minutes and try again."
            )
        } else {
            write!(
                f,
                "HuggingFace rate-limited this connection. Add a free HuggingFace token in \
                 Settings to browse the full model library."
            )
        }
    }
}

impl std::error::Error for RateLimited {}

/// Requests left in HuggingFace's current rate-limit window, as the most recent
/// response reported it. `u64::MAX` until any response has said.
static REMAINING: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);

/// When that window resets, in Unix milliseconds; 0 when not reported.
///
/// `REMAINING` is only ever refreshed by a response, so without this a sweep
/// that stopped at its reserve would leave a low count behind that nothing
/// clears — and the next sweep would refuse to start long after the Hub had
/// restored the allowance.
static RESET_AT_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Requests a sweep leaves unspent in the window.
///
/// The Hub allows an authenticated account 1,000 API requests per five minutes
/// and an anonymous address far fewer, and a 20-page sweep wants ~2,000. Run to
/// exhaustion, the sweep left nothing behind: every search, adapter lookup and
/// download resolution in the next five minutes was refused with 429, which is
/// what "fetching from HuggingFace is broken" looked like from the app. Stopping
/// with this much in hand keeps everything the user does next working, and the
/// merge in `catalog_cache` means the repositories not reached this time are
/// carried over rather than lost.
const SWEEP_RESERVE: u64 = 150;

/// Reads one `key=<number>` parameter of `RateLimit: "api";r=<remaining>;t=<seconds>`
/// (IETF draft syntax, as the Hub sends it).
fn rate_limit_param(value: &str, key: &str) -> Option<u64> {
    value
        .split(';')
        .find_map(|part| part.trim().strip_prefix(key)?.strip_prefix('='))
        .and_then(|n| n.trim().parse().ok())
}

/// Requests remaining in the window.
fn parse_remaining(value: &str) -> Option<u64> {
    rate_limit_param(value, "r")
}

fn record_rate_limit(headers: &reqwest::header::HeaderMap) {
    use std::sync::atomic::Ordering;

    let Some(value) = headers.get("ratelimit").and_then(|v| v.to_str().ok()) else {
        return;
    };
    if let Some(remaining) = parse_remaining(value) {
        REMAINING.store(remaining, Ordering::Relaxed);
        let reset_at = rate_limit_param(value, "t").map_or(0, |secs| now_ms() + secs * 1000);
        RESET_AT_MS.store(reset_at, Ordering::Relaxed);
    }
}

/// True once a sweep has spent the window down to its reserve.
fn sweep_budget_spent() -> bool {
    use std::sync::atomic::Ordering;
    budget_spent(
        REMAINING.load(Ordering::Relaxed),
        RESET_AT_MS.load(Ordering::Relaxed),
        now_ms(),
    )
}

/// The rule behind [`sweep_budget_spent`], kept pure so it can be tested
/// without racing the network tests that update [`REMAINING`].
fn budget_spent(remaining: u64, reset_at_ms: u64, now_ms: u64) -> bool {
    // A window that has rolled over has its whole allowance back, whatever the
    // last response said.
    if reset_at_ms != 0 && now_ms >= reset_at_ms {
        return false;
    }
    remaining < SWEEP_RESERVE + CONCURRENCY as u64
}

/// Per-request timeout. Detail calls are small JSON documents.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);

/// Repos with fewer downloads than this are skipped — the long tail is mostly
/// abandoned experiments and personal scratch uploads.
const MIN_DOWNLOADS: u64 = 50;

fn client(token: Option<&str>) -> Result<reqwest::Client> {
    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(t) = token.map(str::trim).filter(|t| !t.is_empty()) {
        let value = reqwest::header::HeaderValue::from_str(&format!("Bearer {t}"))
            .map_err(|_| anyhow!("HuggingFace token contains invalid characters"))?;
        headers.insert(reqwest::header::AUTHORIZATION, value);
    }
    reqwest::Client::builder()
        .user_agent("Sarathi/0.1.0")
        .timeout(REQUEST_TIMEOUT)
        .default_headers(headers)
        .build()
        .map_err(Into::into)
}

/// Searches the Hub for GGUF repositories.
///
/// `query` is an optional free-text filter, so the UI can search the full Hub
/// rather than only what the default listing returns.
pub async fn search_repos(
    query: Option<&str>,
    limit: u32,
    page: u32,
    token: Option<&str>,
) -> Result<Vec<String>> {
    ids_from_search(&search_url(query, limit, page), token, MIN_DOWNLOADS).await
}

/// Lists the GGUF repositories one publisher owns.
///
/// Unlike [`search_repos`], `org` is matched exactly against the owner half of
/// the repo id, so this returns what NVIDIA *published* rather than what merely
/// mentions NVIDIA. The slug must already be cased as the Hub stores it; see
/// [`org_search_url`].
///
/// The download floor is not applied here. It exists to keep abandoned personal
/// uploads out of a popularity sweep, and a first-party release is not that: a
/// model NVIDIA published last week has few downloads *because it is new*, and
/// dropping it is precisely the failure someone searching "nvidia" would notice.
/// Naming the publisher is the quality signal the floor was standing in for.
pub async fn search_org_repos(
    org: &str,
    limit: u32,
    page: u32,
    token: Option<&str>,
) -> Result<Vec<String>> {
    ids_from_search(&org_search_url(org, limit, page), token, 0).await
}

/// Runs one search request and returns the repo ids it yielded.
async fn ids_from_search(url: &str, token: Option<&str>, min_downloads: u64) -> Result<Vec<String>> {
    let client = client(token)?;

    let resp = client.get(url).send().await?;
    record_rate_limit(resp.headers());
    let status = resp.status();
    if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
        return Err(RateLimited { had_token: token.is_some() }.into());
    }
    if !status.is_success() {
        return Err(anyhow!("HuggingFace search returned {status}"));
    }

    let raw: Vec<RawModelInfo> = resp.json().await?;
    Ok(raw
        .into_iter()
        .filter(|r| r.downloads >= min_downloads)
        .map(|r| r.id)
        .collect())
}

/// Fetches full details for one repository.
pub async fn fetch_repo(repo_id: &str, token: Option<&str>) -> Result<GgufRepo> {
    let client = client(token)?;
    let resp = client.get(detail_url(repo_id)).send().await?;
    record_rate_limit(resp.headers());

    let status = resp.status();
    if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
        return Err(RateLimited { had_token: token.is_some() }.into());
    }
    if !status.is_success() {
        return Err(anyhow!("HuggingFace returned {status} for {repo_id}"));
    }

    let raw: RawModelInfo = resp.json().await?;
    raw.into_repo()
        .ok_or_else(|| anyhow!("{repo_id} has no usable GGUF files"))
}

/// How far along a sweep is.
///
/// A sweep is two phases with very different shapes: a handful of search pages,
/// then a detail request per repository — hundreds of them. Reporting them as
/// one percentage would sit at zero through the first phase and then crawl, so
/// each phase reports its own counts and the UI says which is running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SweepProgress {
    /// Listing candidate repositories. `page` is 1-based.
    Searching { page: u32, pages: u32, found: usize },
    /// Reading each repository's GGUF metadata and file sizes.
    Fetching { done: usize, total: usize },
}

/// Reports sweep progress. Called from the sweep's own task, so implementations
/// must not block.
pub type ProgressFn<'a> = &'a (dyn Fn(SweepProgress) + Send + Sync);

/// Fetches many repositories concurrently, in bounded batches.
///
/// Individual failures are logged and skipped rather than failing the sweep —
/// one unreachable repo must not empty the catalog.
pub async fn fetch_repos(repo_ids: &[String], token: Option<&str>) -> Vec<GgufRepo> {
    fetch_repos_reporting(repo_ids, token, None).await
}

/// [`fetch_repos`], reporting after each batch.
pub async fn fetch_repos_reporting(
    repo_ids: &[String],
    token: Option<&str>,
    progress: Option<ProgressFn<'_>>,
) -> Vec<GgufRepo> {
    let mut out = Vec::with_capacity(repo_ids.len());
    let total = repo_ids.len();
    let mut attempted = 0usize;

    if let Some(report) = progress {
        report(SweepProgress::Fetching { done: 0, total });
    }

    for chunk in repo_ids.chunks(CONCURRENCY) {
        if sweep_budget_spent() {
            log::warn!(
                "[HF_CATALOG] Stopping after {} of {total} repositories to keep {SWEEP_RESERVE} \
                 HuggingFace requests in hand; the rest are kept from the stored library",
                out.len()
            );
            break;
        }

        let futures = chunk.iter().map(|id| async move {
            match fetch_repo(id, token).await {
                Ok(repo) => (Some(repo), false),
                Err(e) => {
                    let limited = e.downcast_ref::<RateLimited>().is_some();
                    if !limited {
                        log::debug!("[HF_CATALOG] Skipping {id}: {e}");
                    }
                    (None, limited)
                }
            }
        });

        let results = futures_util::future::join_all(futures).await;

        // Once the limit is hit, every remaining request will fail the same way.
        // Stop and keep what we have rather than spending hundreds of doomed
        // requests and digging the rate limit deeper.
        let limited = results.iter().any(|(_, l)| *l);
        out.extend(results.into_iter().filter_map(|(r, _)| r));

        // Counted as attempted rather than as kept: a repository skipped for
        // having no usable GGUF is still work done, and a bar that stalls
        // whenever a few are skipped looks like a hang.
        attempted += chunk.len();
        if let Some(report) = progress {
            report(SweepProgress::Fetching { done: attempted, total });
        }

        if limited {
            log::warn!(
                "[HF_CATALOG] Rate limited after {} repositories — keeping partial results. {}",
                out.len(),
                RateLimited { had_token: token.is_some() }
            );
            break;
        }
    }

    out
}

/// A LoRA adapter published for some base model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AdapterListing {
    pub repo_id: String,
    pub name: String,
    pub author: String,
    pub downloads: u64,
    pub likes: u64,
    /// True when the repo ships `.gguf` files, which llama.cpp can load directly.
    ///
    /// Most published adapters are PEFT safetensors and are **not** loadable as
    /// they stand — they need converting to GGUF first. Saying so on the card is
    /// the difference between "download this and it works" and a file that
    /// silently fails to load.
    pub gguf_ready: bool,
    /// Short description derived from the repo name, e.g. `text to sql`.
    pub focus: String,
    /// Bytes the adapter's own weight file occupies, when the Hub reports it.
    ///
    /// Only the weight file, not the repository: an adapter repo also carries a
    /// config, a README and often a tokenizer copy, none of which is downloaded.
    /// Summing the repository would overstate what installing costs.
    ///
    /// `0` means the Hub did not report a size. Shown as unknown rather than as
    /// zero, so a total never quietly understates itself.
    #[serde(default)]
    pub size_bytes: u64,
    /// Capability slot this adapter fills, when its metadata says.
    ///
    /// Read the same way the details panel reads it — the author's tags first,
    /// the repository name only as a fallback — so the grouping shown here and
    /// the slot it lands in after install cannot disagree.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capability: Option<String>,
    /// `stated` when the author's tags said so, `suggested` when it was read out
    /// of the name. Absent when neither settled it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capability_confidence: Option<String>,
    /// Whether Sarathi can actually install this against the current base model.
    ///
    /// Defaults to `true` so a listing cached before this existed keeps working;
    /// the judgement is recomputed on every read anyway.
    #[serde(default = "yes")]
    pub installable: bool,
    /// Why it cannot be installed, in the user's words. `None` when it can.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocked_reason: Option<String>,
    /// What the adapter is for — `Finance`, `Tax · Law`, `Lyrics & music` —
    /// for every adapter, not only the ones that fill a capability slot.
    ///
    /// `None` only in listings cached before this existed; those are refetched.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub use_case: Option<super::use_case::UseCase>,
}

/// Serde default for [`AdapterListing::installable`].
fn yes() -> bool {
    true
}

/// Finds LoRA adapters published for a given base model.
///
/// Uses HuggingFace's `base_model:adapter:` tag, which adapter authors set to
/// declare their parent — so this is a real lookup, not a name-similarity guess.
pub async fn find_adapters(
    base_model_id: &str,
    limit: u32,
    token: Option<&str>,
) -> Result<Vec<AdapterListing>> {
    let base = base_model_id.trim();
    if base.is_empty() || !base.contains('/') {
        // Without an org/name id there is nothing to match against.
        return Ok(Vec::new());
    }

    let client = client(token)?;
    let url = format!(
        "https://huggingface.co/api/models?filter=base_model:adapter:{}&sort=downloads&direction=-1&limit={}&full=true",
        base,
        limit.clamp(1, 50)
    );

    let resp = client.get(&url).send().await?;
    if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
        return Err(RateLimited { had_token: token.is_some() }.into());
    }
    if !resp.status().is_success() {
        return Err(anyhow!("HuggingFace returned {} searching adapters", resp.status()));
    }

    let raw: Vec<RawModelInfo> = resp.json().await?;

    // Carrying the `base_model:adapter:` tag is a claim, not evidence.
    //
    // Repositories publishing merged fine-tunes keep that tag, and marking them
    // `gguf_ready: false` was not enough — they still appeared as adapters the
    // user could try to install, and the failure only surfaced after the
    // download had been started. Dropping them here means the list never offers
    // something that cannot be an adapter.
    Ok(raw
        .into_iter()
        .filter(|r| {
            let files: Vec<(String, u64)> = r
                .siblings
                .iter()
                .map(|s| (s.rfilename.clone(), s.size.unwrap_or(0)))
                .collect();

            let kind = crate::adapter_manager::store::classify_repo_files(&files);
            if kind == crate::adapter_manager::store::RepoKind::FullModel {
                log::debug!(
                    "[ADAPTERS] '{}' claims to be an adapter but ships model weights — not listed",
                    r.id
                );
                return false;
            }
            true
        })
        .map(to_adapter_listing)
        .collect())
}

fn to_adapter_listing(raw: RawModelInfo) -> AdapterListing {
    let author = raw
        .author
        .clone()
        .unwrap_or_else(|| raw.id.split('/').next().unwrap_or("").to_string());

    // Asking "does it contain a .gguf?" is not enough. Some repositories carry
    // the `base_model:adapter:` tag while shipping fully merged weights, and a
    // model GGUF passes that test — the Get button would then start a
    // multi-gigabyte download that can never be bound as an adapter.
    //
    // The installability rule lives in one place so the button and the download
    // cannot disagree about what is loadable.
    let filenames: Vec<String> = raw.siblings.iter().map(|s| s.rfilename.clone()).collect();
    let gguf_ready = crate::adapter_manager::store::check_installable(&filenames).is_ok();

    // What installing actually downloads: the GGUF when one is loadable as it
    // stands, otherwise the PEFT weights that will be converted. Anything else
    // in the repository stays on the Hub.
    let wanted = crate::adapter_manager::store::check_installable(&filenames)
        .ok()
        .or_else(|| {
            filenames
                .iter()
                .find(|f| f.ends_with("adapter_model.safetensors"))
                .cloned()
        });
    let size_bytes = wanted
        .and_then(|name| {
            raw.siblings
                .iter()
                .find(|s| s.rfilename == name)
                .and_then(|s| s.size)
        })
        .unwrap_or(0);

    let short = raw.id.split('/').next_back().unwrap_or(&raw.id);
    let name = short.replace(['-', '_'], " ");

    // Trim only the boilerplate suffixes, keeping the rest of the name intact.
    //
    // An earlier version filtered out family names, digits, and anything ending
    // in "b", trying to distil a "focus". It mangled real names into single
    // meaningless words — "Persim", "Iraqi", "Merged" — because it stripped
    // everything that looked like model metadata and kept whatever was left.
    // The author's own name is more informative than a guess at its meaning.
    let focus = {
        let lowered = name.to_lowercase();
        let mut trimmed = lowered.as_str();
        for suffix in [" lora", " peft", " adapter", " finetune", " ft"] {
            trimmed = trimmed.strip_suffix(suffix).unwrap_or(trimmed);
        }
        let cleaned = trimmed.trim();
        if cleaned.is_empty() { name.clone() } else { cleaned.to_string() }
    };

    // The author's tags outrank the repository name, which is the same order
    // `capability/assign.rs` uses when the adapter is installed.
    let assignment = crate::capability::assign::infer(&raw.id, &raw.tags);

    // From tags and name only; `adapter_discovery` folds in the model card.
    let use_case = super::use_case::from_metadata(&raw.id, &raw.tags, None);

    AdapterListing {
        use_case: Some(use_case),
        size_bytes,
        capability: assignment.as_ref().map(|a| a.capability.clone()),
        capability_confidence: assignment
            .as_ref()
            .map(|a| a.confidence.as_str().to_string()),
        // Judged against the base model by `adapter_discovery`, which is the only
        // caller that knows which model the question is being asked about.
        installable: true,
        blocked_reason: None,
        repo_id: raw.id,
        name,
        author,
        downloads: raw.downloads,
        likes: raw.likes,
        gguf_ready,
        focus: if focus.trim().is_empty() { "general".to_string() } else { focus },
    }
}

/// Converts a discovered repository into the engine's model record.
///
/// Returns `None` when the repo lacks the GGUF metadata needed to reason about
/// it — better to omit a model than to recommend one on invented numbers.
pub fn to_model_metadata(repo: &GgufRepo) -> Option<ModelMetadata> {
    let gguf = repo.gguf.as_ref()?;
    if gguf.total_parameters == 0 {
        return None;
    }

    let arch = estimate_architecture(gguf.total_parameters);

    // The Hub's GGUF metadata does not expose expert counts, so a discovered
    // model starts out recorded as Dense.
    //
    // That used to be harmless — `gguf.total` counts every expert's weights, so
    // the *total* memory figure was right either way. It is no longer harmless:
    // expert counts now decide **placement**. Routed experts can live in system
    // RAM while attention stays on the GPU, and a model recorded as Dense is
    // sized as though all of it needed VRAM. That is the difference between
    // offering a 21B MoE on a 4 GB card and refusing it.
    //
    // `moe_geometry` supplies verified counts for known models; anything not in
    // that table stays Dense rather than being sized on invented numbers. Once
    // downloaded, `ai_engine::gguf_meta` reads the real geometry from the file
    // and the table is no longer consulted.
    let architecture = ModelArchitecture::Dense;

    let mut use_cases = vec!["chat".to_string(), "general".to_string()];
    let haystack = format!("{} {}", repo.repo_id.to_lowercase(), repo.tags.join(" ").to_lowercase());
    if haystack.contains("coder") || haystack.contains("code") {
        use_cases.push("code".to_string());
    }
    if haystack.contains("math") {
        use_cases.push("math".to_string());
    }
    if haystack.contains("instruct") || haystack.contains("chat") {
        use_cases.push("instruct".to_string());
    }

    let mut model = ModelMetadata {
        id: repo.repo_id.clone(),
        name: repo.display_name(),
        family: family_name(repo),
        architecture,
        total_parameters: gguf.total_parameters,
        active_parameters: None,
        num_layers: arch.num_layers,
        num_attention_heads: arch.num_attention_heads,
        num_kv_heads: arch.num_kv_heads,
        head_dimension: arch.head_dimension,
        hidden_size: arch.hidden_size,
        max_context_length: if gguf.context_length > 0 { gguf.context_length } else { 4096 },
        vocab_size: 0,
        default_dtype: "gguf".to_string(),
        use_cases,
        catalog_version: "live_hf_v2".to_string(),
    };

    // Replaces the size-banded guesses above when the model is a verified MoE.
    // Those guesses are inferred from a parameter count, and a MoE model's count
    // is dominated by experts — gpt-oss-20b reports 20.9B but has the 24 layers
    // of a far smaller dense model.
    if let Some(geometry) = super::moe_geometry::lookup(&gguf.architecture, gguf.total_parameters) {
        geometry.apply(&mut model);
    }

    Some(model)
}

/// Runs a full discovery sweep and returns engine-ready model records.
///
/// `pages` controls breadth: each page is up to 100 repositories, so 5 pages
/// sweeps roughly 500 candidates — versus the 16 hardcoded entries this
/// replaces.
/// Families the popularity sweep does not reliably reach.
///
/// The main sweep is `filter=gguf&sort=downloads`, which returns whatever is
/// most downloaded overall. That misses two cases:
///
/// - Labs that publish **safetensors only**, whose GGUF builds exist solely
///   under quantizers' accounts. `author=deepseek-ai&filter=gguf` and
///   `author=moonshotai&filter=gguf` both return nothing, so seeding by
///   organisation would find neither — the conversions live at
///   `unsloth/DeepSeek-V3.2-GGUF` and `unsloth/Kimi-K2-Instruct-GGUF`.
/// - Labs whose most-downloaded GGUFs are a different modality: `nvidia`'s top
///   GGUF results are speech models, so its text models never surface.
///
/// These are therefore *name* seeds, not organisation seeds, and nothing here
/// names a specific repository — whatever the Hub actually returns is what gets
/// listed. Each result still goes through the same `to_model_metadata`
/// conversion and the same compatibility and format checks as the main sweep,
/// so seeding widens what is *considered*, never what is *vouched for*.
pub const SEED_QUERIES: &[&str] = &[
    "DeepSeek",
    "Kimi",
    "Nemotron",
    "GLM",
    "MiniMax",
];

/// Drops repeated ids, keeping the position of the first occurrence.
///
/// The org sweep and the free-text search overlap whenever a publisher's own
/// build also matches the typed word, which for `Qwen/Qwen3-8B-GGUF` on a
/// search for "qwen" is always. Keeping the *first* occurrence is what makes
/// the org sweep worth running: it ran first, so the official copy holds the
/// place it earned there instead of falling back to wherever download-ordered
/// text search happened to put it.
fn dedupe_preserving_order(ids: &mut Vec<String>) {
    let mut seen = std::collections::HashSet::new();
    ids.retain(|id| seen.insert(id.clone()));
}

pub async fn discover(
    query: Option<&str>,
    pages: u32,
    token: Option<&str>,
) -> Result<Vec<ModelMetadata>> {
    let repos = discover_repos(query, pages, token).await?;
    let models: Vec<ModelMetadata> = repos.iter().filter_map(to_model_metadata).collect();

    let finetunes = repos.iter().filter(|r| r.is_finetune).count();
    let adapters = repos.iter().filter(|r| r.is_lora_adapter).count();
    log::info!(
        "[HF_CATALOG] Resolved {} models from {} repositories ({} fine-tunes, {} LoRA adapters)",
        models.len(),
        repos.len(),
        finetunes,
        adapters
    );

    Ok(models)
}

/// Runs a discovery sweep and returns the raw repositories.
///
/// Kept separate from [`discover`] because model cards need the full repository
/// — every quantization with its exact size, the tags, the publisher — while the
/// recommendation engine only needs the distilled [`ModelMetadata`]. Converting
/// early would throw away what the cards display.
pub async fn discover_repos(
    query: Option<&str>,
    pages: u32,
    token: Option<&str>,
) -> Result<Vec<GgufRepo>> {
    discover_repos_reporting(query, pages, token, None).await
}

/// [`discover_repos`], reporting progress as it goes.
///
/// A full authenticated sweep is ~2,000 requests and takes minutes. Without
/// this the UI has nothing to say for the whole of it, which is why the loading
/// state read as a frozen application.
pub async fn discover_repos_reporting(
    query: Option<&str>,
    pages: u32,
    token: Option<&str>,
    progress: Option<ProgressFn<'_>>,
) -> Result<Vec<GgufRepo>> {
    let mut repo_ids = Vec::new();
    let total_pages = pages.max(1);

    // A query that names a publisher gets that publisher's shelf first.
    //
    // The Hub's `search=` is a substring match ordered by downloads, so "nvidia"
    // leads with `nvidia/parakeet-ctc-1.1b` — a speech recogniser — and the
    // Nemotron builds someone actually wanted are further down, if they made the
    // page at all. Asking `author=nvidia` separately puts the official releases
    // at the front of the candidate list, where the front end's relevance
    // ordering keeps them.
    //
    // This runs *in addition to* the free-text search, never instead of it.
    // DeepSeek, Meta, and Moonshot publish no GGUF themselves, so their org
    // query is legitimately empty and the community conversions found by
    // free text are the entire answer. Replacing one with the other would
    // report that DeepSeek has no models.
    let brand = query.and_then(brands::resolve);
    if let Some(brand) = brand {
        for org in brand.orgs {
            match search_org_repos(org, 100, 0, token).await {
                Ok(ids) => {
                    log::info!(
                        "[HF_CATALOG] '{}' resolved to {} — {} repositories published by {org}",
                        query.unwrap_or_default(),
                        brand.label,
                        ids.len()
                    );
                    repo_ids.extend(ids);
                }
                // An org sweep is an enrichment, not the answer. Losing it costs
                // ranking; failing the search would cost the results themselves.
                Err(e) => log::warn!("[HF_CATALOG] Org sweep for {org} failed, skipping: {e}"),
            }
        }
    }

    for page in 0..total_pages {
        // Each listed repository costs a detail request later, so paging on
        // with the window nearly spent only queues work that cannot run.
        if page > 0 && sweep_budget_spent() {
            log::warn!("[HF_CATALOG] Rate-limit window nearly spent; stopping at page {page}");
            break;
        }

        if let Some(report) = progress {
            report(SweepProgress::Searching {
                page: page + 1,
                pages: total_pages,
                found: repo_ids.len(),
            });
        }

        match search_repos(query, 100, page, token).await {
            Ok(ids) if ids.is_empty() => break, // no more results
            Ok(ids) => repo_ids.extend(ids),
            // The first page failing is fatal — unless the org sweep already
            // found the brand's models, in which case there is a real answer to
            // return and erroring would throw it away.
            Err(e) if page == 0 && repo_ids.is_empty() => return Err(e),
            Err(e) => {
                log::warn!("[HF_CATALOG] Page {page} failed, continuing with what we have: {e}");
                break;
            }
        }
    }

    dedupe_preserving_order(&mut repo_ids);

    log::info!("[HF_CATALOG] {} candidate repositories found", repo_ids.len());

    Ok(fetch_repos_reporting(&repo_ids, token, progress).await)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_providers::huggingface::discovery::{GgufMeta, Quantization};

    fn ids(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn the_official_copy_keeps_the_place_the_org_sweep_gave_it() {
        // How a search for "qwen" arrives: the org sweep first, then the
        // download-ordered text search, which repeats the same repository far
        // down its own list behind more popular third-party conversions.
        let mut merged = ids(&[
            "Qwen/Qwen3-8B-GGUF",
            "Qwen/Qwen3-4B-GGUF",
            "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
            "bartowski/Qwen3-8B-GGUF",
            "Qwen/Qwen3-8B-GGUF",
        ]);
        dedupe_preserving_order(&mut merged);

        assert_eq!(
            merged,
            ids(&[
                "Qwen/Qwen3-8B-GGUF",
                "Qwen/Qwen3-4B-GGUF",
                "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
                "bartowski/Qwen3-8B-GGUF",
            ]),
            "the duplicate must collapse into the earlier position, not the later one"
        );
    }

    #[test]
    fn deduping_leaves_a_list_without_repeats_alone() {
        let mut only_search = ids(&["a/one-GGUF", "b/two-GGUF"]);
        dedupe_preserving_order(&mut only_search);
        assert_eq!(only_search, ids(&["a/one-GGUF", "b/two-GGUF"]));

        let mut empty: Vec<String> = Vec::new();
        dedupe_preserving_order(&mut empty);
        assert!(empty.is_empty());
    }

    fn repo(params: u64, arch: &str, id: &str) -> GgufRepo {
        GgufRepo {
            repo_id: id.to_string(),
            author: "someone".into(),
            downloads: 1000,
            likes: 10,
            last_modified: String::new(),
            quantizations: vec![Quantization {
                label: "Q4_K_M".into(),
                filename: "m-Q4_K_M.gguf".into(),
                size_bytes: 4_000_000_000,
                is_sharded: false,
            }],
            gguf: Some(GgufMeta {
                total_parameters: params,
                architecture: arch.to_string(),
                context_length: 32768,
                chat_template: None,
                bos_token: None,
                eos_token: None,
            }),
            base_model: None,
            is_finetune: false,
            is_lora_adapter: false,
            tags: vec![],
        }
    }

    #[test]
    fn a_repo_with_gguf_metadata_converts() {
        let m = to_model_metadata(&repo(7_600_000_000, "qwen2", "bartowski/Qwen2.5-Coder-7B-GGUF"))
            .expect("should convert");

        assert_eq!(m.total_parameters, 7_600_000_000);
        assert_eq!(m.max_context_length, 32768);
        assert_eq!(m.architecture, ModelArchitecture::Dense);
        assert!(m.use_cases.contains(&"code".to_string()), "coder repo should be tagged for code");
        assert_eq!(m.catalog_version, "live_hf_v2");
    }

    #[test]
    fn a_repo_without_gguf_metadata_is_skipped() {
        // Recommending on invented numbers is worse than omitting the model.
        let mut r = repo(1, "llama", "x/y");
        r.gguf = None;
        assert!(to_model_metadata(&r).is_none());

        let mut zero = repo(0, "llama", "x/y");
        zero.gguf.as_mut().unwrap().total_parameters = 0;
        assert!(to_model_metadata(&zero).is_none());
    }

    #[test]
    fn moe_models_keep_their_full_parameter_count() {
        // An architecture with no verified geometry stays Dense, but the total
        // must still include every expert's weights — that is the number memory
        // budgeting depends on.
        let m = to_model_metadata(&repo(46_000_000_000, "mixtral", "x/Mixtral-GGUF")).unwrap();

        assert_eq!(m.total_parameters, 46_000_000_000);
        assert_eq!(m.architecture, ModelArchitecture::Dense);
        assert!(m.active_parameters.is_none(), "must not invent an active-parameter count");
    }

    /// The regression that made expert offload unreachable for the models it
    /// was built for: every discovered repo was recorded as Dense, so the
    /// scorer sized a 21B MoE as though all of it needed VRAM.
    #[test]
    fn a_verified_moe_repo_carries_its_expert_geometry() {
        let m = to_model_metadata(&repo(20_900_000_000, "gpt-oss", "unsloth/gpt-oss-20b-GGUF"))
            .expect("should convert");

        assert!(
            matches!(
                m.architecture,
                ModelArchitecture::MixtureOfExperts { num_experts: 32, active_experts: 4 }
            ),
            "got {:?}",
            m.architecture
        );
        assert_eq!(m.active_parameters, Some(3_600_000_000));
        assert_eq!(
            m.num_layers, 24,
            "verified geometry must replace the size-banded guess"
        );
        assert_eq!(m.total_parameters, 20_900_000_000, "the total still counts every expert");
    }

    #[test]
    fn qwen3_moe_is_recognised_too() {
        let m = to_model_metadata(&repo(30_500_000_000, "qwen3moe", "unsloth/Qwen3-30B-A3B-GGUF"))
            .expect("should convert");

        assert!(matches!(
            m.architecture,
            ModelArchitecture::MixtureOfExperts { num_experts: 128, active_experts: 8 }
        ));
        assert_eq!(m.num_layers, 48);
    }

    #[test]
    fn an_unverified_moe_architecture_stays_dense_rather_than_guessing() {
        let m = to_model_metadata(&repo(236_000_000_000, "deepseek2", "x/DeepSeek-V2-GGUF"))
            .expect("should convert");

        assert_eq!(m.architecture, ModelArchitecture::Dense);
        assert!(m.active_parameters.is_none());
    }

    #[test]
    fn architecture_estimates_scale_with_size() {
        let small = estimate_architecture(1_000_000_000);
        let mid = estimate_architecture(8_000_000_000);
        let large = estimate_architecture(70_000_000_000);

        assert!(small.num_layers < mid.num_layers);
        assert!(mid.num_layers < large.num_layers);
        // Grouped-query attention is near-universal in current open models.
        assert_eq!(mid.num_kv_heads, 8);
    }

    #[test]
    fn a_finetune_inherits_its_parents_family() {
        let mut r = repo(8_000_000_000, "llama", "someone/my-custom-tune-GGUF");
        r.base_model = Some("meta-llama/Llama-3.1-8B".into());
        r.is_finetune = true;

        let m = to_model_metadata(&r).unwrap();
        assert_eq!(m.family, "Llama 3.1 8B", "fine-tunes belong to the parent family");
        assert_eq!(m.id, "someone/my-custom-tune-GGUF", "but keep their own id");
    }

    #[test]
    fn sweep_breadth_depends_on_having_a_token() {
        // A 5-page anonymous sweep is ~500 requests and trips HuggingFace's
        // per-IP limit, which returns 429 and yields an empty catalog.
        assert_eq!(pages_for(None), ANONYMOUS_PAGES);
        assert_eq!(pages_for(Some("   ")), ANONYMOUS_PAGES, "blank token is no token");
        assert_eq!(pages_for(Some("hf_realtoken")), AUTHENTICATED_PAGES);
        assert!(ANONYMOUS_PAGES < AUTHENTICATED_PAGES);
    }

    #[test]
    fn rate_limit_message_names_the_fix_when_there_is_no_token() {
        let anon = RateLimited { had_token: false }.to_string();
        assert!(anon.contains("token"), "must tell the user what to do: {anon}");
        assert!(anon.contains("Settings"), "must say where to do it: {anon}");

        // With a token there is nothing to add, so the advice must differ.
        let authed = RateLimited { had_token: true }.to_string();
        assert!(authed.contains("Wait"), "should advise waiting instead: {authed}");
        assert_ne!(anon, authed);
    }

    #[test]
    fn the_hubs_rate_limit_header_is_understood() {
        // Verbatim shapes from live responses.
        assert_eq!(parse_remaining(r#""api";r=998;t=204"#), Some(998));
        assert_eq!(parse_remaining(r#""api";r=0;t=12"#), Some(0));
        assert_eq!(parse_remaining(r#""api"; r=395 ; t=268"#), Some(395));
        assert_eq!(parse_remaining("garbage"), None);
        assert_eq!(parse_remaining(r#""api";t=204"#), None);
        assert_eq!(rate_limit_param(r#""api";r=998;t=204"#, "t"), Some(204));
        assert_eq!(
            rate_limit_param(r#""api";tr=5;t=9"#, "t"),
            Some(9),
            "a key must match whole, not as a prefix of another"
        );
    }

    #[test]
    fn a_sweep_stops_with_requests_still_in_hand() {
        let now = 1_000_000;
        let later = now + 60_000;
        assert!(!budget_spent(u64::MAX, 0, now), "an unknown window must not block a sweep");
        assert!(!budget_spent(SWEEP_RESERVE + CONCURRENCY as u64 + 50, later, now));
        assert!(
            budget_spent(SWEEP_RESERVE, later, now),
            "the reserve is for the user's next action, not the sweep"
        );
        assert!(budget_spent(0, 0, now), "with no reset reported, a spent count holds");
    }

    #[test]
    fn a_window_that_has_reset_is_not_still_spent() {
        // The count was low when last seen, but the window it described is over.
        assert!(!budget_spent(0, 1_000, 2_000));
        assert!(budget_spent(0, 2_000, 1_000), "still inside the window");
    }

    #[test]
    fn missing_context_length_falls_back_to_a_safe_default() {
        let mut r = repo(3_000_000_000, "llama", "x/y");
        r.gguf.as_mut().unwrap().context_length = 0;

        let m = to_model_metadata(&r).unwrap();
        assert_eq!(m.max_context_length, 4096, "never report a zero context");
    }
}
