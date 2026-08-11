//! IPC for browsing models by category.
//!
//! Returns presentation-ready cards — publisher, licence, popularity, age,
//! categories, and every quantization sized against this machine's memory — so
//! the browser can render and filter without re-deriving anything.
//!
//! ## Where the library comes from
//!
//! Three places, in order of preference:
//!
//! 1. The in-process cache, for repeat visits within a session.
//! 2. [`catalog_cache`] on disk, so a launch shows the library at once instead
//!    of spending minutes sweeping the Hub before drawing anything.
//! 3. A live sweep, which reports progress as it goes.
//!
//! A cache that is merely old is served *and* refreshed: the user gets the
//! listing immediately and it updates underneath them. Only an unusable cache —
//! absent, ancient, or gathered under different credentials — makes anyone wait.
//!
//! Nothing hardware-dependent is ever cached. Cards are rebuilt from the live
//! hardware profile on every read, so the "runs here" column and the MoE offload
//! tag describe the machine the user is on now, not the one the sweep ran on.

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use tauri::{AppHandle, Emitter, Manager};

use crate::model_providers::huggingface::card::{build_card, ModelCard, ModelCategory};
use crate::model_providers::huggingface::catalog_cache::{self, Freshness};
use crate::model_providers::huggingface::discovery::GgufRepo;
use crate::model_providers::huggingface::live_catalog::{self, SweepProgress};
use crate::model_providers::huggingface::moe_fit::{HostCapacity, BROWSE_CONTEXT};

/// How long a browse sweep stays fresh in memory.
///
/// Each sweep costs one search plus a detail request per repository — around a
/// hundred calls. Without this, every visit to the model browser repeated the
/// whole thing, and React's StrictMode doubles it in development: four full
/// sweeps were observed on a single startup, which walks straight into
/// HuggingFace's anonymous rate limit.
///
/// This is the *session* cache. Surviving a restart is
/// [`catalog_cache`]'s job, and it keeps results far longer.
const BROWSE_CACHE_TTL: Duration = Duration::from_secs(600);

/// Event carrying sweep progress to the model browser.
pub const PROGRESS_EVENT: &str = "catalog:progress";

/// Event fired when a background refresh has replaced the cached library, so an
/// open browser can pick up the new results without the user reloading.
pub const UPDATED_EVENT: &str = "catalog:updated";

/// Cached sweep, with the authentication state that produced it.
///
/// The token decides how much of the library a sweep reaches — one search page
/// anonymously against twenty with a token — so results gathered under one state
/// do not describe the other. Without the flag a cached authenticated sweep
/// could be served to an anonymous caller, which showed a full 182-model
/// listing underneath a notice offering to unlock the full library.
type BrowseCache = Mutex<Option<(Instant, bool, Vec<GgufRepo>)>>;

fn browse_cache() -> &'static BrowseCache {
    static CACHE: OnceLock<BrowseCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(None))
}

/// Serialises sweeps so simultaneous callers share one fetch instead of each
/// starting their own — the cache is only written once a sweep completes.
fn browse_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

/// Set while a background refresh is in flight, so a second visit to the
/// browser does not start another sweep of the same thing.
fn refresh_in_flight() -> &'static std::sync::atomic::AtomicBool {
    static FLAG: OnceLock<std::sync::atomic::AtomicBool> = OnceLock::new();
    FLAG.get_or_init(|| std::sync::atomic::AtomicBool::new(false))
}

fn cached_repos(authenticated: bool) -> Option<Vec<GgufRepo>> {
    let guard = browse_cache().lock().ok()?;
    let (fetched_at, was_authenticated, repos) = guard.as_ref()?;
    (*was_authenticated == authenticated && fetched_at.elapsed() < BROWSE_CACHE_TTL)
        .then(|| repos.clone())
}

fn store_repos(authenticated: bool, repos: &[GgufRepo]) {
    if let Ok(mut guard) = browse_cache().lock() {
        *guard = Some((Instant::now(), authenticated, repos.to_vec()));
    }
}

/// Drops the cached sweep so the next browse re-fetches.
///
/// Called when the HuggingFace token changes: the cached results were gathered
/// under the old credentials, and an anonymous sweep reaches far less of the
/// library than an authenticated one. Without this, adding a token appeared to
/// do nothing until the ten-minute TTL expired.
///
/// The stored library is dropped for the same reason — a cache that survives a
/// restart would otherwise outlive the credentials it was gathered under.
pub fn invalidate_browse_cache(app_data_dir: Option<&std::path::Path>) {
    if let Ok(mut guard) = browse_cache().lock() {
        *guard = None;
    }
    if let Some(dir) = app_data_dir {
        catalog_cache::clear(dir);
    }
}

/// Cards plus the facts needed to explain an empty or partial result.
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogPage {
    pub cards: Vec<ModelCard>,
    /// Every category present in these results, so the sidebar only offers
    /// filters that would actually match something.
    pub categories: Vec<CategoryCount>,
    /// Memory available for model weights, used to mark which quantizations fit.
    /// Zero when hardware could not be read — nothing is then marked as fitting.
    pub weight_budget_bytes: u64,
    /// System RAM available to hold a MoE model's offloaded experts. Zero when
    /// unknown, in which case no model is marked as offloadable.
    pub host_ram_bytes: u64,
    /// Where these results came from, so the UI can say "showing a saved copy,
    /// checking for updates" rather than presenting stale data as current.
    pub source: CatalogSource,
    /// How old the served library is, in seconds. `None` for a live sweep.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub age_seconds: Option<i64>,
    /// True when a refresh is running behind this response. The UI shows a quiet
    /// indicator and waits for [`UPDATED_EVENT`].
    pub refreshing: bool,
    /// Set when the sweep was cut short, e.g. by rate limiting. The results are
    /// still usable; the message explains why there are fewer than expected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notice: Option<String>,
}

/// Which of the three sources answered this request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CatalogSource {
    /// Swept from HuggingFace just now.
    Live,
    /// The in-process cache from earlier in this session.
    Session,
    /// The library stored on disk by a previous run.
    Stored,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CategoryCount {
    pub category: ModelCategory,
    pub label: String,
    pub count: usize,
}

/// Sweep progress, as the model browser receives it.
///
/// `fraction` is the honest one: it is `None` while searching, because the
/// number of repositories a sweep will find is not known until the search pages
/// are in. A bar that guesses and then jumps backwards is worse than a bar that
/// waits.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProgressPayload {
    /// `searching`, `fetching`, `caching`, or `done`.
    pub phase: &'static str,
    /// One line describing what is happening, written for a person.
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fraction: Option<f64>,
    pub done: usize,
    pub total: usize,
    /// True when this sweep is a background refresh of an already-shown library,
    /// so the UI can keep it quiet rather than covering the results.
    pub background: bool,
}

impl ProgressPayload {
    fn from_sweep(progress: SweepProgress, background: bool) -> Self {
        match progress {
            SweepProgress::Searching { page, pages, found } => Self {
                phase: "searching",
                message: format!(
                    "Searching HuggingFace — page {page} of {pages}, {found} models found"
                ),
                // Deliberately absent: the denominator of the long phase is not
                // known yet, and searching is a few seconds of a multi-minute
                // sweep.
                fraction: None,
                done: found,
                total: 0,
                background,
            },
            SweepProgress::Fetching { done, total } => Self {
                phase: "fetching",
                message: format!("Reading model details — {done} of {total}"),
                fraction: (total > 0).then(|| done as f64 / total as f64),
                done,
                total,
                background,
            },
        }
    }

    fn simple(phase: &'static str, message: impl Into<String>, background: bool) -> Self {
        Self {
            phase,
            message: message.into(),
            fraction: None,
            done: 0,
            total: 0,
            background,
        }
    }
}

/// Memory this machine can give to model weights.
///
/// Budgets against the same card the runtime would actually load onto, so the
/// "runs here" column and what happens on Load cannot disagree. Returns 0 when
/// no usable GPU is detected, so callers mark nothing as fitting rather than
/// guessing.
///
/// Two things this deliberately does not do. It does not take the largest
/// reported VRAM: an integrated GPU advertises a slice of system RAM as its
/// own, and on this machine the Radeon 780M's 13 GB outranked an RTX 5060's
/// real 8 GB — which marked 9 GB models as fitting on a card that cannot hold
/// them. And it budgets from *total* rather than free VRAM, because a browse
/// listing that shifts as other processes take memory would be unreadable.
fn weight_budget() -> u64 {
    let analyzer = crate::system_analyzer::get_system_analyzer_manager();
    let Some(profile) = analyzer.get_profile() else {
        return 0;
    };
    weight_budget_for(&profile.gpus.current())
}

/// The budget arithmetic, separated from hardware detection so it can be tested.
///
/// Charges the same three costs the VRAM planner does, in the same order: the
/// OS reserve, then the KV cache, then compute buffers. Weights get the rest.
///
/// Omitting the KV cache made this noticeably more generous than the planner —
/// 6.46 GB against its 5.46 GB on an 8 GB card — so a quantization the browser
/// marked "fits" could still fail to fully offload on Load. The KV cost depends
/// on context, and browsing has no session to ask, so it is priced at a typical
/// working context rather than a model's advertised maximum.
fn weight_budget_for(gpus: &[crate::system_analyzer::traits::GpuInfo]) -> u64 {
    const OS_RESERVE: u64 = 900 * 1024 * 1024;

    let Some(gpu) = crate::ai_engine::manager::select_inference_gpu(gpus) else {
        return 0;
    };

    let usable = gpu.vram_total_bytes.saturating_sub(OS_RESERVE);

    // Sized for a mid-range model: the exact per-token figure depends on the
    // model, which differs per card in the listing, so this uses the band the
    // planner applies to everything from ~3–8 GB of weights.
    let kv_bytes = crate::ai_engine::vram_planner::estimate_kv_bytes_per_token(4 * 1024 * 1024 * 1024)
        * u64::from(BROWSE_CONTEXT);

    let after_kv = usable.saturating_sub(kv_bytes);
    after_kv.saturating_sub((after_kv as f64 * 0.12) as u64)
}

/// What this machine can offer a MoE model, read from the live profile.
///
/// Both figures are zero when hardware could not be read, which
/// [`moe_fit`](crate::model_providers::huggingface::moe_fit) treats as "promise
/// nothing" rather than as a permissive default.
fn host_capacity() -> HostCapacity {
    let analyzer = crate::system_analyzer::get_system_analyzer_manager();
    let Some(profile) = analyzer.get_profile() else {
        return HostCapacity::default();
    };

    let vram_total_bytes =
        crate::ai_engine::manager::select_inference_gpu(&profile.gpus.current())
            .map(|gpu| gpu.vram_total_bytes)
            .unwrap_or(0);

    // Routed through the same budget calculator the loader uses, so the listing
    // and the load apply one notion of "usable RAM" rather than two.
    let usable_ram_bytes = crate::model_recommendation::budget::calculate_budget(
        &profile,
        &crate::model_recommendation::traits::BudgetConfig::default(),
    )
    .system_ram
    .usable_for_inference;

    HostCapacity { vram_total_bytes, usable_ram_bytes }
}

fn tally_categories(cards: &[ModelCard]) -> Vec<CategoryCount> {
    ModelCategory::all()
        .iter()
        .filter_map(|cat| {
            let count = cards.iter().filter(|c| c.categories.contains(cat)).count();
            // Offering a filter that matches nothing is a dead end.
            (count > 0).then(|| CategoryCount {
                category: *cat,
                label: cat.label().to_string(),
                count,
            })
        })
        .collect()
}

/// Turns repositories into a page of cards, sized against the current machine.
///
/// Always called on the way out, never cached: this is where hardware enters,
/// and a stored library must not carry another machine's answers.
fn page_from(
    repos: &[GgufRepo],
    source: CatalogSource,
    age_seconds: Option<i64>,
    refreshing: bool,
    notice: Option<String>,
) -> CatalogPage {
    let budget = weight_budget();
    let host = host_capacity();
    let now = chrono::Utc::now();

    // Adapters are not models and must not be browsed as though they were.
    //
    // A LoRA is a patch that cannot load on its own; listing it beside runnable
    // models offered a Download button for something that would never start.
    // They stay discoverable in the place they make sense — the adapter list on
    // the base model they were trained against.
    let cards: Vec<ModelCard> = repos
        .iter()
        .filter(|r| !r.is_lora_adapter)
        .map(|r| build_card(r, budget, host, now))
        .collect();

    CatalogPage {
        categories: tally_categories(&cards),
        cards,
        weight_budget_bytes: budget,
        host_ram_bytes: host.usable_ram_bytes,
        source,
        age_seconds,
        refreshing,
        notice,
    }
}

fn anonymous_notice(token_present: bool, searching: bool) -> Option<String> {
    (!token_present && !searching).then(|| {
        "Showing the most popular models. Add a free HuggingFace token in Settings to browse \
         the full library."
            .to_string()
    })
}

fn data_dir(app: &AppHandle) -> Option<PathBuf> {
    app.path().app_data_dir().ok()
}

/// Runs a sweep, reporting progress to the front end as it goes.
///
/// `background` only changes how the progress is labelled — a refresh behind an
/// already-drawn listing must not look like a blocking load.
async fn sweep(app: &AppHandle, token: Option<&str>, background: bool) -> Result<Vec<GgufRepo>, String> {
    let emitter = app.clone();
    let report = move |p: SweepProgress| {
        let _ = emitter.emit(PROGRESS_EVENT, ProgressPayload::from_sweep(p, background));
    };

    let pages = live_catalog::pages_for(token);
    let repos = live_catalog::discover_repos_reporting(None, pages, token, Some(&report))
        .await
        // Rate limiting is the common, fixable case and its message says what to
        // do, so it is passed through unchanged.
        .map_err(|e| e.to_string())?;

    let _ = app.emit(
        PROGRESS_EVENT,
        ProgressPayload::simple("caching", "Saving the library for next time…", background),
    );

    store_repos(token.is_some(), &repos);
    if let Some(dir) = data_dir(app) {
        let fingerprint = {
            let host = host_capacity();
            catalog_cache::fingerprint(host.vram_total_bytes, host.usable_ram_bytes)
        };
        if let Err(e) = catalog_cache::store(&dir, token.is_some(), &fingerprint, &repos) {
            // A library that cannot be saved is still a library worth showing.
            log::warn!("[CATALOG] Could not store the model library: {e}");
        }
    }

    let _ = app.emit(
        PROGRESS_EVENT,
        ProgressPayload::simple("done", format!("{} models ready", repos.len()), background),
    );

    Ok(repos)
}

/// Refreshes the stored library behind an already-served page.
///
/// Failures are logged and dropped: the user is looking at a working listing,
/// and interrupting it to report that a background check failed would be worse
/// than showing yesterday's results for another hour.
fn spawn_refresh(app: &AppHandle, token: Option<String>) {
    use std::sync::atomic::Ordering;

    if refresh_in_flight().swap(true, Ordering::SeqCst) {
        log::debug!("[CATALOG] A refresh is already running");
        return;
    }

    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        // Held for the whole refresh so a foreground sweep and this one cannot
        // both be hammering the Hub.
        let _guard = browse_lock().lock().await;

        log::info!("[CATALOG] Refreshing the model library in the background");
        match sweep(&app, token.as_deref(), true).await {
            Ok(repos) => {
                log::info!("[CATALOG] Background refresh brought back {} repositories", repos.len());
                let _ = app.emit(UPDATED_EVENT, repos.len());
            }
            Err(e) => log::warn!("[CATALOG] Background refresh failed: {e}"),
        }
        refresh_in_flight().store(false, Ordering::SeqCst);
    });
}

/// Browses the model catalog, optionally filtered by a search term.
///
/// A search queries HuggingFace directly rather than filtering the cached
/// listing: the cache holds only the popular sweep, and someone searching for a
/// specific fine-tune would otherwise be told it does not exist.
#[tauri::command]
pub async fn browse_model_cards(
    app: AppHandle,
    query: Option<String>,
) -> Result<CatalogPage, String> {
    let token = crate::config::hf_token::get();
    let search = query.as_deref().map(str::trim).filter(|q| !q.is_empty());

    // A search is specific and user-initiated, so it always goes to the Hub —
    // serving it from the popular-sweep cache would report that a model the user
    // named does not exist.
    if let Some(term) = search {
        let emitter = app.clone();
        let report = move |p: SweepProgress| {
            let _ = emitter.emit(PROGRESS_EVENT, ProgressPayload::from_sweep(p, false));
        };

        let repos = live_catalog::discover_repos_reporting(Some(term), 1, token.as_deref(), Some(&report))
            .await
            .map_err(|e| e.to_string())?;

        let _ = app.emit(
            PROGRESS_EVENT,
            ProgressPayload::simple("done", format!("{} results", repos.len()), false),
        );
        return Ok(page_from(&repos, CatalogSource::Live, None, false, None));
    }

    let authenticated = token.is_some();
    let notice = anonymous_notice(authenticated, false);

    // Only ever filled by a sweep in this run, so it is by construction recent.
    if let Some(cached) = cached_repos(authenticated) {
        log::debug!("[CATALOG] Serving {} repositories from the session cache", cached.len());
        return Ok(page_from(&cached, CatalogSource::Session, None, false, notice));
    }

    // The stored library, which is what makes a launch instant.
    if let Some(dir) = data_dir(&app) {
        if let Some(stored) = catalog_cache::load(&dir) {
            let now = chrono::Utc::now();
            let freshness = stored.freshness_at(authenticated, now);
            let age = stored.age_at(now).map(|d| d.num_seconds());

            if freshness != Freshness::Expired {
                let host = host_capacity();
                let moved = stored.hardware_changed(&catalog_cache::fingerprint(
                    host.vram_total_bytes,
                    host.usable_ram_bytes,
                ));

                // Hardware moving does not make the stored repositories wrong —
                // cards are rebuilt against the current machine either way — but
                // it does mean a sweep filtered for the old machine may be
                // missing models this one can run.
                let refreshing = freshness == Freshness::Stale || moved;
                if moved {
                    log::info!(
                        "[CATALOG] This machine's memory differs from the stored library's; refreshing"
                    );
                }

                log::info!(
                    "[CATALOG] Serving {} repositories from the stored library ({:?}, {}s old)",
                    stored.repos.len(),
                    freshness,
                    age.unwrap_or(0)
                );

                // Deliberately *not* promoted into the session cache. That cache
                // is timed from when it was filled, so copying a day-old library
                // into it would make the next visit report a fresh sweep and
                // hide the "checking for updates" line. Re-reading a local JSON
                // file costs milliseconds; the cache exists to avoid network
                // sweeps, not file reads. It also means the next read picks up
                // whatever a background refresh has just written.
                if refreshing {
                    spawn_refresh(&app, token.clone());
                }

                return Ok(page_from(&stored.repos, CatalogSource::Stored, age, refreshing, notice));
            }

            log::info!("[CATALOG] The stored library is no longer usable ({freshness:?}); sweeping");
        }
    }

    let _guard = browse_lock().lock().await;

    // Re-check: another caller may have finished the same sweep while this one
    // waited for the lock.
    if let Some(cached) = cached_repos(authenticated) {
        return Ok(page_from(&cached, CatalogSource::Session, None, false, notice));
    }

    let repos = sweep(&app, token.as_deref(), false).await?;
    Ok(page_from(&repos, CatalogSource::Live, Some(0), false, notice))
}

/// Discards every cached library and sweeps again.
///
/// The escape hatch for "I know there is something newer than this". Everything
/// else in this module prefers showing a cached answer.
#[tauri::command]
pub async fn refresh_model_library(app: AppHandle) -> Result<CatalogPage, String> {
    invalidate_browse_cache(data_dir(&app).as_deref());

    let token = crate::config::hf_token::get();
    let _guard = browse_lock().lock().await;

    let repos = sweep(&app, token.as_deref(), false).await?;
    Ok(page_from(
        &repos,
        CatalogSource::Live,
        Some(0),
        false,
        anonymous_notice(token.is_some(), false),
    ))
}

/// Adapters published for a model, plus how usable they are here.
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AdapterPage {
    pub adapters: Vec<live_catalog::AdapterListing>,
    /// How many can be loaded as they are. The rest are PEFT safetensors, which
    /// Sarathi converts to GGUF during installation.
    pub ready_count: usize,
    /// Shown when none are directly usable, explaining what that means.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notice: Option<String>,
}

/// Lists LoRA adapters published for a base model.
///
/// Looks up HuggingFace's `base_model:adapter:` tag, which adapter authors set
/// to declare their parent — a real relationship, not a name-similarity guess.
#[tauri::command]
pub async fn find_model_adapters(base_model_id: String) -> Result<AdapterPage, String> {
    let token = crate::config::hf_token::get();

    let adapters = live_catalog::find_adapters(&base_model_id, 20, token.as_deref())
        .await
        .map_err(|e| e.to_string())?;

    let ready_count = adapters.iter().filter(|a| a.gguf_ready).count();

    let notice = if adapters.is_empty() {
        Some("No LoRA adapters published for this model yet.".to_string())
    } else if ready_count == 0 {
        Some(format!(
            "{} adapter(s) found. None ship GGUF, so Sarathi converts them during install — \
             which needs this base model installed and its family supported.",
            adapters.len()
        ))
    } else {
        None
    };

    Ok(AdapterPage { adapters, ready_count, notice })
}

/// Every category the app knows about, for a sidebar that stays stable while
/// results load.
#[tauri::command]
pub fn list_model_categories() -> Vec<CategoryCount> {
    ModelCategory::all()
        .iter()
        .map(|c| CategoryCount { category: *c, label: c.label().to_string(), count: 0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system_analyzer::traits::GpuInfo;

    fn gpu(model: &str, dedicated: bool, total: u64, cuda: bool) -> GpuInfo {
        GpuInfo {
            vendor: String::new(),
            model: model.to_string(),
            gpu_type: String::new(),
            is_dedicated: dedicated,
            dedicated_video_memory_bytes: if dedicated { total } else { 0 },
            dedicated_system_memory_bytes: 0,
            shared_system_memory_bytes: if dedicated { 0 } else { total },
            total_available_graphics_memory_bytes: total,
            vram_total_bytes: total,
            vram_free_bytes: total,
            driver_version: None,
            vendor_id: None,
            device_id: None,
            compute_capability: None,
            cuda_supported: cuda,
            rocm_supported: false,
            directx_supported: true,
            vulkan_supported: true,
            opencl_supported: true,
            detection_source: "test".into(),
            confidence: "High".into(),
        }
    }

    #[test]
    fn an_integrated_gpus_shared_memory_never_sets_the_budget() {
        // This machine exactly: a Radeon 780M reporting 13 GB of shared system
        // memory next to an RTX 5060 with 8.28 GB of its own.
        //
        // Taking the larger figure produced a ~9.5 GB budget, which marked a
        // 9.1 GB Q6_K as fitting on a card that holds 8.28 GB.
        let gpus = vec![
            gpu("AMD Radeon 780M", false, 13_050_000_000, false),
            gpu("NVIDIA GeForce RTX 5060 Laptop GPU", true, 8_280_000_000, true),
        ];

        let budget = weight_budget_for(&gpus);

        assert!(budget < 8_280_000_000, "budget cannot exceed the card's own VRAM");
        assert!(
            budget < 9_100_000_000,
            "a 9.1 GB Q6_K must not be marked as fitting; budget was {budget}"
        );
        // Roughly 5.5 GB once the KV cache and compute buffers are charged.
        assert!((5_000_000_000..6_200_000_000).contains(&budget), "unexpected budget {budget}");
    }

    #[test]
    fn the_browse_budget_is_not_more_generous_than_the_loader() {
        // These two answer the same question and used to disagree: the listing
        // marked a 6.09 GB Q3_K_M as fitting, then the loader refused to fully
        // offload it because its budget also carried the KV cache.
        let gpus = vec![gpu("NVIDIA GeForce RTX 5060 Laptop GPU", true, 8_280_000_000, true)];
        let browse = weight_budget_for(&gpus);

        let plan_budget = {
            const OS: u64 = 900 * 1024 * 1024;
            let usable = 8_280_000_000u64.saturating_sub(OS);
            let kv = crate::ai_engine::vram_planner::estimate_kv_bytes_per_token(4 * 1024 * 1024 * 1024)
                * u64::from(BROWSE_CONTEXT);
            let after_kv = usable.saturating_sub(kv);
            after_kv.saturating_sub((after_kv as f64 * 0.12) as u64)
        };

        assert!(
            browse <= plan_budget,
            "browse budget {browse} must not exceed the loader's {plan_budget}"
        );
    }

    #[test]
    fn no_usable_gpu_means_nothing_is_marked_as_fitting() {
        // Zero is the signal for "unknown", and `to_option` marks nothing as
        // fitting rather than guessing that everything does.
        assert_eq!(weight_budget_for(&[]), 0);
    }

    #[test]
    fn an_integrated_gpu_still_sets_the_budget_when_it_is_all_there_is() {
        let gpus = vec![gpu("AMD Radeon 780M", false, 13_050_000_000, false)];
        assert!(weight_budget_for(&gpus) > 0);
    }

    // ─── Progress reporting ─────────────────────────────────────────────────

    /// The number of repositories a sweep will find is not known until the
    /// search pages are in, so a fraction during search would be invented — and
    /// a bar that jumps backwards is worse than one that waits.
    #[test]
    fn searching_reports_no_fraction_but_still_says_something() {
        let p = ProgressPayload::from_sweep(
            SweepProgress::Searching { page: 3, pages: 20, found: 240 },
            false,
        );

        assert_eq!(p.phase, "searching");
        assert!(p.fraction.is_none());
        assert!(p.message.contains("3 of 20"), "message: {}", p.message);
        assert!(p.message.contains("240"), "message: {}", p.message);
    }

    #[test]
    fn fetching_reports_a_real_fraction_and_real_counts() {
        let p =
            ProgressPayload::from_sweep(SweepProgress::Fetching { done: 250, total: 1000 }, false);

        assert_eq!(p.phase, "fetching");
        assert_eq!(p.fraction, Some(0.25));
        assert_eq!((p.done, p.total), (250, 1000));
        assert!(p.message.contains("250 of 1000"), "message: {}", p.message);
    }

    /// A sweep that found nothing must not divide by zero.
    #[test]
    fn an_empty_sweep_reports_no_fraction_rather_than_dividing_by_zero() {
        let p = ProgressPayload::from_sweep(SweepProgress::Fetching { done: 0, total: 0 }, false);
        assert!(p.fraction.is_none());
    }

    /// A background refresh must be distinguishable, or the UI covers a working
    /// listing with a loading state the user did not ask for.
    #[test]
    fn background_refreshes_are_labelled_as_such() {
        let fg = ProgressPayload::from_sweep(SweepProgress::Fetching { done: 1, total: 2 }, false);
        let bg = ProgressPayload::from_sweep(SweepProgress::Fetching { done: 1, total: 2 }, true);

        assert!(!fg.background);
        assert!(bg.background);
    }

    #[test]
    fn the_token_notice_appears_only_for_an_anonymous_listing() {
        assert!(anonymous_notice(false, false).is_some());
        assert!(anonymous_notice(true, false).is_none(), "a token holder needs no advice");
        assert!(
            anonymous_notice(false, true).is_none(),
            "a search already reaches the whole Hub, so the notice would be wrong"
        );
    }
}
