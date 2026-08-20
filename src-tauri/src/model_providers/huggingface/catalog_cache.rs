//! The model library, kept on disk between runs.
//!
//! A full sweep is one search request per page plus a detail request per
//! repository — around 2,000 calls with a token. The in-process cache in
//! [`crate::commands::catalog`] made that once per ten minutes instead of once
//! per visit, but it dies with the process: every launch paid for the whole
//! sweep again, and the user watched a spinner for the length of it.
//!
//! This stores the swept repositories so the next launch can show the library
//! immediately and check the Hub behind it.
//!
//! ## What is stored, and what is not
//!
//! Only [`GgufRepo`] records — what the Hub said. Nothing derived from the
//! machine is written down: which quantizations fit, which MoE models can be
//! offloaded here, and what a card looks like are all recomputed from the live
//! hardware profile every time the cache is read. A cache carried to a machine
//! with a different GPU therefore cannot claim a model runs there.
//!
//! The hardware fingerprint is recorded anyway, for one narrow purpose: to
//! notice that the machine has changed and say so. It is never used to decide
//! whether a model fits.
//!
//! ## Two ages, not one
//!
//! [`FRESH_FOR`] is how long a cache is served without doing anything else.
//! Past it the cache is still served — instantly, because a listing that is a
//! day old beats a spinner — and a refresh runs behind it. [`USABLE_FOR`] is
//! the point where the contents are too old to show at all and the sweep is
//! awaited.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::model_providers::huggingface::discovery::GgufRepo;

/// File the swept library is stored in.
pub const CACHE_FILE: &str = "model-library.json";

/// Bumped when the stored shape changes. An older file is discarded rather than
/// coerced — a partial read of a renamed field is worse than one slow sweep.
const CACHE_VERSION: u32 = 1;

/// How long the cache is served without checking the Hub.
///
/// Model repositories are updated over days, not minutes. An hour is short
/// enough that a newly published model appears the same day someone looks for
/// it, and long enough that opening the browser a few times in an afternoon
/// costs nothing.
pub const FRESH_FOR: chrono::Duration = chrono::Duration::hours(1);

/// How long the cache can still be shown while a refresh runs behind it.
///
/// Past this the listing would be misleading rather than merely stale, so the
/// sweep is awaited instead.
pub const USABLE_FOR: chrono::Duration = chrono::Duration::days(7);

/// Ceiling on how many repositories a merged library may hold.
///
/// [`merge`] carries over entries a sweep did not reach, which is what stops a
/// rate-limited refresh from destroying the library. Left unbounded that grows:
/// the Hub reorders by popularity over weeks, so successive sweeps return
/// overlapping but not identical sets, and the union creeps upwards forever.
///
/// A full authenticated sweep considers 2,000 candidates
/// ([`AUTHENTICATED_PAGES`](super::live_catalog::AUTHENTICATED_PAGES) at 100 per
/// page) and keeps rather fewer, since most repositories carry no usable GGUF.
/// Double that leaves room for genuine carry-over while keeping the file, and
/// the card-building pass that reads it, bounded.
///
/// Trimming takes from the tail, which is the oldest carried-over end: the
/// current sweep is written first and is never what gets dropped.
pub const MAX_MERGED_REPOS: usize = 4_000;

/// A swept library as it sits on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CachedLibrary {
    pub version: u32,
    /// When the sweep completed, RFC 3339.
    pub fetched_at: String,
    /// Whether a token was in use. A sweep reaches one search page anonymously
    /// against twenty with a token, so the two are not interchangeable results
    /// and an anonymous cache must not be served as if it were the full library.
    pub authenticated: bool,
    /// The machine the sweep was displayed on. Advisory only — see the module
    /// docs. Absent in caches written before fingerprinting existed.
    #[serde(default)]
    pub hardware_fingerprint: Option<String>,
    pub repos: Vec<GgufRepo>,
}

/// How usable a cache is right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Freshness {
    /// Serve it and do nothing else.
    Fresh,
    /// Serve it immediately, then refresh behind the user.
    Stale,
    /// Too old, or from a different authentication state. Sweep first.
    Expired,
}

impl CachedLibrary {
    /// Age against a clock, so this can be tested without waiting.
    pub fn freshness_at(
        &self,
        authenticated: bool,
        now: chrono::DateTime<chrono::Utc>,
    ) -> Freshness {
        // An anonymous sweep is a different, much smaller result than an
        // authenticated one. Serving one for the other showed a 182-model
        // listing beneath a notice offering to unlock the full library.
        if self.authenticated != authenticated {
            return Freshness::Expired;
        }
        if self.repos.is_empty() {
            return Freshness::Expired;
        }

        let Some(age) = self.age_at(now) else {
            // An unreadable timestamp cannot be trusted to be recent.
            return Freshness::Expired;
        };

        if age < FRESH_FOR {
            Freshness::Fresh
        } else if age < USABLE_FOR {
            Freshness::Stale
        } else {
            Freshness::Expired
        }
    }

    /// How old this cache is, or `None` if the stamp cannot be read.
    ///
    /// A negative age — a file written by a clock ahead of this one — is
    /// reported as zero rather than as an error, since the contents are
    /// certainly not stale.
    pub fn age_at(&self, now: chrono::DateTime<chrono::Utc>) -> Option<chrono::Duration> {
        let then = chrono::DateTime::parse_from_rfc3339(&self.fetched_at).ok()?;
        let age = now.signed_duration_since(then.with_timezone(&chrono::Utc));
        Some(age.max(chrono::Duration::zero()))
    }

    /// Whether this cache was written on hardware other than the current.
    ///
    /// Cards are rebuilt against live hardware regardless, so this changes
    /// nothing about correctness — it only earns a refresh, because a machine
    /// with a bigger card may now be able to run models the previous sweep
    /// filtered out for being unreachable.
    pub fn hardware_changed(&self, fingerprint: &str) -> bool {
        self.hardware_fingerprint.as_deref().is_some_and(|f| f != fingerprint)
    }
}

pub fn cache_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join(CACHE_FILE)
}

/// Reads the cached library, or `None` when there is nothing usable.
///
/// Every failure is non-fatal and logged: a missing, unreadable, malformed or
/// outdated file simply means a sweep is needed, which is the state the app was
/// always in before this existed.
pub fn load(app_data_dir: &Path) -> Option<CachedLibrary> {
    let path = cache_path(app_data_dir);
    if !path.is_file() {
        return None;
    }

    let raw = match std::fs::read_to_string(&path) {
        Ok(r) => r,
        Err(e) => {
            log::warn!("[CATALOG_CACHE] Could not read {}: {e}", path.display());
            return None;
        }
    };

    match serde_json::from_str::<CachedLibrary>(&raw) {
        Ok(cache) if cache.version == CACHE_VERSION => Some(cache),
        Ok(cache) => {
            log::info!(
                "[CATALOG_CACHE] Discarding a version {} cache; this build writes version {CACHE_VERSION}",
                cache.version
            );
            None
        }
        Err(e) => {
            log::warn!("[CATALOG_CACHE] {} is not readable ({e}); re-sweeping", path.display());
            None
        }
    }
}

/// Folds a sweep into whatever is already stored, and returns the union.
///
/// A sweep is not always the whole library. When HuggingFace rate-limits a
/// refresh part-way through, [`live_catalog`] deliberately keeps what it has and
/// stops — so a background refresh can come back with forty repositories where
/// the stored library holds four hundred. Writing that directly would replace a
/// complete library with a truncated one and call it an update, which is the
/// one thing a refresh must never do.
///
/// Merging makes a partial sweep additive instead of destructive: every
/// repository the sweep saw is taken at its new value, and every repository it
/// did not reach keeps the value it already had. A full sweep is unaffected —
/// it covers everything in the cache, so the union is just the sweep.
///
/// The cost of this is that a repository withdrawn from the Hub lingers until a
/// cache is cleared outright, which [`clear`] does when the token changes.
/// Showing one model that has since disappeared is a far smaller harm than
/// losing nine tenths of the library to a rate limit.
///
/// Ordering follows the sweep, because that is the Hub's own popularity
/// ordering and the browser presents results in the order they arrive. Entries
/// held over from the previous cache follow, keeping their relative order.
///
/// Takes the previous repositories directly rather than reading the disk, so
/// the merge can be tested without a filesystem.
pub fn merge(previous: &[GgufRepo], sweep: &[GgufRepo]) -> Vec<GgufRepo> {
    use std::collections::HashSet;

    let swept: HashSet<&str> = sweep.iter().map(|r| r.repo_id.as_str()).collect();

    let mut merged = Vec::with_capacity(previous.len() + sweep.len());
    merged.extend(sweep.iter().cloned());
    // Only those the sweep never reached; anything it did see is already
    // present at its newer value.
    merged.extend(
        previous
            .iter()
            .filter(|r| !swept.contains(r.repo_id.as_str()))
            .cloned(),
    );

    // From the tail, so a sweep's own results are never the ones discarded.
    merged.truncate(MAX_MERGED_REPOS);
    merged
}

/// Writes a completed sweep.
///
/// Written to a temporary file and renamed, so a crash or a second window
/// writing at the same moment cannot leave a half-written cache that the next
/// launch has to discard.
pub fn store(
    app_data_dir: &Path,
    authenticated: bool,
    hardware_fingerprint: &str,
    repos: &[GgufRepo],
) -> Result<(), String> {
    // An empty sweep is a failed sweep — rate limiting, or no network. Writing
    // it would replace a good cache with an empty one and make the next launch
    // show nothing.
    if repos.is_empty() {
        return Err("refusing to cache an empty sweep".into());
    }

    let cache = CachedLibrary {
        version: CACHE_VERSION,
        fetched_at: chrono::Utc::now().to_rfc3339(),
        authenticated,
        hardware_fingerprint: Some(hardware_fingerprint.to_string()),
        repos: repos.to_vec(),
    };

    std::fs::create_dir_all(app_data_dir)
        .map_err(|e| format!("could not prepare {}: {e}", app_data_dir.display()))?;

    let body = serde_json::to_string(&cache).map_err(|e| format!("could not encode: {e}"))?;
    let final_path = cache_path(app_data_dir);
    let temp_path = final_path.with_extension("json.tmp");

    std::fs::write(&temp_path, body)
        .map_err(|e| format!("could not write {}: {e}", temp_path.display()))?;
    std::fs::rename(&temp_path, &final_path).map_err(|e| {
        let _ = std::fs::remove_file(&temp_path);
        format!("could not replace {}: {e}", final_path.display())
    })?;

    log::info!(
        "[CATALOG_CACHE] Stored {} repositories to {}",
        repos.len(),
        final_path.display()
    );
    Ok(())
}

/// Deletes the cache. Used when the token changes, since the stored sweep
/// describes what the previous credentials could reach.
pub fn clear(app_data_dir: &Path) {
    let path = cache_path(app_data_dir);
    match std::fs::remove_file(&path) {
        Ok(()) => log::info!("[CATALOG_CACHE] Cleared {}", path.display()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => log::warn!("[CATALOG_CACHE] Could not clear {}: {e}", path.display()),
    }
}

/// A short, stable description of the memory this machine can offer a model.
///
/// Only the figures that change what is runnable: the inference GPU's VRAM and
/// the RAM available for offloaded experts. Deliberately not a full hardware
/// hash — a driver update or a different monitor must not invalidate the
/// library.
pub fn fingerprint(vram_total_bytes: u64, usable_ram_bytes: u64) -> String {
    // Rounded to whole gigabytes so the ordinary drift in reported free memory
    // does not read as a new machine on every launch.
    format!(
        "vram{}gb-ram{}gb",
        vram_total_bytes / (1024 * 1024 * 1024),
        usable_ram_bytes / (1024 * 1024 * 1024)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_providers::huggingface::discovery::{GgufMeta, Quantization};

    const GB: u64 = 1024 * 1024 * 1024;

    fn now() -> chrono::DateTime<chrono::Utc> {
        chrono::DateTime::parse_from_rfc3339("2026-08-11T12:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc)
    }

    fn repo(id: &str) -> GgufRepo {
        GgufRepo {
            repo_id: id.into(),
            author: "someone".into(),
            downloads: 1000,
            likes: 10,
            last_modified: "2026-07-01T00:00:00Z".into(),
            quantizations: vec![Quantization {
                label: "Q4_K_M".into(),
                filename: "m.gguf".into(),
                size_bytes: 4_000_000_000,
                is_sharded: false,
            }],
            gguf: Some(GgufMeta {
                total_parameters: 7_600_000_000,
                architecture: "qwen2".into(),
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

    fn cache_aged(hours: i64, authenticated: bool) -> CachedLibrary {
        CachedLibrary {
            version: CACHE_VERSION,
            fetched_at: (now() - chrono::Duration::hours(hours)).to_rfc3339(),
            authenticated,
            hardware_fingerprint: Some(fingerprint(8 * GB, 24 * GB)),
            repos: vec![repo("a/b")],
        }
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sarathi_catalog_cache_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_recent_cache_is_served_without_checking_the_hub() {
        assert_eq!(cache_aged(0, true).freshness_at(true, now()), Freshness::Fresh);
    }

    /// The behaviour the whole feature is for: an old cache is still shown at
    /// once, and the network happens behind it.
    #[test]
    fn a_day_old_cache_is_still_shown_while_it_refreshes() {
        assert_eq!(cache_aged(24, true).freshness_at(true, now()), Freshness::Stale);
    }

    #[test]
    fn a_cache_older_than_a_week_is_not_shown_at_all() {
        assert_eq!(cache_aged(24 * 8, true).freshness_at(true, now()), Freshness::Expired);
    }

    /// An anonymous sweep reaches one search page; an authenticated one reaches
    /// twenty. Serving one as the other showed the full library beneath a
    /// notice offering to unlock it.
    #[test]
    fn a_cache_from_a_different_authentication_state_is_not_reused() {
        assert_eq!(cache_aged(0, false).freshness_at(true, now()), Freshness::Expired);
        assert_eq!(cache_aged(0, true).freshness_at(false, now()), Freshness::Expired);
    }

    #[test]
    fn an_empty_or_undated_cache_is_expired_rather_than_trusted() {
        let mut empty = cache_aged(0, true);
        empty.repos.clear();
        assert_eq!(empty.freshness_at(true, now()), Freshness::Expired);

        let mut undated = cache_aged(0, true);
        undated.fetched_at = "not a date".into();
        assert_eq!(undated.freshness_at(true, now()), Freshness::Expired);
    }

    /// A machine whose clock is ahead writes a cache dated in the future. That
    /// is certainly not stale, and must not read as a negative age.
    #[test]
    fn a_cache_from_the_future_is_treated_as_new() {
        let mut ahead = cache_aged(0, true);
        ahead.fetched_at = (now() + chrono::Duration::hours(3)).to_rfc3339();

        assert_eq!(ahead.age_at(now()), Some(chrono::Duration::zero()));
        assert_eq!(ahead.freshness_at(true, now()), Freshness::Fresh);
    }

    #[test]
    fn a_cache_round_trips_through_disk() {
        let dir = temp_dir("roundtrip");
        let fp = fingerprint(8 * GB, 24 * GB);

        store(&dir, true, &fp, &[repo("a/b"), repo("c/d")]).expect("should write");
        let loaded = load(&dir).expect("should read back");

        assert_eq!(loaded.repos.len(), 2);
        assert!(loaded.authenticated);
        assert_eq!(loaded.hardware_fingerprint.as_deref(), Some(fp.as_str()));
        assert_eq!(loaded.freshness_at(true, chrono::Utc::now()), Freshness::Fresh);
    }

    /// A rate-limited sweep returns nothing. Writing that would replace a good
    /// library with an empty one and make the next launch show a blank page.
    #[test]
    fn an_empty_sweep_is_never_written_over_a_good_cache() {
        let dir = temp_dir("empty");
        let fp = fingerprint(8 * GB, 24 * GB);
        store(&dir, true, &fp, &[repo("a/b")]).unwrap();

        assert!(store(&dir, true, &fp, &[]).is_err());
        assert_eq!(load(&dir).expect("the good cache survives").repos.len(), 1);
    }

    /// The defect this exists for: HuggingFace rate-limits a background refresh
    /// part-way through, the sweep keeps what it reached, and writing that
    /// directly replaced a complete library with a fraction of one.
    #[test]
    fn a_partial_sweep_does_not_drop_what_it_never_reached() {
        let previous = vec![repo("a/b"), repo("c/d"), repo("e/f")];
        let partial = vec![repo("a/b")];

        let merged = merge(&previous, &partial);

        assert_eq!(merged.len(), 3, "the two the sweep never reached survive");
        let ids: Vec<&str> = merged.iter().map(|r| r.repo_id.as_str()).collect();
        assert_eq!(ids, vec!["a/b", "c/d", "e/f"]);
    }

    /// A repository the sweep *did* reach must take its new value, otherwise a
    /// refresh could never update anything.
    #[test]
    fn a_swept_repository_wins_over_the_stored_copy() {
        let mut stale = repo("a/b");
        stale.downloads = 1;
        let mut fresh = repo("a/b");
        fresh.downloads = 9_999;

        let merged = merge(&[stale], &[fresh]);

        assert_eq!(merged.len(), 1, "the same repo must not appear twice");
        assert_eq!(merged[0].downloads, 9_999, "the sweep is the newer truth");
    }

    /// A sweep that covers everything is a plain replacement, and must not be
    /// made to grow by merging.
    #[test]
    fn a_full_sweep_merges_to_exactly_itself() {
        let previous = vec![repo("a/b"), repo("c/d")];
        let full = vec![repo("a/b"), repo("c/d")];

        assert_eq!(merge(&previous, &full).len(), 2);
    }

    #[test]
    fn a_sweep_that_finds_something_new_keeps_it_first() {
        let merged = merge(&[repo("old/one")], &[repo("new/one")]);

        let ids: Vec<&str> = merged.iter().map(|r| r.repo_id.as_str()).collect();
        assert_eq!(ids, vec!["new/one", "old/one"], "sweep order leads");
    }

    /// Carrying entries over must not let the library grow without limit: the
    /// Hub reorders over weeks, so successive sweeps overlap without matching,
    /// and an uncapped union creeps upwards on every refresh.
    #[test]
    fn a_merged_library_is_capped_and_drops_the_carried_over_end_first() {
        let previous: Vec<GgufRepo> =
            (0..MAX_MERGED_REPOS).map(|i| repo(&format!("old/{i}"))).collect();
        let sweep = vec![repo("new/one"), repo("new/two")];

        let merged = merge(&previous, &sweep);

        assert_eq!(merged.len(), MAX_MERGED_REPOS, "the cap holds");
        assert_eq!(merged[0].repo_id, "new/one", "the sweep survives at the front");
        assert_eq!(merged[1].repo_id, "new/two");
        assert!(
            !merged.iter().any(|r| r.repo_id == format!("old/{}", MAX_MERGED_REPOS - 1)),
            "the oldest carried-over entries are what gets trimmed"
        );
    }

    #[test]
    fn merging_against_an_empty_cache_is_just_the_sweep() {
        assert_eq!(merge(&[], &[repo("a/b")]).len(), 1);
    }

    #[test]
    fn a_missing_cache_is_normal() {
        assert!(load(&temp_dir("missing")).is_none());
    }

    #[test]
    fn a_malformed_cache_is_discarded_rather_than_fatal() {
        let dir = temp_dir("malformed");
        std::fs::write(cache_path(&dir), "{ not json").unwrap();
        assert!(load(&dir).is_none());
    }

    #[test]
    fn a_cache_from_an_older_version_is_discarded() {
        let dir = temp_dir("oldversion");
        let mut old = cache_aged(0, true);
        old.version = CACHE_VERSION - 1;
        std::fs::write(cache_path(&dir), serde_json::to_string(&old).unwrap()).unwrap();

        assert!(load(&dir).is_none());
    }

    #[test]
    fn clearing_removes_the_file_and_is_safe_to_repeat() {
        let dir = temp_dir("clear");
        store(&dir, true, &fingerprint(8 * GB, 24 * GB), &[repo("a/b")]).unwrap();

        clear(&dir);
        assert!(load(&dir).is_none());
        clear(&dir); // must not panic on a file that is already gone
    }

    #[test]
    fn a_new_gpu_is_noticed_but_a_trivial_change_is_not() {
        let cache = cache_aged(0, true);

        assert!(cache.hardware_changed(&fingerprint(24 * GB, 24 * GB)), "a bigger card matters");
        assert!(
            !cache.hardware_changed(&fingerprint(8 * GB, 24 * GB)),
            "the same machine is the same machine"
        );
        // Free RAM drifts constantly; rounding to whole GB keeps that quiet.
        assert!(!cache.hardware_changed(&fingerprint(
            8 * GB + 300_000_000,
            24 * GB + 500_000_000
        )));
    }

    /// Cards are rebuilt against live hardware on every read, so a cache from
    /// another machine is safe to serve — it just earns a refresh.
    #[test]
    fn hardware_change_does_not_by_itself_expire_a_cache() {
        assert_eq!(cache_aged(0, true).freshness_at(true, now()), Freshness::Fresh);
    }

    #[test]
    fn a_cache_written_before_fingerprinting_still_loads() {
        let dir = temp_dir("nofingerprint");
        let json = serde_json::json!({
            "version": CACHE_VERSION,
            "fetchedAt": now().to_rfc3339(),
            "authenticated": true,
            "repos": [],
        });
        std::fs::write(cache_path(&dir), json.to_string()).unwrap();

        let loaded = load(&dir).expect("a missing fingerprint is not a parse failure");
        assert!(loaded.hardware_fingerprint.is_none());
        assert!(!loaded.hardware_changed("vram8gb-ram24gb"), "unknown is not 'changed'");
    }
}
