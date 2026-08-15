//! One reader for what is on disk, shared by everything that asks.
//!
//! The Storage screen issues `get_installed_models`, `get_storage_summary` and
//! `get_inference_status` together. The first two both used to walk the whole
//! model store and read every GGUF header, so a single refresh paid for the scan
//! twice — and did it on the thread that draws the window.
//!
//! This owns the scan instead:
//!
//! * The **directory walk is always live**. Adding or deleting a model is seen
//!   on the next call with no cache to invalidate, which is what makes the
//!   caching below safe to be aggressive about.
//! * The **header read is memoised** per file, keyed by the path, length and
//!   modification time. That is the expensive part — 96% of the scan — and it
//!   cannot change without one of those three changing with it.
//! * **Concurrent callers share one scan.** A caller that was waiting when a
//!   scan finished takes that scan's answer instead of starting its own, so the
//!   screen's two commands cost one walk however they interleave.
//!
//! Nothing here may run on the UI thread; [`scan_blocking`] says so out loud.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime};

use sysinfo::Disks;

use crate::adapter_manager::AdapterRegistry;
use crate::diagnostics::{assert_off_ui_thread, Stage};
use crate::download_manager::traits::{InstalledModel, StorageSummary};
use crate::model_manager::classify::{classify, Classification};

/// Identifies a file precisely enough that a changed file is never mistaken for
/// the one that was read before.
///
/// Length and modification time together: a rewrite that preserves the length
/// still moves the timestamp, and a truncation still changes the length.
///
/// The timestamp is kept in **nanoseconds**, not seconds. A re-download that
/// finishes within the same second as the previous scan is an ordinary thing to
/// do, and a second-resolution key answers it from the stale entry — NTFS
/// records to 100 ns, so there is no reason to throw that away. A filesystem
/// that reports no mtime gives `None`, which never matches: the header is then
/// read every time, correct if slower.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FileKey {
    path: PathBuf,
    len: u64,
    modified_nanos: Option<u128>,
}

impl FileKey {
    fn of(path: &Path) -> Option<Self> {
        let meta = std::fs::metadata(path).ok()?;
        Some(Self {
            path: path.to_path_buf(),
            len: meta.len(),
            modified_nanos: meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                .map(|d| d.as_nanos()),
        })
    }
}

/// The completed listing, and when the scan that produced it finished.
struct Completed {
    at: Instant,
    models: Arc<Vec<InstalledModel>>,
}

#[derive(Default)]
pub struct ModelStore {
    /// Header classification per file. Never cleared on its own: a stale entry
    /// is unreachable, because its key no longer matches the file on disk.
    headers: Mutex<HashMap<FileKey, Classification>>,
    /// The last finished scan, offered to anyone who was already waiting.
    last: Mutex<Option<Completed>>,
    /// Serialises scans. Async so a waiting caller yields its worker rather
    /// than parking a thread on a lock it will not need.
    scanning: tokio::sync::Mutex<()>,
}

impl ModelStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Every installed model, scanned off the calling thread.
    ///
    /// Concurrent callers collapse onto one scan: whoever gets the lock does the
    /// work, and anyone who was already waiting when it finished takes its
    /// result rather than repeating it.
    pub async fn listing(self: &Arc<Self>, app_data_dir: &Path) -> Arc<Vec<InstalledModel>> {
        let asked_at = Instant::now();
        let _turn = self.scanning.lock().await;

        // Someone else scanned while this call was waiting. Their answer is
        // newer than the question, so it answers it.
        if let Some(done) = self.last.lock().unwrap().as_ref() {
            if done.at >= asked_at {
                log::debug!("[STORAGE] Reusing a scan that finished while this call waited");
                return done.models.clone();
            }
        }

        let store = self.clone();
        let dir = app_data_dir.to_path_buf();
        let models = tokio::task::spawn_blocking(move || store.scan_blocking(&dir))
            .await
            .unwrap_or_else(|e| {
                log::error!("[STORAGE] Scan task failed: {e}");
                Vec::new()
            });

        let models = Arc::new(models);
        *self.last.lock().unwrap() =
            Some(Completed { at: Instant::now(), models: models.clone() });
        models
    }

    /// Disk usage, from the same scan the listing uses.
    pub async fn summary(self: &Arc<Self>, app_data_dir: &Path) -> StorageSummary {
        let installed = self.listing(app_data_dir).await;
        let total_models_bytes = installed.iter().map(|m| m.size_bytes).sum();
        let models_dir = app_data_dir.join("models");

        // Volume enumeration talks to every mounted device, including ones that
        // are asleep or on the network, so it is never done on the caller's
        // thread either.
        let dir = models_dir.clone();
        let (available_disk_space_bytes, total_disk_space_bytes) =
            tokio::task::spawn_blocking(move || disk_space_for(&dir)).await.unwrap_or((0, 0));

        StorageSummary {
            models_directory: models_dir.to_string_lossy().to_string(),
            total_installed_models: installed.len(),
            total_models_bytes,
            available_disk_space_bytes,
            total_disk_space_bytes,
        }
    }

    /// Drops the shared result so the next caller scans afresh.
    ///
    /// Called after a delete or a finished download — changes Sarathi made
    /// itself, which should be visible immediately rather than on the next
    /// natural refresh. The header cache is deliberately kept: those entries are
    /// keyed by file identity, so they are still correct for whatever remains.
    pub fn invalidate(&self) {
        *self.last.lock().unwrap() = None;
    }

    /// One scan, uncached and unshared, on the calling thread.
    ///
    /// For callers outside the running app — tests and the headless verification
    /// binaries — which have no `ModelStore` to share and no UI to protect.
    /// Inside the app, use [`listing`](Self::listing).
    pub fn scan_now(app_data_dir: &Path) -> Vec<InstalledModel> {
        Self::new().scan_blocking(app_data_dir)
    }

    /// Walks the store and builds the listing. Blocking, by design.
    fn scan_blocking(&self, app_data_dir: &Path) -> Vec<InstalledModel> {
        assert_off_ui_thread("storage scan");
        let _stage = Stage::new("storage: full scan");

        let mut installed = Vec::new();
        let models_dir = app_data_dir.join("models");
        if !models_dir.is_dir() {
            return installed;
        }

        let providers = match std::fs::read_dir(&models_dir) {
            Ok(p) => p,
            Err(e) => {
                log::warn!("[STORAGE] Could not read {}: {e}", models_dir.display());
                return installed;
            }
        };

        for provider_entry in providers.flatten() {
            let provider_path = provider_entry.path();
            if !provider_path.is_dir() {
                continue;
            }
            let provider_id =
                provider_path.file_name().unwrap_or_default().to_string_lossy().to_string();

            let Ok(packages) = std::fs::read_dir(&provider_path) else { continue };
            for pkg_entry in packages.flatten() {
                let pkg_path = pkg_entry.path();
                if !pkg_path.is_dir() {
                    continue;
                }
                if let Some(model) = self.read_package(&pkg_path, &provider_id) {
                    installed.push(model);
                }
            }
        }

        log::debug!("[STORAGE] Scan found {} installed model(s)", installed.len());
        installed
    }

    /// One package directory, or `None` when it holds no loadable weight file.
    fn read_package(&self, pkg_path: &Path, provider_id: &str) -> Option<InstalledModel> {
        let folder = pkg_path.file_name().unwrap_or_default().to_string_lossy().to_string();
        let inferred_model_id = folder.replace('_', "/");

        // The read-only resolver. Listing what is on disk must not write to it:
        // a package that cannot be repaired was rewriting its own manifest on
        // every scan, dirtying a file — and waking the virus scanner — for a
        // screen that only wanted to read.
        let (manifest, _needs_repair) =
            AdapterRegistry::resolve_manifest(pkg_path, provider_id, &inferred_model_id).ok()?;

        let gguf_path = pkg_path.join(&manifest.base_model.file_path);
        if !gguf_path.is_file() {
            return None;
        }

        let classification = self.classification_of(&gguf_path, &manifest.base_model.model_name);

        // The header's quantization wins over the manifest's, which records what
        // was asked for rather than what arrived.
        let quantization = classification
            .quantization
            .clone()
            .unwrap_or_else(|| manifest.base_model.quantization.clone());

        let model_id = manifest.base_model.model_id.clone();
        let size_bytes = manifest.base_model.size_bytes;

        Some(InstalledModel {
            id: format!("{}_{}", model_id.replace('/', "_"), quantization),
            model_id,
            model_name: manifest.base_model.model_name.clone(),
            provider_id: manifest.provider_id.clone(),
            quantization,
            format: "GGUF".to_string(),
            backend: "llama.cpp (GGUF)".to_string(),
            file_name: gguf_path.file_name().unwrap_or_default().to_string_lossy().to_string(),
            file_path: gguf_path.to_string_lossy().to_string(),
            size_bytes,
            installed_at: chrono::Utc::now().to_rfc3339(),
            // Present on disk is not the same as usable: a helper file is
            // complete and still cannot be loaded.
            is_ready: size_bytes > 0 && classification.group.is_loadable(),
            checksum: None,
            adapters: Some(manifest.adapters),
            classification: Some(classification),
        })
    }

    /// What the file is, read once per version of the file.
    fn classification_of(&self, gguf_path: &Path, model_name: &str) -> Classification {
        let key = FileKey::of(gguf_path);

        if let Some(key) = &key {
            if let Some(hit) = self.headers.lock().unwrap().get(key) {
                return hit.clone();
            }
        }

        let _stage = Stage::new("storage: read one GGUF header");
        let classification = match crate::ai_engine::gguf_meta::read_gguf_metadata(gguf_path) {
            Ok(meta) => classify(&meta, model_name),
            Err(e) => {
                log::warn!("[STORAGE] Could not classify '{}': {e:#}", gguf_path.display());
                // Not cached: a header that could not be read is usually a
                // download still in flight, and the next look should try again
                // rather than repeat a verdict the file has already outgrown.
                return Classification::unreadable(format!(
                    "Sarathi could not read this file's header: {e}"
                ));
            }
        };

        if let Some(key) = key {
            self.headers.lock().unwrap().insert(key, classification.clone());
        }
        classification
    }
}

/// Free and total bytes on the volume holding `models_dir`.
pub(crate) fn disk_space_for(models_dir: &Path) -> (u64, u64) {
    assert_off_ui_thread("disk volume enumeration");
    let _stage = Stage::new("storage: enumerate volumes");

    let disks = Disks::new_with_refreshed_list();
    let path_str = models_dir.to_string_lossy();
    let drive_prefix = if path_str.len() >= 3 && &path_str[1..3] == ":\\" {
        &path_str[0..3]
    } else {
        "C:\\"
    };

    for disk in &disks {
        let mount = disk.mount_point().to_string_lossy();
        if mount.eq_ignore_ascii_case(drive_prefix) || path_str.starts_with(mount.as_ref()) {
            return (disk.available_space(), disk.total_space());
        }
    }
    (0, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_store(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sarathi_store_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("models/huggingface")).unwrap();
        dir
    }

    /// A minimal but real GGUF header, so the cache is exercised against the
    /// parser rather than a stub of it.
    fn write_gguf(path: &Path, arch: &str, blocks: u32) {
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes()); // tensor count
        out.extend_from_slice(&2u64.to_le_bytes()); // kv count

        let mut kv_str = |key: &str, value: &str, out: &mut Vec<u8>| {
            out.extend_from_slice(&(key.len() as u64).to_le_bytes());
            out.extend_from_slice(key.as_bytes());
            out.extend_from_slice(&8u32.to_le_bytes());
            out.extend_from_slice(&(value.len() as u64).to_le_bytes());
            out.extend_from_slice(value.as_bytes());
        };
        kv_str("general.architecture", arch, &mut out);

        let key = format!("{arch}.block_count");
        out.extend_from_slice(&(key.len() as u64).to_le_bytes());
        out.extend_from_slice(key.as_bytes());
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&blocks.to_le_bytes());

        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&out).unwrap();
    }

    fn package(root: &Path, name: &str, blocks: u32) -> PathBuf {
        let pkg = root.join("models/huggingface").join(name);
        let gguf = pkg.join("base/model.gguf");
        write_gguf(&gguf, "llama", blocks);
        gguf
    }

    #[tokio::test]
    async fn a_second_look_reads_no_header_again() {
        let root = temp_store("cached");
        package(&root, "org_model", 32);
        let store = Arc::new(ModelStore::new());

        let first = store.listing(&root).await;
        assert_eq!(first.len(), 1, "the package should be listed");
        assert_eq!(store.headers.lock().unwrap().len(), 1, "its header should be remembered");

        // Force a real second scan; the walk repeats, the header read must not.
        store.invalidate();
        let second = store.listing(&root).await;
        assert_eq!(second.len(), 1);
        assert_eq!(
            store.headers.lock().unwrap().len(),
            1,
            "a second scan of the same file must not add a second entry"
        );
        let arch = |m: &InstalledModel| m.classification.as_ref().unwrap().architecture.clone();
        assert_eq!(arch(&first[0]), arch(&second[0]), "the cached verdict must be the same one");
    }

    #[tokio::test]
    async fn a_changed_file_is_read_again_rather_than_remembered_wrongly() {
        let root = temp_store("invalidate");
        let gguf = package(&root, "org_model", 32);
        let store = Arc::new(ModelStore::new());

        let before = store.listing(&root).await;
        let before_arch = before[0].classification.as_ref().unwrap().architecture.clone();
        assert_eq!(before_arch, "llama");

        // Rewrite the same path with different contents and a different length,
        // which is what a re-download looks like.
        write_gguf(&gguf, "qwen2", 80);
        store.invalidate();

        let after = store.listing(&root).await;
        assert_eq!(
            after[0].classification.as_ref().unwrap().architecture,
            "qwen2",
            "a rewritten file must be read again, not answered from the old verdict"
        );
        assert_eq!(
            store.headers.lock().unwrap().len(),
            2,
            "the new file identity should be a new entry, not an overwrite of a live one"
        );
    }

    #[tokio::test]
    async fn a_package_added_after_a_scan_is_seen_without_invalidating_anything() {
        // The walk is always live; only header reads are cached. A model that
        // appears on disk must show up even though nothing told the store.
        let root = temp_store("added");
        package(&root, "org_first", 32);
        let store = Arc::new(ModelStore::new());

        assert_eq!(store.listing(&root).await.len(), 1);

        package(&root, "org_second", 40);
        store.invalidate();
        assert_eq!(store.listing(&root).await.len(), 2, "a new package must be found");
    }

    #[tokio::test]
    async fn simultaneous_callers_share_one_scan() {
        // What the Storage screen does: the listing and the summary are asked
        // for together. They must cost one walk, not two.
        let root = temp_store("shared");
        package(&root, "org_model", 32);
        let store = Arc::new(ModelStore::new());

        let a = store.clone();
        let b = store.clone();
        let dir_a = root.clone();
        let dir_b = root.clone();
        let (left, right) = tokio::join!(
            async move { a.listing(&dir_a).await },
            async move { b.summary(&dir_b).await }
        );

        assert_eq!(left.len(), 1);
        assert_eq!(right.total_installed_models, 1);
    }

    #[tokio::test]
    async fn a_missing_store_is_empty_rather_than_an_error() {
        let store = Arc::new(ModelStore::new());
        let nowhere = std::env::temp_dir().join("sarathi_store_does_not_exist");
        let _ = std::fs::remove_dir_all(&nowhere);
        assert!(store.listing(&nowhere).await.is_empty());
    }

    #[tokio::test]
    async fn a_malformed_header_is_reported_rather_than_hiding_the_model() {
        let root = temp_store("malformed");
        let pkg = root.join("models/huggingface/org_broken");
        std::fs::create_dir_all(pkg.join("base")).unwrap();
        std::fs::write(pkg.join("base/model.gguf"), b"not a gguf at all").unwrap();

        let listed = store_listing(&root).await;
        assert_eq!(listed.len(), 1, "a broken file is still something on disk");
        let c = listed[0].classification.as_ref().unwrap();
        assert!(!c.group.is_loadable(), "and it must not be offered as loadable");
    }

    async fn store_listing(root: &Path) -> Arc<Vec<InstalledModel>> {
        Arc::new(ModelStore::new()).listing(root).await
    }
}
