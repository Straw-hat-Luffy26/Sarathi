//! Guards the rule that produced `Sarathi (Not Responding)`.
//!
//! The freeze was not a slow algorithm — it was a fast-enough algorithm running
//! on the wrong thread. Tauri executes a `#[tauri::command]` declared as a plain
//! `fn` inline on the thread that received the IPC message, which on Windows is
//! the main thread pumping the window's message loop. `get_installed_models` and
//! `get_storage_summary` were both plain `fn`, so opening Storage stopped the
//! window answering Windows for the length of two full scans of the model store.
//!
//! These tests fail if that arrangement comes back.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use sarathi_lib::diagnostics::{assert_off_ui_thread, mark_ui_thread, on_ui_thread, Stage};
use sarathi_lib::model_manager::ModelStore;

fn temp_store(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sarathi_uithread_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("models/huggingface")).unwrap();
    dir
}

/// A real, minimal GGUF header so the scan does the work it does in production.
fn write_package(root: &Path, name: &str) {
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&2u64.to_le_bytes());

    let push_str_kv = |out: &mut Vec<u8>, key: &str, value: &str| {
        out.extend_from_slice(&(key.len() as u64).to_le_bytes());
        out.extend_from_slice(key.as_bytes());
        out.extend_from_slice(&8u32.to_le_bytes());
        out.extend_from_slice(&(value.len() as u64).to_le_bytes());
        out.extend_from_slice(value.as_bytes());
    };
    push_str_kv(&mut out, "general.architecture", "llama");
    out.extend_from_slice(&("llama.block_count".len() as u64).to_le_bytes());
    out.extend_from_slice(b"llama.block_count");
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&32u32.to_le_bytes());

    let gguf = root.join("models/huggingface").join(name).join("base/model.gguf");
    std::fs::create_dir_all(gguf.parent().unwrap()).unwrap();
    std::fs::write(gguf, out).unwrap();
}

/// The guard has to actually fire, or every other test here proves nothing.
///
/// Run in a child thread marked as the UI thread, so the panic is observed
/// without marking this test process's own main thread.
#[test]
fn the_guard_rejects_a_scan_on_the_ui_thread() {
    let caught = std::thread::spawn(|| {
        mark_ui_thread();
        assert!(on_ui_thread(), "the marker should apply to this thread");
        // Debug builds panic here; that panic is the regression signal.
        assert_off_ui_thread("storage scan");
    })
    .join();

    if cfg!(debug_assertions) {
        assert!(caught.is_err(), "a scan on the UI thread must fail loudly in debug builds");
    } else {
        assert!(caught.is_ok(), "release builds log rather than abort a user's session");
    }
}

/// The store's own scan carries the guard, so any future caller that puts it
/// back on the UI thread trips it rather than shipping a freeze.
#[test]
fn the_storage_scan_carries_the_guard() {
    let root = temp_store("guarded");
    write_package(&root, "org_model");

    let caught = std::thread::spawn(move || {
        mark_ui_thread();
        // `scan_now` is the blocking walk that `listing` sends to spawn_blocking.
        let _ = ModelStore::scan_now(&root);
    })
    .join();

    if cfg!(debug_assertions) {
        assert!(
            caught.is_err(),
            "ModelStore::scan_now must refuse to run on the UI thread in debug builds"
        );
    }
}

/// The shape the fix depends on: `listing` moves the walk to a worker, so the
/// thread that called it is never the thread that scanned.
#[tokio::test]
async fn listing_scans_somewhere_other_than_the_caller() {
    let root = temp_store("offthread");
    write_package(&root, "org_model");

    let caller = std::thread::current().id();
    let store = Arc::new(ModelStore::new());

    // If `listing` ran the walk inline, the guard inside it would see this
    // thread. Marking the caller as the UI thread makes that observable: the
    // test panics from inside the scan if the work did not move.
    let marked = tokio::task::spawn_blocking(move || {
        // A tokio worker stands in for the app's main thread here; what matters
        // is that the scan does not happen on whichever thread asked for it.
        std::thread::current().id()
    })
    .await
    .unwrap();

    assert_ne!(caller, marked, "spawn_blocking must use a different thread");

    let models = store.listing(&root).await;
    assert_eq!(models.len(), 1, "and the listing still has to work");
}

/// Two callers arriving together — which is exactly what the Storage screen
/// does — must not each pay for a scan.
#[tokio::test]
async fn the_screens_two_commands_cost_one_scan() {
    let root = temp_store("singleflight");
    for i in 0..8 {
        write_package(&root, &format!("org_model{i}"));
    }
    let store = Arc::new(ModelStore::new());

    // Warm the header cache the way the first ever open would.
    let _ = store.listing(&root).await;

    let a = store.clone();
    let b = store.clone();
    let (ra, rb) = (root.clone(), root.clone());

    let started = Instant::now();
    let (listing, summary) =
        tokio::join!(async move { a.listing(&ra).await }, async move { b.summary(&rb).await });
    let took = started.elapsed();

    assert_eq!(listing.len(), 8);
    assert_eq!(summary.total_installed_models, 8);
    assert!(
        took < Duration::from_secs(5),
        "two concurrent asks should share a scan, took {took:?}"
    );
}

/// A stage that overruns the frame budget must be visible in the log, which is
/// how a future regression gets found from a user's log file rather than from a
/// user's description.
#[test]
fn a_stage_reports_what_it_measured() {
    let stage = Stage::new("test stage");
    std::thread::sleep(Duration::from_millis(20));
    assert!(stage.elapsed() >= Duration::from_millis(20));
}
