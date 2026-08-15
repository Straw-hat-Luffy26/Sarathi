//! Keeping expensive work off the thread that draws the window.
//!
//! Tauri runs a `#[tauri::command]` declared as a plain `fn` *inline on the
//! thread that received the IPC message* — on Windows that is the main thread,
//! the one pumping the Win32 message loop for the WebView2 window. A command
//! that reads the disk there stops the window answering `WM_*` messages, and
//! once about five seconds pass without a pump Windows retitles it
//! `Sarathi (Not Responding)` and offers to kill it.
//!
//! Nothing in the type system says which thread a function ends up on, so this
//! module makes it observable instead:
//!
//! * [`mark_ui_thread`] records the main thread once, at startup.
//! * [`assert_off_ui_thread`] fails loudly in debug and logs in release when
//!   something expensive is running there anyway. This is the regression guard —
//!   a future `pub fn` command that scans the store trips it in the test suite
//!   and in any debug run, rather than shipping as an intermittent freeze.
//! * [`Stage`] times a stage of work and reports anything that would blow a
//!   frame budget, so a slow path shows up as a number in the log before a user
//!   has to describe it as "sometimes it hangs".

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// One 60 Hz frame. Work longer than this on the UI thread is visible.
pub const FRAME_BUDGET: Duration = Duration::from_millis(16);

/// Above this, a stage is worth a log line wherever it ran — a background scan
/// this slow is still a user waiting on a spinner.
const SLOW_STAGE: Duration = Duration::from_millis(250);

/// Raw id of the thread that owns the window, or 0 before startup records it.
///
/// `ThreadId` is opaque and not convertible to a number on stable, so the value
/// stored is a hash of it. Equality is all this needs.
static UI_THREAD: AtomicU64 = AtomicU64::new(0);
static UI_THREAD_KNOWN: AtomicBool = AtomicBool::new(false);

fn current_thread_key() -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    std::thread::current().id().hash(&mut hasher);
    hasher.finish()
}

/// Records the calling thread as the one that must never block.
///
/// Called from Tauri's `setup`, which runs on the main thread before the window
/// starts pumping messages.
pub fn mark_ui_thread() {
    UI_THREAD.store(current_thread_key(), Ordering::SeqCst);
    UI_THREAD_KNOWN.store(true, Ordering::SeqCst);
    log::info!(
        "[PERF] UI thread is {:?}; disk and lock work must not run here",
        std::thread::current().id()
    );
}

/// True when the caller is on the thread that draws the window.
///
/// False before [`mark_ui_thread`] has run, and in tests and examples, where
/// there is no window to freeze.
pub fn on_ui_thread() -> bool {
    UI_THREAD_KNOWN.load(Ordering::SeqCst) && UI_THREAD.load(Ordering::SeqCst) == current_thread_key()
}

/// Refuses to let expensive work run on the UI thread.
///
/// Panics in debug so a test or a developer run catches it at the call site;
/// logs an error in release rather than killing a user's session over it. Put
/// this at the top of anything that walks the store, parses a model header,
/// waits on the generation mutex, or talks to a subprocess.
#[track_caller]
pub fn assert_off_ui_thread(what: &str) {
    if !on_ui_thread() {
        return;
    }

    let message = format!(
        "[PERF] '{what}' is running on the UI thread. Windows stops pumping the \
         window's messages for as long as it takes, which is what
         'Sarathi (Not Responding)' is. Make the command `async fn` and run the \
         body under `tokio::task::spawn_blocking`."
    );

    debug_assert!(false, "{message}");
    log::error!("{message}");
}

/// Times a stage and reports it when it finishes.
///
/// Held in a local: `let _s = Stage::new("storage: list packages");`. The report
/// happens on drop, so an early return is still measured.
pub struct Stage {
    name: &'static str,
    started: Instant,
    on_ui: bool,
}

impl Stage {
    pub fn new(name: &'static str) -> Self {
        Self { name, started: Instant::now(), on_ui: on_ui_thread() }
    }

    /// Elapsed time so far, for callers that want to report it themselves.
    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }
}

impl Drop for Stage {
    fn drop(&mut self) {
        let took = self.started.elapsed();

        if self.on_ui && took > FRAME_BUDGET {
            log::error!(
                "[PERF] '{}' held the UI thread for {:.1}ms (budget {}ms)",
                self.name,
                took.as_secs_f64() * 1000.0,
                FRAME_BUDGET.as_millis()
            );
        } else if took > SLOW_STAGE {
            log::info!("[PERF] '{}' took {:.1}ms", self.name, took.as_secs_f64() * 1000.0);
        } else {
            log::debug!("[PERF] '{}' took {:.1}ms", self.name, took.as_secs_f64() * 1000.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Nothing is the UI thread until startup says so, so tests, examples and
    /// the headless verification binaries are never tripped by the guard.
    #[test]
    fn without_a_marked_ui_thread_nothing_is_the_ui_thread() {
        // This process never calls `mark_ui_thread`, so the guard is inert and
        // must not panic in a debug test build.
        assert!(!on_ui_thread());
        assert_off_ui_thread("a scan in a test");
    }

    #[test]
    fn a_marked_thread_recognises_itself_and_no_other() {
        // Run in a child so the marker does not leak into other tests.
        let child = std::thread::spawn(|| {
            mark_ui_thread();
            let here = on_ui_thread();
            let elsewhere = std::thread::spawn(on_ui_thread).join().unwrap();
            (here, elsewhere)
        });

        let (here, elsewhere) = child.join().unwrap();
        assert!(here, "the marked thread must recognise itself");
        assert!(!elsewhere, "another thread must not be mistaken for it");
    }

    #[test]
    fn a_stage_measures_the_work_it_wraps() {
        let stage = Stage::new("test");
        std::thread::sleep(Duration::from_millis(5));
        assert!(stage.elapsed() >= Duration::from_millis(5));
    }
}
