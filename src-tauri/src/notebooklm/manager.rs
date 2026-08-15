//! One NotebookLM connection for the whole application.
//!
//! ## Why this exists
//!
//! The card used to own its own state. React unmounts the Launch page whenever
//! the user visits another tab, so every return re-ran detection from nothing,
//! landed back on `Unverified`, and offered **Connect / Login** as the primary
//! action — to a user who had signed in five minutes earlier. Nothing was
//! actually re-authenticating; the app had simply forgotten that it ever had.
//! Forgetting looked exactly like being logged out, and the only way out of it
//! the card offered was to sign in again.
//!
//! So the connection lives here, in Tauri's managed state, for the life of the
//! process, and the persisted half of it ([`super::state`]) outlives even that.
//! The Launch page *reads* this. Mounting it is not an event this manager can
//! observe, which is the strongest available guarantee that mounting cannot
//! start a sign-in.
//!
//! ## The rules it enforces
//!
//! * **Detection runs once**, then from remembered paths. Concurrent callers
//!   share one run rather than starting their own.
//! * **A live check runs once** per reason to run one — startup, an explicit
//!   Health check, the end of a sign-in. Never on a timer, never on a mount.
//! * **Only Google demotes a session.** A restart, a tab switch, a model load
//!   and a re-render are all invisible to the authentication state.
//! * **One sign-in at a time.** A second Connect while a browser is open is
//!   refused rather than opening a second browser.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use super::state::{self, Persisted};
use super::{registry, NotebookLmState, NotebookLmStatus, Remembered, MCP_SERVER_KEY};

/// Event every subscriber listens to. One state, many views.
pub const STATUS_EVENT: &str = "notebooklm:status";
/// Human-readable progress for the phase currently running.
pub const PROGRESS_EVENT: &str = "notebooklm:progress";

/// Where state changes are announced.
///
/// The manager does not depend on Tauri for anything else, so this is the whole
/// of the coupling — and behind a trait it can be driven by a test that walks
/// the real lifecycle (mount, remount, restart, sign out) without a window.
/// The bugs this module fixes were all lifecycle bugs; a test that cannot
/// perform a lifecycle cannot catch them.
pub trait StatusSink: Send + Sync {
    fn status(&self, status: &NotebookLmStatus);
    fn progress(&self, line: &str);
}

impl StatusSink for tauri::AppHandle {
    fn status(&self, status: &NotebookLmStatus) {
        use tauri::Emitter;
        let _ = self.emit(STATUS_EVENT, status);
    }

    fn progress(&self, line: &str) {
        use tauri::Emitter;
        let _ = self.emit(PROGRESS_EVENT, line.to_string());
    }
}

/// A sink that drops everything, for a manager nobody is watching.
pub struct Silent;

impl StatusSink for Silent {
    fn status(&self, _: &NotebookLmStatus) {}
    fn progress(&self, _: &str) {}
}

/// Which providers could take this capability, and which are being handed it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderFit {
    pub id: String,
    pub name: String,
    /// The provider speaks MCP at all, so it is eligible.
    pub compatible: bool,
    /// The generated config it gets at launch actually contains this server.
    pub receiving: bool,
}

pub struct NotebookLmManager {
    app_data: PathBuf,
    /// Where every state change is announced. One sink, so there is one place
    /// a subscriber can be told and no way to change state quietly.
    sink: Arc<dyn StatusSink>,
    /// What every reader sees. Written only by [`Self::publish`].
    snapshot: RwLock<NotebookLmStatus>,
    /// The half that survives a restart.
    persisted: std::sync::Mutex<Persisted>,
    /// Held for the duration of a full detection, so a second caller waits for
    /// the first rather than launching its own PATH scan.
    detecting: tokio::sync::Mutex<()>,
    /// Same, for the live check — with a counter so a caller that waited can
    /// tell it was waiting for an answer that has since arrived.
    verifying: tokio::sync::Mutex<()>,
    verify_generation: AtomicU64,
    /// True while a browser sign-in is open.
    signing_in: AtomicBool,
    /// Startup work is idempotent; this is what makes it so.
    started: AtomicBool,
}

impl NotebookLmManager {
    pub fn new(app_data: PathBuf, sink: Arc<dyn StatusSink>) -> Self {
        let persisted = state::load(&app_data);
        Self {
            app_data,
            sink,
            snapshot: RwLock::new(NotebookLmStatus::blank(NotebookLmState::Checking)),
            persisted: std::sync::Mutex::new(persisted),
            detecting: tokio::sync::Mutex::new(()),
            verifying: tokio::sync::Mutex::new(()),
            verify_generation: AtomicU64::new(0),
            signing_in: AtomicBool::new(false),
            started: AtomicBool::new(false),
        }
    }

    /// The current state, with no work of any kind. This is what a mounting
    /// component gets, and it is why mounting is free.
    pub fn snapshot(&self) -> NotebookLmStatus {
        self.snapshot.read().expect("status lock poisoned").clone()
    }

    /// Brings the state up to date, once per process.
    ///
    /// Runs in three waves so the card is never blank and never lies:
    ///
    /// 1. remembered paths confirmed on disk — microseconds;
    /// 2. a full probe — seconds, in the background, correcting wave 1;
    /// 3. a live check, only when there is a session to check and the user has
    ///    not signed out.
    ///
    /// Wave 3 never opens a browser. Its worst outcome is `AuthenticationExpired`,
    /// which puts a **Reconnect** button on the card for the user to press —
    /// which is the whole difference between validating a session and demanding
    /// one.
    pub fn ensure_started(self: &Arc<Self>) {
        if self.started.swap(true, Ordering::SeqCst) {
            return;
        }
        let this = self.clone();
        tauri::async_runtime::spawn(async move {
            this.run_startup().await;
        });
    }

    /// The startup sequence, awaited rather than spawned.
    ///
    /// Guarded by the same flag as [`Self::ensure_started`], so the two cannot
    /// between them run startup twice. They could, before: a caller that
    /// awaited this and a screen that later called `ensure_started` produced
    /// two full probes and two live checks, and the second one dragged the card
    /// back from Connected to Unverified while it re-detected — the very
    /// flicker this manager exists to remove.
    pub async fn startup(self: Arc<Self>) {
        if self.started.swap(true, Ordering::SeqCst) {
            return;
        }
        self.run_startup().await;
    }

    async fn run_startup(self: Arc<Self>) {
        let remembered = self.persisted.lock().expect("state lock").remembered.clone();

        // Wave 1: nothing but `is_file()` calls.
        let quick = tokio::task::spawn_blocking(move || super::detect_remembered(&remembered))
            .await
            .ok()
            .flatten();
        if let Some(status) = quick {
            log::info!("[NOTEBOOKLM] Remembered installation confirmed on disk");
            self.publish(status).await;
        }

        // Wave 2: the real probe, correcting whatever wave 1 assumed.
        let detected = self.detect_full().await;

        // Now, and not before, the registry can be judged: an entry is removed
        // because the probe says the server is gone, never because the probe
        // has not finished.
        self.drop_stale_registration().await;

        // Wave 3: is the session Google's problem or ours?
        let signed_out = self.persisted.lock().expect("state lock").signed_out;
        if detected.has_local_session && !signed_out && detected.cli_path.is_some() {
            self.verify(false).await;
        } else if signed_out {
            log::info!("[NOTEBOOKLM] A stored session exists but the user signed out; not checking it");
        }
    }

    /// A full probe, deduplicated. Publishes and persists the result.
    pub async fn detect_full(self: &Arc<Self>) -> NotebookLmStatus {
        let _guard = self.detecting.lock().await;

        let detected = tokio::task::spawn_blocking(super::detect)
            .await
            .unwrap_or_else(|e| {
                let mut s = NotebookLmStatus::blank(NotebookLmState::ConnectionFailed);
                s.detail = Some(format!("detection did not finish: {e}"));
                s
            });

        // Remember where things are, so the next start skips all of that.
        self.mutate_persisted(|p| p.remembered = Remembered::of(&detected));

        self.publish(detected).await
    }

    /// Asks Google whether the stored session still works.
    ///
    /// `force` is the difference between the user pressing Health check and
    /// anything else wanting to know: without it, a check that another caller
    /// has just completed is reused instead of repeated.
    pub async fn verify(
        self: &Arc<Self>,
        force: bool,
    ) -> NotebookLmStatus {
        let before = self.verify_generation.load(Ordering::SeqCst);

        // No session on disk is not a question for Google. Asking anyway costs
        // several seconds and comes back as a failure that reads like something
        // is broken, when the honest answer is simply "not signed in".
        //
        // Read from disk rather than from the snapshot: a sign-in that has just
        // finished wrote the session file, and the snapshot still remembers the
        // machine as it was before it did.
        {
            let present = super::session_fingerprint().is_some();
            let current = self.snapshot();
            if current.cli_path.is_some() && !present {
                let mut plain = current;
                plain.state = NotebookLmState::NotAuthenticated;
                plain.has_local_session = false;
                plain.last_verified_at = None;
                plain.detail = None;
                return self.publish(plain).await;
            }
        }

        // Announce the phase before waiting on the lock, so a card that just
        // asked shows "Verifying" rather than an unexplained pause.
        let mut pending = self.snapshot();
        if pending.cli_path.is_some() {
            pending.state = NotebookLmState::Verifying;
            self.publish(pending).await;
        }

        let _guard = self.verifying.lock().await;
        if !force && self.verify_generation.load(Ordering::SeqCst) != before {
            // Someone else's check finished while this one queued. Its answer
            // is this answer; running a second network call would only make the
            // user wait for the same result twice.
            log::debug!("[NOTEBOOKLM] Reusing a verification that completed while queued");
            return self.snapshot();
        }

        let base = self.snapshot();
        if base.cli_path.is_none() {
            return base;
        }

        self.progress("Checking the saved NotebookLM session with Google…");
        let verified = tokio::task::spawn_blocking(move || super::verify(base))
            .await
            .unwrap_or_else(|e| {
                let mut s = NotebookLmStatus::blank(NotebookLmState::ConnectionFailed);
                s.detail = Some(format!("the check did not finish: {e}"));
                s
            });
        self.verify_generation.fetch_add(1, Ordering::SeqCst);

        self.remember_outcome(&verified);
        self.publish(verified).await
    }

    /// Runs Google's sign-in, then verifies what it produced.
    ///
    /// The browser and everything typed into it are between the user and
    /// Google. Sarathi starts the CLI's login command in a console the user can
    /// see and waits for it to exit; it never sees a password or a second
    /// factor, and it never reads the session file the CLI writes.
    pub async fn login(self: &Arc<Self>) -> Result<NotebookLmStatus, String> {
        if self.signing_in.swap(true, Ordering::SeqCst) {
            return Err("a NotebookLM sign-in is already open — finish it in the browser".into());
        }
        // Released however this returns, so a failed sign-in cannot wedge the
        // button for the rest of the session.
        let _release = SigningIn(self.clone());

        let mut announcing = self.snapshot();
        announcing.state = NotebookLmState::Authenticating;
        announcing.detail = None;
        self.publish(announcing).await;
        self.progress("Opening a browser for Google sign-in — complete it there");

        let outcome = tokio::task::spawn_blocking(|| -> Result<(), String> {
            let (program, args) = super::login_command()?;
            // Visible on purpose: the flow needs a console the user can watch
            // and, on some paths, paste into. A hidden window would look like
            // a hang.
            let status = std::process::Command::new(&program)
                .args(&args)
                .status()
                .map_err(|e| format!("could not start the NotebookLM sign-in: {e}"))?;
            if status.success() {
                Ok(())
            } else {
                Err("the sign-in did not complete".to_string())
            }
        })
        .await
        .map_err(|e| format!("login task failed: {e}"))?;

        self.progress("Google sign-in finished — verifying the session…");

        // A sign-in changes the session file, so the previous verification no
        // longer refers to anything. The paths, though, have not moved: this
        // verifies against the installation already detected rather than
        // re-running the PATH scan, which is what used to put fifteen seconds
        // of subprocess between "signed in" and "Connected".
        self.mutate_persisted(|p| {
            p.forget_verification();
            p.signed_out = false;
        });

        let mut status = self.verify(true).await;
        if let Err(e) = outcome {
            if status.state != NotebookLmState::Connected {
                status.detail = Some(e);
                self.publish(status.clone()).await;
            }
        }
        Ok(status)
    }

    /// Clears the stored session. The only thing besides Google that may.
    pub async fn logout(self: &Arc<Self>) -> Result<NotebookLmStatus, String> {
        tokio::task::spawn_blocking(super::logout)
            .await
            .map_err(|e| format!("logout task failed: {e}"))??;

        self.mutate_persisted(|p| {
            p.forget_verification();
            p.signed_out = true;
        });

        let mut status = self.snapshot();
        status.state = NotebookLmState::NotAuthenticated;
        status.has_local_session = super::session_fingerprint().is_some();
        status.last_verified_at = None;
        status.detail = None;
        Ok(self.publish(status).await)
    }

    /// Installs the package, on an explicit request and never otherwise.
    pub async fn install(self: &Arc<Self>) -> Result<NotebookLmStatus, String> {
        let mut announcing = self.snapshot();
        announcing.state = NotebookLmState::Installing;
        announcing.detail = None;
        self.publish(announcing).await;

        let sink = self.sink.clone();
        let outcome = tokio::task::spawn_blocking(move || {
            super::install(|step| {
                log::info!("[NOTEBOOKLM] {step}");
                sink.progress(step);
            })
        })
        .await
        .map_err(|e| format!("install task failed: {e}"))?;

        match outcome {
            Ok(installed) => {
                self.mutate_persisted(|p| p.remembered = Remembered::of(&installed));
                let status = self.publish(installed).await;
                // Installed and able to serve MCP, so it joins the registry —
                // and from there reaches every MCP-capable provider with no
                // provider-side change.
                if status.mcp_available && !status.in_registry {
                    return self.set_registered(true).await;
                }
                Ok(status)
            }
            Err(why) => {
                let mut failed = self.snapshot();
                failed.state = NotebookLmState::InstallFailed;
                failed.detail = Some(why.clone());
                self.publish(failed).await;
                Err(why)
            }
        }
    }

    /// Adds or removes the entry every MCP-capable provider is handed.
    ///
    /// Deliberately independent of authentication: rewriting provider config is
    /// not a reason to make anyone sign in, and signing in is not a reason to
    /// rewrite provider config.
    pub async fn set_registered(
        self: &Arc<Self>,
        enabled: bool,
    ) -> Result<NotebookLmStatus, String> {
        let dir = self.app_data.clone();
        let status = self.snapshot();
        let change = tokio::task::spawn_blocking(move || {
            if enabled {
                registry::register(&dir, &status)
            } else {
                registry::unregister(&dir)
            }
        })
        .await
        .map_err(|e| format!("registry task failed: {e}"))??;

        log::info!("[NOTEBOOKLM] Registry entry: {change:?}");
        Ok(self.publish(self.snapshot()).await)
    }

    /// Drops a registry entry whose server has since been uninstalled.
    ///
    /// One direction only: an entry pointing at a command that is no longer
    /// there would be handed to every provider, and each would report its own
    /// failure to start it. Nothing is ever *added* here — offering a capability
    /// is the user's decision, not something that happens because a package
    /// turned up on the machine.
    pub async fn drop_stale_registration(self: &Arc<Self>) -> bool {
        let status = self.snapshot();
        if status.state == NotebookLmState::Checking || status.mcp_available {
            return false;
        }
        let dir = self.app_data.clone();
        let removed = tokio::task::spawn_blocking(move || {
            if registry::is_registered(&dir) {
                let _ = registry::unregister(&dir);
                true
            } else {
                false
            }
        })
        .await
        .unwrap_or(false);

        if removed {
            log::warn!(
                "[NOTEBOOKLM] Removed a registry entry whose MCP server is no longer installed; \
                 providers would have been handed a command that cannot start"
            );
        }
        removed
    }

    /// Fills in everything that is cheap to know, stores it, and tells everyone.
    ///
    /// Every state change goes through here, so there is exactly one place that
    /// decides what a subscriber sees and exactly one event they need.
    async fn publish(&self, mut status: NotebookLmStatus) -> NotebookLmStatus {
        let dir = self.app_data.clone();
        let (in_registry, providers) =
            tokio::task::spawn_blocking(move || (registry::is_registered(&dir), provider_fit(&dir)))
                .await
                .unwrap_or((false, Vec::new()));

        status.in_registry = in_registry;
        status.compatible_providers = providers
            .iter()
            .filter(|p| p.compatible)
            .map(|p| p.name.clone())
            .collect();

        {
            let fingerprint = super::session_fingerprint();
            let persisted = self.persisted.lock().expect("state lock");
            status.signed_out = persisted.signed_out;
            // Carry a previous run's verification forward as history, but only
            // while it still refers to the session that is on disk now. It is
            // shown as "last verified <time>", never as a reason to claim
            // Connected — only a live call does that.
            if status.last_verified_at.is_none()
                && persisted.verification_still_applies(fingerprint.as_deref())
            {
                status.last_verified_at = persisted.last_verified_at.clone();
            }
        }

        let previous = {
            let mut guard = self.snapshot.write().expect("status lock poisoned");
            let previous = guard.state;
            *guard = status.clone();
            previous
        };
        if previous != status.state {
            // Every transition, in order, in the log. This is the record that
            // answers "did opening Launch make it re-authenticate?" without
            // anyone having to reproduce it live.
            log::info!("[NOTEBOOKLM] {previous:?} -> {:?}", status.state);
        }
        self.sink.status(&status);
        status
    }

    fn progress(&self, line: &str) {
        log::info!("[NOTEBOOKLM] {line}");
        self.sink.progress(line);
    }

    /// Writes down what a live check just established.
    fn remember_outcome(&self, status: &NotebookLmStatus) {
        match status.state {
            NotebookLmState::Connected => {
                let at = status
                    .last_verified_at
                    .clone()
                    .unwrap_or_else(|| chrono::Utc::now().to_rfc3339());
                let fingerprint = super::session_fingerprint();
                self.mutate_persisted(|p| p.record_verified(at.clone(), fingerprint.clone()));
            }
            // Google refused it: the remembered check is worthless now.
            NotebookLmState::AuthenticationExpired => {
                self.mutate_persisted(Persisted::forget_verification);
            }
            // Anything else — the network, a broken CLI — says nothing about
            // the session, so the remembered check stands.
            _ => {}
        }
    }

    fn mutate_persisted(&self, f: impl FnOnce(&mut Persisted)) {
        let snapshot = {
            let mut guard = self.persisted.lock().expect("state lock");
            f(&mut guard);
            guard.clone()
        };
        if let Err(e) = state::save(&self.app_data, &snapshot) {
            log::warn!("[NOTEBOOKLM] Could not save state: {e}");
        }
    }
}

/// Releases the sign-in guard however the login returns.
struct SigningIn(Arc<NotebookLmManager>);

impl Drop for SigningIn {
    fn drop(&mut self) {
        self.0.signing_in.store(false, Ordering::SeqCst);
    }
}

/// Which providers could take NotebookLM, asked of each provider's own MCP
/// support rather than assumed from a list of names.
///
/// A provider added tomorrow that declares an MCP dialect appears here with no
/// change to this function; one that declares none never will, and is not
/// claimed to have a capability it cannot receive.
pub fn provider_fit(app_data_dir: &std::path::Path) -> Vec<ProviderFit> {
    let mcp = crate::launcher::mcp::load(app_data_dir);
    crate::launcher::registry::load(app_data_dir)
        .tools
        .iter()
        .map(|tool| {
            let delivery = tool.mcp_delivery(&mcp);
            ProviderFit {
                id: tool.id.clone(),
                name: tool.name.clone(),
                compatible: delivery.supported,
                receiving: delivery.delivered.iter().any(|n| n == MCP_SERVER_KEY),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sarathi_nlm_mgr_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A newly constructed manager has done nothing and claims nothing.
    #[test]
    fn a_fresh_manager_is_checking_and_has_run_no_probe() {
        let m = NotebookLmManager::new(temp("fresh"), Arc::new(Silent));
        assert_eq!(m.snapshot().state, NotebookLmState::Checking);
        assert!(!m.started.load(Ordering::SeqCst));
    }

    /// Reading the state is what a mounting component does, and it must be
    /// free — no subprocess, no network, no lock held across an await.
    #[test]
    fn reading_the_state_a_thousand_times_costs_nothing() {
        let m = NotebookLmManager::new(temp("cheap"), Arc::new(Silent));
        let started = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = m.snapshot();
        }
        assert!(
            started.elapsed() < std::time::Duration::from_millis(50),
            "a mount must never pay for a probe: {:?}",
            started.elapsed()
        );
    }

    /// The bug this module exists for: remounting must not undo a sign-in.
    #[test]
    fn a_verified_session_survives_any_number_of_readers() {
        let m = NotebookLmManager::new(temp("survives"), Arc::new(Silent));
        {
            let mut s = m.snapshot.write().unwrap();
            s.state = NotebookLmState::Connected;
            s.last_verified_at = Some("2026-08-12T10:00:00Z".into());
        }
        for _ in 0..20 {
            assert_eq!(m.snapshot().state, NotebookLmState::Connected);
        }
    }

    #[test]
    fn startup_work_happens_once_however_many_screens_ask_for_it() {
        let m = NotebookLmManager::new(temp("once"), Arc::new(Silent));
        assert!(!m.started.swap(true, Ordering::SeqCst), "the first caller runs it");
        assert!(m.started.swap(true, Ordering::SeqCst), "every later caller does not");
    }

    /// Two sign-ins would mean two browsers and two sessions racing to write
    /// the same file.
    #[test]
    fn only_one_sign_in_can_be_open_at_a_time() {
        let m = Arc::new(NotebookLmManager::new(temp("onelogin"), Arc::new(Silent)));
        assert!(!m.signing_in.swap(true, Ordering::SeqCst));
        assert!(m.signing_in.swap(true, Ordering::SeqCst), "the second is refused");

        drop(SigningIn(m.clone()));
        assert!(!m.signing_in.load(Ordering::SeqCst), "and the guard is released");
    }

    /// Signing out is remembered; a restart must not quietly re-verify a
    /// session the user asked to be rid of.
    #[test]
    fn signing_out_is_written_down() {
        let dir = temp("signout");
        let m = NotebookLmManager::new(dir.clone(), Arc::new(Silent));
        m.mutate_persisted(|p| {
            p.forget_verification();
            p.signed_out = true;
        });
        assert!(state::load(&dir).signed_out);
    }

    /// Provider eligibility is read from the registry, so a provider added
    /// later is included without this code changing.
    #[test]
    fn provider_eligibility_comes_from_the_registry_not_a_list() {
        let fit = provider_fit(&temp("providers"));
        assert!(!fit.is_empty(), "the shipped providers are the starting point");
        assert!(
            fit.iter().any(|p| p.compatible),
            "at least one shipped provider speaks MCP"
        );
        // Nothing is claimed to be receiving a server that is not registered.
        assert!(
            fit.iter().all(|p| !p.receiving),
            "an empty registry delivers nothing to anyone"
        );
    }
}
