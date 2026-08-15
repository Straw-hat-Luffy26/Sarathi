//! IPC for the NotebookLM capability card on the Launch screen.
//!
//! Every command here is a thin wrapper over
//! [`crate::notebooklm::manager::NotebookLmManager`], which owns the state for
//! the life of the application. The commands hold no state of their own, and
//! neither does the screen: that is what makes opening Launch a read rather
//! than an initialisation.
//!
//! Every one is `async fn` and does its work under `spawn_blocking`: they run
//! subprocesses, and a subprocess on the UI thread is the freeze this app has
//! already been through once. See [`crate::diagnostics`].
//!
//! Nothing in this file handles a credential. Sign-in is Google's browser flow,
//! started in a console the user can watch; Sarathi observes the exit code.

use std::sync::Arc;

use tauri::{AppHandle, Manager, State};

use crate::notebooklm::manager::{NotebookLmManager, ProviderFit};
use crate::notebooklm::NotebookLmStatus;

/// Shorthand for the one piece of state every command here needs.
type Mgr<'r> = State<'r, Arc<NotebookLmManager>>;

/// The current state, and a guarantee that startup detection is under way.
///
/// This is the only thing the Launch page calls when it mounts. It returns
/// whatever is known right now — `Checking` on the very first call of a
/// session, the real state on every call after — and never blocks on a probe.
/// Subsequent changes arrive on the `notebooklm:status` event, so a card that
/// mounts mid-detection is brought up to date without asking again.
///
/// It cannot start a sign-in. There is no code path from here to a browser.
#[tauri::command]
pub async fn notebooklm_state(manager: Mgr<'_>) -> Result<NotebookLmStatus, String> {
    let manager = manager.inner().clone();
    manager.ensure_started();
    Ok(manager.snapshot())
}

/// Re-runs the full probe. The Refresh button, not the mount.
#[tauri::command]
pub async fn notebooklm_redetect(manager: Mgr<'_>) -> Result<NotebookLmStatus, String> {
    let manager = manager.inner().clone();
    Ok(manager.detect_full().await)
}

/// Verifies the session with a live call, and reports what it found.
///
/// The user pressed Health check, so this always makes the call rather than
/// reusing a recent answer.
#[tauri::command]
pub async fn notebooklm_health_check(manager: Mgr<'_>) -> Result<NotebookLmStatus, String> {
    let manager = manager.inner().clone();
    let status = manager.verify(true).await;
    log::info!(
        "[NOTEBOOKLM] Health check: {:?} (version {})",
        status.state,
        status.version.as_deref().unwrap_or("unknown")
    );
    Ok(status)
}

/// Installs `notebooklm-py[mcp]`, reporting progress as it goes.
///
/// Only ever reached from the Install button. Nothing installs software on this
/// machine without the user asking for it.
#[tauri::command]
pub async fn notebooklm_install(manager: Mgr<'_>) -> Result<NotebookLmStatus, String> {
    manager.inner().clone().install().await
}

/// Adds or removes the NotebookLM entry in the shared MCP registry.
///
/// Separate from installation so a user can keep the package and stop offering
/// it to providers, without uninstalling anything — and separate from
/// authentication, so neither one waits on the other.
#[tauri::command]
pub async fn notebooklm_set_registered(
    manager: Mgr<'_>,
    enabled: bool,
) -> Result<NotebookLmStatus, String> {
    manager.inner().clone().set_registered(enabled).await
}

/// Starts Google sign-in in a console the user can see and complete.
///
/// Reached only from Connect or Reconnect. On success the session is stored by
/// `notebooklm-py` in its own profile directory, which Sarathi never reads.
#[tauri::command]
pub async fn notebooklm_login(manager: Mgr<'_>) -> Result<NotebookLmStatus, String> {
    manager.inner().clone().login().await
}

/// Clears the stored session. The next Connect starts from a clean sign-in.
#[tauri::command]
pub async fn notebooklm_logout(manager: Mgr<'_>) -> Result<NotebookLmStatus, String> {
    manager.inner().clone().logout().await
}

fn app_data(app: &AppHandle) -> Result<std::path::PathBuf, String> {
    app.path()
        .app_data_dir()
        .map_err(|e| format!("could not locate the Sarathi data directory: {e}"))
}

/// Starts detection at launch, off the UI thread, and keeps the registry honest.
///
/// The user may never open Launch, so this exists to make the answer ready
/// before they do rather than to make it available at all. Nothing here can
/// open a browser.
pub fn reconcile_at_startup(manager: Arc<NotebookLmManager>) {
    manager.ensure_started();

    tauri::async_runtime::spawn(async move {
        // Report what startup settled on. The registry is reconciled inside
        // the manager, once the probe has actually answered.
        for _ in 0..240 {
            if !manager.snapshot().state.is_busy() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        }

        let status = manager.snapshot();
        log::info!(
            "[NOTEBOOKLM] Detected: {:?}, version {}, mcp {}, registered {}",
            status.state,
            status.version.as_deref().unwrap_or("unknown"),
            if status.mcp_available { "available" } else { "unavailable" },
            status.in_registry
        );
    });
}

/// Which MCP servers each provider would actually be handed right now.
///
/// The Launch screen renders this instead of the registry, so a provider that
/// receives nothing cannot be shown as though it received everything.
#[tauri::command]
pub async fn mcp_delivery_report(app: AppHandle) -> Result<Vec<ProviderMcpReport>, String> {
    let dir = app_data(&app)?;

    tokio::task::spawn_blocking(move || {
        let registry = crate::launcher::mcp::load(&dir);
        let reg = crate::launcher::registry::load(&dir);

        reg.tools
            .iter()
            .map(|tool| {
                let delivery = tool.mcp_delivery(&registry);
                ProviderMcpReport {
                    tool_id: tool.id.clone(),
                    tool_name: tool.name.clone(),
                    delivery,
                }
            })
            .collect()
    })
    .await
    .map_err(|e| format!("report task failed: {e}"))
}

/// Which providers are eligible for this capability, and which have it.
#[tauri::command]
pub async fn notebooklm_providers(app: AppHandle) -> Result<Vec<ProviderFit>, String> {
    let dir = app_data(&app)?;
    tokio::task::spawn_blocking(move || crate::notebooklm::manager::provider_fit(&dir))
        .await
        .map_err(|e| format!("provider task failed: {e}"))
}

/// One provider's row in the MCP delivery table.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderMcpReport {
    pub tool_id: String,
    pub tool_name: String,
    #[serde(flatten)]
    pub delivery: crate::launcher::spec::McpDelivery,
}

#[cfg(test)]
mod tests {
    use crate::notebooklm::{self, NotebookLmState};

    /// The state machine's one hard rule, asserted at the boundary the UI reads.
    #[test]
    fn detection_alone_never_reports_connected() {
        let status = notebooklm::detect();
        assert!(
            matches!(
                status.state,
                NotebookLmState::NotInstalled
                    | NotebookLmState::NotAuthenticated
                    | NotebookLmState::Unverified
            ),
            "detect() performs no live call, so it cannot know: got {:?}",
            status.state
        );
    }
}
