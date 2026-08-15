//! What Sarathi remembers about NotebookLM between runs.
//!
//! Two things are remembered, and nothing else:
//!
//! * **Where the programs are** — so the next start does not repeat a PATH scan
//!   and a Python interpreter launch to rediscover files that have not moved.
//! * **That a live check once succeeded** — the RFC3339 time of it, and a
//!   size-and-mtime marker for the session file it was checked against.
//!
//! ## What is deliberately not here
//!
//! No cookie, no token, no Google identity, not one byte of the session file's
//! contents. `notebooklm-py` owns the session and keeps it in its own profile
//! directory; Sarathi records only that a check passed and which file it passed
//! against. That is enough to avoid asking the user to sign in again, and not
//! enough to be worth stealing.
//!
//! The fingerprint is what makes "still signed in" a fact rather than an
//! assumption with a timer on it. If the session file is replaced — a
//! `notebooklm login` run from a terminal, a restored profile — the remembered
//! verification no longer refers to the session on disk and is dropped.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::Remembered;

/// File name inside Sarathi's app-data directory.
const FILE: &str = "notebooklm-state.json";

/// The whole of what survives a restart.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Persisted {
    /// Paths resolved by the last full detection.
    pub remembered: Remembered,
    /// RFC3339 time of the last live check that succeeded.
    pub last_verified_at: Option<String>,
    /// Size-and-mtime of the session file that check was made against.
    pub verified_session: Option<String>,
    /// The user pressed Sign out. Survives a restart, because "I signed out"
    /// is a decision and not a transient UI state.
    pub signed_out: bool,
}

impl Persisted {
    /// Whether a remembered verification still refers to the session on disk.
    ///
    /// False after an explicit sign-out, false when there is nothing
    /// remembered, and false when the session file has been replaced since.
    /// Never false merely because time has passed — expiry is something Google
    /// tells us, not something a clock decides.
    pub fn verification_still_applies(&self, current_fingerprint: Option<&str>) -> bool {
        if self.signed_out || self.last_verified_at.is_none() {
            return false;
        }
        match (self.verified_session.as_deref(), current_fingerprint) {
            (Some(then), Some(now)) => then == now,
            // Verified before fingerprints were recorded, or the session file
            // has gone: neither is proof, so re-check rather than claim.
            _ => false,
        }
    }

    /// Records a verification that just succeeded.
    pub fn record_verified(&mut self, at: String, fingerprint: Option<String>) {
        self.last_verified_at = Some(at);
        self.verified_session = fingerprint;
        self.signed_out = false;
    }

    /// Forgets the verification without forgetting where the programs are.
    pub fn forget_verification(&mut self) {
        self.last_verified_at = None;
        self.verified_session = None;
    }
}

pub fn path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join(FILE)
}

/// Reads the remembered state, treating anything unreadable as "nothing
/// remembered" — the cost of being wrong is one extra detection.
pub fn load(app_data_dir: &Path) -> Persisted {
    let file = path(app_data_dir);
    let Ok(raw) = std::fs::read_to_string(&file) else {
        return Persisted::default();
    };
    match serde_json::from_str(&raw) {
        Ok(state) => state,
        Err(e) => {
            log::warn!("[NOTEBOOKLM] Ignoring unreadable {}: {e}", file.display());
            Persisted::default()
        }
    }
}

/// Writes through a temporary file so an interrupted write cannot leave a
/// half-document that the next start would read as "nothing remembered".
pub fn save(app_data_dir: &Path, state: &Persisted) -> Result<(), String> {
    let file = path(app_data_dir);
    if let Some(parent) = file.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("could not prepare {}: {e}", parent.display()))?;
    }
    let body = serde_json::to_string_pretty(state)
        .map_err(|e| format!("could not encode the NotebookLM state: {e}"))?;

    let tmp = file.with_extension("json.tmp");
    std::fs::write(&tmp, format!("{body}\n"))
        .map_err(|e| format!("could not write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, &file)
        .map_err(|e| format!("could not replace {}: {e}", file.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sarathi_nlm_state_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn nothing_remembered_reads_back_as_nothing_rather_than_failing() {
        assert_eq!(load(&temp("empty")), Persisted::default());
    }

    #[test]
    fn what_is_written_is_what_comes_back() {
        let dir = temp("roundtrip");
        let mut state = Persisted {
            remembered: Remembered {
                cli_path: Some(r"C:\py\Scripts\notebooklm.exe".into()),
                mcp_server_path: Some(r"C:\py\Scripts\notebooklm-mcp.exe".into()),
                version: Some("0.8.0".into()),
            },
            ..Persisted::default()
        };
        state.record_verified("2026-08-12T10:00:00Z".into(), Some("120:1760000000".into()));

        save(&dir, &state).unwrap();
        assert_eq!(load(&dir), state);
    }

    /// The rule that stops a restart from becoming a sign-in.
    #[test]
    fn an_unchanged_session_keeps_its_verification_across_a_restart() {
        let mut state = Persisted::default();
        state.record_verified("2026-08-12T10:00:00Z".into(), Some("120:1760000000".into()));

        assert!(state.verification_still_applies(Some("120:1760000000")));
        // Age alone changes nothing: there is no timer here on purpose.
        assert!(state.verification_still_applies(Some("120:1760000000")));
    }

    #[test]
    fn a_replaced_session_file_invalidates_the_remembered_check() {
        let mut state = Persisted::default();
        state.record_verified("2026-08-12T10:00:00Z".into(), Some("120:1760000000".into()));

        assert!(!state.verification_still_applies(Some("998:1799999999")));
        assert!(!state.verification_still_applies(None), "no session file, no claim");
    }

    #[test]
    fn signing_out_invalidates_it_and_survives_a_restart() {
        let dir = temp("signout");
        let mut state = Persisted::default();
        state.record_verified("2026-08-12T10:00:00Z".into(), Some("120:1760000000".into()));
        state.signed_out = true;
        state.forget_verification();
        save(&dir, &state).unwrap();

        let reloaded = load(&dir);
        assert!(reloaded.signed_out);
        assert!(!reloaded.verification_still_applies(Some("120:1760000000")));
    }

    /// Paths are remembered even after a sign-out: knowing where the program
    /// lives is not a credential, and forgetting it only buys a slow start.
    #[test]
    fn signing_out_does_not_forget_where_the_programs_are() {
        let mut state = Persisted {
            remembered: Remembered {
                cli_path: Some("/usr/local/bin/notebooklm".into()),
                ..Remembered::default()
            },
            ..Persisted::default()
        };
        state.signed_out = true;
        state.forget_verification();
        assert!(state.remembered.cli_path.is_some());
    }

    /// Nothing credential-shaped may reach this file.
    #[test]
    fn the_persisted_document_holds_no_secret_material() {
        let mut state = Persisted::default();
        state.remembered.cli_path = Some(r"C:\py\Scripts\notebooklm.exe".into());
        state.record_verified("2026-08-12T10:00:00Z".into(), Some("120:1760000000".into()));

        let json = serde_json::to_string(&state).unwrap().to_lowercase();
        for forbidden in ["cookie", "token", "psid", "secret", "password", "bearer"] {
            assert!(!json.contains(forbidden), "{forbidden} must never be persisted: {json}");
        }
    }
}
