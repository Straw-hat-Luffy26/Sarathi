//! Putting NotebookLM into `mcp.json`, and taking it out again.
//!
//! This is the only place NotebookLM and the MCP registry meet, and it is
//! deliberately one function in each direction. Once the entry is in the file it
//! is an ordinary stdio server: the launcher, the dialects and every provider
//! treat it exactly like `git` or `searxng`, and none of them contains the word
//! NotebookLM. That is what makes "add a capability, every provider gets it"
//! true rather than aspirational.
//!
//! The file is rewritten preserving everything else in it, including the
//! comment block and any server the user added by hand, because it is a file
//! the user is invited to edit.

use std::path::Path;

use crate::launcher::mcp::{user_mcp_path, McpServerSpec};

use super::{mcp_server_spec, NotebookLmStatus, MCP_SERVER_KEY};

/// What a registry write did, so the caller can report it accurately.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegistryChange {
    Added,
    Updated,
    Removed,
    /// Already exactly as it should be.
    Unchanged,
}

/// Whether the registry currently lists NotebookLM.
pub fn is_registered(app_data_dir: &Path) -> bool {
    read_document(app_data_dir)
        .ok()
        .and_then(|doc| doc.get("mcpServers")?.get(MCP_SERVER_KEY).cloned())
        .is_some()
}

/// Adds or updates the NotebookLM entry, from a detected installation.
///
/// Refuses when the package cannot serve MCP: an entry pointing at a command
/// that does not exist would be handed to every provider, and each would report
/// its own failure to start it.
pub fn register(app_data_dir: &Path, status: &NotebookLmStatus) -> Result<RegistryChange, String> {
    let spec = mcp_server_spec(status).ok_or_else(|| {
        "NotebookLM's MCP server is not installed, so there is nothing to register".to_string()
    })?;
    spec.validate(MCP_SERVER_KEY)?;

    let mut doc = read_document(app_data_dir)?;
    let servers = ensure_servers(&mut doc);

    let desired = serde_json::to_value(&spec).map_err(|e| format!("could not encode entry: {e}"))?;
    let change = match servers.get(MCP_SERVER_KEY) {
        Some(existing) if *existing == desired => return Ok(RegistryChange::Unchanged),
        Some(_) => RegistryChange::Updated,
        None => RegistryChange::Added,
    };

    servers.insert(MCP_SERVER_KEY.to_string(), desired);
    write_document(app_data_dir, &doc)?;
    Ok(change)
}

/// Removes the entry, leaving anything else in the file alone.
pub fn unregister(app_data_dir: &Path) -> Result<RegistryChange, String> {
    let mut doc = read_document(app_data_dir)?;
    let servers = ensure_servers(&mut doc);
    if servers.remove(MCP_SERVER_KEY).is_none() {
        return Ok(RegistryChange::Unchanged);
    }
    write_document(app_data_dir, &doc)?;
    Ok(RegistryChange::Removed)
}

/// The whole document, so a rewrite keeps the parts this module does not own.
fn read_document(app_data_dir: &Path) -> Result<serde_json::Value, String> {
    let path = user_mcp_path(app_data_dir);
    if !path.is_file() {
        return Ok(serde_json::json!({ "mcpServers": {} }));
    }
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| format!("could not read {}: {e}", path.display()))?;
    serde_json::from_str(&raw).map_err(|e| {
        format!(
            "{} is not valid JSON ({e}); fix it by hand rather than let Sarathi overwrite it",
            path.display()
        )
    })
}

fn ensure_servers(doc: &mut serde_json::Value) -> &mut serde_json::Map<String, serde_json::Value> {
    if !doc.is_object() {
        *doc = serde_json::json!({});
    }
    let obj = doc.as_object_mut().expect("just made it an object");
    obj.entry("mcpServers").or_insert_with(|| serde_json::json!({}));
    if !obj["mcpServers"].is_object() {
        obj["mcpServers"] = serde_json::json!({});
    }
    obj["mcpServers"].as_object_mut().expect("just made it an object")
}

/// Writes through a temporary file, so an interrupted write cannot leave the
/// registry every provider reads as a truncated document.
fn write_document(app_data_dir: &Path, doc: &serde_json::Value) -> Result<(), String> {
    let path = user_mcp_path(app_data_dir);
    let body = serde_json::to_string_pretty(doc)
        .map_err(|e| format!("could not encode the registry: {e}"))?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("could not prepare {}: {e}", parent.display()))?;
    }

    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, format!("{body}\n"))
        .map_err(|e| format!("could not write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, &path)
        .map_err(|e| format!("could not replace {}: {e}", path.display()))?;
    Ok(())
}

/// The registered entry, ignoring `notebooklm`, for the "no NotebookLM here"
/// assertions the provider tests make.
#[cfg(test)]
pub fn other_server_names(app_data_dir: &Path) -> Vec<String> {
    read_document(app_data_dir)
        .ok()
        .and_then(|d| d.get("mcpServers")?.as_object().cloned())
        .map(|m| m.keys().filter(|k| *k != MCP_SERVER_KEY).cloned().collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sarathi_nlm_reg_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn status() -> NotebookLmStatus {
        let mut s = super::super::detect();
        s.mcp_server_path = Some(r"C:\py\Scripts\notebooklm-mcp.exe".into());
        s.mcp_available = true;
        s
    }

    #[test]
    fn registering_creates_an_entry_the_loader_accepts() {
        let dir = temp("create");
        assert!(!is_registered(&dir));

        assert_eq!(register(&dir, &status()).unwrap(), RegistryChange::Added);
        assert!(is_registered(&dir));

        // The real loader, not a re-parse of our own idea of the format.
        let reg = crate::launcher::mcp::load(&dir);
        assert!(reg.warnings.is_empty(), "own output must validate: {:?}", reg.warnings);
        let entry = reg.servers.get(MCP_SERVER_KEY).expect("listed");
        assert_eq!(entry.command.as_deref(), Some(r"C:\py\Scripts\notebooklm-mcp.exe"));
    }

    #[test]
    fn registering_twice_changes_nothing_the_second_time() {
        let dir = temp("idempotent");
        register(&dir, &status()).unwrap();
        assert_eq!(register(&dir, &status()).unwrap(), RegistryChange::Unchanged);
    }

    #[test]
    fn a_moved_installation_updates_rather_than_duplicates() {
        let dir = temp("moved");
        register(&dir, &status()).unwrap();

        let mut moved = status();
        moved.mcp_server_path = Some(r"D:\other\notebooklm-mcp.exe".into());
        assert_eq!(register(&dir, &moved).unwrap(), RegistryChange::Updated);

        let reg = crate::launcher::mcp::load(&dir);
        assert_eq!(
            reg.servers[MCP_SERVER_KEY].command.as_deref(),
            Some(r"D:\other\notebooklm-mcp.exe")
        );
        assert_eq!(reg.servers.len(), 1, "updated in place, not appended");
    }

    #[test]
    fn a_users_own_servers_and_comments_survive_a_write() {
        let dir = temp("preserve");
        std::fs::write(
            crate::launcher::mcp::user_mcp_path(&dir),
            r#"{
              "_comment": ["hand-written, keep me"],
              "mcpServers": { "mine": { "command": "my-server" } }
            }"#,
        )
        .unwrap();

        register(&dir, &status()).unwrap();

        let raw = std::fs::read_to_string(crate::launcher::mcp::user_mcp_path(&dir)).unwrap();
        assert!(raw.contains("hand-written, keep me"), "the comment block must survive");
        assert_eq!(other_server_names(&dir), vec!["mine".to_string()]);

        unregister(&dir).unwrap();
        assert!(!is_registered(&dir));
        assert_eq!(other_server_names(&dir), vec!["mine".to_string()], "removal is surgical");
    }

    #[test]
    fn an_uninstalled_notebooklm_cannot_be_registered() {
        let dir = temp("absent");
        let mut s = status();
        s.mcp_server_path = None;
        assert!(register(&dir, &s).is_err());
        assert!(!is_registered(&dir));
    }

    #[test]
    fn a_corrupt_registry_is_reported_rather_than_overwritten() {
        let dir = temp("corrupt");
        std::fs::write(crate::launcher::mcp::user_mcp_path(&dir), "{ not json").unwrap();

        let err = register(&dir, &status()).unwrap_err();
        assert!(err.contains("not valid JSON"), "got: {err}");
        assert_eq!(
            std::fs::read_to_string(crate::launcher::mcp::user_mcp_path(&dir)).unwrap(),
            "{ not json",
            "a file Sarathi could not understand must be left exactly as it was"
        );
    }
}
