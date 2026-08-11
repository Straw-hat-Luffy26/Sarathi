//! The shared MCP server registry.
//!
//! One file — `mcp.json` in the app data directory — lists every MCP server on
//! this machine, in the `mcpServers` shape Claude Desktop established and most
//! clients now read. Sarathi hands that same set to every tool it launches, so
//! a server added once is available in all of them.
//!
//! The complication this module exists for is that clients agree on *what* an
//! MCP server is and disagree on how to write it down. opencode nests the
//! command and its arguments in one array under `mcp` and calls the environment
//! `environment`; the rest take `command`/`args`/`env` under `mcpServers`. So
//! the registry is stored once in the common shape and rendered per client.
//!
//! Nothing here starts a process. A server is spawned by the client that reads
//! the generated config, which is what keeps the same registry usable from a
//! client Sarathi never launched.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// File users edit to add their own MCP servers.
pub const USER_MCP_FILE: &str = "mcp.json";

/// How a given client wants its MCP servers written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum McpDialect {
    /// `{"name": {"command", "args", "env"}}` — Claude Code, Cursor, Windsurf,
    /// Continue, and anything else following Claude Desktop's original file.
    #[default]
    Standard,
    /// opencode: command and arguments in one array, `environment` not `env`,
    /// and an explicit `type`/`enabled`.
    Opencode,
}

/// One MCP server, as written in `mcp.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpServerSpec {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Kept in the file but not handed to clients — how a server is turned off
    /// without losing how it was configured.
    #[serde(default)]
    pub disabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl McpServerSpec {
    fn validate(&self, name: &str) -> Result<(), String> {
        if name.trim().is_empty() {
            return Err("an MCP server entry has no name".into());
        }
        if self.command.trim().is_empty() {
            return Err(format!("MCP server '{name}' has no command to run"));
        }
        Ok(())
    }

    fn render(&self, dialect: McpDialect) -> serde_json::Value {
        match dialect {
            McpDialect::Standard => serde_json::json!({
                "command": self.command,
                "args": self.args,
                "env": self.env,
            }),
            McpDialect::Opencode => {
                let mut command = vec![self.command.clone()];
                command.extend(self.args.iter().cloned());
                serde_json::json!({
                    "type": "local",
                    "command": command,
                    "environment": self.env,
                    "enabled": true,
                })
            }
        }
    }
}

/// The registry as loaded, plus anything wrong with it.
#[derive(Debug, Clone, Default)]
pub struct McpRegistry {
    pub servers: BTreeMap<String, McpServerSpec>,
    pub warnings: Vec<String>,
}

/// The on-disk shape.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct McpFile {
    #[serde(default)]
    mcp_servers: BTreeMap<String, McpServerSpec>,
}

pub fn user_mcp_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join(USER_MCP_FILE)
}

/// Loads the registry, skipping entries that could not work.
///
/// A missing file is normal: Sarathi runs perfectly well with no MCP servers,
/// and inventing defaults that point at services the user has not installed
/// would give every client a set of tools that fail on first use.
pub fn load(app_data_dir: &Path) -> McpRegistry {
    let path = user_mcp_path(app_data_dir);
    if !path.is_file() {
        return McpRegistry::default();
    }

    let raw = match std::fs::read_to_string(&path) {
        Ok(r) => r,
        Err(e) => {
            return McpRegistry {
                servers: BTreeMap::new(),
                warnings: vec![format!("Could not read {}: {e}", path.display())],
            }
        }
    };

    let parsed: McpFile = match serde_json::from_str(&raw) {
        Ok(f) => f,
        Err(e) => {
            return McpRegistry {
                servers: BTreeMap::new(),
                warnings: vec![format!(
                    "{} is not valid JSON ({e}). No MCP servers were loaded.",
                    path.display()
                )],
            }
        }
    };

    let mut servers = BTreeMap::new();
    let mut warnings = Vec::new();

    for (name, spec) in parsed.mcp_servers {
        if let Err(reason) = spec.validate(&name) {
            warnings.push(format!("Ignored an entry in {USER_MCP_FILE}: {reason}"));
            continue;
        }
        if spec.disabled {
            continue;
        }
        servers.insert(name, spec);
    }

    McpRegistry { servers, warnings }
}

impl McpRegistry {
    /// The servers as a JSON object in `dialect`, ready to substitute into a
    /// generated client config.
    ///
    /// Always an object, never `null` or absent: a client that finds the key
    /// missing behaves differently from one that finds it empty, and "no
    /// servers" should mean the same thing everywhere.
    pub fn render(&self, dialect: McpDialect) -> String {
        let map: serde_json::Map<String, serde_json::Value> = self
            .servers
            .iter()
            .map(|(name, spec)| (name.clone(), spec.render(dialect)))
            .collect();
        serde_json::Value::Object(map).to_string()
    }

    /// A whole `mcp.json`-shaped document, for clients told to read a file
    /// rather than having servers inlined into their own config.
    pub fn render_document(&self) -> String {
        serde_json::json!({ "mcpServers": serde_json::from_str::<serde_json::Value>(
            &self.render(McpDialect::Standard)
        ).unwrap_or_else(|_| serde_json::json!({})) })
        .to_string()
    }

    pub fn is_empty(&self) -> bool {
        self.servers.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sarathi_mcp_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(dir: &Path, body: &str) {
        std::fs::write(user_mcp_path(dir), body).unwrap();
    }

    const SAMPLE: &str = r#"{
      "mcpServers": {
        "searxng": {
          "command": "npx",
          "args": ["-y", "mcp-searxng"],
          "env": { "SEARXNG_URL": "http://127.0.0.1:8888" }
        }
      }
    }"#;

    #[test]
    fn a_missing_file_is_normal_and_yields_no_servers() {
        let reg = load(&temp_dir("missing"));
        assert!(reg.is_empty());
        assert!(reg.warnings.is_empty(), "not having MCP servers is not a problem");
    }

    #[test]
    fn servers_are_loaded_from_disk() {
        let dir = temp_dir("valid");
        write(&dir, SAMPLE);

        let reg = load(&dir);
        assert_eq!(reg.servers.len(), 1);
        let s = &reg.servers["searxng"];
        assert_eq!(s.command, "npx");
        assert_eq!(s.env["SEARXNG_URL"], "http://127.0.0.1:8888");
    }

    #[test]
    fn the_standard_dialect_keeps_command_and_args_apart() {
        let dir = temp_dir("standard");
        write(&dir, SAMPLE);

        let json: serde_json::Value =
            serde_json::from_str(&load(&dir).render(McpDialect::Standard)).unwrap();

        assert_eq!(json["searxng"]["command"], "npx");
        assert_eq!(json["searxng"]["args"][1], "mcp-searxng");
        assert_eq!(json["searxng"]["env"]["SEARXNG_URL"], "http://127.0.0.1:8888");
    }

    #[test]
    fn the_opencode_dialect_merges_command_and_args_and_renames_env() {
        // opencode reads neither `args` nor `env`; getting this wrong produces a
        // config it accepts and a server it never starts.
        let dir = temp_dir("opencode");
        write(&dir, SAMPLE);

        let json: serde_json::Value =
            serde_json::from_str(&load(&dir).render(McpDialect::Opencode)).unwrap();

        assert_eq!(json["searxng"]["type"], "local");
        assert_eq!(json["searxng"]["command"][0], "npx");
        assert_eq!(json["searxng"]["command"][2], "mcp-searxng");
        assert_eq!(json["searxng"]["environment"]["SEARXNG_URL"], "http://127.0.0.1:8888");
        assert_eq!(json["searxng"]["enabled"], true);
        assert!(json["searxng"]["args"].is_null(), "opencode has no args key");
    }

    #[test]
    fn a_disabled_server_is_not_handed_to_clients() {
        let dir = temp_dir("disabled");
        write(&dir, r#"{"mcpServers":{"off":{"command":"x","disabled":true}}}"#);

        assert!(load(&dir).is_empty());
    }

    #[test]
    fn an_entry_with_no_command_is_skipped_with_a_reason() {
        let dir = temp_dir("nocommand");
        write(
            &dir,
            r#"{"mcpServers":{"broken":{"command":"  "},"fine":{"command":"npx"}}}"#,
        );

        let reg = load(&dir);
        assert!(!reg.servers.contains_key("broken"));
        assert!(reg.servers.contains_key("fine"), "one bad entry must not block the rest");
        assert_eq!(reg.warnings.len(), 1);
        assert!(reg.warnings[0].contains("no command"), "got: {}", reg.warnings[0]);
    }

    #[test]
    fn malformed_json_is_reported_rather_than_silently_ignored() {
        let dir = temp_dir("malformed");
        write(&dir, "{ not json");

        let reg = load(&dir);
        assert!(reg.is_empty());
        assert_eq!(reg.warnings.len(), 1);
        assert!(reg.warnings[0].contains("not valid JSON"));
    }

    #[test]
    fn an_empty_registry_renders_an_empty_object_not_null() {
        // A client that finds the key absent falls back to its own config; one
        // that finds it empty does not. Those must not be the same case.
        let reg = McpRegistry::default();
        assert_eq!(reg.render(McpDialect::Standard), "{}");
        assert_eq!(reg.render(McpDialect::Opencode), "{}");
    }

    #[test]
    fn the_rendered_document_is_a_valid_mcp_json_file() {
        let dir = temp_dir("document");
        write(&dir, SAMPLE);

        let body = load(&dir).render_document();
        let json: serde_json::Value = serde_json::from_str(&body).expect("valid JSON");
        assert_eq!(json["mcpServers"]["searxng"]["command"], "npx");

        // Round-trips: what Sarathi writes, Sarathi can read back.
        let reparsed: McpFile = serde_json::from_str(&body).unwrap();
        assert!(reparsed.mcp_servers.contains_key("searxng"));
    }

    #[test]
    fn rendering_is_stable_across_calls() {
        // Generated configs are written on every launch; an unstable key order
        // would rewrite the file each time and look like a change that matters.
        let dir = temp_dir("stable");
        write(
            &dir,
            r#"{"mcpServers":{"z":{"command":"a"},"a":{"command":"b"},"m":{"command":"c"}}}"#,
        );

        let reg = load(&dir);
        assert_eq!(reg.render(McpDialect::Standard), reg.render(McpDialect::Standard));
        let keys: Vec<&String> = reg.servers.keys().collect();
        assert_eq!(keys, vec!["a", "m", "z"], "sorted, not insertion order");
    }
}
