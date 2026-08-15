//! The shared MCP server registry.
//!
//! One file — `mcp.json` in the app data directory — lists every MCP server on
//! this machine, in the `mcpServers` shape Claude Desktop established and most
//! clients now read. Sarathi hands that same set to every provider it launches,
//! so a server added once is available in all of them.
//!
//! The complication this module exists for is that clients agree on *what* an
//! MCP server is and disagree on how to write it down. opencode nests the
//! command and its arguments in one array under `mcp` and calls the environment
//! `environment`; OpenClaw wants `mcp.servers`; Hermes wants `mcp_servers` in
//! YAML; the rest take `command`/`args`/`env` under `mcpServers`. So the
//! registry is stored once in a common shape and rendered per client.
//!
//! **Nothing here starts a process.** A server is spawned by the client that
//! reads the generated config, which is what keeps the same registry usable
//! from a client Sarathi never launched, and what keeps Sarathi out of the
//! business of supervising other people's subprocesses. See
//! [`crate::launcher::spec::McpSupport`] for how a provider declares that it can
//! read one of these.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// File users edit to add their own MCP servers.
pub const USER_MCP_FILE: &str = "mcp.json";

/// How a client reaches an MCP server.
///
/// Only the three the supported clients actually implement. A transport is not
/// listed here until some provider Sarathi launches can really speak it —
/// advertising one that no client honours produces a config that validates and
/// a server that is never contacted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum McpTransport {
    /// A child process speaking JSON-RPC over stdin/stdout. The common case,
    /// and the only one every supported client can do.
    Stdio,
    /// HTTP with a Server-Sent Events response stream. The older remote shape.
    Sse,
    /// MCP's current remote transport: one HTTP endpoint, streamed responses.
    StreamableHttp,
}

impl McpTransport {
    pub fn is_remote(self) -> bool {
        !matches!(self, Self::Stdio)
    }
}

/// One MCP server, as written in `mcp.json`.
///
/// Deliberately a superset of what any single client takes: the registry
/// records what the *server* is, and each dialect renders the subset its client
/// understands. A field a client cannot express is dropped in that client's
/// rendering rather than being refused here.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpServerSpec {
    // ── stdio ───────────────────────────────────────────────────────────────
    /// Executable to spawn. Required for stdio, absent for a remote server.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
    /// Working directory for the spawned server.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,

    // ── remote ──────────────────────────────────────────────────────────────
    /// Endpoint of a remote server. Required when there is no `command`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Stated explicitly when the endpoint's shape is not the default; see
    /// [`McpServerSpec::transport`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transport: Option<McpTransport>,
    /// Headers sent with every request to a remote server — typically the
    /// authorization the endpoint needs.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,

    // ── common ──────────────────────────────────────────────────────────────
    /// Kept in the file but not handed to clients — how a server is turned off
    /// without losing how it was configured.
    #[serde(default)]
    pub disabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl McpServerSpec {
    /// A local stdio server, which is what most entries are.
    pub fn stdio(command: impl Into<String>) -> Self {
        Self { command: Some(command.into()), ..Self::default() }
    }

    /// How this server is reached.
    ///
    /// An explicit `transport` wins. Otherwise it is inferred: a `command` is
    /// stdio, and a bare `url` is streamable HTTP — MCP's current remote
    /// transport, and the one a server that does not say otherwise will be.
    pub fn transport(&self) -> McpTransport {
        match self.transport {
            Some(t) => t,
            None if self.command.is_some() => McpTransport::Stdio,
            None => McpTransport::StreamableHttp,
        }
    }

    /// Rejects an entry that could not work, naming what is wrong with it.
    ///
    /// Validation happens once, here, so every client receives the same set and
    /// a mistake is reported to the user rather than to a log file inside some
    /// provider's terminal.
    pub fn validate(&self, name: &str) -> Result<(), String> {
        if name.trim().is_empty() {
            return Err("an MCP server entry has no name".into());
        }

        let has_command = self.command.as_deref().is_some_and(|c| !c.trim().is_empty());
        let has_url = self.url.as_deref().is_some_and(|u| !u.trim().is_empty());

        match (has_command, has_url) {
            (false, false) => {
                return Err(format!(
                    "MCP server '{name}' has neither a command to run nor a url to reach"
                ))
            }
            (true, true) => {
                return Err(format!(
                    "MCP server '{name}' has both a command and a url; it can only be one \
                     of a local process and a remote endpoint"
                ))
            }
            _ => {}
        }

        if self.transport() == McpTransport::Stdio && has_url {
            return Err(format!("MCP server '{name}' declares stdio but gives a url"));
        }
        if self.transport().is_remote() && has_command {
            return Err(format!(
                "MCP server '{name}' declares a remote transport but gives a command"
            ));
        }

        if let Some(url) = self.url.as_deref().filter(|u| !u.trim().is_empty()) {
            let scheme_ok = url.starts_with("http://") || url.starts_with("https://");
            if !scheme_ok {
                return Err(format!(
                    "MCP server '{name}' has a url that is not http or https: '{url}'"
                ));
            }
        }

        if self.transport() == McpTransport::Stdio && !self.headers.is_empty() {
            return Err(format!(
                "MCP server '{name}' sets HTTP headers on a local process, which has none"
            ));
        }

        Ok(())
    }

    /// The server as this client wants it written.
    ///
    /// Returns `None` when the client cannot express this server at all — a
    /// remote server handed to a client with no remote transport, say. Silently
    /// rendering it as something else would produce a config the client accepts
    /// and a server it never reaches.
    fn render(&self, dialect: McpDialect) -> Option<serde_json::Value> {
        let transport = self.transport();
        if !dialect.supports(transport) {
            return None;
        }

        let command = self.command.clone().unwrap_or_default();
        let url = self.url.clone().unwrap_or_default();

        let value = match (dialect, transport) {
            // ── Claude Desktop's original file, which Claude Code reads ──────
            (McpDialect::Standard, McpTransport::Stdio) => {
                let mut v = serde_json::json!({
                    "type": "stdio",
                    "command": command,
                    "args": self.args,
                    "env": self.env,
                });
                if let Some(cwd) = &self.cwd {
                    v["cwd"] = serde_json::Value::from(cwd.clone());
                }
                v
            }
            (McpDialect::Standard, McpTransport::Sse) => serde_json::json!({
                "type": "sse", "url": url, "headers": self.headers,
            }),
            (McpDialect::Standard, McpTransport::StreamableHttp) => serde_json::json!({
                "type": "http", "url": url, "headers": self.headers,
            }),

            // ── opencode: one command array, `environment`, explicit enabled ─
            (McpDialect::Opencode, McpTransport::Stdio) => {
                let mut argv = vec![command];
                argv.extend(self.args.iter().cloned());
                let mut v = serde_json::json!({
                    "type": "local",
                    "command": argv,
                    "environment": self.env,
                    "enabled": true,
                });
                if let Some(cwd) = &self.cwd {
                    v["cwd"] = serde_json::Value::from(cwd.clone());
                }
                v
            }
            (McpDialect::Opencode, _) => serde_json::json!({
                "type": "remote", "url": url, "headers": self.headers, "enabled": true,
            }),

            // ── OpenClaw: `mcp.servers`, its own transport spelling ──────────
            (McpDialect::OpenClaw, McpTransport::Stdio) => {
                let mut v = serde_json::json!({
                    "command": command,
                    "args": self.args,
                    "env": self.env,
                });
                if let Some(cwd) = &self.cwd {
                    v["cwd"] = serde_json::Value::from(cwd.clone());
                }
                v
            }
            (McpDialect::OpenClaw, t) => serde_json::json!({
                "url": url,
                "transport": if t == McpTransport::Sse { "sse" } else { "streamable-http" },
                "headers": self.headers,
            }),

            // ── Hermes: `mcp_servers`, with an explicit enabled flag ─────────
            (McpDialect::Hermes, McpTransport::Stdio) => {
                let mut v = serde_json::json!({
                    "command": command,
                    "args": self.args,
                    "env": self.env,
                    "enabled": true,
                });
                if let Some(cwd) = &self.cwd {
                    v["cwd"] = serde_json::Value::from(cwd.clone());
                }
                v
            }
            (McpDialect::Hermes, _) => serde_json::json!({
                "url": url, "headers": self.headers, "enabled": true,
            }),
        };

        Some(value)
    }
}

/// How a given client wants its MCP servers written.
///
/// A dialect is a *translation*, never a policy: it says how to spell a server
/// for one client and knows nothing about which servers exist. Adding a server
/// touches no dialect; adding a client adds one variant here and nothing else.
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
    /// OpenClaw: the same server shape as Standard, but nested under
    /// `mcp.servers` and spelling its remote transports `sse` /
    /// `streamable-http`. See its `McpServerConfig` in `plugin-sdk`.
    #[serde(rename = "openclaw")]
    OpenClaw,
    /// Hermes Agent: `mcp_servers` in its YAML config, each entry carrying an
    /// explicit `enabled`. See `hermes_cli/mcp_config.py`.
    Hermes,
}

impl McpDialect {
    /// Whether a client speaking this dialect can reach a server this way.
    ///
    /// Every supported client does stdio. Remote transports are claimed only
    /// where the client's own configuration schema has a place to put them.
    pub fn supports(self, transport: McpTransport) -> bool {
        match transport {
            McpTransport::Stdio => true,
            // Claude Code (`type: sse|http`), opencode (`type: remote`),
            // OpenClaw (`transport: sse|streamable-http`) and Hermes (`url`)
            // all take a remote endpoint.
            McpTransport::Sse | McpTransport::StreamableHttp => true,
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

    from_str(&raw, &path.display().to_string())
}

/// Parses a registry document. Split out so the rules are testable without a
/// filesystem, and so a caller holding the text can reuse them.
pub fn from_str(raw: &str, source: &str) -> McpRegistry {
    let parsed: McpFile = match serde_json::from_str(raw) {
        Ok(f) => f,
        Err(e) => {
            return McpRegistry {
                servers: BTreeMap::new(),
                warnings: vec![format!(
                    "{source} is not valid JSON ({e}). No MCP servers were loaded."
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
        serde_json::Value::Object(self.render_map(dialect)).to_string()
    }

    /// As [`render`](Self::render), nested the way OpenClaw's config wants it:
    /// `{"servers": {…}}` under the caller's `mcp` key.
    pub fn render_openclaw_mcp(&self) -> String {
        serde_json::json!({ "servers": serde_json::Value::Object(self.render_map(McpDialect::OpenClaw)) })
            .to_string()
    }

    /// The servers as a YAML mapping, indented to sit under a `mcp_servers:`
    /// key at the top level of a document.
    ///
    /// Written as YAML rather than substituted as JSON because Hermes' config
    /// is YAML: a JSON object is valid YAML, but only as a flow mapping on one
    /// line, which a user opening the file to edit it could not work with.
    pub fn render_yaml_block(&self, dialect: McpDialect, indent: usize) -> String {
        let map = self.render_map(dialect);
        if map.is_empty() {
            return format!("{}{{}}\n", " ".repeat(indent));
        }

        let mut out = String::new();
        for (name, value) in map {
            out.push_str(&yaml_entry(&name, &value, indent));
        }
        out
    }

    fn render_map(&self, dialect: McpDialect) -> serde_json::Map<String, serde_json::Value> {
        self.servers
            .iter()
            .filter_map(|(name, spec)| spec.render(dialect).map(|v| (name.clone(), v)))
            .collect()
    }

    /// Names this dialect cannot express, so a caller can say so rather than
    /// letting a server disappear between the registry and a provider.
    pub fn unrepresentable(&self, dialect: McpDialect) -> Vec<String> {
        self.servers
            .iter()
            .filter(|(_, spec)| spec.render(dialect).is_none())
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// A whole `mcp.json`-shaped document, for clients told to read a file
    /// rather than having servers inlined into their own config.
    pub fn render_document(&self) -> String {
        serde_json::json!({ "mcpServers": serde_json::Value::Object(
            self.render_map(McpDialect::Standard)
        ) })
        .to_string()
    }

    pub fn is_empty(&self) -> bool {
        self.servers.is_empty()
    }

    /// Every server name, in the order they are rendered.
    pub fn names(&self) -> Vec<String> {
        self.servers.keys().cloned().collect()
    }
}

/// One `name: {…}` YAML entry, block-styled so the file stays editable.
fn yaml_entry(name: &str, value: &serde_json::Value, indent: usize) -> String {
    let pad = " ".repeat(indent);
    let mut out = format!("{pad}{}:\n", yaml_scalar(name));
    out.push_str(&yaml_value(value, indent + 2));
    out
}

fn yaml_value(value: &serde_json::Value, indent: usize) -> String {
    let pad = " ".repeat(indent);
    match value {
        serde_json::Value::Object(map) if map.is_empty() => format!("{pad}{{}}\n"),
        serde_json::Value::Object(map) => {
            let mut out = String::new();
            for (k, v) in map {
                match v {
                    serde_json::Value::Object(inner) if inner.is_empty() => {
                        out.push_str(&format!("{pad}{}: {{}}\n", yaml_scalar(k)));
                    }
                    serde_json::Value::Array(items) if items.is_empty() => {
                        out.push_str(&format!("{pad}{}: []\n", yaml_scalar(k)));
                    }
                    serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
                        out.push_str(&format!("{pad}{}:\n", yaml_scalar(k)));
                        out.push_str(&yaml_value(v, indent + 2));
                    }
                    scalar => {
                        out.push_str(&format!("{pad}{}: {}\n", yaml_scalar(k), yaml_inline(scalar)));
                    }
                }
            }
            out
        }
        serde_json::Value::Array(items) => {
            let mut out = String::new();
            for item in items {
                out.push_str(&format!("{pad}- {}\n", yaml_inline(item)));
            }
            out
        }
        scalar => format!("{pad}{}\n", yaml_inline(scalar)),
    }
}

/// A scalar as YAML. Everything textual is double-quoted: a Windows path, a
/// version-like string and the word `no` are each mis-typed by YAML's implicit
/// rules if left bare, and quoting is never wrong.
fn yaml_inline(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => yaml_scalar(s),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Null => "null".to_string(),
        other => other.to_string(),
    }
}

fn yaml_scalar(s: &str) -> String {
    serde_json::Value::from(s).to_string()
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

    const REMOTE: &str = r#"{
      "mcpServers": {
        "hosted": {
          "url": "https://mcp.example.com/v1",
          "transport": "streamable-http",
          "headers": { "Authorization": "Bearer abc" }
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
        assert_eq!(s.command.as_deref(), Some("npx"));
        assert_eq!(s.env["SEARXNG_URL"], "http://127.0.0.1:8888");
        assert_eq!(s.transport(), McpTransport::Stdio, "a command is stdio");
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
        assert_eq!(json["searxng"]["type"], "stdio");
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

    /// OpenClaw reads `mcp.servers`, not `mcpServers`. Writing the outer key
    /// wrongly is indistinguishable, from the user's side, from writing nothing.
    #[test]
    fn the_openclaw_dialect_nests_servers_under_its_own_key() {
        let dir = temp_dir("openclaw");
        write(&dir, SAMPLE);

        let json: serde_json::Value =
            serde_json::from_str(&load(&dir).render_openclaw_mcp()).unwrap();

        assert!(json["servers"].is_object(), "OpenClaw's McpConfig is {{servers: …}}");
        assert_eq!(json["servers"]["searxng"]["command"], "npx");
        assert_eq!(json["servers"]["searxng"]["args"][1], "mcp-searxng");
        assert_eq!(json["servers"]["searxng"]["env"]["SEARXNG_URL"], "http://127.0.0.1:8888");
    }

    #[test]
    fn the_hermes_dialect_renders_editable_yaml() {
        let dir = temp_dir("hermes");
        write(&dir, SAMPLE);

        let yaml = load(&dir).render_yaml_block(McpDialect::Hermes, 2);

        assert!(yaml.contains("  \"searxng\":\n"), "got:\n{yaml}");
        assert!(yaml.contains("    \"command\": \"npx\"\n"), "got:\n{yaml}");
        assert!(yaml.contains("      - \"mcp-searxng\"\n"), "args are a block list:\n{yaml}");
        assert!(yaml.contains("\"enabled\": true"), "Hermes takes an explicit enabled");
        // Every scalar quoted: an unquoted Windows path or a bare `no` is
        // re-typed by YAML's implicit rules.
        assert!(!yaml.contains("command: npx"), "scalars must be quoted:\n{yaml}");
    }

    #[test]
    fn a_windows_path_survives_every_dialect() {
        let dir = temp_dir("winpath");
        write(
            &dir,
            r#"{"mcpServers":{"git":{"command":"C:\\Users\\me\\.local\\bin\\mcp-server-git.exe"}}}"#,
        );
        let reg = load(&dir);
        let expected = r"C:\Users\me\.local\bin\mcp-server-git.exe";

        for dialect in [McpDialect::Standard, McpDialect::OpenClaw, McpDialect::Hermes] {
            let json: serde_json::Value = serde_json::from_str(&reg.render(dialect)).unwrap();
            assert_eq!(json["git"]["command"], expected, "{dialect:?}");
        }
        let oc: serde_json::Value =
            serde_json::from_str(&reg.render(McpDialect::Opencode)).unwrap();
        assert_eq!(oc["git"]["command"][0], expected);

        let yaml = reg.render_yaml_block(McpDialect::Hermes, 2);
        assert!(
            yaml.contains(r#""C:\\Users\\me\\.local\\bin\\mcp-server-git.exe""#),
            "backslashes must be escaped inside a quoted YAML scalar:\n{yaml}"
        );
    }

    #[test]
    fn a_remote_server_renders_in_each_clients_spelling() {
        let dir = temp_dir("remote");
        write(&dir, REMOTE);
        let reg = load(&dir);
        assert_eq!(reg.servers["hosted"].transport(), McpTransport::StreamableHttp);

        let std: serde_json::Value = serde_json::from_str(&reg.render(McpDialect::Standard)).unwrap();
        assert_eq!(std["hosted"]["type"], "http");
        assert_eq!(std["hosted"]["url"], "https://mcp.example.com/v1");

        let oc: serde_json::Value = serde_json::from_str(&reg.render(McpDialect::Opencode)).unwrap();
        assert_eq!(oc["hosted"]["type"], "remote");

        let claw: serde_json::Value = serde_json::from_str(&reg.render(McpDialect::OpenClaw)).unwrap();
        assert_eq!(claw["hosted"]["transport"], "streamable-http");

        let her: serde_json::Value = serde_json::from_str(&reg.render(McpDialect::Hermes)).unwrap();
        assert_eq!(her["hosted"]["url"], "https://mcp.example.com/v1");
    }

    #[test]
    fn an_sse_endpoint_keeps_its_own_transport_name() {
        let reg = from_str(
            r#"{"mcpServers":{"s":{"url":"https://x/sse","transport":"sse"}}}"#,
            "test",
        );
        let claw: serde_json::Value = serde_json::from_str(&reg.render(McpDialect::OpenClaw)).unwrap();
        assert_eq!(claw["s"]["transport"], "sse");
        let std: serde_json::Value = serde_json::from_str(&reg.render(McpDialect::Standard)).unwrap();
        assert_eq!(std["s"]["type"], "sse");
    }

    #[test]
    fn a_working_directory_reaches_the_clients_that_take_one() {
        let reg = from_str(r#"{"mcpServers":{"g":{"command":"x","cwd":"C:/repo"}}}"#, "test");
        for dialect in [McpDialect::Standard, McpDialect::Opencode, McpDialect::OpenClaw] {
            let json: serde_json::Value = serde_json::from_str(&reg.render(dialect)).unwrap();
            assert_eq!(json["g"]["cwd"], "C:/repo", "{dialect:?}");
        }
    }

    #[test]
    fn a_disabled_server_is_not_handed_to_clients() {
        let dir = temp_dir("disabled");
        write(&dir, r#"{"mcpServers":{"off":{"command":"x","disabled":true}}}"#);

        assert!(load(&dir).is_empty());
    }

    #[test]
    fn an_entry_with_neither_command_nor_url_is_skipped_with_a_reason() {
        let dir = temp_dir("nocommand");
        write(
            &dir,
            r#"{"mcpServers":{"broken":{"command":"  "},"fine":{"command":"npx"}}}"#,
        );

        let reg = load(&dir);
        assert!(!reg.servers.contains_key("broken"));
        assert!(reg.servers.contains_key("fine"), "one bad entry must not block the rest");
        assert_eq!(reg.warnings.len(), 1);
        assert!(reg.warnings[0].contains("neither a command"), "got: {}", reg.warnings[0]);
    }

    #[test]
    fn contradictory_entries_are_refused_rather_than_guessed_at() {
        for (body, needle) in [
            (r#"{"mcpServers":{"x":{"command":"a","url":"https://y"}}}"#, "both a command and a url"),
            (r#"{"mcpServers":{"x":{"url":"ftp://y"}}}"#, "not http or https"),
            (
                r#"{"mcpServers":{"x":{"command":"a","transport":"sse"}}}"#,
                "remote transport but gives a command",
            ),
            (
                r#"{"mcpServers":{"x":{"command":"a","headers":{"A":"b"}}}}"#,
                "headers on a local process",
            ),
        ] {
            let reg = from_str(body, "test");
            assert!(reg.servers.is_empty(), "should have been refused: {body}");
            assert_eq!(reg.warnings.len(), 1);
            assert!(
                reg.warnings[0].contains(needle),
                "warning should say why; got: {}",
                reg.warnings[0]
            );
        }
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
        for d in [McpDialect::Standard, McpDialect::Opencode, McpDialect::OpenClaw, McpDialect::Hermes] {
            assert_eq!(reg.render(d), "{}", "{d:?}");
        }
        assert_eq!(reg.render_openclaw_mcp(), r#"{"servers":{}}"#);
        assert_eq!(reg.render_yaml_block(McpDialect::Hermes, 2), "  {}\n");
    }

    #[test]
    fn the_rendered_document_is_a_valid_mcp_json_file() {
        let dir = temp_dir("document");
        write(&dir, SAMPLE);

        let body = load(&dir).render_document();
        let json: serde_json::Value = serde_json::from_str(&body).expect("valid JSON");
        assert_eq!(json["mcpServers"]["searxng"]["command"], "npx");

        // Round-trips: what Sarathi writes, Sarathi can read back.
        let reparsed = from_str(&body, "roundtrip");
        assert!(reparsed.servers.contains_key("searxng"));
        assert!(reparsed.warnings.is_empty(), "own output must validate: {:?}", reparsed.warnings);
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

    /// The property the whole design rests on: a server added to the registry
    /// reaches every dialect without any dialect knowing it exists.
    #[test]
    fn a_new_server_reaches_every_dialect_with_no_dialect_specific_code() {
        let reg = from_str(
            r#"{"mcpServers":{
                "brand-new-thing": {"command":"whatever","args":["--x"],"env":{"K":"v"}}
            }}"#,
            "test",
        );

        for dialect in [McpDialect::Standard, McpDialect::Opencode, McpDialect::OpenClaw, McpDialect::Hermes] {
            let json: serde_json::Value = serde_json::from_str(&reg.render(dialect)).unwrap();
            assert!(
                json.get("brand-new-thing").is_some(),
                "{dialect:?} lost a server it was never told about"
            );
            assert!(reg.unrepresentable(dialect).is_empty(), "{dialect:?}");
        }
    }
}
