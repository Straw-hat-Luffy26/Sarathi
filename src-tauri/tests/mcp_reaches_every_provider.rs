//! The MCP contract, enforced.
//!
//! Sarathi's MCP design rests on two claims:
//!
//! 1. **A new MCP server reaches every capable provider with no provider-side
//!    change.** Servers are data in `mcp.json`; providers declare a dialect.
//! 2. **A new provider receives every existing server** the moment it declares
//!    support, with no server-side change.
//!
//! Both were false at some point — Hermes and OpenClaw were handed nothing at
//! all while the startup screen reported their servers connected — and both are
//! the kind of thing that breaks silently, because a provider that receives no
//! servers looks exactly like a provider whose servers had no useful tools.
//! These tests are the alarm.

use std::collections::BTreeMap;

use sarathi_lib::launcher::mcp::{self, McpDialect, McpRegistry, McpServerSpec, McpTransport};
use sarathi_lib::launcher::spec::{
    builtin_tools, fill_placeholders_with, ClientConfig, DetectSpec, LaunchContext, LaunchSpec,
    McpSupport, Protocol, RuntimeSnapshot, ToolSpec, PLACEHOLDER_MCP, PLACEHOLDER_MCP_YAML,
};

/// A registry standing in for "whatever the user has configured", including a
/// server no provider was written with in mind.
fn registry() -> McpRegistry {
    mcp::from_str(
        r#"{
          "mcpServers": {
            "searxng":   {"command":"mcp-searxng","env":{"SEARXNG_URL":"http://127.0.0.1:8888"}},
            "git":       {"command":"C:\\bin\\mcp-server-git.exe"},
            "notebooklm":{"command":"C:\\py\\Scripts\\notebooklm-mcp.exe"},
            "brand-new": {"command":"node","args":["server.js"],"env":{"K":"v"}}
          }
        }"#,
        "test",
    )
}

fn ctx(reg: McpRegistry) -> LaunchContext {
    LaunchContext {
        port: 11435,
        model_id: "Qwen/Qwen2.5-Coder-7B".into(),
        model_name: "Qwen2.5 Coder 7B".into(),
        client_dir: r"C:\clients\x".into(),
        context_length: 32768,
        mcp: reg,
        runtime: RuntimeSnapshot::default(),
    }
}

/// The provider's generated config, as it would be written to disk.
fn rendered(tool: &ToolSpec, ctx: &LaunchContext) -> String {
    let cfg = tool.launch.client_config.as_ref().expect("an MCP provider generates a config");
    fill_placeholders_with(&cfg.contents, ctx, true, cfg.mcp_dialect)
}

/// Every shipped provider that declares MCP support must actually carry every
/// configured server into its generated config — by name, in a document its own
/// parser accepts.
#[test]
fn every_mcp_capable_provider_receives_every_configured_server() {
    let reg = registry();
    let ctx = ctx(reg.clone());
    let mut checked = 0;

    for tool in builtin_tools() {
        let Some(dialect) = tool.mcp.dialect() else { continue };
        checked += 1;

        let body = rendered(&tool, &ctx);
        for name in reg.names() {
            assert!(
                body.contains(&format!("\"{name}\"")),
                "'{}' declares MCP support but its config never mentions '{name}':\n{body}",
                tool.id
            );
        }

        // And the document has to parse, or the provider reads none of it.
        match dialect {
            McpDialect::Hermes => {
                assert!(body.contains("mcp_servers:"), "{}: {body}", tool.id);
                // Block style, one key per line — a flow mapping would be one
                // unreadable line in a file the user is invited to edit.
                assert!(body.contains("\n    \"command\":"), "{}: not block YAML:\n{body}", tool.id);
            }
            _ => {
                let parsed: serde_json::Value = serde_json::from_str(&body)
                    .unwrap_or_else(|e| panic!("{} produced invalid JSON: {e}\n{body}", tool.id));
                assert!(parsed.is_object(), "{}", tool.id);
            }
        }
    }

    assert!(checked >= 4, "expected the four shipped agents to declare MCP, saw {checked}");
}

/// Each provider's servers must land under the key that provider actually
/// reads. Writing them under the wrong key produces a config the client accepts
/// and servers it never starts, which is what happened to OpenClaw.
#[test]
fn servers_land_under_the_key_each_provider_really_reads() {
    let ctx = ctx(registry());

    let expected: BTreeMap<&str, &str> = BTreeMap::from([
        // Claude Code: `--mcp-config` file, Claude Desktop's original shape.
        ("claude-code", "mcpServers"),
        // opencode: top-level `mcp`.
        ("opencode", "mcp"),
        // OpenClaw: `mcp.servers` — see `McpConfig` in its plugin-sdk.
        ("openclaw", "mcp.servers"),
        // Hermes: `mcp_servers` in its YAML — see `hermes_cli/mcp_config.py`.
        ("hermes-agent", "mcp_servers"),
    ]);

    for (id, key) in expected {
        let tool = builtin_tools().into_iter().find(|t| t.id == id).expect(id);
        assert_eq!(tool.mcp.key(), Some(key), "{id} declares the wrong key");

        let body = rendered(&tool, &ctx);
        match key {
            "mcp.servers" => {
                let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
                assert!(
                    parsed["mcp"]["servers"]["searxng"].is_object(),
                    "openclaw must nest under mcp.servers, got:\n{body}"
                );
            }
            "mcp_servers" => {
                // Entries are sorted, so which one comes first is not the
                // point — that they sit indented under the key is.
                assert!(body.contains("\nmcp_servers:\n  \""), "got:\n{body}");
                assert!(body.contains("\n  \"git\":\n"), "got:\n{body}");
            }
            other => {
                let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
                assert!(parsed[other]["searxng"].is_object(), "{id} under {other}:\n{body}");
            }
        }
    }
}

/// Claim 1. A server nobody wrote code for reaches everything.
#[test]
fn a_server_added_today_needs_no_provider_change() {
    let ctx = ctx(mcp::from_str(
        r#"{"mcpServers":{"invented-after-the-fact":{"command":"x","args":["--y"]}}}"#,
        "test",
    ));

    for tool in builtin_tools().into_iter().filter(|t| t.mcp.is_supported()) {
        let body = rendered(&tool, &ctx);
        assert!(
            body.contains("invented-after-the-fact"),
            "'{}' would not have received a server added after it was written:\n{body}",
            tool.id
        );
    }
}

/// Claim 2. A provider added today receives everything already configured, by
/// declaring a dialect and placing the placeholder — and nothing else.
#[test]
fn a_provider_added_today_receives_every_existing_server() {
    let reg = registry();
    let ctx = ctx(reg.clone());

    // Exactly what a new provider's author has to write.
    let newcomer = ToolSpec {
        id: "future-agent".into(),
        name: "Future Agent".into(),
        description: "A provider that did not exist when the servers were added.".into(),
        protocol: Protocol::OpenAi,
        detect: DetectSpec {
            command: "future".into(),
            version_arg: "--version".into(),
            expect: "future".into(),
        },
        install: None,
        launch: LaunchSpec {
            command: "future".into(),
            args: vec![],
            env: Default::default(),
            env_remove: vec![],
            client_config: Some(ClientConfig {
                file_name: "future.json".into(),
                contents: format!("{{\"mcpServers\": {PLACEHOLDER_MCP}}}"),
                mcp_dialect: McpDialect::Standard,
            }),
        },
        mcp: McpSupport::Config { dialect: McpDialect::Standard, key: "mcpServers".into() },
        min_context: None,
        user_defined: true,
    };

    assert!(newcomer.validate().is_ok(), "a correctly declared provider must validate");

    let body = rendered(&newcomer, &ctx);
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    for name in reg.names() {
        assert!(
            parsed["mcpServers"][&name].is_object(),
            "a new provider must inherit '{name}' with no change to the registry"
        );
    }

    let delivery = newcomer.mcp_delivery(&reg);
    assert!(delivery.supported);
    assert_eq!(delivery.delivered.len(), reg.names().len());
    assert!(delivery.dropped.is_empty());
}

/// The declaration has to be binding, or it is decoration. A provider that says
/// it takes MCP servers and then never places them is exactly the bug this
/// whole mechanism exists to prevent, so it must fail validation.
#[test]
fn claiming_mcp_support_without_delivering_is_refused() {
    let mut broken = builtin_tools().into_iter().find(|t| t.id == "claude-code").unwrap();
    let cfg = broken.launch.client_config.as_mut().unwrap();
    cfg.contents = "{\"mcpServers\": {}}".to_string(); // placeholder removed
    broken.launch.args.retain(|a| !a.contains(PLACEHOLDER_MCP));

    let err = broken.validate().unwrap_err();
    assert!(err.contains("declares MCP support"), "got: {err}");
}

/// And the converse: delivering without declaring would leave the UI unable to
/// report it, which is the other half of the honesty rule.
#[test]
fn delivering_mcp_without_declaring_is_also_refused() {
    let mut sneaky = builtin_tools().into_iter().find(|t| t.id == "opencode").unwrap();
    sneaky.mcp = McpSupport::Unsupported { reason: "not stated".into() };

    let err = sneaky.validate().unwrap_err();
    assert!(err.contains("declares no MCP support"), "got: {err}");
}

/// A provider that genuinely has no MCP client must say so, and must be
/// reported as receiving nothing rather than as receiving the registry.
#[test]
fn an_unsupported_provider_reports_what_it_did_not_get() {
    let reg = registry();
    let tool = ToolSpec {
        mcp: McpSupport::Unsupported { reason: "this editor has no MCP client".into() },
        launch: LaunchSpec { client_config: None, ..dummy_launch() },
        ..dummy_tool()
    };

    let delivery = tool.mcp_delivery(&reg);
    assert!(!delivery.supported);
    assert!(delivery.delivered.is_empty(), "nothing was written anywhere");
    assert_eq!(delivery.dropped.len(), reg.names().len());
    assert_eq!(delivery.reason.as_deref(), Some("this editor has no MCP client"));
}

/// A disabled server must reach nobody, and a malformed one must not take the
/// rest of the registry down with it.
#[test]
fn disabled_and_malformed_entries_never_reach_a_provider() {
    let reg = mcp::from_str(
        r#"{"mcpServers":{
            "off":     {"command":"x","disabled":true},
            "broken":  {"description":"no command and no url"},
            "working": {"command":"y"}
        }}"#,
        "test",
    );
    assert_eq!(reg.names(), vec!["working".to_string()]);
    assert_eq!(reg.warnings.len(), 1, "the malformed entry is reported: {:?}", reg.warnings);

    let ctx = ctx(reg);
    for tool in builtin_tools().into_iter().filter(|t| t.mcp.is_supported()) {
        let body = rendered(&tool, &ctx);
        assert!(!body.contains("\"off\""), "{} received a disabled server:\n{body}", tool.id);
        assert!(!body.contains("\"broken\""), "{} received an invalid server:\n{body}", tool.id);
        assert!(body.contains("\"working\""), "{}:\n{body}", tool.id);
    }
}

/// Server environment is what carries an MCP server's own credentials. Losing
/// it produces a server that starts and then fails on its first call.
#[test]
fn server_environment_reaches_every_provider_intact() {
    let reg = mcp::from_str(
        r#"{"mcpServers":{"s":{"command":"x","env":{"API_KEY":"secret-value","URL":"http://h"}}}}"#,
        "test",
    );
    let ctx = ctx(reg);

    for tool in builtin_tools().into_iter().filter(|t| t.mcp.is_supported()) {
        let body = rendered(&tool, &ctx);
        assert!(body.contains("API_KEY"), "{} lost the env key:\n{body}", tool.id);
        assert!(body.contains("secret-value"), "{} lost the env value:\n{body}", tool.id);
        assert!(body.contains("http://h"), "{}:\n{body}", tool.id);
    }
}

/// Remote servers are only claimed where the client can really express them.
#[test]
fn a_remote_server_is_expressed_in_each_clients_own_spelling() {
    let reg = mcp::from_str(
        r#"{"mcpServers":{"hosted":{"url":"https://mcp.example.com/v1","transport":"sse"}}}"#,
        "test",
    );
    assert_eq!(reg.servers["hosted"].transport(), McpTransport::Sse);
    let ctx = ctx(reg.clone());

    for tool in builtin_tools().into_iter().filter(|t| t.mcp.is_supported()) {
        assert!(
            tool.mcp_delivery(&reg).dropped.is_empty(),
            "{} should be able to take a remote server",
            tool.id
        );
        let body = rendered(&tool, &ctx);
        assert!(body.contains("https://mcp.example.com/v1"), "{}:\n{body}", tool.id);
    }
}

/// Windows paths are the ordinary case here, and a lost backslash is a command
/// that cannot be found.
#[test]
fn windows_paths_survive_into_every_generated_config() {
    let reg = mcp::from_str(
        r#"{"mcpServers":{"git":{"command":"C:\\Users\\me\\.local\\bin\\mcp-server-git.exe"}}}"#,
        "test",
    );
    let ctx = ctx(reg);

    for tool in builtin_tools().into_iter().filter(|t| t.mcp.is_supported()) {
        let body = rendered(&tool, &ctx);
        let cfg = tool.launch.client_config.as_ref().unwrap();
        if cfg.mcp_dialect == McpDialect::Hermes {
            assert!(body.contains(r"C:\\Users\\me"), "{}:\n{body}", tool.id);
            continue;
        }
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        let found = serde_json::to_string(&parsed).unwrap();
        assert!(
            found.contains(r"C:\\Users\\me\\.local\\bin\\mcp-server-git.exe"),
            "{} mangled a Windows path:\n{found}",
            tool.id
        );
    }
}

/// NotebookLM is an ordinary server. Nothing in the launcher may know its name
/// — that is the whole test.
#[test]
fn notebooklm_needs_no_provider_specific_handling() {
    let reg = mcp::from_str(
        r#"{"mcpServers":{"notebooklm":{"command":"C:\\py\\Scripts\\notebooklm-mcp.exe"}}}"#,
        "test",
    );
    let ctx = ctx(reg.clone());

    for tool in builtin_tools().into_iter().filter(|t| t.mcp.is_supported()) {
        let body = rendered(&tool, &ctx);
        assert!(body.contains("notebooklm"), "{} did not receive NotebookLM:\n{body}", tool.id);
        assert!(tool.mcp_delivery(&reg).delivered.contains(&"notebooklm".to_string()));
    }

    // And the launcher's own source must not mention it. A grep is a blunt
    // instrument, but a NotebookLM branch inside a provider spec is exactly the
    // thing this architecture exists to make unnecessary.
    let launcher_src = concat!(
        include_str!("../src/launcher/spec.rs"),
        include_str!("../src/launcher/mcp.rs"),
        include_str!("../src/launcher/mod.rs"),
    );
    assert!(
        !launcher_src.to_lowercase().contains("notebooklm"),
        "the launcher names NotebookLM; it should only know about servers in general"
    );
}

fn dummy_tool() -> ToolSpec {
    ToolSpec {
        id: "dummy".into(),
        name: "Dummy".into(),
        description: String::new(),
        protocol: Protocol::OpenAi,
        detect: DetectSpec {
            command: "dummy".into(),
            version_arg: "--version".into(),
            expect: "dummy".into(),
        },
        install: None,
        launch: dummy_launch(),
        mcp: McpSupport::default(),
        min_context: None,
        user_defined: true,
    }
}

fn dummy_launch() -> LaunchSpec {
    LaunchSpec {
        command: "dummy".into(),
        args: vec![],
        env: Default::default(),
        env_remove: vec![],
        client_config: None,
    }
}

/// The YAML placeholder must nest at the indentation it appears at, or the
/// servers land as siblings of `mcp_servers:` and Hermes sees none.
#[test]
fn the_yaml_block_nests_where_the_placeholder_sits() {
    let ctx = ctx(registry());
    let body = fill_placeholders_with(
        &format!("mcp_servers:\n  {PLACEHOLDER_MCP_YAML}\n"),
        &ctx,
        true,
        McpDialect::Hermes,
    );

    assert!(body.starts_with("mcp_servers:\n"), "got:\n{body}");
    for line in body.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        assert!(line.starts_with("  "), "every entry must be indented under the key: {line:?}");
    }
    assert!(body.contains("\n  \"git\":\n"), "got:\n{body}");
}

/// An unused server spec must round-trip: what Sarathi writes into `mcp.json`
/// it has to be able to read back, or a capability that registers itself
/// disappears on the next launch.
#[test]
fn a_written_entry_reads_back_identically() {
    let spec = McpServerSpec {
        args: vec!["--flag".into()],
        env: BTreeMap::from([("K".into(), "v".into())]),
        cwd: Some(r"C:\work".into()),
        ..McpServerSpec::stdio(r"C:\bin\thing.exe")
    };
    let doc = serde_json::json!({ "mcpServers": { "thing": spec } }).to_string();

    let reg = mcp::from_str(&doc, "roundtrip");
    assert!(reg.warnings.is_empty(), "{:?}", reg.warnings);
    let back = &reg.servers["thing"];
    assert_eq!(back.command.as_deref(), Some(r"C:\bin\thing.exe"));
    assert_eq!(back.args, vec!["--flag".to_string()]);
    assert_eq!(back.cwd.as_deref(), Some(r"C:\work"));
    assert_eq!(back.env["K"], "v");
}
