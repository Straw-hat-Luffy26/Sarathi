//! Writes the client configs Sarathi would generate at launch, without launching.
//!
//! The point is to be able to check what a tool will actually receive — and to
//! hand a real generated config to that tool directly, which is how the MCP
//! wiring is verified against opencode and Claude Code rather than asserted.
//!
//! ```text
//! cargo run --example render_client_configs -- <app-data-dir> <out-dir> [port]
//! ```
//!
//! Reads the same `mcp.json` the launcher reads and renders through the same
//! `fill_placeholders_with`, so a config produced here is the one a launch
//! would write.

use std::path::PathBuf;

use sarathi_lib::launcher::{
    mcp,
    spec::{builtin_tools, fill_placeholders_with, resolve_args, resolve_env, LaunchContext},
};

fn main() {
    let mut args = std::env::args().skip(1);
    let app_data = PathBuf::from(args.next().expect("usage: <app-data-dir> <out-dir> [port]"));
    let out_dir = PathBuf::from(args.next().expect("usage: <app-data-dir> <out-dir> [port]"));
    let port: u16 = args.next().map(|p| p.parse().expect("port")).unwrap_or(11435);

    let registry = mcp::load(&app_data);
    for w in &registry.warnings {
        eprintln!("warning: {w}");
    }
    eprintln!("loaded {} MCP server(s) from {}", registry.servers.len(), app_data.display());

    for tool in builtin_tools() {
        let client_dir = out_dir.join(&tool.id);
        std::fs::create_dir_all(&client_dir).expect("create client dir");

        let ctx = LaunchContext {
            port,
            model_id: "local/model".into(),
            model_name: "Local Model".into(),
            client_dir: client_dir.to_string_lossy().to_string(),
            context_length: 32768,
            mcp: registry.clone(),
        };

        if let Some(cfg) = &tool.launch.client_config {
            let body = fill_placeholders_with(&cfg.contents, &ctx, true, cfg.mcp_dialect);
            let path = client_dir.join(&cfg.file_name);
            std::fs::write(&path, &body).expect("write config");
            println!("{}\t{}", tool.id, path.display());
        }

        let launch_args = resolve_args(&tool.launch.args, &ctx);
        if !launch_args.is_empty() {
            println!("{}\targs\t{}", tool.id, launch_args.join(" "));
        }
        let env = resolve_env(&tool.launch.env, &ctx);
        let mut keys: Vec<&String> = env.keys().collect();
        keys.sort();
        for k in keys {
            println!("{}\tenv\t{}={}", tool.id, k, env[k]);
        }
    }
}
