//! Proves OpenClaw's context floor is met end to end, against the real model.
//!
//! OpenClaw's embedded agent refuses any model whose context is under 16000
//! tokens. Sarathi planned every load around an 8192-token working context, so
//! the generated `openclaw.json` said 8192 and the agent exited on
//! `Model context window too small (8192 tokens)` before its first turn.
//!
//! This runs the real path rather than a reconstruction of it: the actual
//! loader, the actual `ToolSpec`, the actual config renderer, and the actual
//! gateway. It then stays up so `openclaw agent --local` can be pointed at the
//! config it wrote and asked to do a turn.
//!
//! Run with:  cargo run --example verify_openclaw_context

use std::path::PathBuf;
use std::sync::Arc;

use sarathi_lib::ai_engine::manager::InferenceManager;
use sarathi_lib::ai_engine::scheduler::GenerationScheduler;
use sarathi_lib::gateway::server::start_gateway;
use sarathi_lib::gateway::state::{GatewayConfig, GatewayState};
use sarathi_lib::launcher::spec::{
    builtin_tools, fill_placeholders_with, resolve_env, LaunchContext, RuntimeSnapshot,
};

fn app_data_dir() -> PathBuf {
    PathBuf::from(std::env::var("APPDATA").expect("APPDATA")).join("com.sarathi.app")
}

/// The loader's own INFO lines are the evidence here, so they go to stdout.
struct Stdout;

impl log::Log for Stdout {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.level() <= log::Level::Info
    }
    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            println!("[{}] {}", record.level(), record.args());
        }
    }
    fn flush(&self) {}
}

#[tokio::main]
async fn main() {
    log::set_boxed_logger(Box::new(Stdout)).unwrap();
    log::set_max_level(log::LevelFilter::Info);

    let app_data = app_data_dir();
    let session: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(app_data.join("session.json")).unwrap())
            .unwrap();
    let provider = session["providerId"].as_str().unwrap().to_string();
    let model_id = session["modelId"].as_str().unwrap().to_string();
    let quant = session["quantization"].as_str().unwrap().to_string();

    println!("\n=== 1. Loading the model the user selected, unchanged ===");
    println!("    {provider} / {model_id} / {quant}");
    let inference = Arc::new(InferenceManager::new());
    let before = inference
        .load_installed_model_direct(&app_data, &provider, &model_id, &quant)
        .expect("load the user's model");
    println!("    loaded context: {}", before.context_length);

    let tool = builtin_tools()
        .into_iter()
        .find(|t| t.id == "openclaw")
        .expect("openclaw is a shipped tool");
    let floor = tool.min_context.expect("openclaw declares a context floor");
    println!("\n=== 2. OpenClaw declares a floor of {floor} tokens ===");

    let after = if before.context_length < floor {
        println!("    {} is below it; raising the load", before.context_length);
        inference
            .ensure_context_at_least(
                &app_data,
                floor,
                &tool.name,
                Some(|status: &str, step: Option<&str>| {
                    println!("    [{status}] {}", step.unwrap_or(""));
                }),
            )
            .expect("raise the loaded context")
    } else {
        before.clone()
    };
    println!("    loaded context now: {}", after.context_length);
    assert!(
        after.context_length >= 16_000,
        "still under OpenClaw's hard minimum: {}",
        after.context_length
    );
    assert_eq!(after.model_id, model_id, "the user's model must not be swapped");

    println!("\n=== 3. Starting the gateway ===");
    // Bound before the config is written, for the same reason the real launch
    // command reads `gateway.port()`: the configured port can be taken, and a
    // config naming one nothing is listening on is worse than no config.
    let scheduler = Arc::new(GenerationScheduler::start(inference.clone()));
    let state = Arc::new(GatewayState::new(scheduler, inference, GatewayConfig::default()));
    let handle = start_gateway(state).await.expect("gateway should bind");
    let port = handle.port;
    println!("    serving http://127.0.0.1:{port}");

    println!("\n=== 4. Writing the config through the real renderer ===");
    let client_dir = app_data.join("clients").join(&tool.id);
    std::fs::create_dir_all(&client_dir).unwrap();

    let ctx = LaunchContext {
        port,
        model_id: after.model_id.clone(),
        model_name: after.model_name.clone(),
        client_dir: client_dir.to_string_lossy().to_string(),
        context_length: after.context_length,
        mcp: sarathi_lib::launcher::mcp::load(&app_data),
        runtime: RuntimeSnapshot {
            quantization: Some(after.quantization.clone()),
            backend: Some(after.backend_used.clone()),
            gpu_layers: Some(after.gpu_layers),
            cpu_moe_layers: Some(after.cpu_moe_layers),
            ..Default::default()
        },
    };

    let cfg = tool.launch.client_config.as_ref().unwrap();
    let body = fill_placeholders_with(&cfg.contents, &ctx, true, cfg.mcp_dialect);
    let config_path = client_dir.join(&cfg.file_name);
    std::fs::write(&config_path, &body).unwrap();

    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    let model = &parsed["models"]["providers"]["sarathi"]["models"][0];
    println!("    {}", config_path.display());
    println!("    id            = {}", model["id"]);
    println!("    contextTokens = {}", model["contextTokens"]);
    println!("    contextWindow = {}", model["contextWindow"]);
    println!("    maxTokens     = {}", model["maxTokens"]);
    assert!(
        model["contextTokens"].as_u64().unwrap() >= 16_000,
        "the key OpenClaw's guard reads is still under its minimum"
    );

    let env = resolve_env(&tool.launch.env, &ctx);
    println!("\n=== 5. Environment OpenClaw is started with ===");
    for key in ["OPENCLAW_CONFIG_PATH", "OPENCLAW_STATE_DIR"] {
        println!("    {key} = {}", env[key]);
    }

    println!("\nREADY on port {port} — run the OpenClaw agent now. Kill this process to stop.");

    loop {
        tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
    }
}
