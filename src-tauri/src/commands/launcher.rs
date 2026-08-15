//! IPC for the Launch screen.
//!
//! Every command reports *why* something cannot happen rather than returning a
//! bare failure — the screen is aimed at people who will not read a log file,
//! so "npm is needed to install this, but it is not on this machine" has to
//! reach the UI intact.

use std::sync::Arc;

use tauri::{AppHandle, Emitter, Manager, State};

use crate::gateway::state::{GatewayState, GatewayStats};
use crate::launcher::{
    self, registry, spec::ToolSpec, LaunchedProcesses, ToolState, ToolStatus,
};

/// Everything the Launch screen renders in one call.
///
/// Bundled deliberately: the screen shows tools and server status together, and
/// two round trips could disagree — a tool marked Running beside a server shown
/// as stopped.
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LaunchOverview {
    pub tools: Vec<ToolStatus>,
    pub gateway: GatewayStats,
    /// Problems loading user tool definitions, surfaced rather than swallowed.
    pub warnings: Vec<String>,
    /// False when no model is loaded. Launching then produces a tool that
    /// connects and fails on its first question, so the UI disables Launch.
    pub can_launch: bool,
    /// Why launching is unavailable, when it is.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocked_reason: Option<String>,
}

/// Path of the file where MCP servers are defined, so the UI can point at it.
///
/// One file for every tool: this is what makes a server added once available
/// in Claude Code, opencode and anything else Sarathi launches.
#[tauri::command]
pub async fn user_mcp_file(app: AppHandle) -> Result<String, String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("could not resolve the app data folder: {e}"))?;
    Ok(launcher::mcp::user_mcp_path(&dir).to_string_lossy().to_string())
}

fn resolve_registry(app: &AppHandle) -> registry::Registry {
    match app.path().app_data_dir() {
        Ok(dir) => registry::load(&dir),
        // Without an app data directory there are no user entries, only the
        // shipped ones — still a usable screen.
        Err(_) => registry::Registry {
            tools: launcher::spec::builtin_tools(),
            warnings: vec!["Could not read the app data folder; custom tools were not loaded.".into()],
        },
    }
}

fn status_for(spec: &ToolSpec, procs: &LaunchedProcesses, detected: ToolState) -> ToolStatus {
    // A process Sarathi started takes precedence: it is the most specific thing
    // we know, and re-detecting would only report "installed".
    let state = match procs.pid_of(&spec.id) {
        Some(pid) => ToolState::Running { pid },
        None => detected,
    };

    // Only offer an install command when it could actually run.
    let install_command = spec.install.as_ref().and_then(|i| {
        launcher::manager_available(i.manager).then(|| i.manager.command_line(&i.package))
    });

    ToolStatus {
        id: spec.id.clone(),
        name: spec.name.clone(),
        description: spec.description.clone(),
        protocol: spec.protocol,
        user_defined: spec.user_defined,
        install_command,
        state,
    }
}

#[tauri::command]
pub async fn get_launch_overview(
    app: AppHandle,
    gateway: State<'_, Arc<GatewayState>>,
    procs: State<'_, Arc<LaunchedProcesses>>,
    detection: State<'_, Arc<launcher::DetectionCache>>,
) -> Result<LaunchOverview, String> {
    // The screen polls this every two seconds. If it ever costs more than a
    // frame, the log says so before a user has to call it "frozen".
    let _stage = crate::diagnostics::Stage::new("launcher: launch overview");

    let reg = resolve_registry(&app);
    let procs = procs.inner().clone();
    let detection = detection.inner().clone();

    // Answer from what is already known, and refresh behind the call when that
    // has gone stale. The screen polls every couple of seconds; a poll that
    // waited on `where` and `--version` for every tool is what made the first
    // paint take minutes, and made each late reply overlap the next.
    let specs = reg.tools.clone();
    let (states, stale) = detection.states(&specs);

    if stale {
        let cache = detection.clone();
        let for_refresh = specs.clone();
        tokio::task::spawn_blocking(move || cache.refresh(&for_refresh));
    }

    // Cheap: a `tasklist` per tool Sarathi actually started, and normally none.
    let tools = tokio::task::spawn_blocking(move || {
        specs
            .iter()
            .zip(states)
            .map(|(s, state)| status_for(s, &procs, state))
            .collect::<Vec<_>>()
    })
    .await
    .map_err(|e| format!("could not check installed tools: {e}"))?;

    let model = gateway.inference.get_loaded_model_info();
    let can_launch = model.is_some();

    // Reported from whether the server is actually up, not assumed.
    //
    // The handle is only managed once a listener is bound, so its presence is
    // the fact rather than an inference. Passing `true` unconditionally meant
    // the dashboard read "Running" while nothing was listening — which is
    // exactly the state a client sees as ConnectionRefused, with the one screen
    // that could have explained it insisting the server was fine.
    let gateway_running = app.try_state::<crate::gateway::GatewayHandle>().is_some();

    Ok(LaunchOverview {
        tools,
        gateway: gateway.stats(gateway_running),
        warnings: reg.warnings,
        can_launch,
        blocked_reason: (!can_launch)
            .then(|| "Load a model first — tools would connect but get no answer.".to_string()),
    })
}

/// Re-detects every tool, ignoring what was cached. The Refresh button.
///
/// Blocks until it has an answer, because the user asked a direct question and
/// a Refresh that returned instantly with the old answer would be worse than a
/// slow one.
#[tauri::command]
pub async fn redetect_tools(
    app: AppHandle,
    detection: State<'_, Arc<launcher::DetectionCache>>,
) -> Result<(), String> {
    let specs = resolve_registry(&app).tools;
    let detection = detection.inner().clone();
    tokio::task::spawn_blocking(move || {
        detection.forget_all();
        detection.refresh(&specs);
    })
    .await
    .map_err(|e| format!("detection task failed: {e}"))
}

/// The exact command an install would run, for the confirmation prompt.
///
/// Separate from [`install_tool`] so the UI can show it and wait for consent.
/// Nothing is executed here.
#[tauri::command]
pub async fn preview_tool_install(app: AppHandle, tool_id: String) -> Result<String, String> {
    let reg = resolve_registry(&app);
    let spec = reg
        .tools
        .iter()
        .find(|t| t.id == tool_id)
        .ok_or_else(|| format!("no tool called '{tool_id}'"))?;

    let install = spec
        .install
        .as_ref()
        .ok_or_else(|| format!("{} cannot be installed by Sarathi", spec.name))?;

    if !launcher::manager_available(install.manager) {
        return Err(format!(
            "{} is needed to install {}, but it is not available on this machine.",
            install.manager.program(),
            spec.name
        ));
    }

    Ok(install.manager.command_line(&install.package))
}

#[tauri::command]
pub async fn install_tool(
    app: AppHandle,
    detection: State<'_, Arc<launcher::DetectionCache>>,
    tool_id: String,
) -> Result<(), String> {
    let reg = resolve_registry(&app);
    let spec = reg
        .tools
        .iter()
        .find(|t| t.id == tool_id)
        .ok_or_else(|| format!("no tool called '{tool_id}'"))?
        .clone();

    // Installs are slow and run an external program. Each line the package
    // manager prints is forwarded to the UI as it arrives, so the card can show
    // what is actually happening instead of a spinner with no end in sight.
    let emitter = app.clone();
    let progress_id = tool_id.clone();

    let outcome = tokio::task::spawn_blocking(move || {
        launcher::install(&spec, |line| {
            let _ = emitter.emit(
                "tool:install-progress",
                serde_json::json!({ "toolId": progress_id, "line": line }),
            );
        })
    })
    .await
    .map_err(|e| format!("install task failed: {e}"))?;

    // Only now: detecting mid-install would have cached "not installed" and
    // held that answer for the whole TTL, right after the user watched it
    // install.
    detection.invalidate(&tool_id);
    outcome
}

#[tauri::command]
pub async fn launch_tool(
    app: AppHandle,
    gateway: State<'_, Arc<GatewayState>>,
    procs: State<'_, Arc<LaunchedProcesses>>,
    tool_id: String,
) -> Result<u32, String> {
    let Some(model) = gateway.inference.get_loaded_model_info() else {
        return Err("Load a model first — the tool would connect but get no answer.".into());
    };

    let reg = resolve_registry(&app);
    let spec = reg
        .tools
        .iter()
        .find(|t| t.id == tool_id)
        .ok_or_else(|| format!("no tool called '{tool_id}'"))?
        .clone();

    // A tool Sarathi already started keeps the terminal it is working in.
    //
    // Launching again would open a second window against the same workspace and
    // the same gateway, with two agents editing the same files. The card's Stop
    // is how a user detaches from one deliberately.
    if let Some(pid) = procs.live_pid(&tool_id) {
        log::info!("[LAUNCH] '{tool_id}' is already running (pid {pid}); not starting another");
        return Ok(pid);
    }

    let data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("could not locate the Sarathi data directory: {e}"))?;

    // A tool that refuses to run below a context floor of its own gets it met
    // before anything is written or started.
    //
    // The generated config states the context the model is really loaded with,
    // so this cannot be satisfied by writing a larger number: the runtime
    // rejects a prompt longer than its context, and an agent sized against a
    // figure Sarathi cannot honour fails on its first real turn instead of at
    // startup. Raising the load is the only version of this that is true.
    let model = match spec.min_context {
        Some(min) if model.context_length < min => {
            log::info!(
                "[LAUNCH] '{tool_id}' will not start below {min} tokens; the model is loaded with {}",
                model.context_length
            );
            let inference = gateway.inference.clone();
            let dir = data_dir.clone();
            let tool_name = spec.name.clone();
            // The model really is unloaded while this runs, so the rest of the
            // app is told rather than left showing a model that is not there.
            let app_for_status = app.clone();
            let status = move |status: &str, step: Option<&str>| {
                let _ = app_for_status.emit(
                    "inference:status",
                    crate::ai_engine::traits::InferenceStatusPayload {
                        status: status.to_string(),
                        step: step.map(|s| s.to_string()),
                        model: None,
                        error: None,
                    },
                );
            };
            tokio::task::spawn_blocking(move || {
                inference.ensure_context_at_least(&dir, min, &tool_name, Some(status))
            })
            .await
            .map_err(|e| format!("context reload task failed: {e}"))?
            .map_err(|e| format!("{e:#}"))?
        }
        _ => model,
    };

    // Then make room for the tool definitions this launch is about to hand it.
    //
    // A preference, not a floor: the model gives what it can and the launch goes
    // ahead either way. Without it, every MCP server Sarathi added made every
    // request larger while the context stayed where it was, until the prompt no
    // longer fit and every turn failed with nothing on screen to say why.
    let model = match spec.preferred_context(&crate::launcher::mcp::load(&data_dir)) {
        Some(wanted) if model.context_length < wanted => {
            log::info!(
                "[LAUNCH] '{tool_id}' is being handed MCP tool definitions; asking for {wanted} \
                 tokens of context (loaded with {})",
                model.context_length
            );
            let inference = gateway.inference.clone();
            let dir = data_dir.clone();
            let tool_name = spec.name.clone();
            let app_for_status = app.clone();
            let status = move |status: &str, step: Option<&str>| {
                let _ = app_for_status.emit(
                    "inference:status",
                    crate::ai_engine::traits::InferenceStatusPayload {
                        status: status.to_string(),
                        step: step.map(|s| s.to_string()),
                        model: None,
                        error: None,
                    },
                );
            };
            tokio::task::spawn_blocking(move || {
                inference.grow_context_towards(&dir, wanted, &tool_name, Some(status))
            })
            .await
            .map_err(|e| format!("context reload task failed: {e}"))?
            // Falling short is not a launch failure; the tool still runs, and a
            // request that does not fit now says so clearly.
            .unwrap_or(model)
        }
        _ => model,
    };

    // One workspace shared by every tool, and a private config directory per
    // tool. Two tools launched together therefore open on the same files while
    // keeping their own provider state isolated from each other and from the
    // user's own installs.
    let workspace = data_dir.join("workspace");
    let client_dir = data_dir.join("clients").join(&tool_id);

    // The live port, not the configured one: they differ if the configured
    // port was taken and the gateway bound elsewhere. Model likewise comes from
    // what is loaded right now, so the tool cannot open on a stale one.
    let port = gateway.port();

    // The same card the loader selected, read rather than recomputed — the
    // startup screen must not be able to disagree with the placement it reports.
    let selected_gpu = crate::system_analyzer::get_system_analyzer_manager()
        .get_profile()
        .and_then(|p| crate::ai_engine::manager::select_inference_gpu(&p.gpus.current()));
    // Loaded per launch rather than cached: a server added to mcp.json should
    // reach the next tool started, without restarting Sarathi.
    let mcp = launcher::mcp::load(&data_dir);
    for warning in &mcp.warnings {
        log::warn!("[LAUNCH] {warning}");
    }
    log::info!("[LAUNCH] Handing '{tool_id}' {} MCP server(s)", mcp.servers.len());

    let ctx = launcher::spec::LaunchContext {
        port,
        model_id: model.model_id.clone(),
        model_name: model.model_name.clone(),
        client_dir: client_dir.to_string_lossy().to_string(),
        context_length: model.context_length,
        mcp,
        // Taken from what is already in hand: the loaded model reports its own
        // placement, and the profile the loader planned against says which card
        // it chose. Nothing is detected a second time here.
        runtime: launcher::spec::RuntimeSnapshot {
            quantization: Some(model.quantization.clone()).filter(|q| !q.is_empty()),
            backend: Some(model.backend_used.clone()).filter(|b| !b.is_empty()),
            gpu_layers: Some(model.gpu_layers),
            cpu_moe_layers: Some(model.cpu_moe_layers),
            gpu_name: selected_gpu.as_ref().map(|g| g.model.clone()),
            vram_total_bytes: selected_gpu.as_ref().map(|g| g.vram_total_bytes),
            gpu_backend_compiled: cfg!(any(feature = "cuda", feature = "vulkan")),
        },
    };
    let model_label = model.model_name.clone();
    let tool_name = spec.name.clone();

    let pid = tokio::task::spawn_blocking(move || launcher::launch(&spec, &ctx, &workspace))
        .await
        .map_err(|e| format!("launch task failed: {e}"))??;

    procs.record(&tool_id, pid, &launcher::console::title_for(&tool_name));
    log::info!(
        "[LAUNCH] Started '{tool_id}' (pid {pid}) as a Sarathi client on port {port}, serving '{model_label}'"
    );
    Ok(pid)
}

/// Stops tracking a tool.
///
/// Sarathi does not kill the process: these are terminals and editors the user
/// may be working in, and closing one from another window would lose their work.
/// The card returns to Ready.
#[tauri::command]
pub async fn forget_tool_process(
    procs: State<'_, Arc<LaunchedProcesses>>,
    tool_id: String,
) -> Result<(), String> {
    procs.forget(&tool_id);
    Ok(())
}

/// Path of the file where custom tools are defined, so the UI can point at it.
#[tauri::command]
pub async fn user_tools_file(app: AppHandle) -> Result<String, String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("could not resolve the app data folder: {e}"))?;
    Ok(registry::user_tools_path(&dir).to_string_lossy().to_string())
}
