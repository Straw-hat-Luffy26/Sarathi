//! System IPC commands for Sarathi

use serde_json::json;
use tauri::AppHandle;
use tauri::Manager;

use crate::core::app_state::{get_app_state, AppStateData};
use crate::system_analyzer::{get_system_analyzer_manager, HardwareProfile, SystemValidationResult};

/// Returns basic application info
#[tauri::command]
pub async fn get_app_info(app: AppHandle) -> Result<serde_json::Value, String> {
    let data_dir = app
        .path()
        .app_data_dir()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    Ok(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "data_dir": data_dir,
    }))
}

/// Returns current application state
#[tauri::command]
pub async fn get_app_state_info() -> Result<AppStateData, String> {
    Ok(get_app_state().get())
}

/// Records activity to log (placeholder for future implementation)
#[tauri::command]
pub async fn log_activity(
    action: String,
    category: String,
    details: Option<String>,
) -> Result<(), String> {
    log::info!("Activity: {} [{}] - {:?}", action, category, details);
    Ok(())
}

/// Retrieves cached hardware profile if available
#[tauri::command]
pub async fn get_hardware_profile() -> Result<Option<HardwareProfile>, String> {
    Ok(get_system_analyzer_manager().get_profile())
}

/// Triggers full system analysis and updates cached profile
#[tauri::command]
pub async fn analyze_system() -> Result<HardwareProfile, String> {
    // Run on a blocking thread since analyze_system() spawns child processes
    tokio::task::spawn_blocking(move || {
        get_system_analyzer_manager()
            .analyze_system()
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}

/// Applies a manual override to a hardware/software profile field
#[tauri::command]
pub async fn override_hardware_value(
    field_path: String,
    value: serde_json::Value,
) -> Result<HardwareProfile, String> {
    get_system_analyzer_manager()
        .override_value(&field_path, value)
        .map_err(|e| e.to_string())
}

/// Reverts a hardware/software field override back to detected value
#[tauri::command]
pub async fn revert_hardware_override(field_path: String) -> Result<HardwareProfile, String> {
    get_system_analyzer_manager()
        .revert_override(&field_path)
        .map_err(|e| e.to_string())
}

/// Evaluates current system hardware readiness for AI model execution
#[tauri::command]
pub async fn validate_system() -> Result<SystemValidationResult, String> {
    let manager = get_system_analyzer_manager();
    if let Some(profile) = manager.get_profile() {
        Ok(profile.validation)
    } else {
        let profile = manager.analyze_system().map_err(|e| e.to_string())?;
        Ok(profile.validation)
    }
}

/// What the inference runtime can actually do on this machine, right now.
///
/// Exists because the most consequential fact about a Sarathi build is invisible
/// from inside it: GPU support in llama.cpp is compiled in, not detected at run
/// time, so a binary built without it will report a healthy NVIDIA card, plan an
/// offload for it, and then run every model on the CPU. Detection and capability
/// are different questions and this answers both.
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeCapability {
    /// True when this binary was compiled with a GPU backend.
    pub gpu_backend_compiled: bool,
    /// Which one — `cuda`, `vulkan`, or `none`.
    pub gpu_backend: &'static str,
    /// The card the loader would place a model on, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_gpu: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_gpu_vram_bytes: Option<u64>,
    /// True when a GPU is present that this build cannot use. The one state
    /// that must never be silent: everything looks right and nothing is.
    pub gpu_present_but_unusable: bool,
    /// Plain sentence for the UI, written for someone who will not read a log.
    pub summary: String,
}

#[tauri::command]
pub async fn get_runtime_capability() -> Result<RuntimeCapability, String> {
    let gpu_backend_compiled = cfg!(any(feature = "cuda", feature = "vulkan"));
    let gpu_backend = if cfg!(feature = "cuda") {
        "cuda"
    } else if cfg!(feature = "vulkan") {
        "vulkan"
    } else {
        "none"
    };

    let analyzer = crate::system_analyzer::get_system_analyzer_manager();
    let gpus = analyzer
        .get_profile()
        .map(|p| p.gpus.current().to_vec())
        .unwrap_or_default();
    let selected = crate::ai_engine::manager::select_inference_gpu(&gpus);

    let gpu_present_but_unusable = selected.is_some() && !gpu_backend_compiled;

    let summary = match (&selected, gpu_backend_compiled) {
        (Some(gpu), true) => format!(
            "Inference runs on {} ({:.1} GB) via {gpu_backend}.",
            gpu.model,
            gpu.vram_total_bytes as f64 / 1e9
        ),
        (Some(gpu), false) => format!(
            "{} was detected, but this build of Sarathi has no GPU backend compiled in, so every \
             model runs on the CPU. Run Sarathi with `npm start`, which builds the GPU backend.",
            gpu.model
        ),
        (None, true) => {
            "No usable GPU was detected. Models run on the CPU.".to_string()
        }
        (None, false) => "No usable GPU was detected, and this build is CPU-only.".to_string(),
    };

    Ok(RuntimeCapability {
        gpu_backend_compiled,
        gpu_backend,
        selected_gpu: selected.as_ref().map(|g| g.model.clone()),
        selected_gpu_vram_bytes: selected.as_ref().map(|g| g.vram_total_bytes),
        gpu_present_but_unusable,
        summary,
    })
}
