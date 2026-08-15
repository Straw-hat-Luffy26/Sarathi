//! Phase 4 Tauri Commands for Model Downloads & Storage Management
//!
//! **Every command here that touches the disk is `async fn` deliberately.**
//! Tauri runs a command declared as a plain `fn` inline on the thread that
//! received the IPC message — the main thread, which on Windows also pumps the
//! window's message loop. A synchronous `get_installed_models` therefore stopped
//! the window answering Windows for as long as the scan took, which is what
//! `Sarathi (Not Responding)` is. Async commands run on the async runtime
//! instead, and the blocking work inside them goes to `spawn_blocking`.

use tauri::{AppHandle, Manager, State};
use std::sync::Arc;
use anyhow::Result;

use crate::download_manager::traits::{DownloadTask, InstalledModel, StorageSummary};
use crate::download_manager::DownloadManager;
use crate::model_manager::{ModelManager, ModelStore};

#[tauri::command]
pub async fn start_model_download(
    app_handle: AppHandle,
    download_mgr: State<'_, Arc<DownloadManager>>,
    model_id: String,
    model_name: String,
    provider_id: String,
    quantization: String,
    format: String,
    backend: String,
    hf_token: Option<String>,
) -> Result<String, String> {
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve AppData directory: {}", e))?;

    download_mgr
        .start_download(
            app_handle.clone(),
            app_data_dir,
            model_id,
            model_name,
            provider_id,
            quantization,
            format,
            backend,
            hf_token,
        )
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn pause_model_download(
    download_mgr: State<'_, Arc<DownloadManager>>,
    task_id: String,
) -> Result<(), String> {
    download_mgr.pause_download(&task_id).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn resume_model_download(
    app_handle: AppHandle,
    download_mgr: State<'_, Arc<DownloadManager>>,
    task_id: String,
    hf_token: Option<String>,
) -> Result<String, String> {
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve AppData directory: {}", e))?;

    download_mgr
        .resume_download(app_handle.clone(), app_data_dir, &task_id, hf_token)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn cancel_model_download(
    app_handle: AppHandle,
    download_mgr: State<'_, Arc<DownloadManager>>,
    task_id: String,
) -> Result<(), String> {
    download_mgr.cancel_download(&app_handle, &task_id).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_active_downloads(
    download_mgr: State<'_, Arc<DownloadManager>>,
) -> Result<Vec<DownloadTask>, String> {
    Ok(download_mgr.list_tasks())
}

#[tauri::command]
pub async fn get_installed_models(
    app_handle: AppHandle,
    store: State<'_, Arc<ModelStore>>,
) -> Result<Vec<InstalledModel>, String> {
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve AppData directory: {}", e))?;

    // Cloned out of the shared listing rather than returned by reference: the
    // result is about to be serialised to the webview anyway, and holding the
    // `Arc` no longer would keep a scan alive past its usefulness.
    Ok(store.inner().listing(&app_data_dir).await.as_ref().clone())
}

#[tauri::command]
pub async fn delete_installed_model(
    app_handle: AppHandle,
    store: State<'_, Arc<ModelStore>>,
    provider_id: String,
    model_id: String,
    quantization: String,
) -> Result<(), String> {
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve AppData directory: {}", e))?;

    // Removing gigabytes is filesystem work; it does not belong on the UI thread
    // any more than reading them does.
    let dir = app_data_dir.clone();
    tokio::task::spawn_blocking(move || {
        ModelManager::delete_installed_model(&dir, &provider_id, &model_id, &quantization)
    })
    .await
    .map_err(|e| format!("delete task failed: {e}"))?
    .map_err(|e| e.to_string())?;

    // Sarathi changed the store itself, so the next look must not be answered
    // from the scan taken before the deletion.
    store.invalidate();
    Ok(())
}

#[tauri::command]
pub async fn get_storage_summary(
    app_handle: AppHandle,
    store: State<'_, Arc<ModelStore>>,
) -> Result<StorageSummary, String> {
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve AppData directory: {}", e))?;

    Ok(store.inner().summary(&app_data_dir).await)
}
