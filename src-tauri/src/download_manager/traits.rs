//! Download Manager Data Types and Structures

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DownloadStatus {
    Resolving,
    Queued,
    Downloading,
    Paused,
    Completed,
    Failed,
    Cancelled,
    Verifying,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadTask {
    pub id: String,
    pub model_id: String,
    pub model_name: String,
    pub provider_id: String,
    pub quantization: String,
    pub format: String,
    pub backend: String,
    pub url: String,
    pub destination_path: String,
    pub temp_path: String,
    pub total_bytes: u64,
    pub downloaded_bytes: u64,
    pub status: DownloadStatus,
    pub speed_bps: f64,
    pub eta_seconds: Option<u64>,
    pub checksum: Option<String>,
    pub error: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadProgressPayload {
    pub task_id: String,
    pub model_id: String,
    pub quantization: String,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub progress_percent: f64,
    pub speed_bps: f64,
    pub speed_formatted: String,
    pub eta_seconds: Option<u64>,
    pub status: DownloadStatus,
    pub error: Option<String>,
    pub package_id: Option<String>,
    pub capability: Option<String>,
    pub item_type: Option<String>, // "base_model" | "adapter"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InstalledModel {
    pub id: String,
    pub model_id: String,
    pub model_name: String,
    pub provider_id: String,
    pub quantization: String,
    pub format: String,
    pub backend: String,
    pub file_name: String,
    pub file_path: String,
    pub size_bytes: u64,
    pub installed_at: String,
    pub is_ready: bool,
    pub checksum: Option<String>,
    pub adapters: Option<std::collections::HashMap<String, crate::adapter_manager::AdapterManifestInfo>>,
    /// What the file on disk actually is, read from its GGUF header.
    ///
    /// Everything above this line comes from the manifest, which records what
    /// was *requested* when the model was downloaded. That is why it is not
    /// enough on its own: a request that fetched the wrong file is recorded just
    /// as faithfully as one that did not.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub classification: Option<crate::model_manager::classify::Classification>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StorageSummary {
    pub models_directory: String,
    pub total_installed_models: usize,
    pub total_models_bytes: u64,
    pub available_disk_space_bytes: u64,
    pub total_disk_space_bytes: u64,
}
