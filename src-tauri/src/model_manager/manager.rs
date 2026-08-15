//! Model Manager Engine
//!
//! Scans installed models directory, provides model deletion functionality,
//! and computes storage usage metrics.

use std::fs;
use std::path::Path;
use anyhow::Result;

use crate::download_manager::traits::{InstalledModel, StorageSummary};

pub struct ModelManager;

impl ModelManager {
    /// Scans `<app_data_dir>/models/` for model packages, on this thread.
    ///
    /// Kept as the plain blocking entry point for tests and the headless
    /// verification binaries. The running app goes through
    /// [`ModelStore`](crate::model_manager::ModelStore) instead, which shares one
    /// scan between callers, caches the header reads, and — the part that
    /// matters — never runs on the thread that draws the window. There is only
    /// one implementation of the walk; this delegates to it.
    pub fn list_installed_models(app_data_dir: &Path) -> Vec<InstalledModel> {
        crate::model_manager::ModelStore::scan_now(app_data_dir)
    }

    /// Deletes an installed model directory and files from disk
    pub fn delete_installed_model(app_data_dir: &Path, provider_id: &str, model_id: &str, quantization: &str) -> Result<()> {
        let clean_model_id = model_id.replace('/', "_");
        let package_dir = app_data_dir
            .join("models")
            .join(provider_id)
            .join(&clean_model_id);

        let legacy_quant_dir = package_dir.join(quantization);

        if package_dir.exists() {
            fs::remove_dir_all(&package_dir)?;
            log::info!("[MODEL_MANAGER] ✓ Deleted model package directory: {:?}", package_dir);
        } else if legacy_quant_dir.exists() {
            fs::remove_dir_all(&legacy_quant_dir)?;
            log::info!("[MODEL_MANAGER] ✓ Deleted model directory: {:?}", legacy_quant_dir);
        }

        Ok(())
    }

    /// Storage usage, on this thread. See [`list_installed_models`] on why the
    /// running app uses `ModelStore::summary` instead.
    ///
    /// [`list_installed_models`]: Self::list_installed_models
    pub fn get_storage_summary(app_data_dir: &Path) -> StorageSummary {
        let installed = Self::list_installed_models(app_data_dir);
        let models_dir = app_data_dir.join("models");
        let (available_disk_space_bytes, total_disk_space_bytes) =
            crate::model_manager::store::disk_space_for(&models_dir);

        StorageSummary {
            models_directory: models_dir.to_string_lossy().to_string(),
            total_installed_models: installed.len(),
            total_models_bytes: installed.iter().map(|m| m.size_bytes).sum(),
            available_disk_space_bytes,
            total_disk_space_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_installed_models_empty() {
        let temp_dir = std::env::temp_dir().join("sarathi_test_models_empty");
        let installed = ModelManager::list_installed_models(&temp_dir);
        assert_eq!(installed.len(), 0);
    }
}
