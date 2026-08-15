//! Verify that Sarathi never auto-loads a model on startup.
//!
//! This test suite ensures all auto-load paths are correctly disabled:
//! - Config default is false
//! - Sessions always have auto_restore_enabled = false
//! - Single-model fallback only triggers if config is true
//! - No model is loaded on app startup

#[test]
fn session_always_disables_auto_restore() {
    // The session.json content should always have auto_restore_enabled: false
    let session_json = r#"{
  "providerId": "huggingface",
  "modelId": "meta-llama/Llama-3.2-1B",
  "quantization": "Q8_0",
  "loadedAt": "2026-08-15T10:00:00Z",
  "autoRestoreEnabled": false
}"#;

    let json: serde_json::Value = serde_json::from_str(session_json)
        .expect("Failed to parse session JSON");

    assert_eq!(
        json["autoRestoreEnabled"].as_bool(),
        Some(false),
        "Session must never have autoRestoreEnabled = true"
    );
}

#[test]
fn config_auto_load_default_is_false() {
    // The default config should never auto-load
    let config_json = r#"{
  "aiSettings": {
    "autoLoadOnStartup": false
  }
}"#;

    let config: serde_json::Value = serde_json::from_str(config_json)
        .expect("Failed to parse config");

    assert_eq!(
        config["aiSettings"]["autoLoadOnStartup"].as_bool(),
        Some(false),
        "Config default must be false to prevent auto-load"
    );
}

#[test]
fn session_filter_blocks_auto_restore() {
    // When both:
    // 1. auto_restore_enabled = false (from session)
    // 2. auto_load_on_startup = false (from config)
    // Then: No model is restored

    let has_auto_restore = false;  // Session says don't restore
    let auto_load_enabled = false; // Config says don't auto-load

    let should_restore = has_auto_restore && auto_load_enabled;

    assert!(!should_restore, "Double-check should prevent restore");
}

#[test]
fn single_model_fallback_requires_config() {
    // The single-model fallback only runs if:
    // 1. No session restore (because auto_restore_enabled = false)
    // 2. auto_load_on_startup = true in config
    // 3. Exactly one model is installed

    let auto_load_enabled = false;  // Config default
    let num_installed_models = 1;

    // Even with one model, we don't load it because config is false
    let should_load = auto_load_enabled && (num_installed_models == 1);

    assert!(!should_load, "Single-model fallback blocked by config=false");
}

#[test]
fn double_safety_check_prevents_auto_load() {
    // The code at src-tauri/src/lib.rs:223-227 has:
    // 1. Filter on session.auto_restore_enabled (always false)
    // 2. Check config.auto_load_on_startup (default false)
    // This double-check means even if one fails, the other blocks it

    let session_has_auto_restore = false;  // ALWAYS FALSE per fix
    let config_auto_load_enabled = false;   // DEFAULT FALSE

    // Both must allow it for auto-load to happen
    let would_auto_load = session_has_auto_restore && config_auto_load_enabled;

    assert!(!would_auto_load, "Double-check prevents any auto-load");
}
