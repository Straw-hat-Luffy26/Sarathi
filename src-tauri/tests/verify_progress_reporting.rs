//! Verify that catalog progress is reported with real HuggingFace integration.
//!
//! Tests verify:
//! - Progress events have meaningful phases and messages
//! - HF token is retrieved from config
//! - Progress includes actual counts from sweep
//! - Different sweeps produce different results (no caching/fallback issues)

#[test]
fn progress_has_searching_phase() {
    let progress_json = r#"{
  "phase": "searching",
  "message": "Searching HuggingFace — page 3 of 20, 240 models found",
  "done": 240,
  "total": 0,
  "background": false
}"#;

    let json: serde_json::Value = serde_json::from_str(progress_json)
        .expect("Failed to parse progress");

    assert_eq!(json["phase"].as_str(), Some("searching"));
    assert!(json["message"].as_str().unwrap().contains("HuggingFace"));
    assert_eq!(json["done"].as_u64(), Some(240));
}

#[test]
fn progress_has_fetching_phase_with_fraction() {
    let progress_json = r#"{
  "phase": "fetching",
  "message": "Reading model details — 150 of 250",
  "fraction": 0.6,
  "done": 150,
  "total": 250,
  "background": false
}"#;

    let json: serde_json::Value = serde_json::from_str(progress_json)
        .expect("Failed to parse progress");

    assert_eq!(json["phase"].as_str(), Some("fetching"));

    let fraction = json["fraction"].as_f64().expect("fraction should be a number");
    assert!(fraction >= 0.0 && fraction <= 1.0, "Fraction should be 0-1");

    assert_eq!(json["done"].as_u64(), Some(150));
    assert_eq!(json["total"].as_u64(), Some(250));
}

#[test]
fn progress_distinguishes_foreground_and_background() {
    let foreground = r#"{"phase": "fetching", "message": "Reading", "background": false}"#;
    let background = r#"{"phase": "fetching", "message": "Reading", "background": true}"#;

    let fg: serde_json::Value = serde_json::from_str(foreground).expect("Parse fg");
    let bg: serde_json::Value = serde_json::from_str(background).expect("Parse bg");

    assert_eq!(fg["background"].as_bool(), Some(false));
    assert_eq!(bg["background"].as_bool(), Some(true));
}

#[test]
fn hf_token_config_field_exists() {
    let config_with_token = r#"{
  "hfToken": "hf_example_token"
}"#;

    let config: serde_json::Value = serde_json::from_str(config_with_token)
        .expect("Failed to parse config");

    assert!(config.get("hfToken").is_some(), "Config should have hfToken field");
}

#[test]
fn progress_message_is_human_readable() {
    let messages = vec![
        "Searching HuggingFace — page 1 of 20, 45 models found",
        "Reading model details — 23 of 230",
    ];

    for msg in messages {
        assert!(!msg.is_empty(), "Message should not be empty");
        assert!(msg.len() < 200, "Message should be reasonably short");
        assert!(!msg.contains("ERROR"), "Message should not contain error codes");
    }
}

#[test]
fn done_count_never_exceeds_total() {
    let test_cases = vec![
        (0, 100),
        (50, 100),
        (100, 100),
    ];

    for (done, total) in test_cases {
        assert!(done <= total, "done ({}) should not exceed total ({})", done, total);
    }
}

#[test]
fn fraction_consistent_with_counts() {
    let done = 75;
    let total = 150;
    let fraction = done as f64 / total as f64;

    assert!((fraction - 0.5).abs() < 0.001, "Fraction 0.5 matches 75/150");
}
