//! The bug that made a broken model look like a silent one.
//!
//! A generation failure used to travel as `finish_reason: "error: …"`. The
//! gateway forwarded it faithfully, which meant the client received HTTP 200,
//! `content: null`, and a `finish_reason` no OpenAI or Anthropic client knows.
//! Every one of them rendered exactly nothing. The user saw `Worked for 11s`
//! and an empty answer, and the actual reason — a prompt six times longer than
//! the loaded context — sat unread in the payload.
//!
//! These tests pin the two halves of the fix: failures are typed rather than
//! smuggled through a string, and they come back as HTTP errors that say what
//! to do.

use sarathi_lib::ai_engine::traits::{GenerationError, GenerationErrorKind, StreamChunk};

const OVERFLOW: &str = "Prompt is 48021 tokens but the model is loaded with a 8192-token \
     context. Load the model with a larger context, or send a shorter prompt.";

#[test]
fn a_context_overflow_is_a_client_error_with_a_code_clients_branch_on() {
    let failure = GenerationError::classify(OVERFLOW);

    assert_eq!(failure.kind, GenerationErrorKind::ContextLengthExceeded);
    assert_eq!(failure.status(), 400, "retrying this unchanged cannot work, so not a 5xx");
    assert_eq!(failure.code(), "context_length_exceeded");
}

#[test]
fn a_model_that_cannot_take_tools_says_so_specifically() {
    let failure = GenerationError::classify(
        "Qwen2.5-3B cannot be given tools: its chat template renders without them",
    );
    assert_eq!(failure.kind, GenerationErrorKind::ToolsUnsupported);
    assert_eq!(failure.status(), 400);
}

/// Anything unrecognised must still be an error, not a blank answer.
#[test]
fn an_unrecognised_failure_is_still_reported_rather_than_swallowed() {
    let failure = GenerationError::classify("the backend fell over");
    assert_eq!(failure.kind, GenerationErrorKind::Inference);
    assert_eq!(failure.status(), 500);
    assert!(!failure.message.is_empty(), "a failure with no message is the original bug");
}

/// The structural guarantee: a failure is a field of its own, so no consumer
/// can mistake it for a normal ending.
#[test]
fn a_failure_is_carried_as_a_failure_and_not_as_a_finish_reason() {
    let chunk = StreamChunk {
        text: String::new(),
        is_final: true,
        tokens_generated: Some(0),
        finish_reason: Some("error".to_string()),
        error: Some(GenerationError::classify(OVERFLOW)),
    };

    assert!(chunk.error.is_some(), "the gateway branches on this, not on the string");
    assert!(
        !chunk.finish_reason.as_deref().unwrap_or("").starts_with("error:"),
        "the old smuggling format must not come back: a client renders this field nowhere"
    );

    // And a normal ending still carries no error, so the branch is unambiguous.
    let normal = StreamChunk {
        text: "Hello!".into(),
        is_final: true,
        tokens_generated: Some(3),
        finish_reason: Some("stop".to_string()),
        error: None,
    };
    assert!(normal.error.is_none());
}

/// The regression guard for the sizing half.
///
/// Six MCP servers put ~43 000 tokens of tool schema into every request. A tool
/// that receives them must ask for room to hold them; asking for a fixed 8192
/// is what broke every provider at once.
#[test]
fn a_tool_given_mcp_servers_asks_for_room_to_hold_them() {
    use sarathi_lib::launcher::mcp::{McpRegistry, McpServerSpec};
    use sarathi_lib::launcher::spec::builtin_tools;

    let mut registry = McpRegistry::default();
    let claude = builtin_tools()
        .into_iter()
        .find(|t| t.id == "claude-code")
        .expect("Claude Code is a shipped tool");

    let bare = claude.preferred_context(&registry).expect("an MCP client wants room");

    for name in ["crawl4ai", "git", "notebooklm", "playwright", "research", "searxng"] {
        registry
            .servers
            .insert(name.to_string(), McpServerSpec::stdio(format!("{name}-server")));
    }
    let loaded = claude.preferred_context(&registry).expect("still an MCP client");

    assert!(
        loaded > bare,
        "adding servers must move the number: {bare} -> {loaded}"
    );
    assert!(
        loaded >= 48_000,
        "six servers measured 43k tokens of schema plus 5k of conversation; \
         asking for {loaded} would fail on the first real turn"
    );
}

/// A tool with no MCP support is not made to pay for tools it cannot receive.
#[test]
fn a_tool_without_mcp_support_asks_for_nothing_extra() {
    use sarathi_lib::launcher::mcp::McpRegistry;
    use sarathi_lib::launcher::spec::{McpSupport, ToolSpec};

    let mut plain: ToolSpec = sarathi_lib::launcher::spec::builtin_tools()
        .into_iter()
        .find(|t| t.id == "claude-code")
        .unwrap();
    plain.mcp = McpSupport::default();

    assert_eq!(plain.preferred_context(&McpRegistry::default()), None);
}
