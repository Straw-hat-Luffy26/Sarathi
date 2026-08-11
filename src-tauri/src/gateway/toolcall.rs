//! Recovering tool calls from what a local model actually emits.
//!
//! A hosted API returns tool calls as structured fields. A GGUF returns text,
//! and the shape of that text is decided by the chat template baked into the
//! model — so this is a parser, not a protocol.
//!
//! Three formats cover the models Sarathi serves:
//!
//! - ChatML family (Qwen, Hermes, most fine-tunes): `<tool_call>{…}</tool_call>`
//! - Llama 3.1: bare `{"name":…,"parameters":…}`, sometimes after `<|python_tag|>`
//! - Mistral / Mixtral: `[TOOL_CALLS] [{…}]`
//!
//! Everything else falls through as ordinary prose. That direction of failure is
//! the safe one: text that was meant as an answer arrives as an answer, whereas
//! a greedy parser turns a code sample about JSON into a phantom tool call.

use serde::Serialize;

/// One call the model asked for, in the shape both protocols serialise from.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ToolCall {
    /// Stable within a response; clients echo it back on the tool result.
    pub id: String,
    pub name: String,
    /// JSON object of arguments, as a string — what the OpenAI schema requires.
    pub arguments: String,
}

/// What a completion turned out to be.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedCompletion {
    /// Prose with any tool-call syntax removed. Empty when the model only called.
    pub text: String,
    pub calls: Vec<ToolCall>,
}

impl ParsedCompletion {
    pub fn is_tool_use(&self) -> bool {
        !self.calls.is_empty()
    }

    /// `tool_calls` when the model called something, otherwise the model's own
    /// reason. Clients branch on this to decide whether to run a tool.
    pub fn finish_reason<'a>(&self, natural: &'a str) -> &'a str
    where
        'a: 'a,
    {
        if self.is_tool_use() {
            "tool_calls"
        } else {
            natural
        }
    }
}

/// Extracts tool calls from raw model output.
///
/// `seq` seeds the generated call ids so two calls in one response never
/// collide, which some clients use as a map key.
pub fn parse(output: &str) -> ParsedCompletion {
    for extract in [extract_tagged, extract_mistral, extract_bare_json] {
        let (text, calls) = extract(output);
        if !calls.is_empty() {
            return ParsedCompletion { text: text.trim().to_string(), calls };
        }
    }

    ParsedCompletion { text: output.to_string(), calls: Vec::new() }
}

fn call_id(index: usize) -> String {
    format!("call_{:08x}{index}", fastrand_seed())
}

/// A per-process seed. Ids only need to be unique within one response, and
/// pulling in a random crate for that would be a dependency for nothing.
fn fastrand_seed() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0)
}

/// Turns one parsed JSON object into a call, if it looks like one.
///
/// Templates disagree on the argument key — `arguments` in ChatML and Mistral,
/// `parameters` in Llama 3.1 — so both are accepted.
fn call_from_value(value: &serde_json::Value, index: usize) -> Option<ToolCall> {
    let name = value.get("name")?.as_str()?.trim();
    if name.is_empty() {
        return None;
    }

    let args = value
        .get("arguments")
        .or_else(|| value.get("parameters"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));

    // Some models emit the arguments already stringified. Passing that through
    // as a JSON string would give the client `"\"{\\\"q\\\":…\""` to parse.
    let arguments = match args {
        serde_json::Value::String(s) => s,
        other => other.to_string(),
    };

    Some(ToolCall { id: call_id(index), name: name.to_string(), arguments })
}

/// `<tool_call>{…}</tool_call>`, the ChatML convention.
///
/// The closing tag is optional: models truncated at the token limit routinely
/// emit a complete JSON object and never close the tag, and discarding a call
/// that is entirely readable would be perverse.
fn extract_tagged(output: &str) -> (String, Vec<ToolCall>) {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";

    let mut text = String::new();
    let mut calls = Vec::new();
    let mut rest = output;

    while let Some(start) = rest.find(OPEN) {
        text.push_str(&rest[..start]);
        let after = &rest[start + OPEN.len()..];

        let (body, consumed) = match after.find(CLOSE) {
            Some(end) => (&after[..end], end + CLOSE.len()),
            None => (after, after.len()),
        };

        if let Ok(value) = serde_json::from_str::<serde_json::Value>(body.trim()) {
            if let Some(call) = call_from_value(&value, calls.len()) {
                calls.push(call);
            }
        }
        rest = &after[consumed..];
    }

    text.push_str(rest);
    (text, calls)
}

/// `[TOOL_CALLS] [{…}, {…}]`, the Mistral convention.
fn extract_mistral(output: &str) -> (String, Vec<ToolCall>) {
    const MARKER: &str = "[TOOL_CALLS]";

    let Some(start) = output.find(MARKER) else {
        return (output.to_string(), Vec::new());
    };

    let text = output[..start].to_string();
    let payload = output[start + MARKER.len()..].trim();

    let Some(json) = first_json_value(payload) else {
        return (output.to_string(), Vec::new());
    };

    let items = match &json {
        serde_json::Value::Array(items) => items.clone(),
        other => vec![other.clone()],
    };

    let calls: Vec<ToolCall> = items
        .iter()
        .enumerate()
        .filter_map(|(i, v)| call_from_value(v, i))
        .collect();

    if calls.is_empty() {
        return (output.to_string(), Vec::new());
    }
    (text, calls)
}

/// A bare `{"name":…,"parameters":…}` object, the Llama 3.1 convention.
///
/// The strictest of the three, because it is the one that could misfire: the
/// whole completion must be that object and nothing else. A reply that merely
/// *contains* JSON — an answer showing an example payload — is prose.
fn extract_bare_json(output: &str) -> (String, Vec<ToolCall>) {
    let trimmed = output
        .trim()
        .trim_start_matches("<|python_tag|>")
        .trim();

    if !trimmed.starts_with('{') && !trimmed.starts_with('[') {
        return (output.to_string(), Vec::new());
    }

    let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) else {
        return (output.to_string(), Vec::new());
    };

    let items = match &value {
        serde_json::Value::Array(items) => items.clone(),
        other => vec![other.clone()],
    };

    // Requiring an argument key as well as a name is what stops a model that
    // answers a question *about* JSON with `{"name": "Ada"}` being read as a
    // call to a tool named Ada.
    if !items.iter().all(|v| {
        v.get("name").is_some() && (v.get("arguments").is_some() || v.get("parameters").is_some())
    }) {
        return (output.to_string(), Vec::new());
    }

    let calls: Vec<ToolCall> = items
        .iter()
        .enumerate()
        .filter_map(|(i, v)| call_from_value(v, i))
        .collect();

    if calls.is_empty() {
        return (output.to_string(), Vec::new());
    }
    (String::new(), calls)
}

/// Reads the first complete JSON value at the start of `input`.
///
/// Needed because a model may follow its call with trailing prose, which
/// `serde_json::from_str` rejects as trailing characters.
fn first_json_value(input: &str) -> Option<serde_json::Value> {
    let mut de = serde_json::Deserializer::from_str(input).into_iter::<serde_json::Value>();
    de.next()?.ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args_of(c: &ToolCall) -> serde_json::Value {
        serde_json::from_str(&c.arguments).expect("arguments must be valid JSON")
    }

    #[test]
    fn plain_prose_is_not_a_tool_call() {
        let parsed = parse("The capital of France is Paris.");
        assert!(!parsed.is_tool_use());
        assert_eq!(parsed.text, "The capital of France is Paris.");
    }

    #[test]
    fn chatml_tool_call_is_extracted() {
        let parsed = parse(
            r#"<tool_call>{"name": "searxng_web_search", "arguments": {"query": "rust async"}}</tool_call>"#,
        );

        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].name, "searxng_web_search");
        assert_eq!(args_of(&parsed.calls[0])["query"], "rust async");
        assert_eq!(parsed.text, "", "the call syntax must not leak into the reply");
    }

    #[test]
    fn prose_around_a_tagged_call_is_kept() {
        let parsed = parse(
            "Let me look that up.\n<tool_call>{\"name\":\"research_search\",\"arguments\":{\"query\":\"x\"}}</tool_call>",
        );
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.text, "Let me look that up.");
    }

    #[test]
    fn two_calls_in_one_reply_both_survive_with_distinct_ids() {
        let parsed = parse(
            r#"<tool_call>{"name":"a","arguments":{}}</tool_call><tool_call>{"name":"b","arguments":{}}</tool_call>"#,
        );

        assert_eq!(parsed.calls.len(), 2);
        assert_eq!(parsed.calls[0].name, "a");
        assert_eq!(parsed.calls[1].name, "b");
        assert_ne!(parsed.calls[0].id, parsed.calls[1].id, "ids are used as map keys");
    }

    /// Hitting the token limit mid-tag is common, and the call is still readable.
    #[test]
    fn an_unclosed_tag_still_yields_the_call() {
        let parsed = parse(r#"<tool_call>{"name":"research_ask","arguments":{"question":"why"}}"#);
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].name, "research_ask");
    }

    #[test]
    fn llama_uses_parameters_where_chatml_uses_arguments() {
        let parsed = parse(r#"{"name": "git_log", "parameters": {"repo_path": "/tmp/x"}}"#);
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(args_of(&parsed.calls[0])["repo_path"], "/tmp/x");
    }

    #[test]
    fn the_llama_python_tag_is_stripped() {
        let parsed = parse(r#"<|python_tag|>{"name":"md","parameters":{"url":"https://x"}}"#);
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].name, "md");
    }

    #[test]
    fn mistral_tool_calls_are_extracted() {
        let parsed = parse(r#"[TOOL_CALLS] [{"name": "web_url_read", "arguments": {"url": "https://x"}}]"#);
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].name, "web_url_read");
    }

    #[test]
    fn mistral_trailing_prose_does_not_defeat_the_parse() {
        let parsed = parse(r#"[TOOL_CALLS] [{"name":"a","arguments":{}}] and then I will summarise"#);
        assert_eq!(parsed.calls.len(), 1);
    }

    /// The regression this parser's strictness exists for: a reply that merely
    /// contains JSON is an answer, not a call.
    #[test]
    fn json_in_an_ordinary_answer_is_not_mistaken_for_a_call() {
        for prose in [
            r#"{"name": "Ada Lovelace"}"#,
            r#"{"result": 42}"#,
            "Here is an example: {\"name\": \"x\", \"arguments\": {}} — note the shape.",
            "[1, 2, 3]",
        ] {
            let parsed = parse(prose);
            assert!(!parsed.is_tool_use(), "should be prose, got a call from: {prose}");
            assert_eq!(parsed.text, prose);
        }
    }

    #[test]
    fn malformed_json_inside_the_tag_is_left_as_text() {
        let parsed = parse("<tool_call>{not json at all}</tool_call>");
        assert!(!parsed.is_tool_use());
    }

    #[test]
    fn stringified_arguments_are_not_double_encoded() {
        // Some models emit the arguments already as a string; passing that
        // through verbatim would hand the client an escaped blob to unwrap.
        let parsed = parse(r#"<tool_call>{"name":"a","arguments":"{\"q\":\"hi\"}"}</tool_call>"#);
        assert_eq!(args_of(&parsed.calls[0])["q"], "hi");
    }

    #[test]
    fn a_call_with_no_arguments_gets_an_empty_object() {
        let parsed = parse(r#"<tool_call>{"name":"research_list_notebooks"}</tool_call>"#);
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(args_of(&parsed.calls[0]), serde_json::json!({}));
    }

    #[test]
    fn finish_reason_reports_tool_use_and_otherwise_defers() {
        let called = parse(r#"<tool_call>{"name":"a","arguments":{}}</tool_call>"#);
        assert_eq!(called.finish_reason("stop"), "tool_calls");

        let prose = parse("hello");
        assert_eq!(prose.finish_reason("length"), "length");
    }

    #[test]
    fn an_unnamed_call_is_rejected() {
        // A call with no name cannot be dispatched; better as text than as a
        // tool call the client fails on.
        let parsed = parse(r#"<tool_call>{"arguments":{"q":"x"}}</tool_call>"#);
        assert!(!parsed.is_tool_use());
    }
}
