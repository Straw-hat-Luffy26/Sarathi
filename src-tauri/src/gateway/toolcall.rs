//! Recovering tool calls from what a local model actually emits.
//!
//! A hosted API returns tool calls as structured fields. A GGUF returns text,
//! and the shape of that text is decided by the chat template baked into the
//! model — so this is a parser, not a protocol.
//!
//! Five formats cover the models Sarathi serves:
//!
//! - ChatML family (Qwen, Hermes, most fine-tunes): `<tool_call>{…}</tool_call>`
//! - LFM2 (Liquid): `<|tool_call_start|>[name(k='v')]<|tool_call_end|>` — Python
//!   call syntax, not JSON
//! - Llama 3.1: bare `{"name":…,"parameters":…}`, sometimes after `<|python_tag|>`
//! - Mistral / Mixtral: `[TOOL_CALLS] [{…}]`
//! - A fenced ```json block, which no template asks for and smaller models
//!   emit anyway — a correct call wearing markdown.
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
    for extract in [
        extract_tagged,
        extract_lfm2,
        extract_mistral,
        extract_fenced_json,
        extract_xml_attributes,
        extract_bare_json,
    ] {
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

/// `<|tool_call_start|>[name(arg='value')]<|tool_call_end|>`, LFM2's convention.
///
/// Liquid's models write calls as *Python call syntax* rather than JSON — their
/// chat template builds `func_name(k=v, …)` and wraps the list in its own
/// tokens. It is the format the model was trained to emit, so a parser that
/// only knows JSON drops a perfectly good call from a model that did everything
/// right. Read the template's `render_tool_calls` macro to see the shape.
fn extract_lfm2(output: &str) -> (String, Vec<ToolCall>) {
    const OPEN: &str = "<|tool_call_start|>";
    const CLOSE: &str = "<|tool_call_end|>";

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

        // The list brackets are the template's, not part of any call.
        let inner = body.trim().trim_start_matches('[').trim_end_matches(']');
        for call in split_top_level(inner, ',') {
            if let Some(parsed) = python_call(call.trim(), calls.len()) {
                calls.push(parsed);
            }
        }
        rest = &after[consumed..];
    }

    text.push_str(rest);
    (text, calls)
}

/// `name(key='value', other=3)` into a call with JSON arguments.
fn python_call(source: &str, index: usize) -> Option<ToolCall> {
    let open = source.find('(')?;
    if !source.ends_with(')') {
        return None;
    }
    let name = source[..open].trim();
    if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-') {
        return None;
    }

    let mut args = serde_json::Map::new();
    for pair in split_top_level(&source[open + 1..source.len() - 1], ',') {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }
        let Some(eq) = split_top_level_once(pair, '=') else { continue };
        let (key, raw) = (pair[..eq].trim(), pair[eq + 1..].trim());
        if key.is_empty() {
            continue;
        }
        args.insert(key.to_string(), python_literal(raw));
    }

    Some(ToolCall {
        id: call_id(index),
        name: name.to_string(),
        arguments: serde_json::Value::Object(args).to_string(),
    })
}

/// A single argument value, in the forms the template can produce.
fn python_literal(raw: &str) -> serde_json::Value {
    // Quoted string — the template single-quotes every string argument.
    let unquoted = raw
        .strip_prefix('\'')
        .and_then(|s| s.strip_suffix('\''))
        .or_else(|| raw.strip_prefix('"').and_then(|s| s.strip_suffix('"')));
    if let Some(s) = unquoted {
        return serde_json::Value::from(s);
    }

    // An object or array went through `tojson`, so it is already JSON.
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(raw) {
        return v;
    }

    // Python's spelling of the booleans and null, then anything else as text.
    match raw {
        "True" => serde_json::Value::Bool(true),
        "False" => serde_json::Value::Bool(false),
        "None" => serde_json::Value::Null,
        other => serde_json::Value::from(other),
    }
}

/// Splits on `sep`, ignoring separators inside quotes, brackets or braces.
fn split_top_level(input: &str, sep: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut depth = 0i32;
    let mut quote: Option<char> = None;

    for (i, c) in input.char_indices() {
        match (quote, c) {
            (Some(q), c) if c == q => quote = None,
            (Some(_), _) => {}
            (None, '\'') | (None, '"') => quote = Some(c),
            (None, '(') | (None, '[') | (None, '{') => depth += 1,
            (None, ')') | (None, ']') | (None, '}') => depth -= 1,
            (None, c) if c == sep && depth == 0 => {
                parts.push(&input[start..i]);
                start = i + c.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(&input[start..]);
    parts
}

/// Byte offset of the first top-level `sep`, for splitting `key=value` once.
fn split_top_level_once(input: &str, sep: char) -> Option<usize> {
    let mut depth = 0i32;
    let mut quote: Option<char> = None;
    for (i, c) in input.char_indices() {
        match (quote, c) {
            (Some(q), c) if c == q => quote = None,
            (Some(_), _) => {}
            (None, '\'') | (None, '"') => quote = Some(c),
            (None, '(') | (None, '[') | (None, '{') => depth += 1,
            (None, ')') | (None, ']') | (None, '}') => depth -= 1,
            (None, c) if c == sep && depth == 0 => return Some(i),
            _ => {}
        }
    }
    None
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

/// A call inside a fenced code block: ```` ```json {"name":…,"arguments":…} ``` ````.
///
/// Not a format any template asks for — it is what a model does anyway. A 7B
/// told about tools through its system prompt frequently answers in the
/// markdown it was trained to write code in, and the call is entirely correct
/// apart from its wrapper. Refusing it means a model that *did* decide to use a
/// tool is reported as having chosen not to, which is the same silent failure
/// as dropping the definitions in the first place.
///
/// Held to the same standard as a bare object: a name *and* an argument key,
/// so a fenced example in an answer about JSON is not mistaken for a call.
fn extract_fenced_json(output: &str) -> (String, Vec<ToolCall>) {
    const FENCE: &str = "```";

    let mut calls = Vec::new();
    let mut text = String::new();
    let mut rest = output;

    while let Some(start) = rest.find(FENCE) {
        let after_open = &rest[start + FENCE.len()..];
        // The info string, e.g. `json`, up to the first newline.
        let Some(nl) = after_open.find('\n') else { break };
        let info = after_open[..nl].trim();
        let body_start = nl + 1;

        let Some(close) = after_open[body_start..].find(FENCE) else { break };
        let body = &after_open[body_start..body_start + close];
        let consumed = start + FENCE.len() + body_start + close + FENCE.len();

        // Only fences that claim to hold JSON, or none at all — a ```python
        // block is code the user asked for, not a call.
        //
        // `xml` is included because these models label the fence by habit
        // rather than by content: the body observed inside a ```xml fence was
        // the same JSON call object as inside a ```json one. The body still has
        // to parse as a call, so a genuine XML snippet is unaffected.
        let plausible = info.is_empty()
            || info.eq_ignore_ascii_case("json")
            || info.eq_ignore_ascii_case("xml");
        // Inside a fence the model has already declared this is data, so a
        // call buried in markup — `<tools>{…}</tools>` was one of the shapes
        // observed — is worth digging out. `calls_from_json_text` still
        // insists on a name and arguments, so wrapping cannot manufacture one.
        let found = plausible
            .then(|| {
                calls_from_json_text(body, calls.len()).or_else(|| {
                    let open = body.find('{')?;
                    calls_from_json_text(&body[open..], calls.len())
                })
            })
            .flatten();

        // Whatever the model said *before* the fence is prose either way. A
        // model that narrates before it calls ("I will search.") should not
        // lose the narration to the call.
        text.push_str(&rest[..start]);
        match found {
            Some(mut found) => calls.append(&mut found),
            None => text.push_str(&rest[start..consumed]),
        }
        rest = &rest[consumed..];
    }

    if calls.is_empty() {
        return (output.to_string(), Vec::new());
    }
    text.push_str(rest);
    (text, calls)
}

/// Parses `body` as one call or an array of them, requiring both keys.
/// Repairs the one malformation small models produce often enough to matter.
///
/// Qwen-class models trained on Jinja chat templates sometimes copy the
/// template's own delimiters into their answer and emit
/// `{{"name": "search", "arguments": {…}}}` — the call, correct in every
/// respect, wrapped in a doubled brace. Observed on three runs out of three
/// against `searxng_web_search`. Rejecting it loses a call the model got right,
/// and the user sees the raw JSON printed as prose instead of a search.
///
/// Deliberately narrow: only a leading `{{` with a matching trailing `}}`, and
/// only after the text has already failed to parse as JSON. Anything else is
/// left to fail, because guessing at broken output is how a parser starts
/// inventing calls nobody made.
/// Returns the text after one surplus opening brace, when there is one.
///
/// The brace count is usually wrong too — the observed emission opens with
/// `{{` and closes with a single `}}`, so it is not merely "wrapped", it is
/// unbalanced. Trimming the front and letting [`first_json_value`] read the
/// object that follows handles both the balanced and the unbalanced form
/// without guessing at what the closing braces were meant to be.
fn unwrap_doubled_braces(body: &str) -> Option<&str> {
    let trimmed = body.trim();
    let inner = trimmed.strip_prefix('{')?.trim_start();
    inner.starts_with('{').then_some(inner)
}

fn calls_from_json_text(body: &str, offset: usize) -> Option<Vec<ToolCall>> {
    let value = match serde_json::from_str::<serde_json::Value>(body.trim()) {
        Ok(value) => value,
        // Trailing markup or a surplus brace: read the first complete value
        // and let the name/arguments check below decide whether it is a call.
        Err(_) => match unwrap_doubled_braces(body) {
            Some(inner) => first_json_value(inner)?,
            None => first_json_value(body.trim())?,
        },
    };
    let items = match &value {
        serde_json::Value::Array(items) => items.clone(),
        other => vec![other.clone()],
    };

    if items.is_empty()
        || !items.iter().all(|v| {
            v.get("name").is_some()
                && (v.get("arguments").is_some() || v.get("parameters").is_some())
        })
    {
        return None;
    }

    let calls: Vec<ToolCall> = items
        .iter()
        .enumerate()
        .filter_map(|(i, v)| call_from_value(v, offset + i))
        .collect();

    (!calls.is_empty()).then_some(calls)
}

/// A bare `{"name":…,"parameters":…}` object, the Llama 3.1 convention.
///
/// The strictest of them, because it is the one that could misfire: the whole
/// completion must be that object and nothing else. A reply that merely
/// *contains* JSON — an answer showing an example payload — is prose.
fn extract_bare_json(output: &str) -> (String, Vec<ToolCall>) {
    let trimmed = output
        .trim()
        .trim_start_matches("<|python_tag|>")
        .trim();

    if !trimmed.starts_with('{') && !trimmed.starts_with('[') {
        return (output.to_string(), Vec::new());
    }

    // Same repair as the fenced path: a call the model doubled the braces on.
    let repaired = match serde_json::from_str::<serde_json::Value>(trimmed) {
        Ok(value) => Some(value),
        Err(_) => unwrap_doubled_braces(trimmed).and_then(first_json_value),
    };

    let Some(value) = repaired else {
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

/// `<function name="x" arguments='{…}'/>` and `<tool name="x" …/>`.
///
/// Not a format any chat template asks for — it is what a small model produces
/// when it has been shown JSON tool schemas and reaches for markup anyway.
/// Measured on Qwen2.5 through this gateway, four consecutive attempts to call
/// one tool produced four different shapes, two of them this one. Each was
/// returned to the client as prose, so the tool was never run and the user got
/// a paragraph of XML instead of a search.
///
/// Kept narrow on purpose: an element must carry both a `name` and an
/// `arguments` attribute, and the arguments must parse as a JSON object. A
/// document that merely contains angle brackets cannot match.
fn extract_xml_attributes(output: &str) -> (String, Vec<ToolCall>) {
    const ELEMENTS: [&str; 3] = ["<function", "<tool ", "<tool_call "];

    let mut calls = Vec::new();
    for element in ELEMENTS {
        let mut rest = output;
        while let Some(start) = rest.find(element) {
            let after = &rest[start + element.len()..];
            let Some(end) = after.find("/>").or_else(|| after.find('>')) else { break };
            let attrs = &after[..end];

            if let (Some(name), Some(args)) = (attribute(attrs, "name"), attribute(attrs, "arguments"))
            {
                if !name.is_empty() {
                    if let Ok(value) = serde_json::from_str::<serde_json::Value>(args.trim()) {
                        if value.is_object() {
                            calls.push(ToolCall {
                                id: call_id(calls.len()),
                                name: name.to_string(),
                                arguments: value.to_string(),
                            });
                        }
                    }
                }
            }
            rest = &after[end..];
        }
        if !calls.is_empty() {
            break;
        }
    }

    if calls.is_empty() {
        return (output.to_string(), Vec::new());
    }
    // The model wrapped the whole answer in markup; there is no prose to keep.
    (String::new(), calls)
}

/// `key="value"` or `key='value'` out of an attribute list.
fn attribute<'a>(attrs: &'a str, key: &str) -> Option<&'a str> {
    let at = attrs.find(key)?;
    let after = attrs[at + key.len()..].trim_start();
    let after = after.strip_prefix('=')?.trim_start();
    let quote = after.chars().next().filter(|c| *c == '"' || *c == '\'')?;
    let body = &after[quote.len_utf8()..];
    let end = body.find(quote)?;
    Some(&body[..end])
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
mod doubled_brace_tests {
    use super::*;

    /// Verbatim from Qwen2.5 answering a `searxng_web_search` prompt through
    /// the gateway. Three runs out of three produced this; every one was
    /// returned to the client as prose, so the search never happened.
    #[test]
    fn a_call_wrapped_in_the_templates_own_braces_is_still_a_call() {
        let raw = r#"{{"name": "searxng_web_search", "arguments": {"query": "weather in Delhi"}}}"#;
        let parsed = parse(raw);

        assert_eq!(parsed.calls.len(), 1, "got: {parsed:?}");
        assert_eq!(parsed.calls[0].name, "searxng_web_search");
        assert!(parsed.calls[0].arguments.contains("Delhi"));
        assert_eq!(parsed.finish_reason("stop"), "tool_calls");
    }

    #[test]
    fn the_same_call_inside_a_mislabelled_xml_fence_is_recognised() {
        let raw = "```xml\n{\"name\": \"searxng_web_search\", \"arguments\": {\"query\": \"x\"}}\n```";
        assert_eq!(parse(raw).calls.len(), 1, "the fence's label is not the body");
    }

    /// The other three shapes the same model produced for the same prompt, all
    /// captured verbatim from the gateway.
    #[test]
    fn every_shape_this_model_actually_emitted_is_recognised() {
        let observed = [
            r#"{{"name": "searxng_web_search", "arguments": {"query": "weather in Delhi"}}"#,
            "```xml\n<tools>\n  {\"name\": \"searxng_web_search\", \"arguments\": {\"query\": \"weather\"}}\n</tools>\n```",
            "```xml\n<function name=\"searxng_web_search\" arguments='{\"query\": \"weather\"}'/>\n```",
            "```xml\n<tools>\n  <tool name=\"searxng_web_search\" arguments='{\"query\": \"weather\"}' />\n</tools>\n```",
        ];

        for raw in observed {
            let parsed = parse(raw);
            assert_eq!(parsed.calls.len(), 1, "not recognised: {raw}");
            assert_eq!(parsed.calls[0].name, "searxng_web_search");
            assert!(parsed.calls[0].arguments.contains("weather"), "{:?}", parsed.calls[0]);
        }
    }

    #[test]
    fn markup_that_is_not_a_call_stays_prose() {
        for prose in [
            "<function>do a thing</function>",
            "<tool name=\"x\" />",
            "<function name=\"x\" arguments='not json'/>",
            "Use the <tool> element to declare one.",
        ] {
            assert!(parse(prose).calls.is_empty(), "invented a call from: {prose}");
        }
    }

    /// The repair must not start inventing calls out of ordinary output.
    #[test]
    fn nothing_that_is_not_a_call_becomes_one() {
        for prose in [
            "Here is a JSON object: {{\"a\": 1}}",
            "{{\"arguments\": {\"query\": \"x\"}}}",
            "```xml\n<note>hello</note>\n```",
            "```python\nprint({\"name\": \"x\", \"arguments\": {}})\n```",
            "I could search for that if you like.",
        ] {
            assert!(parse(prose).calls.is_empty(), "invented a call from: {prose}");
        }
    }

    /// A single brace pair is ordinary JSON and must keep working.
    #[test]
    fn the_normal_form_is_untouched() {
        let raw = r#"<tool_call>{"name": "git_log", "arguments": {"n": 3}}</tool_call>"#;
        assert_eq!(parse(raw).calls.len(), 1);
    }
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

    /// What a 7B actually emits when its system prompt lists tools: a correct
    /// call wearing a markdown fence.
    #[test]
    fn a_call_in_a_fenced_json_block_is_recognised() {
        let out = "```json
{\"name\": \"searxng_web_search\", \"arguments\": {\"query\": \"btc\"}}
```";
        let parsed = parse(out);
        assert_eq!(parsed.calls.len(), 1, "got: {parsed:?}");
        assert_eq!(parsed.calls[0].name, "searxng_web_search");
        assert_eq!(parsed.calls[0].arguments, r#"{"query":"btc"}"#);
        assert!(parsed.text.trim().is_empty(), "the fence should not survive as prose");
    }

    #[test]
    fn a_fence_with_no_language_still_counts() {
        let out = "```
{\"name\": \"git_log\", \"arguments\": {}}
```";
        assert_eq!(parse(out).calls.len(), 1);
    }

    #[test]
    fn prose_around_a_fenced_call_is_kept() {
        let out = "I will search.
```json
{\"name\": \"s\", \"arguments\": {}}
```
Stand by.";
        let parsed = parse(out);
        assert_eq!(parsed.calls.len(), 1);
        assert!(parsed.text.contains("I will search."), "got: {:?}", parsed.text);
        assert!(parsed.text.contains("Stand by."), "got: {:?}", parsed.text);
    }

    #[test]
    fn several_fenced_calls_all_arrive_with_distinct_ids() {
        let out = "```json
{\"name\":\"a\",\"arguments\":{}}
```
```json
{\"name\":\"b\",\"arguments\":{}}
```";
        let parsed = parse(out);
        assert_eq!(parsed.calls.len(), 2);
        assert_ne!(parsed.calls[0].id, parsed.calls[1].id, "ids are map keys for some clients");
    }

    /// The false positives this must not produce.
    #[test]
    fn ordinary_fenced_code_is_never_mistaken_for_a_call() {
        for out in [
            // A python block the user asked for.
            "```python
print({'name': 'x', 'arguments': {}})
```",
            // JSON that is the answer, not a call.
            "```json
{\"name\": \"Ada Lovelace\", \"born\": 1815}
```",
            // A schema being explained.
            "```json
{\"type\": \"object\", \"properties\": {}}
```",
        ] {
            let parsed = parse(out);
            assert!(parsed.calls.is_empty(), "false positive on: {out}
{parsed:?}");
            assert_eq!(parsed.text, out, "the answer must survive untouched");
        }
    }

    /// The tagged form still wins — it is what the templates actually ask for.
    #[test]
    fn the_tagged_form_is_still_preferred_over_a_fence() {
        let out = "<tool_call>{\"name\":\"tagged\",\"arguments\":{}}</tool_call>";
        let parsed = parse(out);
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].name, "tagged");
    }

    // ─── LFM2's Python-call syntax ──────────────────────────────────────────

    #[test]
    fn an_lfm2_tool_call_is_extracted() {
        let out = "<|tool_call_start|>[searxng_web_search(query='bitcoin price')]<|tool_call_end|>";
        let parsed = parse(out);
        assert_eq!(parsed.calls.len(), 1, "got: {parsed:?}");
        assert_eq!(parsed.calls[0].name, "searxng_web_search");
        assert_eq!(parsed.calls[0].arguments, r#"{"query":"bitcoin price"}"#);
    }

    #[test]
    fn lfm2_arguments_keep_their_types() {
        let out = "<|tool_call_start|>[f(s='x', n=3, ok=True, off=False, nil=None, o={\"a\": 1})]<|tool_call_end|>";
        let parsed = parse(out);
        let args: serde_json::Value =
            serde_json::from_str(&parsed.calls[0].arguments).expect("valid JSON");
        assert_eq!(args["s"], "x");
        assert_eq!(args["n"], 3);
        assert_eq!(args["ok"], true);
        assert_eq!(args["off"], false);
        assert!(args["nil"].is_null());
        assert_eq!(args["o"]["a"], 1);
    }

    #[test]
    fn a_comma_inside_an_lfm2_argument_does_not_split_it() {
        let out = "<|tool_call_start|>[search(query='rust, ownership and borrowing')]<|tool_call_end|>";
        let parsed = parse(out);
        assert_eq!(parsed.calls.len(), 1, "got: {parsed:?}");
        let args: serde_json::Value = serde_json::from_str(&parsed.calls[0].arguments).unwrap();
        assert_eq!(args["query"], "rust, ownership and borrowing");
    }

    #[test]
    fn two_lfm2_calls_in_one_list_both_survive() {
        let out = "<|tool_call_start|>[a(x='1'), b(y='2')]<|tool_call_end|>";
        let parsed = parse(out);
        assert_eq!(parsed.calls.len(), 2);
        assert_eq!(parsed.calls[0].name, "a");
        assert_eq!(parsed.calls[1].name, "b");
        assert_ne!(parsed.calls[0].id, parsed.calls[1].id);
    }

    #[test]
    fn an_lfm2_call_with_no_arguments_is_still_a_call() {
        let parsed = parse("<|tool_call_start|>[list_notebooks()]<|tool_call_end|>");
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].arguments, "{}");
    }

    #[test]
    fn prose_around_an_lfm2_call_is_kept() {
        let out = "Let me look.<|tool_call_start|>[f(a='b')]<|tool_call_end|>Done.";
        let parsed = parse(out);
        assert_eq!(parsed.calls.len(), 1);
        assert!(parsed.text.contains("Let me look."), "got: {:?}", parsed.text);
        assert!(parsed.text.contains("Done."), "got: {:?}", parsed.text);
    }

    #[test]
    fn text_that_merely_looks_like_a_call_is_not_one() {
        // No markers, so nothing here should be read as a call.
        let parsed = parse("call foo(bar='baz') when you need it");
        assert!(parsed.calls.is_empty(), "got: {parsed:?}");
    }
}
