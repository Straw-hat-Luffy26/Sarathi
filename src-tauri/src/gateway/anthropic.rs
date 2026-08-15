//! Anthropic Messages protocol (`/v1/messages`).
//!
//! Spoken by Claude Code, which is pointed here with `ANTHROPIC_BASE_URL`.
//!
//! Three differences from the OpenAI shape drive this module:
//!
//! 1. `system` is a **top-level field**, not a message with `role: "system"`.
//! 2. Message `content` is commonly an **array of typed blocks**, not a string.
//! 3. Streaming uses **named SSE events** in a required order, not one repeated
//!    chunk type. Clients parse by event name, so the sequence must be exact:
//!    `message_start` → `content_block_start` → `content_block_delta`* →
//!    `content_block_stop` → `message_delta` → `message_stop`.

use serde::{Deserialize, Serialize};

use crate::ai_engine::traits::{ChatMessage, GenerationParams};

// ─── Request ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct MessagesRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<AnthropicMessage>,
    /// Top-level system prompt. String or array of blocks.
    #[serde(default)]
    pub system: Option<SystemPrompt>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    /// Tools Claude Code is offering, in Anthropic's flat
    /// `{name, description, input_schema}` form — every MCP server it has
    /// connected arrives here.
    #[serde(default)]
    pub tools: Vec<AnthropicTool>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub input_schema: Option<serde_json::Value>,
}

impl AnthropicTool {
    /// Rewrites into the OpenAI `{type, function}` shape.
    ///
    /// Chat templates are written against that shape almost universally — it is
    /// what `tokenizer.apply_chat_template(tools=…)` documents — so converting
    /// once here is what lets one template path serve both protocols.
    pub fn to_openai(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description.clone().unwrap_or_default(),
                "parameters": self
                    .input_schema
                    .clone()
                    .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}})),
            }
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

impl SystemPrompt {
    pub fn to_text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Blocks(b) => blocks_to_text(b),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: MessageContent,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct ContentBlock {
    #[serde(rename = "type", default)]
    pub block_type: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
    // ── tool_use blocks ──
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub input: Option<serde_json::Value>,
    // ── tool_result blocks ──
    #[serde(default)]
    pub content: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_use_id: Option<String>,
}

/// Flattens content blocks to the text the model should see.
///
/// Images are still dropped — the local GGUF path has no vision — but tool
/// blocks are not: skipping them, which is what this did, erased the model's
/// own tool call and the result that came back, so a second turn saw a
/// conversation where it had apparently said nothing.
fn blocks_to_text(blocks: &[ContentBlock]) -> String {
    let mut parts: Vec<String> = Vec::new();

    for b in blocks {
        match b.block_type.as_deref().unwrap_or("text") {
            "text" => {
                if let Some(t) = b.text.as_deref() {
                    parts.push(t.to_string());
                }
            }
            // Re-rendered in the syntax the model emits, so a replayed call
            // reads to it exactly as its own output did.
            "tool_use" => {
                if let Some(name) = b.name.as_deref() {
                    let input = b.input.clone().unwrap_or_else(|| serde_json::json!({}));
                    parts.push(format!(
                        "<tool_call>{{\"name\": {}, \"arguments\": {}}}</tool_call>",
                        serde_json::Value::from(name),
                        input
                    ));
                }
            }
            "tool_result" => {
                let body = match b.content.as_ref() {
                    Some(serde_json::Value::String(s)) => s.clone(),
                    Some(serde_json::Value::Array(items)) => items
                        .iter()
                        .filter_map(|i| i.get("text").and_then(|t| t.as_str()))
                        .collect::<Vec<_>>()
                        .join("\n"),
                    Some(other) => other.to_string(),
                    None => b.text.clone().unwrap_or_default(),
                };
                parts.push(format!("<tool_response>\n{body}\n</tool_response>"));
            }
            _ => {}
        }
    }

    parts.join("\n")
}

impl MessageContent {
    pub fn to_text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Blocks(b) => blocks_to_text(b),
        }
    }
}

impl MessagesRequest {
    /// Flattens into Sarathi's internal list, hoisting `system` to the front as
    /// a system-role message so the rest of the pipeline sees a uniform shape.
    pub fn to_chat_messages(&self) -> Vec<ChatMessage> {
        let mut out = Vec::with_capacity(self.messages.len() + 1);

        if let Some(system) = &self.system {
            let text = system.to_text();
            if !text.trim().is_empty() {
                out.push(ChatMessage::new("system", text));
            }
        }

        out.extend(
            self.messages
                .iter()
                .map(|m| ChatMessage::new(m.role.clone(), m.content.to_text())),
        );

        out
    }

    /// `max_tokens` is required by the Anthropic spec, so unlike the OpenAI path
    /// it is normally present and authoritative.
    pub fn to_generation_params(&self) -> GenerationParams {
        let base = GenerationParams::default();
        GenerationParams {
            temperature: self.temperature.unwrap_or(base.temperature),
            top_p: self.top_p.unwrap_or(base.top_p),
            top_k: self.top_k.unwrap_or(base.top_k),
            max_tokens: self.max_tokens.unwrap_or(base.max_tokens),
            min_p: base.min_p,
            repeat_penalty: base.repeat_penalty,
            mirostat: base.mirostat,
            tools: if self.tool_choice_is_none() {
                Vec::new()
            } else {
                self.tools.iter().map(AnthropicTool::to_openai).collect()
            },
        }
    }

    fn tool_choice_is_none(&self) -> bool {
        self.tool_choice
            .as_ref()
            .and_then(|c| c.get("type"))
            .and_then(|t| t.as_str())
            == Some("none")
    }
}

// ─── Non-streaming response ─────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub role: &'static str,
    pub model: String,
    pub content: Vec<ResponseBlock>,
    pub stop_reason: String,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

/// A block in the assistant's reply: prose, or a tool the model wants run.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ResponseBlock {
    Text(TextBlock),
    ToolUse(ToolUseBlock),
}

#[derive(Debug, Clone, Serialize)]
pub struct TextBlock {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolUseBlock {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub id: String,
    pub name: String,
    /// An object, unlike the OpenAI schema's JSON string.
    pub input: serde_json::Value,
}

impl ToolUseBlock {
    fn from_call(c: &crate::gateway::toolcall::ToolCall) -> Self {
        Self {
            kind: "tool_use",
            id: c.id.clone(),
            name: c.name.clone(),
            // Arguments arrive as a string; Anthropic wants the parsed object,
            // and a malformed one is better sent as `{}` than as a block the
            // client rejects outright.
            input: serde_json::from_str(&c.arguments).unwrap_or_else(|_| serde_json::json!({})),
        }
    }
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl MessagesResponse {
    pub fn new(id: String, model: String, text: String, stop_reason: &str, output_tokens: u32) -> Self {
        let parsed = crate::gateway::toolcall::parse(&text);

        let mut content: Vec<ResponseBlock> = Vec::new();
        if !parsed.text.is_empty() {
            content.push(ResponseBlock::Text(TextBlock { kind: "text", text: parsed.text }));
        }
        content.extend(
            parsed.calls.iter().map(|c| ResponseBlock::ToolUse(ToolUseBlock::from_call(c))),
        );

        // A reply with no blocks at all is rejected by strict clients, so an
        // empty completion still sends an empty text block.
        if content.is_empty() {
            content.push(ResponseBlock::Text(TextBlock { kind: "text", text: String::new() }));
        }

        let stop = if parsed.calls.is_empty() { stop_reason.to_string() } else { "tool_use".into() };

        Self {
            id,
            kind: "message",
            role: "assistant",
            model,
            content,
            stop_reason: stop,
            stop_sequence: None,
            usage: Usage { input_tokens: 0, output_tokens },
        }
    }
}

/// Maps Sarathi's finish reasons onto Anthropic's vocabulary.
pub fn map_stop_reason(finish_reason: &str) -> &'static str {
    match finish_reason {
        "length" => "max_tokens",
        "cancelled" => "stop_sequence",
        _ => "end_turn",
    }
}

// ─── Streaming events ───────────────────────────────────────────────────────

/// Builds the SSE event bodies. Each returns `(event_name, json_payload)`.
pub struct StreamEvents;

impl StreamEvents {
    pub fn message_start(id: &str, model: &str) -> (&'static str, String) {
        let payload = serde_json::json!({
            "type": "message_start",
            "message": {
                "id": id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": { "input_tokens": 0, "output_tokens": 0 }
            }
        });
        ("message_start", payload.to_string())
    }

    pub fn content_block_start() -> (&'static str, String) {
        let payload = serde_json::json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": { "type": "text", "text": "" }
        });
        ("content_block_start", payload.to_string())
    }

    pub fn content_block_delta(text: &str) -> (&'static str, String) {
        let payload = serde_json::json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": { "type": "text_delta", "text": text }
        });
        ("content_block_delta", payload.to_string())
    }

    pub fn content_block_stop() -> (&'static str, String) {
        Self::content_block_stop_at(0)
    }

    // ── Indexed variants ──
    //
    // A reply containing prose *and* tool calls is several blocks, and the
    // index is how the client tells them apart. The zero-index helpers above
    // stay for the plain streaming path, which only ever has one block.

    pub fn content_block_start_at(index: u32) -> (&'static str, String) {
        let payload = serde_json::json!({
            "type": "content_block_start",
            "index": index,
            "content_block": { "type": "text", "text": "" }
        });
        ("content_block_start", payload.to_string())
    }

    pub fn content_block_delta_at(index: u32, text: &str) -> (&'static str, String) {
        let payload = serde_json::json!({
            "type": "content_block_delta",
            "index": index,
            "delta": { "type": "text_delta", "text": text }
        });
        ("content_block_delta", payload.to_string())
    }

    pub fn content_block_stop_at(index: u32) -> (&'static str, String) {
        let payload = serde_json::json!({ "type": "content_block_stop", "index": index });
        ("content_block_stop", payload.to_string())
    }

    pub fn tool_use_start(index: u32, id: &str, name: &str) -> (&'static str, String) {
        let payload = serde_json::json!({
            "type": "content_block_start",
            "index": index,
            // `input` opens empty and is filled by the deltas below; a client
            // that sees a populated one here has nothing to accumulate into.
            "content_block": { "type": "tool_use", "id": id, "name": name, "input": {} }
        });
        ("content_block_start", payload.to_string())
    }

    pub fn tool_use_delta(index: u32, partial_json: &str) -> (&'static str, String) {
        let payload = serde_json::json!({
            "type": "content_block_delta",
            "index": index,
            "delta": { "type": "input_json_delta", "partial_json": partial_json }
        });
        ("content_block_delta", payload.to_string())
    }

    pub fn message_delta(stop_reason: &str, output_tokens: u32) -> (&'static str, String) {
        let payload = serde_json::json!({
            "type": "message_delta",
            "delta": { "stop_reason": stop_reason, "stop_sequence": null },
            "usage": { "output_tokens": output_tokens }
        });
        ("message_delta", payload.to_string())
    }

    pub fn message_stop() -> (&'static str, String) {
        let payload = serde_json::json!({ "type": "message_stop" });
        ("message_stop", payload.to_string())
    }

    /// Mid-stream failure. Clients surface this instead of hanging.
    pub fn error(message: &str) -> (&'static str, String) {
        let payload = serde_json::json!({
            "type": "error",
            "error": { "type": "api_error", "message": message }
        });
        ("error", payload.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> MessagesRequest {
        serde_json::from_str(json).expect("should parse")
    }

    // ─── Tools ──────────────────────────────────────────────────────────────

    #[test]
    fn claude_codes_tools_are_rewritten_into_the_shape_templates_expect() {
        // Anthropic sends a flat {name, description, input_schema}; chat
        // templates are written against OpenAI's {type, function}.
        let req = parse(
            r#"{"max_tokens":1024,"messages":[{"role":"user","content":"hi"}],
                "tools":[{"name":"research_search","description":"Search notes",
                          "input_schema":{"type":"object",
                            "properties":{"query":{"type":"string"}}}}]}"#,
        );

        let tools = req.to_generation_params().tools;
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "research_search");
        assert_eq!(tools[0]["function"]["description"], "Search notes");
        assert_eq!(tools[0]["function"]["parameters"]["properties"]["query"]["type"], "string");
    }

    #[test]
    fn a_tool_with_no_schema_still_gets_a_usable_parameters_object() {
        let req = parse(
            r#"{"max_tokens":1,"messages":[],"tools":[{"name":"research_list_notebooks"}]}"#,
        );
        let tools = req.to_generation_params().tools;
        assert_eq!(tools[0]["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn tool_choice_none_withholds_the_tools() {
        let req = parse(
            r#"{"max_tokens":1,"messages":[],"tool_choice":{"type":"none"},
                "tools":[{"name":"x"}]}"#,
        );
        assert!(req.to_generation_params().tools.is_empty());
    }

    #[test]
    fn a_previous_tool_exchange_survives_into_the_prompt() {
        // Tool blocks used to be dropped along with images, which erased both
        // the model's own call and the result that answered it.
        let req = parse(
            r#"{"max_tokens":1024,"messages":[
                {"role":"user","content":"search for vec0"},
                {"role":"assistant","content":[
                  {"type":"text","text":"Looking that up."},
                  {"type":"tool_use","id":"tu_1","name":"searxng_web_search",
                   "input":{"query":"vec0"}}]},
                {"role":"user","content":[
                  {"type":"tool_result","tool_use_id":"tu_1",
                   "content":[{"type":"text","text":"Title: sqlite-vec"}]}]}]}"#,
        );

        let msgs = req.to_chat_messages();
        assert_eq!(msgs.len(), 3);
        assert!(msgs[1].content.contains("Looking that up."));
        assert!(msgs[1].content.contains("<tool_call>"), "got: {}", msgs[1].content);
        assert!(msgs[1].content.contains(r#""query":"vec0""#), "got: {}", msgs[1].content);
        assert!(msgs[2].content.contains("Title: sqlite-vec"), "got: {}", msgs[2].content);
    }

    #[test]
    fn images_are_still_dropped_rather_than_failing_the_request() {
        let req = parse(
            r#"{"max_tokens":1,"messages":[{"role":"user","content":[
                {"type":"text","text":"describe"},
                {"type":"image","source":{"type":"base64","data":"..."}}]}]}"#,
        );
        assert_eq!(req.to_chat_messages()[0].content, "describe");
    }

    #[test]
    fn a_tool_call_comes_back_as_a_tool_use_block() {
        let resp = MessagesResponse::new(
            "msg_1".into(),
            "m".into(),
            r#"Checking.<tool_call>{"name":"research_search","arguments":{"query":"vec0"}}</tool_call>"#
                .into(),
            "end_turn",
            10,
        );
        let json: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&resp).unwrap()).unwrap();

        assert_eq!(json["stop_reason"], "tool_use", "Claude Code branches on this");
        assert_eq!(json["content"][0]["type"], "text");
        assert_eq!(json["content"][0]["text"], "Checking.");

        let block = &json["content"][1];
        assert_eq!(block["type"], "tool_use");
        assert_eq!(block["name"], "research_search");
        // An object here, unlike the OpenAI schema's JSON string.
        assert_eq!(block["input"]["query"], "vec0");
        assert!(block["id"].as_str().is_some_and(|s| !s.is_empty()));
    }

    #[test]
    fn an_ordinary_answer_keeps_its_single_text_block() {
        let resp =
            MessagesResponse::new("msg_1".into(), "m".into(), "Hello.".into(), "end_turn", 2);
        let json: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&resp).unwrap()).unwrap();

        assert_eq!(json["stop_reason"], "end_turn");
        assert_eq!(json["content"].as_array().unwrap().len(), 1);
        assert_eq!(json["content"][0]["text"], "Hello.");
    }

    #[test]
    fn an_empty_completion_still_produces_a_block() {
        // Strict clients reject a message with no content blocks at all.
        let resp = MessagesResponse::new("msg_1".into(), "m".into(), String::new(), "end_turn", 0);
        assert_eq!(resp.content.len(), 1);
    }

    #[test]
    fn tool_use_stream_events_carry_the_block_index() {
        // A reply with prose and a call is two blocks; the index is the only
        // thing telling a client which delta belongs to which.
        let (name, payload) = StreamEvents::tool_use_start(1, "tu_1", "research_search");
        assert_eq!(name, "content_block_start");
        let json: serde_json::Value = serde_json::from_str(&payload).unwrap();
        assert_eq!(json["index"], 1);
        assert_eq!(json["content_block"]["type"], "tool_use");
        assert_eq!(json["content_block"]["name"], "research_search");
        assert_eq!(json["content_block"]["input"], serde_json::json!({}), "filled by deltas");

        let (dname, dpayload) = StreamEvents::tool_use_delta(1, r#"{"query":"vec0"}"#);
        assert_eq!(dname, "content_block_delta");
        let dj: serde_json::Value = serde_json::from_str(&dpayload).unwrap();
        assert_eq!(dj["index"], 1);
        assert_eq!(dj["delta"]["type"], "input_json_delta");
        assert_eq!(dj["delta"]["partial_json"], r#"{"query":"vec0"}"#);
    }

    #[test]
    fn top_level_system_becomes_a_leading_system_message() {
        let req = parse(
            r#"{"model":"claude-sonnet-4-5","max_tokens":1024,
                "system":"You are a helpful assistant.",
                "messages":[{"role":"user","content":"Hi"}]}"#,
        );

        let msgs = req.to_chat_messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[0].content, "You are a helpful assistant.");
        assert_eq!(msgs[1].role, "user");
    }

    #[test]
    fn block_style_content_is_flattened() {
        // Claude Code sends content as blocks in normal operation.
        let req = parse(
            r#"{"max_tokens":100,"messages":[{"role":"user","content":[
                {"type":"text","text":"first"},
                {"type":"text","text":"second"}]}]}"#,
        );
        assert_eq!(req.to_chat_messages()[0].content, "first\nsecond");
    }

    #[test]
    fn non_text_blocks_are_skipped() {
        let req = parse(
            r#"{"max_tokens":100,"messages":[{"role":"user","content":[
                {"type":"text","text":"look"},
                {"type":"image","source":{"type":"base64","data":"..."}}]}]}"#,
        );
        assert_eq!(req.to_chat_messages()[0].content, "look");
    }

    #[test]
    fn a_block_style_system_prompt_is_supported() {
        let req = parse(
            r#"{"max_tokens":100,
                "system":[{"type":"text","text":"Rule one"},{"type":"text","text":"Rule two"}],
                "messages":[{"role":"user","content":"go"}]}"#,
        );
        assert_eq!(req.to_chat_messages()[0].content, "Rule one\nRule two");
    }

    #[test]
    fn an_empty_system_prompt_adds_no_message() {
        let req = parse(r#"{"max_tokens":100,"system":"   ","messages":[{"role":"user","content":"hi"}]}"#);
        let msgs = req.to_chat_messages();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "user");
    }

    #[test]
    fn client_sampling_is_honoured() {
        let req = parse(
            r#"{"max_tokens":512,"temperature":0.2,"top_p":0.5,"top_k":10,
                "messages":[{"role":"user","content":"hi"}]}"#,
        );
        let p = req.to_generation_params();
        assert_eq!(p.max_tokens, 512);
        assert_eq!(p.temperature, 0.2);
        assert_eq!(p.top_k, 10);
    }

    #[test]
    fn unknown_fields_do_not_break_parsing() {
        let req = parse(
            r#"{"model":"claude-sonnet-4-5","max_tokens":1024,
                "messages":[{"role":"user","content":"hi"}],
                "tools":[],"metadata":{"user_id":"x"},"stop_sequences":["\n"]}"#,
        );
        assert_eq!(req.to_chat_messages().len(), 1);
    }

    #[test]
    fn stop_reasons_map_to_anthropic_vocabulary() {
        assert_eq!(map_stop_reason("length"), "max_tokens");
        assert_eq!(map_stop_reason("stop"), "end_turn");
        assert_eq!(map_stop_reason("cancelled"), "stop_sequence");
        assert_eq!(map_stop_reason("anything else"), "end_turn");
    }

    #[test]
    fn stream_events_carry_their_declared_type() {
        // Clients dispatch on the event name, and validate the inner "type".
        let (name, body) = StreamEvents::message_start("msg_1", "qwen");
        assert_eq!(name, "message_start");
        assert!(body.contains(r#""type":"message_start""#));
        assert!(body.contains(r#""role":"assistant""#));

        let (name, body) = StreamEvents::content_block_delta("Hello");
        assert_eq!(name, "content_block_delta");
        assert!(body.contains(r#""type":"text_delta""#));
        assert!(body.contains(r#""text":"Hello""#));

        let (name, body) = StreamEvents::message_delta("end_turn", 7);
        assert_eq!(name, "message_delta");
        assert!(body.contains(r#""stop_reason":"end_turn""#));
        assert!(body.contains(r#""output_tokens":7"#));

        assert_eq!(StreamEvents::message_stop().0, "message_stop");
        assert_eq!(StreamEvents::content_block_stop().0, "content_block_stop");
        assert_eq!(StreamEvents::error("boom").0, "error");
    }

    #[test]
    fn delta_text_is_json_escaped() {
        // Newlines and quotes must not corrupt the SSE payload.
        let (_, body) = StreamEvents::content_block_delta("line\n\"quoted\"");
        assert!(body.contains(r#"\n"#));
        assert!(body.contains(r#"\""#));
        // Round-trips as valid JSON.
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["delta"]["text"], "line\n\"quoted\"");
    }
}
