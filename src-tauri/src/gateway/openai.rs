//! OpenAI-compatible protocol (`/v1/chat/completions`).
//!
//! Spoken by opencode, openclaw, Cursor, and Continue.dev.
//!
//! The `model` field is accepted and echoed back but does not select anything —
//! Sarathi always serves whichever model the desktop app has loaded. Clients
//! hardcode names like `gpt-4o` that mean nothing here, and failing those
//! requests would break every tool out of the box.

use serde::{Deserialize, Serialize};

use crate::ai_engine::traits::{ChatMessage, GenerationParams};

// ─── Request ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<OpenAiMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Newer clients send this instead of `max_tokens`.
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// Tools the client is offering, in `{"type":"function","function":{…}}` form.
    ///
    /// This is where every MCP server a client has connected arrives. Dropping
    /// the field — which is what serde did while it was unmodelled — let the
    /// client list its tools while the model was never told they existed.
    #[serde(default)]
    pub tools: Vec<serde_json::Value>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiMessage {
    pub role: String,
    /// Usually a string, but the spec also allows an array of content parts.
    #[serde(default)]
    pub content: Option<MessageContent>,
    /// Present on an assistant turn that called a tool. Replayed into the
    /// prompt so the model can see what it already asked for; without it a
    /// multi-step tool conversation loses its own half of the exchange.
    #[serde(default)]
    pub tool_calls: Vec<serde_json::Value>,
    /// Present on a `role: "tool"` turn, tying the result to its call.
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct ContentPart {
    #[serde(default)]
    pub text: Option<String>,
}

impl MessageContent {
    /// Flattens to plain text. Non-text parts (images) are dropped — the local
    /// GGUF path has no vision support, and silently ignoring them beats
    /// rejecting an otherwise valid request.
    pub fn to_text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|p| p.text.as_deref())
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

impl ChatCompletionRequest {
    /// Converts to Sarathi's internal message list.
    ///
    /// Tool turns are carried **both** ways. The structured `tool_calls`,
    /// `tool_call_id` and `name` travel through to the chat template, which is
    /// how a tool-aware model reads its own previous calls and ties a result
    /// back to the call it answers. The same calls are also rendered into the
    /// content as `<tool_call>{…}</tool_call>`, so a template that only knows
    /// about text still sees a coherent conversation instead of an assistant
    /// turn that says nothing.
    pub fn to_chat_messages(&self) -> Vec<ChatMessage> {
        self.messages
            .iter()
            .map(|m| {
                let mut content = m.content.as_ref().map(|c| c.to_text()).unwrap_or_default();

                for call in &m.tool_calls {
                    let function = call.get("function").unwrap_or(call);
                    let name = function.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let args = function
                        .get("arguments")
                        .and_then(|v| v.as_str().map(str::to_string))
                        .unwrap_or_else(|| {
                            function.get("arguments").map(|v| v.to_string()).unwrap_or_default()
                        });
                    if !name.is_empty() {
                        if !content.is_empty() {
                            content.push('\n');
                        }
                        content.push_str(&format!(
                            "<tool_call>{{\"name\": {}, \"arguments\": {}}}</tool_call>",
                            serde_json::Value::from(name),
                            if args.is_empty() { "{}".to_string() } else { args }
                        ));
                    }
                }

                // A bare result is ambiguous once several tools are in flight,
                // so the name travels with it when the client sent one.
                if m.role == "tool" {
                    if let Some(name) = m.name.as_deref().filter(|n| !n.is_empty()) {
                        content = format!("{name} returned:\n{content}");
                    }
                }

                ChatMessage {
                    role: m.role.clone(),
                    content,
                    timestamp: None,
                    tool_calls: m.tool_calls.clone(),
                    tool_call_id: m.tool_call_id.clone(),
                    name: m.name.clone(),
                }
            })
            .collect()
    }

    /// Layers client-supplied sampling over Sarathi's defaults.
    ///
    /// Only fields the client actually sent are applied, so omitted values keep
    /// the model's configured defaults rather than snapping to OpenAI's.
    pub fn to_generation_params(&self) -> GenerationParams {
        let base = GenerationParams::default();
        GenerationParams {
            temperature: self.temperature.unwrap_or(base.temperature),
            top_p: self.top_p.unwrap_or(base.top_p),
            max_tokens: self
                .max_tokens
                .or(self.max_completion_tokens)
                .unwrap_or(base.max_tokens),
            repeat_penalty: self
                .frequency_penalty
                .map(|f| 1.0 + f.clamp(0.0, 1.0))
                .unwrap_or(base.repeat_penalty),
            top_k: base.top_k,
            min_p: base.min_p,
            mirostat: base.mirostat,
            // `tool_choice: "none"` means the client wants prose this turn even
            // though it has tools connected. Honouring it here keeps the model
            // from being handed a tool list it has been told not to use.
            tools: if self.tool_choice_is_none() { Vec::new() } else { self.tools.clone() },
        }
    }

    fn tool_choice_is_none(&self) -> bool {
        self.tool_choice.as_ref().and_then(|c| c.as_str()) == Some("none")
    }
}

// ─── Response ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ResponseMessage,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseMessage {
    pub role: &'static str,
    /// Null rather than empty when the model only called a tool — clients treat
    /// an empty string as an empty answer and render it as a blank reply.
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<OpenAiToolCall>,
}

/// A tool call in the shape the OpenAI schema defines.
#[derive(Debug, Clone, Serialize)]
pub struct OpenAiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize)]
pub struct FunctionCall {
    pub name: String,
    /// A JSON *string*, not an object — the schema is specific about this and
    /// clients call `JSON.parse` on it.
    pub arguments: String,
}

impl From<&crate::gateway::toolcall::ToolCall> for OpenAiToolCall {
    fn from(c: &crate::gateway::toolcall::ToolCall) -> Self {
        Self {
            id: c.id.clone(),
            kind: "function",
            function: FunctionCall { name: c.name.clone(), arguments: c.arguments.clone() },
        }
    }
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl ChatCompletionResponse {
    pub fn new(id: String, created: u64, model: String, text: String, finish_reason: &str, completion_tokens: u32) -> Self {
        // The model's output is text either way; whether it was a tool call is
        // decided by parsing it, because a GGUF has no other channel to say so.
        let parsed = crate::gateway::toolcall::parse(&text);
        let finish = parsed.finish_reason(finish_reason).to_string();
        let calls: Vec<OpenAiToolCall> = parsed.calls.iter().map(Into::into).collect();

        Self {
            id,
            object: "chat.completion",
            created,
            model,
            choices: vec![Choice {
                index: 0,
                message: ResponseMessage {
                    role: "assistant",
                    content: (!parsed.text.is_empty()).then_some(parsed.text),
                    tool_calls: calls,
                },
                finish_reason: finish,
            }],
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens,
                total_tokens: completion_tokens,
            },
        }
    }
}

// ─── Streaming chunks ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<DeltaToolCall>,
}

/// A tool call inside a streaming delta.
///
/// `index` is what lets a client assemble calls arriving in pieces. Sarathi
/// sends each call whole, but the field is required by the schema and clients
/// key on it regardless.
#[derive(Debug, Clone, Serialize)]
pub struct DeltaToolCall {
    pub index: u32,
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: FunctionCall,
}

impl ChatCompletionChunk {
    /// First chunk: announces the assistant role, carries no text.
    pub fn opening(id: &str, created: u64, model: &str) -> Self {
        Self::with_delta(
            id,
            created,
            model,
            Delta { role: Some("assistant"), ..Delta::default() },
            None,
        )
    }

    pub fn text(id: &str, created: u64, model: &str, text: String) -> Self {
        Self::with_delta(id, created, model, Delta { content: Some(text), ..Delta::default() }, None)
    }

    /// One chunk carrying every tool call the model asked for.
    pub fn tool_calls(
        id: &str,
        created: u64,
        model: &str,
        calls: &[crate::gateway::toolcall::ToolCall],
    ) -> Self {
        let deltas = calls
            .iter()
            .enumerate()
            .map(|(i, c)| DeltaToolCall {
                index: i as u32,
                id: c.id.clone(),
                kind: "function",
                function: FunctionCall { name: c.name.clone(), arguments: c.arguments.clone() },
            })
            .collect();

        Self::with_delta(id, created, model, Delta { tool_calls: deltas, ..Delta::default() }, None)
    }

    pub fn closing(id: &str, created: u64, model: &str, finish_reason: &str) -> Self {
        Self::with_delta(id, created, model, Delta::default(), Some(finish_reason.to_string()))
    }

    fn with_delta(id: &str, created: u64, model: &str, delta: Delta, finish_reason: Option<String>) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk",
            created,
            model: model.to_string(),
            choices: vec![ChunkChoice { index: 0, delta, finish_reason }],
        }
    }
}

// ─── Model listing ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelEntry {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

impl ModelList {
    pub fn single(model_id: Option<String>, created: u64) -> Self {
        let data = match model_id {
            Some(id) => vec![ModelEntry { id, object: "model", created, owned_by: "sarathi" }],
            None => vec![],
        };
        Self { object: "list", data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> ChatCompletionRequest {
        serde_json::from_str(json).expect("should parse")
    }

    #[test]
    fn parses_a_typical_request() {
        let req = parse(
            r#"{"model":"gpt-4o","messages":[
                {"role":"system","content":"Be brief."},
                {"role":"user","content":"Hello"}],
               "stream":true,"temperature":0.3}"#,
        );

        let msgs = req.to_chat_messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[1].content, "Hello");
        assert!(req.stream);
        assert_eq!(req.to_generation_params().temperature, 0.3);
    }

    #[test]
    fn omitted_sampling_falls_back_to_sarathi_defaults() {
        let req = parse(r#"{"messages":[{"role":"user","content":"hi"}]}"#);
        let params = req.to_generation_params();
        let base = GenerationParams::default();

        assert_eq!(params.temperature, base.temperature);
        assert_eq!(params.top_p, base.top_p);
        assert_eq!(params.max_tokens, base.max_tokens);
        assert!(!req.stream, "stream must default to false");
    }

    #[test]
    fn accepts_array_style_content() {
        // Newer clients send content as an array of typed parts.
        let req = parse(
            r#"{"messages":[{"role":"user","content":[
                {"type":"text","text":"line one"},
                {"type":"text","text":"line two"}]}]}"#,
        );
        assert_eq!(req.to_chat_messages()[0].content, "line one\nline two");
    }

    #[test]
    fn image_parts_are_dropped_rather_than_failing_the_request() {
        let req = parse(
            r#"{"messages":[{"role":"user","content":[
                {"type":"text","text":"describe"},
                {"type":"image_url","image_url":{"url":"data:..."}}]}]}"#,
        );
        assert_eq!(req.to_chat_messages()[0].content, "describe");
    }

    #[test]
    fn max_completion_tokens_is_honoured_as_an_alias() {
        let req = parse(r#"{"messages":[],"max_completion_tokens":123}"#);
        assert_eq!(req.to_generation_params().max_tokens, 123);
    }

    // ─── Tools ──────────────────────────────────────────────────────────────

    #[test]
    fn tools_reach_the_generation_params() {
        // The regression this exists for: `tools` was unmodelled, so serde
        // dropped it and every MCP server a client had connected was invisible
        // to the model — listed in the client, never callable.
        let req = parse(
            r#"{"messages":[{"role":"user","content":"search"}],
                "tools":[{"type":"function","function":{"name":"searxng_web_search",
                          "parameters":{"type":"object","properties":{"query":{"type":"string"}}}}}]}"#,
        );

        let params = req.to_generation_params();
        assert_eq!(params.tools.len(), 1);
        assert_eq!(params.tools[0]["function"]["name"], "searxng_web_search");
    }

    #[test]
    fn tool_choice_none_withholds_the_tools_from_the_model() {
        let req = parse(
            r#"{"messages":[],"tool_choice":"none",
                "tools":[{"type":"function","function":{"name":"x"}}]}"#,
        );
        assert!(req.to_generation_params().tools.is_empty());

        let auto = parse(
            r#"{"messages":[],"tool_choice":"auto",
                "tools":[{"type":"function","function":{"name":"x"}}]}"#,
        );
        assert_eq!(auto.to_generation_params().tools.len(), 1);
    }

    #[test]
    fn a_completion_containing_a_tool_call_serialises_as_one() {
        let resp = ChatCompletionResponse::new(
            "id".into(),
            0,
            "m".into(),
            r#"<tool_call>{"name":"research_search","arguments":{"query":"vec0"}}</tool_call>"#.into(),
            "stop",
            10,
        );
        let json: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&resp).unwrap()).unwrap();

        let choice = &json["choices"][0];
        assert_eq!(choice["finish_reason"], "tool_calls", "clients branch on this");
        let call = &choice["message"]["tool_calls"][0];
        assert_eq!(call["type"], "function");
        assert_eq!(call["function"]["name"], "research_search");
        // A JSON *string*, per the schema — clients call JSON.parse on it.
        let args: serde_json::Value =
            serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["query"], "vec0");
        assert!(choice["message"]["content"].is_null(), "no prose means null, not empty");
    }

    #[test]
    fn an_ordinary_answer_is_unchanged_by_the_tool_path() {
        let resp =
            ChatCompletionResponse::new("id".into(), 0, "m".into(), "Hello.".into(), "stop", 2);
        let json: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&resp).unwrap()).unwrap();

        assert_eq!(json["choices"][0]["message"]["content"], "Hello.");
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert!(json["choices"][0]["message"]["tool_calls"].is_null(), "omitted when unused");
    }

    #[test]
    fn a_previous_tool_exchange_is_replayed_into_the_prompt() {
        // Without this the model's second turn sees a conversation in which it
        // apparently said nothing and a result appeared from nowhere.
        let req = parse(
            r#"{"messages":[
                {"role":"user","content":"search for vec0"},
                {"role":"assistant","content":null,"tool_calls":[
                  {"id":"call_1","type":"function","function":{
                     "name":"searxng_web_search","arguments":"{\"query\":\"vec0\"}"}}]},
                {"role":"tool","tool_call_id":"call_1","name":"searxng_web_search",
                 "content":"Title: sqlite-vec"}]}"#,
        );

        let msgs = req.to_chat_messages();
        assert_eq!(msgs.len(), 3);
        assert!(msgs[1].content.contains("<tool_call>"), "got: {}", msgs[1].content);
        assert!(msgs[1].content.contains("searxng_web_search"));
        assert!(msgs[1].content.contains(r#""query":"vec0""#), "got: {}", msgs[1].content);
        assert_eq!(msgs[2].role, "tool");
        assert!(msgs[2].content.contains("searxng_web_search returned:"));
        assert!(msgs[2].content.contains("Title: sqlite-vec"));
    }

    #[test]
    fn a_streaming_chunk_can_carry_tool_calls() {
        let calls = crate::gateway::toolcall::parse(
            r#"<tool_call>{"name":"a","arguments":{"k":1}}</tool_call>"#,
        )
        .calls;
        let chunk = ChatCompletionChunk::tool_calls("id", 0, "m", &calls);
        let json: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&chunk).unwrap()).unwrap();

        let delta = &json["choices"][0]["delta"];
        assert_eq!(delta["tool_calls"][0]["index"], 0, "clients key assembly on index");
        assert_eq!(delta["tool_calls"][0]["function"]["name"], "a");
    }

    #[test]
    fn unknown_fields_do_not_break_parsing() {
        // Clients send plenty we do not model; rejecting them would be fatal.
        let req = parse(
            r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],
                "tools":[],"tool_choice":"auto","seed":42,"user":"abc","n":1}"#,
        );
        assert_eq!(req.to_chat_messages().len(), 1);
    }

    #[test]
    fn streaming_chunks_have_the_right_shape() {
        let opening = ChatCompletionChunk::opening("id1", 1, "m");
        let json = serde_json::to_string(&opening).unwrap();
        assert!(json.contains(r#""object":"chat.completion.chunk""#));
        assert!(json.contains(r#""role":"assistant""#));
        // No content key on the opening chunk.
        assert!(!json.contains(r#""content""#));

        let text = ChatCompletionChunk::text("id1", 1, "m", "Hi".into());
        assert!(serde_json::to_string(&text).unwrap().contains(r#""content":"Hi""#));

        let closing = ChatCompletionChunk::closing("id1", 1, "m", "stop");
        assert!(serde_json::to_string(&closing).unwrap().contains(r#""finish_reason":"stop""#));
    }

    #[test]
    fn model_list_is_empty_when_nothing_is_loaded() {
        assert!(ModelList::single(None, 0).data.is_empty());
        assert_eq!(ModelList::single(Some("qwen".into()), 0).data.len(), 1);
    }
}
