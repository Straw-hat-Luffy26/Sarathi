//! AI engine traits and data types
//!
//! Defines the abstract interface for AI inference backends,
//! runtime status tracking, generation parameters, and message types.

use anyhow::Result;
use serde::{Deserialize, Serialize};

// ─── Runtime Status ──────────────────────────────────────────────────────────

/// Represents the current state of the inference runtime
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum RuntimeStatus {
    /// No model loaded, runtime idle
    NotLoaded,
    /// Model is being loaded into memory (step description included)
    Loading(String),
    /// Model loaded and ready for inference
    Ready,
    /// Actively generating tokens
    Generating,
    /// Model is being unloaded from memory
    Unloading,
    /// An error occurred
    Error(String),
}

impl std::fmt::Display for RuntimeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeStatus::NotLoaded => write!(f, "NotLoaded"),
            RuntimeStatus::Loading(step) => write!(f, "Loading: {}", step),
            RuntimeStatus::Ready => write!(f, "Ready"),
            RuntimeStatus::Generating => write!(f, "Generating"),
            RuntimeStatus::Unloading => write!(f, "Unloading"),
            RuntimeStatus::Error(e) => write!(f, "Error: {}", e),
        }
    }
}

// ─── Backend Type Enum ───────────────────────────────────────────────────────

pub enum AIBackendType { Ollama, LlamaCpp, VLLM, Custom(String) }

// ─── Chat Message Types ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
    /// Calls this assistant turn made, as the client reported them.
    ///
    /// Carried structurally rather than only as text because the *next* turn's
    /// prompt has to re-render them in the model's own syntax — a chat template
    /// reads `message.tool_calls` to do that. Flattening them into content and
    /// nothing else made every tool conversation malformed from its second turn
    /// on, which reads as the model ignoring results it was given.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<serde_json::Value>,
    /// For a `tool` turn: which call this is the result of.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// For a `tool` turn: which tool produced it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    /// A plain turn, which is most of them.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            timestamp: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatResponse {
    pub message: String,
    pub model: String,
    pub tokens_used: u32,
    pub generation_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamChunk {
    /// The token text fragment
    pub text: String,
    /// Whether this is the final chunk
    pub is_final: bool,
    /// Running count of tokens generated so far
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_generated: Option<u32>,
    /// Reason generation finished (e.g., "stop", "length", "cancelled")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// Why generation produced nothing, when it did.
    ///
    /// A field of its own rather than a prefix on `finish_reason`. Errors used
    /// to travel as `finish_reason: "error: …"`, which every consumer forwarded
    /// faithfully into a response field that no OpenAI or Anthropic client
    /// renders: the client saw `content: null`, a `finish_reason` it did not
    /// recognise, and HTTP 200, so it displayed nothing at all. The message was
    /// in the payload the whole time, in the one place nobody looks. That is
    /// what "Worked for 11s" and no output was.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<GenerationError>,
}

/// A generation failure, in a shape the gateway can turn into a real HTTP
/// error with a code clients branch on.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct GenerationError {
    /// Machine-readable kind, mapped to the dialect's own error type.
    pub kind: GenerationErrorKind,
    /// One sentence the user can act on.
    pub message: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GenerationErrorKind {
    /// The prompt does not fit the loaded context. The caller's problem, and
    /// the one thing they can actually do something about.
    ContextLengthExceeded,
    /// The model cannot be given tools at all.
    ToolsUnsupported,
    /// Anything else the runtime reported.
    Inference,
}

impl GenerationError {
    /// Classifies a runtime failure from what it says.
    ///
    /// String matching, because the runtime's errors are `anyhow` chains built
    /// for people to read. The two cases singled out here are the two a client
    /// can respond to differently: shorten the request, or stop sending tools.
    pub fn classify(message: impl Into<String>) -> Self {
        let message = message.into();
        let lower = message.to_lowercase();
        let kind = if lower.contains("token context") || lower.contains("context length") {
            GenerationErrorKind::ContextLengthExceeded
        } else if lower.contains("cannot be given tools") || lower.contains("tool definitions") {
            GenerationErrorKind::ToolsUnsupported
        } else {
            GenerationErrorKind::Inference
        };
        Self { kind, message }
    }

    /// The HTTP status a client should see.
    ///
    /// Both singled-out kinds are 400: the request as sent cannot be served, and
    /// retrying it unchanged will fail again. A 500 would invite exactly that
    /// retry.
    pub fn status(&self) -> u16 {
        match self.kind {
            GenerationErrorKind::ContextLengthExceeded
            | GenerationErrorKind::ToolsUnsupported => 400,
            GenerationErrorKind::Inference => 500,
        }
    }

    /// The `type` field, named as each dialect's own documentation names it.
    pub fn code(&self) -> &'static str {
        match self.kind {
            GenerationErrorKind::ContextLengthExceeded => "context_length_exceeded",
            GenerationErrorKind::ToolsUnsupported => "tools_unsupported",
            GenerationErrorKind::Inference => "inference_error",
        }
    }
}

// ─── Generation Parameters ───────────────────────────────────────────────────

/// Parameters controlling token generation/sampling
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationParams {
    /// Sampling temperature (0.0 = greedy, higher = more random). Default: 0.7
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold. Default: 0.9
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// Top-k sampling. Default: 40
    #[serde(default = "default_top_k")]
    pub top_k: u32,
    /// Maximum tokens to generate. Default: 2048
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// Min-p sampling threshold. Default: 0.05
    #[serde(default = "default_min_p")]
    pub min_p: f32,
    /// Repeat penalty. Default: 1.1
    #[serde(default = "default_repeat_penalty")]
    pub repeat_penalty: f32,
    /// Mirostat sampling (0 = disabled, 1 = v1, 2 = v2). Default: 0
    #[serde(default = "default_mirostat")]
    pub mirostat: u32,
    /// Tool definitions the model may call, in OpenAI's `{type, function}` shape.
    ///
    /// Carried alongside sampling rather than as its own argument so it reaches
    /// the runtime through the existing `generate(messages, params, cb)` path.
    /// Chat templates that support tool use read a `tools` variable; leaving it
    /// empty is what made every MCP server registered against Sarathi list its
    /// tools and never be called.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<serde_json::Value>,
}

fn default_temperature() -> f32 { 0.7 }
fn default_top_p() -> f32 { 0.9 }
fn default_top_k() -> u32 { 40 }
fn default_min_p() -> f32 { 0.05 }
fn default_max_tokens() -> u32 { 2048 }
fn default_repeat_penalty() -> f32 { 1.1 }
fn default_mirostat() -> u32 { 0 }

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            temperature: default_temperature(),
            top_p: default_top_p(),
            top_k: default_top_k(),
            min_p: default_min_p(),
            max_tokens: default_max_tokens(),
            repeat_penalty: default_repeat_penalty(),
            mirostat: default_mirostat(),
            tools: Vec::new(),
        }
    }
}

// ─── Model Load Configuration ────────────────────────────────────────────────

/// Configuration passed to the runtime when loading a model
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelLoadConfig {
    /// Absolute path to the GGUF file
    pub model_path: String,
    /// Model identifier (e.g., "meta-llama/Llama-3.2-1B")
    pub model_id: String,
    /// Human-readable model name
    pub model_name: String,
    /// Quantization label (e.g., "Q8_0")
    pub quantization: String,
    /// Context length in tokens (from Phase 3 recommendation or dynamic calculation)
    pub context_length: u32,
    /// Number of model layers to offload to GPU (0 = CPU-only)
    pub gpu_layers: u32,
    /// For a Mixture-of-Experts model, how many layers' *routed experts* are
    /// kept in system RAM rather than VRAM. 0 for dense models and for MoE
    /// models that fit outright.
    ///
    /// This is orthogonal to `gpu_layers`: a MoE model is split by tensor, so
    /// every layer still goes to the GPU while the bulk of the expert weight
    /// stays in RAM. Reducing `gpu_layers` instead would evict attention and
    /// the KV cache, which is exactly what must stay resident.
    #[serde(default)]
    pub cpu_moe_layers: u32,
    /// Number of CPU threads for inference
    pub threads: u32,
    /// Model chat template (e.g., "chatml", "llama3", "gemma", "mistral")
    #[serde(default = "default_chat_template")]
    pub chat_template: String,
    /// Stop tokens for generation cutoff
    #[serde(default)]
    pub stop_tokens: Vec<String>,
}

fn default_chat_template() -> String {
    "chatml".to_string()
}

// ─── Loaded Model Info ───────────────────────────────────────────────────────

/// Information about the currently loaded model
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoadedModelInfo {
    pub model_id: String,
    pub model_name: String,
    pub quantization: String,
    pub file_path: String,
    pub context_length: u32,
    pub gpu_layers: u32,
    /// Layers whose routed experts were placed in system RAM. Reported so the
    /// UI can say where the weights actually went rather than implying the
    /// whole model is on the GPU.
    #[serde(default)]
    pub cpu_moe_layers: u32,
    pub threads: u32,
    pub backend_used: String,
    pub loaded_at: String,
    pub chat_template: String,
    /// Where the prompt formatting actually came from: `"gguf"` when the model's
    /// own template is in use, or `"fallback:<name>"` when a hand-written one is.
    #[serde(default = "default_template_source")]
    pub template_source: String,
    pub stop_tokens: Vec<String>,
    #[serde(default = "default_model_family")]
    pub model_family: String,
    pub active_adapter: Option<String>,
}

fn default_model_family() -> String {
    "Generic".to_string()
}

fn default_template_source() -> String {
    "unknown".to_string()
}

// ─── Inference Status Payload (for Tauri events) ─────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InferenceStatusPayload {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<LoadedModelInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// ─── Abstract Backend Trait ──────────────────────────────────────────────────

pub trait AIBackend: Send + Sync {
    fn name(&self) -> &'static str { "Unknown" }
    fn backend_type(&self) -> AIBackendType { AIBackendType::Ollama }
    fn load_model(&self, _config: ModelLoadConfig) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn unload_model(&self, _model_id: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn is_model_loaded(&self, _model_id: &str) -> Result<bool> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn chat(&self, _messages: Vec<ChatMessage>) -> Result<ChatResponse> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn chat_stream(&self, _messages: Vec<ChatMessage>) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn list_loaded_models(&self) -> Result<Vec<String>> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn get_model_info(&self, _model_id: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
}

pub trait ContextManager: Send + Sync {
    fn add_message(&self, _message: ChatMessage) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn get_context(&self) -> Result<Vec<ChatMessage>> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn clear_context(&self) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn trim_context(&self, _max_tokens: u32) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn get_token_count(&self, _messages: Vec<ChatMessage>) -> Result<u32> { Err(anyhow::anyhow!("Not yet implemented")) }
}

pub trait MemoryManager: Send + Sync {
    fn save_conversation(&self, _id: &str, _messages: Vec<ChatMessage>) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn load_conversation(&self, _id: &str) -> Result<Vec<ChatMessage>> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn list_conversations(&self) -> Result<Vec<String>> { Err(anyhow::anyhow!("Not yet implemented")) }
    fn delete_conversation(&self, _id: &str) -> Result<()> { Err(anyhow::anyhow!("Not yet implemented")) }
}
