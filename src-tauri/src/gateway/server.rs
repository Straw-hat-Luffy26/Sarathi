//! Axum server: routes, handlers, and SSE streaming for both protocols.

use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
// Only futures_util's StreamExt is imported: tokio_stream's version defines the
// same combinators and importing both makes every call ambiguous.
use futures_util::stream::{self, Stream, StreamExt};
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::ai_engine::scheduler::{
    CancelOnDrop, Canceller, GenerationHandle, GenerationJob, JobOrigin,
};
use crate::ai_engine::traits::{
    ChatMessage, GenerationError, GenerationParams, StreamChunk,
};
use crate::gateway::anthropic::{self, MessagesRequest, MessagesResponse, StreamEvents};
use crate::gateway::guard::origin_guard;
use crate::gateway::openai::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ModelList};
use crate::gateway::state::{client_label, GatewayState};

/// Running server; call [`GatewayHandle::stop`] to shut it down.
///
/// **The handle must be kept alive for as long as the server should run.**
/// Dropping it drops the shutdown sender, which resolves the graceful-shutdown
/// future and stops the server accepting connections — within milliseconds of
/// it reporting that it is listening. Store it in application state rather than
/// letting it fall out of scope.
#[derive(Debug)]
pub struct GatewayHandle {
    pub port: u16,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
}

impl GatewayHandle {
    pub fn stop(mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
    }
}

/// Binds the gateway and serves until shutdown.
///
/// A bind failure (usually the port being in use) is returned rather than
/// panicking, so the desktop app still opens and the user can choose another port.
/// Binds `addr`, retrying briefly while it is still held.
///
/// A port busy at startup is nearly always this app's own previous instance
/// finishing its shutdown — restarting Sarathi is exactly when this happens, and
/// the socket clears in well under a second. Retrying keeps the address the user
/// configured, which matters because anything pointed at it by hand would
/// otherwise be left behind by a fallback.
async fn bind_with_retry(
    addr: std::net::SocketAddr,
) -> std::io::Result<tokio::net::TcpListener> {
    const GRACE: std::time::Duration = std::time::Duration::from_secs(3);
    const POLL: std::time::Duration = std::time::Duration::from_millis(150);

    let deadline = std::time::Instant::now() + GRACE;
    loop {
        match tokio::net::TcpListener::bind(addr).await {
            Ok(listener) => return Ok(listener),
            Err(e) if std::time::Instant::now() < deadline => {
                log::debug!("[GATEWAY] {addr} busy ({e}); retrying");
                tokio::time::sleep(POLL).await;
            }
            Err(e) => return Err(e),
        }
    }
}

pub async fn start_gateway(state: Arc<GatewayState>) -> anyhow::Result<GatewayHandle> {
    let requested_port = state.port();
    let app = router(state.clone());

    // Loopback only. Never 0.0.0.0 — that would expose the model to the network.
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], requested_port));
    let listener = match bind_with_retry(addr).await {
        Ok(listener) => listener,
        Err(first) => {
            // Binding once and giving up left the gateway down for the entire
            // session whenever the port was momentarily busy — most often this
            // app's own previous instance still releasing its socket after a
            // quick restart. Every client then got ConnectionRefused with
            // nothing to explain it, and no amount of retrying on their side
            // could recover, because nothing was ever going to listen.
            //
            // An OS-assigned port always succeeds. The bound port is read back
            // below and published to state, and tools are handed that at launch
            // rather than the configured value, so a fallback stays invisible.
            log::warn!(
                "[GATEWAY] Could not bind 127.0.0.1:{requested_port} ({first}); falling back to a free port"
            );
            let any_port = std::net::SocketAddr::from(([127, 0, 0, 1], 0));
            tokio::net::TcpListener::bind(any_port).await.map_err(|e| {
                anyhow::anyhow!(
                    "could not bind gateway to 127.0.0.1:{requested_port} ({first}), \
                     nor to any free port ({e})"
                )
            })?
        }
    };

    // Read the port back from the socket rather than trusting the request.
    // Port 0 means "any free port", and the OS picks the real one — reporting
    // the requested value would hand callers a port nothing is listening on,
    // and log a connection string that cannot work.
    let port = listener
        .local_addr()
        .map_err(|e| anyhow::anyhow!("bound gateway but could not read its address: {e}"))?
        .port();

    if port != requested_port {
        log::warn!(
            "[GATEWAY] Configured port {requested_port} was unavailable; serving on {port} instead"
        );
    }
    // Published so the dashboard and every launched tool use the live address.
    state.set_port(port);

    let (tx, rx) = tokio::sync::oneshot::channel::<()>();

    tokio::spawn(async move {
        let server = axum::serve(listener, app).with_graceful_shutdown(async {
            let _ = rx.await;
        });
        if let Err(e) = server.await {
            log::error!("[GATEWAY] Server stopped with error: {e}");
        }
    });

    log::info!("[GATEWAY] Listening on http://127.0.0.1:{port}");
    Ok(GatewayHandle { port, shutdown: Some(tx) })
}

pub fn router(state: Arc<GatewayState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(openai_chat))
        .route("/v1/messages", post(anthropic_messages))
        .layer(axum::middleware::from_fn(origin_guard))
        .with_state(state)
}

// ─── Simple endpoints ───────────────────────────────────────────────────────

async fn health(State(state): State<Arc<GatewayState>>) -> impl IntoResponse {
    let model = state.inference.get_loaded_model_info();
    Json(serde_json::json!({
        "status": "ok",
        "modelLoaded": model.is_some(),
        "model": model.map(|m| m.model_id),
    }))
}

async fn list_models(State(state): State<Arc<GatewayState>>) -> impl IntoResponse {
    let id = state.inference.get_loaded_model_info().map(|m| m.model_id);
    Json(ModelList::single(id, now_secs()))
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn short_id(prefix: &str) -> String {
    format!("{prefix}{}", uuid::Uuid::new_v4().simple())
}

fn json_of<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

/// The model being served, or a 503 explaining what to do about it.
///
/// A clear message here is what stops users debugging their client config for an
/// hour when the real problem is that no model is loaded.
fn active_model(state: &GatewayState) -> Result<String, Response> {
    match state.inference.get_loaded_model_info() {
        Some(info) => Ok(info.model_id),
        None => Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": {
                    "type": "no_model_loaded",
                    "message": "No model is loaded. Open Sarathi and load a model, then retry."
                }
            })),
        )
            .into_response()),
    }
}

/// What arrived, in numbers, before anything is done with it.
///
/// Sarathi could see that a request had failed but not why it was so large,
/// because nothing recorded the shape of what clients send. An agentic client
/// carrying six MCP servers sends tool definitions that dwarf the conversation
/// — and that is invisible unless it is counted. Sizes only: no message
/// content, no tool arguments, nothing a user typed.
#[derive(Clone, Copy)]
struct Shape {
    tools: usize,
    tool_chars: usize,
}

fn log_request_shape(
    client: &str,
    dialect: &str,
    stream: bool,
    messages: &[ChatMessage],
    params: &GenerationParams,
) -> Shape {
    let message_chars: usize = messages.iter().map(|m| m.content.len()).sum();
    let tool_chars: usize =
        params.tools.iter().map(|t| serde_json::to_string(t).map_or(0, |s| s.len())).sum();

    // A rough token estimate, four characters to the token. The real count
    // comes from the tokenizer a moment later; this is here so the *split*
    // between conversation and tool definitions is visible at a glance.
    log::info!(
        "[GATEWAY] REQUEST_START client={client} dialect={dialect} stream={stream} \
         messages={} message_chars={message_chars} tools={} tool_schema_chars={tool_chars} \
         (~{}k tokens of conversation, ~{}k of tool definitions) max_tokens={}",
        messages.len(),
        params.tools.len(),
        message_chars / 4000,
        tool_chars / 4000,
        params.max_tokens,
    );

    Shape { tools: params.tools.len(), tool_chars }
}

fn submit(
    state: &GatewayState,
    messages: Vec<ChatMessage>,
    params: GenerationParams,
    client: &str,
) -> Result<GenerationHandle, Response> {
    state
        .scheduler
        .submit(GenerationJob {
            messages,
            params,
            // Capability routing is opt-in for gateway traffic: external tools
            // send their own system prompts and sampling, and overriding those
            // silently can break output they depend on.
            capability: None,
            origin: JobOrigin::Gateway { client: client.to_string() },
        })
        .map_err(|e| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": { "type": "scheduler_unavailable", "message": e.to_string() }
                })),
            )
                .into_response()
        })
}

/// A finished generation, or the reason there isn't one.
struct Answer {
    text: String,
    finish: String,
    tokens: u32,
    failure: Option<GenerationError>,
}

/// Drains the whole answer for non-streaming requests.
async fn collect(handle: &mut GenerationHandle) -> Answer {
    let mut answer =
        Answer { text: String::new(), finish: "stop".to_string(), tokens: 0, failure: None };

    while let Some(chunk) = handle.chunks.recv().await {
        answer.text.push_str(&chunk.text);
        if let Some(n) = chunk.tokens_generated {
            answer.tokens = n;
        }
        if chunk.is_final {
            if let Some(reason) = chunk.finish_reason {
                answer.finish = reason;
            }
            answer.failure = chunk.error;
            break;
        }
    }
    answer
}

/// How long the gateway will wait for the first chunk before committing to a
/// streaming response.
///
/// Every failure worth reporting — a prompt that does not fit, a model that
/// cannot take tools, no model at all — is decided during prefill, before a
/// single token exists. Waiting for that decision means it can be answered with
/// a real HTTP status, which every client displays, instead of an empty 200 that
/// every client renders as nothing.
///
/// Bounded because the alternative failure is worse: on a slow CPU prefill the
/// first token can be a minute away, and holding the response headers back that
/// long denies the client the keep-alive that tells it the connection is live.
/// Past this point the stream starts, and a later failure is reported in-band.
const FIRST_CHUNK_BUDGET: Duration = Duration::from_secs(20);

/// Waits for the first chunk, so an immediate failure is still an HTTP error.
async fn peek(handle: &mut GenerationHandle) -> Option<StreamChunk> {
    tokio::time::timeout(FIRST_CHUNK_BUDGET, handle.chunks.recv()).await.ok().flatten()
}

/// The dialect an error has to be phrased in.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Dialect {
    OpenAi,
    Anthropic,
}

/// Says what filled the context, when it was mostly tool definitions.
///
/// "Send a shorter prompt" is not advice anybody can act on: the prompt belongs
/// to the agent, and the user never saw it. What they can act on is the fact
/// that 122 tool definitions from their MCP servers took 43 000 of the 48 000
/// tokens — because the remedy is to hand this provider fewer servers, or to
/// use a model with more room.
fn explain_overflow(failure: &GenerationError, tools: usize, tool_chars: usize) -> GenerationError {
    if failure.kind != crate::ai_engine::traits::GenerationErrorKind::ContextLengthExceeded
        || tools == 0
    {
        return failure.clone();
    }

    GenerationError {
        kind: failure.kind,
        message: format!(
            "{} Most of it is tool definitions: this request carried {tools} tools \
             (~{}k tokens of schema) from the MCP servers Sarathi gave this provider. \
             Remove servers from mcp.json, or load a model with a longer context.",
            failure.message,
            tool_chars / 4000,
        ),
    }
}

/// A failure, in the shape the calling client's own SDK expects.
///
/// Both dialects document an error envelope and both sets of clients render it.
/// What neither renders is a 200 whose content is null, which is what Sarathi
/// used to send.
fn error_response(dialect: Dialect, failure: &GenerationError) -> Response {
    let status =
        StatusCode::from_u16(failure.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    log::warn!("[GATEWAY] Returning HTTP {} [{}]: {}", status, failure.code(), failure.message);

    let body = match dialect {
        Dialect::OpenAi => serde_json::json!({
            "error": {
                "message": failure.message,
                "type": "invalid_request_error",
                "code": failure.code(),
            }
        }),
        Dialect::Anthropic => serde_json::json!({
            "type": "error",
            "error": { "type": failure.code(), "message": failure.message },
        }),
    };
    (status, Json(body)).into_response()
}

/// The same failure, for a stream that has already sent its headers.
///
/// Second best, and only reached when generation ran past
/// [`FIRST_CHUNK_BUDGET`] before failing. Both dialects define this frame, so
/// the client sees the reason rather than a stream that simply stops.
fn error_event(dialect: Dialect, failure: &GenerationError) -> Event {
    match dialect {
        Dialect::OpenAi => Event::default().data(json_of(&serde_json::json!({
            "error": { "message": failure.message, "type": "invalid_request_error", "code": failure.code() }
        }))),
        Dialect::Anthropic => Event::default().event("error").data(json_of(&serde_json::json!({
            "type": "error",
            "error": { "type": failure.code(), "message": failure.message },
        }))),
    }
}

/// How a generation ended, shared between the body and tail of a stream.
#[derive(Clone)]
struct Outcome(Arc<Mutex<(String, u32)>>);

impl Outcome {
    fn new() -> Self {
        Self(Arc::new(Mutex::new(("stop".to_string(), 0))))
    }
    fn record(&self, reason: String, tokens: u32) {
        if let Ok(mut g) = self.0.lock() {
            *g = (reason, tokens);
        }
    }
    fn get(&self) -> (String, u32) {
        self.0.lock().map(|g| g.clone()).unwrap_or_else(|_| ("stop".to_string(), 0))
    }
}

// ─── OpenAI ─────────────────────────────────────────────────────────────────

async fn openai_chat(
    State(state): State<Arc<GatewayState>>,
    headers: HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let client = client_label(headers.get("user-agent").and_then(|v| v.to_str().ok()));
    state.record_request(&client, "openai", now_millis());

    let model = match active_model(&state) {
        Ok(m) => m,
        Err(resp) => {
            state.finish_request();
            return resp;
        }
    };

    let (messages, params) = (req.to_chat_messages(), req.to_generation_params());
    let shape = log_request_shape(&client, "openai", req.stream, &messages, &params);

    let mut handle = match submit(&state, messages, params, &client) {
        Ok(h) => h,
        Err(resp) => {
            state.finish_request();
            return resp;
        }
    };

    // Armed for the whole handler, not just the response stream.
    //
    // Nothing else notices a client that hangs up during prefill: the
    // scheduler only learns of a dropped receiver when it next emits a token,
    // and prefill emits none. A 48 000-token prompt therefore ran to
    // completion — minutes of CPU, blocking every request queued behind it —
    // for a client that had already gone. Holding the guard here means the
    // handler future being dropped is itself the signal.
    let mut guard = CancelOnDrop::new(handle.canceller());

    let id = short_id("chatcmpl-");
    let created = now_secs();

    if !req.stream {
        let answer = collect(&mut handle).await;
        guard.disarm();
        state.finish_request();
        if let Some(failure) = &answer.failure {
            return error_response(Dialect::OpenAi, &explain_overflow(failure, shape.tools, shape.tool_chars));
        }
        return Json(ChatCompletionResponse::new(
            id,
            created,
            model,
            answer.text,
            &answer.finish,
            answer.tokens,
        ))
        .into_response();
    }

    // A tool call cannot be recognised until the whole of it has arrived — it is
    // JSON inside a tag, and half of it is not a call. So when the client has
    // offered tools, the answer is collected and then emitted as one correct
    // stream. The latency this costs is real, but it only applies to the
    // agentic path, where no one is watching tokens appear anyway.
    //
    // It also means a failure on the agentic path — the path where prompts are
    // largest and overflow likeliest — is always known before anything is sent,
    // so it can always be a real HTTP error.
    if !req.tools.is_empty() {
        let answer = collect(&mut handle).await;
        guard.disarm();
        state.finish_request();
        if let Some(failure) = &answer.failure {
            return error_response(Dialect::OpenAi, &explain_overflow(failure, shape.tools, shape.tool_chars));
        }
        return Sse::new(openai_buffered_stream(
            answer.text,
            answer.finish,
            answer.tokens,
            id,
            created,
            model,
        ))
        .keep_alive(KeepAlive::default())
        .into_response();
    }

    // Wait for the first chunk before committing to a 200, so a request that
    // fails during prefill is answered rather than left blank.
    let first = peek(&mut handle).await;
    if let Some(failure) = first.as_ref().and_then(|c| c.error.as_ref()) {
        guard.disarm();
        state.finish_request();
        return error_response(Dialect::OpenAi, &explain_overflow(failure, shape.tools, shape.tool_chars));
    }

    Sse::new(openai_stream(state.clone(), handle, guard, first, id, created, model))
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Replays a completed answer as a well-formed OpenAI stream.
///
/// Used when tools were offered; see [`openai_chat`].
fn openai_buffered_stream(
    text: String,
    finish: String,
    _tokens: u32,
    id: String,
    created: u64,
    model: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let parsed = crate::gateway::toolcall::parse(&text);
    let reason = parsed.finish_reason(&finish).to_string();

    let mut events = vec![Ok(Event::default()
        .data(json_of(&ChatCompletionChunk::opening(&id, created, &model))))];

    if !parsed.text.is_empty() {
        events.push(Ok(Event::default().data(json_of(&ChatCompletionChunk::text(
            &id, created, &model, parsed.text.clone(),
        )))));
    }
    if !parsed.calls.is_empty() {
        events.push(Ok(Event::default().data(json_of(&ChatCompletionChunk::tool_calls(
            &id, created, &model, &parsed.calls,
        )))));
    }

    events.push(Ok(Event::default().data(json_of(&ChatCompletionChunk::closing(
        &id, created, &model, &reason,
    )))));
    events.push(Ok(Event::default().data("[DONE]")));

    stream::iter(events)
}

/// `opening chunk` → `text chunk`* → `closing chunk` → `[DONE]`.
///
/// All parts share one id, which clients use to correlate the stream.
fn openai_stream(
    state: Arc<GatewayState>,
    handle: GenerationHandle,
    mut guard: CancelOnDrop,
    first: Option<StreamChunk>,
    id: String,
    created: u64,
    model: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let canceller: Canceller = handle.canceller();
    // The handler's guard, moved in: it now covers submission, prefill and the
    // whole stream without a gap between them.
    // The chunk already taken off the channel by `peek` goes back on the front,
    // or the client loses the first token of every answer.
    let chunks = stream::iter(first).chain(UnboundedReceiverStream::new(handle.chunks));

    let (head_id, head_model) = (id.clone(), model.clone());
    let head = stream::once(async move {
        Ok(Event::default().data(json_of(&ChatCompletionChunk::opening(&head_id, created, &head_model))))
    });

    let (body_id, body_model) = (id.clone(), model.clone());
    let body = chunks.filter_map(move |chunk: StreamChunk| {
        if chunk.is_final {
            guard.disarm();
        }
        let event = if let Some(failure) = &chunk.error {
            // Past the point where an HTTP status was still possible.
            Some(error_event(Dialect::OpenAi, failure))
        } else if chunk.is_final {
            let reason = chunk.finish_reason.clone().unwrap_or_else(|| "stop".to_string());
            Some(Event::default().data(json_of(&ChatCompletionChunk::closing(
                &body_id, created, &body_model, &reason,
            ))))
        } else if chunk.text.is_empty() {
            None
        } else {
            Some(Event::default().data(json_of(&ChatCompletionChunk::text(
                &body_id, created, &body_model, chunk.text,
            ))))
        };
        async move { event.map(Ok) }
    });

    let tail = stream::once(async move {
        // The client has read everything; make sure nothing keeps generating.
        canceller.cancel();
        state.finish_request();
        Ok(Event::default().data("[DONE]"))
    });

    head.chain(body).chain(tail)
}

// ─── Anthropic ──────────────────────────────────────────────────────────────

async fn anthropic_messages(
    State(state): State<Arc<GatewayState>>,
    headers: HeaderMap,
    Json(req): Json<MessagesRequest>,
) -> Response {
    let client = client_label(headers.get("user-agent").and_then(|v| v.to_str().ok()));
    state.record_request(&client, "anthropic", now_millis());

    let model = match active_model(&state) {
        Ok(m) => m,
        Err(resp) => {
            state.finish_request();
            return resp;
        }
    };

    let (messages, params) = (req.to_chat_messages(), req.to_generation_params());
    let shape = log_request_shape(&client, "anthropic", req.stream, &messages, &params);

    let mut handle = match submit(&state, messages, params, &client) {
        Ok(h) => h,
        Err(resp) => {
            state.finish_request();
            return resp;
        }
    };

    // Armed for the whole handler, not just the response stream.
    //
    // Nothing else notices a client that hangs up during prefill: the
    // scheduler only learns of a dropped receiver when it next emits a token,
    // and prefill emits none. A 48 000-token prompt therefore ran to
    // completion — minutes of CPU, blocking every request queued behind it —
    // for a client that had already gone. Holding the guard here means the
    // handler future being dropped is itself the signal.
    let mut guard = CancelOnDrop::new(handle.canceller());

    let id = short_id("msg_");

    if !req.stream {
        let answer = collect(&mut handle).await;
        guard.disarm();
        state.finish_request();
        if let Some(failure) = &answer.failure {
            return error_response(Dialect::Anthropic, &explain_overflow(failure, shape.tools, shape.tool_chars));
        }
        let stop = anthropic::map_stop_reason(&answer.finish);
        return Json(MessagesResponse::new(id, model, answer.text, stop, answer.tokens))
            .into_response();
    }

    // Same reasoning as the OpenAI path: a tool call is only recognisable whole,
    // and collecting first means an agentic request that overflows the context
    // is reported as an error rather than as an empty answer.
    if !req.tools.is_empty() {
        let answer = collect(&mut handle).await;
        guard.disarm();
        state.finish_request();
        if let Some(failure) = &answer.failure {
            return error_response(Dialect::Anthropic, &explain_overflow(failure, shape.tools, shape.tool_chars));
        }
        let stop = anthropic::map_stop_reason(&answer.finish);
        return Sse::new(anthropic_buffered_stream(answer.text, stop, answer.tokens, id, model))
            .keep_alive(KeepAlive::default())
            .into_response();
    }

    let first = peek(&mut handle).await;
    if let Some(failure) = first.as_ref().and_then(|c| c.error.as_ref()) {
        guard.disarm();
        state.finish_request();
        return error_response(Dialect::Anthropic, &explain_overflow(failure, shape.tools, shape.tool_chars));
    }

    Sse::new(anthropic_stream(state.clone(), handle, guard, first, id, model))
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Replays a completed answer as the exact event sequence Anthropic clients
/// require, with a `tool_use` block per call the model made.
fn anthropic_buffered_stream(
    text: String,
    stop_reason: &'static str,
    tokens: u32,
    id: String,
    model: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    fn sse((name, data): (&'static str, String)) -> Result<Event, Infallible> {
        Ok(Event::default().event(name).data(data))
    }

    let parsed = crate::gateway::toolcall::parse(&text);
    let stop = if parsed.calls.is_empty() { stop_reason } else { "tool_use" };

    let mut events = vec![sse(StreamEvents::message_start(&id, &model))];

    // Blocks are indexed across the whole message, so text and tool_use share
    // one counter — a client keys its assembly on that index.
    let mut index = 0u32;
    if !parsed.text.is_empty() {
        events.push(sse(StreamEvents::content_block_start_at(index)));
        events.push(sse(StreamEvents::content_block_delta_at(index, &parsed.text)));
        events.push(sse(StreamEvents::content_block_stop_at(index)));
        index += 1;
    }

    for call in &parsed.calls {
        let input: serde_json::Value =
            serde_json::from_str(&call.arguments).unwrap_or_else(|_| serde_json::json!({}));
        events.push(sse(StreamEvents::tool_use_start(index, &call.id, &call.name)));
        // Sent whole rather than as partial JSON: Sarathi has the complete
        // arguments by this point, and splitting them would only give the
        // client something to reassemble.
        events.push(sse(StreamEvents::tool_use_delta(index, &input.to_string())));
        events.push(sse(StreamEvents::content_block_stop_at(index)));
        index += 1;
    }

    if index == 0 {
        events.push(sse(StreamEvents::content_block_start_at(0)));
        events.push(sse(StreamEvents::content_block_stop_at(0)));
    }

    events.push(sse(StreamEvents::message_delta(stop, tokens)));
    events.push(sse(StreamEvents::message_stop()));

    stream::iter(events)
}

/// The exact event order Anthropic clients require:
/// `message_start` → `content_block_start` → `content_block_delta`* →
/// `content_block_stop` → `message_delta` → `message_stop`.
fn anthropic_stream(
    state: Arc<GatewayState>,
    handle: GenerationHandle,
    mut guard: CancelOnDrop,
    first: Option<StreamChunk>,
    id: String,
    model: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let canceller: Canceller = handle.canceller();
    // See the OpenAI stream: the guard comes from the handler, so the gap
    // between submitting and streaming is covered too.
    // Put back what `peek` took, or the first token never reaches the client.
    let chunks = stream::iter(first).chain(UnboundedReceiverStream::new(handle.chunks));
    let outcome = Outcome::new();

    fn sse((name, data): (&'static str, String)) -> Result<Event, Infallible> {
        Ok(Event::default().event(name).data(data))
    }

    let head = stream::iter(vec![
        sse(StreamEvents::message_start(&id, &model)),
        sse(StreamEvents::content_block_start()),
    ]);

    let body_outcome = outcome.clone();
    let body = chunks.filter_map(move |chunk: StreamChunk| {
        let event = if let Some(failure) = &chunk.error {
            // Too late for a status code, but the client still gets the reason.
            guard.disarm();
            body_outcome.record("error".to_string(), 0);
            Some(Ok(error_event(Dialect::Anthropic, failure)))
        } else if chunk.is_final {
            // Generation ended on its own, so dropping the guard must not read
            // as an abandonment.
            guard.disarm();
            // Terminal events belong to `tail`; just record how it ended.
            let reason = anthropic::map_stop_reason(chunk.finish_reason.as_deref().unwrap_or("stop"));
            body_outcome.record(reason.to_string(), chunk.tokens_generated.unwrap_or(0));
            None
        } else if chunk.text.is_empty() {
            None
        } else {
            Some(sse(StreamEvents::content_block_delta(&chunk.text)))
        };
        async move { event }
    });

    let tail_outcome = outcome.clone();
    let tail = stream::once(async move {
        canceller.cancel();
        state.finish_request();
        tail_outcome.get()
    })
    .flat_map(|(stop_reason, tokens)| {
        stream::iter(vec![
            sse(StreamEvents::content_block_stop()),
            sse(StreamEvents::message_delta(&stop_reason, tokens)),
            sse(StreamEvents::message_stop()),
        ])
    });

    head.chain(body).chain(tail)
}

#[cfg(test)]
mod server_tests {
    use super::*;
    use crate::ai_engine::scheduler::GenerationScheduler;
    use crate::gateway::state::GatewayConfig;

    fn test_state(port: u16) -> Arc<GatewayState> {
        let inference = Arc::new(crate::ai_engine::manager::InferenceManager::new());
        let scheduler = Arc::new(GenerationScheduler::start(inference.clone()));
        Arc::new(GatewayState::new(
            scheduler,
            inference,
            GatewayConfig { enabled: true, port, apply_capabilities: false },
        ))
    }

    /// Port 0 asks the OS for a free port, so tests never collide with a running
    /// app or with each other.
    async fn start_on_ephemeral_port() -> (GatewayHandle, u16) {
        let handle = start_gateway(test_state(0)).await.expect("gateway should bind");
        let port = handle.port;
        assert_ne!(port, 0, "handle must report the OS-assigned port, not the requested 0");
        (handle, port)
    }

    #[tokio::test]
    async fn the_server_answers_while_the_handle_is_held() {
        let (_handle, port) = start_on_ephemeral_port().await;

        let body = reqwest::get(format!("http://127.0.0.1:{port}/health"))
            .await
            .expect("health should respond")
            .text()
            .await
            .expect("health should have a body");

        assert!(body.contains("\"status\""), "unexpected health body: {body}");
    }

    /// Regression: the handle owns the graceful-shutdown sender, so letting it
    /// fall out of scope stopped the server milliseconds after it logged
    /// "Listening". The app reported a healthy gateway while refusing every
    /// connection, which is exactly what happened in practice.
    #[tokio::test]
    async fn dropping_the_handle_stops_the_server() {
        let (handle, port) = start_on_ephemeral_port().await;
        let url = format!("http://127.0.0.1:{port}/health");

        assert!(
            reqwest::get(&url).await.is_ok(),
            "server should answer while the handle is alive"
        );

        drop(handle);
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        assert!(
            reqwest::get(&url).await.is_err(),
            "server should stop once the handle drops — if this ever fails, the \
             shutdown wiring changed and lib.rs may no longer need to retain it"
        );
    }

    #[tokio::test]
    async fn a_taken_port_falls_back_instead_of_leaving_the_gateway_down() {
        // Previously this returned an error and the app ran with no gateway for
        // the rest of the session: every client got ConnectionRefused, retrying
        // could never help because nothing would ever listen, and the dashboard
        // still reported the server as running. Serving somewhere is strictly
        // better than serving nowhere, and callers are handed the live port
        // rather than the configured one.
        let (_first, port) = start_on_ephemeral_port().await;

        let state = test_state(port);
        let second = start_gateway(state.clone())
            .await
            .expect("a taken port must not stop the gateway from starting");

        assert_ne!(second.port, port, "it must move off the port already in use");
        assert_ne!(second.port, 0, "the fallback must report a real bound port");

        // The live port has to be published, or launched tools are handed an
        // address nothing is listening on.
        assert_eq!(
            state.port(),
            second.port,
            "state must carry the port that was actually bound"
        );

        // And it must genuinely be accepting connections there.
        assert!(
            tokio::net::TcpStream::connect(("127.0.0.1", second.port))
                .await
                .is_ok(),
            "the fallback port must be live"
        );
    }
}
