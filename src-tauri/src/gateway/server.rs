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

/// Characters per token, for sizing a prompt before the tokenizer sees it.
///
/// Deliberately pessimistic. English prose runs about four characters to the
/// token, but tool schemas are JSON — braces, quotes, colons and snake_case keys
/// all tokenize densely — and an estimate that runs *over* the real count
/// silently reintroduces the overflow this is here to prevent. Measured against
/// a real failure: 125 schemas estimated at 46k tokens on the four-character
/// rule actually tokenized to a 49.5k prompt.
const CHARS_PER_TOKEN: usize = 3;

/// Tokens held back for chat-template scaffolding and estimation error.
const FIT_MARGIN_TOKENS: usize = 1024;

/// Drops tool definitions that cannot fit the loaded model's context.
///
/// ## Why this exists
///
/// Sarathi hands every MCP server in `mcp.json` to every provider it launches,
/// and an agentic client re-sends all of their schemas on every single turn. Six
/// servers came to 125 tools and ~46k tokens — so a one-word "Hii" arrived as a
/// 49,478-token prompt against a 32,768-token context and was rejected outright.
/// The user had done nothing wrong and the conversation could not proceed at
/// all; the only advice on offer was to hand-edit `mcp.json`.
///
/// Sizing what a model is given to what that model can hold is Sarathi's job,
/// the same way sizing a download to the GPU is. So the tool list is trimmed to
/// fit and the request goes through, rather than failing whole.
///
/// ## What gets dropped
///
/// The tail of the list. Clients put their own built-in tools first and append
/// MCP-provided ones, so trimming from the end keeps the tools an agent cannot
/// work without — reading and writing files, running commands — and sheds the
/// long tail of optional integrations. Nothing is reordered, because a client
/// that does put something essential last would be worse served by a cleverer
/// rule that guessed which those were.
///
/// Returns how many were removed. Zero is the normal case and costs one pass
/// over the schemas.
fn fit_tools_to_context(
    messages: &[ChatMessage],
    params: &mut GenerationParams,
    context_length: u32,
) -> usize {
    if params.tools.is_empty() || context_length == 0 {
        return 0;
    }

    let conversation_tokens: usize =
        messages.iter().map(|m| m.content.len()).sum::<usize>() / CHARS_PER_TOKEN;

    // Room the answer itself needs. Without this a prompt could fit exactly and
    // then overflow the moment generation started.
    let reserved = conversation_tokens
        .saturating_add(params.max_tokens as usize)
        .saturating_add(FIT_MARGIN_TOKENS);

    let Some(mut remaining) = (context_length as usize).checked_sub(reserved) else {
        // The conversation alone does not fit. Nothing to be gained by keeping
        // tools; the overflow error will explain the real problem.
        let dropped = params.tools.len();
        params.tools.clear();
        return dropped;
    };

    let before = params.tools.len();
    let mut kept = 0;
    for tool in &params.tools {
        let cost = serde_json::to_string(tool).map_or(0, |s| s.len()) / CHARS_PER_TOKEN;
        match remaining.checked_sub(cost) {
            Some(left) => {
                remaining = left;
                kept += 1;
            }
            None => break,
        }
    }

    params.tools.truncate(kept);
    before - kept
}

/// Applies [`fit_tools_to_context`] against whatever model is loaded, and says
/// so when it had to take something away.
///
/// Logged at warning level rather than silently: an agent that suddenly cannot
/// see a tool it used last week needs *some* trace explaining why, and the fix
/// — fewer MCP servers, or a model with a longer context — belongs in the log
/// next to the evidence.
fn trim_tools_for_model(
    state: &GatewayState,
    messages: &[ChatMessage],
    params: &mut GenerationParams,
    client: &str,
) {
    let Some(info) = state.inference.get_loaded_model_info() else {
        return;
    };

    let offered = params.tools.len();
    let dropped = fit_tools_to_context(messages, params, info.context_length);
    if dropped > 0 {
        log::warn!(
            "[GATEWAY] {client}: {offered} tools would not fit {}'s {}-token context; \
             sent the first {} and dropped {dropped}. Disable MCP servers in mcp.json, \
             or load a model with a longer context, to stop this happening.",
            info.model_id,
            info.context_length,
            params.tools.len(),
        );
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

    let (messages, mut params) = (req.to_chat_messages(), req.to_generation_params());
    trim_tools_for_model(&state, &messages, &mut params, &client);
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

    // Wait for the first chunk before committing to a 200, so a request that
    // fails during prefill is answered rather than left blank.
    //
    // This is also what lets the tool-carrying path stream. It used to `collect`
    // the entire answer first, reasoning that a tool call is only recognisable
    // whole and that "no one is watching tokens appear" on the agentic path. The
    // first half is true of the *call*; it is not true of the prose beside it,
    // and every agentic client sends tools on every request — so in practice
    // nothing served through this gateway ever streamed, and
    // time-to-first-visible-token was the entire generation time. Peeking gives
    // the same guarantee collecting did — an overflow is a real HTTP error
    // rather than an empty 200 — without paying for the rest of the answer.
    let first = peek(&mut handle).await;
    if let Some(failure) = first.as_ref().and_then(|c| c.error.as_ref()) {
        guard.disarm();
        state.finish_request();
        return error_response(Dialect::OpenAi, &explain_overflow(failure, shape.tools, shape.tool_chars));
    }

    // With tools offered, the same stream runs through a sieve that holds back
    // anything that might be the opening of a call.
    if !req.tools.is_empty() {
        return Sse::new(openai_tool_stream(
            state.clone(), handle, guard, first, id, created, model,
        ))
        .keep_alive(KeepAlive::default())
        .into_response();
    }

    Sse::new(openai_stream(state.clone(), handle, guard, first, id, created, model))
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Streams an answer that may contain tool calls.
///
/// Identical to [`openai_stream`] except that text passes through a
/// [`StreamSieve`](crate::gateway::toolcall::StreamSieve), which releases prose
/// immediately and withholds anything that could be the opening of a call. At
/// the end the accumulated output is parsed once: a call is emitted as a proper
/// `tool_calls` delta, and text the sieve held back that turned out to be
/// ordinary prose is released.
fn openai_tool_stream(
    state: Arc<GatewayState>,
    handle: GenerationHandle,
    mut guard: CancelOnDrop,
    first: Option<StreamChunk>,
    id: String,
    created: u64,
    model: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let canceller: Canceller = handle.canceller();
    let chunks = stream::iter(first).chain(UnboundedReceiverStream::new(handle.chunks));

    let (head_id, head_model) = (id.clone(), model.clone());
    let head = stream::once(async move {
        Ok(Event::default().data(json_of(&ChatCompletionChunk::opening(&head_id, created, &head_model))))
    });

    let mut sieve = crate::gateway::toolcall::StreamSieve::new();
    let (body_id, body_model) = (id.clone(), model.clone());
    let body = chunks.flat_map(move |chunk: StreamChunk| {
        let mut events: Vec<Result<Event, Infallible>> = Vec::new();

        if chunk.is_final {
            guard.disarm();
        }

        if let Some(failure) = &chunk.error {
            // Past the point where an HTTP status was still possible.
            events.push(Ok(error_event(Dialect::OpenAi, failure)));
        } else if chunk.is_final {
            let natural = chunk.finish_reason.clone().unwrap_or_else(|| "stop".to_string());
            let parsed = crate::gateway::toolcall::parse(sieve.full());

            if let Some(rest) = sieve.finish(&parsed) {
                events.push(Ok(Event::default().data(json_of(&ChatCompletionChunk::text(
                    &body_id, created, &body_model, rest,
                )))));
            }
            if !parsed.calls.is_empty() {
                events.push(Ok(Event::default().data(json_of(&ChatCompletionChunk::tool_calls(
                    &body_id, created, &body_model, &parsed.calls,
                )))));
            }
            events.push(Ok(Event::default().data(json_of(&ChatCompletionChunk::closing(
                &body_id, created, &body_model, parsed.finish_reason(&natural),
            )))));
        } else if !chunk.text.is_empty() {
            if let Some(out) = sieve.push(&chunk.text) {
                events.push(Ok(Event::default().data(json_of(&ChatCompletionChunk::text(
                    &body_id, created, &body_model, out,
                )))));
            }
        }

        stream::iter(events)
    });

    let tail = stream::once(async move {
        // The client has read everything; make sure nothing keeps generating.
        canceller.cancel();
        state.finish_request();
        Ok(Event::default().data("[DONE]"))
    });

    head.chain(body).chain(tail)
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

    let (messages, mut params) = (req.to_chat_messages(), req.to_generation_params());
    trim_tools_for_model(&state, &messages, &mut params, &client);
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

    // Peeking rather than collecting, for the reason given on the OpenAI path:
    // it keeps the guarantee that an overflow is a real HTTP error, without
    // withholding the answer until it is complete. This is the path Claude Code
    // and every other agentic client uses, and it always carries tools — so
    // collecting here meant nothing ever streamed.
    let first = peek(&mut handle).await;
    if let Some(failure) = first.as_ref().and_then(|c| c.error.as_ref()) {
        guard.disarm();
        state.finish_request();
        return error_response(Dialect::Anthropic, &explain_overflow(failure, shape.tools, shape.tool_chars));
    }

    if !req.tools.is_empty() {
        return Sse::new(anthropic_tool_stream(state.clone(), handle, guard, first, id, model))
            .keep_alive(KeepAlive::default())
            .into_response();
    }

    Sse::new(anthropic_stream(state.clone(), handle, guard, first, id, model))
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Streams an Anthropic answer that may contain tool calls.
///
/// Text streams into content block 0 as it is produced, filtered by a
/// [`StreamSieve`](crate::gateway::toolcall::StreamSieve). Any calls are emitted
/// as `tool_use` blocks after that one closes, which is the same block ordering
/// the buffered form produced — clients key their assembly on the index, so the
/// numbering has to stay stable.
fn anthropic_tool_stream(
    state: Arc<GatewayState>,
    handle: GenerationHandle,
    mut guard: CancelOnDrop,
    first: Option<StreamChunk>,
    id: String,
    model: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let canceller: Canceller = handle.canceller();
    let chunks = stream::iter(first).chain(UnboundedReceiverStream::new(handle.chunks));
    let outcome = Outcome::new();
    let calls: Arc<Mutex<Vec<crate::gateway::toolcall::ToolCall>>> = Arc::new(Mutex::new(Vec::new()));

    fn sse((name, data): (&'static str, String)) -> Result<Event, Infallible> {
        Ok(Event::default().event(name).data(data))
    }

    let head = stream::iter(vec![
        sse(StreamEvents::message_start(&id, &model)),
        sse(StreamEvents::content_block_start()),
    ]);

    let mut sieve = crate::gateway::toolcall::StreamSieve::new();
    let body_outcome = outcome.clone();
    let body_calls = calls.clone();
    let body = chunks.flat_map(move |chunk: StreamChunk| {
        let mut events: Vec<Result<Event, Infallible>> = Vec::new();

        if let Some(failure) = &chunk.error {
            // Too late for a status code, but the client still gets the reason.
            guard.disarm();
            body_outcome.record("error".to_string(), 0);
            events.push(Ok(error_event(Dialect::Anthropic, failure)));
        } else if chunk.is_final {
            // Generation ended on its own, so dropping the guard must not read
            // as an abandonment.
            guard.disarm();

            let parsed = crate::gateway::toolcall::parse(sieve.full());
            if let Some(rest) = sieve.finish(&parsed) {
                events.push(sse(StreamEvents::content_block_delta(&rest)));
            }

            let natural =
                anthropic::map_stop_reason(chunk.finish_reason.as_deref().unwrap_or("stop"));
            let reason = if parsed.calls.is_empty() { natural } else { "tool_use" };
            if let Ok(mut slot) = body_calls.lock() {
                *slot = parsed.calls;
            }
            // Terminal events belong to `tail`; just record how it ended.
            body_outcome.record(reason.to_string(), chunk.tokens_generated.unwrap_or(0));
        } else if !chunk.text.is_empty() {
            if let Some(out) = sieve.push(&chunk.text) {
                events.push(sse(StreamEvents::content_block_delta(&out)));
            }
        }

        stream::iter(events)
    });

    let tail_outcome = outcome.clone();
    let tail = stream::once(async move {
        canceller.cancel();
        state.finish_request();
        let (reason, tokens) = tail_outcome.get();
        let calls = calls.lock().map(|g| g.clone()).unwrap_or_default();
        (reason, tokens, calls)
    })
    .flat_map(|(stop_reason, tokens, calls)| {
        // Block 0 is the text block opened in `head`; tool_use blocks follow it.
        let mut events = vec![sse(StreamEvents::content_block_stop())];

        for (offset, call) in calls.iter().enumerate() {
            let index = offset as u32 + 1;
            let input: serde_json::Value =
                serde_json::from_str(&call.arguments).unwrap_or_else(|_| serde_json::json!({}));
            events.push(sse(StreamEvents::tool_use_start(index, &call.id, &call.name)));
            // Sent whole rather than as partial JSON: the arguments are complete
            // by this point, and splitting them would only give the client
            // something to reassemble.
            events.push(sse(StreamEvents::tool_use_delta(index, &input.to_string())));
            events.push(sse(StreamEvents::content_block_stop_at(index)));
        }

        events.push(sse(StreamEvents::message_delta(&stop_reason, tokens)));
        events.push(sse(StreamEvents::message_stop()));
        stream::iter(events)
    });

    head.chain(body).chain(tail)
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
mod fit_tests {
    use super::*;

    /// A tool schema of roughly `chars` characters, as an MCP server would send.
    fn tool(name: &str, chars: usize) -> serde_json::Value {
        serde_json::json!({
            "name": name,
            "description": "x".repeat(chars),
            "input_schema": { "type": "object" },
        })
    }

    fn params(tools: Vec<serde_json::Value>, max_tokens: u32) -> GenerationParams {
        GenerationParams { tools, max_tokens, ..Default::default() }
    }

    fn msg(content: &str) -> ChatMessage {
        ChatMessage::new("user", content)
    }

    /// The reported failure, reproduced: 125 tools of schema against a
    /// 32,768-token context and a one-word message. It must now fit.
    #[test]
    fn the_reported_overflow_is_trimmed_until_it_fits() {
        // ~184k chars of schema across 125 tools, which is what produced the
        // 49,478-token prompt.
        let tools: Vec<_> = (0..125).map(|i| tool(&format!("t{i}"), 1470)).collect();
        let mut p = params(tools, 4096);
        let messages = vec![msg("Hii")];

        let dropped = fit_tools_to_context(&messages, &mut p, 32_768);

        assert!(dropped > 0, "125 schemas cannot fit a 32k context");
        assert!(!p.tools.is_empty(), "trimming must not strip the agent of every tool");

        let kept_tokens: usize = p
            .tools
            .iter()
            .map(|t| serde_json::to_string(t).unwrap().len() / CHARS_PER_TOKEN)
            .sum();
        assert!(
            kept_tokens + 4096 + FIT_MARGIN_TOKENS <= 32_768,
            "what survived still has to fit: {kept_tokens} tokens of schema"
        );
    }

    /// The ordinary case must cost nothing: a tool set that fits is untouched.
    #[test]
    fn a_tool_set_that_fits_is_left_alone() {
        let tools: Vec<_> = (0..8).map(|i| tool(&format!("t{i}"), 400)).collect();
        let mut p = params(tools, 1024);

        assert_eq!(fit_tools_to_context(&[msg("hello")], &mut p, 32_768), 0);
        assert_eq!(p.tools.len(), 8);
    }

    /// Clients list their own built-ins first and append MCP tools, so the head
    /// of the list is what an agent cannot work without.
    #[test]
    fn trimming_takes_from_the_tail_and_keeps_the_order() {
        let tools = vec![
            tool("Read", 300),
            tool("Write", 300),
            tool("Bash", 300),
            tool("mcp_huge", 60_000),
        ];
        let mut p = params(tools, 512);

        assert_eq!(fit_tools_to_context(&[msg("hi")], &mut p, 4096), 1);
        let names: Vec<&str> = p.tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert_eq!(names, vec!["Read", "Write", "Bash"]);
    }

    /// When the conversation alone overflows, no tool selection can save it —
    /// and the overflow error should describe that, not a tool problem.
    #[test]
    fn a_conversation_that_cannot_fit_drops_every_tool() {
        let mut p = params(vec![tool("a", 100)], 512);
        let huge = "x".repeat(500_000);

        assert_eq!(fit_tools_to_context(&[msg(&huge)], &mut p, 8192), 1);
        assert!(p.tools.is_empty());
    }

    #[test]
    fn a_request_with_no_tools_is_untouched() {
        let mut p = params(vec![], 512);
        assert_eq!(fit_tools_to_context(&[msg("hi")], &mut p, 8192), 0);
    }

    /// An unknown context must not be treated as "zero room for anything".
    #[test]
    fn an_unknown_context_length_trims_nothing() {
        let mut p = params(vec![tool("a", 100)], 512);
        assert_eq!(fit_tools_to_context(&[msg("hi")], &mut p, 0), 0);
        assert_eq!(p.tools.len(), 1);
    }

    /// Room for the answer is reserved too, or a prompt could fit exactly and
    /// then overflow the moment generation began.
    #[test]
    fn the_reply_gets_room_reserved_for_it() {
        // Sized so the two budgets genuinely differ: 20 schemas of ~1,000
        // tokens each against a 16k context, where the answer's share decides
        // how many survive.
        let tools: Vec<_> = (0..20).map(|i| tool(&format!("t{i}"), 3000)).collect();

        let mut small = params(tools.clone(), 256);
        let mut large = params(tools, 8192);
        fit_tools_to_context(&[msg("hi")], &mut small, 16_384);
        fit_tools_to_context(&[msg("hi")], &mut large, 16_384);

        assert!(
            large.tools.len() < small.tools.len(),
            "a bigger max_tokens must leave less room for schemas"
        );
    }
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
