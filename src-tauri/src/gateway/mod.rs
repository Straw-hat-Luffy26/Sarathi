//! Local HTTP gateway — lets external coding tools use Sarathi's loaded model.
//!
//! Sarathi is the engine room: it owns the model, and tools like Claude Code,
//! opencode, and openclaw connect to it rather than loading their own.
//!
//! Two protocol surfaces are served, because the ecosystem is split:
//!
//! - `POST /v1/chat/completions` — OpenAI shape (opencode, openclaw, Cursor)
//! - `POST /v1/messages` — Anthropic shape (Claude Code, via `ANTHROPIC_BASE_URL`)
//!
//! Both funnel into the same [`GenerationScheduler`], so external requests and
//! the desktop app share one model and take turns rather than competing for it.

pub mod anthropic;
pub mod guard;
pub mod openai;
pub mod server;
pub mod state;
pub mod toolcall;

pub use server::{start_gateway, GatewayHandle};
pub use state::{GatewayConfig, GatewayState, GatewayStats, ClientActivity};

/// Default port.
///
/// Not Ollama's 11434, and no longer 11435 either: ARJUN, which grew out of this
/// codebase, kept 11435 for its own gateway. With both installed, whichever
/// started second fell back to a random port, and every tool pointed at 11435
/// quietly talked to the *other* application's model. 11535 is used by neither.
pub const DEFAULT_PORT: u16 = 11535;
