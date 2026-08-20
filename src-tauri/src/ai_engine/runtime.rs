//! LlamaCpp Runtime — In-process GGUF model inference via llama-cpp-2
//!
//! Provides model loading, token generation with streaming, and resource management.
//! Designed to be wrapped by InferenceManager for thread-safe Tauri integration.

use std::ffi::CString;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use sha2::{Digest, Sha256};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;

use crate::ai_engine::lora_binding::LoraAdapterCache;
use crate::ai_engine::traits::*;
use crate::capability::CapabilityBackend;

/// Buffer-type override patterns keeping the routed experts of the first
/// `n_layers` layers in system RAM.
///
/// This is precisely what `--n-cpu-moe N` does. There is no distinct C API for
/// that flag: llama.cpp implements it in `common/arg.cpp` as a loop pushing one
/// buffer-type override per layer, each built from `llm_ffn_exps_block_regex(i)`
/// and bound to the CPU buffer type.
///
/// The binding's own [`LlamaModelParams::add_cpu_moe_override`] is deliberately
/// **not** used. Its regex is `\.ffn_(up|down|gate)_(ch|)exps`, which drops the
/// `gate_up` alternative that upstream carries in `common/common.h`. Models with
/// *fused* expert tensors — `blk.N.ffn_gate_up_exps`, registered in
/// `llama-arch.cpp` and shipped by gpt-oss — would not match, and the regex
/// cannot fall through to the `gate` branch either: after `gate` it requires
/// `_`, then optionally `ch`, then `exps`, while the tensor has `_up_exps`
/// there. The offload would silently do nothing and the model would spill or
/// run out of memory with nothing explaining why.
fn cpu_moe_override_patterns(n_layers: u32) -> Vec<CString> {
    (0..n_layers)
        // The pattern is generated, so it can never contain an interior NUL.
        .filter_map(|i| CString::new(format!(r"blk\.{i}\.ffn_(up|down|gate|gate_up)_(ch|)exps")).ok())
        .collect()
}

/// Core inference runtime wrapping llama.cpp via safe Rust bindings.
///
/// This struct owns the model, context, and backend. It is NOT thread-safe by itself;
/// the InferenceManager wraps it in Arc<Mutex<>> for safe concurrent access.
/// A live inference context plus the exact tokens its KV cache holds.
///
/// ## Why this exists
///
/// A context used to be created for every request and dropped at the end of it,
/// which meant the KV cache was thrown away and the *entire* prompt was decoded
/// again on every turn. Measured on an RTX 5060 with Qwen2.5-Coder-7B Q4_0 and a
/// ~2,850-token agent prompt, that was ~1,200 ms of prefill per request, paid
/// again on turn two and turn three even though nothing before the last user
/// message had changed. Prefill was the whole of time-to-first-token; decode was
/// already running at ~50 tok/s.
///
/// Keeping the context alive lets an unchanged prefix stay in the cache, so a
/// follow-up turn only decodes what the user actually added.
///
/// ## What is tracked
///
/// `tokens` mirrors, exactly, what has been decoded into sequence 0 — prompt
/// tokens *and* the tokens generated from them. Generated tokens matter as much
/// as prompt tokens: the next turn's prompt replays the assistant's own reply,
/// so leaving it cached extends the reusable prefix past the previous prompt.
///
/// It is only ever appended to after a successful `decode`, so it cannot claim
/// more than the cache holds. Any failure or cancellation drops the whole
/// session rather than trying to repair it — a mismatch between this list and
/// the real cache would produce silently wrong output, which is far worse than
/// one rebuilt context.
struct GenerationSession {
    /// Borrows the model in `LlamaCppRuntime::model`, with the lifetime erased.
    ///
    /// SAFETY: the referent is a `Box<LlamaModel>` owned by the same struct, so
    /// its address is stable for as long as the box lives. The borrow is only
    /// sound while that model is alive, which is upheld by dropping this session
    /// before the model in every path that clears or replaces it
    /// (`end_session`, called from `unload_model` and `load_model`).
    ctx: LlamaContext<'static>,
    /// Tokens currently decoded into sequence 0, in order.
    tokens: Vec<LlamaToken>,
    /// Context window this was built with. A different one needs a new context.
    n_ctx: u32,
    /// The LoRA binding in force, as (path, scale bits).
    ///
    /// A bound adapter is part of the context's state and cannot be swapped
    /// without invalidating everything decoded under it, so a request wanting a
    /// different adapter gets a fresh context.
    adapter_key: Option<(std::path::PathBuf, u32)>,
}

// SAFETY: `LlamaContext` holds raw pointers into llama.cpp and so is not `Send`
// by default. Nothing about a llama.cpp context is thread-*affine*, though — it
// may be used from any thread provided it is never used from two at once, which
// is exactly the guarantee the surrounding types already give:
//
//   - The only owner is `LlamaCppRuntime`, reachable solely through
//     `InferenceManager`'s `Mutex<LlamaCppRuntime>`, so every use holds the lock.
//   - Generation is funnelled through `GenerationScheduler`'s single worker
//     thread, which processes one job at a time.
//
// Before this session existed the context was a local inside `generate`, so the
// question never arose; storing it between requests is what surfaces it. The
// mutex, not the thread identity, is what makes this sound.
unsafe impl Send for GenerationSession {}

/// How much of `prompt` is already decoded in `cached`.
///
/// Capped one token short of the prompt: llama.cpp draws the first sampled token
/// from the logits of the last decoded position, so at least one prompt token
/// must always go through `decode`. Reusing all of them would leave the sampler
/// with no logits to read.
fn reusable_prefix(cached: &[LlamaToken], prompt: &[LlamaToken]) -> usize {
    let ceiling = prompt.len().saturating_sub(1);
    cached
        .iter()
        .zip(prompt.iter())
        .take(ceiling)
        .take_while(|(a, b)| a == b)
        .count()
}

pub struct LlamaCppRuntime {
    backend: Option<LlamaBackend>,
    /// Boxed so the model has a stable heap address.
    ///
    /// [`GenerationSession`] holds a context that borrows this model, with the
    /// borrow's lifetime erased. Moving a `LlamaModel` by value would move the
    /// pointee and dangle that borrow; moving a `Box` moves only the pointer.
    model: Option<Box<LlamaModel>>,
    /// The live context and the tokens currently sitting in its KV cache.
    ///
    /// Kept between requests so an unchanged prompt prefix does not have to be
    /// decoded again. See [`GenerationSession`].
    ///
    /// MUST be dropped before `model`: it borrows it. Every path that clears or
    /// replaces the model goes through [`Self::end_session`] first.
    session: Option<GenerationSession>,
    loaded_info: Option<LoadedModelInfo>,
    /// The chat template baked into the loaded GGUF, when it ships one.
    ///
    /// Rendering through llama.cpp's own template engine is the only way to be
    /// sure the prompt matches what the model was trained on. The hand-written
    /// templates in `format_chat_prompt_with_template` are guesses keyed off a
    /// family name, and guessing wrong produces confident nonsense rather than
    /// an error — so they are only a fallback for GGUFs that carry no template.
    native_template: Option<NativeChatTemplate>,
    is_generating: Arc<AtomicBool>,
    /// LoRA adapters initialised against the currently loaded model.
    /// Cleared on unload — entries are only valid for the model they were built from.
    adapter_cache: LoraAdapterCache,
}

impl LlamaCppRuntime {
    /// Drops the cached context, freeing its KV cache and compute buffers.
    ///
    /// This is the only correct way to get rid of a session, and it must run
    /// before the model is replaced or dropped: the context borrows the model.
    /// Called on unload, on load, and whenever a decode fails — a session whose
    /// token list might not match its cache has to go, because reusing a prefix
    /// that is not really there produces confident nonsense rather than an error.
    fn end_session(&mut self) {
        if self.session.take().is_some() {
            log::debug!("[RUNTIME] Inference context released");
        }
    }

    /// Creates a new unloaded runtime instance
    pub fn new() -> Self {
        Self {
            backend: None,
            model: None,
            session: None,
            loaded_info: None,
            native_template: None,
            is_generating: Arc::new(AtomicBool::new(false)),
            adapter_cache: LoraAdapterCache::new(),
        }
    }

    /// Returns the current runtime status
    pub fn status(&self) -> RuntimeStatus {
        if self.is_generating.load(Ordering::Relaxed) {
            RuntimeStatus::Generating
        } else if self.model.is_some() {
            RuntimeStatus::Ready
        } else {
            RuntimeStatus::NotLoaded
        }
    }

    /// Returns info about the currently loaded model, if any
    pub fn loaded_model_info(&self) -> Option<&LoadedModelInfo> {
        self.loaded_info.as_ref()
    }

    /// Returns a clone of the generation cancellation flag
    pub fn cancel_flag(&self) -> Arc<AtomicBool> {
        self.is_generating.clone()
    }

    /// Loads a GGUF model from disk into memory.
    ///
    /// `status_callback` is called with human-readable step descriptions.
    pub fn load_model<F>(&mut self, config: &ModelLoadConfig, mut status_cb: F) -> Result<LoadedModelInfo>
    where
        F: FnMut(&str),
    {
        // Ensure no model is already loaded
        if self.model.is_some() {
            self.unload_model()?;
        }

        log::info!(
            "[STAGE 4 RUNTIME] load_model entered: path='{}', id='{}', ctx={}, gpu_layers={}, threads={}",
            config.model_path, config.model_id, config.context_length, config.gpu_layers, config.threads
        );

        let model_path = Path::new(&config.model_path);
        let path_exists = model_path.exists();
        log::info!("[STAGE 4 RUNTIME] Model path check: '{:?}' exists={}", model_path, path_exists);
        if !path_exists {
            let err = anyhow!("[STAGE 4 RUNTIME ERROR] GGUF model file not found at '{:?}'", model_path);
            log::error!("{}", err);
            return Err(err);
        }

        // Step 1: Initialize backend (keep alive across model loads/reloads)
        status_cb("Initializing llama.cpp backend...");
        if self.backend.is_none() {
            log::info!("[STAGE 4 RUNTIME] Initializing process-global LlamaBackend instance...");
            match LlamaBackend::init() {
                Ok(b) => {
                    log::info!("[STAGE 4 RUNTIME] LlamaBackend initialized successfully!");
                    self.backend = Some(b);
                }
                Err(e) => {
                    let err = anyhow!("[STAGE 4 RUNTIME ERROR] LlamaBackend::init() failed: {:?}", e);
                    log::error!("{}", err);
                    return Err(err);
                }
            }
        } else {
            log::info!("[STAGE 4 RUNTIME] Reusing active process-global LlamaBackend instance");
        }
        let backend = self.backend.as_ref().unwrap();

        // Step 2: Configure model parameters
        status_cb("Configuring model parameters...");
        log::info!(
            "[STAGE 4 RUNTIME] Configuring LlamaModelParams (gpu_layers={}, threads={})",
            config.gpu_layers, config.threads
        );

        // GPU offload only takes effect when llama.cpp was compiled with a GPU
        // backend. Requesting layers without one is a silent no-op inside
        // llama.cpp, which previously made CPU-only builds appear GPU-accelerated
        // in the logs and the UI. Surface the mismatch instead of hiding it.
        let gpu_backend_compiled = cfg!(any(feature = "cuda", feature = "vulkan"));
        if config.gpu_layers > 0 && !gpu_backend_compiled {
            log::warn!(
                "[STAGE 4 RUNTIME WARN] {} GPU layers requested, but this binary was built \
                 without a GPU backend — llama.cpp will ignore the request and run on CPU. \
                 Rebuild with `--features cuda` (needs the CUDA Toolkit) or `--features vulkan`.",
                config.gpu_layers
            );
        }

        let effective_gpu_layers = if gpu_backend_compiled { config.gpu_layers } else { 0 };

        // Pinning experts to CPU is meaningless without a GPU backend — every
        // tensor is already in system RAM — so it is reported as not applied
        // rather than silently requested.
        let effective_cpu_moe = if gpu_backend_compiled { config.cpu_moe_layers } else { 0 };
        if config.cpu_moe_layers > 0 && !gpu_backend_compiled {
            log::warn!(
                "[STAGE 4 RUNTIME WARN] Expert offload for {} layers requested, but this binary \
                 was built without a GPU backend — every tensor is on CPU already.",
                config.cpu_moe_layers
            );
        }

        // Declared before `model_params` so it is dropped *after* them: Rust
        // drops locals in reverse declaration order, and `add_cpu_buft_override`
        // stores a borrowed pointer with no lifetime tie recorded on the params
        // (see the crate's SAFETY note on `tensor_buft_override_patterns`), so
        // these strings must outlive the params that point into them.
        let expert_patterns = cpu_moe_override_patterns(effective_cpu_moe);

        let model_params = {
            let mut params = Box::pin(LlamaModelParams::default().with_n_gpu_layers(effective_gpu_layers));
            for pattern in &expert_patterns {
                params.as_mut().add_cpu_buft_override(pattern);
            }
            params
        };

        if effective_cpu_moe > 0 {
            log::info!(
                "[STAGE 4 RUNTIME] MoE expert offload: routed experts of {} layer(s) pinned to \
                 system RAM via {} buffer-type override(s); all layers still requested on GPU",
                effective_cpu_moe,
                expert_patterns.len()
            );
        }

        // Recorded by the preflight below and used only to explain a failure —
        // a load that fails for both GPU and CPU is almost always the build not
        // recognising the architecture, and naming it is the difference between
        // an actionable message and "NullResult".
        let mut preflight_architecture: Option<String> = None;

        // Use the path exactly as resolved upstream. Rewriting separators here
        // corrupts absolute paths on every non-Windows platform.
        let clean_path = config.model_path.clone();
        let path_obj = Path::new(&clean_path);
        let exists = path_obj.exists();
        let size = std::fs::metadata(&clean_path).map(|m| m.len()).unwrap_or(0);

        log::info!(
            "[STAGE 4 RUNTIME PARAMETER AUDIT]\n  -> Raw model_path: '{}'\n  -> Cleaned model_path: '{}'\n  -> File exists: {}\n  -> File size: {} bytes\n  -> Requested gpu_layers: {}\n  -> Context length: {}\n  -> Threads: {}\n  -> Backend active: true",
            config.model_path,
            clean_path,
            exists,
            size,
            config.gpu_layers,
            config.context_length,
            config.threads
        );

        match std::fs::File::open(&clean_path) {
            Ok(mut f) => {
                let mut header_buf = [0u8; 8];
                use std::io::Read;
                if f.read_exact(&mut header_buf).is_err() || &header_buf[0..4] != b"GGUF" {
                    let err = anyhow!(
                        "[STAGE 4 RUNTIME ERROR] Corrupted/Invalid GGUF header at '{}': Expected magic bytes 'GGUF' (0x46554747), found {:?}",
                        clean_path, String::from_utf8_lossy(&header_buf[0..4])
                    );
                    log::error!("{}", err);
                    return Err(err);
                }
                let gguf_ver = u32::from_le_bytes([header_buf[4], header_buf[5], header_buf[6], header_buf[7]]);
                log::info!("[STAGE 4 RUNTIME AUDIT] std::fs::File::open & GGUF header check succeeded for '{}' (Magic='GGUF', Version={})", clean_path, gguf_ver);
            }
            Err(e) => {
                let err = anyhow!("[STAGE 4 RUNTIME ERROR] std::fs::File::open failed for '{}': {:?}", clean_path, e);
                log::error!("{}", err);
                return Err(err);
            }
        }

        // Is this file a model at all?
        //
        // The magic-bytes check above only proves the file is *a* GGUF. A vision
        // projector, an MTP head and an EAGLE-3 speculative-decoding draft are
        // all valid GGUFs, and llama.cpp answers a request to load one with a
        // null pointer — which surfaced as "GPU error: NullResult, CPU error:
        // NullResult" and sent people looking for a hardware fault.
        //
        // Asked before the load rather than after it fails, because afterwards
        // there is nothing left to inspect: a null carries no reason.
        match crate::ai_engine::gguf_meta::read_gguf_metadata(path_obj) {
            Ok(meta) => {
                if let Some(reason) = meta.role.refusal(&meta.architecture) {
                    let err = anyhow!("[STAGE 4 RUNTIME ERROR] {reason}");
                    log::error!("{}", err);
                    return Err(err);
                }
                log::info!(
                    "[STAGE 4 RUNTIME AUDIT] '{}' is a loadable model (architecture '{}', {} layers, {})",
                    clean_path,
                    meta.architecture,
                    meta.block_count,
                    if meta.is_moe() {
                        format!("MoE with {} experts", meta.expert_count)
                    } else {
                        "dense".to_string()
                    }
                );
                preflight_architecture = Some(meta.architecture);
            }
            // A header this cannot parse is not grounds to refuse the load.
            // llama.cpp reads more of the format than this does, and rejecting a
            // model it would have loaded is the worse error.
            Err(e) => log::warn!(
                "[STAGE 4 RUNTIME WARN] Could not classify '{}' before loading ({e:#}); \
                 handing it to llama.cpp anyway",
                clean_path
            ),
        }

        // Shard completeness check for split GGUFs (e.g., -00001-of-00002.gguf)
        if clean_path.contains("-00001-of-") {
            if let Some(pos) = clean_path.find("-00001-of-") {
                let suffix = &clean_path[pos + 10..];
                if let Some(end_pos) = suffix.find(".gguf") {
                    let total_shards_str = &suffix[..end_pos];
                    if let Ok(total_shards) = total_shards_str.parse::<u32>() {
                        for shard_idx in 2..=total_shards {
                            let shard_name = format!("-{:05}-of-{}", shard_idx, suffix);
                            let shard_path_str = clean_path.replace(&format!("-00001-of-{}", suffix), &shard_name);
                            let shard_path = Path::new(&shard_path_str);
                            if !shard_path.exists() || std::fs::metadata(&shard_path_str).map(|m| m.len()).unwrap_or(0) == 0 {
                                let err = anyhow!(
                                    "[STAGE 4 RUNTIME ERROR] Split GGUF shard missing or incomplete: shard {} of {} at '{}'",
                                    shard_idx, total_shards, shard_path_str
                                );
                                log::error!("{}", err);
                                return Err(err);
                            }
                            log::info!("[STAGE 4 RUNTIME AUDIT] Verified split GGUF shard {}/{} exists: '{}'", shard_idx, total_shards, shard_path_str);
                        }
                    }
                }
            }
        }

        // Step 3: Load model from GGUF file
        status_cb("Loading model weights from GGUF file...");
        log::info!(
            "\n==================== [LLAMA.CPP RUNTIME LOAD CONFIGURATION] ====================\n\
             - GGUF File Path:     {} (Source: Manifest / Resolver)\n\
             - Model ID:           {} (Source: Manifest)\n\
             - Quantization:       {} (Source: Model Manifest / User Selection)\n\
             - GPU Layers Offload: {} (Source: Hardware Profile Estimator)\n\
             - CPU Threads:        {} (Source: System Hardware Collector)\n\
             - Context Length:     {} tokens (Source: GGUF / config.json max_position_embeddings)\n\
             - Chat Template:      {} (Source: GGUF / tokenizer_config.json / Jinja)\n\
             - Stop Sequences:     {:?} (Source: GGUF / tokenizer.json added_tokens)\n\
             =================================================================================",
            clean_path,
            config.model_id,
            config.quantization,
            config.gpu_layers,
            config.threads,
            config.context_length,
            config.chat_template,
            config.stop_tokens
        );

        let model_result = LlamaModel::load_from_file(&backend, &clean_path, &model_params);

        let (model, actual_backend) = match model_result {
            Ok(m) => {
                let desc = if effective_gpu_layers > 0 && effective_cpu_moe > 0 {
                    // Names the split explicitly: a user whose experts went to
                    // system RAM should not be told the model is on the GPU.
                    format!(
                        "llama.cpp (GPU + experts of {} layers on CPU)",
                        effective_cpu_moe
                    )
                } else if effective_gpu_layers > 0 {
                    format!("llama.cpp (GPU offload: {} layers)", effective_gpu_layers)
                } else if config.gpu_layers > 0 {
                    "llama.cpp (CPU — built without GPU backend)".to_string()
                } else {
                    "llama.cpp (CPU)".to_string()
                };
                log::info!("[STAGE 4 RUNTIME] LlamaModel::load_from_file succeeded with backend: {}", desc);
                (m, desc)
            }
            Err(e) => {
                log::warn!(
                    "[STAGE 4 RUNTIME WARN] LlamaModel::load_from_file(gpu_layers={}) failed with error: {:?}. Attempting CPU fallback (gpu_layers=0)...",
                    config.gpu_layers, e
                );
                let cpu_params = LlamaModelParams::default().with_n_gpu_layers(0);
                match LlamaModel::load_from_file(&backend, &clean_path, &cpu_params) {
                    Ok(cpu_model) => {
                        log::info!("[STAGE 4 RUNTIME] CPU fallback LlamaModel::load_from_file succeeded!");
                        (cpu_model, "llama.cpp (CPU Fallback)".to_string())
                    }
                    Err(e2) => {
                        // Failing on CPU too rules out the offload plan, the
                        // card and its driver: those are the only things the
                        // two attempts differ in. What is left is the file, and
                        // llama.cpp's way of saying it cannot read one is a null
                        // pointer with no reason attached — so the reason has to
                        // be supplied here.
                        let cause = match &preflight_architecture {
                            Some(arch) => format!(
                                "This build of llama.cpp does not support the '{arch}' \
                                 architecture. It is a valid model file; the bundled runtime is \
                                 too old to read it, or the architecture is not implemented \
                                 upstream. Check for a Sarathi update, or pick a model built on \
                                 an architecture this version supports."
                            ),
                            None => "The file could not be read as a model. It may be truncated, \
                                     corrupted, or not a model at all — a failed or interrupted \
                                     download is the usual cause."
                                .to_string(),
                        };
                        let err = anyhow!(
                            "[STAGE 4 RUNTIME ERROR] {cause} (loading '{}' failed on GPU and CPU \
                             alike; GPU error: {:?}, CPU error: {:?})",
                            clean_path,
                            e,
                            e2
                        );
                        log::error!("{}", err);
                        return Err(err);
                    }
                }
            }
        };

        log::info!("[RUNTIME] Model loaded successfully via {}", actual_backend);

        let backend_desc = actual_backend;

        // Read the chat template out of the GGUF itself. The profile's
        // `chat_template` is only a family-name guess derived from the model id,
        // and it has been wrong in practice (a Gemma model profiled with Llama-3
        // tokens), so the model's own template wins whenever it has one.
        let native_template = match model.chat_template(None) {
            Ok(handle) => {
                let source = model.meta_val_str("tokenizer.chat_template").ok();
                log::info!(
                    "[RUNTIME] ✓ Using chat template embedded in the GGUF (profile guessed '{}', Jinja source {})",
                    config.chat_template,
                    if source.is_some() { "available" } else { "unavailable" }
                );
                let piece = |t| {
                    model
                        .token_to_str(t, llama_cpp_2::model::Special::Tokenize)
                        .unwrap_or_default()
                };
                Some(NativeChatTemplate {
                    source,
                    handle,
                    bos_token: piece(model.token_bos()),
                    eos_token: piece(model.token_eos()),
                })
            }
            Err(e) => {
                log::warn!(
                    "[RUNTIME] GGUF ships no chat template ({e:?}); falling back to the hand-written '{}' template. \
                     If output looks malformed, this is the first thing to suspect.",
                    config.chat_template
                );
                None
            }
        };

        let template_source = if native_template.is_some() {
            "gguf".to_string()
        } else {
            format!("fallback:{}", config.chat_template)
        };

        // Now that llama.cpp has parsed the GGUF, write the real architecture and
        // token metadata back to profile.json. Profiles are generated at download
        // time, before the `.part` file is finalized, so the extractor finds no
        // GGUF to read and the profile keeps whatever defaults it started with.
        // This is the first moment the truth is available.
        let runtime_meta = Self::extract_runtime_metadata(&model, native_template.is_some());
        let effective_stop_tokens = Self::sync_profile_metadata(&clean_path, &runtime_meta)
            .unwrap_or_else(|| config.stop_tokens.clone());

        // Step 4: Store state
        status_cb("Model loaded and ready for inference");
        let fam_str = format!("{:?}", crate::model_intelligence::MetadataExtractor::infer_family_from_string(&config.model_id));
        let info = LoadedModelInfo {
            model_id: config.model_id.clone(),
            model_name: config.model_name.clone(),
            quantization: config.quantization.clone(),
            file_path: config.model_path.clone(),
            context_length: config.context_length,
            // Report what was actually applied, not what was requested.
            gpu_layers: effective_gpu_layers,
            cpu_moe_layers: effective_cpu_moe,
            threads: config.threads,
            backend_used: backend_desc,
            loaded_at: chrono::Utc::now().to_rfc3339(),
            chat_template: config.chat_template.clone(),
            template_source,
            stop_tokens: effective_stop_tokens,
            model_family: fam_str,
            active_adapter: None,
        };

        // Any context from a previously loaded model borrows *that* model and
        // must not outlive it.
        self.end_session();

        self.model = Some(Box::new(model));
        self.native_template = native_template;
        self.loaded_info = Some(info.clone());

        log::info!(
            "[RUNTIME] ✓ Model ready: {} ({}) — context={}, gpu_layers={}, threads={}, backend={}",
            info.model_name, info.quantization, info.context_length,
            info.gpu_layers, info.threads, info.backend_used
        );

        Ok(info)
    }

    /// Reads architecture and token facts out of a model llama.cpp has loaded.
    fn extract_runtime_metadata(
        model: &LlamaModel,
        has_native_chat_template: bool,
    ) -> crate::model_intelligence::profile::RuntimeGgufMetadata {
        let piece = |t| {
            model
                .token_to_str(t, llama_cpp_2::model::Special::Tokenize)
                .ok()
                .filter(|s| !s.is_empty())
        };

        let architecture = model.meta_val_str("general.architecture").ok();

        // A model's end-of-turn token is often distinct from its EOS — Gemma
        // ends turns with `<end_of_turn>` while `<eos>` barely appears in chat
        // data — so record it separately when the GGUF declares one.
        let eot_token = model
            .meta_val_str("tokenizer.ggml.eot_token_id")
            .ok()
            .and_then(|id| id.trim().parse::<i32>().ok())
            .and_then(|id| piece(llama_cpp_2::token::LlamaToken(id)));

        crate::model_intelligence::profile::RuntimeGgufMetadata {
            architecture,
            bos_token: piece(model.token_bos()),
            eos_token: piece(model.token_eos()),
            eot_token,
            context_length: model.n_ctx_train(),
            has_native_chat_template,
        }
    }

    /// Writes runtime-read metadata back to the package's `profile.json`.
    ///
    /// Returns the corrected stop tokens when the profile was updated, so the
    /// current session uses them too rather than waiting for the next load.
    /// Best-effort: a model still runs fine if its profile cannot be written.
    fn sync_profile_metadata(
        gguf_path: &str,
        meta: &crate::model_intelligence::profile::RuntimeGgufMetadata,
    ) -> Option<Vec<String>> {
        // <package_dir>/base/<file>.gguf — the profile lives beside `base/`.
        let package_dir = Path::new(gguf_path).parent()?.parent()?;
        let profile_path = package_dir.join("profile.json");

        let content = std::fs::read_to_string(&profile_path).ok()?;
        let mut profile =
            serde_json::from_str::<crate::model_intelligence::ModelProfile>(&content).ok()?;

        let changed = profile.apply_runtime_metadata(meta);
        let stop_tokens = profile.tokens.stop_tokens.clone();

        if changed {
            log::info!(
                "[RUNTIME] Corrected profile from GGUF metadata: architecture={}, eos={:?}, stop_tokens={:?}",
                profile.architecture, profile.tokens.eos_token, stop_tokens
            );
            if let Err(e) =
                crate::model_intelligence::ModelIntelligenceManager::write_profile(package_dir, &profile)
            {
                log::warn!("[RUNTIME] Could not persist corrected profile: {e:#}");
            }
        }

        Some(stop_tokens)
    }

    /// Unloads the current model, freeing all RAM/VRAM
    pub fn unload_model(&mut self) -> Result<()> {
        self.is_generating.store(false, Ordering::Relaxed);

        if let Some(ref info) = self.loaded_info {
            log::info!("[RUNTIME] Unloading model: {} ({})", info.model_name, info.quantization);
        }

        // Release the context BEFORE the model: it holds a borrow of the model
        // whose lifetime the compiler can no longer check for us.
        self.end_session();

        // Release adapter handles BEFORE dropping the model. Each adapter was
        // initialised against this model; using one after the model is freed
        // would dereference a dangling pointer.
        self.adapter_cache.clear();

        // Drop model to free RAM/VRAM tensors
        self.model = None;
        self.native_template = None;
        self.loaded_info = None;
        // Keep self.backend alive for process lifetime

        log::info!("[RUNTIME] ✓ Model unloaded, RAM/VRAM resources freed");
        Ok(())
    }

    /// Generates tokens for the given messages.
    ///
    /// Calls `token_cb` for each generated token. Returns the total generation.
    /// Respects the `is_generating` flag for cancellation.
    pub fn generate<F>(
        &mut self,
        messages: &[ChatMessage],
        params: &GenerationParams,
        token_cb: F,
    ) -> Result<String>
    where
        F: FnMut(StreamChunk),
    {
        self.generate_with_capability(messages, params, None, token_cb)
    }

    /// Generates tokens with an optional capability backend applied.
    ///
    /// When `capability_backend` is [`CapabilityBackend::LoraAdapter`], the
    /// adapter is bound to the freshly created context *before* the prompt is
    /// decoded, so prefill and generation both run against adapted weights.
    ///
    /// A LoRA binding failure is never fatal: it is logged and generation
    /// proceeds on the base model, matching the graceful-degradation contract
    /// in [`crate::capability`].
    pub fn generate_with_capability<F>(
        &mut self,
        messages: &[ChatMessage],
        params: &GenerationParams,
        capability_backend: Option<&CapabilityBackend>,
        mut token_cb: F,
    ) -> Result<String>
    where
        F: FnMut(StreamChunk),
    {
        // Destructured so `adapter_cache` can be borrowed mutably while `model`
        // is borrowed immutably — these are disjoint fields.
        let Self {
            backend,
            model,
            session,
            loaded_info,
            native_template,
            is_generating,
            adapter_cache,
        } = self;

        let model_box = model
            .as_ref()
            .ok_or_else(|| anyhow!("No model loaded"))?;
        let model: &LlamaModel = model_box;
        let backend = backend
            .as_ref()
            .ok_or_else(|| anyhow!("Backend not initialized"))?;
        let config = loaded_info
            .as_ref()
            .ok_or_else(|| anyhow!("No model info available"))?;

        is_generating.store(true, Ordering::Relaxed);
        let cancel_flag = is_generating.clone();

        // Stage timings for the whole request. Kept because "the model is slow"
        // is not actionable without knowing which stage spent the time — prompt
        // rendering, context allocation, prefill and decode have entirely
        // different fixes.
        let t_start = std::time::Instant::now();
        let mut t_ctx_ready = t_start;
        let mut t_prefill_done = t_start;

        // Render the prompt with the model's own template when it has one, and
        // only fall back to the hand-written approximations otherwise.
        //
        // The fallbacks carry no tools. When the caller supplied some, taking a
        // fallback would answer as though the model had chosen not to call
        // anything, which is a different and much worse thing than saying the
        // model cannot be given tools — see `tools_unsupported`.
        let (prompt, template_source) = match native_template.as_ref() {
            Some(tmpl) => match render_with_native_template(model, tmpl, messages, &params.tools) {
                Ok(p) => (p, "gguf"),
                Err(e) if !params.tools.is_empty() => {
                    is_generating.store(false, Ordering::Relaxed);
                    return Err(tools_unsupported(
                        &config.model_name,
                        params.tools.len(),
                        &format!("its chat template could not be rendered ({e:#})"),
                    ));
                }
                Err(e) => {
                    log::warn!(
                        "[RUNTIME] The GGUF's own chat template failed to render ({e:#}); \
                         falling back to the hand-written '{}' template.",
                        config.chat_template
                    );
                    (
                        format_chat_prompt_with_template(messages, &config.chat_template),
                        "fallback",
                    )
                }
            },
            None if !params.tools.is_empty() => {
                is_generating.store(false, Ordering::Relaxed);
                return Err(tools_unsupported(
                    &config.model_name,
                    params.tools.len(),
                    "it ships no chat template of its own, and the hand-written \
                     fallbacks cannot carry tool definitions",
                ));
            }
            None => (
                format_chat_prompt_with_template(messages, &config.chat_template),
                "fallback",
            ),
        };

        // Rendered, but did the template actually use them? A template that
        // never mentions `tools` renders a perfectly good plain-chat prompt and
        // drops the definitions on the floor, which looks identical from here.
        if !params.tools.is_empty() {
            if let Some(missing) = first_tool_name_missing(&prompt, &params.tools) {
                is_generating.store(false, Ordering::Relaxed);
                return Err(tools_unsupported(
                    &config.model_name,
                    params.tools.len(),
                    &format!(
                        "its chat template renders without them — '{missing}' does not appear \
                         in the prompt, so the model would never learn the tool exists"
                    ),
                ));
            }
        }

        // Compute SHA-256 digest of final prompt sent to llama.cpp
        let mut hasher = Sha256::new();
        hasher.update(prompt.as_bytes());
        let prompt_sha256 = format!("{:x}", hasher.finalize());

        let has_memory_injection = prompt.contains("Known User Information & Preferences")
            || prompt.contains("Recalled Context & Facts")
            || prompt.contains("Shreyash")
            || prompt.contains("User Workspace & Project Context");

        log::info!(
            "\n==================== [LLAMA.CPP RUNTIME PROMPT TRACE] ====================\n\
             - Prompt SHA-256 Hash:   {}\n\
             - Prompt Total Length:   {} chars\n\
             - Memory Injected:       {}\n\
             - Template Format:       {} (rendered via: {})\n\
             - Input Message Count:   {}\n\
             ==========================================================================",
            prompt_sha256, prompt.len(), has_memory_injection, config.chat_template, template_source, messages.len()
        );

        // Decide BOS from what the rendered prompt actually starts with, rather
        // than from the template name.
        //
        // `AddBos::Always` does not blindly prepend a token — it sets llama.cpp's
        // `add_special`, which honours the model's own `add_bos_token` metadata.
        // The previous rule forced `Never` for every "gemma" and "chatml"
        // template, which left Gemma running with no BOS at all; Gemma degrades
        // into multilingual noise without it. The only case that genuinely needs
        // `Never` is a template that already emitted BOS itself (some Jinja
        // templates interpolate `{{ bos_token }}`), which we detect directly.
        let bos_str = model
            .token_to_str(model.token_bos(), llama_cpp_2::model::Special::Tokenize)
            .unwrap_or_default();
        let add_bos = if !bos_str.is_empty() && prompt.starts_with(bos_str.as_str()) {
            AddBos::Never
        } else {
            AddBos::Always
        };

        let prompt_tokens = model
            .str_to_token(&prompt, add_bos)
            .map_err(|e| anyhow!("Tokenization failed: {:?}", e))?;

        let n_prompt_tokens = prompt_tokens.len();
        log::info!("[RUNTIME] Prompt tokenized: {} tokens", n_prompt_tokens);

        if n_prompt_tokens == 0 {
            is_generating.store(false, Ordering::Relaxed);
            return Err(anyhow!("Empty prompt after tokenization"));
        }

        // Create context for this generation
        let ctx_size = std::num::NonZeroU32::new(config.context_length)
            .unwrap_or(std::num::NonZeroU32::new(2048).unwrap());

        // Refused before a context is allocated: a prompt at or over the window
        // is one of the ways llama.cpp aborts the process, and an error the user
        // can read beats the window disappearing.
        if n_prompt_tokens >= ctx_size.get() as usize {
            is_generating.store(false, Ordering::Relaxed);
            return Err(anyhow!(
                "Prompt is {} tokens but the model is loaded with a {}-token context. \
                 Load the model with a larger context, or send a shorter prompt.",
                n_prompt_tokens,
                ctx_size.get()
            ));
        }

        // Batch size is left at whatever this llama.cpp build considers sane and
        // the prefill below is chunked to match, rather than inflated to cover
        // the whole prompt in one decode.
        //
        // Both matter. llama.cpp *aborts the process* — it does not return an
        // error — when a single decode carries more tokens than n_batch, which
        // is how the app vanished (0xc0000409, FAST_FAIL_FATAL_APP_EXIT) the
        // moment a real client connected. Sizing n_batch up to the prompt fixed
        // that but allocated buffers proportional to the largest prompt seen,
        // which on a 32k certified context is memory no short turn uses.
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(ctx_size))
            .with_n_threads(config.threads as i32)
            .with_n_threads_batch(config.threads as i32);

        // What this request needs bound, so a session built under a different
        // adapter is not silently reused.
        let wanted_adapter: Option<(std::path::PathBuf, u32)> = match capability_backend {
            Some(CapabilityBackend::LoraAdapter { path, scale }) => {
                Some((path.clone(), scale.to_bits()))
            }
            _ => None,
        };

        // Reuse the existing context when it is compatible with this request.
        //
        // Only two things make a context unusable: a different context window,
        // and a different LoRA binding. Neither can be changed in place — the
        // window sizes the KV cache, and an adapter is baked into everything
        // already decoded — so either one means starting over.
        let reuse_session = session
            .as_ref()
            .is_some_and(|s| s.n_ctx == ctx_size.get() && s.adapter_key == wanted_adapter);

        if !reuse_session && session.is_some() {
            log::debug!("[RUNTIME] Context not reusable for this request; rebuilding");
            *session = None;
        }

        let mut active_adapter_label: Option<String> = None;

        if session.is_none() {
            let mut ctx = model
                .new_context(backend, ctx_params)
                .map_err(|e| anyhow!("Failed to create inference context: {:?}", e))?;

            // Bound before anything is decoded, so prefill runs against the
            // adapted weights. A fresh context has no stale binding to clear.
            if let Some(CapabilityBackend::LoraAdapter { path, scale }) = capability_backend {
                match adapter_cache.get_or_init(model, path) {
                    Ok(adapter) => match crate::ai_engine::lora_binding::bind_adapter(&mut ctx, adapter, *scale) {
                        Ok(()) => {
                            let label = path
                                .file_name()
                                .map(|n| n.to_string_lossy().to_string())
                                .unwrap_or_else(|| "adapter".to_string());
                            log::info!("[RUNTIME] Generating with LoRA adapter '{}' at scale {:.2}", label, scale);
                            active_adapter_label = Some(label);
                        }
                        Err(e) => log::warn!(
                            "[RUNTIME WARN] LoRA bind failed, continuing on base model: {:#}",
                            e
                        ),
                    },
                    Err(e) => log::warn!(
                        "[RUNTIME WARN] LoRA adapter init failed, continuing on base model: {:#}",
                        e
                    ),
                }
            }

            // SAFETY: `ctx` borrows `model`, which is the `Box<LlamaModel>` held
            // in `self.model`. A box keeps its contents at a fixed heap address,
            // so that borrow stays valid however the runtime itself is moved.
            // The erased lifetime is upheld by `end_session`, which drops this
            // context before the model is ever replaced or freed.
            let ctx: LlamaContext<'static> = unsafe { std::mem::transmute(ctx) };
            *session = Some(GenerationSession {
                ctx,
                tokens: Vec::new(),
                n_ctx: ctx_size.get(),
                adapter_key: wanted_adapter,
            });
        } else if let Some(CapabilityBackend::LoraAdapter { path, .. }) = capability_backend {
            // Same adapter as the live context, so the binding is already in
            // place; only the label for reporting has to be recovered.
            active_adapter_label = path.file_name().map(|n| n.to_string_lossy().to_string());
        }

        let live = session.as_mut().expect("a session was just established");
        t_ctx_ready = std::time::Instant::now();

        // Prefill in chunks the size of this context's own batch.
        //
        // Prefill is the dominant cost of a request — on a CPU-only build a
        // coding agent's system prompt measured ~98s — and decoding it as one
        // call meant a client that had already hung up was discovered only
        // after all of it had been paid for. Chunking gives a cancellation
        // point between batches, so an abandoned request stops within one
        // chunk instead of running to completion.
        let prefill_chunk = (live.ctx.n_batch().max(1)) as usize;
        let mut batch = LlamaBatch::new(prefill_chunk, 1);

        // How much of this prompt the cache already holds.
        //
        // A follow-up turn repeats its whole history — system prompt, every
        // previous exchange, and the assistant's last reply — before adding the
        // new user message. All of that is already decoded, so only the tail is
        // new work.
        let reuse = reusable_prefix(&live.tokens, &prompt_tokens);

        // Drop everything after the divergence point. Positions from `reuse`
        // onward describe a different conversation and would otherwise be
        // attended to as though they belonged to this one.
        if live.tokens.len() > reuse {
            live.ctx
                .clear_kv_cache_seq(Some(0), Some(reuse as u32), None)
                .map_err(|e| anyhow!("Failed to trim the KV cache: {e:?}"))?;
            live.tokens.truncate(reuse);
        }

        // From here the session's token list must track the cache exactly. If a
        // decode fails or is cancelled the session is destroyed rather than left
        // claiming tokens the cache does not hold — reusing a prefix that is not
        // really there would produce confident nonsense instead of an error.
        let prefill_result = (|| -> Result<()> {
            for (chunk_index, chunk) in prompt_tokens[reuse..].chunks(prefill_chunk).enumerate() {
                // The flag is cleared by the canceller, so "not generating" here
                // means someone asked us to stop.
                if !cancel_flag.load(Ordering::Relaxed) {
                    let done = reuse + chunk_index * prefill_chunk;
                    log::info!(
                        "[RUNTIME] Prefill cancelled after {}/{} prompt tokens",
                        done, n_prompt_tokens
                    );
                    return Err(anyhow!("Generation cancelled during prompt prefill"));
                }

                batch.clear();
                let base = reuse + chunk_index * prefill_chunk;
                for (offset, &token) in chunk.iter().enumerate() {
                    let pos = base + offset;
                    // Only the final prompt token needs logits — that is the one
                    // the first sampled token is drawn from.
                    let wants_logits = pos == n_prompt_tokens - 1;
                    batch
                        .add(token, pos as i32, &[0], wants_logits)
                        .map_err(|e| anyhow!("Failed to add token to batch: {:?}", e))?;
                }

                live.ctx
                    .decode(&mut batch)
                    .map_err(|e| anyhow!("Failed to decode prompt batch: {:?}", e))?;

                // Appended only once the cache really holds them.
                live.tokens.extend_from_slice(chunk);
            }
            Ok(())
        })();

        if let Err(e) = prefill_result {
            is_generating.store(false, Ordering::Relaxed);
            *session = None;
            return Err(e);
        }

        let live = session.as_mut().expect("the session survives a successful prefill");

        t_prefill_done = std::time::Instant::now();
        let fresh = n_prompt_tokens - reuse;
        log::info!(
            "[PERF] ctx_alloc={}ms prefill={}ms ({} new, {} reused, {:.1} tok/s)",
            t_ctx_ready.duration_since(t_start).as_millis(),
            t_prefill_done.duration_since(t_ctx_ready).as_millis(),
            fresh,
            reuse,
            fresh as f64
                / t_prefill_done.duration_since(t_ctx_ready).as_secs_f64().max(1e-9)
        );

        let eos_token = model.token_eos();

        // Build effective template-driven stop sequences list.
        //
        // These additions are guesses derived from the family name, so they are
        // only appropriate when the hand-written template is what produced the
        // prompt. When the GGUF's own template is in use, the vocab's
        // end-of-generation tokens are authoritative and guessing here would
        // just reintroduce mismatched tokens.
        let mut effective_stop_tokens = config.stop_tokens.clone();
        let lower_temp = config.chat_template.to_lowercase();

        if template_source != "fallback" {
            // Nothing to add — `is_eog_token` handles termination.
        } else if lower_temp.contains("chatml") || lower_temp.contains("qwen") {
            for st in &["<|im_end|>", "<|im_start|>", "<|endoftext|>"] {
                if !effective_stop_tokens.iter().any(|s| s == st) {
                    effective_stop_tokens.push(st.to_string());
                }
            }
        } else if lower_temp.contains("gemma") {
            for st in &["<end_of_turn>", "<start_of_turn>"] {
                if !effective_stop_tokens.iter().any(|s| s == st) {
                    effective_stop_tokens.push(st.to_string());
                }
            }
        } else if lower_temp.contains("llama3") || lower_temp.contains("llama-3") || lower_temp.contains("llama") {
            for st in &["<|eot_id|>", "<|end_of_text|>", "</s>"] {
                if !effective_stop_tokens.iter().any(|s| s == st) {
                    effective_stop_tokens.push(st.to_string());
                }
            }
        }

        // Log complete runtime generation parameters & provenance immediately before generation
        log::info!(
            "\n==================== [LLAMA.CPP RUNTIME GENERATION CONFIGURATION] ====================\n\
             - Model ID:           {} (Source: Manifest)\n\
             - Chat Template:      {} (Source: GGUF / tokenizer_config.json / Jinja)\n\
             - EOS Token ID:       {:?} (Source: GGUF model.token_eos)\n\
             - Context Length:     {} tokens (Source: GGUF / config.json max_position_embeddings)\n\
             - Threads (batch):    {} (Source: System Hardware Collector)\n\
             - Temperature:        {} (Source: generation_config.json / User Override)\n\
             - Top-P:              {} (Source: generation_config.json / User Override)\n\
             - Top-K:              {} (Source: generation_config.json / User Override)\n\
             - Min-P:              {} (Source: generation_config.json / Model Profile)\n\
             - Repeat Penalty:     {} (Source: generation_config.json / Model Profile)\n\
             - Mirostat Mode:      {} (Source: Model Profile / Fallback)\n\
             - Effective Stop Tokens: {:?} (Source: Metadata + Template Defaults)\n\
             =======================================================================================",
            config.model_id,
            config.chat_template,
            eos_token,
            config.context_length,
            config.threads,
            params.temperature,
            params.top_p,
            params.top_k,
            params.min_p,
            params.repeat_penalty,
            params.mirostat,
            effective_stop_tokens
        );

        // Set up sampler chain directly consumed by llama.cpp.
        //
        // `penalties` must come first so repetition suppression applies to the
        // full distribution before truncation. It was previously absent from the
        // chain entirely, so `repeat_penalty` was logged above but never took
        // effect — capability sampling profiles depend on it being real.
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(1234);

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::penalties(64, params.repeat_penalty, 0.0, 0.0),
            LlamaSampler::temp(params.temperature),
            LlamaSampler::top_k(params.top_k as i32),
            LlamaSampler::top_p(params.top_p, 1),
            LlamaSampler::min_p(params.min_p, 1),
            // Previously a hardcoded seed, which made every regeneration of the
            // same prompt byte-identical regardless of temperature.
            LlamaSampler::dist(seed),
        ]);

        // Autoregressive generation loop
        let mut generated_text = String::new();
        let mut n_generated: u32 = 0;
        let mut n_cur = n_prompt_tokens as i32;

        loop {
            // Check cancellation
            if !cancel_flag.load(Ordering::Relaxed) {
                log::info!("[RUNTIME] Generation cancelled by user after {} tokens", n_generated);
                token_cb(StreamChunk {
                    text: String::new(),
                    is_final: true,
                    tokens_generated: Some(n_generated),
                    finish_reason: Some("cancelled".to_string()),
                    error: None,
                });
                break;
            }

            // Check max tokens
            if n_generated >= params.max_tokens {
                log::info!("[RUNTIME] Max tokens reached: {}", n_generated);
                token_cb(StreamChunk {
                    text: String::new(),
                    is_final: true,
                    tokens_generated: Some(n_generated),
                    finish_reason: Some("length".to_string()),
                    error: None,
                });
                break;
            }

            // Check context window
            if n_cur as u32 >= config.context_length {
                log::warn!("[RUNTIME] Context window exhausted at {} tokens", n_cur);
                token_cb(StreamChunk {
                    text: String::new(),
                    is_final: true,
                    tokens_generated: Some(n_generated),
                    finish_reason: Some("context_length".to_string()),
                    error: None,
                });
                break;
            }

            // Sample next token
            let new_token_id = sampler.sample(&live.ctx, -1);

            // Stop on any end-of-generation token the model itself declares.
            // `token_eos()` is only one of them — Gemma, for instance, ends turns
            // with `<end_of_turn>` rather than `<eos>` — and asking the vocab is
            // reliable where string matching on detokenized text is not.
            if new_token_id == eos_token || model.is_eog_token(new_token_id) {
                log::info!("[RUNTIME] End-of-generation token {} reached after {} tokens", new_token_id, n_generated);
                token_cb(StreamChunk {
                    text: String::new(),
                    is_final: true,
                    tokens_generated: Some(n_generated),
                    finish_reason: Some("stop".to_string()),
                    error: None,
                });
                break;
            }

            // Detokenize
            let token_str = model
                .token_to_str(new_token_id, llama_cpp_2::model::Special::Tokenize)
                .unwrap_or_default();

            // Match only the stop strings this model actually declares.
            //
            // Two previous rules made this fire far too eagerly: any token
            // beginning with `<|` was treated as a stop token whenever *some*
            // configured stop string happened to begin with `<|`, and ChatML's
            // control tokens were hardcoded regardless of the model's family.
            // Between them, a model could be cut off mid-answer by ordinary
            // output. End-of-generation is handled by the vocab check above.
            let combined_check = format!("{}{}", generated_text, token_str);
            let is_stop_str = effective_stop_tokens
                .iter()
                .any(|st| !st.is_empty() && (token_str.contains(st) || combined_check.ends_with(st)));

            if is_stop_str {
                log::info!("[RUNTIME] Stop token sequence '{}' (Token ID: {}) detected after {} tokens", token_str, new_token_id, n_generated);
                token_cb(StreamChunk {
                    text: String::new(),
                    is_final: true,
                    tokens_generated: Some(n_generated),
                    finish_reason: Some("stop".to_string()),
                    error: None,
                });
                break;
            }

            generated_text.push_str(&token_str);
            n_generated += 1;

            // Filter out <think>...</think> reasoning blocks for UI stream
            let mut emit_text = token_str.clone();
            if emit_text.contains("<think>") || emit_text.contains("</think>") || emit_text.contains("<|im_start|>") || emit_text.contains("<|im_end|>") {
                emit_text = emit_text
                    .replace("<think>", "")
                    .replace("</think>", "")
                    .replace("<|im_start|>", "")
                    .replace("<|im_end|>", "");
            }

            if !emit_text.is_empty() {
                // Emit clean token to callback
                token_cb(StreamChunk {
                    text: emit_text,
                    is_final: false,
                    tokens_generated: Some(n_generated),
                    finish_reason: None,
                    error: None,
                });
            }

            // Prepare next batch with the new token
            batch.clear();
            batch
                .add(new_token_id, n_cur, &[0], true)
                .map_err(|e| anyhow!("Failed to add generated token: {:?}", e))?;

            n_cur += 1;

            // Decode
            //
            // A failure here leaves the cache in a state the session's token
            // list can no longer describe, so the session goes rather than being
            // trusted on the next request.
            if let Err(e) = live.ctx.decode(&mut batch) {
                is_generating.store(false, Ordering::Relaxed);
                *session = None;
                return Err(anyhow!("Decode failed at token {}: {:?}", n_generated, e));
            }

            // Recorded after the decode, so the list never claims more than the
            // cache holds. Generated tokens are worth keeping for the same
            // reason prompt tokens are: the next turn replays this reply as part
            // of its history, so caching it extends the reusable prefix past the
            // end of the current prompt.
            live.tokens.push(new_token_id);
        }

        is_generating.store(false, Ordering::Relaxed);
        {
            let decode_s = t_prefill_done.elapsed().as_secs_f64();
            log::info!(
                "[PERF] decode={}ms ({} tok, {:.1} tok/s) total={}ms",
                (decode_s * 1000.0) as u64,
                n_generated,
                n_generated as f64 / decode_s.max(1e-9),
                t_start.elapsed().as_millis()
            );
        }
        log::info!(
            "[RUNTIME] Generation complete: {} tokens, {} chars, adapter={}",
            n_generated,
            generated_text.len(),
            active_adapter_label.as_deref().unwrap_or("none")
        );

        Ok(generated_text)
    }

    /// Signals the generation loop to stop
    pub fn stop_generation(&self) {
        log::info!("[RUNTIME] Stop generation requested");
        self.is_generating.store(false, Ordering::Relaxed);
    }
}

/// Everything needed to frame a prompt the way a given model expects.
///
/// Held together because the two rendering routes need different things: the
/// Jinja source drives the faithful path, and llama.cpp's handle drives the
/// built-in path used when that source will not render.
pub struct NativeChatTemplate {
    /// Raw Jinja from the GGUF's `tokenizer.chat_template` key, when present.
    source: Option<String>,
    /// llama.cpp's own handle on the same template.
    handle: LlamaChatTemplate,
    bos_token: String,
    eos_token: String,
}

/// Renders messages through the chat template baked into the GGUF.
///
/// Two routes, tried in order of faithfulness:
///
/// 1. The template's actual Jinja, rendered with minijinja. This is the same
///    thing HuggingFace evaluates, so it reproduces the exact framing a model
///    was trained on — including custom formats nothing else knows about.
/// 2. llama.cpp's `apply_chat_template`, which recognises a fixed set of
///    well-known formats and rejects anything outside it.
///
/// Falling through both leaves the hand-written family templates, which are
/// approximations and can be badly wrong for an unusual model.
fn render_with_native_template(
    model: &LlamaModel,
    tmpl: &NativeChatTemplate,
    messages: &[ChatMessage],
    tools: &[serde_json::Value],
) -> Result<String> {
    if let Some(src) = &tmpl.source {
        match render_jinja_chat_template(src, messages, &tmpl.bos_token, &tmpl.eos_token, tools) {
            Ok(prompt) if !prompt.trim().is_empty() => return Ok(prompt),
            Ok(_) => log::warn!("[RUNTIME] Jinja chat template rendered an empty prompt; trying llama.cpp's engine"),
            Err(e) => log::warn!("[RUNTIME] Jinja chat template failed to render ({e:#}); trying llama.cpp's engine"),
        }
    }

    if !tools.is_empty() {
        // llama.cpp's own template engine takes messages only, so falling
        // through to it would drop the tools. The caller turns this into a
        // refusal rather than an answer; returning an error here is what makes
        // that possible.
        return Err(anyhow!(
            "llama.cpp's template engine cannot carry the {} tool definition(s) in this request",
            tools.len()
        ));
    }

    let chat: Vec<LlamaChatMessage> = messages
        .iter()
        .map(|m| {
            LlamaChatMessage::new(m.role.clone(), m.content.clone())
                .map_err(|e| anyhow!("Message rejected by chat template: {e:?}"))
        })
        .collect::<Result<_>>()?;

    // `add_ass = true` leaves the assistant tag open so the model completes a
    // reply instead of generating a fresh turn header of its own.
    let prompt = model
        .apply_chat_template(&tmpl.handle, &chat, true)
        .map_err(|e| anyhow!("apply_chat_template failed: {e:?}"))?;

    if prompt.trim().is_empty() {
        return Err(anyhow!("chat template rendered an empty prompt"));
    }

    Ok(prompt)
}

/// The error a caller gets when tools were asked for and cannot be delivered.
///
/// Deliberately not a silent degradation. A client that connected five MCP
/// servers and gets prose back has no way to tell that from a model deciding
/// none of its tools applied; this says which model, how many tools, and why.
fn tools_unsupported(model_name: &str, tool_count: usize, why: &str) -> anyhow::Error {
    anyhow!(
        "This model cannot be given tools: {model_name} was sent {tool_count} tool \
         definition(s) but {why}. Sarathi will not answer as though no tools were \
         offered — load a model whose chat template supports tool calling, or have the \
         client send `tool_choice: \"none\"` to ask for prose deliberately."
    )
}

/// The first tool whose name does not appear in the rendered prompt.
///
/// A cheap, template-agnostic check that the definitions actually landed. Names
/// are what a model needs in order to emit a call at all, so a name absent from
/// the prompt means the tool was not passed on, whatever the template did.
fn first_tool_name_missing(prompt: &str, tools: &[serde_json::Value]) -> Option<String> {
    for tool in tools {
        let name = tool
            .get("function")
            .and_then(|f| f.get("name"))
            .or_else(|| tool.get("name"))
            .and_then(|n| n.as_str())
            .unwrap_or_default();
        if !name.is_empty() && !prompt.contains(name) {
            return Some(name.to_string());
        }
    }
    None
}

/// Rewrites template constructs minijinja has no parser for.
///
/// `{% generation %}` / `{% endgeneration %}` come from HuggingFace's
/// `return_assistant_tokens_mask`: they mark which span of the rendered text is
/// the assistant's, so a trainer can mask it. They contribute nothing to the
/// output, and minijinja — which implements Jinja2 proper, not transformers'
/// extensions — rejects the whole template on sight of one. LFM2.5 uses them,
/// which is why a model whose template *does* support tools rendered as though
/// it did not.
///
/// The replacement keeps each tag's whitespace-control markers, because
/// `{%- x -%}` trims either side and deleting the tag outright would leave that
/// whitespace in the prompt.
fn strip_generation_tags(src: &str) -> std::borrow::Cow<'_, str> {
    if !src.contains("generation") {
        return std::borrow::Cow::Borrowed(src);
    }

    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    let mut rewrote = false;

    while let Some(open) = rest.find("{%") {
        let Some(close_rel) = rest[open..].find("%}") else { break };
        let close = open + close_rel + 2;
        let tag = &rest[open..close];

        // `{%- generation -%}` and `{% endgeneration %}`, and every spacing in
        // between. The body is exactly one word, so anything else — a variable
        // called `generation_prompt`, say — is left alone.
        let inner = tag[2..tag.len() - 2].trim_end_matches('-').trim_start_matches('-').trim();
        let is_generation = inner == "generation" || inner == "endgeneration";

        out.push_str(&rest[..open]);
        if is_generation {
            // A `set` of an unused name is a real statement that emits nothing,
            // so the whitespace-control markers keep behaving as they did.
            let left = if tag.starts_with("{%-") { "-" } else { "" };
            let right = if tag.ends_with("-%}") { "-" } else { "" };
            let value = if inner == "endgeneration" { "false" } else { "true" };
            out.push_str(&format!("{{%{left} set __sarathi_generation_span = {value} {right}%}}"));
            rewrote = true;
        } else {
            out.push_str(tag);
        }
        rest = &rest[close..];
    }

    if !rewrote {
        return std::borrow::Cow::Borrowed(src);
    }
    out.push_str(rest);
    std::borrow::Cow::Owned(out)
}

/// Python string and mapping methods that chat templates take for granted.
///
/// HuggingFace templates are written to run under Jinja2 *with Python objects
/// underneath*, so they call `content.endswith(…)`, `s.split(…)`,
/// `d.items()` — methods of the underlying type, not Jinja filters. minijinja
/// implements Jinja faithfully and therefore has none of them, and one call to
/// a missing method aborts the whole render.
///
/// Before this, that abort was indistinguishable from "this model does not
/// support tools": LFM2.5's template calls `endswith` while rendering a
/// previous assistant turn, so a tool conversation rendered fine on turn one
/// and failed on turn two.
///
/// Only methods with unambiguous semantics are implemented. Anything else still
/// raises, so a template using something genuinely unsupported fails loudly
/// rather than rendering something subtly wrong.
fn python_method(
    _state: &minijinja::State<'_, '_>,
    value: &minijinja::value::Value,
    method: &str,
    args: &[minijinja::value::Value],
) -> Result<minijinja::value::Value, minijinja::Error> {
    use minijinja::value::Value;
    use minijinja::{Error, ErrorKind};

    let unsupported = || {
        Error::new(
            ErrorKind::UnknownMethod,
            format!("no Python-compatible method named {method}"),
        )
    };
    let arg_str = |i: usize| -> Option<String> { args.get(i)?.as_str().map(str::to_string) };

    // ── mappings ────────────────────────────────────────────────────────────
    if let Ok(iter) = value.try_iter() {
        if value.kind() == minijinja::value::ValueKind::Map {
            match method {
                "items" => {
                    let pairs: Vec<Value> = iter
                        .map(|k| {
                            let v = value.get_item(&k).unwrap_or(Value::UNDEFINED);
                            Value::from(vec![k, v])
                        })
                        .collect();
                    return Ok(Value::from(pairs));
                }
                "keys" => return Ok(Value::from(iter.collect::<Vec<_>>())),
                "values" => {
                    let vals: Vec<Value> = iter
                        .map(|k| value.get_item(&k).unwrap_or(Value::UNDEFINED))
                        .collect();
                    return Ok(Value::from(vals));
                }
                "get" => {
                    let default = args.get(1).cloned().unwrap_or(Value::from(()));
                    let found = args
                        .first()
                        .and_then(|k| value.get_item(k).ok())
                        .filter(|v| !v.is_undefined());
                    return Ok(found.unwrap_or(default));
                }
                _ => {}
            }
        }
    }

    // ── strings ─────────────────────────────────────────────────────────────
    let Some(s) = value.as_str() else { return Err(unsupported()) };

    let out = match method {
        "startswith" => Value::from(s.starts_with(arg_str(0).ok_or_else(unsupported)?.as_str())),
        "endswith" => Value::from(s.ends_with(arg_str(0).ok_or_else(unsupported)?.as_str())),
        "strip" => Value::from(match arg_str(0) {
            Some(chars) => s.trim_matches(|c| chars.contains(c)).to_string(),
            None => s.trim().to_string(),
        }),
        "lstrip" => Value::from(match arg_str(0) {
            Some(chars) => s.trim_start_matches(|c| chars.contains(c)).to_string(),
            None => s.trim_start().to_string(),
        }),
        "rstrip" => Value::from(match arg_str(0) {
            Some(chars) => s.trim_end_matches(|c| chars.contains(c)).to_string(),
            None => s.trim_end().to_string(),
        }),
        "lower" => Value::from(s.to_lowercase()),
        "upper" => Value::from(s.to_uppercase()),
        "title" => Value::from(
            s.split(' ')
                .map(|w| {
                    let mut c = w.chars();
                    match c.next() {
                        Some(f) => f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase(),
                        None => String::new(),
                    }
                })
                .collect::<Vec<_>>()
                .join(" "),
        ),
        "capitalize" => {
            let mut c = s.chars();
            Value::from(match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase(),
                None => String::new(),
            })
        }
        "replace" => Value::from(s.replace(
            arg_str(0).ok_or_else(unsupported)?.as_str(),
            arg_str(1).ok_or_else(unsupported)?.as_str(),
        )),
        "split" => {
            let parts: Vec<Value> = match arg_str(0) {
                Some(sep) => s.split(sep.as_str()).map(Value::from).collect(),
                None => s.split_whitespace().map(Value::from).collect(),
            };
            Value::from(parts)
        }
        "rsplit" => {
            let sep = arg_str(0).ok_or_else(unsupported)?;
            let mut parts: Vec<Value> = s.rsplit(sep.as_str()).map(Value::from).collect();
            parts.reverse();
            Value::from(parts)
        }
        "count" => Value::from(s.matches(arg_str(0).ok_or_else(unsupported)?.as_str()).count()),
        "find" => Value::from(
            s.find(arg_str(0).ok_or_else(unsupported)?.as_str())
                .map_or(-1i64, |i| i as i64),
        ),
        _ => return Err(unsupported()),
    };
    Ok(out)
}

/// Puts tool calls into the shape chat templates are written against.
///
/// The OpenAI wire format carries `function.arguments` as a **string** of JSON.
/// HuggingFace templates are written against `tokenizer.apply_chat_template`,
/// where it is a **mapping** — LFM2's iterates `func_args.items()` to rebuild
/// its `name(k=v)` syntax, and a string has no `.items()`, so the whole render
/// aborts. Decoding it here is the difference between a tool conversation that
/// continues past its first turn and one that dies on the second.
fn normalise_tool_calls(calls: &[serde_json::Value]) -> Vec<serde_json::Value> {
    calls
        .iter()
        .map(|call| {
            let mut call = call.clone();
            let Some(args) = call.pointer("/function/arguments") else { return call };
            let serde_json::Value::String(text) = args else { return call };

            // Only when it really is JSON. An argument string that is not an
            // object is left exactly as the client sent it.
            if let Ok(decoded) = serde_json::from_str::<serde_json::Value>(text) {
                if decoded.is_object() {
                    if let Some(slot) = call.pointer_mut("/function/arguments") {
                        *slot = decoded;
                    }
                }
            }
            call
        })
        .collect()
}

/// A chat message as a chat template expects to see it.
///
/// Templates are written against Python dicts, so they reach for `.get(...)`
/// on optional fields like `reasoning` and `tool_calls`. A plain map value has
/// no such method and rendering aborts, so this supplies dict semantics:
/// subscripting known keys, and `.get` returning a default instead of raising.
///
/// `tool_calls` and `tool_call_id` are carried structurally rather than being
/// flattened into the text, because that is the half of the round trip the
/// model has to *read*: a template renders a previous assistant turn's calls in
/// the model's own syntax, and a tool result has to be tied back to the call it
/// answers. Without them the second turn of every tool conversation is
/// malformed, which looks like the model ignoring its own tool results.
#[derive(Debug)]
struct TemplateMessage {
    role: String,
    content: String,
    tool_calls: Vec<serde_json::Value>,
    tool_call_id: Option<String>,
    name: Option<String>,
}

impl minijinja::value::Object for TemplateMessage {
    fn get_value(self: &std::sync::Arc<Self>, key: &minijinja::value::Value) -> Option<minijinja::value::Value> {
        match key.as_str()? {
            "role" => Some(minijinja::value::Value::from(self.role.clone())),
            "content" => Some(minijinja::value::Value::from(self.content.clone())),
            // Absent rather than empty when there are none: templates test
            // `if message.tool_calls`, and an empty list is falsy either way,
            // but `is defined` must be false for a plain turn.
            "tool_calls" if !self.tool_calls.is_empty() => Some(
                minijinja::value::Value::from_serialize(&normalise_tool_calls(&self.tool_calls)),
            ),
            "tool_call_id" => self
                .tool_call_id
                .as_ref()
                .map(|id| minijinja::value::Value::from(id.clone())),
            "name" => self.name.as_ref().map(|n| minijinja::value::Value::from(n.clone())),
            _ => None,
        }
    }

    fn call_method(
        self: &std::sync::Arc<Self>,
        _state: &minijinja::State<'_, '_>,
        method: &str,
        args: &[minijinja::value::Value],
    ) -> std::result::Result<minijinja::value::Value, minijinja::Error> {
        match method {
            // Python's `dict.get(key[, default])`: absent keys yield the
            // default (None when unspecified) rather than erroring.
            "get" => {
                let none = minijinja::value::Value::from(());
                let default = args.get(1).cloned().unwrap_or(none.clone());
                Ok(args
                    .first()
                    .and_then(|k| self.get_value(k))
                    .unwrap_or(default))
            }
            _ => Err(minijinja::Error::new(
                minijinja::ErrorKind::UnknownMethod,
                format!("chat message has no method named {method}"),
            )),
        }
    }
}

/// Renders a HuggingFace-style Jinja chat template.
///
/// The context mirrors what `tokenizer.apply_chat_template` provides, since
/// templates are written against it: a `messages` list of role/content maps,
/// the special tokens as variables, and `add_generation_prompt` set so the
/// template leaves the assistant turn open for the model to complete.
pub fn render_jinja_chat_template(
    template_src: &str,
    messages: &[ChatMessage],
    bos_token: &str,
    eos_token: &str,
    tools: &[serde_json::Value],
) -> Result<String> {
    use minijinja::{context, Environment};

    let mut env = Environment::new();
    // Chat templates are whitespace-sensitive; keeping blocks and newlines
    // verbatim is what makes the output match the training format.
    env.set_keep_trailing_newline(true);
    // Chat templates are written against Python objects; see `python_method`.
    env.set_unknown_method_callback(python_method);
    let prepared = strip_generation_tags(template_src);
    env.add_template("chat", &prepared)
        .map_err(|e| anyhow!("chat template is not valid Jinja: {e}"))?;

    let tmpl = env
        .get_template("chat")
        .map_err(|e| anyhow!("chat template unavailable: {e}"))?;

    let msgs: Vec<_> = messages
        .iter()
        .map(|m| {
            minijinja::value::Value::from_object(TemplateMessage {
                role: m.role.clone(),
                content: m.content.clone(),
                tool_calls: m.tool_calls.clone(),
                tool_call_id: m.tool_call_id.clone(),
                name: m.name.clone(),
            })
        })
        .collect();

    // Tool definitions go in as plain data, which is the shape templates expect:
    // they subscript `tool.function.name` and `.parameters` directly. An empty
    // list still has to be *declared* so templates that branch on it take the
    // plain-chat path rather than failing on an undefined name.
    let tool_values: Vec<minijinja::value::Value> = tools
        .iter()
        .map(minijinja::value::Value::from_serialize)
        .collect();

    tmpl.render(context! {
        messages => msgs,
        bos_token => bos_token,
        eos_token => eos_token,
        add_generation_prompt => true,
        tools => tool_values,
        enable_thinking => false,
    })
    .map_err(|e| anyhow!("chat template failed to render: {e}"))
}

/// Formats chat messages into a prompt string based on Model Profile template.
///
/// Fallback only — used when a GGUF ships no chat template of its own. These are
/// approximations of each family's real format, so prefer the model's embedded
/// template (see [`render_with_native_template`]) wherever one exists.
pub fn format_chat_prompt_with_template(messages: &[ChatMessage], template_name: &str) -> String {
    let mut prompt = String::new();
    let lower_temp = template_name.to_lowercase();

    if lower_temp.contains("chatml") || lower_temp.contains("qwen") {
        for msg in messages {
            prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content));
        }
        prompt.push_str("<|im_start|>assistant\n");
    } else if lower_temp.contains("llama3") || lower_temp.contains("llama-3") || lower_temp.contains("llama_3") {
        // The role must sit between the header tags with no newline of its own;
        // a stray one here does not match how Llama-3 was trained.
        for msg in messages {
            prompt.push_str(&format!("<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>", msg.role, msg.content));
        }
        prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    } else if lower_temp.contains("gemma") {
        for msg in messages {
            let role = if msg.role == "assistant" { "model" } else { &msg.role };
            prompt.push_str(&format!("<start_of_turn>{}\n{}<end_of_turn>\n", role, msg.content));
        }
        prompt.push_str("<start_of_turn>model\n");
    } else if lower_temp.contains("mistral") {
        for msg in messages {
            if msg.role == "user" {
                prompt.push_str(&format!("[INST] {} [/INST]", msg.content));
            } else if msg.role == "assistant" {
                prompt.push_str(&format!(" {}\n", msg.content));
            }
        }
    } else {
        // Fallback to ChatML structure for unknown models
        for msg in messages {
            prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content));
        }
        prompt.push_str("<|im_start|>assistant\n");
    }

    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── KV cache prefix reuse ──────────────────────────────────────────────

    fn toks(ids: &[i32]) -> Vec<LlamaToken> {
        ids.iter().copied().map(LlamaToken).collect()
    }

    /// The ordinary case: a follow-up turn repeats everything said so far and
    /// appends to it, so only the appended part is new work.
    #[test]
    fn a_continued_conversation_reuses_everything_before_the_new_turn() {
        let cached = toks(&[1, 2, 3, 4]);
        let prompt = toks(&[1, 2, 3, 4, 5, 6]);

        assert_eq!(reusable_prefix(&cached, &prompt), 4);
    }

    /// At least one prompt token must always be decoded: the first sampled
    /// token is drawn from the logits of the last decoded position, so reusing
    /// the whole prompt would leave the sampler nothing to read.
    #[test]
    fn an_identical_prompt_still_decodes_its_final_token() {
        let same = toks(&[1, 2, 3, 4]);

        assert_eq!(reusable_prefix(&same, &same), 3, "one short of the full prompt");
    }

    /// Regenerating the same turn is the case that would be tempting to reuse
    /// entirely, and must not be.
    #[test]
    fn a_cache_longer_than_the_prompt_is_capped_by_the_prompt() {
        let cached = toks(&[1, 2, 3, 4, 5, 6, 7]);
        let prompt = toks(&[1, 2, 3]);

        assert_eq!(reusable_prefix(&cached, &prompt), 2);
    }

    /// Editing an earlier message, or switching conversation, diverges partway
    /// through. Everything from the divergence on has to be decoded again.
    #[test]
    fn a_divergent_conversation_reuses_only_the_shared_head() {
        let cached = toks(&[1, 2, 3, 99, 99]);
        let prompt = toks(&[1, 2, 3, 42, 42]);

        assert_eq!(reusable_prefix(&cached, &prompt), 3);
    }

    #[test]
    fn a_prompt_sharing_nothing_reuses_nothing() {
        assert_eq!(reusable_prefix(&toks(&[9, 9, 9]), &toks(&[1, 2, 3])), 0);
    }

    /// A fresh context has an empty cache, which is simply "reuse nothing"
    /// rather than a special case.
    #[test]
    fn an_empty_cache_reuses_nothing() {
        assert_eq!(reusable_prefix(&[], &toks(&[1, 2, 3])), 0);
    }

    /// A single-token prompt has no reusable part at all, and the cap must not
    /// underflow computing that.
    #[test]
    fn a_one_token_prompt_is_all_new_work() {
        assert_eq!(reusable_prefix(&toks(&[1]), &toks(&[1])), 0);
        assert_eq!(reusable_prefix(&[], &[]), 0);
    }

    // ─── MoE expert offload patterns ────────────────────────────────────────

    /// Matches a tensor name against a generated pattern the way llama.cpp does.
    fn pattern_matches(pattern: &CString, tensor: &str) -> bool {
        // The patterns are fixed-shape: `blk\.<i>\.ffn_(a|b|…)_(ch|)exps`.
        // Rather than pull in a regex engine for a test, expand the two
        // alternations and check the literal forms.
        let raw = pattern.to_str().expect("generated pattern is valid UTF-8");
        let prefix = raw.trim_end_matches(r"\.ffn_(up|down|gate|gate_up)_(ch|)exps");
        let block = prefix.replace(r"\.", ".");

        ["up", "down", "gate", "gate_up"].iter().any(|proj| {
            ["ch", ""]
                .iter()
                .any(|infix| tensor.contains(&format!("{block}.ffn_{proj}_{infix}exps")))
        })
    }

    /// The regression this design exists to prevent.
    ///
    /// The crate's own `add_cpu_moe_override()` omits the `gate_up` alternative
    /// that upstream carries, so a model with fused expert tensors would load
    /// with the offload silently doing nothing. If anyone "simplifies" the
    /// pattern generation back to the crate helper, this must fail loudly.
    #[test]
    fn generated_patterns_match_fused_gate_up_expert_tensors() {
        let patterns = cpu_moe_override_patterns(1);
        assert_eq!(patterns.len(), 1);

        assert!(
            pattern_matches(&patterns[0], "blk.0.ffn_gate_up_exps.weight"),
            "fused expert tensors must match, or gpt-oss offloads nothing: {:?}",
            patterns[0]
        );
    }

    #[test]
    fn generated_patterns_match_split_expert_tensors() {
        let patterns = cpu_moe_override_patterns(1);

        for tensor in [
            "blk.0.ffn_up_exps.weight",
            "blk.0.ffn_down_exps.weight",
            "blk.0.ffn_gate_exps.weight",
            "blk.0.ffn_down_chexps.weight",
        ] {
            assert!(pattern_matches(&patterns[0], tensor), "{tensor} must match");
        }
    }

    /// `--n-cpu-moe N` covers the *first* N layers, one override each.
    #[test]
    fn one_pattern_is_generated_per_offloaded_layer() {
        let patterns = cpu_moe_override_patterns(22);
        assert_eq!(patterns.len(), 22);

        assert!(patterns[0].to_str().unwrap().starts_with(r"blk\.0\."));
        assert!(patterns[21].to_str().unwrap().starts_with(r"blk\.21\."));

        // Layer 22 is beyond the requested depth and must stay resident.
        assert!(!pattern_matches(&patterns[21], "blk.22.ffn_gate_up_exps.weight"));
    }

    #[test]
    fn a_depth_of_zero_generates_no_overrides() {
        assert!(cpu_moe_override_patterns(0).is_empty());
    }

    #[test]
    fn patterns_target_only_their_own_layer() {
        let patterns = cpu_moe_override_patterns(3);

        assert!(pattern_matches(&patterns[1], "blk.1.ffn_up_exps.weight"));
        assert!(!pattern_matches(&patterns[1], "blk.0.ffn_up_exps.weight"));
    }

    /// Attention and the KV cache are what expert offload exists to keep on the
    /// GPU, so they must never be caught by these patterns.
    #[test]
    fn patterns_never_match_attention_or_shared_tensors() {
        let patterns = cpu_moe_override_patterns(4);

        for tensor in [
            "blk.0.attn_q.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_gate_inp.weight",
            "blk.0.ffn_up_shexp.weight",
            "token_embd.weight",
            "output_norm.weight",
        ] {
            for pattern in &patterns {
                assert!(
                    !pattern_matches(pattern, tensor),
                    "{tensor} must stay resident but matched {pattern:?}"
                );
            }
        }
    }

    #[test]
    fn test_runtime_initial_status() {
        let runtime = LlamaCppRuntime::new();
        assert_eq!(runtime.status(), RuntimeStatus::NotLoaded);
        assert!(runtime.loaded_model_info().is_none());
    }

    fn sample_messages() -> Vec<ChatMessage> {
        vec![
            ChatMessage::new("system", "You are helpful."),
            ChatMessage::new("user", "Hello"),
        ]
    }

    #[test]
    fn chatml_template_uses_im_start_markers() {
        let prompt = format_chat_prompt_with_template(&sample_messages(), "chatml");

        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("You are helpful."));
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.contains("Hello"));
        // Must end primed for the assistant turn, or the model continues the
        // user's message instead of replying.
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn llama3_template_uses_header_markers() {
        let prompt = format_chat_prompt_with_template(&sample_messages(), "llama3");

        assert!(prompt.contains("<|start_header_id|>"));
        assert!(prompt.contains("<|eot_id|>"));
        assert!(prompt.contains("You are helpful."));
        assert!(prompt.contains("Hello"));
    }

    #[test]
    fn llama3_role_sits_inside_the_header_tags() {
        let prompt = format_chat_prompt_with_template(&sample_messages(), "llama3");

        // A newline between the role and `<|end_header_id|>` is not the format
        // Llama-3 was trained on, and the closing tag never had one — so the
        // opening tags disagreed with the assistant tag the prompt ends with.
        assert!(prompt.contains("<|start_header_id|>system<|end_header_id|>\n\n"));
        assert!(prompt.contains("<|start_header_id|>user<|end_header_id|>\n\n"));
        assert!(!prompt.contains("<|start_header_id|>system\n"));
        assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn gemma_template_maps_assistant_onto_model_turns() {
        let prompt = format_chat_prompt_with_template(&sample_messages(), "gemma");

        // Gemma names the assistant role "model"; emitting "assistant" would put
        // a token the model never saw in training at the start of every reply.
        assert!(prompt.contains("<start_of_turn>user\n"));
        assert!(prompt.ends_with("<start_of_turn>model\n"));
        assert!(!prompt.contains("<start_of_turn>assistant"));
    }

    #[test]
    fn jinja_template_renders_roles_and_generation_prompt() {
        let src = "{{- bos_token -}}\
                   {%- for m in messages -%}<|turn>{{ m['role'] }}\n{{ m['content'] }}<turn|>\n{%- endfor -%}\
                   {%- if add_generation_prompt -%}<|turn>model\n{%- endif -%}";

        let out = render_jinja_chat_template(src, &sample_messages(), "<bos>", "<eos>", &[]).unwrap();

        // `{%-` trims the preceding newline, so the turns run together — which
        // is exactly the whitespace handling a chat template depends on.
        assert_eq!(
            out,
            "<bos><|turn>system\nYou are helpful.<turn|><|turn>user\nHello<turn|><|turn>model"
        );
    }

    #[test]
    fn jinja_messages_support_python_dict_get() {
        // Real templates probe optional fields with `.get(...)`. Without dict
        // semantics minijinja aborts with "map has no method named get", and
        // the whole template silently falls back to a guessed format.
        let src = "{%- for m in messages -%}\
                   {{ m.get('role') }}:{{ m.get('tool_calls', 'none') }};\
                   {%- endfor -%}";

        let out = render_jinja_chat_template(src, &sample_messages(), "<bos>", "<eos>", &[]).unwrap();

        assert_eq!(out, "system:none;user:none;");
    }

    #[test]
    fn tool_definitions_reach_the_template() {
        // The gap this closes: `tools` was hardcoded to an empty list, so a
        // tool-capable template took its plain-chat branch every time and the
        // model was never told any tool existed.
        let src = "{%- if tools -%}\
                   TOOLS:{% for t in tools %}{{ t.function.name }},{% endfor %}\
                   {%- else -%}NO_TOOLS{%- endif -%}";

        let tools = vec![
            serde_json::json!({"type":"function","function":{"name":"searxng_web_search"}}),
            serde_json::json!({"type":"function","function":{"name":"research_ask"}}),
        ];

        let with = render_jinja_chat_template(src, &sample_messages(), "<b>", "<e>", &tools).unwrap();
        assert_eq!(with, "TOOLS:searxng_web_search,research_ask,");

        let without = render_jinja_chat_template(src, &sample_messages(), "<b>", "<e>", &[]).unwrap();
        assert_eq!(without, "NO_TOOLS", "an empty list must still take the plain branch");
    }

    #[test]
    fn a_tool_parameter_schema_survives_into_the_template() {
        // Templates render the schema verbatim into the prompt; a value that
        // arrives stringified or flattened gives the model an unusable spec.
        let src = "{{ tools[0].function.parameters.properties.query.type }}";
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "s",
                "parameters": {"type":"object","properties":{"query":{"type":"string"}}}
            }
        })];

        let out = render_jinja_chat_template(src, &sample_messages(), "<b>", "<e>", &tools).unwrap();
        assert_eq!(out, "string");
    }

    #[test]
    fn a_broken_jinja_template_errors_rather_than_panicking() {
        // Callers fall back to a hand-written template on error, so this must
        // surface as Err rather than taking the process down.
        let out = render_jinja_chat_template("{% for %}", &sample_messages(), "<bos>", "<eos>", &[]);
        assert!(out.is_err());
    }

    #[test]
    fn qwen_is_treated_as_chatml() {
        let qwen = format_chat_prompt_with_template(&sample_messages(), "qwen");
        let chatml = format_chat_prompt_with_template(&sample_messages(), "chatml");
        assert_eq!(qwen, chatml);
    }

    #[test]
    fn an_unknown_template_still_produces_a_usable_prompt() {
        // Must never return empty: an empty prompt tokenizes to nothing and the
        // runtime rejects the request outright.
        let prompt = format_chat_prompt_with_template(&sample_messages(), "not-a-real-template");

        assert!(!prompt.trim().is_empty());
        assert!(prompt.contains("Hello"));
    }

    #[test]
    fn test_generate_fails_without_model() {
        let mut runtime = LlamaCppRuntime::new();
        let messages = vec![ChatMessage::new("user", "test")];
        let result = runtime.generate(&messages, &GenerationParams::default(), |_| {});
        assert!(result.is_err());
    }

    #[test]
    fn test_unload_without_model() {
        let mut runtime = LlamaCppRuntime::new();
        // Should succeed even with no model loaded
        assert!(runtime.unload_model().is_ok());
    }

    static TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Loads a real installed model and reports what the fix actually reads
    /// from it: the embedded chat template, the true BOS/EOS tokens, and the
    /// prompt that now gets sent.
    ///
    /// Ignored by default because it loads several gigabytes of weights.
    /// Run with: `cargo test --lib gguf_metadata_matches_the_model -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn gguf_metadata_matches_the_model() {
        let _guard = TEST_MUTEX.lock().unwrap();
        let gguf_path = r"C:\Users\lenovo\AppData\Roaming\com.sarathi.app\models\huggingface\yuxinlu1_gemma-4-12B-coder-fable5-composer2.5-v1-GGUF\base\gemma4-coding-Q2_K.gguf";
        if !std::path::Path::new(gguf_path).exists() {
            println!("Skipping: {gguf_path} not present");
            return;
        }

        let backend = LlamaBackend::init().expect("backend init");
        let params = LlamaModelParams::default().with_n_gpu_layers(0);
        let model = LlamaModel::load_from_file(&backend, gguf_path, &params).expect("model load");

        let meta = LlamaCppRuntime::extract_runtime_metadata(&model, model.chat_template(None).is_ok());
        println!("architecture      : {:?}", meta.architecture);
        println!("bos_token         : {:?}", meta.bos_token);
        println!("eos_token         : {:?}", meta.eos_token);
        println!("eot_token         : {:?}", meta.eot_token);
        println!("n_ctx_train       : {}", meta.context_length);
        println!("native template   : {}", meta.has_native_chat_template);

        // The profile on disk claimed architecture "llama" with `<|eot_id|>`;
        // the model itself must say otherwise.
        assert!(meta.has_native_chat_template, "GGUF should carry its own chat template");
        assert_ne!(meta.eos_token.as_deref(), Some("<|eot_id|>"), "EOS must not be Llama-3's");

        println!(
            "--- embedded template source ---\n{}\n--------------------------------",
            model.meta_val_str("tokenizer.chat_template").unwrap_or_else(|e| format!("<unreadable: {e:?}>"))
        );

        let piece = |t| model.token_to_str(t, llama_cpp_2::model::Special::Tokenize).unwrap_or_default();
        let tmpl = NativeChatTemplate {
            source: model.meta_val_str("tokenizer.chat_template").ok(),
            handle: model.chat_template(None).unwrap(),
            bos_token: piece(model.token_bos()),
            eos_token: piece(model.token_eos()),
        };

        let rendered = render_with_native_template(&model, &tmpl, &sample_messages(), &[])
            .expect("the model's own template must render");
        println!("--- rendered prompt ---\n{rendered}\n-----------------------");

        assert!(rendered.contains("Hello"));
        // This model frames turns as `<|turn>role … <turn|>`. The hand-written
        // "gemma" fallback emits `<start_of_turn>`, which this model was never
        // trained on — that mismatch is what produced the garbled output.
        assert!(rendered.contains("<|turn>user"), "expected the model's own turn markers");
        assert!(!rendered.contains("<start_of_turn>"), "must not use the guessed Gemma framing");
    }

    #[test]
    fn test_load_real_installed_gguf() {
        let _guard = TEST_MUTEX.lock().unwrap();
        let gguf_path = r"C:\Users\lenovo\AppData\Roaming\com.sarathi.app\models\huggingface\meta-Llama_Llama-3.2-1B\base\Llama-3.2-1B-Instruct-Q8_0.gguf";
        if !std::path::Path::new(gguf_path).exists() {
            println!("Skipping real GGUF test, file does not exist at {}", gguf_path);
            return;
        }

        let mut runtime = LlamaCppRuntime::new();
        let config = ModelLoadConfig {
            model_path: gguf_path.to_string(),
            model_id: "meta-llama/Llama-3.2-1B".to_string(),
            model_name: "Llama 3.2 1B".to_string(),
            quantization: "Q8_0".to_string(),
            context_length: 65536,
            gpu_layers: 999,
            cpu_moe_layers: 0,
            threads: 4,
            chat_template: "llama3".to_string(),
            stop_tokens: vec![],
        };

        println!("Testing load_model for real GGUF at {}", gguf_path);
        let res = runtime.load_model(&config, |step| {
            println!("Step: {}", step);
        });

        match &res {
            Ok(info) => println!("SUCCESSFULLY LOADED MODEL: {:?}", info),
            Err(e) => println!("FAILED TO LOAD MODEL: {:?}", e),
        }

        assert!(res.is_ok(), "Real GGUF load failed: {:?}", res.err());
    }

    #[test]
    fn test_load_real_installed_gguf_cpu() {
        let _guard = TEST_MUTEX.lock().unwrap();
        let gguf_path = r"C:\Users\lenovo\AppData\Roaming\com.sarathi.app\models\huggingface\meta-Llama_Llama-3.2-1B\base\Llama-3.2-1B-Instruct-Q8_0.gguf";
        if !std::path::Path::new(gguf_path).exists() {
            println!("Skipping real GGUF CPU test, file does not exist at {}", gguf_path);
            return;
        }

        let mut runtime = LlamaCppRuntime::new();
        let config = ModelLoadConfig {
            model_path: gguf_path.to_string(),
            model_id: "meta-llama/Llama-3.2-1B".to_string(),
            model_name: "Llama 3.2 1B".to_string(),
            quantization: "Q8_0".to_string(),
            context_length: 4096,
            gpu_layers: 0,
            cpu_moe_layers: 0,
            threads: 4,
            chat_template: "llama3".to_string(),
            stop_tokens: vec![],
        };

        println!("Testing load_model (CPU mode, gpu_layers=0) for GGUF at {}", gguf_path);
        let res = runtime.load_model(&config, |step| {
            println!("Step: {}", step);
        });

        match &res {
            Ok(info) => println!("SUCCESSFULLY LOADED CPU MODEL: {:?}", info),
            Err(e) => println!("FAILED TO LOAD CPU MODEL: {:?}", e),
        }

        assert!(res.is_ok(), "CPU GGUF load failed: {:?}", res.err());
    }

    #[test]
    fn test_simulate_gui_load_model_flow() {
        let _guard = TEST_MUTEX.lock().unwrap();
        use crate::adapter_manager::AdapterRegistry;
        use crate::ai_engine::manager::InferenceManager;

        let app_data_dir = std::path::PathBuf::from(r"C:\Users\lenovo\AppData\Roaming\com.sarathi.app");
        let provider_id = "huggingface";
        let model_id = "meta-llama/Llama-3.2-1B";
        let quantization = "Q8_0";

        println!("Simulating exact GUI load flow for provider='{}', model='{}'", provider_id, model_id);

        let package_dir = AdapterRegistry::resolve_package_dir(&app_data_dir, provider_id, model_id);
        if !package_dir.exists() || !package_dir.join("manifest.json").exists() {
            println!("Skipping Llama test: package dir or manifest does not exist at {:?}", package_dir);
            return;
        }

        let manifest = AdapterRegistry::read_manifest(&package_dir).expect("Failed to read manifest");
        println!("Manifest: model_name='{}', file_path='{}'", manifest.base_model.model_name, manifest.base_model.file_path);

        let gguf_path = match InferenceManager::resolve_gguf_path(&package_dir, &manifest) {
            Ok(p) => p,
            Err(e) => {
                println!("Skipping Llama test: GGUF file not found: {}", e);
                return;
            }
        };
        println!("Resolved GGUF path: {}", gguf_path);

        let profile = crate::model_intelligence::ModelIntelligenceManager::get_or_create_profile(&package_dir, &manifest).expect("Failed to get profile");
        let config = InferenceManager::build_load_config(&app_data_dir, &gguf_path, model_id, &manifest, quantization, &profile, None).expect("Failed to build load config");
        println!("Built ModelLoadConfig: path='{}', ctx={}, gpu_layers={}, threads={}", config.model_path, config.context_length, config.gpu_layers, config.threads);

        let mut runtime = LlamaCppRuntime::new();
        let res = runtime.load_model(&config, |step| println!("Step: {}", step));
        println!("Simulated GUI Load Result: {:?}", res);
        assert!(res.is_ok(), "GUI simulated load failed: {:?}", res.err());
    }

    #[test]
    fn test_simulate_gui_load_qwen_coder_7b() {
        let _guard = TEST_MUTEX.lock().unwrap();
        use crate::adapter_manager::AdapterRegistry;
        use crate::ai_engine::manager::InferenceManager;

        let app_data_dir = std::path::PathBuf::from(r"C:\Users\lenovo\AppData\Roaming\com.sarathi.app");
        let provider_id = "huggingface";
        let model_id = "Qwen/Qwen2.5-Coder-7B";
        let quantization = "Q4_0";

        println!("Simulating exact GUI load flow for provider='{}', model='{}'", provider_id, model_id);

        let package_dir = AdapterRegistry::resolve_package_dir(&app_data_dir, provider_id, model_id);
        if !package_dir.exists() || !package_dir.join("manifest.json").exists() {
            println!("Skipping Qwen test: package dir or manifest does not exist at {:?}", package_dir);
            return;
        }

        let manifest = AdapterRegistry::read_manifest(&package_dir).expect("Failed to read manifest");
        let gguf_path = match InferenceManager::resolve_gguf_path(&package_dir, &manifest) {
            Ok(p) => p,
            Err(e) => {
                println!("Skipping Qwen test: GGUF file not found: {}", e);
                return;
            }
        };
        println!("Resolved GGUF path: {}", gguf_path);

        let profile = crate::model_intelligence::ModelIntelligenceManager::get_or_create_profile(&package_dir, &manifest).expect("Failed to get profile");
        let config = InferenceManager::build_load_config(&app_data_dir, &gguf_path, model_id, &manifest, quantization, &profile, None).expect("Failed to build load config");

        let mut runtime = LlamaCppRuntime::new();
        let res = runtime.load_model(&config, |step| println!("Step: {}", step));
        println!("Simulated Qwen Load Result: {:?}", res);
        assert!(res.is_ok(), "Qwen load failed: {:?}", res.err());
    }

    #[test]
    fn test_generic_multi_model_sequential_session_load() {
        let _guard = TEST_MUTEX.lock().unwrap();
        use crate::adapter_manager::AdapterRegistry;
        use crate::ai_engine::manager::InferenceManager;
        use crate::model_intelligence::{MetadataExtractor, ModelFamily};

        // 1. Verify 4 distinct Model Families are detected generically without model-name hardcoding
        assert_eq!(MetadataExtractor::infer_family_from_string("meta-llama/Llama-3.2-1B"), ModelFamily::Llama);
        assert_eq!(MetadataExtractor::infer_family_from_string("Qwen/Qwen2.5-Coder-7B"), ModelFamily::Qwen);
        assert_eq!(MetadataExtractor::infer_family_from_string("google/gemma-2-9b"), ModelFamily::Gemma);
        assert_eq!(MetadataExtractor::infer_family_from_string("mistralai/Mistral-7B-v0.3"), ModelFamily::Mistral);

        let app_data_dir = std::path::PathBuf::from(r"C:\Users\lenovo\AppData\Roaming\com.sarathi.app");
        let mut runtime = LlamaCppRuntime::new();

        // 2. Load Model 1 (Llama-3.2-1B) in runtime session
        let pkg1 = AdapterRegistry::resolve_package_dir(&app_data_dir, "huggingface", "meta-llama/Llama-3.2-1B");
        if pkg1.exists() && pkg1.join("manifest.json").exists() {
            let manifest1 = AdapterRegistry::read_manifest(&pkg1).unwrap();
            if let Ok(path1) = InferenceManager::resolve_gguf_path(&pkg1, &manifest1) {
            let profile1 = crate::model_intelligence::ModelIntelligenceManager::refresh_profile(&pkg1, &manifest1).unwrap();
            let cfg1 = InferenceManager::build_load_config(&app_data_dir, &path1, "meta-llama/Llama-3.2-1B", &manifest1, "Q8_0", &profile1, None).unwrap();

            println!("\n=== [SEQUENTIAL LOAD TEST 1] Loading Llama-3.2-1B ===");
            let res1 = runtime.load_model(&cfg1, |s| println!("Load step: {}", s));
            assert!(res1.is_ok(), "Failed sequential load 1 (Llama)");
            assert_eq!(cfg1.chat_template, "llama3");

            println!("=== [SEQUENTIAL UNLOAD TEST 1] Unloading Llama-3.2-1B ===");
            assert!(runtime.unload_model().is_ok());
            } // if let Ok(path1)
        }

        // 3. Load Model 2 (Qwen2.5-Coder-7B) into SAME runtime session without process restart
        let pkg2 = AdapterRegistry::resolve_package_dir(&app_data_dir, "huggingface", "Qwen/Qwen2.5-Coder-7B");
        if pkg2.exists() && pkg2.join("manifest.json").exists() {
            let manifest2 = AdapterRegistry::read_manifest(&pkg2).unwrap();
            if let Ok(path2) = InferenceManager::resolve_gguf_path(&pkg2, &manifest2) {
            let profile2 = crate::model_intelligence::ModelIntelligenceManager::refresh_profile(&pkg2, &manifest2).unwrap();
            let cfg2 = InferenceManager::build_load_config(&app_data_dir, &path2, "Qwen/Qwen2.5-Coder-7B", &manifest2, "Q4_0", &profile2, None).unwrap();

            println!("\n=== [SEQUENTIAL LOAD TEST 2] Loading Qwen2.5-Coder-7B into SAME session ===");
            let res2 = runtime.load_model(&cfg2, |s| println!("Load step: {}", s));
            assert!(res2.is_ok(), "Failed sequential load 2 (Qwen)");
            assert_eq!(cfg2.chat_template, "chatml");

            println!("=== [SEQUENTIAL UNLOAD TEST 2] Unloading Qwen2.5-Coder-7B ===");
            assert!(runtime.unload_model().is_ok());
            } // if let Ok(path2)
        }

        println!("=== SEQUENTIAL MULTI-MODEL LOAD TEST SUCCEEDED CLEANLY ===");
    }

    #[test]
    fn test_phase5_2_automatic_runtime_configuration_verification() {
        let _guard = TEST_MUTEX.lock().unwrap();
        use crate::adapter_manager::AdapterRegistry;

        let app_data_dir = std::path::PathBuf::from(r"C:\Users\lenovo\AppData\Roaming\com.sarathi.app");

        // Model 1: Llama 3.2 1B
        let pkg1 = AdapterRegistry::resolve_package_dir(&app_data_dir, "huggingface", "meta-llama/Llama-3.2-1B");
        if pkg1.exists() && pkg1.join("manifest.json").exists() {
            let manifest1 = AdapterRegistry::read_manifest(&pkg1).unwrap();
            let profile1 = crate::model_intelligence::ModelIntelligenceManager::refresh_profile(&pkg1, &manifest1).unwrap();
            println!("\n[PHASE 5.2 VERIFICATION] Llama 3.2 1B Profile Extracted:");
            println!("  Family: {:?}", profile1.model_family);
            println!("  Chat Template: {}", profile1.chat_template);
            println!("  Context Length: {}", profile1.recommended_params.context_length);
            println!("  Stop Tokens: {:?}", profile1.tokens.stop_tokens);
            println!("  Sampling Params: temp={}, top_p={}, repeat_penalty={}, min_p={}",
                profile1.recommended_params.temperature,
                profile1.recommended_params.top_p,
                profile1.recommended_params.repeat_penalty,
                profile1.recommended_params.min_p
            );
            assert_eq!(profile1.model_family, crate::model_intelligence::ModelFamily::Llama);
            assert_eq!(profile1.chat_template, "llama3");
        }

        // Model 2: Qwen 2.5 Coder 7B
        let pkg2 = AdapterRegistry::resolve_package_dir(&app_data_dir, "huggingface", "Qwen/Qwen2.5-Coder-7B");
        if pkg2.exists() && pkg2.join("manifest.json").exists() {
            let manifest2 = AdapterRegistry::read_manifest(&pkg2).unwrap();
            let profile2 = crate::model_intelligence::ModelIntelligenceManager::refresh_profile(&pkg2, &manifest2).unwrap();
            println!("\n[PHASE 5.2 VERIFICATION] Qwen 2.5 Coder 7B Profile Extracted:");
            println!("  Family: {:?}", profile2.model_family);
            println!("  Chat Template: {}", profile2.chat_template);
            println!("  Context Length: {}", profile2.recommended_params.context_length);
            println!("  Stop Tokens: {:?}", profile2.tokens.stop_tokens);
            println!("  Sampling Params: temp={}, top_p={}, repeat_penalty={}, min_p={}",
                profile2.recommended_params.temperature,
                profile2.recommended_params.top_p,
                profile2.recommended_params.repeat_penalty,
                profile2.recommended_params.min_p
            );
            assert_eq!(profile2.model_family, crate::model_intelligence::ModelFamily::Qwen);
            assert_eq!(profile2.chat_template, "chatml");
        }

        println!("\n=== PHASE 5.2 AUTOMATIC RUNTIME CONFIGURATION VERIFIED ===\n");
    }

    // ─── Tool schemas reaching the model ────────────────────────────────────

    /// The construct that made a tool-capable model look tool-incapable.
    #[test]
    fn generation_tags_are_rewritten_rather_than_rejected() {
        let src = "a{%- generation -%}b{% endgeneration %}c";
        let out = super::strip_generation_tags(src);
        assert!(!out.contains("generation -%}"), "got: {out}");
        assert!(out.contains("{%- set"), "whitespace control must survive: {out}");
        assert!(out.contains("{% set"), "and so must its absence: {out}");

        // minijinja has to accept the result, which is the only thing that
        // matters about it.
        let mut env = minijinja::Environment::new();
        env.add_template("t", &out).expect("rewritten template must parse");
        assert_eq!(env.get_template("t").unwrap().render(()).unwrap(), "abc");
    }

    #[test]
    fn a_template_without_the_tags_is_left_untouched() {
        let src = "{% if tools %}x{% endif %}";
        assert!(matches!(super::strip_generation_tags(src), std::borrow::Cow::Borrowed(_)));
    }

    /// A name that never made it into the prompt is a tool the model cannot
    /// call, whatever the template did with the list.
    #[test]
    fn a_tool_absent_from_the_prompt_is_detected() {
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": { "name": "searxng_web_search" }
        })];
        assert_eq!(
            super::first_tool_name_missing("a prompt with no tools", &tools).as_deref(),
            Some("searxng_web_search")
        );
        assert_eq!(
            super::first_tool_name_missing("List of tools: [searxng_web_search]", &tools),
            None
        );
    }

    /// The real thing: LFM2.5's own template, which uses `{% generation %}`
    /// and builds a "List of tools:" system prompt. Before the rewrite this
    /// failed to parse and every tool was dropped.
    #[test]
    fn the_real_lfm25_template_renders_and_carries_its_tools() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/lfm2.5-chat-template.jinja");
        let Ok(src) = std::fs::read_to_string(&path) else {
            eprintln!("fixture missing, skipping: {}", path.display());
            return;
        };

        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "searxng_web_search",
                "description": "Search the web",
                "parameters": {"type":"object","properties":{"query":{"type":"string"}}}
            }
        })];

        let prompt = super::render_jinja_chat_template(
            &src,
            &[ChatMessage::new("user", "find me something")],
            "<|startoftext|>",
            "<|im_end|>",
            &tools,
        )
        .expect("LFM2.5's template must render");

        assert!(
            prompt.contains("searxng_web_search"),
            "the tool name has to reach the model:
{prompt}"
        );
        assert!(super::first_tool_name_missing(&prompt, &tools).is_none());
    }

    /// And the same template with no tools still renders an ordinary prompt.
    #[test]
    fn the_real_template_still_renders_plain_chat() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/lfm2.5-chat-template.jinja");
        let Ok(src) = std::fs::read_to_string(&path) else { return };

        let prompt = super::render_jinja_chat_template(
            &src,
            &[ChatMessage::new("user", "hello")],
            "<|startoftext|>",
            "<|im_end|>",
            &[],
        )
        .expect("plain chat must still render");
        assert!(prompt.contains("hello"), "got: {prompt}");
        assert!(!prompt.contains("List of tools"), "no tools means no tool list: {prompt}");
    }

    /// A previous assistant turn's calls and the tool result that answered them
    /// have to survive into the next prompt, or the model cannot see its own
    /// conversation.
    #[test]
    fn a_tool_call_and_its_result_survive_into_the_next_prompt() {
        let template = concat!(
            "{% for m in messages %}",
            "[{{ m.role }}]",
            "{% if m.tool_calls is defined %}",
            "{% for c in m.tool_calls %}CALL:{{ c.function.name }}{% endfor %}",
            "{% endif %}",
            "{% if m.tool_call_id is defined %}FOR:{{ m.tool_call_id }}{% endif %}",
            "{{ m.content }}",
            "{% endfor %}"
        );

        let assistant = ChatMessage {
            tool_calls: vec![serde_json::json!({
                "id": "call_1",
                "function": {"name": "searxng_web_search", "arguments": "{}"}
            })],
            ..ChatMessage::new("assistant", "")
        };
        let result = ChatMessage {
            tool_call_id: Some("call_1".into()),
            name: Some("searxng_web_search".into()),
            ..ChatMessage::new("tool", "three results")
        };

        let out = super::render_jinja_chat_template(
            template,
            &[ChatMessage::new("user", "search"), assistant, result],
            "",
            "",
            &[],
        )
        .expect("renders");

        assert!(out.contains("CALL:searxng_web_search"), "the call was lost: {out}");
        assert!(out.contains("FOR:call_1"), "the result lost its call id: {out}");
        assert!(out.contains("three results"), "the result body was lost: {out}");
    }

    /// The round trip's real failure mode: OpenAI sends `arguments` as a JSON
    /// string, HuggingFace templates iterate it as a mapping.
    #[test]
    fn tool_call_arguments_are_decoded_for_the_template() {
        let calls = vec![serde_json::json!({
            "id": "call_1",
            "function": {"name": "search", "arguments": "{\"query\":\"btc\"}"}
        })];
        let out = super::normalise_tool_calls(&calls);
        assert!(out[0]["function"]["arguments"].is_object(), "got: {out:?}");
        assert_eq!(out[0]["function"]["arguments"]["query"], "btc");
    }

    #[test]
    fn arguments_that_are_not_json_are_left_alone() {
        let calls = vec![serde_json::json!({
            "function": {"name": "f", "arguments": "not json at all"}
        })];
        let out = super::normalise_tool_calls(&calls);
        assert_eq!(out[0]["function"]["arguments"], "not json at all");
    }

    /// The whole second turn, through LFM2.5's real template: the model has to
    /// see its own previous call and the result that answered it.
    #[test]
    fn the_real_template_renders_a_full_tool_round_trip() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/lfm2.5-chat-template.jinja");
        let Ok(src) = std::fs::read_to_string(&path) else { return };

        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {"name": "searxng_web_search", "description": "Search",
                         "parameters": {"type":"object","properties":{"query":{"type":"string"}}}}
        })];

        let assistant = ChatMessage {
            tool_calls: vec![serde_json::json!({
                "id": "call_1",
                "function": {"name": "searxng_web_search", "arguments": "{\"query\":\"btc\"}"}
            })],
            ..ChatMessage::new("assistant", "")
        };
        let result = ChatMessage {
            tool_call_id: Some("call_1".into()),
            name: Some("searxng_web_search".into()),
            ..ChatMessage::new("tool", "BTC is $91,432.18")
        };

        let prompt = super::render_jinja_chat_template(
            &src,
            &[ChatMessage::new("user", "price of btc?"), assistant, result],
            "<|startoftext|>",
            "<|im_end|>",
            &tools,
        )
        .expect("a tool round trip must render");

        assert!(prompt.contains("searxng_web_search"), "tool name lost:
{prompt}");
        assert!(prompt.contains("BTC is $91,432.18"), "the result was lost:
{prompt}");
        assert!(
            prompt.contains("<|tool_call_start|>"),
            "the previous call must be re-rendered in the model's own syntax:
{prompt}"
        );
    }

    /// The Python methods HF templates assume. One missing method aborts a
    /// whole render, so these are load-bearing rather than nice-to-have.
    #[test]
    fn python_string_methods_are_available_to_templates() {
        let cases = [
            (r#"{{ "abc" if "hello world".endswith("world") else "no" }}"#, "abc"),
            (r#"{{ "yes" if "hello".startswith("he") else "no" }}"#, "yes"),
            (r#"{{ "  x  ".strip() }}"#, "x"),
            (r#"{{ "xxaxx".strip("x") }}"#, "a"),
            (r#"{{ "  x".lstrip() }}"#, "x"),
            (r#"{{ "x  ".rstrip() }}"#, "x"),
            (r#"{{ "AB".lower() }}"#, "ab"),
            (r#"{{ "ab".upper() }}"#, "AB"),
            (r#"{{ "a-b".replace("-", "+") }}"#, "a+b"),
            (r#"{{ "a,b,c".split(",")[1] }}"#, "b"),
            (r#"{{ "a b".title() }}"#, "A B"),
            (r#"{{ "ab".capitalize() }}"#, "Ab"),
            (r#"{{ "aXbXc".count("X") }}"#, "2"),
            (r#"{{ "hello".find("ll") }}"#, "2"),
            (r#"{{ "hello".find("zz") }}"#, "-1"),
        ];

        for (src, expected) in cases {
            let out = super::render_jinja_chat_template(src, &[], "", "", &[])
                .unwrap_or_else(|e| panic!("{src} failed: {e}"));
            assert_eq!(out.trim(), expected, "{src}");
        }
    }

    #[test]
    fn a_genuinely_unknown_method_still_fails_loudly() {
        // The shim must not swallow real mistakes into a silently wrong prompt.
        let err = super::render_jinja_chat_template(
            r#"{{ "x".no_such_method() }}"#, &[], "", "", &[],
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("no_such_method"), "got: {err:#}");
    }

    #[test]
    fn mapping_methods_work_for_tool_arguments() {
        let template = r#"{% for m in messages %}{% for c in m.tool_calls %}{% for k, v in c.function.arguments.items() %}{{ k }}={{ v }};{% endfor %}{% endfor %}{% endfor %}"#;

        let assistant = ChatMessage {
            tool_calls: vec![serde_json::json!({
                "function": {"name": "f", "arguments": "{\"a\":1,\"b\":\"two\"}"}
            })],
            ..ChatMessage::new("assistant", "")
        };

        let out = super::render_jinja_chat_template(template, &[assistant], "", "", &[])
            .expect("items() must work on decoded arguments");
        assert!(out.contains("a=1"), "got: {out}");
        assert!(out.contains("b=two"), "got: {out}");
    }
}
