//! GGUF header metadata, read before the model is loaded.
//!
//! Deciding where a model's weights go has to happen *before* a `LlamaModel`
//! exists, and `llama-cpp-2` only exposes `n_layer()`, `n_params()` and
//! `meta_val_str()` on an already-loaded model. This reads the same figures
//! straight from the file's key-value header.
//!
//! Only the header is touched. Arrays are seeked past rather than read — the
//! tokenizer vocabulary alone is typically 128k strings — so this costs a few
//! kilobytes of I/O on a 12 GB file.
//!
//! Two planner inputs depend on it:
//!
//! - `block_count`, which [`vram_planner`](super::vram_planner) otherwise guesses
//!   at 32 (`ASSUMED_LAYERS_FALLBACK`). gpt-oss-20b has 24.
//! - the exact KV cache cost, which the size-banded estimate gets badly wrong for
//!   MoE: it bands on *file size*, but a MoE file is large because of experts
//!   while its KV cost is driven by attention. For gpt-oss-20b the band returns
//!   256 KB/token against a real 48 KB — a 5× over-estimate that would consume an
//!   entire 4 GB card's weight budget on its own.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// v1 used 32-bit lengths throughout and is not produced by any current
/// converter. Refusing it is better than misreading it as v2.
const MIN_VERSION: u32 = 2;

/// Guards against allocating or looping against a corrupt length.
const MAX_KV_COUNT: u64 = 1 << 20;
const MAX_STRING_BYTES: u64 = 64 << 20;

/// KV cache elements are f16 in llama.cpp regardless of weight quantization.
const KV_BYTES_PER_ELEMENT: u64 = 2;

/// gate, up and down projections per expert. A fused `gate_up` tensor is
/// `embedding_length × (expert_ff_length × 2)` — the same parameter total, so
/// this holds for both tensor layouts.
const PROJECTIONS_PER_EXPERT: u64 = 3;

/// Ceiling on the share of a file attributed to routed experts.
///
/// The expert figure is computed from header geometry while the total may come
/// from a different key; a disagreement must not claim the whole file is experts
/// and leave nothing resident.
const MAX_EXPERT_FRACTION: f64 = 0.95;

/// What a GGUF file actually is.
///
/// Not every `.gguf` is a model. Repositories ship modules that exist to serve
/// a model — a vision projector, a multi-token-prediction head, an EAGLE-3
/// speculative-decoding draft — and llama.cpp cannot load any of them on their
/// own. Asked to, it returns a null pointer, which reaches the user as
/// `NullResult` and explains nothing.
///
/// The distinction is read from the file rather than from its name. A name is a
/// convention that the next side-car format is free to ignore; declaring a
/// target model is what *makes* something a side-car.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufRole {
    /// A complete model. The only thing that can be loaded and talked to.
    Model,
    /// A LoRA patch. Applied on top of a model, never loaded instead of one.
    Adapter,
    /// A module that serves some other model, with what it declared about that
    /// relationship — used verbatim in the error, since it is the evidence.
    Auxiliary { evidence: String },
}

impl GgufRole {
    /// One sentence for someone who tried to load this and could not.
    pub fn refusal(&self, architecture: &str) -> Option<String> {
        match self {
            Self::Model => None,
            Self::Adapter => Some(format!(
                "This file is a LoRA adapter for a '{architecture}' model, not a model. \
                 Install it from the base model's page — on its own there is nothing to run."
            )),
            Self::Auxiliary { evidence } => Some(format!(
                "This file is a '{architecture}' helper module rather than a model — {evidence}. \
                 Modules like speculative-decoding drafts, vision projectors and \
                 multi-token-prediction heads accelerate or extend a model that must be \
                 downloaded separately; llama.cpp cannot load one by itself. \
                 Download the model this was built for instead."
            )),
        }
    }
}

/// The header figures the planner needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufMetadata {
    pub architecture: String,
    /// Whether this file is a model at all. Checked before loading, because
    /// llama.cpp's answer for everything else is an unexplained null.
    pub role: GgufRole,
    pub block_count: u32,
    pub embedding_length: u32,
    /// 0 for a dense model.
    pub expert_count: u32,
    /// Experts consulted per token — 4 of 32 for gpt-oss. This is what makes
    /// expert offload viable: only this share crosses PCIe per token.
    pub expert_used_count: u32,
    pub expert_ff_length: u32,
    pub head_count_kv: u32,
    pub key_length: u32,
    pub value_length: u32,
    /// From `general.parameter_count`, which not every converter writes.
    pub parameter_count: Option<u64>,
    /// Training context, from `{arch}.context_length`.
    pub context_length: u32,
    /// True when the file carries image-tower metadata — `{arch}.vision.*` keys,
    /// or the `clip.*` block a projector uses.
    ///
    /// Structural rather than nominal: a multimodal model declares the tower it
    /// has, whatever its architecture happens to be called.
    pub has_vision: bool,
    /// True when the file declares a pooling strategy.
    ///
    /// Only a model whose output is a sentence embedding needs to say how its
    /// token vectors are reduced to one vector. A generative model never states
    /// it, which is what makes this the honest test for an embedding model
    /// rather than matching `bert` in the architecture name.
    pub has_pooling: bool,
    /// `general.file_type`, the quantization the file was written with.
    ///
    /// Read here so what Storage displays comes from the file rather than from
    /// its name — the two disagreed when a helper module was registered under
    /// the label parsed out of its filename.
    pub file_type: Option<u32>,
}

impl GgufMetadata {
    pub fn is_moe(&self) -> bool {
        self.expert_count > 0
    }

    /// Exact f16 KV cache cost per token.
    ///
    /// `layers × kv_heads × (key_dim + value_dim) × 2`. The key and value
    /// dimensions carry the factor of two that the size-banded estimate spells
    /// out separately.
    pub fn kv_bytes_per_token(&self) -> u64 {
        u64::from(self.block_count)
            * u64::from(self.head_count_kv)
            * (u64::from(self.key_length) + u64::from(self.value_length))
            * KV_BYTES_PER_ELEMENT
    }

    /// Routed-expert parameters across every layer.
    pub fn expert_params(&self) -> u64 {
        u64::from(self.block_count)
            .saturating_mul(u64::from(self.expert_count))
            .saturating_mul(PROJECTIONS_PER_EXPERT)
            .saturating_mul(u64::from(self.embedding_length))
            .saturating_mul(u64::from(self.expert_ff_length))
    }

    /// Parameters actually used per token.
    ///
    /// Everything outside the routed experts, plus the share of experts the
    /// router consults. Returns `None` when the total is unknown, since the
    /// non-expert remainder cannot be derived without it.
    pub fn active_params(&self, total_params: Option<u64>) -> Option<u64> {
        let total = total_params.or(self.parameter_count)?;
        let experts = self.expert_params();

        if experts == 0 || self.expert_count == 0 {
            return Some(total);
        }
        // A disagreement between header geometry and the stated total must not
        // underflow into a nonsense figure.
        let dense = total.saturating_sub(experts);
        let used = experts / u64::from(self.expert_count) * u64::from(self.expert_used_count);

        Some(dense.saturating_add(used))
    }

    /// Bytes of routed-expert weight inside a file of `model_bytes`.
    ///
    /// Derived as a share of the real file size rather than summed from tensor
    /// dimensions, because per-tensor byte sizes are not recoverable from the
    /// header across mixed quantizations. This assumes experts are quantized
    /// comparably to the rest of the model, which holds for the targets
    /// (gpt-oss-20b is uniformly MXFP4).
    ///
    /// Returns 0 when the file is dense or the geometry is unusable, which the
    /// caller reads as "no expert offload available".
    pub fn expert_bytes(&self, model_bytes: u64, total_params: Option<u64>) -> u64 {
        let experts = self.expert_params();
        let total = total_params.or(self.parameter_count).unwrap_or(0);
        if experts == 0 || total == 0 {
            return 0;
        }

        let fraction = (experts as f64 / total as f64).min(MAX_EXPERT_FRACTION);
        (model_bytes as f64 * fraction) as u64
    }
}

/// Reads the header of the GGUF file at `path`.
pub fn read_gguf_metadata(path: &Path) -> Result<GgufMetadata> {
    let file = File::open(path)
        .with_context(|| format!("could not open GGUF file '{}'", path.display()))?;
    let mut reader = BufReader::new(file);
    parse_gguf_metadata(&mut reader)
        .with_context(|| format!("could not read GGUF header of '{}'", path.display()))
}

/// Parses a GGUF header from any seekable stream.
///
/// Split from [`read_gguf_metadata`] so the format handling is testable against
/// synthetic headers without writing multi-gigabyte fixtures.
pub fn parse_gguf_metadata<R: Read + Seek>(r: &mut R) -> Result<GgufMetadata> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).context("file is too short to be a GGUF")?;
    if &magic != GGUF_MAGIC {
        bail!(
            "not a GGUF file: expected magic 'GGUF', found {:?}",
            String::from_utf8_lossy(&magic)
        );
    }

    let version = read_u32(r)?;
    if version < MIN_VERSION {
        bail!("unsupported GGUF version {version}; {MIN_VERSION} is the minimum");
    }

    let _tensor_count = read_u64(r)?;
    let kv_count = read_u64(r)?;
    if kv_count > MAX_KV_COUNT {
        bail!("GGUF header claims {kv_count} metadata entries, which is not credible");
    }

    let mut kv: HashMap<String, Scalar> = HashMap::new();
    // Every key that was present, including the arrays whose values are skipped.
    //
    // Role detection asks whether a key *exists* — `target_layers` is an array,
    // so a map of decoded scalars alone would not know it was ever there.
    let mut keys: HashSet<String> = HashSet::new();
    for _ in 0..kv_count {
        let key = read_string(r)?;
        let value_type = read_u32(r)?;
        let value = read_value(r, value_type)?;
        if let Some(value) = value {
            kv.insert(key.clone(), value);
        }
        keys.insert(key);
    }

    from_kv(&kv, &keys)
}

/// Decides what the file is from what it says about itself.
///
/// Three signals, in order of how conclusive they are:
///
/// 1. `general.type` — converters that know they are emitting a non-model write
///    it. `adapter` is the common one.
/// 2. A declared *target*. `{arch}.target_hidden_size` and `{arch}.target_layers`
///    say "I attach to a model with this shape", which no standalone model has
///    any reason to state. EAGLE-3 drafts, Medusa heads and MTP modules all
///    declare it — this is the general rule, not a list of their names.
/// 3. Projector tensors. A vision projector is keyed under `clip.*`, which again
///    describes a component rather than a model.
fn classify(architecture: &str, kv: &HashMap<String, Scalar>, keys: &HashSet<String>) -> GgufRole {
    if let Some(declared) = kv.get("general.type").and_then(Scalar::as_str) {
        match declared.trim().to_ascii_lowercase().as_str() {
            "adapter" | "lora" => return GgufRole::Adapter,
            "projector" | "clip" | "mmproj" => {
                return GgufRole::Auxiliary {
                    evidence: format!("it declares itself as '{declared}'"),
                }
            }
            _ => {}
        }
    }

    for suffix in ["target_hidden_size", "target_layers", "target_model", "target_vocab_size"] {
        let key = format!("{architecture}.{suffix}");
        if keys.contains(&key) {
            return GgufRole::Auxiliary {
                evidence: format!(
                    "it declares a target model to attach to (`{key}`), which a \
                     standalone model has no reason to state"
                ),
            };
        }
    }

    if keys.iter().any(|k| k.starts_with("clip.")) {
        return GgufRole::Auxiliary {
            evidence: "it carries `clip.*` projector metadata rather than model layers".to_string(),
        };
    }

    GgufRole::Model
}

fn from_kv(kv: &HashMap<String, Scalar>, keys: &HashSet<String>) -> Result<GgufMetadata> {
    let architecture = kv
        .get("general.architecture")
        .and_then(Scalar::as_str)
        .ok_or_else(|| anyhow!("GGUF header has no general.architecture"))?
        .to_string();

    let get = |suffix: &str| {
        kv.get(&format!("{architecture}.{suffix}"))
            .and_then(Scalar::as_u32)
    };

    let context_length = get("context_length").unwrap_or(0);

    let role = classify(&architecture, kv, keys);

    // Required of a model, because everything downstream is per-layer — but not
    // of a side-car, which frequently omits it. Bailing here would replace a
    // precise "this is a draft model, not a model" with a parse error that says
    // only that a key is missing.
    let block_count = match get("block_count") {
        Some(n) => n,
        None if role != GgufRole::Model => 0,
        None => bail!("GGUF header has no {architecture}.block_count"),
    };
    let embedding_length = get("embedding_length").unwrap_or(0);
    let head_count = get("attention.head_count").unwrap_or(0);

    // A GGUF may store head_count_kv as a per-layer array, which is skipped
    // during parsing. Falling back to head_count over-states the KV cost rather
    // than under-stating it, which is the safe direction — see vram_planner.
    let head_count_kv = get("attention.head_count_kv").unwrap_or(head_count);

    // llama.cpp applies the same default when these keys are absent.
    let default_head_dim = embedding_length.checked_div(head_count).unwrap_or(0);
    let key_length = get("attention.key_length").unwrap_or(default_head_dim);
    let value_length = get("attention.value_length").unwrap_or(default_head_dim);
    let expert_count = get("expert_count").unwrap_or(0);
    let expert_used_count = get("expert_used_count").unwrap_or(0);
    let expert_ff_length = get("expert_feed_forward_length").unwrap_or(0);

    // Every `get` is done, so the closure's borrow of `architecture` has ended
    // and it can be moved into the result.
    Ok(GgufMetadata {
        architecture,
        role,
        block_count,
        embedding_length,
        expert_count,
        expert_used_count,
        expert_ff_length,
        head_count_kv,
        key_length,
        value_length,
        parameter_count: kv.get("general.parameter_count").and_then(Scalar::as_u64),
        context_length,
        has_vision: keys.iter().any(|k| k.contains(".vision.") || k.starts_with("clip.")),
        has_pooling: keys.iter().any(|k| k.ends_with(".pooling_type")),
        file_type: kv.get("general.file_type").and_then(Scalar::as_u32),
    })
}

/// Names the quantization from GGUF's `general.file_type`.
///
/// The enum is llama.cpp's own and its values are stable — a file that says 15
/// is Q4_K_M on every machine that reads it. Returns `None` for a value this
/// build does not know, so the caller falls back to the filename rather than
/// inventing a label.
pub fn file_type_label(file_type: u32) -> Option<&'static str> {
    Some(match file_type {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        38 => "MXFP4",
        _ => return None,
    })
}

// ─── Value decoding ─────────────────────────────────────────────────────────

/// A scalar metadata value. Arrays are skipped rather than represented — none of
/// the figures this module needs is stored as one.
#[derive(Debug, Clone, PartialEq)]
enum Scalar {
    U(u64),
    I(i64),
    F(f64),
    Bool(bool),
    Str(String),
}

impl Scalar {
    fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U(v) => u32::try_from(*v).ok(),
            Self::I(v) => u32::try_from(*v).ok(),
            _ => None,
        }
    }

    fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U(v) => Some(*v),
            Self::I(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// Reads one value, or `Ok(None)` for an array that was skipped.
fn read_value<R: Read + Seek>(r: &mut R, value_type: u32) -> Result<Option<Scalar>> {
    let scalar = match value_type {
        0 => Scalar::U(u64::from(read_n::<_, 1>(r)?[0])),
        1 => Scalar::I(i64::from(read_n::<_, 1>(r)?[0] as i8)),
        2 => Scalar::U(u64::from(u16::from_le_bytes(read_n::<_, 2>(r)?))),
        3 => Scalar::I(i64::from(i16::from_le_bytes(read_n::<_, 2>(r)?))),
        4 => Scalar::U(u64::from(read_u32(r)?)),
        5 => Scalar::I(i64::from(i32::from_le_bytes(read_n::<_, 4>(r)?))),
        6 => Scalar::F(f64::from(f32::from_le_bytes(read_n::<_, 4>(r)?))),
        7 => Scalar::Bool(read_n::<_, 1>(r)?[0] != 0),
        8 => Scalar::Str(read_string(r)?),
        9 => {
            skip_array(r)?;
            return Ok(None);
        }
        10 => Scalar::U(read_u64(r)?),
        11 => Scalar::I(i64::from_le_bytes(read_n::<_, 8>(r)?)),
        12 => Scalar::F(f64::from_le_bytes(read_n::<_, 8>(r)?)),
        other => bail!("unknown GGUF value type {other}"),
    };
    Ok(Some(scalar))
}

/// Byte width of a fixed-size value type, or `None` for strings and arrays.
fn scalar_width(value_type: u32) -> Option<u64> {
    match value_type {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

fn skip_array<R: Read + Seek>(r: &mut R) -> Result<()> {
    let element_type = read_u32(r)?;
    let len = read_u64(r)?;

    match scalar_width(element_type) {
        Some(width) => {
            let bytes = width
                .checked_mul(len)
                .ok_or_else(|| anyhow!("GGUF array length {len} overflows"))?;
            seek_forward(r, bytes)
        }
        // Strings are variable-length, so each has to be stepped over.
        None if element_type == 8 => {
            for _ in 0..len {
                let bytes = read_u64(r)?;
                if bytes > MAX_STRING_BYTES {
                    bail!("GGUF string of {bytes} bytes is not credible");
                }
                seek_forward(r, bytes)?;
            }
            Ok(())
        }
        None if element_type == 9 => bail!("nested GGUF arrays are not supported"),
        None => bail!("unknown GGUF array element type {element_type}"),
    }
}

/// Largest skip worth reading through the buffer rather than seeking.
///
/// `Seek` on a `BufReader` throws the buffer away, because the implementation
/// cannot know the target is inside it. That is the right trade for a long jump
/// and a disaster for a short one: the tokenizer's `merges` array is ~293k
/// strings of a few bytes each, and seeking past them one at a time turned a
/// header read into ~400k `ReadFile` calls — 950 ms on an already-warm file.
/// Reading a short skip through the buffer instead answers it from memory.
///
/// 64 KiB is comfortably above any single token or merge and far below the
/// tensor data a real seek is for.
const DISCARD_LIMIT: u64 = 64 << 10;

fn seek_forward<R: Read + Seek>(r: &mut R, bytes: u64) -> Result<()> {
    if bytes <= DISCARD_LIMIT {
        let skipped = std::io::copy(&mut r.by_ref().take(bytes), &mut std::io::sink())
            .context("GGUF header ended unexpectedly")?;
        if skipped != bytes {
            bail!("GGUF header ended after {skipped} of {bytes} skipped bytes");
        }
        return Ok(());
    }

    let offset = i64::try_from(bytes)
        .map_err(|_| anyhow!("GGUF skip of {bytes} bytes overflows a file offset"))?;
    r.seek(SeekFrom::Current(offset))
        .context("GGUF header ended unexpectedly")?;
    Ok(())
}

fn read_n<R: Read, const N: usize>(r: &mut R) -> Result<[u8; N]> {
    let mut buf = [0u8; N];
    r.read_exact(&mut buf).context("GGUF header ended unexpectedly")?;
    Ok(buf)
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    Ok(u32::from_le_bytes(read_n::<_, 4>(r)?))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    Ok(u64::from_le_bytes(read_n::<_, 8>(r)?))
}

fn read_string<R: Read>(r: &mut R) -> Result<String> {
    let len = read_u64(r)?;
    if len > MAX_STRING_BYTES {
        bail!("GGUF string of {len} bytes is not credible");
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf).context("GGUF header ended unexpectedly")?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // ─── Synthetic header construction ──────────────────────────────────────

    fn gguf_string(s: &str) -> Vec<u8> {
        let mut out = (s.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(s.as_bytes());
        out
    }

    fn kv_str(key: &str, value: &str) -> Vec<u8> {
        let mut out = gguf_string(key);
        out.extend_from_slice(&8u32.to_le_bytes());
        out.extend(gguf_string(value));
        out
    }

    fn kv_u32(key: &str, value: u32) -> Vec<u8> {
        let mut out = gguf_string(key);
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&value.to_le_bytes());
        out
    }

    fn kv_u64(key: &str, value: u64) -> Vec<u8> {
        let mut out = gguf_string(key);
        out.extend_from_slice(&10u32.to_le_bytes());
        out.extend_from_slice(&value.to_le_bytes());
        out
    }

    /// A string array, as the tokenizer vocabulary is stored.
    fn kv_string_array(key: &str, values: &[&str]) -> Vec<u8> {
        let mut out = gguf_string(key);
        out.extend_from_slice(&9u32.to_le_bytes());
        out.extend_from_slice(&8u32.to_le_bytes());
        out.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for v in values {
            out.extend(gguf_string(v));
        }
        out
    }

    fn header(entries: Vec<Vec<u8>>) -> Cursor<Vec<u8>> {
        let mut out = GGUF_MAGIC.to_vec();
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes()); // tensor count
        out.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        for e in entries {
            out.extend(e);
        }
        Cursor::new(out)
    }

    /// gpt-oss-20b's real geometry: 24 layers, 8 KV heads, 64-wide K and V.
    fn gpt_oss_entries() -> Vec<Vec<u8>> {
        vec![
            kv_str("general.architecture", "gpt-oss"),
            kv_u64("general.parameter_count", 20_900_000_000),
            kv_u32("gpt-oss.block_count", 24),
            kv_u32("gpt-oss.embedding_length", 2880),
            kv_u32("gpt-oss.attention.head_count", 64),
            kv_u32("gpt-oss.attention.head_count_kv", 8),
            kv_u32("gpt-oss.attention.key_length", 64),
            kv_u32("gpt-oss.attention.value_length", 64),
            kv_u32("gpt-oss.expert_count", 32),
            kv_u32("gpt-oss.expert_used_count", 4),
            kv_u32("gpt-oss.expert_feed_forward_length", 2880),
        ]
    }

    // ─── Tests ──────────────────────────────────────────────────────────────

    #[test]
    fn reads_the_geometry_a_moe_plan_needs() {
        let meta = parse_gguf_metadata(&mut header(gpt_oss_entries())).unwrap();

        assert_eq!(meta.architecture, "gpt-oss");
        assert_eq!(meta.block_count, 24, "the planner otherwise assumes 32");
        assert_eq!(meta.expert_count, 32);
        assert!(meta.is_moe());
    }

    /// The figure the size-banded estimate gets 5× wrong for MoE.
    #[test]
    fn kv_cost_per_token_is_exact() {
        let meta = parse_gguf_metadata(&mut header(gpt_oss_entries())).unwrap();

        // 24 layers × 8 KV heads × (64 + 64) × 2 bytes
        assert_eq!(meta.kv_bytes_per_token(), 49_152);

        // At the working context this is a few hundred MB, not the ~2 GB the
        // banded estimate would charge a 4 GB card.
        assert!(meta.kv_bytes_per_token() * 8192 < 512 * 1024 * 1024);
    }

    #[test]
    fn a_dense_model_reports_no_experts() {
        let meta = parse_gguf_metadata(&mut header(vec![
            kv_str("general.architecture", "llama"),
            kv_u32("llama.block_count", 32),
            kv_u32("llama.embedding_length", 4096),
            kv_u32("llama.attention.head_count", 32),
            kv_u32("llama.attention.head_count_kv", 8),
        ]))
        .unwrap();

        assert!(!meta.is_moe());
        assert_eq!(meta.expert_params(), 0);
        assert_eq!(meta.expert_bytes(8 * 1024 * 1024 * 1024, None), 0);
    }

    #[test]
    fn head_dimension_defaults_to_embedding_over_heads_when_absent() {
        // llama.cpp applies the same default; 4096 / 32 = 128.
        let meta = parse_gguf_metadata(&mut header(vec![
            kv_str("general.architecture", "llama"),
            kv_u32("llama.block_count", 32),
            kv_u32("llama.embedding_length", 4096),
            kv_u32("llama.attention.head_count", 32),
            kv_u32("llama.attention.head_count_kv", 8),
        ]))
        .unwrap();

        assert_eq!(meta.key_length, 128);
        assert_eq!(meta.value_length, 128);
        assert_eq!(meta.kv_bytes_per_token(), 32 * 8 * 256 * 2);
    }

    #[test]
    fn expert_bytes_are_a_share_of_the_real_file_size() {
        let meta = parse_gguf_metadata(&mut header(gpt_oss_entries())).unwrap();
        let file_bytes = 12_800_000_000u64;

        let experts = meta.expert_bytes(file_bytes, None);

        // Experts dominate a MoE file, but never all of it — the attention stack
        // has to stay resident for the split to be worth making.
        assert!(experts > file_bytes / 2, "got {experts} of {file_bytes}");
        assert!(experts < file_bytes, "got {experts} of {file_bytes}");
    }

    #[test]
    fn expert_share_is_capped_when_the_geometry_disagrees_with_the_total() {
        // A parameter_count far too small would otherwise claim the entire file
        // is experts, leaving nothing resident.
        let mut entries = gpt_oss_entries();
        entries[1] = kv_u64("general.parameter_count", 1_000_000);

        let meta = parse_gguf_metadata(&mut header(entries)).unwrap();
        let file_bytes = 12_800_000_000u64;

        assert!(meta.expert_bytes(file_bytes, None) <= (file_bytes as f64 * 0.95) as u64);
    }

    #[test]
    fn a_caller_supplied_total_wins_over_the_header() {
        let meta = parse_gguf_metadata(&mut header(gpt_oss_entries())).unwrap();
        let file_bytes = 12_800_000_000u64;

        let from_header = meta.expert_bytes(file_bytes, None);
        let from_caller = meta.expert_bytes(file_bytes, Some(20_900_000_000 * 2));

        assert!(
            from_caller < from_header,
            "doubling the total should halve the expert share ({from_caller} vs {from_header})"
        );
    }

    /// The vocabulary is 128k+ strings; reading it would defeat the point of
    /// touching only the header.
    #[test]
    fn string_arrays_are_stepped_over_rather_than_read() {
        let mut entries = gpt_oss_entries();
        entries.insert(
            1,
            kv_string_array("tokenizer.ggml.tokens", &["hello", "world", "<|end|>"]),
        );

        let meta = parse_gguf_metadata(&mut header(entries)).unwrap();

        assert_eq!(meta.block_count, 24, "keys after the array must still be found");
        assert_eq!(meta.expert_count, 32);
    }

    #[test]
    fn numeric_arrays_are_stepped_over_too() {
        let mut entries = gpt_oss_entries();
        let mut arr = gguf_string("tokenizer.ggml.token_type");
        arr.extend_from_slice(&9u32.to_le_bytes());
        arr.extend_from_slice(&5u32.to_le_bytes()); // INT32 elements
        arr.extend_from_slice(&4u64.to_le_bytes());
        arr.extend_from_slice(&[0u8; 16]);
        entries.insert(1, arr);

        let meta = parse_gguf_metadata(&mut header(entries)).unwrap();

        assert_eq!(meta.block_count, 24);
    }

    /// Only a fraction of the experts fires per token — the reason offloading
    /// them to system RAM is affordable at all.
    #[test]
    fn active_params_count_only_the_experts_the_router_uses() {
        let meta = parse_gguf_metadata(&mut header(gpt_oss_entries())).unwrap();
        let total = 20_900_000_000u64;

        let active = meta.active_params(Some(total)).unwrap();

        assert!(active < total / 2, "4 of 32 experts fire, got {active} of {total}");
        assert!(active > 0);
    }

    #[test]
    fn a_dense_model_has_every_parameter_active() {
        let meta = parse_gguf_metadata(&mut header(vec![
            kv_str("general.architecture", "llama"),
            kv_u32("llama.block_count", 32),
        ]))
        .unwrap();

        assert_eq!(meta.active_params(Some(8_000_000_000)), Some(8_000_000_000));
    }

    #[test]
    fn active_params_are_unknown_without_a_total() {
        let mut entries = gpt_oss_entries();
        entries.remove(1); // general.parameter_count

        let meta = parse_gguf_metadata(&mut header(entries)).unwrap();
        assert_eq!(meta.active_params(None), None);
    }

    #[test]
    fn a_file_that_is_not_gguf_is_rejected() {
        let mut cursor = Cursor::new(b"NOTGGUF and then some".to_vec());

        let err = parse_gguf_metadata(&mut cursor).unwrap_err().to_string();
        assert!(err.contains("not a GGUF file"), "got: {err}");
    }

    #[test]
    fn a_truncated_header_is_an_error_rather_than_a_panic() {
        let full = header(gpt_oss_entries()).into_inner();

        for cut in [2, 8, 20, 40, 60] {
            let mut cursor = Cursor::new(full[..cut.min(full.len())].to_vec());
            assert!(
                parse_gguf_metadata(&mut cursor).is_err(),
                "a header cut at {cut} bytes must not parse"
            );
        }
    }

    #[test]
    fn a_header_without_an_architecture_is_rejected() {
        let mut cursor = header(vec![kv_u32("llama.block_count", 32)]);

        let err = parse_gguf_metadata(&mut cursor).unwrap_err().to_string();
        assert!(err.contains("general.architecture"), "got: {err}");
    }

    #[test]
    fn a_header_without_a_block_count_is_rejected() {
        // Without it there is no per-layer expert size, so no N can be computed.
        let mut cursor = header(vec![kv_str("general.architecture", "llama")]);

        let err = parse_gguf_metadata(&mut cursor).unwrap_err().to_string();
        assert!(err.contains("block_count"), "got: {err}");
    }

    #[test]
    fn an_absurd_metadata_count_is_refused_before_looping() {
        let mut out = GGUF_MAGIC.to_vec();
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&u64::MAX.to_le_bytes());

        let err = parse_gguf_metadata(&mut Cursor::new(out)).unwrap_err().to_string();
        assert!(err.contains("not credible"), "got: {err}");
    }

    #[test]
    fn version_1_is_refused_rather_than_misread() {
        let mut out = GGUF_MAGIC.to_vec();
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());

        let err = parse_gguf_metadata(&mut Cursor::new(out)).unwrap_err().to_string();
        assert!(err.contains("version 1"), "got: {err}");
    }

    #[test]
    fn every_scalar_type_round_trips() {
        // A converter may store block_count as any integer width.
        for (type_tag, bytes) in [
            (0u32, vec![24u8]),
            (2, 24u16.to_le_bytes().to_vec()),
            (4, 24u32.to_le_bytes().to_vec()),
            (5, 24i32.to_le_bytes().to_vec()),
            (10, 24u64.to_le_bytes().to_vec()),
            (11, 24i64.to_le_bytes().to_vec()),
        ] {
            let mut entry = gguf_string("llama.block_count");
            entry.extend_from_slice(&type_tag.to_le_bytes());
            entry.extend_from_slice(&bytes);

            let meta = parse_gguf_metadata(&mut header(vec![
                kv_str("general.architecture", "llama"),
                entry,
            ]))
            .unwrap();

            assert_eq!(meta.block_count, 24, "value type {type_tag} did not decode");
        }
    }

    // ─── What the file actually is ──────────────────────────────────────────
    //
    // Every fixture below is transcribed from the header of a real file that
    // was downloaded through Sarathi and failed to load with
    // "GPU error: NullResult, CPU error: NullResult".

    /// `eagle3-gpt-oss-20b-BF16.gguf`, from `ggml-org/gpt-oss-20b-GGUF`.
    ///
    /// 1.72 GB, 16 tensors, one block. It is the speculative-decoding draft head
    /// for gpt-oss-20b, not gpt-oss-20b — but `general.type` says `model` and
    /// `general.name` says `gpt-oss-20b`, so nothing about its identity is
    /// available except the target it declares.
    fn eagle3_draft_entries() -> Vec<Vec<u8>> {
        vec![
            kv_str("general.architecture", "eagle3"),
            kv_str("general.type", "model"),
            kv_str("general.name", "gpt-oss-20b"),
            kv_u32("eagle3.block_count", 1),
            kv_u32("eagle3.embedding_length", 2880),
            kv_u32("eagle3.attention.head_count", 64),
            kv_u32("eagle3.attention.head_count_kv", 8),
            kv_u32("eagle3.target_hidden_size", 2880),
            // An array, so its value is skipped during parsing — the presence of
            // the key is the whole signal, which is why keys are tracked apart
            // from decoded values.
            kv_string_array("eagle3.target_layers", &["2", "12", "21"]),
        ]
    }

    #[test]
    fn an_eagle3_draft_is_not_a_model() {
        let meta = parse_gguf_metadata(&mut header(eagle3_draft_entries())).unwrap();

        assert!(
            matches!(meta.role, GgufRole::Auxiliary { .. }),
            "got {:?}",
            meta.role
        );
        let refusal = meta.role.refusal(&meta.architecture).expect("must refuse");
        assert!(refusal.contains("eagle3"), "names the architecture: {refusal}");
        assert!(refusal.contains("separately"), "says what to do instead: {refusal}");
    }

    /// The signal has to survive an array-valued declaration on its own, since
    /// some converters write only `target_layers`.
    #[test]
    fn a_target_declared_only_as_an_array_is_still_detected() {
        let meta = parse_gguf_metadata(&mut header(vec![
            kv_str("general.architecture", "eagle3"),
            kv_u32("eagle3.block_count", 1),
            kv_string_array("eagle3.target_layers", &["2", "12", "21"]),
        ]))
        .unwrap();

        assert!(matches!(meta.role, GgufRole::Auxiliary { .. }), "got {:?}", meta.role);
    }

    /// The evidence is quoted into the message, so a future side-car nobody has
    /// seen still produces an explanation rather than a shrug.
    #[test]
    fn the_refusal_quotes_the_evidence_it_was_based_on() {
        let meta = parse_gguf_metadata(&mut header(eagle3_draft_entries())).unwrap();
        let refusal = meta.role.refusal(&meta.architecture).unwrap();

        assert!(refusal.contains("target_hidden_size"), "got: {refusal}");
    }

    /// `adapter.gguf` from an installed LoRA: architecture `gemma4`,
    /// `general.type = adapter`, and no `block_count` at all.
    #[test]
    fn a_lora_adapter_is_recognised_even_without_a_block_count() {
        let meta = parse_gguf_metadata(&mut header(vec![
            kv_str("general.architecture", "gemma4"),
            kv_str("general.type", "adapter"),
        ]))
        .expect("a missing block_count must not turn into a parse error here");

        assert_eq!(meta.role, GgufRole::Adapter);
        let refusal = meta.role.refusal(&meta.architecture).unwrap();
        assert!(refusal.contains("base model"), "points at the fix: {refusal}");
    }

    /// A vision projector, the other side-car that ships beside real models.
    #[test]
    fn a_vision_projector_is_not_a_model() {
        let meta = parse_gguf_metadata(&mut header(vec![
            kv_str("general.architecture", "clip"),
            kv_u32("clip.block_count", 27),
            kv_u32("clip.vision.embedding_length", 1152),
        ]))
        .unwrap();

        assert!(matches!(meta.role, GgufRole::Auxiliary { .. }), "got {:?}", meta.role);
    }

    /// The half that must not regress: every real model still loads. A false
    /// positive here bricks the app for models that work today.
    #[test]
    fn real_models_are_never_refused() {
        // gpt-oss-20b, the MoE this whole path exists for.
        let moe = parse_gguf_metadata(&mut header(gpt_oss_entries())).unwrap();
        assert_eq!(moe.role, GgufRole::Model);
        assert!(moe.role.refusal(&moe.architecture).is_none());
        assert!(moe.is_moe(), "and it is still recognised as MoE");

        // A dense model, declaring its type explicitly as real converters do.
        let dense = parse_gguf_metadata(&mut header(vec![
            kv_str("general.architecture", "qwen2"),
            kv_str("general.type", "model"),
            kv_u32("qwen2.block_count", 28),
            kv_u32("qwen2.embedding_length", 3584),
            kv_u32("qwen2.attention.head_count", 28),
            kv_u32("qwen2.attention.head_count_kv", 4),
        ]))
        .unwrap();
        assert_eq!(dense.role, GgufRole::Model);
        assert!(dense.role.refusal(&dense.architecture).is_none());
    }

    /// A model that merely mentions a target-shaped word in its *name* is a
    /// model. Only a declared key counts, or the check becomes the name-matching
    /// it was written to replace.
    #[test]
    fn a_models_name_never_decides_its_role() {
        let meta = parse_gguf_metadata(&mut header(vec![
            kv_str("general.architecture", "llama"),
            kv_str("general.name", "Draft Speculator Eagle3 Chat"),
            kv_u32("llama.block_count", 32),
        ]))
        .unwrap();

        assert_eq!(meta.role, GgufRole::Model, "the name is not evidence");
    }
}
