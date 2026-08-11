//! Reading a remote GGUF's header before committing to the download.
//!
//! Everything upstream of this works on names and sizes: the Hub's file listing
//! gives a path and a byte count, and a quantization label is parsed out of the
//! filename. That is enough to *offer* a download and not enough to be sure what
//! is being offered. `ggml-org/gpt-oss-20b-GGUF` ships the model beside an
//! EAGLE-3 speculative-decoding draft, and the draft's name parses as cleanly as
//! the model's does.
//!
//! A GGUF's header says what the file is, and it sits at the front of the file.
//! HTTP range requests mean it can be read without the other twelve gigabytes —
//! typically a couple of megabytes, against a download that would otherwise be
//! wasted in full before anyone found out.
//!
//! This is the authority. The filename rules upstream are a cheap pre-filter
//! that keep obvious side-cars out of the listing; this decides.

use anyhow::{anyhow, bail, Context, Result};

use crate::ai_engine::gguf_meta::{parse_gguf_metadata, GgufMetadata, GgufRole};

/// First slice fetched. Enough for the header of most models.
///
/// The header's size is dominated by the tokenizer vocabulary, which is stored
/// as a string array — 32k–200k entries. Small vocabularies fit well inside
/// this; large ones need a second request, which is the trade being made:
/// one small request usually, rather than one large request always.
const FIRST_CHUNK: u64 = 2 << 20;

/// Ceiling on how much of a file is read to find the end of its header.
///
/// A 200k-token vocabulary runs to roughly 4 MB. Past this the file is not
/// shaped like anything Sarathi can use, and reading further to prove it would
/// cost more than the download it is trying to avoid.
const MAX_CHUNK: u64 = 32 << 20;

/// Attempts made when a header read fails for a reason that may not repeat.
///
/// A 12 GB download hangs on this answer, so a slow link is worth waiting out.
const ATTEMPTS: u32 = 3;

/// Per-request budget.
///
/// Generous on purpose: two megabytes over a slow connection can take a while,
/// and the alternative to waiting is downloading the whole file to find out.
const REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(90);

/// What the file turned out to be.
#[derive(Debug, Clone)]
pub struct GgufProbe {
    pub filename: String,
    /// Bytes actually read to reach a verdict, for the log.
    pub header_bytes_read: u64,
    pub metadata: GgufMetadata,
}

/// The outcome of choosing a file from a repository.
#[derive(Debug, Clone)]
pub enum Selection {
    /// The file's own header was read and says it is a model.
    Verified(GgufProbe),
    /// No header could be read — the Hub was unreachable or too slow.
    ///
    /// The best-ranked candidate is named anyway rather than failing the
    /// download outright, because being offline is not evidence that a file is
    /// bad and `ai_engine::gguf_meta` refuses a helper at load time regardless.
    /// What this must never become is a *different* file than the one asked for:
    /// silently substituting one is the bug this whole path exists to prevent.
    Unverified { filename: String, reason: String },
}

impl GgufProbe {
    /// Whether this file can be loaded and talked to on its own.
    pub fn is_standalone_model(&self) -> bool {
        self.metadata.role == GgufRole::Model
    }

    /// Why it cannot be, when it cannot.
    pub fn refusal(&self) -> Option<String> {
        self.metadata.role.refusal(&self.metadata.architecture)
    }
}

fn resolve_url(repo_id: &str, filename: &str) -> String {
    format!("https://huggingface.co/{repo_id}/resolve/main/{filename}")
}

/// Fetches bytes `from..=to` of a URL.
///
/// A server that ignores `Range` and sends the whole file would be a
/// twelve-gigabyte surprise, so a 200 response is refused outright rather than
/// read. HuggingFace honours ranges on `resolve/` URLs; this guards against a
/// proxy that does not.
async fn fetch_range(
    client: &reqwest::Client,
    url: &str,
    from: u64,
    to: u64,
    token: Option<&str>,
) -> Result<Vec<u8>> {
    let mut req = client.get(url).header(reqwest::header::RANGE, format!("bytes={from}-{to}"));
    if let Some(t) = token.map(str::trim).filter(|t| !t.is_empty()) {
        req = req.bearer_auth(t);
    }

    let resp = req.send().await.context("could not reach HuggingFace")?;
    let status = resp.status();

    if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
        bail!("HuggingFace rate-limited the metadata check; try again shortly");
    }
    if status == reqwest::StatusCode::OK {
        bail!("the server ignored the range request, so the header cannot be read cheaply");
    }
    if status != reqwest::StatusCode::PARTIAL_CONTENT {
        bail!("HuggingFace returned {status} for the header of {url}");
    }

    Ok(resp.bytes().await.context("could not read the header")?.to_vec())
}

/// Reads a remote GGUF's header and reports what the file is.
///
/// Grows the read only when the header proves longer than the slice fetched, so
/// the common case costs one small request.
pub async fn probe_remote(repo_id: &str, filename: &str, token: Option<&str>) -> Result<GgufProbe> {
    let mut last = None;
    for attempt in 1..=ATTEMPTS {
        match probe_once(repo_id, filename, token).await {
            Ok(probe) => return Ok(probe),
            Err(e) => {
                log::debug!("[HF_PROBE] {filename} attempt {attempt}/{ATTEMPTS} failed: {e:#}");
                last = Some(e);
            }
        }
    }
    Err(last.unwrap_or_else(|| anyhow!("could not read {filename}")))
}

async fn probe_once(repo_id: &str, filename: &str, token: Option<&str>) -> Result<GgufProbe> {
    let url = resolve_url(repo_id, filename);
    let client = reqwest::Client::builder()
        .user_agent("Sarathi/0.1.0")
        .timeout(REQUEST_TIMEOUT)
        .build()?;

    // Accumulated, not re-fetched. A header that needs 16 MB used to cost
    // 2+4+8+16 MB because each attempt restarted from byte zero; appending makes
    // the total equal to whatever the header actually is. That matters most for
    // exactly the models this check is for — a large mixture-of-experts model
    // has a 200k-token vocabulary, and the vocabulary is most of the header.
    let mut buffer: Vec<u8> = Vec::with_capacity(FIRST_CHUNK as usize);
    let mut want = FIRST_CHUNK;
    let mut last_err = None;

    while want <= MAX_CHUNK {
        let have = buffer.len() as u64;
        let chunk = fetch_range(&client, &url, have, want - 1, token).await?;
        let short = (chunk.len() as u64) < want - have;
        buffer.extend_from_slice(&chunk);

        let read = buffer.len() as u64;
        let mut cursor = std::io::Cursor::new(&buffer);

        match parse_gguf_metadata(&mut cursor) {
            Ok(metadata) => {
                log::info!(
                    "[HF_PROBE] {filename}: architecture '{}', role {:?}, {} layers, {} experts \
                     (header read in {} KB)",
                    metadata.architecture,
                    metadata.role,
                    metadata.block_count,
                    metadata.expert_count,
                    read / 1024
                );
                return Ok(GgufProbe {
                    filename: filename.to_string(),
                    header_bytes_read: read,
                    metadata,
                });
            }
            Err(e) => {
                // The server returned less than asked for, so the whole file is
                // already in hand. The parse failure is real rather than a
                // truncation, and reading further is not possible.
                if short {
                    return Err(e.context(format!("{filename} is not a readable GGUF")));
                }
                last_err = Some(e);
                want *= 2;
            }
        }
    }

    Err(last_err
        .unwrap_or_else(|| anyhow!("header not found"))
        .context(format!("{filename}'s header exceeds {} MB, which no usable model has", MAX_CHUNK >> 20)))
}

/// Picks the file in a repository that is actually the model.
///
/// `candidates` is the shortlist the caller assembled from names and sizes, best
/// guess first. Each is probed in turn and the first standalone model wins, so a
/// correct guess costs one request and a wrong one costs one more.
///
/// Returns the probe of the chosen file, or an error naming what every candidate
/// turned out to be — which is the message that would have explained the
/// original failure: not "NullResult" but "the file offered is a
/// speculative-decoding draft".
pub async fn select_model_file(
    repo_id: &str,
    candidates: &[String],
    token: Option<&str>,
) -> Result<Selection> {
    if candidates.is_empty() {
        bail!("no GGUF files to choose from in {repo_id}");
    }

    // Kept apart, because they mean opposite things. "This is a draft model" is
    // a fact about the file and settles it forever. "The request timed out" is a
    // fact about the network and settles nothing — treating the two alike is how
    // a slow connection turned a request for Q4_K_M into a Q5_K_M download.
    let mut refused: Vec<String> = Vec::new();
    let mut unreachable: Vec<String> = Vec::new();

    for filename in candidates {
        match probe_remote(repo_id, filename, token).await {
            Ok(probe) if probe.is_standalone_model() => return Ok(Selection::Verified(probe)),
            Ok(probe) => {
                let why = probe.refusal().unwrap_or_else(|| "not a standalone model".into());
                log::info!("[HF_PROBE] Skipping {filename}: {why}");
                refused.push(format!("{filename} is {}", short_reason(&probe)));
            }
            Err(e) => {
                log::warn!("[HF_PROBE] Could not read {filename}'s header: {e:#}");
                unreachable.push(filename.clone());
            }
        }
    }

    // Every candidate was read, and none was a model. This is conclusive, and
    // starting a multi-gigabyte download of something that cannot load is
    // exactly what the check exists to prevent.
    if unreachable.is_empty() {
        bail!(
            "No standalone model found in {repo_id}. {}. Helper modules such as \
             speculative-decoding drafts, vision projectors and LoRA adapters need the model they \
             were built for, which is published separately.",
            refused.join("; ")
        );
    }

    // Some file could not be reached. Fall back to the best-ranked candidate —
    // the one the caller asked for — rather than to whichever file happened to
    // answer. Downloading a different build than was requested is a worse
    // outcome than downloading an unverified one, because the second is caught
    // at load time and the first is not caught at all.
    let preferred = candidates
        .iter()
        .find(|c| unreachable.contains(c))
        .cloned()
        .unwrap_or_else(|| candidates[0].clone());

    Ok(Selection::Unverified {
        filename: preferred,
        reason: format!(
            "HuggingFace could not be reached to check {} file header(s) before downloading",
            unreachable.len()
        ),
    })
}

fn short_reason(probe: &GgufProbe) -> String {
    match &probe.metadata.role {
        GgufRole::Model => "a model".to_string(),
        GgufRole::Adapter => format!("a LoRA adapter for {}", probe.metadata.architecture),
        GgufRole::Auxiliary { .. } => {
            format!("a '{}' helper module", probe.metadata.architecture)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The URL shape the probe reads from. Getting this wrong turns every probe
    /// into a 404 and every download into an unchecked one.
    #[test]
    fn header_urls_point_at_the_raw_file() {
        assert_eq!(
            resolve_url("ggml-org/gpt-oss-20b-GGUF", "gpt-oss-20b-MXFP4.gguf"),
            "https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-MXFP4.gguf"
        );
    }

    #[test]
    fn an_empty_repository_is_an_error_rather_than_a_panic() {
        let err = tauri::async_runtime::block_on(select_model_file("a/b", &[], None)).unwrap_err();
        assert!(err.to_string().contains("no GGUF files"), "got: {err}");
    }

    /// The growth ceiling has to leave room for a real vocabulary. A 200k-token
    /// tokenizer is about 4 MB of strings, so a limit below that would reject
    /// models it should accept.
    #[test]
    fn the_header_budget_covers_a_large_tokenizer() {
        assert!(FIRST_CHUNK >= 1 << 20);
        assert!(MAX_CHUNK >= 8 << 20, "a 200k-token vocabulary needs several MB");
        assert!(MAX_CHUNK > FIRST_CHUNK);
    }
}
