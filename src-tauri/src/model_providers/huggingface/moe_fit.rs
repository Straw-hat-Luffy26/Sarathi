//! Can this machine actually run this MoE model, before downloading it?
//!
//! A Mixture-of-Experts model is placed by *tensor*, not by layer: routed
//! experts move to system RAM while attention, the KV cache, the router and the
//! shared experts stay on the card. That is what makes a 12.8 GB model usable on
//! an 8 GB one — and it is invisible to the browse listing, which sizes every
//! quantization against the VRAM weight budget and therefore marks all of
//! gpt-oss-20b "too large" on hardware that runs it perfectly well.
//!
//! This module closes that gap by asking [`vram_planner::plan_moe_offload`] —
//! the same planner the loader uses at load time — whether a plan exists for
//! this machine. Nothing here is a specification lookup: the answer is computed
//! from the GPU's real VRAM, the system's real usable RAM, and the actual size
//! of the file that would be downloaded. Change the hardware and the answer
//! changes.
//!
//! Two deliberate limits:
//!
//! - Only architectures with a **verified** geometry in
//!   [`moe_geometry`](super::moe_geometry) are considered. The Hub's GGUF
//!   metadata does not expose expert counts, and a model marked runnable on
//!   invented numbers is worse than one left unmarked.
//! - The planner is asked about the *exact* file, so a repository's Q4 may be
//!   offloadable here while its Q8 is not. Marking the repository as a whole
//!   would tell someone a download works when the one they press does not.

use crate::ai_engine::vram_planner::{self, MoeGeometry, MoeOffloadPlan};
use crate::model_providers::huggingface::discovery::GgufRepo;
use crate::model_providers::huggingface::moe_geometry::{self, KnownMoe};
use crate::model_recommendation::estimator::moe_expert_fraction;

/// Context the listing plans against.
///
/// Browsing has no session to ask, so placement is priced at a typical working
/// context rather than a model's advertised maximum — the same figure the VRAM
/// weight budget in [`crate::commands::catalog`] uses, so the two columns of the
/// same table cannot be answering different questions.
pub const BROWSE_CONTEXT: u32 = 8192;

/// Bytes per element in llama.cpp's KV cache. FP16, regardless of how the
/// weights are quantized.
const KV_BYTES_PER_ELEMENT: u64 = 2;

/// What this machine can offer a model, read once per sweep.
///
/// Both figures come from the live hardware profile. Zero means "not detected",
/// which every function here treats as "cannot promise anything" rather than as
/// a permissive default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HostCapacity {
    /// Total VRAM of the card the runtime would actually load onto.
    ///
    /// Total rather than free, for the reason the browse weight budget gives:
    /// a listing that shifts as other processes take memory would be unreadable.
    /// It is also the conservative direction here — the loader plans against
    /// free VRAM when it knows it, which is never larger.
    pub vram_total_bytes: u64,
    /// System RAM available to hold offloaded experts, after the OS reserve.
    ///
    /// This is usually the binding constraint rather than VRAM: a 21B model's
    /// experts are ~12 GB, which a 16 GB laptop cannot hold however well the
    /// card is planned.
    pub usable_ram_bytes: u64,
}

impl HostCapacity {
    /// True when there is enough of a hardware picture to judge placement.
    pub fn is_known(&self) -> bool {
        self.vram_total_bytes > 0 && self.usable_ram_bytes > 0
    }
}

/// A MoE placement this machine can actually execute.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExpertOffload {
    /// Routed experts of this many leading layers move to system RAM.
    /// Zero means the model fits in VRAM outright and offload is unnecessary.
    pub cpu_moe_layers: u32,
    pub total_layers: u32,
    /// System RAM the offloaded experts occupy.
    pub host_bytes: u64,
    /// Parameters actually used per token — why a model this size is quick
    /// despite living partly in system RAM.
    pub active_parameters: u64,
    /// One line for the UI, in the terms someone choosing a download needs.
    pub note: String,
}

/// Reads a verified MoE geometry for a repository, if it has one.
///
/// Returns `None` for dense models and for MoE architectures whose expert
/// counts have not been verified — the caller must treat that as "unknown",
/// never as "dense".
pub fn known_geometry(repo: &GgufRepo) -> Option<&'static KnownMoe> {
    let gguf = repo.gguf.as_ref()?;
    moe_geometry::lookup(&gguf.architecture, gguf.total_parameters)
}

/// Builds the planner's input for one quantization of a known MoE model.
///
/// `file_bytes` is the size of the specific GGUF that would be downloaded, so
/// the expert share is priced against the real download rather than against the
/// unquantized checkpoint.
fn geometry_for(known: &KnownMoe, file_bytes: u64) -> Option<MoeGeometry> {
    let fraction = moe_expert_fraction(
        known.num_experts,
        known.active_experts,
        known.total_params,
        known.active_params,
    )?;

    // Exact, not the size-banded estimate: that bands on file size, and a MoE
    // file's size is dominated by experts rather than by attention — which is
    // precisely the part the KV cache is proportional to.
    let kv_bytes_per_token = 2
        * u64::from(known.num_layers)
        * u64::from(known.num_kv_heads)
        * u64::from(known.head_dimension)
        * KV_BYTES_PER_ELEMENT;

    Some(MoeGeometry {
        total_layers: known.num_layers,
        expert_bytes: (file_bytes as f64 * fraction) as u64,
        kv_bytes_per_token,
        active_params: known.active_params,
    })
}

/// Plans expert offload for one quantization, or explains why there is none.
///
/// Returns `Ok` with a placement this machine can execute, or `Err` with the
/// planner's reason — which names whether RAM or VRAM was the blocker, and is
/// worth showing rather than reducing to "too large".
pub fn plan_for(
    repo_id: &str,
    known: &KnownMoe,
    file_bytes: u64,
    host: HostCapacity,
) -> Result<ExpertOffload, String> {
    if !host.is_known() {
        return Err("This machine's memory could not be read, so placement is unknown".into());
    }
    let Some(geom) = geometry_for(known, file_bytes) else {
        return Err(format!("{repo_id} has no usable expert geometry"));
    };

    let plan = vram_planner::plan_moe_offload(
        repo_id,
        host.vram_total_bytes,
        host.usable_ram_bytes,
        file_bytes,
        BROWSE_CONTEXT,
        &geom,
    );

    if !plan.fits {
        return Err(plan.reason);
    }

    let per_layer = geom.expert_bytes / u64::from(geom.total_layers.max(1));
    Ok(ExpertOffload {
        cpu_moe_layers: plan.cpu_moe_layers,
        total_layers: geom.total_layers,
        host_bytes: u64::from(plan.cpu_moe_layers) * per_layer,
        active_parameters: geom.active_params,
        note: note_for(&plan, geom.total_layers, geom.active_params, per_layer),
    })
}

/// The one-line explanation shown beside a size.
///
/// The planner's own `reason` is an audit trail — it names byte budgets and
/// reserves, which is right for a log and wrong for a table cell. This says the
/// thing a person choosing a download needs: whether it runs, what it costs in
/// system memory, and why it is not as slow as its size suggests.
fn note_for(
    plan: &MoeOffloadPlan,
    total_layers: u32,
    active_params: u64,
    per_layer_bytes: u64,
) -> String {
    if plan.cpu_moe_layers == 0 {
        return format!(
            "Fits in VRAM as it is — {} active per token, no offload needed",
            as_billions(active_params)
        );
    }

    format!(
        "Runs by keeping the experts of {} of {} layers in system memory ({:.1} GB). \
         Only {} parameters are used per token, so it stays responsive.",
        plan.cpu_moe_layers,
        total_layers,
        as_gb(u64::from(plan.cpu_moe_layers) * per_layer_bytes),
        as_billions(active_params)
    )
}

fn as_billions(params: u64) -> String {
    if params == 0 {
        return "an unknown number of".to_string();
    }
    format!("{:.1}B", params as f64 / 1e9)
}

fn as_gb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_providers::huggingface::discovery::{GgufMeta, Quantization};

    const GB: u64 = 1024 * 1024 * 1024;
    /// gpt-oss-20b's MXFP4 build.
    const GPT_OSS_BYTES: u64 = 12_800_000_000;

    fn gpt_oss() -> &'static KnownMoe {
        moe_geometry::lookup("gpt-oss", 20_900_000_000).expect("verified entry")
    }

    fn repo(arch: &str, params: u64) -> GgufRepo {
        GgufRepo {
            repo_id: "unsloth/gpt-oss-20b-GGUF".into(),
            author: "unsloth".into(),
            downloads: 1000,
            likes: 10,
            last_modified: String::new(),
            quantizations: vec![Quantization {
                label: "MXFP4".into(),
                filename: "m.gguf".into(),
                size_bytes: GPT_OSS_BYTES,
                is_sharded: false,
            }],
            gguf: Some(GgufMeta {
                total_parameters: params,
                architecture: arch.into(),
                context_length: 131072,
                chat_template: None,
                bos_token: None,
                eos_token: None,
            }),
            base_model: None,
            is_finetune: false,
            is_lora_adapter: false,
            tags: vec![],
        }
    }

    fn host(vram: u64, ram: u64) -> HostCapacity {
        HostCapacity { vram_total_bytes: vram, usable_ram_bytes: ram }
    }

    #[test]
    fn a_verified_moe_repo_is_recognised() {
        assert!(known_geometry(&repo("gpt-oss", 20_900_000_000)).is_some());
    }

    #[test]
    fn a_dense_repo_has_no_geometry() {
        assert!(known_geometry(&repo("llama", 8_000_000_000)).is_none());
    }

    /// An unverified MoE architecture must not be sized on invented counts —
    /// the whole point of keeping the table small.
    #[test]
    fn an_unverified_moe_architecture_is_not_classified() {
        assert!(known_geometry(&repo("deepseek2", 236_000_000_000)).is_none());
    }

    /// The headline case: a 12.8 GB model that the VRAM-only budget calls "too
    /// large" on every card this app targets, and which in fact runs on all of
    /// them given the RAM for its experts.
    #[test]
    fn a_21b_moe_runs_on_a_4gb_card_with_enough_system_ram() {
        let plan = plan_for("gpt-oss-20b", gpt_oss(), GPT_OSS_BYTES, host(4 * GB, 24 * GB))
            .expect("should be offloadable");

        assert!(plan.cpu_moe_layers > 0, "some experts must move to make this fit");
        assert!(plan.cpu_moe_layers <= plan.total_layers);
        assert_eq!(plan.total_layers, 24);
        assert_eq!(plan.active_parameters, 3_600_000_000);
    }

    /// Classification is per machine, not per model. The same file on a bigger
    /// card keeps more experts resident.
    #[test]
    fn a_bigger_card_keeps_more_experts_resident() {
        let small = plan_for("m", gpt_oss(), GPT_OSS_BYTES, host(4 * GB, 24 * GB)).unwrap();
        let large = plan_for("m", gpt_oss(), GPT_OSS_BYTES, host(8 * GB, 24 * GB)).unwrap();

        assert!(
            large.cpu_moe_layers < small.cpu_moe_layers,
            "more VRAM must mean fewer experts on CPU ({} vs {})",
            large.cpu_moe_layers,
            small.cpu_moe_layers
        );
    }

    #[test]
    fn a_card_large_enough_needs_no_offload_at_all() {
        let plan = plan_for("m", gpt_oss(), GPT_OSS_BYTES, host(24 * GB, 32 * GB)).unwrap();

        assert_eq!(plan.cpu_moe_layers, 0);
        assert_eq!(plan.host_bytes, 0);
        assert!(plan.note.contains("no offload needed"), "note: {}", plan.note);
    }

    /// The refusal that matters most: a machine limited by RAM must be told so,
    /// because "buy a bigger GPU" would be the wrong conclusion.
    #[test]
    fn a_machine_without_ram_for_the_experts_is_refused_and_says_why() {
        let err = plan_for("m", gpt_oss(), GPT_OSS_BYTES, host(4 * GB, 6 * GB))
            .expect_err("6 GB cannot hold ~11 GB of experts");

        assert!(err.contains("system RAM"), "got: {err}");
        assert!(err.contains("limited by RAM"), "got: {err}");
    }

    /// Never claim a model runs on hardware that could not be read.
    #[test]
    fn unknown_hardware_promises_nothing() {
        assert!(plan_for("m", gpt_oss(), GPT_OSS_BYTES, host(0, 24 * GB)).is_err());
        assert!(plan_for("m", gpt_oss(), GPT_OSS_BYTES, host(4 * GB, 0)).is_err());
        assert!(plan_for("m", gpt_oss(), GPT_OSS_BYTES, HostCapacity::default()).is_err());
    }

    /// Classification is per quantization, so a repo's small build can be
    /// runnable while its large one is not.
    #[test]
    fn each_quantization_is_judged_on_its_own_size() {
        let tight = host(4 * GB, 14 * GB);

        let small = plan_for("m", gpt_oss(), 7_000_000_000, tight);
        let huge = plan_for("m", gpt_oss(), 42_000_000_000, tight);

        assert!(small.is_ok(), "a 7 GB build fits: {small:?}");
        assert!(huge.is_err(), "a 42 GB build cannot, and must not be offered");
    }

    /// The expert share has to come from the shared formula, or the listing and
    /// the loader will disagree about how much can leave the card.
    #[test]
    fn the_expert_share_matches_the_shared_formula() {
        let known = gpt_oss();
        let geom = geometry_for(known, GPT_OSS_BYTES).expect("splittable");

        let fraction = moe_expert_fraction(
            known.num_experts,
            known.active_experts,
            known.total_params,
            known.active_params,
        )
        .unwrap();

        assert_eq!(geom.expert_bytes, (GPT_OSS_BYTES as f64 * fraction) as u64);
        assert!(
            geom.expert_bytes > GPT_OSS_BYTES / 2,
            "experts dominate a MoE file; got {}",
            geom.expert_bytes
        );
    }

    /// The KV figure must be the exact one. Using the size-banded estimate bands
    /// on file size, which for a MoE is dominated by experts — it would price
    /// gpt-oss's 49 KB/token cache at 256 KB and refuse plans that do fit.
    #[test]
    fn kv_cost_is_computed_from_geometry_not_from_file_size() {
        let geom = geometry_for(gpt_oss(), GPT_OSS_BYTES).unwrap();

        // 2 × 24 layers × 8 KV heads × 64 dim × 2 bytes.
        assert_eq!(geom.kv_bytes_per_token, 49_152);
        assert!(
            geom.kv_bytes_per_token
                < vram_planner::estimate_kv_bytes_per_token(GPT_OSS_BYTES),
            "the banded estimate overprices a MoE's attention"
        );
    }

    #[test]
    fn the_note_says_what_it_costs_and_why_it_is_still_quick() {
        let plan = plan_for("m", gpt_oss(), GPT_OSS_BYTES, host(4 * GB, 24 * GB)).unwrap();

        assert!(plan.note.contains("system memory"), "note: {}", plan.note);
        assert!(plan.note.contains("3.6B"), "note: {}", plan.note);
        assert!(plan.note.contains("per token"), "note: {}", plan.note);
    }

    #[test]
    fn planning_never_panics_across_extremes() {
        for file in [0u64, 1, GPT_OSS_BYTES, u64::MAX] {
            for h in [host(0, 0), host(4 * GB, 24 * GB), host(u64::MAX, u64::MAX)] {
                let _ = plan_for("m", gpt_oss(), file, h);
            }
        }
    }
}
