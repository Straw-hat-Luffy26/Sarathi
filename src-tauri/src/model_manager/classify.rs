//! What an installed model is, read from the file rather than from its name.
//!
//! Storage used to be a flat list, and everything it displayed came from the
//! manifest — which records what was *requested* at download time. That is how
//! an EAGLE-3 draft came to sit in the list as "gpt oss 20b · BF16": the
//! manifest faithfully reported a request that had fetched the wrong file.
//!
//! Here the GGUF header is the source. It says which architecture the weights
//! are, whether there are routed experts, whether there is an image tower, and
//! which quantization was actually written — so the shelf a model sits on and
//! the label under its name both describe the file on disk.
//!
//! ## One classifier, two screens
//!
//! The categories are [`ModelCategory`], the same enum Discover files models
//! under, produced by the same [`categorize`] call. A model must not be
//! "Vision" while browsing and "Other" once installed; sharing the function is
//! what makes that impossible rather than merely unlikely.
//!
//! Installed models get the better evidence of the two. Discover works from the
//! Hub's summary and a table of verified geometries; here the weights are on
//! disk, so `verified_moe` is a fact rather than a lookup.

use serde::{Deserialize, Serialize};

use crate::ai_engine::gguf_meta::{file_type_label, GgufMetadata, GgufRole};
use crate::model_providers::huggingface::card::{categorize, ModelCategory};

/// The shelf a model belongs on in Storage.
///
/// Deliberately coarse. These are groups a person scans past to find something,
/// not a taxonomy — the finer detail lives in [`Classification::categories`],
/// which is shared with Discover's filters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelGroup {
    /// Routed experts: only a fraction of the weights run per token, which is
    /// what allows a model far larger than the card to run at all.
    MixtureOfExperts,
    /// An ordinary transformer. Every weight participates in every token.
    Dense,
    /// Carries an image tower, so it can be shown pictures as well as text.
    Vision,
    /// Produces sentence vectors rather than replies. Cannot be chatted with,
    /// so it is grouped apart rather than listed beside models that can.
    Embedding,
    /// A readable GGUF that is not a standalone model — a LoRA adapter, a
    /// speculative-decoding draft, a projector. Listed so a file that is taking
    /// up disk space is never invisible, but never offered as something to load.
    Auxiliary,
    /// The header could not be read. Neither claimed to work nor assumed broken.
    Unknown,
}

impl ModelGroup {
    pub fn label(&self) -> &'static str {
        match self {
            Self::MixtureOfExperts => "Mixture of experts",
            Self::Dense => "Dense",
            Self::Vision => "Vision",
            Self::Embedding => "Embedding",
            Self::Auxiliary => "Helper files",
            Self::Unknown => "Unrecognised",
        }
    }

    /// One line explaining what the group means, for a heading that would
    /// otherwise be jargon.
    pub fn description(&self) -> &'static str {
        match self {
            Self::MixtureOfExperts => {
                "Only some of the weights run for each word, so these can be far larger than \
                 your graphics card and still be quick."
            }
            Self::Dense => "Every weight runs for every word. The ordinary kind.",
            Self::Vision => "These can be shown images as well as text.",
            Self::Embedding => {
                "These turn text into vectors for search and comparison. They do not hold \
                 conversations."
            }
            Self::Auxiliary => {
                "Files that support a model rather than being one. They cannot be loaded on \
                 their own."
            }
            Self::Unknown => "Sarathi could not read these files well enough to say.",
        }
    }

    /// Whether a model in this group can be loaded and talked to.
    pub fn is_loadable(&self) -> bool {
        matches!(self, Self::MixtureOfExperts | Self::Dense | Self::Vision)
    }

    /// Display order: the things you can chat with, then the rest.
    pub fn all() -> &'static [ModelGroup] {
        &[
            Self::MixtureOfExperts,
            Self::Dense,
            Self::Vision,
            Self::Embedding,
            Self::Auxiliary,
            Self::Unknown,
        ]
    }
}

/// Everything Storage needs to describe one installed file truthfully.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Classification {
    pub group: ModelGroup,
    /// Shared with Discover, so the same model files the same way on both.
    pub categories: Vec<ModelCategory>,
    /// GGUF `general.architecture`, e.g. `qwen3moe`, `gpt-oss`, `llama`.
    pub architecture: String,
    pub is_moe: bool,
    pub expert_count: u32,
    /// Experts consulted per token. The reason a large MoE stays responsive.
    pub expert_used_count: u32,
    pub block_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameter_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    /// Quantization as the file declares it, which is not always what the
    /// filename says. `None` when the header does not carry a known type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    /// Set when the file is not a standalone model, explaining why.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub not_loadable_reason: Option<String>,
}

impl Classification {
    /// What to say when the header could not be read at all.
    ///
    /// A file that cannot be classified is still on disk and still taking up
    /// space, so it is described as unknown rather than hidden.
    pub fn unreadable(reason: String) -> Self {
        Self {
            group: ModelGroup::Unknown,
            categories: Vec::new(),
            architecture: String::new(),
            is_moe: false,
            expert_count: 0,
            expert_used_count: 0,
            block_count: 0,
            parameter_count: None,
            context_length: None,
            quantization: None,
            not_loadable_reason: Some(reason),
        }
    }
}

/// Classifies a model from its header and its name.
///
/// `display_name` feeds only the shared [`categorize`] call, which is what keeps
/// Storage and Discover in agreement — the *group* is decided by the header
/// alone, so a model named "vision-chat" that has no image tower is filed as
/// dense.
pub fn classify(meta: &GgufMetadata, display_name: &str) -> Classification {
    // A file that is not a model has no business being grouped as one, whatever
    // its architecture or geometry might otherwise suggest.
    if let Some(reason) = meta.role.refusal(&meta.architecture) {
        return Classification {
            group: ModelGroup::Auxiliary,
            categories: auxiliary_categories(&meta.role),
            architecture: meta.architecture.clone(),
            is_moe: false,
            expert_count: 0,
            expert_used_count: 0,
            block_count: meta.block_count,
            parameter_count: meta.parameter_count,
            context_length: (meta.context_length > 0).then_some(meta.context_length),
            quantization: meta.file_type.and_then(file_type_label).map(str::to_string),
            not_loadable_reason: Some(reason),
        };
    }

    // The same call Discover makes, so a model is filed identically on both
    // screens. The architecture string is passed as well as the name, which is
    // what lets `qwen3vl` be recognised without anyone listing model names.
    let mut categories = categorize(
        display_name,
        &[],
        meta.parameter_count.unwrap_or(0),
        meta.context_length,
        &meta.architecture,
        meta.is_moe(),
    );

    // A multimodal model does not always carry its image tower in the same file:
    // llama.cpp keeps the projector in a separate `mmproj` GGUF, so
    // `Qwen3-VL-8B-Instruct-Q4_K_M.gguf` declares no vision keys at all.
    //
    // Its *architecture* still does. `qwen3vl`, `llava`, `gemma3vision` — the
    // architecture id is llama.cpp's own identifier for the model family, chosen
    // by the converter rather than by whoever named the repository, which makes
    // it a property of the weights in a way a display name is not.
    //
    // This has to be checked separately from the shared categoriser, whose
    // vision test looks for `-vl` with a hyphen and therefore matches a filename
    // like `Qwen3-VL-8B` while missing the display name "Qwen3 VL 8B Instruct"
    // that Storage actually holds.
    let arch = meta.architecture.to_ascii_lowercase();
    let arch_is_multimodal = arch.ends_with("vl")
        || arch.ends_with("vlm")
        || arch.contains("vision")
        || arch.contains("llava")
        || arch.contains("clip");

    let looks_multimodal =
        meta.has_vision || arch_is_multimodal || categories.contains(&ModelCategory::Vision);

    // The shelf and the category are two views of one decision, so whichever
    // signal fired, both must reflect it. Without this a model can sit on the
    // Vision shelf while Discover's Vision filter passes it over.
    if looks_multimodal && !categories.contains(&ModelCategory::Vision) {
        categories.push(ModelCategory::Vision);
    }

    // Order matters. An embedding model is checked first because it cannot chat
    // at all, which outranks anything else true of it; vision before MoE because
    // a multimodal mixture-of-experts is more usefully found under the thing
    // that changes how you use it.
    let group = if meta.has_pooling {
        ModelGroup::Embedding
    } else if looks_multimodal {
        ModelGroup::Vision
    } else if meta.is_moe() {
        ModelGroup::MixtureOfExperts
    } else {
        ModelGroup::Dense
    };

    Classification {
        group,
        categories,
        architecture: meta.architecture.clone(),
        is_moe: meta.is_moe(),
        expert_count: meta.expert_count,
        expert_used_count: meta.expert_used_count,
        block_count: meta.block_count,
        parameter_count: meta.parameter_count,
        context_length: (meta.context_length > 0).then_some(meta.context_length),
        quantization: meta.file_type.and_then(file_type_label).map(str::to_string),
        not_loadable_reason: None,
    }
}

fn auxiliary_categories(role: &GgufRole) -> Vec<ModelCategory> {
    match role {
        GgufRole::Adapter => vec![ModelCategory::LoraAdapter],
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(architecture: &str) -> GgufMetadata {
        GgufMetadata {
            architecture: architecture.to_string(),
            role: GgufRole::Model,
            block_count: 32,
            embedding_length: 4096,
            expert_count: 0,
            expert_used_count: 0,
            expert_ff_length: 0,
            head_count_kv: 8,
            key_length: 128,
            value_length: 128,
            parameter_count: Some(8_000_000_000),
            context_length: 32768,
            has_vision: false,
            has_pooling: false,
            file_type: Some(15),
        }
    }

    #[test]
    fn an_ordinary_model_is_dense() {
        let c = classify(&meta("qwen2"), "Qwen2.5 Coder 7B");
        assert_eq!(c.group, ModelGroup::Dense);
        assert!(!c.is_moe);
        assert!(c.group.is_loadable());
    }

    /// Routed experts are read from the header, so this holds for any MoE
    /// architecture rather than the two the Hub table happens to know.
    #[test]
    fn routed_experts_make_a_model_moe_whatever_the_architecture_is_called() {
        let mut m = meta("some-future-moe");
        m.expert_count = 64;
        m.expert_used_count = 8;

        let c = classify(&m, "Something 40B A4B");

        assert_eq!(c.group, ModelGroup::MixtureOfExperts);
        assert_eq!(c.expert_count, 64);
        assert_eq!(c.expert_used_count, 8);
        assert!(c.categories.contains(&ModelCategory::MixtureOfExperts));
    }

    #[test]
    fn an_image_tower_makes_a_model_vision() {
        let mut m = meta("qwen3vl");
        m.has_vision = true;

        let c = classify(&m, "Qwen3 VL 8B Instruct");

        assert_eq!(c.group, ModelGroup::Vision);
        assert!(c.categories.contains(&ModelCategory::Vision));
    }

    /// The real Qwen3-VL case: llama.cpp keeps the projector in a separate
    /// `mmproj` file, so the model's own header declares no vision keys. It is
    /// still a vision model, and Discover already says so — Storage agreeing
    /// with it is the whole point of sharing the classifier.
    #[test]
    fn a_multimodal_model_is_shelved_as_vision_even_with_its_projector_elsewhere() {
        let m = meta("qwen3vl");
        assert!(!m.has_vision, "the fixture reproduces the real header");

        // The display name Storage actually holds — spaced, not hyphenated. An
        // earlier version passed the *filename* here, which contains `-VL` and
        // so matched the shared heuristic; that made the test pass while the
        // application shelved the model as dense.
        let c = classify(&m, "Qwen3 VL 8B Instruct");

        assert_eq!(c.group, ModelGroup::Vision);
        assert!(c.categories.contains(&ModelCategory::Vision));
    }

    /// The architecture is the signal, so a model with an unremarkable name is
    /// still shelved correctly — and one whose name merely mentions vision is
    /// not, unless something about the weights says so.
    #[test]
    fn the_architecture_decides_multimodality_not_the_display_name() {
        assert_eq!(classify(&meta("qwen3vl"), "My Fine Tune").group, ModelGroup::Vision);
        assert_eq!(classify(&meta("llava"), "assistant v2").group, ModelGroup::Vision);

        // `llama` is not a vision architecture and no tower is declared.
        assert_eq!(classify(&meta("llama"), "Night Chat 8B").group, ModelGroup::Dense);
    }

    /// The two screens must never disagree. Whatever Discover's shared
    /// categoriser calls a vision model, Storage shelves as one.
    #[test]
    fn the_vision_shelf_and_the_vision_category_never_disagree() {
        for (arch, name) in [
            ("qwen3vl", "Qwen3 VL 8B"),
            ("llama", "Llava 7B"),
            ("gemma3", "Gemma 3 Vision"),
            ("qwen2", "Qwen2.5 Coder 7B"),
        ] {
            let c = classify(&meta(arch), name);
            assert_eq!(
                c.group == ModelGroup::Vision,
                c.categories.contains(&ModelCategory::Vision),
                "{name} was shelved {:?} but categorised {:?}",
                c.group,
                c.categories
            );
        }
    }

    /// Only a model whose output is an embedding declares how it pools.
    #[test]
    fn a_pooling_strategy_marks_an_embedding_model() {
        let mut m = meta("bert");
        m.has_pooling = true;

        let c = classify(&m, "nomic embed text v1.5");

        assert_eq!(c.group, ModelGroup::Embedding);
        assert!(!c.group.is_loadable(), "an embedding model cannot be chatted with");
    }

    #[test]
    fn a_lora_adapter_is_never_grouped_as_a_model() {
        let mut m = meta("gemma4");
        m.role = GgufRole::Adapter;

        let c = classify(&m, "some coding lora");

        assert_eq!(c.group, ModelGroup::Auxiliary);
        assert!(!c.group.is_loadable());
        assert!(c.not_loadable_reason.is_some());
        assert!(c.categories.contains(&ModelCategory::LoraAdapter));
    }

    /// The file that started all of this. It must land on the helper shelf, not
    /// beside the model whose name it borrows.
    #[test]
    fn a_speculative_decoding_draft_is_grouped_as_a_helper() {
        let mut m = meta("eagle3");
        m.role = GgufRole::Auxiliary { evidence: "it declares a target model".into() };
        m.block_count = 1;

        let c = classify(&m, "gpt oss 20b");

        assert_eq!(c.group, ModelGroup::Auxiliary);
        assert!(!c.group.is_loadable());
        assert!(c.not_loadable_reason.unwrap().contains("helper module"));
    }

    /// Storage must show the quantization the file was written with. The
    /// manifest records what was *asked for*, which is how "BF16" ended up
    /// under a file that is nothing of the sort.
    #[test]
    fn quantization_is_read_from_the_header_not_the_filename() {
        let mut m = meta("gpt-oss");
        m.file_type = Some(38);
        assert_eq!(classify(&m, "gpt oss 20b").quantization.as_deref(), Some("MXFP4"));

        // An unknown type is left unstated rather than guessed at.
        m.file_type = Some(9999);
        assert!(classify(&m, "gpt oss 20b").quantization.is_none());
    }

    #[test]
    fn an_unreadable_file_is_neither_loadable_nor_hidden() {
        let c = Classification::unreadable("header truncated".into());
        assert_eq!(c.group, ModelGroup::Unknown);
        assert!(!c.group.is_loadable());
        assert!(c.not_loadable_reason.is_some());
    }

    /// Every group needs a heading and a sentence, or the UI has nothing to
    /// print for one that is added later.
    #[test]
    fn every_group_can_describe_itself() {
        for g in ModelGroup::all() {
            assert!(!g.label().is_empty(), "{g:?}");
            assert!(!g.description().is_empty(), "{g:?}");
        }
    }
}
