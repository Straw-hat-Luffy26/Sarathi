//! Recognising publisher names in a search box.
//!
//! Typing "nvidia" is a question about a *publisher*, not a substring. The Hub's
//! `search=` parameter cannot tell the difference: it matches the text anywhere
//! in a repo id and orders by download count, so "nvidia" answers with
//! `nvidia/parakeet-ctc-1.1b` — a speech recogniser with 1.8 million downloads —
//! before it reaches a single model that can hold a conversation.
//!
//! The Hub does have an exact publisher filter, `author=`. Three things stop it
//! from being a drop-in replacement, and each one is why this table exists:
//!
//! 1. **`author=` is case-sensitive and takes the exact org slug.** `author=Qwen`
//!    returns 54 repositories; `author=qwen` returns zero. Nobody types the
//!    capitalisation, so the typed word has to be translated into the slug.
//!
//! 2. **The name people use is rarely the slug.** "deepseek" is `deepseek-ai`,
//!    "meta" is `meta-llama`, "z.ai" is `zai-org`, "kimi" is `moonshotai`. A
//!    brand is also often known by its *model* line rather than its company —
//!    people search "nemotron", "gemma", "phi", "granite", and mean the org.
//!
//! 3. **Some publishers ship no GGUF at all.** `author=deepseek-ai&filter=gguf`
//!    returns nothing, and so do `moonshotai` and `meta-llama` — those weights
//!    reach Sarathi only through quantizers like `unsloth` and `bartowski`. An
//!    org-only search for those brands would confidently report that DeepSeek
//!    has no models, which is worse than the ranking problem it set out to fix.
//!    [`Brand::self_publishes_gguf`] records which brands have that shape so the
//!    search can keep the free-text sweep that finds the community builds.
//!
//! Every slug below was verified against the live API rather than guessed, and
//! the two are easy to get wrong in opposite directions: `openai-community`
//! hosts the GPT-2 re-uploads while `openai` hosts gpt-oss, and `Falconsai` is
//! an unrelated account that outranks `tiiuae` on a "falcon" search.

/// A publisher Sarathi can recognise by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Brand {
    /// Display name, shown on cards and in the search hint.
    pub label: &'static str,
    /// Exact HuggingFace org slugs, cased as the Hub stores them.
    ///
    /// A list rather than one value because a few brands genuinely publish from
    /// more than one account.
    pub orgs: &'static [&'static str],
    /// What a person might type to mean this brand: the company, its model
    /// lines, and the spellings in common use.
    ///
    /// Compared after [`normalize`], so punctuation and case are not repeated
    /// here — `z.ai`, `Z AI`, and `zai` are all one entry.
    pub aliases: &'static [&'static str],
    /// Whether the org publishes GGUF builds itself.
    ///
    /// `false` means the brand's models exist in Sarathi only as community
    /// conversions, so an org-scoped query alone would return an empty list.
    pub self_publishes_gguf: bool,
}

/// Publishers worth recognising by name.
///
/// The bar for inclusion is that someone would plausibly type the name into the
/// search box expecting that publisher's models — not that the account is good.
/// Judging conversion quality is [`curation`](super::curation)'s job and stays
/// there; this table only decides what a word *refers to*.
pub const BRANDS: &[Brand] = &[
    // ── Labs that publish their own GGUF ────────────────────────────────
    Brand {
        label: "Qwen",
        orgs: &["Qwen"],
        aliases: &["qwen", "qwq", "tongyi", "alibaba"],
        self_publishes_gguf: true,
    },
    Brand {
        label: "Google",
        orgs: &["google"],
        aliases: &["google", "gemma", "codegemma", "deepmind"],
        self_publishes_gguf: true,
    },
    Brand {
        label: "Microsoft",
        orgs: &["microsoft"],
        aliases: &["microsoft", "phi", "bitnet"],
        self_publishes_gguf: true,
    },
    Brand {
        label: "NVIDIA",
        orgs: &["nvidia"],
        aliases: &["nvidia", "nemotron"],
        self_publishes_gguf: true,
    },
    Brand {
        label: "Mistral AI",
        orgs: &["mistralai"],
        aliases: &["mistral", "mistralai", "mixtral", "magistral", "devstral", "codestral"],
        self_publishes_gguf: true,
    },
    Brand {
        label: "IBM Granite",
        orgs: &["ibm-granite"],
        aliases: &["ibm", "granite", "ibmgranite"],
        self_publishes_gguf: true,
    },
    Brand {
        label: "Z.ai",
        orgs: &["zai-org"],
        aliases: &["zai", "zaiorg", "glm", "chatglm", "codegeex", "thudm"],
        self_publishes_gguf: true,
    },
    Brand {
        label: "OpenAI",
        orgs: &["openai"],
        aliases: &["openai", "gptoss"],
        self_publishes_gguf: true,
    },
    Brand {
        label: "InternLM",
        orgs: &["internlm"],
        aliases: &["internlm"],
        self_publishes_gguf: true,
    },
    Brand {
        label: "Liquid AI",
        orgs: &["LiquidAI"],
        aliases: &["liquid", "liquidai", "lfm"],
        self_publishes_gguf: true,
    },
    Brand {
        label: "Nous Research",
        orgs: &["NousResearch"],
        aliases: &["nous", "nousresearch", "hermes"],
        self_publishes_gguf: true,
    },
    // ── Labs whose GGUF builds come from the community ──────────────────
    //
    // These publish weights, not GGUF. Searching their org returns nothing, so
    // the free-text sweep is what actually answers the question.
    Brand {
        label: "DeepSeek",
        orgs: &["deepseek-ai"],
        aliases: &["deepseek", "deepseekai"],
        self_publishes_gguf: false,
    },
    Brand {
        label: "Meta",
        orgs: &["meta-llama"],
        aliases: &["meta", "metallama", "llama", "codellama"],
        self_publishes_gguf: false,
    },
    Brand {
        label: "Moonshot AI",
        orgs: &["moonshotai"],
        aliases: &["moonshot", "moonshotai", "kimi"],
        self_publishes_gguf: false,
    },
    Brand {
        label: "Allen AI",
        orgs: &["allenai"],
        aliases: &["allenai", "ai2", "olmo", "tulu"],
        self_publishes_gguf: false,
    },
    Brand {
        label: "MiniMax",
        orgs: &["MiniMaxAI"],
        aliases: &["minimax", "minimaxai"],
        self_publishes_gguf: false,
    },
    Brand {
        label: "Cohere",
        orgs: &["CohereLabs"],
        aliases: &["cohere", "coherelabs", "command", "commandr", "aya"],
        self_publishes_gguf: false,
    },
    Brand {
        label: "TII Falcon",
        orgs: &["tiiuae"],
        aliases: &["falcon", "tii", "tiiuae"],
        self_publishes_gguf: false,
    },
    Brand {
        label: "01.AI",
        orgs: &["01-ai"],
        aliases: &["01ai", "yi"],
        self_publishes_gguf: false,
    },
    Brand {
        label: "Baichuan",
        orgs: &["baichuan-inc"],
        aliases: &["baichuan", "baichuaninc"],
        self_publishes_gguf: false,
    },
    // ── Conversion specialists ──────────────────────────────────────────
    //
    // Not model authors, but people search for them by name for the same
    // reason — "show me what unsloth built" is a publisher question.
    Brand {
        label: "Unsloth",
        orgs: &["unsloth"],
        aliases: &["unsloth"],
        self_publishes_gguf: true,
    },
    Brand {
        label: "Bartowski",
        orgs: &["bartowski"],
        aliases: &["bartowski"],
        self_publishes_gguf: true,
    },
    Brand {
        label: "LM Studio Community",
        orgs: &["lmstudio-community"],
        aliases: &["lmstudio", "lmstudiocommunity"],
        self_publishes_gguf: true,
    },
    Brand {
        label: "ggml.ai",
        orgs: &["ggml-org"],
        aliases: &["ggml", "ggmlorg", "ggmlai", "llamacpp"],
        self_publishes_gguf: true,
    },
];

/// Strips the characters that differ between how a name is written and how it
/// is typed.
///
/// `Z.ai`, `z ai`, and `zai` are the same publisher; `ibm-granite` and
/// `IBM Granite` are the same account. Unlike the front end's equivalent, dots
/// are removed here: this compares *names*, where a dot is punctuation, not
/// model ids, where a dot is part of a version number.
pub fn normalize(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// The brand a single word names, if any.
fn brand_for_word(word: &str) -> Option<&'static Brand> {
    let n = normalize(word);
    if n.is_empty() {
        return None;
    }
    BRANDS.iter().find(|b| b.aliases.iter().any(|a| *a == n))
}

/// The brand a search query is asking about, if any.
///
/// Two ways to match, in order:
///
/// 1. **The whole query is the name.** "deepseek", "lm studio", "Z.ai".
/// 2. **The query opens with the name.** "nvidia nemotron 4b", "qwen coder 7b".
///    People lead with the publisher and narrow from there, so the first word is
///    where the brand lives.
///
/// Only the leading word is considered, deliberately. Brand names appear *inside*
/// model names constantly — `DeepSeek-R1-Distill-Qwen-7B` carries two — and
/// scanning the whole query would let a search for a DeepSeek distill get
/// answered with Qwen's catalogue.
pub fn resolve(query: &str) -> Option<&'static Brand> {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return None;
    }
    brand_for_word(trimmed).or_else(|| trimmed.split_whitespace().next().and_then(brand_for_word))
}

/// Whether `repo_id` was published by one of `brand`'s own accounts.
///
/// This is what separates an official release from a community conversion of it,
/// which is the distinction the whole feature exists to draw.
pub fn is_official(brand: &Brand, repo_id: &str) -> bool {
    match repo_id.split_once('/') {
        Some((owner, _)) => brand.orgs.iter().any(|o| o.eq_ignore_ascii_case(owner)),
        None => false,
    }
}

/// The brand that owns `repo_id`, if it is a recognised publisher.
///
/// Used to label cards, so it matches on the org slug rather than on aliases —
/// a repo named `bartowski/nvidia_Nemotron-GGUF` is published by Bartowski and
/// says so.
pub fn brand_of_repo(repo_id: &str) -> Option<&'static Brand> {
    let (owner, _) = repo_id.split_once('/')?;
    BRANDS
        .iter()
        .find(|b| b.orgs.iter().any(|o| o.eq_ignore_ascii_case(owner)))
}

/// Whether a repository's *name* declares that it came from `token`.
///
/// Conversion repos lead with the source: `nvidia_Llama-3.1-Nemotron-Nano-4B`,
/// `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF`, `DeepSeek-Coder-V2-Lite`. That
/// leading word is the uploader stating whose model this is, and it is the only
/// evidence available for the repositories that publish no `base_model` tag —
/// `lmstudio-community` and `RichardErkhov` both do this.
///
/// A **prefix**, never a substring, and that restriction is the whole rule:
/// `QQZ2026/Qwen3.6-27B-NVIDIA-NVFP4-no-MTP-GGUF` is a *Qwen* model with NVIDIA's
/// name buried in the middle describing a quantization format. Matching anywhere
/// would file it under NVIDIA; matching only the front files it under Qwen,
/// which is what it is.
///
/// The character after the token must be a separator or a digit, so `gemma-4`
/// and `Qwen3` match while `Gemmable-4-12B` — a different model entirely — does
/// not.
fn name_declares(repo_id: &str, token: &str) -> bool {
    let Some((_, name)) = repo_id.split_once('/') else {
        return false;
    };
    let name = name.to_ascii_lowercase();
    let token = token.to_ascii_lowercase();
    let Some(rest) = name.strip_prefix(&token) else {
        return false;
    };
    match rest.chars().next() {
        // The name is exactly the token, e.g. `unsloth/gemma`.
        None => true,
        Some(c) => c == '-' || c == '_' || c == '.' || c.is_ascii_digit(),
    }
}

/// Whose *model* this repository holds, as opposed to who uploaded it.
///
/// The distinction is the point. NVIDIA publishes exactly one GGUF chat model;
/// the other ~90 NVIDIA models a person can actually run were converted by
/// `bartowski`, `unsloth`, `lmstudio-community`, and `RichardErkhov`. Ranking on
/// the publisher alone therefore answers "show me NVIDIA models" with one card
/// and leaves the rest ordered by download count among unrelated repositories.
/// [`brand_of_repo`] says who uploaded it; this says whose work it is.
///
/// Two sources of evidence, strongest first:
///
/// 1. **The `base_model` tag.** The Hub's own record of what this was built
///    from, written by the uploader's tooling rather than by hand.
/// 2. **The repository name.** See [`name_declares`]. Needed because a
///    meaningful minority of conversions ship no `base_model` tag at all.
///
/// Only one hop is followed. A re-quantization *of a re-quantization* — its
/// `base_model` pointing at `unsloth/...-GGUF` rather than at the lab — is
/// somebody's edit of somebody's conversion, and reporting it as a first-party
/// model would overstate what it is.
pub fn brand_of_model(repo_id: &str, base_model: Option<&str>) -> Option<&'static Brand> {
    if let Some(parent) = base_model {
        if let Some(brand) = brand_of_repo(parent) {
            return Some(brand);
        }
    }
    BRANDS.iter().find(|b| {
        b.orgs.iter().any(|o| name_declares(repo_id, o))
            || b.aliases.iter().any(|a| name_declares(repo_id, a))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_names_people_type_resolve_to_the_slugs_the_hub_stores() {
        // Each of these fails as a literal `author=` value: the Hub wants
        // `Qwen`, `deepseek-ai`, `zai-org`, `moonshotai`.
        assert_eq!(resolve("qwen").unwrap().orgs, &["Qwen"]);
        assert_eq!(resolve("deepseek").unwrap().orgs, &["deepseek-ai"]);
        assert_eq!(resolve("z.ai").unwrap().orgs, &["zai-org"]);
        assert_eq!(resolve("kimi").unwrap().orgs, &["moonshotai"]);
        assert_eq!(resolve("NVIDIA").unwrap().orgs, &["nvidia"]);
        assert_eq!(resolve("unsloth").unwrap().orgs, &["unsloth"]);
    }

    #[test]
    fn a_model_line_names_its_publisher() {
        // Nobody searches "tiiuae"; they search "falcon".
        assert_eq!(resolve("nemotron").unwrap().label, "NVIDIA");
        assert_eq!(resolve("gemma").unwrap().label, "Google");
        assert_eq!(resolve("phi").unwrap().label, "Microsoft");
        assert_eq!(resolve("falcon").unwrap().label, "TII Falcon");
        assert_eq!(resolve("granite").unwrap().label, "IBM Granite");
    }

    #[test]
    fn a_brand_followed_by_a_model_still_resolves() {
        assert_eq!(resolve("nvidia nemotron 4b").unwrap().label, "NVIDIA");
        assert_eq!(resolve("qwen coder 7b").unwrap().label, "Qwen");
        assert_eq!(resolve("  unsloth  gemma  ").unwrap().label, "Unsloth");
    }

    #[test]
    fn a_brand_buried_mid_query_does_not_hijack_the_search() {
        // `DeepSeek-R1-Distill-Qwen-7B` names two brands. Answering with Qwen's
        // catalogue would be answering a question that was not asked.
        assert_eq!(resolve("deepseek r1 distill qwen").unwrap().label, "DeepSeek");
        assert_eq!(resolve("llama 3 nemotron").unwrap().label, "Meta");
    }

    #[test]
    fn ordinary_searches_are_not_brands() {
        assert!(resolve("").is_none());
        assert!(resolve("   ").is_none());
        assert!(resolve("7b instruct").is_none());
        assert!(resolve("coding model").is_none());
    }

    #[test]
    fn official_releases_are_told_apart_from_conversions_of_them() {
        let nvidia = resolve("nvidia").unwrap();
        assert!(is_official(nvidia, "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF"));
        // The org name inside a quantizer's repo name is not authorship.
        assert!(!is_official(nvidia, "bartowski/nvidia_Nemotron-GGUF"));
        assert!(!is_official(nvidia, "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF"));
        assert!(!is_official(nvidia, "nvidia"));
    }

    #[test]
    fn a_repo_reports_the_brand_that_published_it() {
        assert_eq!(brand_of_repo("Qwen/Qwen3-8B-GGUF").unwrap().label, "Qwen");
        // Published by Unsloth, about a DeepSeek model. The publisher is Unsloth.
        assert_eq!(brand_of_repo("unsloth/DeepSeek-V3.2-GGUF").unwrap().label, "Unsloth");
        assert!(brand_of_repo("SomeRandomPerson/my-merge-GGUF").is_none());
        assert!(brand_of_repo("no-slash-here").is_none());
    }

    #[test]
    fn brands_without_their_own_gguf_are_flagged_so_search_keeps_the_free_text_sweep() {
        // Verified against the live API: `author=deepseek-ai&filter=gguf` is
        // empty, and so are meta-llama and moonshotai.
        assert!(!resolve("deepseek").unwrap().self_publishes_gguf);
        assert!(!resolve("meta").unwrap().self_publishes_gguf);
        assert!(!resolve("kimi").unwrap().self_publishes_gguf);
        assert!(resolve("qwen").unwrap().self_publishes_gguf);
    }

    #[test]
    fn a_conversion_reports_the_lab_whose_model_it_holds() {
        // Every one of these is what a search for "nvidia" actually returns:
        // NVIDIA's models, uploaded by somebody else. Ranking them as unrelated
        // is what left the results looking like the search had not worked.
        let by_tag = brand_of_model(
            "bartowski/nvidia_Llama-3.1-Nemotron-Nano-4B-v1.1-GGUF",
            Some("nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"),
        );
        assert_eq!(by_tag.unwrap().label, "NVIDIA");

        // No `base_model` tag published — the name is the only evidence there is.
        let by_name = brand_of_model("RichardErkhov/nvidia_-_OpenMath2-Llama3.1-8B-gguf", None);
        assert_eq!(by_name.unwrap().label, "NVIDIA");

        let lmstudio = brand_of_model("lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF", None);
        assert_eq!(lmstudio.unwrap().label, "NVIDIA");
    }

    #[test]
    fn a_name_mentioning_a_brand_mid_string_is_not_that_brands_model() {
        // A Qwen model quantized to NVIDIA's NVFP4 format. The word "NVIDIA"
        // describes the *format*, not the author, and filing it under NVIDIA
        // would put a Qwen model at the top of an NVIDIA search.
        let b = brand_of_model("QQZ2026/Qwen3.6-27B-NVIDIA-NVFP4-no-MTP-GGUF", None);
        assert_eq!(b.unwrap().label, "Qwen", "the leading word decides, not any word");

        // `Gemmable` is not `gemma`: the character after the token has to be a
        // separator or a digit, and `b` is neither.
        assert!(brand_of_model("Mia-AiLab/Gemmable-4-12B-MTP-GGUF", None).is_none());
        // `Kimiko` is not `Kimi`, for the same reason.
        assert!(brand_of_model("TheBloke/MythoMax-L2-Kimiko-v2-13B-GGUF", None).is_none());
    }

    #[test]
    fn the_base_model_tag_outranks_the_name() {
        // Named for Qwen, built from DeepSeek's distill. The tag is the Hub's
        // own record and wins.
        let b = brand_of_model(
            "MaziyarPanahi/DeepSeek-R1-0528-Qwen3-8B-GGUF",
            Some("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"),
        );
        assert_eq!(b.unwrap().label, "DeepSeek");
    }

    #[test]
    fn a_requantization_of_a_conversion_is_not_credited_to_the_lab() {
        // `base_model` points at another quantizer's GGUF, not at NVIDIA. Only
        // one hop is followed, so this is somebody's edit of somebody's
        // conversion rather than an NVIDIA release.
        let b = brand_of_model(
            "meshllm/Nemo-Omni-30B-UD-Q4_K_XL-layers",
            Some("unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF"),
        );
        assert_eq!(b.unwrap().label, "Unsloth", "credited to who it was actually built from");
    }

    #[test]
    fn a_model_with_no_recognisable_source_claims_no_brand() {
        assert!(brand_of_model("SomePerson/my-private-merge-GGUF", None).is_none());
        assert!(brand_of_model("no-slash-here", None).is_none());
    }

    #[test]
    fn no_alias_is_claimed_by_two_brands() {
        let mut seen: Vec<&str> = Vec::new();
        for b in BRANDS {
            for a in b.aliases {
                assert!(!seen.contains(a), "alias '{a}' is claimed twice ({})", b.label);
                seen.push(a);
            }
        }
    }

    #[test]
    fn every_alias_is_already_normalized() {
        // An alias with a capital or a dash could never match, because the typed
        // word is normalized before comparison and the table is not.
        for b in BRANDS {
            for a in b.aliases {
                assert_eq!(normalize(a), *a, "alias '{a}' ({}) is not normalized", b.label);
            }
        }
    }
}
