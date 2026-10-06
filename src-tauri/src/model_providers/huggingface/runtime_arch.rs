//! Which GGUF architectures the bundled llama.cpp can load.
//!
//! The Hub reports each GGUF's `general.architecture`, and new families appear
//! there before the llama.cpp release Sarathi links against knows them. Listing
//! such a model offers a download that can only fail at load time, after the
//! user has waited for tens of gigabytes. Checking the architecture up front
//! turns that into a model that is simply not offered yet.
//!
//! The list is llama.cpp's own `LLM_ARCH_NAMES` table for the version pinned in
//! `Cargo.lock` (`llama-cpp-sys-2` 0.1.153). It changes only when that
//! dependency is bumped, and the test below compares it with the vendored
//! source whenever that source is on disk, so a bump that adds architectures
//! fails loudly instead of quietly hiding them.

/// `LLM_ARCH_NAMES` from `llama.cpp/src/llama-arch.cpp`, minus the `clip`
/// projector and the `(unknown)` sentinel, neither of which is a loadable model.
const LOADABLE: &[&str] = &[
    "llama", "llama4", "deci", "falcon", "grok", "gpt2", "gptj", "gptneox", "mpt",
    "baichuan", "starcoder", "refact", "bert", "modern-bert", "nomic-bert",
    "nomic-bert-moe", "neo-bert", "jina-bert-v2", "jina-bert-v3", "eurobert", "bloom",
    "stablelm", "qwen", "qwen2", "qwen2moe", "qwen2vl", "qwen3", "qwen3moe",
    "qwen3next", "qwen3vl", "qwen3vlmoe", "qwen35", "qwen35moe", "phi2", "phi3",
    "phimoe", "plamo", "plamo2", "plamo3", "codeshell", "orion", "internlm2",
    "minicpm", "minicpm3", "gemma", "gemma2", "gemma3", "gemma3n", "gemma4",
    "gemma4-assistant", "gemma-embedding", "starcoder2", "mamba", "mamba2", "jamba",
    "falcon-h1", "xverse", "command-r", "cohere2", "dbrx", "olmo", "olmo2", "olmoe",
    "openelm", "arctic", "deepseek", "deepseek2", "deepseek2-ocr", "deepseek32",
    "chatglm", "glm4", "glm4moe", "glm-dsa", "bitnet", "t5", "t5encoder", "jais",
    "jais2", "nemotron", "nemotron_h", "nemotron_h_moe", "exaone", "exaone4",
    "exaone-moe", "rwkv6", "rwkv6qwen2", "rwkv7", "arwkv7", "granite", "granitemoe",
    "granitehybrid", "chameleon", "wavtokenizer-dec", "plm", "bailingmoe",
    "bailingmoe2", "dots1", "arcee", "afmoe", "ernie4_5", "ernie4_5-moe",
    "hunyuan-moe", "hunyuan-dense", "hunyuan_vl", "smollm3", "gpt-oss", "lfm2",
    "lfm2moe", "dream", "smallthinker", "llada", "llada-moe", "seed_oss", "grovemoe",
    "apertus", "minimax-m2", "cogvlm", "rnd1", "pangu-embedded", "mistral3",
    "mistral4", "paddleocr", "mimo2", "step35", "llama-embed", "maincoder",
    "kimi-linear", "talkie", "mellum",
];

/// True when the bundled llama.cpp can load a GGUF of this architecture.
///
/// An empty string means the Hub did not say, which is not evidence that the
/// model is unloadable, so it passes.
pub fn is_loadable(architecture: &str) -> bool {
    let arch = architecture.trim();
    arch.is_empty() || LOADABLE.iter().any(|known| known.eq_ignore_ascii_case(arch))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_families_are_loadable() {
        for arch in ["llama", "qwen2", "qwen3moe", "qwen35", "qwen35moe", "gemma4", "gpt-oss", "glm-dsa"] {
            assert!(is_loadable(arch), "{arch} must be loadable");
        }
    }

    #[test]
    fn architectures_newer_than_the_runtime_are_not() {
        // Both seen on the Hub's most-downloaded GGUF page.
        assert!(!is_loadable("qwen4exp"));
        assert!(!is_loadable("inkling"));
        assert!(!is_loadable("clip"), "a projector is not a model");
    }

    #[test]
    fn an_unreported_architecture_is_given_the_benefit_of_the_doubt() {
        assert!(is_loadable(""));
        assert!(is_loadable("  "));
    }

    /// Keeps [`LOADABLE`] in step with the llama.cpp actually linked.
    ///
    /// Reads the version `Cargo.lock` pins, then that version's vendored
    /// `llama-arch.cpp` from the cargo registry. Skips quietly when the source
    /// is not on disk (a fresh CI cache), because the check is about drift, not
    /// about the environment.
    #[test]
    fn the_list_matches_the_vendored_llama_cpp() {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let Ok(lock) = std::fs::read_to_string(manifest.join("Cargo.lock")) else {
            return;
        };
        let Some(version) = lock
            .split("[[package]]")
            .find(|p| p.contains("name = \"llama-cpp-sys-2\""))
            .and_then(|p| p.lines().find_map(|l| l.trim().strip_prefix("version = \"")))
            .map(|v| v.trim_end_matches('"').to_string())
        else {
            return;
        };

        let cargo_home = std::env::var_os("CARGO_HOME")
            .map(std::path::PathBuf::from)
            .or_else(|| std::env::var_os("USERPROFILE").map(|h| std::path::PathBuf::from(h).join(".cargo")))
            .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cargo")));
        let Some(registry) = cargo_home.map(|h| h.join("registry").join("src")) else {
            return;
        };
        let Some(source) = std::fs::read_dir(&registry).ok().and_then(|dirs| {
            dirs.flatten()
                .map(|d| {
                    d.path()
                        .join(format!("llama-cpp-sys-2-{version}"))
                        .join("llama.cpp/src/llama-arch.cpp")
                })
                .find(|p| p.is_file())
        }) else {
            return;
        };

        let text = std::fs::read_to_string(&source).expect("readable llama-arch.cpp");
        let table = text
            .split("LLM_ARCH_NAMES")
            .nth(1)
            .and_then(|rest| rest.split("};").next())
            .expect("LLM_ARCH_NAMES table");
        // The sentinel is spelled `(unknown)`; anything that is not an
        // identifier is not an architecture name.
        let vendored: Vec<&str> = table
            .split('"')
            .skip(1)
            .step_by(2)
            .filter(|name| {
                !name.is_empty()
                    && *name != "clip"
                    && name
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
            })
            .collect();

        for name in &vendored {
            assert!(
                LOADABLE.contains(name),
                "llama-cpp-sys-2 {version} loads '{name}', which LOADABLE omits — update runtime_arch.rs"
            );
        }
        for name in LOADABLE {
            assert!(
                vendored.contains(name),
                "LOADABLE lists '{name}', which llama-cpp-sys-2 {version} does not know"
            );
        }
    }
}
