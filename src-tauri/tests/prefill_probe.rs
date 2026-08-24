//! TEMPORARY diagnostic probe — delete after use.
//!
//! Establishes whether the `NTokensZero` prefill failure is caused by
//! multi-chunk prefill in general, or by the `nemotron_h` hybrid architecture.

use sarathi_lib::ai_engine::runtime::LlamaCppRuntime;
use sarathi_lib::ai_engine::traits::{ChatMessage, GenerationParams, ModelLoadConfig};

fn probe(label: &str, path: &str, template: &str) {
    if !std::path::Path::new(path).exists() {
        println!("SKIP {label}: not installed");
        return;
    }

    let mut rt = LlamaCppRuntime::new();
    let config = ModelLoadConfig {
        model_path: path.to_string(),
        model_id: label.to_string(),
        model_name: label.to_string(),
        quantization: "Q4".to_string(),
        context_length: 65536,
        gpu_layers: 999,
        cpu_moe_layers: 0,
        threads: 4,
        chat_template: template.to_string(),
        stop_tokens: vec![],
    };
    if let Err(e) = rt.load_model(&config, |_| {}) {
        println!("SKIP {label}: load failed: {e:#}");
        return;
    }

    // "word " is roughly one token; enough to land on either side of the
    // 2048-token n_batch boundary the prefill loop chunks at.
    for words in [400usize, 2500, 6000, 23000] {
        let filler = "word ".repeat(words);
        let msgs = [
            ChatMessage::new("system", "You are helpful."),
            ChatMessage::new("user", format!("{filler}\nReply with OK.")),
        ];
        let params = GenerationParams { max_tokens: 4, ..GenerationParams::default() };

        match rt.generate(&msgs, &params, |_| {}) {
            Ok(_) => println!("  {label}  ~{words} words: OK"),
            Err(e) => println!("  {label}  ~{words} words: FAIL {e:#}"),
        }
    }
}

#[test]
#[ignore]
fn prefill_probe() {
    let root = r"C:\Users\lenovo\AppData\Roaming\com.sarathi.app\models";
    probe(
        "nemotron_h-4B",
        &format!(r"{root}\local\NVIDIA_Nemotron3-Nano-4B\base\NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf"),
        "llama3",
    );
    probe(
        "llama-3.2-1B (control)",
        &format!(r"{root}\huggingface\meta-Llama_Llama-3.2-1B\base\Llama-3.2-1B-Instruct-Q8_0.gguf"),
        "llama3",
    );
}
