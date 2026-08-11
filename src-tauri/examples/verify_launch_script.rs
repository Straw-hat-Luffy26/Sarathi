//! Prints the terminal script Sarathi would generate for each shipped tool.
//!
//! The banner is the part a user actually sees when Launch works, so it is
//! worth reading rather than inferring from tests. Every value below comes from
//! a `LaunchContext` — the same struct the command builds from the live gateway
//! port and the currently loaded model.
//!
//! Run with:  cargo run --example verify_launch_script

use std::path::Path;

use sarathi_lib::launcher::console::script_for;
use sarathi_lib::launcher::mcp::McpRegistry;
use sarathi_lib::launcher::spec::{builtin_tools, resolve_args, LaunchContext, RuntimeSnapshot};

fn main() {
    let ctx = LaunchContext {
        port: 11435,
        model_id: "unsloth/gpt-oss-20b-GGUF".into(),
        model_name: "gpt-oss 20B".into(),
        client_dir: r"C:\Users\lenovo\AppData\Roaming\com.sarathi.app\clients\claude-code".into(),
        context_length: 8192,
        mcp: McpRegistry::default(),
        runtime: RuntimeSnapshot {
            quantization: Some("Q4_0".into()),
            backend: Some("llama.cpp (GPU offload: 999 layers)".into()),
            gpu_layers: Some(999),
            cpu_moe_layers: Some(0),
            gpu_name: Some("NVIDIA GeForce RTX 5060 Laptop GPU".into()),
            vram_total_bytes: Some(8_151_000_000),
            gpu_backend_compiled: true,
        },
    };

    for spec in builtin_tools() {
        let args = resolve_args(&spec.launch.args, &ctx);
        let program = Path::new(r"C:\Users\lenovo\AppData\Roaming\npm").join(format!(
            "{}.cmd",
            spec.launch.command
        ));

        println!("{}", "=".repeat(78));
        println!("{}  ({})", spec.name, spec.id);
        println!("{}", "=".repeat(78));
        let out = std::env::temp_dir().join(format!("sarathi-yatra-{}.ps1", spec.id));
        std::fs::write(&out, script_for(&spec, &ctx, &program, &args)).unwrap();
        println!("wrote {}", out.display());
    }

}
