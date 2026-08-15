//! The Dharma Yatra — Sarathi's startup screen, and the terminal it runs in.
//!
//! ## Why this exists at all
//!
//! `main.rs` carries
//! `#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]`,
//! so a release build of Sarathi is a GUI process with **no console**. A child
//! spawned from it inherits that — which is to say, inherits nothing. Claude
//! Code, opencode and Hermes are terminal agents: with nowhere to draw they exit
//! at once or run invisibly.
//!
//! So Sarathi opens the terminal itself, and having opened it, says what the
//! user is about to be connected to before handing the screen over.
//!
//! ## The naming
//!
//! From the Bhagavad Gita's chariot: the charioteer who counsels the warrior.
//!
//! | Term  | What it is here                                    |
//! |-------|----------------------------------------------------|
//! | Ratha | the chariot — Sarathi's local gateway               |
//! | Yoddha| the warrior — the model that is loaded              |
//! | Astra | the weapons — MCP tools handed to the provider      |
//! | Sena  | the army — the runtime and hardware behind it       |
//! | Chakra| the Sudarshan Chakra, spinning while the yatra runs |
//!
//! It is an identity, not a theme applied to a generic terminal: the four
//! panels are the four things that have to be true before a coding agent can
//! answer a question locally, and each is named for the part of the chariot it
//! corresponds to.
//!
//! ## Why PowerShell rather than a batch file
//!
//! The screen needs colour, cursor addressing and a timed redraw. `cmd` has
//! none of those without spawning a process per frame. PowerShell has ANSI, a
//! millisecond timer, and `$Host.UI.RawUI.WindowSize` for laying out against the
//! real terminal — and it can `&` the provider afterwards so the agent inherits
//! the same console rather than a new one.
//!
//! Nothing here decides anything. Every value is read from the
//! [`LaunchContext`] the launcher already built; a field that is `None` prints
//! as an explicit unknown rather than being filled in.

use std::path::{Path, PathBuf};

use crate::launcher::spec::{LaunchContext, ToolSpec};

/// Name of the generated script, written into the tool's own config directory
/// beside the config Sarathi already writes for it.
pub const SCRIPT_NAME: &str = "sarathi-launch.ps1";

/// How long the Chakra turns before the provider takes over.
///
/// Long enough to read the panels, short enough that someone launching a tool
/// for the tenth time today is not waiting on ceremony.
const YATRA_FRAMES: u32 = 24;
const FRAME_MS: u32 = 55;

/// Escapes a value for a PowerShell single-quoted string.
///
/// Single quotes are the only literal string form that interpolates nothing, so
/// `$`, backticks and `;` in a model name are inert. The one character that
/// still matters is the quote itself, which doubles.
fn ps_literal(value: &str) -> String {
    let cleaned: String = value
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    format!("'{}'", cleaned.replace('\'', "''"))
}

/// A value that may not be known, as a PowerShell literal.
fn ps_opt(value: Option<&str>) -> String {
    match value.map(str::trim).filter(|v| !v.is_empty()) {
        Some(v) => ps_literal(v),
        None => "$null".to_string(),
    }
}

/// Formats bytes as GB for display, or `$null` when unknown.
fn ps_vram(bytes: Option<u64>) -> String {
    match bytes {
        Some(b) if b > 0 => ps_literal(&format!("{:.1} GB VRAM", b as f64 / 1e9)),
        _ => "$null".to_string(),
    }
}

/// How the model's placement reads in one line.
///
/// Derived from what the runtime *reported*, never from what was hoped for: a
/// build with no GPU backend says so, and a model that went to the CPU says
/// that, however much VRAM the machine has.
fn placement_line(ctx: &LaunchContext) -> Option<String> {
    let r = &ctx.runtime;

    if !r.gpu_backend_compiled {
        return Some("CPU only - no GPU backend in this build".to_string());
    }

    match (r.gpu_layers, r.cpu_moe_layers) {
        (Some(0), _) | (None, _) => Some("CPU placement".to_string()),
        (Some(_), Some(n)) if n > 0 => Some(format!("GPU, experts of {n} layer(s) in RAM")),
        (Some(999), _) => Some("all layers on GPU".to_string()),
        (Some(n), _) => Some(format!("{n} layers on GPU")),
    }
}

/// The title the generated script gives its window.
///
/// Also how the window is found again: `start` returns as soon as it has handed
/// the script to a new console, so the pid Sarathi receives belongs to a wrapper
/// that is already gone. The window it opened is the thing that is still there.
pub fn title_for(tool_name: &str) -> String {
    format!("Sarathi - {tool_name}")
}

/// Builds the startup script: the Dharma Yatra, then the provider.
pub fn script_for(spec: &ToolSpec, ctx: &LaunchContext, program: &Path, args: &[String]) -> String {
    let r = &ctx.runtime;

    // What this provider was actually given, not what the registry holds.
    //
    // Reading the registry here is how the screen came to announce five MCP
    // servers "CONNECTED" to providers whose generated config contained none:
    // the two are the same list right up until a provider cannot be handed
    // them, which is exactly the case worth reporting.
    let delivery = spec.mcp_delivery(&ctx.mcp);
    let astra: Vec<String> = delivery.delivered.iter().map(|name| ps_literal(name)).collect();
    // Stated separately so the screen can distinguish "none configured" from
    // "configured, and this provider cannot take them".
    let astra_note = ps_literal(&match (&delivery.supported, delivery.dropped.len()) {
        (false, 0) => String::new(),
        (false, _) => delivery
            .reason
            .clone()
            .unwrap_or_else(|| "this provider takes no MCP servers".to_string()),
        (true, 0) => String::new(),
        (true, n) => format!("{n} configured server(s) this provider cannot read"),
    });

    let mut s = String::new();

    s.push_str(&format!(
        r#"# Generated by Sarathi. Rewritten on every launch.
$ErrorActionPreference = 'Continue'
$Host.UI.RawUI.WindowTitle = {title}

# ── Values, all supplied by the Sarathi runtime ────────────────────────────
$Provider   = {provider}
$Protocol   = {protocol}
$Port       = {port}
$Yoddha     = {model}
$ModelId    = {model_id}
$Quant      = {quant}
$Context    = {context}
$Backend    = {backend}
$GpuName    = {gpu}
$Vram       = {vram}
$Placement  = {placement}
$Astra      = @({astra})
$AstraNote  = {astra_note}

# ── Terminal capability ───────────────────────────────────────────────────
# VT processing is enabled on Windows 10+ consoles but not guaranteed: a
# redirected stream or an old host has none, and printing raw escapes would be
# worse than printing nothing. Everything degrades to plain text.
$Fancy = $true
try {{
    if ($Host.UI.RawUI -eq $null) {{ $Fancy = $false }}
    if ($env:TERM -eq 'dumb') {{ $Fancy = $false }}
    $null = [Console]::WindowWidth
}} catch {{ $Fancy = $false }}

$E = [char]27
function C($code, $text) {{ if ($Fancy) {{ "$E[${{code}}m$text$E[0m" }} else {{ $text }} }}

$Gold  = '38;5;179'   # bronze, not yellow — the accent
$Dim   = '38;5;245'
$Warm  = '38;5;223'   # parchment, for values
$Good  = '38;5;108'
$Bad   = '38;5;167'

# The chakra: eight spokes, each frame one step around. Box-drawing and the
# dharma wheel are in every Windows console font that ships today; if the
# console cannot render them the ASCII fallback still turns.
if ($Fancy) {{ $Spokes = @([char]0x2E22,[char]0x2E23,[char]0x2E24,[char]0x2E25) }} else {{ $Spokes = @('|','/','-','\') }}
$Wheel = @('|','/','-','\')
if ($Fancy) {{ $Hub = [char]0x2638 }} else {{ $Hub = '*' }}

function Width {{
    $w = 0
    try {{ $w = [Console]::WindowWidth }} catch {{ $w = 0 }}
    if ($w -lt 48) {{ $w = 72 }}          # redirected or headless host
    if ($w -gt 100) {{ $w = 100 }}        # long lines are harder to read, not easier
    return $w
}}

function Centre($text, $w) {{
    $pad = [Math]::Max(0, [int](($w - $text.Length) / 2))
    (' ' * $pad) + $text
}}

# A row of two labelled columns, laid out against the real terminal width so a
# narrow window wraps gracefully instead of tearing.
function Row($leftVal, $rightVal, $w) {{
    $col = [Math]::Max(12, [int](($w - 4) / 2))
    $l = if ($leftVal) {{ [string]$leftVal }} else {{ '' }}
    $r = if ($rightVal) {{ [string]$rightVal }} else {{ '' }}
    # Colour codes are characters too, so a coloured value is longer than it
    # looks. Truncating on the raw length would cut mid-escape and leave the
    # terminal stuck in a colour, so only uncoloured text is trimmed.
    if (-not $Fancy) {{
        if ($l.Length -gt $col) {{ $l = $l.Substring(0, [Math]::Max(1, $col - 1)) + '.' }}
        if ($r.Length -gt $col) {{ $r = $r.Substring(0, [Math]::Max(1, $col - 1)) + '.' }}
    }}
    $pad = [Math]::Max(0, $col - $l.Length)
    ('  ' + $l + (' ' * $pad) + '  ' + $r)
}}

function Dot($ok) {{ if ($ok) {{ C $Good ([char]0x25CF) }} else {{ C $Dim ([char]0x25CB) }} }}
"#,
        title = ps_literal(&title_for(&spec.name)),
        provider = ps_literal(&spec.name),
        protocol = ps_literal(spec.protocol.label()),
        port = ctx.port,
        model = ps_opt(Some(&ctx.model_name)),
        model_id = ps_opt(Some(&ctx.model_id)),
        quant = ps_opt(r.quantization.as_deref()),
        context = ctx.context_length,
        backend = ps_opt(r.backend.as_deref()),
        gpu = ps_opt(r.gpu_name.as_deref()),
        vram = ps_vram(r.vram_total_bytes),
        placement = ps_opt(placement_line(ctx).as_deref()),
        astra = astra.join(", "),
        astra_note = astra_note,
    ));

    s.push_str(
        &r#"
# ── The Dharma Yatra ──────────────────────────────────────────────────────
$w = Width
Clear-Host
Write-Host ''
Write-Host (Centre (C $Gold ("$Hub  S A R A T H I  $Hub")) $w)
Write-Host (Centre (C $Dim 'THE LOCAL AI CHARIOTEER') $w)
Write-Host ''

# The chakra turns while the panels are already on screen, so the wait is the
# animation rather than a pause before it.
$chakraRow = 4
for ($i = 0; $i -lt FRAMES; $i++) {
    $g = $Wheel[$i % $Wheel.Count]
    $frame = "$Hub $g $Hub"
    if ($Fancy) {
        try { [Console]::SetCursorPosition(0, $chakraRow) } catch { }
    }
    Write-Host (Centre (C $Gold $frame) $w) -NoNewline
    Write-Host ''
    if (-not $Fancy) { break }
    Start-Sleep -Milliseconds DELAY
}
Write-Host ''

$modelKnown = [bool]$Yoddha
$gpuActive  = $Placement -and ($Placement -notmatch 'CPU')

Write-Host (Row (C $Gold 'RATHA') (C $Gold 'YODDHA') $w)
Write-Host (Row ((Dot $true) + ' ' + (C $Warm 'ONLINE')) ((Dot $modelKnown) + ' ' + (C $Warm $(if ($Yoddha) { $Yoddha } else { 'no model loaded' }))) $w)
Write-Host (Row (C $Dim "127.0.0.1:$Port") (C $Dim $(if ($Quant) { $Quant } else { '-' })) $w)
Write-Host (Row (C $Dim 'Sarathi Gateway') (C $Dim "$Context tokens context") $w)
Write-Host ''

Write-Host (Row (C $Gold 'ASTRA') (C $Gold 'SENA') $w)
$senaLines = @()
$senaLines += (Dot ([bool]$Backend)) + ' ' + (C $Warm $(if ($Backend) { $Backend } else { 'runtime unknown' }))
$senaLines += (Dot ([bool]$GpuName)) + ' ' + (C $Dim $(if ($GpuName) { $GpuName } else { 'no GPU detected' }))
$senaLines += (Dot ([bool]$Vram))    + ' ' + (C $Dim $(if ($Vram) { $Vram } else { '-' }))
$senaLines += (Dot $gpuActive)       + ' ' + (C $Dim $(if ($Placement) { $Placement } else { 'placement unknown' }))

$astraLines = @()
if ($Astra.Count -eq 0) {
    $astraLines += (Dot $false) + ' ' + (C $Dim $(if ($AstraNote) { $AstraNote } else { 'no MCP servers configured' }))
} else {
    # Handed over, not connected: this provider was given these servers in its
    # config. Whether it starts them and lists their tools is its own business,
    # and Sarathi does not claim to know from here.
    foreach ($a in $Astra) { $astraLines += (Dot $true) + ' ' + (C $Warm $a) }
    if ($AstraNote) { $astraLines += (Dot $false) + ' ' + (C $Dim $AstraNote) }
}

$rows = [Math]::Max($astraLines.Count, $senaLines.Count)
for ($i = 0; $i -lt $rows; $i++) {
    $l = if ($i -lt $astraLines.Count) { $astraLines[$i] } else { '' }
    $r = if ($i -lt $senaLines.Count)  { $senaLines[$i] }  else { '' }
    Write-Host (Row $l $r $w)
}
Write-Host ''

$bar = ([string][char]0x2500) * 6
Write-Host (Centre (C $Gold "$bar  DHARMA YATRA  $bar") $w)
Write-Host ''

function Step($name, $ok, $note) {
    $mark = if ($ok) { C $Good ([char]0x2713) } else { C $Bad ([char]0x00D7) }
    $state = if ($ok) { C $Warm $note } else { C $Dim $note }
    Write-Host ('    ' + (C $Dim $name.PadRight(12)) + $mark + ' ' + $state)
}

Step 'Ratha'  $true        'ONLINE'
Step 'Yoddha' $modelKnown  $(if ($modelKnown) { 'LOADED' } else { 'NOT LOADED' })
# "HANDED OVER", not "CONNECTED". Sarathi writes the config; the provider makes
# the connection, and only it knows whether the server started and answered.
Step 'Astra'  ($Astra.Count -gt 0) $(if ($Astra.Count -gt 0) { "$($Astra.Count) HANDED OVER" } else { 'NONE' })
Step 'Sena'   ([bool]$Backend) $(if ($Backend) { 'READY' } else { 'UNKNOWN' })
Write-Host ''

Write-Host (Centre (C $Gold "$Hub  READY  $Hub") $w)
Write-Host ''
Write-Host (Centre (C $Dim "Provider: $Provider  ($Protocol)") $w)
Write-Host (Centre (C $Dim "Ratha: Sarathi     Yoddha: $(if ($Yoddha) { $Yoddha } else { 'none' })") $w)
Write-Host ''

# The animation is over before the provider is invoked: the cursor is left at a
# fresh line, nothing is still redrawing, and the agent owns the screen from
# here. Anything else would fight the provider for the cursor.
if ($Fancy) { Write-Host "$E[0m" -NoNewline }

"#
        .replace("FRAMES", &YATRA_FRAMES.to_string())
        .replace("DELAY", &FRAME_MS.to_string()),
    );

    // ── Hand over ───────────────────────────────────────────────────────────
    s.push_str(&format!("$SarathiProgram = {}\n", ps_literal(&program.display().to_string())));
    if args.is_empty() {
        s.push_str("$SarathiArgs = @()\n");
    } else {
        let quoted: Vec<String> = args.iter().map(|a| ps_literal(a)).collect();
        s.push_str(&format!("$SarathiArgs = @({})\n", quoted.join(", ")));
    }

    s.push_str(
        r#"
& $SarathiProgram @SarathiArgs
$SarathiExit = $LASTEXITCODE

# A provider that dies at once would otherwise close the window over its own
# error. Held open only on failure: a clean exit is the user closing the agent
# deliberately, and does not need dismissing.
if ($SarathiExit -ne 0 -and $null -ne $SarathiExit) {
    Write-Host ''
    Write-Host (C '38;5;167' "$Provider exited with code $SarathiExit.")
    Write-Host (C '38;5;245' 'This window is held open so the error above can be read.')
    [void](Read-Host 'Press Enter to close')
}
"#,
    );

    s
}

/// Writes the launch script into the tool's config directory.
pub fn write_script(
    client_dir: &Path,
    spec: &ToolSpec,
    ctx: &LaunchContext,
    program: &Path,
    args: &[String],
) -> Result<PathBuf, String> {
    std::fs::create_dir_all(client_dir)
        .map_err(|e| format!("could not prepare {}'s config directory: {e}", spec.name))?;

    let path = client_dir.join(SCRIPT_NAME);
    std::fs::write(&path, script_for(spec, ctx, program, args))
        .map_err(|e| format!("could not write {}'s launch script: {e}", spec.name))?;
    Ok(path)
}

/// Whether a console window with this title is open.
#[cfg(windows)]
pub fn window_open(title: &str) -> bool {
    let out = crate::system_analyzer::process_utils::create_hidden_command("tasklist")
        .args(["/FI", &format!("WINDOWTITLE eq {title}"), "/NH"])
        .output();

    match out {
        Ok(o) => {
            let text = String::from_utf8_lossy(&o.stdout).to_ascii_lowercase();
            text.contains("powershell") || text.contains("cmd.exe") || text.contains("conhost")
        }
        Err(_) => false,
    }
}

#[cfg(not(windows))]
pub fn window_open(_title: &str) -> bool {
    false
}

/// Whether a process id still belongs to a running process.
pub fn is_running(pid: u32) -> bool {
    #[cfg(windows)]
    {
        let out = crate::system_analyzer::process_utils::create_hidden_command("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/NH"])
            .output();

        match out {
            Ok(o) => String::from_utf8_lossy(&o.stdout).contains(&pid.to_string()),
            Err(_) => true,
        }
    }

    #[cfg(not(windows))]
    {
        crate::system_analyzer::process_utils::create_hidden_command("kill")
            .args(["-0", &pid.to_string()])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launcher::mcp::{McpRegistry, McpServerSpec};
    use crate::launcher::spec::{builtin_tools, RuntimeSnapshot};

    fn registry(names: &[&str]) -> McpRegistry {
        let mut servers = std::collections::BTreeMap::new();
        for n in names {
            servers.insert(
                (*n).to_string(),
                serde_json::from_value::<McpServerSpec>(serde_json::json!({
                    "command": "x"
                }))
                .expect("a minimal server spec"),
            );
        }
        McpRegistry { servers, warnings: vec![] }
    }

    /// A machine with a model loaded on the GPU and tools connected.
    fn ctx_loaded() -> LaunchContext {
        LaunchContext {
            port: 11435,
            model_id: "LiquidAI/LFM2.5-8B-A1B-GGUF".into(),
            model_name: "LFM2.5 8B A1B".into(),
            client_dir: r"C:\clients\claude-code".into(),
            context_length: 8192,
            mcp: registry(&["sarathi-research", "filesystem"]),
            runtime: RuntimeSnapshot {
                quantization: Some("Q4_0".into()),
                backend: Some("llama.cpp (GPU offload: 999 layers)".into()),
                gpu_layers: Some(999),
                cpu_moe_layers: Some(0),
                gpu_name: Some("NVIDIA GeForce RTX 5060 Laptop GPU".into()),
                vram_total_bytes: Some(8_151_000_000),
                gpu_backend_compiled: true,
            },
        }
    }

    /// Nothing loaded, no GPU backend — every panel has to degrade.
    fn ctx_empty() -> LaunchContext {
        LaunchContext {
            port: 11435,
            model_id: String::new(),
            model_name: String::new(),
            client_dir: r"C:\clients\opencode".into(),
            context_length: 0,
            mcp: McpRegistry::default(),
            runtime: RuntimeSnapshot::default(),
        }
    }

    fn claude() -> ToolSpec {
        builtin_tools().into_iter().find(|t| t.id == "claude-code").expect("shipped")
    }

    #[test]
    fn the_screen_names_the_provider_the_model_and_the_gateway() {
        let s = script_for(&claude(), &ctx_loaded(), Path::new(r"C:\npm\claude.cmd"), &[]);

        assert!(s.contains("S A R A T H I"), "identity: {s}");
        assert!(s.contains("'Claude Code'"), "the provider being launched");
        assert!(s.contains("'LFM2.5 8B A1B'"), "the loaded model");
        assert!(s.contains("$Port       = 11435"), "the live gateway port");
        assert!(s.contains("'Q4_0'"), "the quantization actually loaded");
    }

    /// Every panel value has to come from the context. A screen that prints a
    /// plausible model name when none is loaded is worse than a blank one.
    #[test]
    fn nothing_is_invented_when_the_runtime_knows_nothing() {
        let s = script_for(&claude(), &ctx_empty(), Path::new("x"), &[]);

        assert!(s.contains("$Yoddha     = $null"), "no model: {s}");
        assert!(s.contains("$Backend    = $null"));
        assert!(s.contains("$GpuName    = $null"));
        assert!(s.contains("$Vram       = $null"));
        assert!(s.contains("$Astra      = @()"));
        // And the script has the branches that render those as words.
        assert!(s.contains("no model loaded"));
        assert!(s.contains("no GPU detected"));
        assert!(s.contains("no MCP servers configured"));
    }

    /// The placement line is the one claim that must never outrun the evidence.
    #[test]
    fn placement_reports_what_the_runtime_did_not_what_was_hoped_for() {
        let mut cpu_build = ctx_loaded();
        cpu_build.runtime.gpu_backend_compiled = false;
        assert_eq!(
            placement_line(&cpu_build).as_deref(),
            Some("CPU only - no GPU backend in this build"),
            "a GPU in the machine does not make a CPU build a GPU one"
        );

        let mut cpu_placed = ctx_loaded();
        cpu_placed.runtime.gpu_layers = Some(0);
        assert_eq!(placement_line(&cpu_placed).as_deref(), Some("CPU placement"));

        assert_eq!(placement_line(&ctx_loaded()).as_deref(), Some("all layers on GPU"));

        let mut partial = ctx_loaded();
        partial.runtime.gpu_layers = Some(24);
        partial.runtime.cpu_moe_layers = Some(0);
        assert_eq!(placement_line(&partial).as_deref(), Some("24 layers on GPU"));

        let mut moe = ctx_loaded();
        moe.runtime.cpu_moe_layers = Some(14);
        assert_eq!(
            placement_line(&moe).as_deref(),
            Some("GPU, experts of 14 layer(s) in RAM"),
            "MoE expert offload is a distinct state and reads as one"
        );
    }

    /// The animation must be finished before the agent is invoked, or the two
    /// fight over the cursor.
    #[test]
    fn the_chakra_stops_before_the_provider_starts() {
        let s = script_for(&claude(), &ctx_loaded(), Path::new("claude.cmd"), &[]);

        let spin = s.find("Start-Sleep -Milliseconds").expect("animation");
        let hand = s.find("& $SarathiProgram").expect("handover");
        assert!(spin < hand, "the loop has to be above the invocation");
    }

    #[test]
    fn the_provider_and_its_arguments_survive_quoting() {
        let s = script_for(
            &claude(),
            &ctx_loaded(),
            Path::new(r"C:\Program Files\npm\claude.cmd"),
            &["--mcp-config".into(), r"C:\App Data\mcp.json".into()],
        );

        assert!(s.contains(r"$SarathiProgram = 'C:\Program Files\npm\claude.cmd'"), "{s}");
        assert!(s.contains(r"'--mcp-config', 'C:\App Data\mcp.json'"), "{s}");
    }

    /// Values come from repository metadata and a user's own mcp.json, so a
    /// quote in one must not end the string it sits in.
    #[test]
    fn a_quote_in_a_name_cannot_escape_its_string() {
        let mut hostile = ctx_loaded();
        hostile.model_name = "it's a '; Remove-Item C:\\ ; '".into();

        let s = script_for(&claude(), &hostile, Path::new("x"), &[]);

        assert!(!s.contains("; Remove-Item C:\\ ; '\n"), "injection survived: {s}");
        assert!(s.contains("''"), "quotes must be doubled: {s}");
    }

    #[test]
    fn the_layout_is_measured_against_the_real_terminal() {
        let s = script_for(&claude(), &ctx_loaded(), Path::new("x"), &[]);
        assert!(s.contains("[Console]::WindowWidth"), "must read the window: {s}");
        assert!(s.contains("Substring"), "and truncate rather than tear: {s}");
    }

    /// A console without VT still has to produce a readable screen.
    #[test]
    fn everything_degrades_when_the_terminal_cannot_do_ansi() {
        let s = script_for(&claude(), &ctx_loaded(), Path::new("x"), &[]);

        assert!(s.contains("$Fancy = $false"), "a way to turn it off: {s}");
        assert!(s.contains("if ($Fancy)"), "and branches that honour it");
        assert!(s.contains(r"@('|','/','-','\')"), "an ASCII wheel for fallback");
    }

    #[test]
    fn every_shipped_provider_produces_a_usable_screen() {
        for spec in builtin_tools() {
            let s = script_for(&spec, &ctx_loaded(), Path::new("prog.cmd"), &spec.launch.args);

            assert!(s.contains("S A R A T H I"), "{}", spec.id);
            assert!(s.contains(&format!("$Provider   = '{}'", spec.name)), "{}", spec.id);
            assert!(s.contains("& $SarathiProgram"), "{}", spec.id);
            assert!(s.contains("DHARMA YATRA"), "{}", spec.id);
        }
    }

    #[test]
    fn a_live_process_is_distinguished_from_a_dead_one() {
        assert!(is_running(std::process::id()));
        assert!(!is_running(4_294_967_294));
    }
}
