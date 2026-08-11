//! Giving a launched tool a terminal of its own.
//!
//! ## Why this exists
//!
//! `main.rs` carries
//! `#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]`,
//! so a release build of Sarathi is a GUI process with **no console**. A child
//! spawned from it inherits that — which is to say, inherits nothing. Claude
//! Code, opencode and Hermes are terminal agents: with no console they have
//! nowhere to draw, and either exit at once or run invisibly. Either way the
//! user clicks Launch and sees nothing happen.
//!
//! A debug build is a *console* subsystem binary, so during development the
//! child attaches to the terminal `npm run tauri:dev` is running in and appears
//! to work. That difference is why this was easy to miss: the bug is invisible
//! in the environment it was written in and total in the one it ships to.
//!
//! `CREATE_NEW_CONSOLE` is the fix. It is not inherited and does not depend on
//! the parent having a console, so the tool gets a real window either way.
//!
//! ## Why a script rather than a command line
//!
//! The terminal has to open *saying* something — which model is loaded, which
//! gateway, which tool — before the agent takes the screen. Threading a banner
//! and a program path and its arguments through one `cmd /c "…"` string means
//! quoting them all against `cmd`'s parser at once. Writing a small script and
//! running that keeps each piece separate, and makes the result something a
//! test can read.

use std::path::{Path, PathBuf};

use crate::launcher::spec::{LaunchContext, ToolSpec};

/// Name of the generated script, written into the tool's own config directory
/// beside the config Sarathi already writes for it.
pub const SCRIPT_NAME: &str = "sarathi-launch.cmd";

/// Escapes a value for `echo` inside a batch file.
///
/// `cmd` treats `&`, `<`, `>`, `|` and `^` as syntax, and `%` as the start of a
/// variable. A model name is unlikely to contain any of them, but it comes from
/// a repository someone else named and is not Sarathi's to trust.
fn echo_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '^' | '&' | '<' | '>' | '|' => {
                out.push('^');
                out.push(ch);
            }
            '%' => out.push_str("%%"),
            // A control character would corrupt the script; a space is fine.
            c if c.is_control() => out.push(' '),
            c => out.push(c),
        }
    }
    out
}

/// One banner line, or a blank line for an empty string.
fn line(text: &str) -> String {
    if text.is_empty() {
        // `echo` with nothing after it prints "ECHO is on."; `echo(` prints a
        // blank line and is the idiom for it.
        "echo(".to_string()
    } else {
        format!("echo   {}", echo_escape(text))
    }
}

/// Builds the script that opens the terminal, says what it is, and runs the tool.
///
/// Every value comes from the live [`LaunchContext`] — the port the gateway
/// actually bound, and the model actually loaded — so the banner cannot claim a
/// model that is not serving.
pub fn script_for(spec: &ToolSpec, ctx: &LaunchContext, program: &Path, args: &[String]) -> String {
    let base = format!("http://127.0.0.1:{}", ctx.port);

    let mut s = String::new();
    s.push_str("@echo off\r\n");
    // The window's own title, so the tool is identifiable from the taskbar
    // without reading its contents.
    s.push_str(&format!("title {}\r\n", echo_escape(&title_for(&spec.name))));
    s.push_str("setlocal\r\n");

    for text in [
        "",
        "  Sarathi - local model gateway",
        "  ---------------------------------------------------------------",
        &format!("  Provider    Sarathi (local)  {base}"),
        &format!("  Model       {}", ctx.model_name),
        &format!("  Model id    {}", ctx.model_id),
        &format!("  Context     {} tokens", ctx.context_length),
        &format!("  Launching   {} ({})", spec.name, spec.protocol.label()),
        &format!("  MCP servers {}", ctx.mcp.servers.len()),
        "  ---------------------------------------------------------------",
        "  Requests from this tool are answered by the model above.",
        "  Nothing is sent to a hosted provider.",
        "",
    ] {
        s.push_str(&line(text));
        s.push_str("\r\n");
    }

    // `call` so a `.cmd` shim returns here instead of ending the script, which
    // is what lets the exit code be reported below.
    s.push_str(&format!("call \"{}\"", program.display()));
    for arg in args {
        s.push_str(&format!(" \"{arg}\""));
    }
    s.push_str("\r\n");

    // A tool that fails immediately is the thing that made this bug invisible:
    // the window would open and vanish before anything could be read. Holding it
    // on a non-zero exit is what turns a silent failure into a message.
    s.push_str("set SARATHI_EXIT=%ERRORLEVEL%\r\n");
    s.push_str("if not \"%SARATHI_EXIT%\"==\"0\" (\r\n");
    s.push_str(&line(""));
    s.push_str("\r\n");
    s.push_str(&format!(
        "  echo   {} exited with code %SARATHI_EXIT%.\r\n",
        echo_escape(&spec.name)
    ));
    s.push_str("  echo   The window is being held open so the error above can be read.\r\n");
    s.push_str("  pause\r\n");
    s.push_str(")\r\n");

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

/// The title the generated script gives its window.
///
/// Also how the window is found again: `start` returns as soon as it has handed
/// the script to a new console, so the pid Sarathi receives belongs to a wrapper
/// that is already gone. The window it opened is the thing that is still there.
pub fn title_for(tool_name: &str) -> String {
    format!("Sarathi - {tool_name}")
}

/// Whether a console window with this title is open.
///
/// This is what "is the tool still running?" means for a terminal agent: the
/// user closes the window when they are finished, and Sarathi is not told.
#[cfg(windows)]
pub fn window_open(title: &str) -> bool {
    let out = crate::system_analyzer::process_utils::create_hidden_command("tasklist")
        .args(["/FI", &format!("WINDOWTITLE eq {title}"), "/NH"])
        .output();

    match out {
        // With no match `tasklist` prints an INFO line rather than a row, so a
        // row is the signal. Matching on the image name avoids reading that
        // line as a result.
        Ok(o) => {
            let text = String::from_utf8_lossy(&o.stdout).to_ascii_lowercase();
            text.contains("cmd.exe") || text.contains("conhost")
        }
        // Unable to ask is not evidence that it closed.
        Err(_) => false,
    }
}

#[cfg(not(windows))]
pub fn window_open(_title: &str) -> bool {
    false
}

/// Whether a process id still belongs to a running process.
///
/// Used so a second click on Launch reconnects to the terminal already open
/// rather than starting a duplicate. A pid that has exited is treated as gone,
/// which is the safe direction: at worst a new window opens.
pub fn is_running(pid: u32) -> bool {
    #[cfg(windows)]
    {
        // `tasklist` filtered to the pid prints a header and nothing else when
        // there is no such process, so the pid has to appear in the output.
        let out = crate::system_analyzer::process_utils::create_hidden_command("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/NH"])
            .output();

        match out {
            Ok(o) => {
                let text = String::from_utf8_lossy(&o.stdout);
                text.contains(&pid.to_string())
            }
            // Unable to ask is not evidence that it stopped.
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
    use crate::launcher::spec::builtin_tools;

    fn ctx() -> LaunchContext {
        LaunchContext {
            port: 11435,
            model_id: "unsloth/gpt-oss-20b-GGUF".into(),
            model_name: "gpt-oss 20B".into(),
            client_dir: r"C:\Users\x\AppData\Roaming\com.sarathi.app\clients\claude-code".into(),
            context_length: 8192,
            mcp: crate::launcher::mcp::McpRegistry::default(),
        }
    }

    fn claude() -> ToolSpec {
        builtin_tools().into_iter().find(|t| t.id == "claude-code").expect("shipped")
    }

    /// The three things the terminal has to say, all from live state.
    #[test]
    fn the_banner_names_sarathi_the_model_and_the_tool() {
        let s = script_for(&claude(), &ctx(), Path::new(r"C:\npm\claude.cmd"), &[]);

        assert!(s.contains("Sarathi - local model gateway"), "{s}");
        assert!(s.contains("gpt-oss 20B"), "the loaded model, not a placeholder: {s}");
        assert!(s.contains("unsloth/gpt-oss-20b-GGUF"), "{s}");
        assert!(s.contains("Claude Code"), "which tool is being launched: {s}");
        assert!(s.contains("http://127.0.0.1:11435"), "the gateway it is connected to: {s}");
    }

    /// The model is whatever is loaded now. A banner that could print a stale or
    /// invented name would be worse than none.
    #[test]
    fn the_banner_follows_the_live_context() {
        let mut other = ctx();
        other.model_name = "Qwen2.5 Coder 7B".into();
        other.model_id = "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF".into();
        other.port = 9999;

        let s = script_for(&claude(), &other, Path::new("x"), &[]);

        assert!(s.contains("Qwen2.5 Coder 7B"));
        assert!(s.contains("Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"));
        assert!(s.contains("http://127.0.0.1:9999"));
        assert!(!s.contains("gpt-oss"), "nothing from another launch may survive: {s}");
        assert!(!s.contains("11435"), "{s}");
    }

    /// The program and its arguments are quoted. A path through
    /// `C:\Program Files\` splits on the space otherwise, and the tool never runs.
    #[test]
    fn the_program_and_arguments_are_quoted() {
        let s = script_for(
            &claude(),
            &ctx(),
            Path::new(r"C:\Program Files\npm\claude.cmd"),
            &["--mcp-config".into(), r"C:\App Data\clients\claude-code\mcp.json".into()],
        );

        assert!(s.contains(r#"call "C:\Program Files\npm\claude.cmd""#), "{s}");
        assert!(s.contains(r#""C:\App Data\clients\claude-code\mcp.json""#), "{s}");
    }

    /// `call`, not a bare invocation: a `.cmd` shim would otherwise end the
    /// script and the exit code could never be reported.
    #[test]
    fn a_cmd_shim_returns_to_the_script() {
        let s = script_for(&claude(), &ctx(), Path::new("claude.cmd"), &[]);
        assert!(s.contains("call \""), "{s}");
        assert!(s.contains("%ERRORLEVEL%"), "the exit code has to be captured: {s}");
    }

    /// The failure this whole module exists to make visible: a tool that dies at
    /// once must leave its error on screen rather than closing over it.
    #[test]
    fn a_failing_tool_holds_the_window_open() {
        let s = script_for(&claude(), &ctx(), Path::new("claude.cmd"), &[]);

        assert!(s.contains("pause"), "{s}");
        assert!(s.contains("exited with code"), "and says what happened: {s}");
    }

    /// A window that fails cleanly must not pause — the user closed the agent
    /// deliberately and does not need to dismiss a prompt about it.
    #[test]
    fn a_clean_exit_does_not_hold_the_window() {
        let s = script_for(&claude(), &ctx(), Path::new("claude.cmd"), &[]);
        assert!(
            s.contains("if not \"%SARATHI_EXIT%\"==\"0\""),
            "the pause has to be conditional: {s}"
        );
    }

    /// Values reaching `echo` come from repository metadata, so cmd's operators
    /// have to be neutralised rather than trusted not to appear.
    #[test]
    fn cmd_operators_in_a_model_name_cannot_break_the_script() {
        let mut hostile = ctx();
        hostile.model_name = "evil & echo pwned > file.txt | more %PATH%".into();

        let s = script_for(&claude(), &hostile, Path::new("claude.cmd"), &[]);

        // Every operator that reaches the file has to be carrying its caret.
        // Asserting on the escaped form alone would pass even if a second,
        // unescaped copy were also present, so this walks the text.
        let chars: Vec<char> = s.chars().collect();
        for (i, c) in chars.iter().enumerate() {
            if matches!(c, '&' | '<' | '>' | '|') {
                assert_eq!(
                    chars.get(i.wrapping_sub(1)),
                    Some(&'^'),
                    "unescaped {c} at {i} in: {s}"
                );
            }
        }

        assert!(s.contains("%%PATH%%"), "a variable must not expand: {s}");
    }

    #[test]
    fn every_shipped_tool_produces_a_usable_script() {
        for spec in builtin_tools() {
            let s = script_for(&spec, &ctx(), Path::new("prog.cmd"), &spec.launch.args);

            assert!(s.starts_with("@echo off"), "{}: {s}", spec.id);
            assert!(s.contains(&spec.name), "{}: banner must name the tool", spec.id);
            assert!(s.contains("gpt-oss 20B"), "{}: banner must name the model", spec.id);
            assert!(s.contains("call \"prog.cmd\""), "{}: {s}", spec.id);
        }
    }

    /// This process is certainly running, and a pid that high is certainly not.
    #[test]
    fn a_live_process_is_distinguished_from_a_dead_one() {
        assert!(is_running(std::process::id()));
        assert!(!is_running(4_294_967_294));
    }
}
