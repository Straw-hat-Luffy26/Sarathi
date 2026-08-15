//! Launch section — start a coding tool already connected to the gateway.
//!
//! The point is that a non-technical user never copies a URL or opens a
//! terminal: press Launch and the tool comes up talking to the local model.
//!
//! Connection is made **at launch only**, by handing the child process the
//! gateway address in its environment. No config file is written, so nothing
//! of the user's is modified and there is nothing to undo. The trade-off is
//! deliberate: opening the tool by other means will not be connected.

pub mod console;
pub mod mcp;
pub mod registry;
pub mod spec;

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::launcher::spec::{
    fill_placeholders_with, output_identifies_tool, resolve_args, resolve_env, LaunchContext,
    PackageManager, ToolSpec,
};
use crate::system_analyzer::process_utils::create_hidden_command;

/// What state a tool is in, and therefore what the card offers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "state", rename_all = "camelCase")]
pub enum ToolState {
    /// Nobody has looked yet. Distinct from `NotInstalled` because "we do not
    /// know" and "it is not there" put different things on the card, and
    /// showing the second while meaning the first offers an Install button for
    /// something already installed.
    Checking,
    /// Checked and absent — offer Install.
    NotInstalled,
    /// Present and identified — offer Launch.
    Ready { version: String },
    /// Started by Sarathi and still alive — offer Stop.
    Running { pid: u32 },
    /// Something is wrong that the user must see, e.g. the package manager is
    /// missing. Kept separate from `NotInstalled` because the fix differs.
    Problem { reason: String },
}

/// A tool plus its current state, as sent to the UI.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolStatus {
    pub id: String,
    pub name: String,
    pub description: String,
    pub protocol: spec::Protocol,
    pub user_defined: bool,
    /// Command line Install would run, for the confirmation prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub install_command: Option<String>,
    #[serde(flatten)]
    pub state: ToolState,
}

/// Tracks processes Sarathi started, so cards can show Running and offer Stop.
#[derive(Default)]
pub struct LaunchedProcesses {
    /// Tool id to the pid Sarathi received and the title of the window it
    /// opened. Both are kept because neither answers on its own: on Windows the
    /// pid belongs to a `start` wrapper that exits immediately, and elsewhere
    /// there is no window to look for.
    inner: Mutex<HashMap<String, (u32, String)>>,
}

impl LaunchedProcesses {
    pub fn record(&self, tool_id: &str, pid: u32, window_title: &str) {
        if let Ok(mut m) = self.inner.lock() {
            m.insert(tool_id.to_string(), (pid, window_title.to_string()));
        }
    }

    pub fn forget(&self, tool_id: &str) {
        if let Ok(mut m) = self.inner.lock() {
            m.remove(tool_id);
        }
    }

    pub fn pid_of(&self, tool_id: &str) -> Option<u32> {
        self.inner.lock().ok()?.get(tool_id).map(|(pid, _)| *pid)
    }

    pub fn running_count(&self) -> usize {
        self.inner.lock().map(|m| m.len()).unwrap_or(0)
    }

    /// The tool's process, if one is tracked *and* still alive.
    ///
    /// A pid alone is not enough to answer "is it running": the user may have
    /// closed the terminal, which Sarathi is not told about. A stale entry is
    /// dropped here so the next launch opens a window rather than reporting a
    /// process that no longer exists.
    pub fn live_pid(&self, tool_id: &str) -> Option<u32> {
        let (pid, title) = self.inner.lock().ok()?.get(tool_id).cloned()?;

        // Either signal is enough. The window is the reliable one on Windows,
        // where the pid belongs to a wrapper that has already exited; the pid
        // covers the platforms that open no window of their own.
        if console::window_open(&title) || console::is_running(pid) {
            return Some(pid);
        }
        self.forget(tool_id);
        None
    }
}

/// Tool detection, remembered between refreshes of the Launch screen.
///
/// Detecting one tool costs a `where` scan and a `--version` run: on the
/// machine this was measured on, 1.8s and 3.2s respectively, per tool, because
/// each of these programs is a Node or Bun bundle that has to boot before it
/// will say its own version. Four shipped tools made that roughly twenty
/// seconds — and the Launch screen was re-running all of it every two seconds,
/// so refreshes overlapped, the blocking pool filled with `where` processes,
/// and the first paint the user was waiting for never arrived. That is the
/// "Checking what's installed…" that lasted minutes.
///
/// Nothing about the answer changes on a two-second timescale, so it is
/// computed once — in the background, at startup, before anyone opens the
/// screen — and re-computed only when it goes stale or when Sarathi does
/// something that could have changed it.
pub struct DetectionCache {
    entries: Mutex<HashMap<String, (std::time::Instant, ToolState)>>,
    /// Set while a background refresh is running, so a burst of polls produces
    /// one sweep rather than one each.
    refreshing: std::sync::atomic::AtomicBool,
}

/// How long a detection stands before it is refreshed in the background. Long
/// enough that polling is free, short enough that an install done in a terminal
/// shows up without restarting Sarathi.
const DETECTION_TTL: std::time::Duration = std::time::Duration::from_secs(30);

impl Default for DetectionCache {
    fn default() -> Self {
        Self { entries: Mutex::new(HashMap::new()), refreshing: false.into() }
    }
}

impl DetectionCache {
    /// The best answer available right now, with no waiting.
    ///
    /// Returns `Checking` for a tool never yet detected and the previous answer
    /// for one whose detection has gone stale, and starts a background sweep
    /// when either is true. It never runs a subprocess on the caller's thread.
    pub fn states(&self, specs: &[ToolSpec]) -> (Vec<ToolState>, bool) {
        let now = std::time::Instant::now();
        let entries = self.entries.lock().expect("detection cache poisoned");

        let mut stale = false;
        let states = specs
            .iter()
            .map(|spec| match entries.get(&spec.id) {
                Some((at, state)) if now.duration_since(*at) < DETECTION_TTL => state.clone(),
                Some((_, state)) => {
                    stale = true;
                    state.clone()
                }
                None => {
                    stale = true;
                    ToolState::Checking
                }
            })
            .collect();

        (states, stale)
    }

    /// Detects every tool, concurrently, and stores the results.
    ///
    /// Concurrent because the four probes share nothing: run one after another
    /// they add up, run together they cost whatever the slowest one costs.
    pub fn refresh(&self, specs: &[ToolSpec]) {
        use std::sync::atomic::Ordering;
        if self.refreshing.swap(true, Ordering::SeqCst) {
            return;
        }
        let _release = scopeguard(|| self.refreshing.store(false, Ordering::SeqCst));
        let _stage = crate::diagnostics::Stage::new("launcher: detect every tool");

        let detected: Vec<(String, ToolState)> = std::thread::scope(|scope| {
            let handles: Vec<_> = specs
                .iter()
                .map(|spec| scope.spawn(move || (spec.id.clone(), detect(spec))))
                .collect();
            handles.into_iter().filter_map(|h| h.join().ok()).collect()
        });

        let now = std::time::Instant::now();
        let mut entries = self.entries.lock().expect("detection cache poisoned");
        for (id, state) in detected {
            entries.insert(id, (now, state));
        }
    }

    /// Forgets one tool, so the next look re-detects it.
    ///
    /// Called after an install or a launch: those are the moments the cached
    /// answer is known to be wrong, and waiting out the TTL would show the user
    /// the state before the thing they just did.
    pub fn invalidate(&self, tool_id: &str) {
        self.entries.lock().expect("detection cache poisoned").remove(tool_id);
    }

    /// Forgets everything, for the Refresh button.
    ///
    /// Refresh has to mean refresh. Serving the cache to somebody who pressed
    /// it would be the same lie in the other direction from the one the cache
    /// fixes.
    pub fn forget_all(&self) {
        self.entries.lock().expect("detection cache poisoned").clear();
    }

    /// Runs the first sweep at startup, off the UI thread.
    pub fn warm(app: &tauri::AppHandle) {
        use tauri::Manager;
        let app = app.clone();
        tauri::async_runtime::spawn(async move {
            let Some(cache) = app.try_state::<std::sync::Arc<DetectionCache>>() else { return };
            let cache = cache.inner().clone();
            let specs = match app.path().app_data_dir() {
                Ok(dir) => registry::load(&dir).tools,
                Err(_) => spec::builtin_tools(),
            };
            let _ = tokio::task::spawn_blocking(move || cache.refresh(&specs)).await;
        });
    }
}

/// Runs a closure on the way out of a scope, whatever happens in it.
fn scopeguard<F: FnMut()>(f: F) -> impl Drop {
    struct Guard<F: FnMut()>(F);
    impl<F: FnMut()> Drop for Guard<F> {
        fn drop(&mut self) {
            (self.0)();
        }
    }
    Guard(f)
}

/// Locates a real executable for `command`.
///
/// Uses `where` on Windows and `command -v` elsewhere, then requires the result
/// to be an actual file. A shell built-in resolves to no path, which is how the
/// `continue` false positive is excluded before anything is run.
fn resolve_executable(command: &str) -> Option<PathBuf> {
    let (finder, args) = if cfg!(windows) {
        ("where", vec![command])
    } else {
        ("sh", vec!["-c"])
    };

    // Hidden: probing runs on every overview refresh, and a GUI app must not
    // flash a console window for each tool it checks.
    let output = if cfg!(windows) {
        create_hidden_command(finder).args(&args).output().ok()?
    } else {
        create_hidden_command("sh")
            .args(["-c", &format!("command -v {command}")])
            .output()
            .ok()?
    };

    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let candidates: Vec<PathBuf> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(PathBuf::from)
        .filter(|p| p.is_file())
        .collect();

    // `where` lists the extensionless shim first — npm and claude both ship one
    // next to their `.cmd`. That shim is a real file, so it passes every check
    // here, but CreateProcess cannot run it: spawning a bare name only ever
    // appends `.exe`. Picking it is what produced "program not found" at the
    // moment of install, long after detection had reported success.
    if cfg!(windows) {
        let runnable = candidates.iter().find(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "exe" | "cmd" | "bat"))
        });
        if let Some(path) = runnable {
            return Some(path.clone());
        }
    }

    candidates.into_iter().next()
}

/// Checks whether a tool is genuinely installed.
///
/// Three steps, all of which must pass:
/// 1. a real executable exists — rules out shell built-ins;
/// 2. running it with its version argument succeeds;
/// 3. the output names the tool — rules out an unrelated program of the same name.
pub fn detect(spec: &ToolSpec) -> ToolState {
    let Some(exe) = resolve_executable(&spec.detect.command) else {
        return ToolState::NotInstalled;
    };

    let output = create_hidden_command(&exe)
        .arg(&spec.detect.version_arg)
        .stdin(Stdio::null())
        .output();

    let Ok(out) = output else {
        return ToolState::NotInstalled;
    };

    // Some tools print their version to stderr.
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    if !output_identifies_tool(&combined, &spec.detect.expect) {
        return ToolState::NotInstalled;
    }

    let version = combined.lines().next().unwrap_or("").trim().to_string();
    ToolState::Ready { version }
}

/// True when the package manager an entry needs is available.
///
/// Memoised for the life of the process. This is a `where` scan, and it was
/// running once per tool on every refresh of a screen that refreshes every two
/// seconds — asking, four times over, whether npm is still installed. A package
/// manager appearing while Sarathi is open is worth a restart; twenty seconds
/// of subprocess a minute is not.
pub fn manager_available(manager: PackageManager) -> bool {
    static KNOWN: Mutex<Option<Vec<(PackageManager, bool)>>> = Mutex::new(None);

    let mut known = KNOWN.lock().expect("manager cache poisoned");
    let cache = known.get_or_insert_with(Vec::new);
    if let Some((_, found)) = cache.iter().find(|(m, _)| *m == manager) {
        return *found;
    }
    let found = resolve_executable(manager.program()).is_some();
    cache.push((manager, found));
    found
}

/// Installs a tool through its package manager.
///
/// Sarathi never downloads an installer itself — it delegates to a manager the
/// user already has, which verifies its own sources. The caller is responsible
/// for showing [`PackageManager::command_line`] and getting consent first.
///
/// `on_line` receives each line the package manager prints, as it prints it, so
/// the caller can show real progress instead of an indefinite spinner.
pub fn install(spec: &ToolSpec, mut on_line: impl FnMut(&str)) -> Result<(), String> {
    let Some(install) = &spec.install else {
        return Err(format!("{} cannot be installed by Sarathi", spec.name));
    };

    let Some(manager_exe) = resolve_executable(install.manager.program()) else {
        return Err(format!(
            "{} is needed to install {}, but it is not available on this machine.",
            install.manager.program(),
            spec.name
        ));
    };

    let mut child = create_hidden_command(&manager_exe)
        .args(install.manager.install_args(&install.package))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not run {}: {e}", install.manager.program()))?;

    // Streamed rather than collected at the end: an install pulls megabytes and
    // takes a minute, and a button that only says "Installing…" for that long
    // is indistinguishable from one that has hung. npm reports progress on
    // stderr and results on stdout, so both are pumped into one channel.
    let (tx, rx) = std::sync::mpsc::channel::<String>();

    let stderr_pipe = child.stderr.take();
    let tx_err = tx.clone();
    let stderr_thread = std::thread::spawn(move || {
        if let Some(pipe) = stderr_pipe {
            for line in BufReader::new(pipe).lines().map_while(Result::ok) {
                let _ = tx_err.send(line);
            }
        }
    });

    let stdout_pipe = child.stdout.take();
    let stdout_thread = std::thread::spawn(move || {
        if let Some(pipe) = stdout_pipe {
            for line in BufReader::new(pipe).lines().map_while(Result::ok) {
                let _ = tx.send(line);
            }
        }
    });

    // Ends when both senders drop, i.e. both pipes reached EOF.
    let mut tail: Vec<String> = Vec::new();
    for line in rx {
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        on_line(&line);
        tail.push(line);
        if tail.len() > 3 {
            tail.remove(0);
        }
    }

    let _ = stdout_thread.join();
    let _ = stderr_thread.join();

    let status = child
        .wait()
        .map_err(|e| format!("could not run {}: {e}", install.manager.program()))?;

    if status.success() {
        return Ok(());
    }

    // Surface the manager's own message — it usually says exactly what is wrong.
    Err(format!(
        "{} failed to install {}: {}",
        install.manager.program(),
        spec.name,
        if tail.is_empty() { "no details reported".to_string() } else { tail.join(" ") }
    ))
}

/// Starts a tool with the gateway address in its environment.
///
/// Returns the child process id. The child is detached — closing Sarathi does
/// not kill a terminal the user is working in.
pub fn launch(spec: &ToolSpec, ctx: &LaunchContext, workspace: &std::path::Path) -> Result<u32, String> {
    let env = resolve_env(&spec.launch.env, ctx);

    // Resolved rather than spawned by bare name, for the same reason as detect:
    // on Windows these tools are `.cmd` shims. Not hidden — a terminal agent
    // needs its console.
    let program = resolve_executable(&spec.launch.command)
        .ok_or_else(|| format!("could not start {}: not found on PATH", spec.name))?;

    // Written fresh on every launch, so a model changed in Sarathi is picked up
    // the next time the tool starts rather than persisting from a stale file.
    // The same applies to MCP servers: a server added to `mcp.json` reaches
    // every tool on its next launch without either being reconfigured.
    if let Some(config) = &spec.launch.client_config {
        let dir = std::path::Path::new(&ctx.client_dir);
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("could not prepare {}'s config directory: {e}", spec.name))?;

        let body = fill_placeholders_with(&config.contents, ctx, true, config.mcp_dialect);
        std::fs::write(dir.join(&config.file_name), body)
            .map_err(|e| format!("could not write {}'s config: {e}", spec.name))?;
    }

    std::fs::create_dir_all(workspace)
        .map_err(|e| format!("could not prepare the Sarathi workspace: {e}")) ?;

    // Filled, not passed raw: an argument may name a generated config file, and
    // that path is only known once the client directory is.
    let args = resolve_args(&spec.launch.args, ctx);

    // On Windows the tool is run through a generated script in its own console.
    //
    // Sarathi is a GUI process in release builds, so it has no console for a
    // child to inherit — and these tools are terminal agents with nowhere to
    // draw without one. Handing the child `CREATE_NEW_CONSOLE` gives it a real
    // window whether or not Sarathi has one, and the script prints which model
    // and gateway it is connected to before the agent takes the screen.
    #[cfg(windows)]
    let mut cmd = {
        let script = console::write_script(
            std::path::Path::new(&ctx.client_dir),
            spec,
            ctx,
            &program,
            &args,
        )?;

        // `start`, not `CREATE_NEW_CONSOLE`.
        //
        // The flag alone is not enough. Rust's `Command` always sets
        // STARTF_USESTDHANDLES and, for the default inherit, fills it with the
        // parent's handles — so the child gets a new console window *and* keeps
        // writing to the old one, which is empty in a GUI build. Observed
        // directly: the banner appeared in the launching terminal instead of the
        // new window.
        //
        // `start` hands the job to the shell, which detaches the child properly
        // and gives it the new console's own handles. The empty first argument
        // is `start`'s title parameter — without it, a quoted script path is
        // taken as the title and nothing runs.
        let mut cmd = std::process::Command::new("cmd.exe");
        // `start` detaches the child into its own console with its own
        // handles; `CREATE_NEW_CONSOLE` alone does not, because Rust's Command
        // always fills STARTF_USESTDHANDLES with the parent's handles and the
        // child keeps writing to those. The empty argument is `start`'s title
        // parameter — without it a quoted path is taken as the title.
        //
        // PowerShell rather than cmd: the startup screen needs ANSI, a timer and
        // the real window size, and it has to `&` the provider afterwards so the
        // agent inherits this console rather than another one.
        cmd.arg("/c")
            .arg("start")
            .arg("")
            .arg("powershell")
            .arg("-NoProfile")
            .arg("-ExecutionPolicy")
            .arg("Bypass")
            .arg("-File")
            .arg(&script);
        cmd
    };

    // Elsewhere the terminal the user launched Sarathi from is the console, and
    // opening another would need a terminal emulator this cannot assume exists.
    #[cfg(not(windows))]
    let mut cmd = {
        let mut cmd = std::process::Command::new(&program);
        cmd.args(&args);
        cmd
    };

    cmd.current_dir(workspace);

    // Stripped before setting ours: a tool that reads a provider key from the
    // user's environment will use it regardless of what Sarathi supplies.
    for key in &spec.launch.env_remove {
        cmd.env_remove(key);
    }
    for (k, v) in &env {
        cmd.env(k, v);
    }

    let child = cmd
        .spawn()
        .map_err(|e| format!("could not start {}: {e}", spec.name))?;

    Ok(child.id())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launcher::spec::{builtin_tools, DetectSpec, LaunchSpec, Protocol};

    fn spec_for(command: &str, expect: &str) -> ToolSpec {
        ToolSpec {
            id: command.into(),
            name: command.into(),
            description: String::new(),
            protocol: Protocol::OpenAi,
            detect: DetectSpec {
                command: command.into(),
                version_arg: "--version".into(),
                expect: expect.into(),
            },
            install: None,
            launch: LaunchSpec {
                command: command.into(),
                args: vec![],
                env: HashMap::new(),
                env_remove: vec![],
                client_config: None,
            },
            mcp: crate::launcher::spec::McpSupport::default(),
            min_context: None,
            user_defined: false,
        }
    }

    #[test]
    fn a_command_that_does_not_exist_is_not_installed() {
        let state = detect(&spec_for("definitely-not-a-real-program-xyz", "xyz"));
        assert_eq!(state, ToolState::NotInstalled);
    }

    /// Regression: a naive PATH lookup reported Continue as installed, because
    /// `continue` is a shell keyword. Requiring a real executable file excludes it.
    #[test]
    fn shell_builtins_are_not_mistaken_for_installed_tools() {
        for builtin in ["continue", "break", "return"] {
            assert_eq!(
                detect(&spec_for(builtin, builtin)),
                ToolState::NotInstalled,
                "'{builtin}' is a shell keyword, not an installed tool"
            );
        }
    }

    #[test]
    fn a_real_program_with_the_wrong_identity_is_not_installed() {
        // A program exists at this name, but it is not the tool we want.
        let cmd = if cfg!(windows) { "cmd" } else { "sh" };
        let state = detect(&spec_for(cmd, "definitely-not-in-that-output"));

        assert_eq!(state, ToolState::NotInstalled, "identity check must reject it");
    }

    #[test]
    fn npm_is_detected_on_this_machine() {
        // npm is present here, so this exercises the success path end to end.
        assert!(manager_available(PackageManager::Npm));
    }

    #[test]
    fn a_tool_with_no_install_route_says_so_rather_than_failing_oddly() {
        let spec = spec_for("something", "something");
        assert!(spec.install.is_none());

        let err = install(&spec, |_| {}).unwrap_err();
        assert!(err.contains("cannot be installed"), "got: {err}");
    }

    #[test]
    fn launching_a_missing_program_reports_the_tool_name() {
        let spec = spec_for("definitely-not-a-real-program-xyz", "xyz");

        let ctx = LaunchContext {
            port: 11435,
            model_id: "Qwen/Qwen2.5-3B".into(),
            model_name: "Qwen2.5-3B".into(),
            client_dir: std::env::temp_dir().join("sarathi-test-client").to_string_lossy().into(),
            context_length: 8192,
            mcp: crate::launcher::mcp::McpRegistry::default(),
            runtime: crate::launcher::spec::RuntimeSnapshot::default(),
        };
        let err = launch(&spec, &ctx, &std::env::temp_dir()).unwrap_err();
        assert!(err.contains("could not start"), "got: {err}");
        assert!(err.contains(&spec.name), "the message should name the tool: {err}");
    }

    #[test]
    fn running_processes_are_tracked_per_tool() {
        let procs = LaunchedProcesses::default();
        assert_eq!(procs.running_count(), 0);
        assert_eq!(procs.pid_of("claude-code"), None);

        procs.record("claude-code", 1234, "Sarathi - Claude Code");
        assert_eq!(procs.pid_of("claude-code"), Some(1234));
        assert_eq!(procs.running_count(), 1);

        procs.forget("claude-code");
        assert_eq!(procs.pid_of("claude-code"), None);
        assert_eq!(procs.running_count(), 0);
    }

    #[test]
    fn detecting_every_shipped_tool_never_panics() {
        // Runs the real detection path against the real machine. Claude Code is
        // installed here and opencode is not, so both branches are exercised.
        for spec in builtin_tools() {
            let state = detect(&spec);
            match state {
                ToolState::Ready { ref version } => {
                    assert!(!version.is_empty(), "{} reported ready with no version", spec.id)
                }
                ToolState::NotInstalled => {}
                other => panic!("unexpected state for {}: {other:?}", spec.id),
            }
        }
    }
}
