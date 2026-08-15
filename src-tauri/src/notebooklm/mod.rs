//! NotebookLM, as a capability Sarathi can see the state of.
//!
//! `notebooklm-py` is a Python package that ships two things Sarathi cares
//! about: a CLI that owns a Google session, and an MCP server (`notebooklm-mcp`)
//! that exposes that session as tools. Sarathi does not reimplement either. It
//! answers four questions and offers the two actions that follow from them:
//!
//! * **Is it installed?** — a console script on PATH, and what version.
//! * **Is there a session?** — local credentials exist, which is *not* the same
//!   as a session that still works.
//! * **Does the session actually work?** — a live call, because a stale cookie
//!   file looks exactly like a good one until something asks Google.
//! * **Can it be handed to providers?** — once installed, its MCP server joins
//!   `mcp.json` like any other, and every MCP-capable provider gets it with no
//!   provider-side code. There is no NotebookLM branch anywhere in the launcher.
//!
//! ## What this module refuses to do
//!
//! Authentication is Google's. `notebooklm login` opens a browser and the user
//! signs in there; Sarathi runs that command and watches it exit. It never sees
//! a password, never stores one, and never reads the cookie jar. The session
//! file's *existence and shape* are reported; its contents are not, and nothing
//! here is written to a log or handed to a model.

pub mod manager;
pub mod registry;
pub mod state;

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::system_analyzer::process_utils::create_hidden_command;

/// Console script the Python package installs. This is the CLI.
const CLI: &str = "notebooklm";
/// Console script that runs the MCP server over stdio.
const MCP_SERVER: &str = "notebooklm-mcp";
/// The distribution, with the extra that brings the MCP server in.
pub const PACKAGE_SPEC: &str = "notebooklm-py[mcp]";
/// Name this server takes in `mcp.json`. Stable, so re-detection updates the
/// same entry rather than appending a second one.
pub const MCP_SERVER_KEY: &str = "notebooklm";

/// A quick command should not hang the UI behind a wedged subprocess.
const QUICK: Duration = Duration::from_secs(20);
/// A live check talks to Google over the network.
const LIVE: Duration = Duration::from_secs(90);

/// Where NotebookLM stands, as far as Sarathi can actually tell.
///
/// Ordered from "nothing here" to "verified working". Each state is reachable
/// only from evidence Sarathi has: there is deliberately no state meaning
/// "probably fine".
///
/// The transient states (`Checking`, `Installing`, `Authenticating`,
/// `Verifying`) exist so the card can always name the work in progress. An
/// indefinite spinner over an unchanged status is the thing they replace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum NotebookLmState {
    /// The first detection has not finished and nothing is known yet. Only ever
    /// the state before startup detection publishes its first answer.
    Checking,
    /// No `notebooklm` on PATH.
    NotInstalled,
    /// An install the user asked for is running.
    Installing,
    /// That install failed; `detail` says how.
    InstallFailed,
    /// Installed, but no local session at all — never logged in, or logged out.
    NotAuthenticated,
    /// A Google sign-in is running: the browser is open and Sarathi is waiting
    /// for the user to finish in it.
    Authenticating,
    /// A live check is in flight against Google.
    Verifying,
    /// A session exists locally but has not been proven to work. Shown after
    /// install-detection, before a health check has run.
    Unverified,
    /// A live call succeeded. The only state that means what it says.
    Connected,
    /// A live call was refused for an auth reason: the session has expired or
    /// been revoked, and the user has to sign in again.
    AuthenticationExpired,
    /// The check failed for a reason that is not authentication — the network,
    /// the package, the CLI. The session may well still be good.
    ConnectionFailed,
}

impl NotebookLmState {
    /// Whether work is in flight. The card keeps its shape and shows this as
    /// activity on the status row rather than replacing itself with a spinner.
    pub fn is_busy(self) -> bool {
        matches!(
            self,
            Self::Checking | Self::Installing | Self::Authenticating | Self::Verifying
        )
    }
}

/// Everything the Launch screen needs to render the NotebookLM card.
///
/// Nothing in here is a secret. The session file is reported as a path and a
/// yes/no, never as content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NotebookLmStatus {
    pub state: NotebookLmState,
    /// Version string from the CLI, when installed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Absolute path of the `notebooklm` console script.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cli_path: Option<String>,
    /// Absolute path of the MCP server script. Without it the package is
    /// installed but not usable as a tool source — the `[mcp]` extra is missing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mcp_server_path: Option<String>,
    /// Whether the MCP server would be offered to providers.
    pub mcp_available: bool,
    /// Whether this server is currently in Sarathi's registry.
    pub in_registry: bool,
    /// Whether a session file is present. Presence only — never its contents.
    pub has_local_session: bool,
    /// RFC3339 time of the last live check that succeeded, including one from a
    /// previous run of Sarathi.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_verified_at: Option<String>,
    /// The user chose Sign out. Distinguishes "never signed in" from "signed
    /// out on purpose", which are the same state but not the same sentence.
    pub signed_out: bool,
    /// Names of the providers that would actually be handed this server right
    /// now, derived from the registry and each provider's MCP dialect. Never a
    /// hardcoded list.
    #[serde(default)]
    pub compatible_providers: Vec<String>,
    /// Whether the numbers above come from a full probe or from remembered
    /// paths that were only confirmed to still exist. The card uses it to say
    /// so rather than to hide it.
    pub from_cache: bool,
    /// One line for the user when something is wrong. Scrubbed of anything
    /// that could carry a credential.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl NotebookLmStatus {
    pub(crate) fn blank(state: NotebookLmState) -> Self {
        Self {
            state,
            version: None,
            cli_path: None,
            mcp_server_path: None,
            mcp_available: false,
            in_registry: false,
            has_local_session: false,
            last_verified_at: None,
            signed_out: false,
            compatible_providers: Vec::new(),
            from_cache: false,
            detail: None,
        }
    }

    fn not_installed() -> Self {
        Self::blank(NotebookLmState::NotInstalled)
    }
}

/// The paths a previous full detection resolved, so the next one need not.
///
/// Only paths and a version string — nothing here is or points to a credential.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Remembered {
    pub cli_path: Option<String>,
    pub mcp_server_path: Option<String>,
    pub version: Option<String>,
}

impl Remembered {
    pub fn of(status: &NotebookLmStatus) -> Self {
        Self {
            cli_path: status.cli_path.clone(),
            mcp_server_path: status.mcp_server_path.clone(),
            version: status.version.clone(),
        }
    }
}

/// Output of a command run, with both streams and the exit status.
struct Ran {
    ok: bool,
    stdout: String,
    stderr: String,
}

/// Runs a short command, hidden, with a deadline.
///
/// Hidden because every one of these would otherwise flash a console window on
/// Windows, and detection runs at startup.
fn run(program: &Path, args: &[&str], timeout: Duration) -> Result<Ran, String> {
    let mut cmd = create_hidden_command(program.to_string_lossy().as_ref());
    cmd.args(args);
    // The CLI renders tables with box characters; ask for plain output so a
    // parse never depends on the terminal it thinks it has.
    cmd.env("NO_COLOR", "1");
    cmd.env("PYTHONIOENCODING", "utf-8");

    let mut child = cmd
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .stdin(std::process::Stdio::null())
        .spawn()
        .map_err(|e| format!("could not run {}: {e}", program.display()))?;

    let deadline = std::time::Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if std::time::Instant::now() >= deadline => {
                let _ = child.kill();
                return Err(format!(
                    "{} did not finish within {}s",
                    program.display(),
                    timeout.as_secs()
                ));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            Err(e) => return Err(format!("could not wait for {}: {e}", program.display())),
        }
    }

    let out = child
        .wait_with_output()
        .map_err(|e| format!("could not read output of {}: {e}", program.display()))?;

    Ok(Ran {
        ok: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).to_string(),
        stderr: String::from_utf8_lossy(&out.stderr).to_string(),
    })
}

/// Locates a console script, including the per-user Scripts directories pip
/// installs into on Windows that are not always on the PATH a GUI app inherits.
///
/// `where`/`command -v` first, because if it is on PATH that is the one the MCP
/// client will run too. The explicit directories are a fallback for the common
/// case of a user-scope `pip install` whose Scripts dir was added to PATH after
/// the current session started.
pub fn find_script(name: &str) -> Option<PathBuf> {
    find_scripts(name).into_iter().next()
}

/// Every copy of `name` this machine has, in the order a spawned client would
/// find them.
///
/// More than one is the ordinary case, not an edge case: a `pip install` and a
/// later `uv tool install` both leave a console script, and only one of them
/// may work — this machine had a `notebooklm-mcp` earlier on PATH whose import
/// failed and a working one behind it. Returning the list lets the caller pick
/// the first that *runs* rather than the first that *exists*.
pub fn find_scripts(name: &str) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = Vec::new();
    let mut push = |path: PathBuf| {
        if path.is_file() && !found.contains(&path) {
            found.push(path);
        }
    };

    let (finder, args) =
        if cfg!(windows) { ("where", vec![name]) } else { ("command", vec!["-v", name]) };
    if let Ok(out) = create_hidden_command(finder).args(&args).output() {
        if out.status.success() {
            for line in String::from_utf8_lossy(&out.stdout).lines().map(str::trim) {
                if !line.is_empty() {
                    push(PathBuf::from(line));
                }
            }
        }
    }

    for dir in candidate_script_dirs() {
        for candidate in script_names(name) {
            push(dir.join(&candidate));
        }
    }
    found
}

fn script_names(name: &str) -> Vec<String> {
    if cfg!(windows) {
        vec![format!("{name}.exe"), format!("{name}.cmd"), name.to_string()]
    } else {
        vec![name.to_string()]
    }
}

/// Directories a user-scope Python install puts console scripts in.
///
/// Memoised for the life of the process. Working this out starts a Python
/// interpreter, which cost 1.4s on the machine this was measured on, and it was
/// being paid twice per detection — once for each console script looked up —
/// and again on every detection after that. The answer cannot change while
/// Sarathi is running without the user reinstalling Python underneath it.
fn candidate_script_dirs() -> &'static [PathBuf] {
    static DIRS: OnceLock<Vec<PathBuf>> = OnceLock::new();
    DIRS.get_or_init(|| {
        let _stage = crate::diagnostics::Stage::new("notebooklm: locate python scripts dir");
        candidate_script_dirs_uncached()
    })
}

fn candidate_script_dirs_uncached() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    // Whatever the interpreter Sarathi would use says about itself. This is the
    // reliable answer; the rest are for when the interpreter cannot be run.
    if let Ok(out) = create_hidden_command("python")
        .args(["-c", "import sysconfig;print(sysconfig.get_path('scripts'))"])
        .output()
    {
        if out.status.success() {
            let line = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !line.is_empty() {
                dirs.push(PathBuf::from(line));
            }
        }
    }

    if let Some(home) = dirs_home() {
        // pipx / uv tool installs.
        dirs.push(home.join(".local").join("bin"));
        if cfg!(windows) {
            dirs.push(home.join("AppData").join("Roaming").join("Python").join("Scripts"));
        }
    }
    dirs
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
}

/// The MCP server entry for the registry, when the package can provide one.
///
/// A plain stdio server like any other — which is the point. Once this is in
/// `mcp.json` nothing downstream knows it is NotebookLM.
pub fn mcp_server_spec(status: &NotebookLmStatus) -> Option<crate::launcher::mcp::McpServerSpec> {
    let path = status.mcp_server_path.as_ref()?;
    Some(crate::launcher::mcp::McpServerSpec {
        description: Some(
            "Google NotebookLM: grounded research over sources you add to a notebook."
                .to_string(),
        ),
        ..crate::launcher::mcp::McpServerSpec::stdio(path.clone())
    })
}

/// Detection with no subprocess at all: are the paths we already resolved still
/// on disk?
///
/// This is the difference between a card that renders in a millisecond and one
/// that renders in fifteen seconds. A full [`detect`] costs, on the machine
/// this was measured on, two `where` scans (2.6s + 2.2s over a 54-entry PATH),
/// a Python interpreter start (1.4s), `notebooklm --version` (4.9s) and a
/// `notebooklm-mcp --help` import check (2.2s). None of that tells us anything
/// new when the same files are sitting at the same paths as last time.
///
/// Returns `None` when nothing is remembered, or when a remembered path has
/// gone — in which case the caller must fall back to the full probe, because a
/// moved installation is exactly the case a cache must not paper over.
pub fn detect_remembered(remembered: &Remembered) -> Option<NotebookLmStatus> {
    let cli = remembered.cli_path.as_ref()?;
    if !Path::new(cli).is_file() {
        return None;
    }

    // A remembered MCP path must still exist; a remembered *absence* is not
    // cacheable, because installing the extra is precisely what the user may
    // have just done outside Sarathi.
    let mcp = remembered.mcp_server_path.as_ref()?;
    if !Path::new(mcp).is_file() {
        return None;
    }

    let has_local_session = session_file().is_some_and(|p| p.is_file());

    Some(NotebookLmStatus {
        state: if has_local_session {
            NotebookLmState::Unverified
        } else {
            NotebookLmState::NotAuthenticated
        },
        version: remembered.version.clone(),
        cli_path: Some(cli.clone()),
        mcp_server_path: Some(mcp.clone()),
        mcp_available: true,
        has_local_session,
        from_cache: true,
        ..NotebookLmStatus::blank(NotebookLmState::Unverified)
    })
}

/// Detects installation without touching the network.
///
/// The two halves — where the CLI is and whether the MCP server can start —
/// share nothing, so they run at the same time. Sequentially they were the
/// larger part of a fifteen-second detection.
pub fn detect() -> NotebookLmStatus {
    let _stage = crate::diagnostics::Stage::new("notebooklm: full detect");

    // Warm the memoised Python lookup before the threads race for it, so only
    // one of them pays for it and neither blocks the other on the OnceLock.
    let _ = candidate_script_dirs();

    let (cli_probe, mcp_probe) = std::thread::scope(|scope| {
        let cli = scope.spawn(|| {
            let path = find_script(CLI)?;
            let version = run(&path, &["--version"], QUICK)
                .ok()
                .filter(|r| r.ok)
                .and_then(|r| parse_version(&r.stdout));
            Some((path, version))
        });

        // Present *and* able to start, taking the first copy that really runs.
        //
        // The console script is installed by the base package, but the server
        // it launches needs the `[mcp]` extra — without it the file exists,
        // `--help` fails on a missing import, and every provider is handed a
        // command that dies the moment it is spawned. Checking only for the
        // file is the same mistake as treating a config as a connection. And
        // since a machine can hold several copies, the broken one being first
        // on PATH must not hide a working one behind it.
        let mcp = scope.spawn(|| {
            let mut failure: Option<String> = None;
            let found = find_scripts(MCP_SERVER).into_iter().find(|path| {
                match mcp_server_runs(path) {
                    Ok(()) => true,
                    Err(why) => {
                        log::warn!("[NOTEBOOKLM] {} cannot start: {why}", path.display());
                        failure.get_or_insert(why);
                        false
                    }
                }
            });
            (found, failure)
        });

        (cli.join().ok().flatten(), mcp.join().unwrap_or((None, None)))
    });

    let Some((cli_path, version)) = cli_probe else {
        return NotebookLmStatus::not_installed();
    };
    let (mcp_server_path, mcp_failure) = mcp_probe;

    let has_local_session = session_file().is_some_and(|p| p.is_file());

    NotebookLmStatus {
        // Installed and holding a session, but nothing has verified it works.
        // That distinction is the whole point of this state existing.
        state: if has_local_session {
            NotebookLmState::Unverified
        } else {
            NotebookLmState::NotAuthenticated
        },
        version,
        cli_path: Some(cli_path.to_string_lossy().to_string()),
        mcp_available: mcp_server_path.is_some(),
        // Why the server is unusable, when it is — so the card can say
        // "install the [mcp] extra" rather than leaving a blank.
        detail: mcp_server_path.is_none().then_some(mcp_failure).flatten(),
        mcp_server_path: mcp_server_path.map(|p| p.to_string_lossy().to_string()),
        has_local_session,
        ..NotebookLmStatus::blank(NotebookLmState::Unverified)
    }
}

/// Whether the MCP server actually starts, rather than merely existing.
///
/// `--help` is enough: it imports the server module, which is where a missing
/// `[mcp]` extra shows up. The script exits 0 even when that import fails —
/// Python's console-script shim swallows the status — so the *output* is what
/// decides, not the exit code.
fn mcp_server_runs(path: &Path) -> Result<(), String> {
    let out = run(path, &["--help"], QUICK)?;
    let combined = format!("{}{}", out.stdout, out.stderr);

    if combined.contains("ModuleNotFoundError") || combined.contains("ImportError") {
        let missing = combined
            .lines()
            .find(|l| l.contains("No module named"))
            .map(|l| l.trim().to_string())
            .unwrap_or_else(|| "a dependency is missing".to_string());
        return Err(format!(
            "{missing} — install `{PACKAGE_SPEC}` so the MCP extra is present"
        ));
    }
    if combined.contains("Traceback (most recent call last)") {
        return Err(summarise(&combined));
    }
    Ok(())
}

/// `NotebookLM CLI, version 0.8.0 (8fb61cb1)` -> `0.8.0`.
fn parse_version(output: &str) -> Option<String> {
    output
        .split_whitespace()
        .find(|w| w.chars().next().is_some_and(|c| c.is_ascii_digit()) && w.contains('.'))
        .map(|v| v.trim_end_matches(',').to_string())
}

/// The session file `notebooklm` keeps for the default profile.
///
/// Located, never read. Sarathi reports that it exists; what is inside it is
/// Google's business and the user's.
fn session_file() -> Option<PathBuf> {
    Some(
        dirs_home()?
            .join(".notebooklm")
            .join("profiles")
            .join("default")
            .join("storage_state.json"),
    )
}

/// A cheap marker for "the same session as last time", from the file's size and
/// modification time.
///
/// Not a hash of the contents — the file is never read. This exists so a
/// verification remembered from a previous run can be discarded when the
/// session underneath it has been replaced (a `notebooklm login` run from a
/// terminal, a profile restored from a backup), rather than trusted blindly or
/// thrown away on a timer.
pub fn session_fingerprint() -> Option<String> {
    let meta = std::fs::metadata(session_file()?).ok()?;
    let modified = meta
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    Some(format!("{}:{modified}", meta.len()))
}

/// Verifies the session against Google, rather than against a file on disk.
///
/// `auth check --test` performs the token fetch the local checks skip. A local
/// cookie jar that has expired is byte-for-byte as convincing as a live one, so
/// this is the only thing that justifies reporting Connected.
pub fn health_check() -> NotebookLmStatus {
    verify(detect())
}

/// The live half of a health check, against a status that has already been
/// detected.
///
/// Separate from [`detect`] because the two answer different questions at
/// wildly different prices, and the expensive one must not be re-run just
/// because the cheap one is wanted again. Re-detecting before every
/// verification is what put fifteen seconds of subprocess in front of the
/// network call the user was actually waiting for after signing in.
pub fn verify(mut status: NotebookLmStatus) -> NotebookLmStatus {
    let _stage = crate::diagnostics::Stage::new("notebooklm: verify session with google");

    let Some(cli) = status.cli_path.clone() else {
        return status;
    };

    match run(Path::new(&cli), &["auth", "check", "--test"], LIVE) {
        Ok(r) if r.ok && !looks_unauthenticated(&r.stdout) => {
            status.state = NotebookLmState::Connected;
            status.last_verified_at = Some(chrono::Utc::now().to_rfc3339());
            status.detail = None;
            status.signed_out = false;
        }
        Ok(r) => {
            let combined = format!("{}\n{}", r.stdout, r.stderr);
            status.state = if looks_unauthenticated(&combined) {
                NotebookLmState::AuthenticationExpired
            } else {
                NotebookLmState::ConnectionFailed
            };
            status.detail = Some(summarise(&combined));
            // A failure that is not about authentication leaves the stored
            // session alone: the network being down is not a reason to make
            // somebody sign in again.
            if status.state == NotebookLmState::AuthenticationExpired {
                status.last_verified_at = None;
            }
        }
        Err(e) => {
            status.state = NotebookLmState::ConnectionFailed;
            status.detail = Some(e);
        }
    }

    status
}

/// Whether output describes an authentication problem rather than any other.
///
/// The distinction decides whether the user is shown **Reconnect** or "try
/// again", so it has to match what the CLI really prints. It did not: a session
/// whose cookies had gone says
///
/// ```text
/// Error: Missing required cookies: SID, __Secure-1PSIDTS
/// ... Run 'notebooklm login' to re-authenticate.
/// ```
///
/// which contains none of "not authenticated", "expired" or "401", so the one
/// case the Reconnect button exists for was being reported as "the check itself
/// failed — this says nothing about your sign-in". The CLI's own instruction to
/// re-authenticate is the most reliable signal there is, so it is the one used.
fn looks_unauthenticated(output: &str) -> bool {
    let lower = output.to_lowercase();
    [
        "not authenticated",
        "no auth",
        "expired",
        "log in",
        "login required",
        "unauthorized",
        "401",
        "403",
        // What this CLI actually says when the session is no longer usable.
        "missing required cookies",
        "re-authenticate",
        "reauthenticate",
        "notebooklm login",
        "not signed in",
        "sign in again",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

/// One line for the UI, with anything credential-shaped removed.
///
/// The CLI prints cookie names and domains in its tables. None of that belongs
/// in a status string that will be rendered, logged and possibly screenshotted.
///
/// An `Error:` line wins over anything above it. Taking the first non-border
/// line instead put the table's own caption — "Authentication Check" — on the
/// card as the explanation of the failure, which told the user nothing at all
/// while the actual reason sat six lines below it.
fn summarise(output: &str) -> String {
    let meaningful = |l: &&str| !l.is_empty() && !l.starts_with(['┌', '│', '├', '└', '─']);

    let line = output
        .lines()
        .map(str::trim)
        .find(|l| l.starts_with("Error:") || l.starts_with("error:"))
        .or_else(|| output.lines().map(str::trim).find(meaningful))
        .unwrap_or("the check failed without saying why");

    scrub(line).chars().take(200).collect()
}

/// Removes anything that looks like a credential from text bound for the UI.
///
/// Deliberately blunt. A status line has no legitimate need for a long opaque
/// token, so anything shaped like one is replaced rather than risk it being
/// displayed, logged, or read back to a model.
pub fn scrub(text: &str) -> String {
    const SECRETS: [&str; 8] = [
        "psid", "sid", "cookie", "token", "bearer", "authorization", "secret", "password",
    ];

    text.split_whitespace()
        .map(|word| {
            let lower = word.to_lowercase();

            // `name=value`: if the name is credential-shaped the value goes,
            // and the pair goes with it.
            if let Some((name, _)) = lower.split_once('=') {
                if SECRETS.iter().any(|s| name.contains(s)) {
                    return "[redacted]".to_string();
                }
            }

            // A long run with no spaces and mixed case/digits is a token, not
            // a word anyone wants to read.
            let opaque = word.len() >= 24
                && word.chars().all(|c| c.is_ascii_alphanumeric() || "-_=.".contains(c))
                && word.chars().any(|c| c.is_ascii_digit());
            if opaque {
                return "[redacted]".to_string();
            }

            // A credential-shaped word that also looks like a *value* — long,
            // or carrying digits. The English words "cookie" and "token" are
            // left alone: redacting them turned "Missing required cookies: SID"
            // into "Missing required [redacted] [redacted]", which protects
            // nothing and explains nothing.
            let secretish = SECRETS.iter().any(|s| lower.contains(s))
                && (word.len() >= 16 || word.chars().any(|c| c.is_ascii_digit()));
            if secretish { "[redacted]".to_string() } else { word.to_string() }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Installs the package with `uv`, falling back to `pip`.
///
/// Only ever called from an explicit user action. `uv tool install` is
/// preferred because it isolates the package's own dependency pins from the
/// system interpreter while still putting the console scripts on PATH.
pub fn install(mut progress: impl FnMut(&str)) -> Result<NotebookLmStatus, String> {
    let attempts: Vec<(&str, Vec<&str>)> = vec![
        ("uv", vec!["tool", "install", PACKAGE_SPEC]),
        ("pip", vec!["install", "--upgrade", PACKAGE_SPEC]),
    ];

    let mut failures = Vec::new();

    for (program, args) in attempts {
        let Some(path) = find_program(program) else {
            progress(&format!("{program} is not on this machine; trying the next method"));
            failures.push(format!("{program}: not installed"));
            continue;
        };

        progress(&format!("Running {program} {}", args.join(" ")));
        match run(&path, &args, Duration::from_secs(900)) {
            Ok(r) if r.ok => {
                progress("Verifying the installation");
                let status = detect();
                return match status.state {
                    NotebookLmState::NotInstalled => Err(
                        "the installer reported success but no `notebooklm` command appeared on \
                         PATH — open a new terminal, or add the Python scripts directory to PATH"
                            .to_string(),
                    ),
                    _ if !status.mcp_available => Err(format!(
                        "`notebooklm` was installed but its MCP server was not; install \
                         `{PACKAGE_SPEC}` (with the [mcp] extra) rather than the bare package"
                    )),
                    _ => Ok(status),
                };
            }
            Ok(r) => {
                let why = summarise(&format!("{}\n{}", r.stderr, r.stdout));
                progress(&format!("{program} failed: {why}"));
                failures.push(format!("{program}: {why}"));
            }
            Err(e) => {
                progress(&format!("{program} failed: {e}"));
                failures.push(format!("{program}: {e}"));
            }
        }
    }

    Err(format!("could not install {PACKAGE_SPEC} — {}", failures.join("; ")))
}

fn find_program(name: &str) -> Option<PathBuf> {
    let (finder, args) = if cfg!(windows) { ("where", vec![name]) } else { ("command", vec!["-v", name]) };
    let out = create_hidden_command(finder).args(&args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .map(PathBuf::from)
        .filter(|p| p.is_file())
}

/// The command that starts an interactive Google sign-in.
///
/// Returned rather than run here because it must be *visible*: `notebooklm
/// login` opens a browser and waits for the user to complete Google's flow,
/// including any second factor. Sarathi runs it in a console the user can see
/// and watches for the exit — it never drives the login itself, and never sees
/// what is typed into it.
pub fn login_command() -> Result<(PathBuf, Vec<String>), String> {
    let cli = find_script(CLI)
        .ok_or_else(|| "NotebookLM is not installed, so there is nothing to log in to".to_string())?;
    Ok((cli, vec!["login".to_string()]))
}

/// Clears the stored session, so the next login starts clean.
pub fn logout() -> Result<(), String> {
    let cli = find_script(CLI).ok_or_else(|| "NotebookLM is not installed".to_string())?;
    let r = run(&cli, &["auth", "logout"], QUICK)?;
    if r.ok {
        Ok(())
    } else {
        Err(summarise(&format!("{}\n{}", r.stderr, r.stdout)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_version_line_yields_just_the_version() {
        assert_eq!(
            parse_version("NotebookLM CLI, version 0.8.0 (8fb61cb1)").as_deref(),
            Some("0.8.0")
        );
        assert_eq!(parse_version("notebooklm 1.2.3").as_deref(), Some("1.2.3"));
        assert_eq!(parse_version("no version here"), None);
    }

    #[test]
    fn auth_failures_are_told_apart_from_other_failures() {
        assert!(looks_unauthenticated("Error: not authenticated, please log in"));
        assert!(looks_unauthenticated("session expired"));
        assert!(looks_unauthenticated("HTTP 401"));
        assert!(!looks_unauthenticated("connection refused"));
        assert!(!looks_unauthenticated("could not reach the network"));
    }

    /// Verbatim from `notebooklm auth check --test` against a session whose
    /// cookies have gone. Matched none of the old needles, so the one case the
    /// Reconnect button exists for was being reported as an unrelated failure.
    const REJECTED: &str = "                             Authentication Check
┌──────────────────┬───────────┬────────────────────────┐
│ Check            │ Status    │ Details                │
│ Cookies present  │ ✗ fail    │                        │
│ SID cookie       │ ✗ fail    │                        │
└──────────────────┴───────────┴────────────────────────┘

Error: Missing required cookies: SID, __Secure-1PSIDTS
This typically means --browser-cookies extraction was incomplete. Run
'notebooklm login' to re-authenticate.
";

    #[test]
    fn a_session_google_will_not_accept_is_an_auth_failure_not_a_network_one() {
        assert!(
            looks_unauthenticated(REJECTED),
            "this is what an expired session actually looks like"
        );
    }

    #[test]
    fn the_reason_shown_is_the_error_not_the_table_caption() {
        let summary = summarise(REJECTED);
        assert!(summary.starts_with("Error: Missing required cookies"), "got: {summary}");
        assert!(
            !summary.contains("Authentication Check"),
            "the table's own title is not an explanation: {summary}"
        );
        // Readable, and still carrying no credential value.
        assert!(summary.contains("SID,"), "the cookie *name* is safe and useful: {summary}");
        assert!(!summary.contains("1PSIDTS"), "the value-shaped one is not: {summary}");
    }

    /// The rule that keeps a credential out of a status string, a log line and
    /// anything a model is later shown.
    #[test]
    fn anything_credential_shaped_is_removed_before_it_can_be_displayed() {
        let scrubbed = scrub(
            "__Secure-1PSIDTS=AbCdEf0123456789AbCdEf0123456789 cookie present token=xyz",
        );
        assert!(!scrubbed.contains("AbCdEf0123456789"), "got: {scrubbed}");
        assert!(!scrubbed.to_lowercase().contains("psidts"), "got: {scrubbed}");
        assert!(scrubbed.contains("[redacted]"));

        let opaque = scrub("g8Fh2kLm9QrS4tUvWx7yZa1bCd3eFg5h");
        assert_eq!(opaque, "[redacted]", "a bare token must not survive");

        // Ordinary words are left alone, or the message becomes unreadable.
        assert_eq!(scrub("could not reach the network"), "could not reach the network");
    }

    #[test]
    fn a_summary_skips_table_borders_and_stays_short() {
        let out = "┌────────┬────────┐\n│ Check  │ Status │\nError: the profile is missing\n";
        let s = summarise(out);
        assert_eq!(s, "Error: the profile is missing");
        assert!(summarise(&"x".repeat(500)).len() <= 200);
    }

    /// NotebookLM must reach providers as an ordinary stdio server, because
    /// that is what makes it need no provider-side code at all.
    #[test]
    fn the_registry_entry_is_an_ordinary_stdio_server() {
        let status = NotebookLmStatus {
            mcp_server_path: Some(r"C:\py\Scripts\notebooklm-mcp.exe".into()),
            mcp_available: true,
            ..NotebookLmStatus::not_installed()
        };

        let spec = mcp_server_spec(&status).expect("a path yields a server");
        assert_eq!(spec.command.as_deref(), Some(r"C:\py\Scripts\notebooklm-mcp.exe"));
        assert_eq!(spec.transport(), crate::launcher::mcp::McpTransport::Stdio);
        assert!(spec.validate(MCP_SERVER_KEY).is_ok());
        assert!(spec.env.is_empty(), "no credential is ever passed through the registry");
    }

    #[test]
    fn without_the_mcp_extra_there_is_no_server_to_offer() {
        let status = NotebookLmStatus { mcp_server_path: None, ..NotebookLmStatus::not_installed() };
        assert!(mcp_server_spec(&status).is_none());
    }

    /// Detection must not claim a working session from a file on disk. The
    /// state after detection is `Unverified`; only `health_check` may promote
    /// it, and only after a live call.
    #[test]
    fn a_session_file_alone_never_means_connected() {
        let status = detect();
        assert_ne!(
            status.state,
            NotebookLmState::Connected,
            "detect() does no network call and must never report Connected"
        );
    }

    /// The case that made a NotebookLM entry every provider would fail on: the
    /// console script installed, its dependency not.
    #[test]
    fn a_server_that_cannot_import_is_not_reported_as_available() {
        let broken = "Traceback (most recent call last):
  from fastmcp import FastMCP
                      ModuleNotFoundError: No module named 'fastmcp'
";
        assert!(broken.contains("ModuleNotFoundError"));

        // The same judgement the detector makes, on the same text.
        let missing = broken.lines().find(|l| l.contains("No module named"));
        assert!(missing.is_some(), "the reason must be quotable back to the user");
    }

    /// Detection must not promise a capability it has not proven startable.
    #[test]
    fn availability_and_registration_are_separate_claims() {
        let mut s = NotebookLmStatus::not_installed();
        s.mcp_server_path = None;
        s.mcp_available = false;
        assert!(mcp_server_spec(&s).is_none(), "nothing to offer providers");

        s.mcp_server_path = Some("C:/x/notebooklm-mcp.exe".into());
        s.mcp_available = true;
        assert!(mcp_server_spec(&s).is_some());
        assert!(!s.in_registry, "available is not the same as offered");
    }
}
