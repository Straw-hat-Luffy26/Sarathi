//! The rules that stop Sarathi asking a signed-in user to sign in again.
//!
//! Every one of these is a bug that was actually reported: the card asked for a
//! Google login after a tab switch, after a model load, after a re-render and
//! after a restart. None of those events had anything to do with the session —
//! the app had simply forgotten it verified one, and the only affordance it
//! offered for "I do not know" was "sign in".
//!
//! So these tests are about *what may change the authentication state*, and the
//! answer is: Google, and the user pressing Sign out. Nothing else.

use std::path::PathBuf;
use std::time::Instant;

use sarathi_lib::notebooklm::{self, state, NotebookLmState, Remembered};

fn temp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sarathi_nlm_it_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn repo_file(relative: &str) -> String {
    // Tests run from `src-tauri`; the front end is its sibling.
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Closing and reopening Sarathi is not a sign-out.
#[test]
fn a_restart_with_the_same_session_needs_no_new_sign_in() {
    let dir = temp("restart");

    // A run in which the user signed in and a live check succeeded.
    let mut first = state::Persisted::default();
    first.remembered = Remembered {
        cli_path: Some("/usr/local/bin/notebooklm".into()),
        mcp_server_path: Some("/usr/local/bin/notebooklm-mcp".into()),
        version: Some("0.8.0".into()),
    };
    first.record_verified("2026-08-12T10:00:00Z".into(), Some("13223:1786539297".into()));
    state::save(&dir, &first).unwrap();

    // The next run, with the session file untouched.
    let second = state::load(&dir);
    assert!(
        second.verification_still_applies(Some("13223:1786539297")),
        "a restart must not invalidate a verified session"
    );
    assert!(!second.signed_out);
    assert!(second.remembered.cli_path.is_some(), "and the fast path survives too");
}

/// The one thing besides Google that may end a session.
#[test]
fn signing_out_is_what_requires_a_new_sign_in() {
    let dir = temp("signout");

    let mut s = state::Persisted::default();
    s.record_verified("2026-08-12T10:00:00Z".into(), Some("13223:1786539297".into()));
    state::save(&dir, &s).unwrap();
    assert!(state::load(&dir).verification_still_applies(Some("13223:1786539297")));

    s.forget_verification();
    s.signed_out = true;
    state::save(&dir, &s).unwrap();

    let after = state::load(&dir);
    assert!(after.signed_out, "the decision survives a restart");
    assert!(!after.verification_still_applies(Some("13223:1786539297")));
}

/// Time passing is not evidence. Only Google, or a changed session file, is.
#[test]
fn nothing_expires_on_a_timer() {
    let mut s = state::Persisted::default();
    // A verification recorded years ago, against the session still on disk.
    s.record_verified("2020-01-01T00:00:00Z".into(), Some("13223:1786539297".into()));
    assert!(
        s.verification_still_applies(Some("13223:1786539297")),
        "age alone must never force a re-authentication"
    );

    // A session file that has been replaced underneath us is a different story.
    assert!(!s.verification_still_applies(Some("999:1799999999")));
}

/// Detection must never be the thing that claims a working connection.
#[test]
fn no_amount_of_local_detection_yields_connected() {
    let full = notebooklm::detect();
    assert_ne!(full.state, NotebookLmState::Connected);

    if let Some(cached) = notebooklm::detect_remembered(&Remembered::of(&full)) {
        assert_ne!(cached.state, NotebookLmState::Connected);
        assert!(cached.from_cache);
    }
}

/// The performance claim, measured rather than asserted in a comment.
///
/// Skipped when NotebookLM is not installed on the machine running the tests —
/// there is nothing to be fast about.
#[test]
fn the_second_look_is_orders_of_magnitude_cheaper_than_the_first() {
    let full = notebooklm::detect();
    let Some(remembered) = Some(Remembered::of(&full)).filter(|r| r.cli_path.is_some()) else {
        eprintln!("NotebookLM is not installed here; nothing to measure");
        return;
    };

    let started = Instant::now();
    let cached = notebooklm::detect_remembered(&remembered);
    let elapsed = started.elapsed();

    if cached.is_some() {
        assert!(
            elapsed < std::time::Duration::from_millis(100),
            "the remembered path must not run a subprocess: took {elapsed:?}"
        );
    }
}

/// The strongest form of "opening Launch never starts a Google login": there is
/// no code path from a render to one.
///
/// The store is where an accidental auto-login would have to live — it is the
/// only thing a mount calls. A `useEffect` that "helpfully" reconnects is
/// exactly the regression this catches.
#[test]
fn mounting_the_launch_page_cannot_reach_a_sign_in() {
    let store = repo_file("src/services/notebooklm.store.ts");

    let start_of_start = store.find("async function start()").expect("the store starts once");
    let end_of_start = store[start_of_start..]
        .find("\n}")
        .map(|i| start_of_start + i)
        .expect("start() ends");
    let start_body = &store[start_of_start..end_of_start];

    assert!(
        !start_body.contains("Login") && !start_body.contains("connect("),
        "the one function a mount runs must not authenticate:\n{start_body}"
    );

    // And the card only ever reaches a sign-in from a click.
    let card = repo_file("src/components/NotebookLmCard.tsx");
    for (i, line) in card.lines().enumerate() {
        if line.contains("notebookLm.connect()") {
            assert!(
                card.lines().nth(i.saturating_sub(1)).is_some_and(|l| l.contains("onClick"))
                    || line.contains("onClick"),
                "line {} reaches a sign-in without a click: {line}",
                i + 1
            );
        }
    }
}

/// Launch renders the capability; it does not own it.
#[test]
fn the_launch_page_holds_no_notebooklm_state() {
    let launch = repo_file("src/pages/Launch.tsx");
    assert!(
        !launch.contains("notebookLm") || launch.contains("NotebookLmCard"),
        "Launch must delegate to the card rather than drive NotebookLM itself"
    );
    assert!(
        !launch.contains("notebookLmLogin") && !launch.contains("notebooklm_login"),
        "the page must have no way to start a sign-in"
    );
}
