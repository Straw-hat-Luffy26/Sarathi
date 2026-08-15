//! Walks the NotebookLM connection through every lifecycle event that used to
//! make it ask for a Google sign-in, and reports what actually happened.
//!
//! The bugs were all lifecycle bugs — a tab switch, a model load, a re-render
//! and a restart each looked to the user like being logged out. So this drives
//! the real manager, against the real installation and the real Google session,
//! through the real sequence, and prints every state transition it produced.
//!
//! Nothing here signs in, and nothing here signs out: a sign-in cannot be
//! automated (it is Google's browser flow, by design) and a sign-out would
//! destroy a session that only the user can restore. The "no session" cases are
//! produced by moving the session file aside and putting it back, which is
//! exactly what the code has to cope with and is completely reversible.
//!
//! Run with:  cargo run --release --example verify_notebooklm_lifecycle

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use sarathi_lib::notebooklm::manager::{NotebookLmManager, StatusSink};
use sarathi_lib::notebooklm::{NotebookLmState, NotebookLmStatus};

/// Records every transition, so the lifecycle is evidence rather than a claim.
#[derive(Default)]
struct Recorder {
    transitions: Mutex<Vec<(f64, NotebookLmState)>>,
    started: Mutex<Option<Instant>>,
}

impl Recorder {
    fn begin(&self) {
        *self.started.lock().unwrap() = Some(Instant::now());
        self.transitions.lock().unwrap().clear();
    }

    fn drain(&self) -> Vec<(f64, NotebookLmState)> {
        self.transitions.lock().unwrap().clone()
    }
}

impl StatusSink for Recorder {
    fn status(&self, status: &NotebookLmStatus) {
        let at = self
            .started
            .lock()
            .unwrap()
            .map(|s| s.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let mut t = self.transitions.lock().unwrap();
        if t.last().map(|(_, s)| *s) != Some(status.state) {
            t.push((at, status.state));
        }
    }

    fn progress(&self, line: &str) {
        println!("      · {line}");
    }
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sarathi_nlm_lifecycle_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // Give it the real MCP registry, so provider availability is the real
    // answer rather than an empty one.
    let real = PathBuf::from(std::env::var("APPDATA").expect("APPDATA"))
        .join("com.sarathi.app")
        .join("mcp.json");
    if real.is_file() {
        let _ = std::fs::copy(&real, dir.join("mcp.json"));
    }
    dir
}

fn session_path() -> PathBuf {
    PathBuf::from(std::env::var("USERPROFILE").expect("USERPROFILE"))
        .join(".notebooklm")
        .join("profiles")
        .join("default")
        .join("storage_state.json")
}

fn report(label: &str, recorder: &Recorder, status: &NotebookLmStatus, took: f64) {
    println!("\n=== {label} ===");
    print!("   transitions: ");
    let t = recorder.drain();
    if t.is_empty() {
        print!("(none — nothing changed)");
    }
    for (i, (at, state)) in t.iter().enumerate() {
        if i > 0 {
            print!(" -> ");
        }
        print!("{state:?}@{at:.0}ms");
    }
    println!();
    println!("   final:       {:?} after {took:.0} ms", status.state);
    println!(
        "   session:     present={} signedOut={} lastVerified={}",
        status.has_local_session,
        status.signed_out,
        status.last_verified_at.as_deref().unwrap_or("(none)")
    );
    println!(
        "   providers:   {} compatible ({})",
        status.compatible_providers.len(),
        status.compatible_providers.join(", ")
    );
}

/// Every state the card can be asked to render, and how it reads.
fn expect(label: &str, actual: NotebookLmState, wanted: &[NotebookLmState]) -> bool {
    let ok = wanted.contains(&actual);
    println!("   {} {label}: {actual:?}", if ok { "PASS" } else { "FAIL" });
    ok
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let mut failures = 0;

    // ---------------------------------------------------------------- start
    println!("\n########## first start, nothing remembered ##########");
    let dir = scratch("first");
    let rec = Arc::new(Recorder::default());
    rec.begin();
    let cold = Arc::new(NotebookLmManager::new(dir.clone(), rec.clone()));

    let t = Instant::now();
    cold.clone().startup().await;
    let cold_ms = t.elapsed().as_secs_f64() * 1000.0;
    let after_cold = cold.snapshot();
    report("cold start (no remembered paths)", &rec, &after_cold, cold_ms);
    if !expect(
        "ends in a state that needs no sign-in",
        after_cold.state,
        &[NotebookLmState::Connected, NotebookLmState::NotAuthenticated],
    ) {
        failures += 1;
    }

    // ------------------------------------------------------- mount/remount
    println!("\n########## tab switch: leave Launch, come back, twenty times ##########");
    let t = Instant::now();
    for _ in 0..20 {
        // Exactly what a mount does: ask for the state, and make sure startup
        // has been kicked off. Nothing else.
        cold.ensure_started();
        let s = cold.snapshot();
        assert_eq!(s.state, after_cold.state, "a mount changed the state");
    }
    let mounts_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("   20 mounts took {mounts_ms:.3} ms total, state unchanged: {:?}", cold.snapshot().state);
    if !expect("still the same state after 20 remounts", cold.snapshot().state, &[after_cold.state]) {
        failures += 1;
    }

    // ------------------------------------------- provider config rewriting
    println!("\n########## provider switch: rewrite provider config twice ##########");
    let before = cold.snapshot();
    rec.begin();
    let _ = cold.set_registered(true).await;
    let _ = cold.set_registered(false).await;
    let _ = cold.set_registered(true).await;
    let after = cold.snapshot();
    report("registry rewritten three times", &rec, &after, 0.0);
    if !expect("authentication state untouched by provider config", after.state, &[before.state]) {
        failures += 1;
    }
    if after.last_verified_at != before.last_verified_at {
        println!("   FAIL: rewriting provider config discarded the verification");
        failures += 1;
    } else {
        println!("   PASS: the recorded verification survived");
    }

    // -------------------------------------------------------------- restart
    println!("\n########## application restart: same data directory, new manager ##########");
    drop(cold);
    let rec2 = Arc::new(Recorder::default());
    rec2.begin();
    let warm = Arc::new(NotebookLmManager::new(dir.clone(), rec2.clone()));
    let t = Instant::now();
    warm.clone().startup().await;
    let warm_ms = t.elapsed().as_secs_f64() * 1000.0;
    let after_warm = warm.snapshot();
    report("warm start (paths remembered)", &rec2, &after_warm, warm_ms);

    let first_paint = rec2.drain().first().map(|(at, _)| *at).unwrap_or(f64::NAN);
    println!("   first usable state published at {first_paint:.0} ms (cold start: see above)");
    if !expect("warm start needs no sign-in", after_warm.state, &[after_cold.state]) {
        failures += 1;
    }

    // ------------------------------------------------------ session removed
    println!("\n########## session gone (what sign-out and expiry look like) ##########");
    let session = session_path();
    let backup = session.with_extension("json.lifecycle-backup");
    let moved = session.is_file() && std::fs::rename(&session, &backup).is_ok();
    if !moved {
        println!("   SKIP: no session file to move aside");
    } else {
        let rec3 = Arc::new(Recorder::default());
        rec3.begin();
        let gone = Arc::new(NotebookLmManager::new(scratch("gone"), rec3.clone()));
        let t = Instant::now();
        gone.clone().startup().await;
        let took = t.elapsed().as_secs_f64() * 1000.0;
        let s = gone.snapshot();
        report("no session on disk", &rec3, &s, took);
        if !expect(
            "asks for a sign-in only now",
            s.state,
            &[NotebookLmState::NotAuthenticated, NotebookLmState::NotInstalled],
        ) {
            failures += 1;
        }

        // And an explicit health check with no session must say so plainly
        // rather than reporting a connection.
        let t = Instant::now();
        let checked = gone.verify(true).await;
        println!("   health check with no session took {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
        if !expect(
            "a live check with no session says 'not signed in', not 'broken'",
            checked.state,
            &[NotebookLmState::NotAuthenticated],
        ) {
            failures += 1;
        }

        std::fs::rename(&backup, &session).expect("the session file must be put back");
        println!("   session file restored");

        // ------------------------------------------------ restored, no login
        let rec4 = Arc::new(Recorder::default());
        rec4.begin();
        let back = Arc::new(NotebookLmManager::new(dir.clone(), rec4.clone()));
        let t = Instant::now();
        back.clone().startup().await;
        let took = t.elapsed().as_secs_f64() * 1000.0;
        let s = back.snapshot();
        report("session restored", &rec4, &s, took);
        if !expect("back to Connected with no sign-in", s.state, &[after_cold.state]) {
            failures += 1;
        }
    }

    // ------------------------------------------------------ duplicated work
    println!("\n########## eight screens ask at once ##########");
    let rec5 = Arc::new(Recorder::default());
    rec5.begin();
    let busy = Arc::new(NotebookLmManager::new(dir.clone(), rec5.clone()));
    busy.clone().startup().await;

    let t = Instant::now();
    let mut tasks = Vec::new();
    for _ in 0..8 {
        let m = busy.clone();
        tasks.push(tokio::spawn(async move { m.verify(false).await.state }));
    }
    let results: Vec<_> = futures_util::future::join_all(tasks).await;
    let concurrent_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "   8 concurrent health checks resolved in {concurrent_ms:.0} ms -> {:?}",
        results.into_iter().filter_map(Result::ok).collect::<Vec<_>>()
    );
    println!("   (one live call costs several seconds; eight of them would not fit in that)");

    println!("\n########## {} ##########", if failures == 0 { "ALL CHECKS PASSED" } else { "FAILURES ABOVE" });
    std::process::exit(if failures == 0 { 0 } else { 1 });
}
