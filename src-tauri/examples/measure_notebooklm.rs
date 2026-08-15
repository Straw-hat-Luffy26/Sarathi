//! Measures every stage of the NotebookLM and provider detection path.
//!
//! "Checking what's installed…" lasting minutes, and a Google sign-in that
//! looked like it had failed for half a minute after it succeeded, were both
//! subprocess cost nobody had counted. This counts it, per stage, against the
//! real machine — so the fix is a number rather than an opinion.
//!
//! Run with:  cargo run --release --example measure_notebooklm

use std::path::{Path, PathBuf};
use std::time::Instant;

use sarathi_lib::launcher;
use sarathi_lib::notebooklm::{self, manager, state, Remembered};

fn app_data_dir() -> PathBuf {
    PathBuf::from(std::env::var("APPDATA").expect("APPDATA")).join("com.sarathi.app")
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

fn timed<T>(label: &str, f: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let out = f();
    println!("{label:<58} {:>10.0} ms", ms(started));
    out
}

/// The pieces a full NotebookLM detection is made of, priced individually.
fn notebooklm_stages() {
    println!("\n--- notebooklm: what a full detection pays for ---");

    let cli = timed("find the notebooklm CLI (PATH scan + python sysconfig)", || {
        notebooklm::find_script("notebooklm")
    });
    println!("   -> {:?}", cli.as_ref().map(|p| p.display().to_string()));

    let servers = timed("find every notebooklm-mcp on this machine", || {
        notebooklm::find_scripts("notebooklm-mcp")
    });
    println!("   -> {} copies", servers.len());
    for s in &servers {
        println!("      {}", s.display());
    }

    timed("second PATH scan (now that the python lookup is memoised)", || {
        notebooklm::find_script("notebooklm")
    });
}

fn detection_paths(app_data: &Path) {
    println!("\n--- notebooklm: detection, cold and warm ---");

    let full = timed("detect(): full probe, concurrent", notebooklm::detect);
    println!(
        "   -> {:?}, version {}, mcp {}",
        full.state,
        full.version.as_deref().unwrap_or("unknown"),
        if full.mcp_available { "available" } else { "unavailable" }
    );

    let remembered = Remembered::of(&full);
    let cached = timed("detect_remembered(): paths confirmed on disk", || {
        notebooklm::detect_remembered(&remembered)
    });
    println!("   -> {:?}", cached.as_ref().map(|s| s.state));

    timed("session_fingerprint(): is this the session we verified?", || {
        notebooklm::session_fingerprint()
    });

    timed("provider_fit(): who could receive this", || {
        manager::provider_fit(app_data)
    });

    println!("\n--- notebooklm: the live half ---");
    let verified = timed("verify(): the one call that can say Connected", || {
        notebooklm::verify(full.clone())
    });
    println!("   -> {:?}", verified.state);

    println!("\n--- persisted state ---");
    let persisted = state::load(app_data);
    println!(
        "   remembered cli: {:?}",
        persisted.remembered.cli_path.as_deref().unwrap_or("(none)")
    );
    println!(
        "   last verified:  {}",
        persisted.last_verified_at.as_deref().unwrap_or("(never)")
    );
    println!("   signed out:     {}", persisted.signed_out);
    println!(
        "   verification still applies: {}",
        persisted.verification_still_applies(notebooklm::session_fingerprint().as_deref())
    );
}

/// The other half of the Launch screen's cost: one `where` and one `--version`
/// per provider, which the screen used to re-run every two seconds.
fn provider_detection(app_data: &Path) {
    println!("\n--- providers: the Launch grid ---");
    let specs = launcher::registry::load(app_data).tools;

    let started = Instant::now();
    for spec in &specs {
        let each = Instant::now();
        let state = launcher::detect(spec);
        println!("   {:<20} {:>8.0} ms  {:?}", spec.name, ms(each), state);
    }
    println!("{:<58} {:>10.0} ms", "sequential, every tool (the old overview call)", ms(started));

    let cache = launcher::DetectionCache::default();
    timed("concurrent, every tool (one cache refresh)", || cache.refresh(&specs));
    timed("from cache (what a poll now costs)", || cache.states(&specs));

    let mut total = 0.0;
    for _ in 0..100 {
        let t = Instant::now();
        let _ = cache.states(&specs);
        total += ms(t);
    }
    println!("{:<58} {:>10.3} ms", "average of 100 cached reads", total / 100.0);
}

fn main() {
    let app_data = app_data_dir();
    println!("app data: {}", app_data.display());

    notebooklm_stages();
    detection_paths(&app_data);
    provider_detection(&app_data);
}
