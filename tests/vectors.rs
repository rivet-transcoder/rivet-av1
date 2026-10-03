//! The AOM AV1 test vectors (av1-1-b8-*, av1-1-b10-*): every shown frame's
//! MD5 must match.
//!
//! Run `tools/fetch-vectors.sh` first (about 7 MB into tests/vectors/), or
//! point `AV1_VECTOR_DIR` at a directory holding them. Without them this
//! test reports that it skipped, unless `AV1_REQUIRE_VECTORS=1` makes the
//! absence a failure. `AV1_VECTOR=substr` restricts the run.

mod common;

#[test]
fn test_vectors() {
    let mut paths = common::vectors_in(&common::vector_dir());
    if paths.is_empty() {
        assert!(
            std::env::var("AV1_REQUIRE_VECTORS").is_err(),
            "no test vectors: run tools/fetch-vectors.sh"
        );
        eprintln!("test vectors not downloaded (tools/fetch-vectors.sh); skipped");
        return;
    }
    if let Ok(f) = std::env::var("AV1_VECTOR") {
        paths.retain(|p| p.to_string_lossy().contains(&f));
    }
    let results = common::run_all(&paths);
    let passed = results.iter().filter(|o| o.passed()).count();
    for o in &results {
        if !o.passed() {
            eprintln!(
                "FAIL {}: {}/{} frames ok; {}",
                o.name,
                o.matched,
                o.expected,
                o.failure.as_deref().unwrap_or("")
            );
        }
    }
    eprintln!("{passed}/{} vectors pass", results.len());
    let unexpected: Vec<_> = results
        .iter()
        .filter(|o| !o.passed() && !KNOWN_FAILURES.iter().any(|k| o.name.contains(k)))
        .collect();
    assert!(unexpected.is_empty(), "{} vectors failed", unexpected.len());
}

/// Vectors this decoder does not pass yet (see the README).
const KNOWN_FAILURES: &[&str] = &[];
