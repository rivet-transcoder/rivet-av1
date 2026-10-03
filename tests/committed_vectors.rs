//! The test vectors committed in tests/data (see its README): every shown
//! frame must match its published MD5. Runs without any download.

mod common;

#[test]
fn committed_vectors() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let paths = common::vectors_in(&dir);
    assert!(paths.len() >= 12, "tests/data is incomplete");
    let results = common::run_all(&paths);
    let failed: Vec<_> = results.iter().filter(|o| !o.passed()).collect();
    for o in &failed {
        eprintln!(
            "FAIL {}: {}/{} frames; {}",
            o.name,
            o.matched,
            o.expected,
            o.failure.as_deref().unwrap_or("")
        );
    }
    assert!(failed.is_empty());
}
