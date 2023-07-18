// HACK(eddyb) easier dep-tracking if we let `rustc` do it.
const SRC_LIB_RS_CONTENTS: &str = include_str!("src/lib.rs");

const EXPECTED_SRC_LIB_RS_PREFIX: &str = "\
//! Port of LLVM's APFloat software floating-point implementation from the
//! following C++ sources (please update commit hash when backporting):
//! https://github.com/llvm/llvm-project/commit/";

fn main() {
    // HACK(eddyb) disable the default of re-running the build script on *any*
    // change to *the entire source tree* (i.e. the default is roughly `./`).
    println!("cargo:rerun-if-changed=build.rs");

    let llvm_commit_hash = SRC_LIB_RS_CONTENTS
        .strip_prefix(EXPECTED_SRC_LIB_RS_PREFIX)
        .ok_or(())
        .map_err(|_| format!("expected `src/lib.rs` to start with:\n\n{EXPECTED_SRC_LIB_RS_PREFIX}"))
        .and_then(|commit_hash_plus_rest_of_file| {
            Ok(commit_hash_plus_rest_of_file
                .split_once('\n')
                .ok_or("expected `src/lib.rs` to have more than 3 lines")?)
        })
        .and_then(|(commit_hash, _)| {
            if commit_hash.len() != 40 || !commit_hash.chars().all(|c| matches!(c, '0'..='9'|'a'..='f')) {
                Err(format!("expected `src/lib.rs` to have a valid commit hash, found {commit_hash:?}"))
            } else {
                Ok(commit_hash)
            }
        })
        .unwrap_or_else(|e| {
            eprintln!("\n{e}\n");
            panic!("failed to validate `src/lib.rs`'s commit hash (see above)")
        });

    let expected_version_metadata = format!("+llvm-{}", &llvm_commit_hash[..12]);
    let actual_version = env!("CARGO_PKG_VERSION");
    if !actual_version.ends_with(&expected_version_metadata) {
        eprintln!("\nexpected version ending in `{expected_version_metadata}`, found `{actual_version}`\n");
        panic!("failed to validate Cargo package version (see above)");
    }
}
