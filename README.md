# `rustc_apfloat` (Rust port of C++ `llvm::APFloat` library)

## 🚧 Work In Progress 🚧

**NOTE**: the repo (and [`rustc_apfloat-git-history-extraction`](https://github.com/LykenSol/rustc_apfloat-git-history-extraction)) might be public already, but only for convenience of discussion, see [relevant Zulip topic](https://rust-lang.zulipchat.com/#narrow/stream/231349-t-core.2Flicensing/topic/apfloat) for more details.

## Testing

`rustc_apfloat` contains ports of all the tests from the C++ `llvm::APFloat` code,
in `tests/ieee.rs` and `tests/ppc.rs`.

For tests specific to `rustc_apfloat` (without C++ equivalents), `tests/downstream.rs`
is used (which mainly contains tests for now-fixed bugs, found by fuzzing).

### Fuzzing

As `llvm::APFloat` tests are far from comprehensive, the only option for in-depth
comparisons between the original C++ code and the Rust port (and between them and
hardware floating-point behavior) is to employ *fuzzing*.

The fuzzing infrastructure lives in `fuzz/` and requires `cargo-afl`, but also
involves an automated build of the original C++ `llvm::APFloat` code with `clang`
(to be able to instrument it via LLVM, in the same way `cargo-afl` does for the
Rust code), and has been prototyped and tested on Linux (and is unlikely to work
on other platforms, or even some Linux distros, though it mostly assumes UNIX).

Example usage:  
<sub>(**TODO**: maybe move this to `fuzz/README.md` and/or expand on it)</sub>

```sh
# Install `cargo-afl` (used below to build/run the fuzzing binary).
cargo install afl

# Build the fuzzing binary (`target/release/rustc_apfloat-fuzz`).
cargo afl build -p rustc_apfloat-fuzz --release

# Seed the inputs for a run `foo` (while not ideal, even this one minimal input works).
mkdir fuzz/in-foo && echo > fuzz/in-foo/empty

# Start the fuzzing run `foo`, which should bring up the AFL++ progress TUI
# (see also `cargo run -p rustc_apfloat-fuzz` for extra flags available).
cargo afl fuzz -i fuzz/in-foo -o fuzz/out-foo target/release/rustc_apfloat-fuzz
```

To visualize the fuzzing testcases, you can use the `decode` subcommand:
```sh
cargo run -p rustc_apfloat-fuzz decode fuzz/out-foo/default/crashes/*
```
(this will work even while `cargo afl fuzz`, i.e. AFL++, is running)
