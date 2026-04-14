# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.4+llvm-038f7debfda0](https://github.com/rust-lang/rustc_apfloat/compare/rustc_apfloat-v0.2.3+llvm-038f7debfda0...rustc_apfloat-v0.2.4+llvm-038f7debfda0) - 2026-04-14

### Other

- Fix `-xux` to `-eux` in the download script
- Enable checking status on host floats
- Add `Float::max_int_bits`
- Replace mix/max combo with `Ord::clamp`
- Bump LLVM commit to https://github.com/llvm/llvm-project/commit/e60b91df1357e6a5f66840581f4d5f57e258c0b4
- Skip decoding README.txt files
- Add a subcommand for generating corpus input
- Create a justfile for invoking common commands
- Change the save directory to `fuzz/runs` rather than `target`
- Add an assembly host implementation on x86
- Adjust print and exit options
- Download LLVM after the cache action
- Move host floats to a separate module
- Move `Op` and `Airity` into `main.rs`
- Extract `FpKind` out of the macro
- Move main toward the top of the file
- Ignore ICE output in git
- Move the bruteforce entrypoint to its separate module
- Port exhaustive tests to the new fuzz setup
- Switch from panics to an error that we can panic from later
- Skip running cxx versions of f8e4m3fn fma
- Improve flexibility of the fuzz binary
- Extract brute force to a separate module
- Download and cache LLVM for the fuzz build
- Build and test all crates in the workspace
- Use the 2024 edition
- Check in generated Rust source
- Replace generated C++ with a source file
- Add a `check-cfg` for `cfg(fuzzing)`
- Remove comments about u128 ABI issues
- Replace shelling out in build.rs with Rust code
- Check in relevant files rather than creating
- Add caching for the test job
- Set a MSRV of 1.65 and test it in CI
- Bump actions/checkout to the latest version (v6)
- Pin Ubuntu versions to 24.04
- Add a timeout for all jobs
- Add a script for downloading the correct LLVM version
- Deny warnings and enable backtraces
- Also run on pushes to `main`
- Invoke `bash` directly rather than using `sh`
- Don't include .github in the shipped crate
- clarify comments for min/max operations

## [0.2.3+llvm-462a31f5a5ab](https://github.com/rust-lang/rustc_apfloat/compare/rustc_apfloat-v0.2.2+llvm-462a31f5a5ab...rustc_apfloat-v0.2.3+llvm-462a31f5a5ab) - 2025-06-11

### Other

- Switch to release-plz for managing releases
- Fix `sig::add_or_sub` and `IeeeFloat::normalize`
- Use runtime env in build script ([#17](https://github.com/rust-lang/rustc_apfloat/pull/17))
- Fix type
