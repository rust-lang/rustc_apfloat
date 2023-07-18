# `rustc_apfloat`<br><sub>(Rust port of the C++ `llvm::APFloat` "softfloat" library)</sub>

## History

LLVM's `APFloat` (aka `llvm::APFloat`) software floating-point (or "softfloat")
library was first ported to Rust (and named `rustc_apfloat`) back in 2017,
in the Rust pull request [`rust-lang/rust#43554`](https://github.com/rust-lang/rust/pull/43554),
as part of an effort to expand Rust compile-time capabilities without sacrificing
determinism (and therefore soundness, if the type-system was involved).

<sub>Note: while using the original C++ `llvm::APFloat` directly would've been an option,
certain high-level API design differences made in the Rust port, without behavioral impact
(C++ raw pointers and dynamic allocations vs Rust generics, traits and `#![no_std]`),
made the Rust port more appealing from a determinism standpoint (mostly thanks to
lacking all 3 of: `unsafe` code, host floating-point use, `std` access - and only
allocating to handle the arbitrary precision needed for conversions to/from decimal),
*even though there was a chance it had correctness issues unique to it*.</sub>

However, that port had a fatal flaw: it was added to the `rust-lang/rust` repository
without its unique licensing status (as a port of a C++ library with its own license)
being properly tracked, communicated, taken into account, etc.  
The end result was years of limbo, mostly chronicled in the Rust issue
[`rust-lang/rust#55993`](https://github.com/rust-lang/rust/issues/55993), in which
the in-tree port couldn't really receive proper updated or even maintenance, due
due to its unclear status.

### Revival (as `rust-lang/rustc_apfloat`)

This repository (`rust-lang/rustc_apfloat`) is the result of a 2022 plan on
[the relevant Zulip topic](https://rust-lang.zulipchat.com/#narrow/stream/231349-t-core.2Flicensing/topic/apfloat), fully put into motion during 2023:
* the `git` history of the in-tree `compiler/rustc_apfloat` library was extracted  
  (see the separate [`rustc_apfloat-git-history-extraction`](https://github.com/LykenSol/rustc_apfloat-git-history-extraction) repository for more details)
* only commits that were *both* necessary *and* had clear copyright status, were kept
* any missing functionality or bug fixes, would have to be either be re-contributed,  
  or rebuilt from the ground up (mostly the latter ended up being done, see below)

Most changes since the original port had been aesthetic (e.g. spell-checking, `rustfmt`),
so little was lost in the process.

Starting from that much smaller "trusted" base:
* everything could use LLVM's new (since 2019) license, "`Apache-2.0 WITH LLVM-exception`"  
  (see the ["Licensing"](#licensing) section below and/or [LICENSE-DETAILS.md](./LICENSE-DETAILS.md) for more details)
* new facilities were built (benchmarks, and [a fuzzer comparing Rust/C++/hardware](#fuzzing))
* excessive testing was performed (via a combination of fuzzing and bruteforce search)
* latent bugs were discovered (e.g. LLVM issues
[#63895](https://github.com/llvm/llvm-project/issues/63895) and
[#63938](https://github.com/llvm/llvm-project/issues/63938))
* the port has been forwarded in time, to include upstream (`llvm/llvm-project`) changes   
  to `llvm::APFloat` over the years (since 2017), removing the need for selective backports

## Versioning

As this is, for the time being, a "living port", tracking upstream (`llvm/llvm-project`)  
`llvm::APFloat` changes, the `rustc_apfloat` crate will have versions of the form:

```
0.X.Y+llvm-ZZZZZZZZZZZZ
```
* `X` is always bumped after semver-incompatible API changes,  
  or when updating the upstream (`llvm/llvm-project`) commit the port is based on
* `Y` is only bumped when other parts of the version don't need to be (e.g. for bug fixes)
* `+llvm-ZZZZZZZZZZZZ` is ["version metadata"](https://doc.rust-lang.org/cargo/reference/resolver.html#version-metadata) (which Cargo itself ignores),  
  and `ZZZZZZZZZZZZ` always holds the first 12 hexadecimal digits of  
  the upstream (`llvm/llvm-project`) `git` commit hash the port is based on


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

## Licensing

This project is licensed under the [Apache 2] with [LLVM exception] license.
For a more complete discussion of this project's licensing, see [LICENSE-DETAILS.md](./LICENSE-DETAILS.md).

[Apache 2]: https://spdx.org/licenses/Apache-2.0.html
[LLVM Exception]: https://spdx.org/licenses/LLVM-exception.html
