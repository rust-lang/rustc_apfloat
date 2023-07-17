use std::process::{Command, ExitCode};

mod ops;

fn main() -> std::io::Result<ExitCode> {
    // HACK(eddyb) disable the default of re-running the build script on *any*
    // change to *the entire source tree* (i.e. the default is roughly `./`).
    println!("cargo:rerun-if-changed=build.rs");

    // NOTE(eddyb) `rustc_apfloat`'s own `build.rs` validated the version string.
    let (_, llvm_commit_hash) = env!("CARGO_PKG_VERSION").split_once("+llvm-").unwrap();
    assert_eq!(llvm_commit_hash.len(), 12);

    let out_dir = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    std::fs::write(out_dir.join("generated_fuzz_ops.rs"), ops::generate_rust())?;

    // FIXME(eddyb) add a way to disable the C++ build below, or automatically
    // disable it if on an unsupported target (e.g. Windows).
    let cxx = true;
    if !cxx {
        return Ok(ExitCode::SUCCESS);
    }

    let mut cxx_exported_symbols = vec![];
    std::fs::write(
        out_dir.join("cxx_apf_fuzz.cpp"),
        ops::generate_cxx(&mut cxx_exported_symbols),
    )?;

    // HACK(eddyb) work around https://github.com/rust-lang/cargo/issues/3676,
    // by removing the env vars that Cargo appears to hardcode.
    const CARGO_HARDCODED_ENV_VARS: &[(&str, &str)] = &[
        ("SSL_CERT_DIR", "/etc/pki/tls/certs"),
        ("SSL_CERT_FILE", "/etc/pki/tls/certs/ca-bundle.crt"),
    ];
    for &(var_name, cargo_hardcoded_value) in CARGO_HARDCODED_ENV_VARS {
        if let Ok(value) = std::env::var(var_name) {
            if value == cargo_hardcoded_value {
                std::env::remove_var(var_name);
            }
        }
    }

    let sh_script_exit_status = Command::new("sh")
        .args(["-c", SH_SCRIPT])
        .envs([
            ("llvm_project_git_hash", llvm_commit_hash),
            ("cxx_apf_fuzz_exports", &cxx_exported_symbols.join(",")),
            (
                "cxx_apf_fuzz_is_fuzzing",
                if cfg!(fuzzing) { "1" } else { "0" },
            ),
        ])
        .status()?;
    Ok(if sh_script_exit_status.success() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

// HACK(eddyb) should avoid shelling out, but for now this will suffice.
const SH_SCRIPT: &str = r#"
set -e

llvm_project_tgz_url="https://codeload.github.com/llvm/llvm-project/tar.gz/$llvm_project_git_hash"
curl -sS "$llvm_project_tgz_url" | tar -C "$OUT_DIR" -xz
llvm="$OUT_DIR"/llvm-project-"$llvm_project_git_hash"/llvm

mkdir -p "$OUT_DIR"/fake-config/llvm/Config
touch "$OUT_DIR"/fake-config/llvm/Config/{abi-breaking,config,llvm-config}.h

# HACK(eddyb) we want standard `assert`s to work, but `NDEBUG` also controls
# unrelated LLVM facilities that are spread all over the place and it's harder
# to compile all of them, than do this workaround where we shadow `assert.h`.
echo -e '#undef NDEBUG\n#include_next <assert.h>\n#define NDEBUG' \
  > "$OUT_DIR"/fake-config/assert.h

# HACK(eddyb) bypass `$llvm/include/llvm/Support/DataTypes.h.cmake`.
mkdir -p "$OUT_DIR"/fake-config/llvm/Support
echo -e '#include <'{math,inttypes,stdint,sys/types}'.h>\n' \
  > "$OUT_DIR"/fake-config/llvm/Support/DataTypes.h

# FIXME(eddyb) maybe split `$clang_codegen_flags` into front-end vs back-end.
clang_codegen_flags="-g -fPIC -fno-exceptions -O3 -march=native"

# HACK(eddyb) first compile all the source files into one `.bc` file:
# - "unity build" (w/ `--include`) lets `-o` specify path (no `--out-dir` sadly)
# - LLVM `.bc` intermediate allows the steps below to reduce dependencies
echo | clang++ -x c++ - -std=c++17 \
  $clang_codegen_flags \
  -I "$llvm"/include \
  -I "$OUT_DIR"/fake-config \
  -DNDEBUG -DHAVE_UNISTD_H -DLLVM_ON_UNIX -DLLVM_ENABLE_THREADS=0 \
  --include="$llvm"/lib/Support/{APInt,APFloat,SmallVector,ErrorHandling}.cpp \
  --include="$OUT_DIR"/cxx_apf_fuzz.cpp \
  -c -emit-llvm -o "$OUT_DIR"/cxx_apf_fuzz.bc

# HACK(eddyb) use the `internalize` pass (+ O3) to prune everything unexported.
opt_passes='internalize,default<O3>'
opt_flags=""
# FIXME(eddyb) this was just the `internalize` hack, but had to move `sancov` here, to
# replicate https://github.com/rust-fuzz/afl.rs/blob/8ece4f9/src/bin/cargo-afl.rs#L370-L372
# *after* `internalize` & optimizations (to avoid instrumenting dead code).
if [ "$cxx_apf_fuzz_is_fuzzing" == "1" ]; then
    opt_passes="$opt_passes,sancov-module"
    opt_flags="--sanitizer-coverage-level=3 \
               --sanitizer-coverage-trace-pc-guard \
               --sanitizer-coverage-prune-blocks=0"
fi
opt \
  --internalize-public-api-list="$cxx_apf_fuzz_exports" \
  --passes="$opt_passes" \
  $opt_flags \
  "$OUT_DIR"/cxx_apf_fuzz.bc \
  -o "$OUT_DIR"/cxx_apf_fuzz.opt.bc

# HACK(eddyb) let Clang do the rest of the work, from the pruned `.bc`.
# FIXME(eddyb) maybe split `$clang_codegen_flags` into front-end vs back-end.
clang++ $clang_codegen_flags \
  "$OUT_DIR"/cxx_apf_fuzz.opt.bc \
  -c -o "$OUT_DIR"/cxx_apf_fuzz.o

llvm-ar rc "$OUT_DIR"/libcxx_apf_fuzz.a "$OUT_DIR"/cxx_apf_fuzz.o

echo cargo:rustc-link-search=native="$OUT_DIR"
echo cargo:rustc-link-lib=cxx_apf_fuzz
echo cargo:rustc-link-lib=stdc++
"#;
