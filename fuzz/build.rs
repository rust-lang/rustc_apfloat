use std::env;
use std::io;
use std::path::PathBuf;
use std::process::Command;

// NB: Any new symbols exported from the C++ source file need to be listed here,
// everything else will get pruned.
const CXX_EXPORTED_SYMBOLS: &[&str] = &[
    "check_error",
    "cxx_apf_eval_op_brainf16",
    "cxx_apf_eval_op_ieee16",
    "cxx_apf_eval_op_ieee32",
    "cxx_apf_eval_op_ieee64",
    "cxx_apf_eval_op_ieee128",
    "cxx_apf_eval_op_ppcdoubledouble",
    "cxx_apf_eval_op_f8e5m2",
    "cxx_apf_eval_op_f8e4m3fn",
    "cxx_apf_eval_op_x87_f80",
];

fn main() -> io::Result<()> {
    // Only rerun if sources run by the fuzzer change
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=cxx");
    println!("cargo::rerun-if-changed=src/apf_fuzz.cpp");

    // NOTE(eddyb) `rustc_apfloat`'s own `build.rs` validated the version string.
    let (_, llvm_commit_hash) = env!("CARGO_PKG_VERSION").split_once("+llvm-").unwrap();
    assert_eq!(llvm_commit_hash.len(), 12);

    // FIXME(eddyb) add a way to disable the C++ build below, or automatically
    // disable it if on an unsupported target (e.g. Windows).
    let cxx = true;
    if !cxx {
        return Ok(());
    }

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR unset"));
    let target_dir = env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.parent().unwrap().join("target"));
    let llvm_root = target_dir.join(format!("llvm-downloads/llvm-project-{llvm_commit_hash}"));

    if !llvm_root.try_exists().is_ok_and(|val| val) {
        panic!(
            "llvm dir `{llvm_root:?}` does not exist or cannot be reached. \
            Perhaps you need to run etc/download-llvm.sh?"
        )
    }

    let llvm_dir = llvm_root.join("llvm");
    let bc_out = out_dir.join("cxx_apf_fuzz.bc");
    let bc_opt_out = out_dir.join("cxx_apf_fuzz.opt.bc");
    let obj_out = out_dir.join("cxx_apf_fuzz.o");
    let archive = out_dir.join("libcxx_apf_fuzz.a");

    // Flags could probably be split between the frontend and backend.
    let clang_codegen_flags = ["-g", "-fPIC", "-fno-exceptions", "-O3", "-march=native"];

    // Note that all commands clear the environment to work around
    // https://github.com/rust-lang/cargo/issues/3676.

    // HACK(eddyb) first compile all the source files into one `.bc` file:
    // - "unity build" (w/ `--include`) lets `-o` specify path (no `--out-dir` sadly)
    // - LLVM `.bc` intermediate allows the steps below to reduce dependencies
    let mut clang = Command::new("clang++");
    clang
        .env_clear()
        .args(["-xc++", "-", "-std=c++17"])
        .args(clang_codegen_flags)
        .arg("-I")
        .arg(llvm_dir.join("include"))
        .arg("-I")
        .arg(llvm_dir.join("lib/Support"))
        .arg("-I")
        .arg(manifest_dir.join("cxx/include"))
        .args([
            "-DNDEBUG",
            "-DHAVE_UNISTD_H",
            "-DLLVM_ON_UNIX",
            "-DLLVM_ENABLE_THREADS=0",
        ])
        .arg("--include")
        .arg(manifest_dir.join("cxx/fuzz_unity_build.cpp"))
        .arg("--include")
        .arg(manifest_dir.join("src/apf_fuzz.cpp"))
        .args(["-c", "-emit-llvm", "-o"])
        .arg(&bc_out);
    println!("+ {clang:?}");
    assert!(clang.status()?.success());

    // Use the `internalize` pass (+ O3) to prune everything unexported.
    let mut opt = Command::new("opt");
    opt.env_clear()
        .arg("--internalize-public-api-list")
        .arg(CXX_EXPORTED_SYMBOLS.join(","))
        .arg(&bc_out)
        .arg("-o")
        .arg(&bc_opt_out);

    // FIXME this was just the `internalize` hack, but had to move `sancov` here, to
    // replicate https://github.com/rust-fuzz/afl.rs/blob/8ece4f9/src/bin/cargo-afl.rs#L370-L372
    // *after* `internalize` & optimizations (to avoid instrumenting dead code).
    let mut passes = "--passes=internalize,default<O3>".to_string();
    if cfg!(fuzzing) {
        opt.args([
            "--sanitizer-coverage-level=3",
            "--sanitizer-coverage-trace-pc-guard",
            "--sanitizer-coverage-prune-blocks=0",
        ]);
        passes.push_str(",sancov-module");
    }

    opt.arg(passes);
    println!("+ {opt:?}");
    assert!(opt.status()?.success());

    // Let Clang do the rest of the work, from the pruned `.bc`.
    let mut clang_final = Command::new("clang++");
    clang_final
        .env_clear()
        .args(clang_codegen_flags)
        .arg(bc_opt_out)
        .args(["-c", "-o"])
        .arg(&obj_out);
    eprintln!("+ {clang_final:?}");
    assert!(clang_final.status()?.success());

    // Construct a linkable archive.
    let mut ar = Command::new("ar");
    ar.env_clear().arg("rc").arg(archive).arg(&obj_out);
    println!("+ {ar:?}");
    assert!(ar.status()?.success());

    println!(
        "cargo:rustc-link-search=native={}",
        out_dir.to_str().unwrap()
    );
    println!("cargo:rustc-link-lib=cxx_apf_fuzz");
    println!("cargo:rustc-link-lib=stdc++");

    Ok(())
}
