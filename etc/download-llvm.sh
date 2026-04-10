#!/bin/bash
# Download the version of LLVM matching this crate's port version to test
# against.

set -xux

# Find the version and extract everything after the `-` to get the LLVM hash
version="$(
    cargo metadata --no-deps --format-version=1 |
    jq -r '.packages | .[] | select(.name == "rustc_apfloat") | .version'
)"
llvm_hash="${version##*-}"

target_dir="${CARGO_TARGET_DIR:-target}"
out_dir="$target_dir/llvm-downloads"
mkdir -p "$out_dir"

if [ -d "$out_dir/llvm-project-$llvm_hash" ] && [ "${1:-}" != "-f" ]; then
    echo Download already exists. Pass '-f' to overwrite.
    exit
fi

tgz_url="https://codeload.github.com/llvm/llvm-project/tar.gz/$llvm_hash"
curl -sS "$tgz_url" | tar -C "$out_dir" -xz
