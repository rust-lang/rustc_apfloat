#!/bin/bash

# Launch the fuzzer with multiple parallel jobs. Requires tmux.
#
# Taken from: <https://github.com/rust-fuzz/afl.rs/issues/132#issuecomment-997827086>

set -euxo pipefail

# Detect cores
all_cores="$(nproc)"
used_cores="$((all_cores - 2))"
in_dir="${FUZZ_IN:-fuzz/runs/in}"
sync_dir="${FUZZ_OUT:-fuzz/runs/out}"
tmux_window=afl

if [[ "$used_cores" -lt 2 ]]; then
    echo "Error: used_cores < 2"
    exit 1
fi

function print_usage() {
    set +x
    echo "Usage: $0 [-j PROCS]"
    echo ""
    echo "Options:"
    echo "  -o DIR      Output directory"
    echo "  -j PROCS    Number of parallel jobs to use (default: $used_cores)"
    echo "  -h,--help   Print this help and exit"
    set -x
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -j) used_cores="$2"; shift ;;
        -o) sync_dir="$2"; shift ;;
        -h|--help) print_usage; exit 0 ;;
    esac
    shift
done

echo "Using $used_cores out of $all_cores cores"


# Make sure we have at least one input file
mkdir -p "$in_dir"
echo > "$in_dir/empty"

# Start main node
tmux new -d -s "afl01" -n $tmux_window \
    "cargo afl fuzz -i $in_dir -o $sync_dir -M fuzzer01 target/release/rustc_apfloat-fuzz"
echo "Spawned main instance afl01"

# Start secondary instances
for i in $(seq -f "%02.0f" 2 "$used_cores"); do
    tmux new -d -s "afl$i" -n $tmux_window \
        cargo afl fuzz -i $in_dir -o "$sync_dir" -S "fuzzer$i" target/release/rustc_apfloat-fuzz
    echo "Spawned secondary instance afl$i"
done

set +x

# Show status output
echo ""
echo "Tmux sessions:"
tmux ls | grep afl
echo ""
echo "Tmux cheatsheet (shell):"
echo "  Attach:"
echo "    tmux attach -t afl01"
echo "  Kill all sessions:"
echo "    tmux kill-server"
echo ""
echo "Tmux chatsheet (inside tmux):"
echo "  List sessions:"
echo "    Ctrl-b s"
echo "  Switch to next session:"
echo "    Ctrl-b )"
echo "  Switch to prev session:"
echo "    Ctrl-b ("
echo "  Detach:"
echo "    Ctrl-b d"
