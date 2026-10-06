# Allow overriding the fuzz directories
fuzz_in_unmin := env("FUZZ_IN_UNMIN", "fuzz/runs/in-unmin")
fuzz_in := env("FUZZ_IN", "fuzz/runs/in")
fuzz_out := env("FUZZ_OUT", "fuzz/runs/out")
fuzz_bin := env("FUZZ_BIN", "target/release/rustc_apfloat-fuzz")

alias f := fuzz
alias fb := fuzz-build
alias fp := fuzz-parallel
alias fa := fuzz-attach
alias fq := fuzz-parallel-quit
alias d := decode
alias t := test

_default:
    {{ just_executable() }} --list

# Run non-fuzzing tests
test:
    cargo test --workspace

# Create directories and build the executable, but don't start fuzzing.
fuzz-build:
    mkdir -p "{{ fuzz_in }}"
    echo > "{{ fuzz_in }}/empty"
    cargo afl build -p rustc_apfloat-fuzz --release

# Generate a corpus for fuzzing then run `cmin`
gen: fuzz-build
    rm -rf "{{ fuzz_in_unmin }}" "{{ fuzz_in }}"
    cargo run -p rustc_apfloat-fuzz -- corpus "{{ fuzz_in_unmin }}"
    cargo afl cmin \
        -i "{{ fuzz_in_unmin }}" \
        -o "{{ fuzz_in }}" \
        -T "{{ num_cpus() }}" \
        "{{ fuzz_bin }}"

# Build the instrumented executable and fuzz it. See also: `fuzz-parallel`.
fuzz: fuzz-build
    cargo afl fuzz -i "{{ fuzz_in }}" -o "{{ fuzz_out }}" "{{ fuzz_bin }}"

# Start fuzzing in parallel. Note this must be stopped with fuzz-parallel-quit (see fuzz-parallel.sh).
fuzz-parallel *args: fuzz-build
    etc/fuzz-parallel.sh {{ args }}

# Attach to a running parallel fuzz session
fuzz-attach:
    tmux attach -t afl01

# Stop parallel fuzzing
fuzz-parallel-quit:
    tmux list-sessions | cut -d':' -f1 | grep afl | xargs -iSESS tmux kill-session -t SESS

all-crashes := '"' + fuzz_out + '"/*/crashes/*'

# Print the result of crashes in the fuzz output directory
decode *paths=all-crashes:
    ls {{ all-crashes }}
    cargo run -p rustc_apfloat-fuzz -- decode {{ paths }}
