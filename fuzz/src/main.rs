#![feature(f16)]
#![feature(f128)]
#![allow(internal_features)] // for the below config
#![feature(cfg_target_has_reliable_f16_f128)]

mod apf_fuzz;
mod exhaustive;

use apf_fuzz::FuzzOp;
use clap::{CommandFactory, Parser, Subcommand};
use io::IsTerminal;
use rustc_apfloat::ieee::{Double, Single};
use rustc_apfloat::{Float, FloatConvert, Round, Status, StatusAnd, ieee};
use std::io::{self, Read};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::{fmt, fs};

use crate::apf_fuzz::Op;

#[derive(Clone, Parser, Debug)]
struct Args {
    /// Disable comparison with C++ (LLVM's original) APFloat
    #[arg(long)]
    ignore_cxx: bool,

    /// Disable comparison with hardware floating-point
    #[arg(long)]
    ignore_hard: bool,

    /// Disable erasure of NaN sign mismatches with hardware floating-point operations
    #[arg(long)]
    strict_hard_nan_sign: bool,

    /// Disable erasure of "which NaN input propagates" mismatches with hardware floating-point operations
    #[arg(long)]
    strict_hard_nan_input_choice: bool,

    /// Hide FMA NaN mismatches for `a * b + NaN` when `a * b` generates a new NaN
    // HACK(eddyb) this is opt-in, not opt-out, because the APFloat behavior, of
    // generating a new NaN (instead of propagating the existing one) is dubious,
    // and may end up changing over time, so the only purpose this serves is to
    // enable fuzzing against hardware without wasting time on these mismatches.
    #[arg(long)]
    ignore_fma_nan_generate_vs_propagate: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Clone, Subcommand, Debug)]
enum Commands {
    /// Check the input from stdin similar to what the fuzzer will run
    Check {
        /// The file to check. If unspecified or `-`, read from stdin.
        file: Option<PathBuf>,
    },
    /// Decode fuzzing in/out testcases (binary serialized `FuzzOp`s)
    Decode { files: Vec<PathBuf> },

    /// Exhaustively test all possible ops and inputs (for 8-bit formats, by default)
    Bruteforce {
        /// Minimum bit-width of floating-point format to test
        #[arg(long, default_value_t = 0)]
        min_width: usize,

        /// Maximum bit-width of floating-point format to test
        #[arg(long, default_value_t = 8)]
        max_width: usize,

        /// Show failures as they happen (useful for larger formats)
        #[arg(short, long)]
        verbose: bool,

        /// Limit testing to FMA ops, and only non-trivial ones (addend != 0.0)
        // HACK(eddyb) this happens to be useful for producing example inputs
        // for one FMA bug (https://github.com/llvm/llvm-project/issues/63895).
        #[arg(long)]
        only_non_trivial_fma: bool,
    },
}

/// Trait implemented for types that describe a floating-point format supported
/// by `rustc_apfloat`, but which themselves only carry the binary representation
/// (instead of `rustc_apfloat` types or native/hardware floating-point types).
///
/// Only some types that implement `FloatRepr` have native/hardware equivalents
/// (i.e. `f32`/`f64` in Rust), and so `hard_eval_fuzz_op_if_supported` will
/// only return `Some` for those.
///
/// Because of the C++ interop (exposed via the `cxx_apf_eval_fuzz_op` method),
/// all types implementing this trait *must* be annotated with `#[repr(C, packed)]`,
/// and `ops.rs` *must* also ensure exactly matching layout for the C++ counterpart.
trait FloatRepr: Copy + Default + Eq + fmt::Display + fmt::Debug {
    type RustcApFloat: Float + FloatConvert<ieee::Single> + FloatConvert<ieee::Double> + fmt::Debug;
    type Repr;

    const BIT_WIDTH: usize = Self::RustcApFloat::BITS;
    const BYTE_LEN: usize = (Self::BIT_WIDTH + 7) / 8;

    // Eventually we will have assembly implementations.
    const HOST_SUPPORTS_FP_ENV: bool = false;

    const NAME: &'static str;

    #[cfg_attr(not(test), expect(dead_code))]
    const KIND: FpKind;

    // HACK(eddyb) this has to be overwritable because we have more than one
    // format with the same `BIT_WIDTH`, so it's not unambiguous on its own.
    const REPR_TAG: u8 = Self::BIT_WIDTH as u8;

    fn short_lowercase_name() -> String {
        Self::NAME.to_ascii_lowercase().replace("ieee", "f")
    }

    fn to_ap(self) -> Self::RustcApFloat;
    fn from_ap(x: Self::RustcApFloat) -> Self;

    // FIXME(const) `[u8; Self::BYTE_LEN]` would be better but requires MGCA.
    fn from_le_bytes(bytes: &[u8]) -> Self;
    fn write_as_le_bytes_into(self, out_bytes: &mut Vec<u8>);

    fn to_bits_u128(self) -> u128;
    fn from_bits_u128(bits: u128) -> Self;

    // HACK(eddyb) this avoids needing another trait (or an `enum` of all formats).
    fn cxx_apf_eval_fuzz_op(op: FuzzOp<Self>) -> Self;
    fn cxx_apf_eval_fuzz_op2(op: Op, rm: Round, a: Self, b: Self, c: Self) -> StatusAnd<Self>;
    // HACK(eddyb) this avoids dealing with separate traits and other complications.
    fn hard_eval_fuzz_op_if_supported(op: FuzzOp<Self>) -> Option<Self>;
    fn host_eval_fuzz_op_if_supported2(
        op: Op,
        rm: Round,
        a: Self,
        b: Self,
        c: Self,
    ) -> Option<StatusAnd<Self>>;
}

macro_rules! float_reprs {
    ($($name:ident($repr:ty) {
        type RustcApFloat = $rs_apf_ty:ty;
        const REPR_TAG = $repr_tag:expr;
        extern fn = $cxx_apf_eval_fuzz_op:ident;
        $(type HardFloat = $hard_float_ty:ty;)?
    })+) => {
        // HACK(eddyb) helper macro used to actually handle all types uniformly.
        macro_rules! dispatch_any_float_repr_by_repr_tag {
            (match $repr_tag_value:ident { for<$ty_var:ident: FloatRepr> => $e:expr }) => {
                // NOTE(eddyb) this doubles as an overlap check: `REPR_TAG`
                // values across *all* `FloatRepr` `impl` *must* be unique.
                #[deny(unreachable_patterns)]
                match $repr_tag_value {
                    $($name::REPR_TAG => {
                        type $ty_var = $name;
                        $e;
                    })+
                    _ => {}
                }
            }
        }

        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Debug, PartialEq)]
        enum FpKind {
            $($name = $repr_tag,)+
        }

        impl FpKind {
            fn from_u8(tag: u8) -> Option<Self> {
                let v = match tag {
                    $(x if x == Self::$name.to_u8() => Self::$name,)+
                    _ => return None,
                };
                Some(v)
            }

            fn to_u8(self) -> u8 {
                self as u8
            }
        }

        $(
            #[repr(C)]
            #[derive(Copy, Clone, Default, PartialEq, Eq)]
            struct $name($repr);

            impl FloatRepr for $name {
                type RustcApFloat = $rs_apf_ty;
                type Repr = $repr;

                const NAME: &'static str = stringify!($name);
                const KIND: FpKind = FpKind::$name;

                const REPR_TAG: u8 = $repr_tag;

                fn to_ap(self) -> Self::RustcApFloat {
                    Self::RustcApFloat::from_bits(self.to_bits_u128())
                }

                fn from_ap(x: Self::RustcApFloat) -> Self {
                    Self::from_bits_u128(x.to_bits())
                }

                fn from_le_bytes(bytes: &[u8]) -> Self {
                    // HACK(eddyb) this allows using e.g. `u128` to hold 80 bits.
                    let mut repr_bytes = [0; std::mem::size_of::<$repr>()];
                    repr_bytes[..Self::BYTE_LEN].copy_from_slice(
                        <&[u8; Self::BYTE_LEN]>::try_from(bytes).unwrap()
                    );
                    Self(<$repr>::from_le_bytes(repr_bytes))
                }
                fn write_as_le_bytes_into(self, out_bytes: &mut Vec<u8>) {
                    out_bytes.extend(&self.0.to_le_bytes()[..Self::BYTE_LEN]);
                }

                fn to_bits_u128(self) -> u128 {
                    self.0.into()
                }
                fn from_bits_u128(bits: u128) -> Self {
                    Self(bits.try_into().unwrap())
                }

                // FIXME: we are discarding status results here and not making use of rounding mode.
                fn cxx_apf_eval_fuzz_op(op: FuzzOp<Self>) -> Self {
                    let (ai, bi, ci)= match op {
                        FuzzOp::Neg(a) => (a.0, 0, 0),
                        FuzzOp::Add(a, b) => (a.0, b.0, 0),
                        FuzzOp::Sub(a, b) => (a.0, b.0, 0),
                        FuzzOp::Mul(a, b) => (a.0, b.0, 0),
                        FuzzOp::Div(a, b) => (a.0, b.0, 0),
                        FuzzOp::Rem(a, b) => (a.0, b.0, 0),
                        FuzzOp::MulAdd(a, b, c) => (a.0, b.0, c.0),
                        FuzzOp::FToI128ToF(a) => (a.0, 0, 0),
                        FuzzOp::FToU128ToF(a) => (a.0, 0, 0),
                        FuzzOp::FToSingleToF(a) => (a.0, 0, 0),
                        FuzzOp::FToDoubleToF(a) => (a.0, 0, 0),
                    };

                    let mut out = 0;
                    let _status = cxx::$cxx_apf_eval_fuzz_op(op.tag(), 0, ai, bi, ci, &mut out);
                    Self(out)
                }

                fn cxx_apf_eval_fuzz_op2(
                    op: Op, rm: Round, a: Self, b: Self, c: Self
                ) -> StatusAnd<Self>
                {

                    let rm = round_to_u8(rm);
                    let mut out = 0;
                    let status = cxx::$cxx_apf_eval_fuzz_op(op.to_u8(), rm, a.0, b.0, c.0, &mut out);
                    let status = cxx::decode_status(status);

                    status.and(Self(out))
                }

                fn hard_eval_fuzz_op_if_supported(_op: FuzzOp<Self>) -> Option<Self> {
                    None $(.or(Some(
                        Self(_op.map(|Self(x)| <$hard_float_ty>::from_bits(x)).eval_hard().to_bits())
                    )))?
                }

                #[allow(unused_variables)]
                fn host_eval_fuzz_op_if_supported2(
                    op: Op, rm: Round, a: Self, b: Self, c: Self
                ) -> Option<StatusAnd<Self>> {
                    None $(.or(
                        Some(eval_host::<$hard_float_ty>(op, rm, a.0, b.0, c.0)?.map(Self))
                    ))?
                }
            }

            impl fmt::Debug for $name {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(f, "{:#0width$x}", self.0, width=(Self::BIT_WIDTH / 4))
                }
            }

            float_reprs!(@display($(via $hard_float_ty)?) for $name);
        )+
    };
    (@display() for $name:ident) => {
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                // HACK(eddyb) this may not be accurate if the fuzzing finds a
                // printing bug specific to `rustc_apfloat`, but we only use
                // this for printing the "human-friendly" representation, with
                // the underlying bit-pattern being the actual trusted value.
                self.to_ap().fmt(f)
            }
        }
    };
    (@display(via $hard_float_ty:ty) for $name:ident) => {
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                // FIXME(eddyb) maybe require that both `rustc_apfloat` and the
                // native float printing agree? (or at least sanity check them)
                <$hard_float_ty>::from_bits(self.0).fmt(f)
            }
        }
    }
}

// FIXME: missing PowerPC semantics
float_reprs! {
    Ieee16(u16) {
        type RustcApFloat = rustc_apfloat::ieee::Half;
        const REPR_TAG = 16;
        extern fn = cxx_apf_eval_op_ieee16;
    }
    Ieee32(u32) {
        type RustcApFloat = rustc_apfloat::ieee::Single;
        const REPR_TAG = 32;
        extern fn = cxx_apf_eval_op_ieee32;
        type HardFloat = f32;
    }
    Ieee64(u64) {
        type RustcApFloat = rustc_apfloat::ieee::Double;
        const REPR_TAG = 64;
        extern fn = cxx_apf_eval_op_ieee64;
        type HardFloat = f64;
    }
    Ieee128(u128) {
        type RustcApFloat = rustc_apfloat::ieee::Quad;
        const REPR_TAG = 128;
        extern fn = cxx_apf_eval_op_ieee128;
    }

    // Non-standard IEEE-like formats.
    F8E5M2(u8) {
        type RustcApFloat = rustc_apfloat::ieee::Float8E5M2;
        const REPR_TAG = 8 + 0;
        extern fn = cxx_apf_eval_op_f8e5m2;
    }
    F8E4M3FN(u8) {
        type RustcApFloat = rustc_apfloat::ieee::Float8E4M3FN;
        const REPR_TAG = 8 + 1;
        extern fn = cxx_apf_eval_op_f8e4m3fn;
    }
    BrainF16(u16) {
        type RustcApFloat = rustc_apfloat::ieee::BFloat;
        const REPR_TAG = 16 + 1;
        extern fn = cxx_apf_eval_op_brainf16;
    }
    X87_F80(u128) {
        type RustcApFloat = rustc_apfloat::ieee::X87DoubleExtended;
        const REPR_TAG = 80;
        extern fn = cxx_apf_eval_op_x87_f80;
    }
}

struct FuzzOpEvalOutputs<F: FloatRepr> {
    rs_apf: F,
    cxx_apf: Option<F>,
    hard: Option<F>,
}

impl<F: FloatRepr> FuzzOpEvalOutputs<F> {
    fn all_match(self) -> bool {
        [self.cxx_apf, self.hard]
            .into_iter()
            .flatten()
            .all(|x| x == self.rs_apf)
    }
}

impl<F: FloatRepr> FuzzOp<F>
// FIXME(eddyb) such bounds shouldn't be here, but `FloatRepr` can't imply them.
where
    ieee::Single: FloatConvert<F::RustcApFloat>,
    ieee::Double: FloatConvert<F::RustcApFloat>,
{
    // FIXME(eddyb) add `seed` subcommand using this method, and a set
    // of basic examples, e.g. every `FuzzOp` variant with `0.0` for all inputs
    // (and/or maybe testcases from known and/or fixed bugs, too).
    #[allow(unused)]
    fn encode_into(self, out_bytes: &mut Vec<u8>) {
        out_bytes.push(self.tag());
        self.map(|x| x.write_as_le_bytes_into(out_bytes));
    }

    fn eval(self, cli_args: &Args) -> FuzzOpEvalOutputs<F> {
        let mut out = FuzzOpEvalOutputs {
            rs_apf: F::from_bits_u128(
                self.map(F::to_bits_u128)
                    .map(F::RustcApFloat::from_bits)
                    .eval_rs_apf()
                    .to_bits(),
            ),
            cxx_apf: if !cli_args.ignore_cxx {
                Some(F::cxx_apf_eval_fuzz_op(self))
            } else {
                None
            },
            hard: if !cli_args.ignore_hard {
                F::hard_eval_fuzz_op_if_supported(self)
            } else {
                None
            },
        };

        out.hard = out.hard.map(|out_hard| {
            let mut out_hard_bits = out_hard.to_bits_u128();

            // HACK(eddyb) to avoid putting this behind a `HasHardFloat` bound,
            // we hardcode some aspects of the IEEE binary float representation,
            // relying on `rustc_apfloat`-provided constants as a source of truth.
            let sign_bit_mask = 1 << (F::BIT_WIDTH - 1);
            let exp_mask = F::RustcApFloat::INFINITY.to_bits();
            let sig_mask = (1 << exp_mask.trailing_zeros()) - 1;
            assert_eq!(
                sign_bit_mask | exp_mask | sig_mask,
                !0 >> (128 - F::BIT_WIDTH)
            );
            assert!((sig_mask + 1).is_power_of_two());
            assert!(((exp_mask | sig_mask) + 1).is_power_of_two());
            assert_eq!(
                sign_bit_mask.count_ones() + exp_mask.count_ones() + sig_mask.count_ones(),
                F::BIT_WIDTH as u32
            );
            let qnan_bit_mask = (sig_mask + 1) >> 1;
            assert!(qnan_bit_mask.is_power_of_two());
            assert_eq!(exp_mask | qnan_bit_mask, F::RustcApFloat::NAN.to_bits());

            let is_nan = |bits| {
                let is_nan = (bits & exp_mask) == exp_mask && (bits & sig_mask) != 0;
                assert_eq!(F::RustcApFloat::from_bits(bits).is_nan(), is_nan);
                is_nan
            };

            // Allow using CLI flags to toggle whether differences vs hardware are
            // erased (by copying e.g. signs from the `rustc_apfloat` result) or kept.
            // FIXME(eddyb) figure out how much we can really validate against hardware.
            let mut strict_nan_bits_mask = !0;
            if !cli_args.strict_hard_nan_sign {
                strict_nan_bits_mask &= !sign_bit_mask;
            };

            let rs_apf_bits = out.rs_apf.to_bits_u128();
            if is_nan(out_hard_bits) && is_nan(rs_apf_bits) {
                out_hard_bits &= strict_nan_bits_mask;
                out_hard_bits |= rs_apf_bits & !strict_nan_bits_mask;

                // There is still a NaN payload difference, check if they both
                // are propagated inputs (verbatim or at most "quieted" if SNaN),
                // because in some cases with multiple NaN inputs, something
                // (hardware or even e.g. LLVM passes or instruction selection)
                // along the way from Rust code to final results, can end up
                // causing a different input NaN to get propagated to the result.
                if !cli_args.strict_hard_nan_input_choice && out_hard_bits != rs_apf_bits {
                    let out_nan_is_propagated_input = |out_nan_bits| {
                        assert!(is_nan(out_nan_bits));
                        let mut found_any_matching_inputs = false;
                        self.map(F::to_bits_u128).map(|in_bits| {
                            // NOTE(eddyb) this `is_nan` check is important, as
                            // `INFINITY.to_bits() | qnan_bit_mask == NAN.to_bits()`,
                            // i.e. seeting the QNaN is more than enough to turn
                            // a non-NaN (infinities, specifically) into a NaN.
                            if is_nan(in_bits) {
                                // Make sure to "quiet" (i.e. turn SNaN into QNaN)
                                // the input first, as propagation does (in the
                                // default exception handling mode, at least).
                                if (in_bits | qnan_bit_mask) & strict_nan_bits_mask
                                    == out_nan_bits & strict_nan_bits_mask
                                {
                                    found_any_matching_inputs = true;
                                }
                            }
                        });
                        found_any_matching_inputs
                    };
                    if out_nan_is_propagated_input(out_hard_bits)
                        && out_nan_is_propagated_input(rs_apf_bits)
                    {
                        out_hard_bits = rs_apf_bits;
                    }
                }

                // HACK(eddyb) last chance to hide a NaN payload difference,
                // in this case for FMAs of the form `a * b + NaN`, when `a * b`
                // generates a new NaN (which hardware can ignore in favor of the
                // existing NaN, but APFloat returns the fresh new NaN instead).
                if cli_args.ignore_fma_nan_generate_vs_propagate && out_hard_bits != rs_apf_bits {
                    if let FuzzOp::MulAdd(a, b, c) = self.map(F::to_bits_u128) {
                        if !is_nan(a)
                            && !is_nan(b)
                            && is_nan(c)
                            && out_hard_bits & strict_nan_bits_mask
                                == (c | qnan_bit_mask) & strict_nan_bits_mask
                            && rs_apf_bits == F::RustcApFloat::NAN.to_bits()
                        {
                            out_hard_bits = rs_apf_bits;
                        }
                    }
                }
            }

            F::from_bits_u128(out_hard_bits)
        });

        out
    }

    fn print_op_and_eval_outputs(self, cli_args: &Args) {
        println!(
            "  {}.{:?}",
            F::short_lowercase_name(),
            self.map(FloatPrintHelper)
        );

        // HACK(eddyb) this lets us show all files even if some cause panics.
        let FuzzOpEvalOutputs {
            rs_apf,
            cxx_apf,
            hard,
        } = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.eval(cli_args))) {
            Ok(out) => out,
            Err(_) => {
                // HACK(eddyb) this tries to reproduce assertion failures in C++.
                if !cli_args.ignore_cxx {
                    F::cxx_apf_eval_fuzz_op(self);
                }
                return;
            }
        };
        let print = |x, label| {
            print!("   => {:?} ({label})", FloatPrintHelper(x));
            if x != rs_apf {
                print!(" <- {}!!! MISMATCH !!!{}", term().red_b, term().rst)
            }
            println!();
        };
        print(rs_apf, "Rust / rustc_apfloat");
        cxx_apf.map(|x| print(x, "C++ / llvm::APFloat"));
        hard.map(|x| print(x, "native hardware floats"));
    }

    /// [`Commands::Bruteforce`] implementation (for a specific choice of `F`),
    /// returning `Err(mismatch_count)` if there were any mismatches.
    //
    // HACK(eddyb) this is a method here because of the bounds `eval` needs, which
    // are thankfully on the whole `impl`, so `Self::eval` is callable.
    fn bruteforce(cli_args: &Args) -> Result<(), NonZeroUsize>
    where
        F: Send + 'static,
    {
        exhaustive::run_exhaustive::<F>(cli_args)
    }
}

fn main() {
    let cli_args = Args::parse();

    if let Some(cmd) = &cli_args.command {
        match cmd {
            Commands::Check { file } => {
                let mut buf = Vec::new();
                let reader: &mut dyn Read = match file {
                    Some(fname) if fname == "-" => &mut io::stdin(),
                    Some(fname) => &mut fs::File::open(fname).unwrap(),
                    None => &mut io::stdin(),
                };
                reader.read_to_end(&mut buf).unwrap();
                match decode_eval_check(&buf, &cli_args, Mode::Assert) {
                    Ok(()) => (),
                    Err(e) => println!("error: {e} (no panic raised)"),
                }
            }
            Commands::Decode { files } => run_decode_subcmd(files, &cli_args),
            Commands::Bruteforce { .. } => {
                let mut any_mismatches = false;
                for repr_tag in 0..=u8::MAX {
                    dispatch_any_float_repr_by_repr_tag!(match repr_tag {
                        for<F: FloatRepr> => {
                            any_mismatches |= FuzzOp::<F>::bruteforce(&cli_args).is_err();
                        }
                    });
                }
                if any_mismatches {
                    // FIXME(eddyb) use `fn main() -> ExitStatus`.
                    std::process::exit(1);
                }
            }
        }
        return;
    }

    // HACK(eddyb) `#[cfg(fuzzing)] {...}` used instead of `if cfg!(fuzzing) {...}`
    // because the latter can still cause the `afl` crate to be linked, and it
    // depends on native libraries that are only available under `cargo afl ...`.
    #[cfg(fuzzing)]
    if true {
        // FIXME(eddyb) make the first argument (panic hook choice) a CLI toggle.
        afl::fuzz(true, |buf| {
            // Discard decoding errors
            let _ = decode_eval_check(&buf, &cli_args, Mode::Assert);
        });

        return;
    }

    // FIXME(eddyb) add better docs for all of this.
    // FIXME(eddyb) add `seed` subcommand using `FuzzOp::encode_into`, and a set
    // of basic examples, e.g. every `FuzzOp` variant with `0.0` for all inputs
    // (and/or maybe testcases from known and/or fixed bugs, too).
    Args::command().print_long_help().unwrap();
    eprintln!(
        "\n\
        To fuzz `rustc_apfloat`, you must use `cargo afl`:\n\
        - `cargo install afl`\n\
        - build with `cargo afl build -p rustc_apfloat-fuzz --release`\n\
        - seed with `mkdir fuzz/in-foo && echo > fuzz/in-foo/empty`\n\
        - run with `cargo afl fuzz -i fuzz/in-foo -o fuzz/out-foo target/release/rustc_apfloat-fuzz`\n\
        "
    );
    std::process::exit(1);
}

/// Errors from improperly formed inputs that cause an exit from the fuzzer but do not raise
/// a test failing error.
#[derive(Clone, Copy, Debug)]
enum DecodeError {
    LenShorterThan(u8, Option<u16>),
    LenNotExactly(u8, Option<u16>),
    InvalidOpcode(u8),
    InvalidKind(u8),
    InvalidRound(u8),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::LenShorterThan(exp, act) => write!(
                f,
                "invalid length; expected at least {exp} bytes, got {act:?}"
            ),
            DecodeError::LenNotExactly(exp, act) => {
                write!(f, "invalid length; expected {exp} bytes, got {act:?}")
            }
            DecodeError::InvalidOpcode(v) => write!(f, "invalid opcode {v:#04x}"),
            DecodeError::InvalidKind(v) => write!(f, "invalid opcode {v:#04x}"),
            DecodeError::InvalidRound(v) => write!(f, "invalid opcode {v:#04x}"),
        }
    }
}

/// Configuration for the current operation.
#[derive(Debug)]
struct EvalCfg {
    kind: FpKind,
    op: Op,
    rm: Round,
    run_cxx: bool,
    run_host: bool,
    ignore_cxx: Option<&'static str>,
    ignore_cxx_status: Option<&'static str>,
    cli_strict_host_nan_sign: bool,
    cli_strict_host_nan_input_choice: bool,
    cli_ignore_fma_nan_generate_vs_propagate: bool,
}

impl EvalCfg {
    fn new(kind: FpKind, op: Op, rm: Round, cli_args: &Args) -> Self {
        let mut ret = Self {
            kind,
            op,
            rm,
            run_cxx: !cli_args.ignore_cxx,
            run_host: !cli_args.ignore_hard,
            ignore_cxx: None,
            ignore_cxx_status: None,
            cli_strict_host_nan_sign: false,
            cli_strict_host_nan_input_choice: false,
            cli_ignore_fma_nan_generate_vs_propagate: false,
        };

        // FIXME: these are XFAILS / ignores and should periodically be reviewed
        if ret.kind == FpKind::X87_F80 {
            // LLVM never seems to set INVALID
            ret.ignore_cxx_status = Some("LLVM does not set INVALID on X87_F80")
        }
        // FIXME(f8): We often disagree with LLVM, more research is neeed to see which
        // implementation is correct.
        if ret.kind == FpKind::F8E4M3FN {
            ret.ignore_cxx = Some("f8e4m3fn may be broken");
        }
        if ret.kind == FpKind::F8E5M2 {
            ret.ignore_cxx = Some("f8e5m2 may be broken");
        }

        // Apply CLI config
        ret.cli_strict_host_nan_sign |= cli_args.strict_hard_nan_sign;
        ret.cli_strict_host_nan_input_choice |= cli_args.strict_hard_nan_input_choice;
        ret.cli_ignore_fma_nan_generate_vs_propagate |=
            cli_args.ignore_fma_nan_generate_vs_propagate;

        ret
    }

    /// Extract from a blob with CLI overrides. Return the decoded config and the remaining
    /// bytestream.
    fn decode<'a>(data: &'a [u8], cli_args: &Args) -> Result<(EvalCfg, &'a [u8]), DecodeError> {
        let Some((&[kind_tag, op_tag, rm_tag], rest)) = data.split_first_chunk() else {
            return Err(DecodeError::LenShorterThan(3, data.len().try_into().ok()));
        };

        let Some(kind) = FpKind::from_u8(kind_tag) else {
            return Err(DecodeError::InvalidKind(op_tag));
        };

        let Some(op) = Op::from_u8(op_tag) else {
            return Err(DecodeError::InvalidOpcode(op_tag));
        };

        let Some(rm) = round_from_u8(rm_tag) else {
            return Err(DecodeError::InvalidRound(op_tag));
        };

        let ret = EvalCfg::new(kind, op, rm, cli_args);
        Ok((ret, rest))
    }
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    /// Print the results of checking but don't panic. Used for decoding error results.
    PrintOnly,
    /// Exit on failure, used when actually running the fuzzer.
    Assert,
}

/// Decode and evaluate all passed files without exiting on mismatches.
fn run_decode_subcmd(files: &[PathBuf], cli_args: &Args) {
    let mut buf = Vec::new();
    for path in files {
        println!("{}{}:{}", term().dim, path.display(), term().rst);

        buf.clear();
        let mut f = fs::File::open(path).unwrap();
        f.read_to_end(&mut buf).unwrap();

        match decode_eval_check(&buf, cli_args, Mode::PrintOnly) {
            Ok(()) => (),
            Err(e) => println!("error decoding file: {e}"),
        }
    }
}

/// Main runner: decode a config, inputs based on that config, and then evaluate the results
/// for Rust, LLVM APFloat, and the host.
fn decode_eval_check(data: &[u8], cli_args: &Args, mode: Mode) -> Result<(), DecodeError> {
    let (cfg, data) = EvalCfg::decode(data, cli_args)?;
    match cfg.kind {
        FpKind::Ieee16 => {
            let (a, b, c, r) = decode_for_ty_eval::<Ieee16>(&cfg, data)?;
            r.check_all(&cfg, a, b, c, mode);
        }
        FpKind::Ieee32 => {
            let (a, b, c, r) = decode_for_ty_eval::<Ieee32>(&cfg, data)?;
            r.check_all(&cfg, a, b, c, mode);
        }
        FpKind::Ieee64 => {
            let (a, b, c, r) = decode_for_ty_eval::<Ieee64>(&cfg, data)?;
            r.check_all(&cfg, a, b, c, mode);
        }
        FpKind::Ieee128 => {
            let (a, b, c, r) = decode_for_ty_eval::<Ieee128>(&cfg, data)?;
            r.check_all(&cfg, a, b, c, mode);
        }
        FpKind::F8E5M2 => {
            let (a, b, c, r) = decode_for_ty_eval::<F8E5M2>(&cfg, data)?;
            r.check_all(&cfg, a, b, c, mode);
        }
        FpKind::F8E4M3FN => {
            let (a, b, c, r) = decode_for_ty_eval::<F8E4M3FN>(&cfg, data)?;
            r.check_all(&cfg, a, b, c, mode);
        }
        FpKind::BrainF16 => {
            let (a, b, c, r) = decode_for_ty_eval::<BrainF16>(&cfg, data)?;
            r.check_all(&cfg, a, b, c, mode);
        }
        FpKind::X87_F80 => {
            let (a, b, c, r) = decode_for_ty_eval::<X87_F80>(&cfg, data)?;
            r.check_all(&cfg, a, b, c, mode);
        }
    }

    Ok(())
}

/// Decode operands for a given type and operation, then evaluate.
fn decode_for_ty_eval<F: FloatRepr>(
    cfg: &EvalCfg,
    data: &[u8],
) -> Result<(F, F, F, FuzzOpEvalOutputs2<F>), DecodeError>
where
    Single: FloatConvert<<F as FloatRepr>::RustcApFloat>,
    Double: FloatConvert<<F as FloatRepr>::RustcApFloat>,
{
    let (a, b, c) = decode_operands::<F>(cfg.op, data)?;

    // Evaluate the APFloat version as well as all possible references.
    let r = FuzzOpEvalOutputs2 {
        rs_apf: eval_rust_ap(cfg.op, cfg.rm, a, b, c),
        cxx_apf: cfg
            .run_cxx
            .then(|| F::cxx_apf_eval_fuzz_op2(cfg.op, cfg.rm, a, b, c)),
        host: cfg
            .run_host
            .then(|| F::host_eval_fuzz_op_if_supported2(cfg.op, cfg.rm, a, b, c))
            .flatten(),
    };

    Ok((a, b, c, r))
}

/// Given a N-arity opcode, decode N `F`s from `data` and return zero in any remaining
/// positions.
fn decode_operands<F: FloatRepr>(op: Op, data: &[u8]) -> Result<(F, F, F), DecodeError> {
    let arity = op.airity() as usize;
    let req_len = arity * F::BYTE_LEN;
    if data.len() != req_len {
        return Err(DecodeError::LenNotExactly(
            req_len.try_into().unwrap(),
            data.len().try_into().ok(),
        ));
    }

    let mut ret: (F, F, F) = Default::default();
    ret.0 = F::from_le_bytes(&data[..F::BYTE_LEN]);
    if arity > 1 {
        ret.1 = F::from_le_bytes(&data[F::BYTE_LEN..(F::BYTE_LEN * 2)]);
    }
    if arity > 2 {
        ret.2 = F::from_le_bytes(&data[(F::BYTE_LEN * 2)..(F::BYTE_LEN * 3)]);
    }

    Ok(ret)
}

/// Collected outputs from all input sources.
struct FuzzOpEvalOutputs2<F> {
    rs_apf: StatusAnd<F>,
    cxx_apf: Option<StatusAnd<F>>,
    host: Option<StatusAnd<F>>,
}

impl<F: FloatRepr> FuzzOpEvalOutputs2<F> {
    /// Validate that outputs are correct. May unconitionally print all values or exit on error
    /// base on the mode (mismatches always print).
    fn check_all(&self, cfg: &EvalCfg, a: F, b: F, c: F, mode: Mode) {
        let (always_print, always_assert) = match mode {
            Mode::PrintOnly => (true, false),
            Mode::Assert => (false, true),
        };

        let print_float = |x: StatusAnd<F>,
                           label,
                           error,
                           stat_error,
                           ignore: Option<&str>,
                           ignore_stat: Option<&str>| {
            print!(
                "   => {:?} {:?} ({label})",
                FloatPrintHelper(x.value),
                x.status
            );
            if error {
                let c = if ignore.is_some() {
                    term().yel_b
                } else {
                    term().red_b
                };
                print!(" <- {c}!!! MISMATCH ");
                if let Some(reason) = ignore {
                    print!("(ignored, {reason}) ");
                }
                print!("!!!{}", term().rst);
            } else if stat_error {
                let c = if ignore.is_some() {
                    term().yel_b
                } else {
                    term().red_b
                };
                print!(" <- {c}!!! STATUS MISMATCH ");
                if let Some(reason) = ignore_stat {
                    print!("(ignored, {reason}) ");
                }
                print!("!!!{}", term().rst);
            }

            println!();
        };

        #[derive(Debug)]
        struct Results {
            cxx_error: bool,
            cxx_ignore: Option<&'static str>,
            cxx_stat_error: bool,
            cxx_stat_ignore: Option<&'static str>,
            host_error: bool,
            host_ignore: Option<&'static str>,
            host_stat_error: bool,
            host_stat_ignore: Option<&'static str>,
        }

        let mut res = Results {
            cxx_error: false,
            cxx_ignore: cfg.ignore_cxx,
            cxx_stat_error: false,
            cxx_stat_ignore: cfg.ignore_cxx_status,
            host_error: false,
            host_ignore: None,
            host_stat_error: false,
            host_stat_ignore: (!F::HOST_SUPPORTS_FP_ENV).then_some("host does not support fpenv"),
        };

        if let Some(cxx_res) = self.cxx_apf {
            res.cxx_error = self.rs_apf.value != cxx_res.value;
            res.cxx_stat_error = self.rs_apf.status != cxx_res.status;
            res.cxx_ignore = res
                .cxx_ignore
                .or_else(|| ignore_cxx(cfg, a, b, c, self.rs_apf.value, cxx_res.value));
        }

        if let Some(host_res) = self.host {
            res.host_error = self.rs_apf.value != host_res.value;
            res.host_stat_error = self.rs_apf.status != host_res.status;
            res.host_ignore = res
                .host_ignore
                .or_else(|| ignore_host(cfg, a, b, c, self.rs_apf.value, host_res.value));
        }

        let failure = (res.cxx_error && res.cxx_ignore.is_none())
            || (res.cxx_stat_error && res.cxx_ignore.is_none() && res.cxx_stat_ignore.is_none())
            || (res.host_error && res.host_ignore.is_none())
            || (res.host_stat_error && res.host_ignore.is_none() && res.host_stat_ignore.is_none());

        if always_print || failure {
            print!("  {}.{:?}(", F::short_lowercase_name(), cfg.op,);

            let airity = cfg.op.airity() as u8;
            print!("{:?}", FloatPrintHelper(a));
            if airity > 1 {
                print!(", {:?}", FloatPrintHelper(b));
            }
            if airity > 2 {
                print!(", {:?}", FloatPrintHelper(c));
            }
            println!(", {:?})", cfg.rm);

            print_float(
                self.rs_apf,
                "Rust / rustc_apfloat",
                false,
                false,
                None,
                None,
            );
            if let Some(cxx_res) = self.cxx_apf {
                print_float(
                    cxx_res,
                    "C++ / llvm::APFloat",
                    res.cxx_error,
                    res.cxx_stat_error,
                    res.cxx_ignore,
                    res.cxx_stat_ignore,
                );
            }
            if let Some(host_res) = self.host {
                print_float(
                    host_res,
                    "native host floats",
                    res.host_error,
                    res.host_stat_error,
                    res.host_ignore,
                    res.host_stat_ignore,
                );
            }
        }

        if always_assert && failure {
            panic!("mismatched results: {cfg:#?}\n{res:#?}");
        }
    }
}

/// Return `Some(reason)` if we can ignore mismatches against the host, `None` otherwise.
fn ignore_cxx<F: FloatRepr>(
    cfg: &EvalCfg,
    a: F,
    _b: F,
    _c: F,
    rs_apf: F,
    cxx_res: F,
) -> Option<&'static str> {
    let rs_apf_bits = rs_apf.to_bits_u128();
    let cxx_bits = cxx_res.to_bits_u128();
    if rs_apf_bits == cxx_bits {
        return None;
    }

    let Masks { qnan_bit_mask, .. } = Masks::for_float::<F>();

    // For the F1->F2->F1 conversions where F1 and F2 are the same type, it seems like LLVM
    // doesn't actually do a conversion which means that sNaNs do not wind up quiet.
    if ((cfg.kind == FpKind::Ieee32 && cfg.op == Op::FToSingleToF)
        || (cfg.kind == FpKind::Ieee64 && cfg.op == Op::FToDoubleToF))
        && a.to_ap().is_signaling()
        && (cxx_bits | qnan_bit_mask) == rs_apf_bits
    {
        return Some("unquieted sNaN for same-size float");
    }

    None
}

/// Return `Some(reason)` if we can ignore mismatches against the host, `None` otherwise.
fn ignore_host<F: FloatRepr>(
    cfg: &EvalCfg,
    a: F,
    b: F,
    c: F,
    rs_apf: F,
    host_res: F,
) -> Option<&'static str> {
    let rs_apf_bits = rs_apf.to_bits_u128();
    let host_bits = host_res.to_bits_u128();
    if rs_apf_bits == host_bits {
        return None;
    }

    let Masks {
        sign_bit_mask,
        exp_mask,
        sig_mask,
        qnan_bit_mask,
    } = Masks::for_float::<F>();

    let is_nan = |bits| {
        let is_nan = (bits & exp_mask) == exp_mask && (bits & sig_mask) != 0;
        assert_eq!(F::RustcApFloat::from_bits(bits).is_nan(), is_nan);
        is_nan
    };

    // Everything else is for handling NaNs.
    if !(is_nan(host_bits) && is_nan(rs_apf_bits)) {
        return None;
    }

    // Allow using CLI flags to toggle whether differences vs hardware are
    // erased (by copying e.g. signs from the `rustc_apfloat` result) or kept.
    let zero_sign_mask = if cfg.cli_strict_host_nan_sign {
        u128::MAX
    } else {
        !sign_bit_mask
    };

    let host_zero_sign = host_bits & zero_sign_mask;
    let rs_apf_zero_sign = rs_apf_bits & zero_sign_mask;

    if host_zero_sign == rs_apf_zero_sign {
        return Some("ignoring NaN sign");
    }

    // There is still a NaN payload difference, check if they both
    // are propagated inputs (verbatim or at most "quieted" if SNaN),
    // because in some cases with multiple NaN inputs, something
    // (hardware or even e.g. LLVM passes or instruction selection)
    // along the way from Rust code to final results, can end up
    // causing a different input NaN to get propagated to the result.
    if !cfg.cli_strict_host_nan_input_choice {
        let mut host_propagated_any = true;
        let mut rs_apf_propagated_any = true;

        for m in [a, b, c] {
            let in_bits = m.to_bits_u128();
            // NOTE(eddyb) this `is_nan` check is important, as
            // `INFINITY.to_bits() | qnan_bit_mask == NAN.to_bits()`,
            // i.e. seeting the QNaN is more than enough to turn
            // a non-NaN (infinities, specifically) into a NaN.
            if !is_nan(in_bits) {
                continue;
            }

            // Make sure to "quiet" (i.e. turn SNaN into QNaN)
            // the input first, as propagation does (in the
            // default exception handling mode, at least).
            let in_quiet = in_bits | qnan_bit_mask;
            let in_zero_sign = in_quiet & zero_sign_mask;

            if in_zero_sign == host_zero_sign {
                host_propagated_any = true;
            }

            if in_zero_sign == rs_apf_zero_sign {
                rs_apf_propagated_any = true;
            }
        }

        // Note that this allows propagating any NaN, even if not the same.
        if host_propagated_any && rs_apf_propagated_any {
            return Some("both are propagated inputs");
        }
    }

    // HACK(eddyb) last chance to hide a NaN payload difference,
    // in this case for FMAs of the form `a * b + NaN`, when `a * b`
    // generates a new NaN (which hardware can ignore in favor of the
    // existing NaN, but APFloat returns the fresh default NaN instead).
    if cfg.cli_ignore_fma_nan_generate_vs_propagate {
        if cfg.op == Op::MulAdd
            && !is_nan(a.to_bits_u128())
            && !is_nan(b.to_bits_u128())
            && is_nan(c.to_bits_u128())
            && host_zero_sign == (c.to_bits_u128() | qnan_bit_mask) & zero_sign_mask
            && rs_apf_bits == F::RustcApFloat::NAN.to_bits()
        {
            return Some("fresh NaN from FMA");
        }
    }

    // None of the patterns matched, we're not going to ignore the mismatch
    None
}

/// Execute the requested operation as an AP float with the given rounding mode.
fn eval_rust_ap<F: FloatRepr>(op: Op, rm: Round, a: F, b: F, c: F) -> StatusAnd<F>
where
    Single: FloatConvert<<F as FloatRepr>::RustcApFloat>,
    Double: FloatConvert<<F as FloatRepr>::RustcApFloat>,
{
    let res = match op {
        Op::Neg => Status::OK.and(-a.to_ap()),
        Op::Add => a.to_ap().add_r(b.to_ap(), rm),
        Op::Sub => a.to_ap().sub_r(b.to_ap(), rm),
        Op::Mul => a.to_ap().mul_r(b.to_ap(), rm),
        Op::Div => a.to_ap().div_r(b.to_ap(), rm),
        // FIXME: rem disregards rounding mode
        Op::Rem => a.to_ap() % b.to_ap(),
        Op::MulAdd => a.to_ap().mul_add_r(b.to_ap(), c.to_ap(), rm),
        // FIXME: the below operations discard a status. We should turn them into
        // unidirectional operations.
        Op::FToI128ToF => {
            F::RustcApFloat::from_i128_r(a.to_ap().to_i128_r(128, rm, &mut false).value, rm)
        }
        Op::FToU128ToF => {
            F::RustcApFloat::from_u128_r(a.to_ap().to_u128_r(128, rm, &mut false).value, rm)
        }
        Op::FToSingleToF => FloatConvert::<F::RustcApFloat>::convert_r(
            FloatConvert::<ieee::Single>::convert_r(a.to_ap(), rm, &mut false).value,
            rm,
            &mut false,
        ),
        Op::FToDoubleToF => FloatConvert::<F::RustcApFloat>::convert_r(
            FloatConvert::<ieee::Double>::convert_r(a.to_ap(), rm, &mut false).value,
            rm,
            &mut false,
        ),
    };

    res.map(F::from_ap)
}

/// Execute the requested operation on the host with the given rounding mode, if possible. If
/// the operation is not possible for whatever reason, return `None`.
fn eval_host<F: HostFloat>(
    op: Op,
    rm: Round,
    a: F::UInt,
    b: F::UInt,
    c: F::UInt,
) -> Option<StatusAnd<F::UInt>> {
    let a = F::from_bits(a);
    let b = F::from_bits(b);
    let c = F::from_bits(c);

    let res = match op {
        Op::Neg => Status::OK.and(a.neg()),
        Op::Add => a.add_r(b, rm)?,
        Op::Sub => a.sub_r(b, rm)?,
        Op::Mul => a.mul_r(b, rm)?,
        Op::Div => a.div_r(b, rm)?,
        // FIXME: rem disregards rounding mode
        Op::Rem => Status::OK.and(a.rem(b)),
        Op::MulAdd => a.mul_add_r(b, c, rm)?,
        // FIXME: the below operations discard a status. We should turn them into
        // unidirectional operations.
        Op::FToI128ToF => F::from_i128_r(a.to_i128_r(rm)?.value, rm)?,
        Op::FToU128ToF => F::from_u128_r(a.to_u128_r(rm)?.value, rm)?,
        Op::FToSingleToF => F::from_single_r(a.to_single_r(rm)?.value, rm)?,
        Op::FToDoubleToF => F::from_double_r(a.to_double_r(rm)?.value, rm)?,
    };

    Some(res.map(F::to_bits))
}

/// Abstraction over host float operations. If the requested rounding mode is not supported,
/// return `None`.
trait HostFloat: Copy + Sized + fmt::Debug {
    type UInt: Copy + fmt::LowerHex;
    fn from_bits(bits: Self::UInt) -> Self;
    fn to_bits(self) -> Self::UInt;
    fn neg(self) -> Self;
    fn add_r(self, other: Self, rm: Round) -> Option<StatusAnd<Self>>;
    fn sub_r(self, other: Self, rm: Round) -> Option<StatusAnd<Self>>;
    fn mul_r(self, other: Self, rm: Round) -> Option<StatusAnd<Self>>;
    fn div_r(self, other: Self, rm: Round) -> Option<StatusAnd<Self>>;
    fn rem(self, other: Self) -> Self;
    fn mul_add_r(self, mul: Self, add: Self, rm: Round) -> Option<StatusAnd<Self>>;
    fn to_i128_r(self, rm: Round) -> Option<StatusAnd<i128>>;
    fn from_i128_r(x: i128, rm: Round) -> Option<StatusAnd<Self>>;
    fn to_u128_r(self, rm: Round) -> Option<StatusAnd<u128>>;
    fn from_u128_r(x: u128, rm: Round) -> Option<StatusAnd<Self>>;
    fn to_double_r(self, rm: Round) -> Option<StatusAnd<f64>>;
    fn from_double_r(x: f64, rm: Round) -> Option<StatusAnd<Self>>;
    fn to_single_r(self, rm: Round) -> Option<StatusAnd<f32>>;
    fn from_single_r(x: f32, rm: Round) -> Option<StatusAnd<Self>>;
}

macro_rules! impl_host_float {
    ($ty:ty, $ity:ty) => {
        impl HostFloat for $ty {
            type UInt = $ity;
            fn from_bits(bits: Self::UInt) -> Self {
                Self::from_bits(bits)
            }
            fn to_bits(self) -> Self::UInt {
                self.to_bits()
            }
            fn neg(self) -> Self {
                -self
            }
            fn add_r(self, other: Self, rm: Round) -> Option<StatusAnd<Self>> {
                match rm {
                    Round::NearestTiesToEven => Some(Status::OK.and(self + other)),
                    _ => None,
                }
            }
            fn sub_r(self, other: Self, rm: Round) -> Option<StatusAnd<Self>> {
                match rm {
                    Round::NearestTiesToEven => Some(Status::OK.and(self - other)),
                    _ => None,
                }
            }
            fn mul_r(self, other: Self, rm: Round) -> Option<StatusAnd<Self>> {
                match rm {
                    Round::NearestTiesToEven => Some(Status::OK.and(self * other)),
                    _ => None,
                }
            }
            fn div_r(self, other: Self, rm: Round) -> Option<StatusAnd<Self>> {
                match rm {
                    Round::NearestTiesToEven => Some(Status::OK.and(self / other)),
                    _ => None,
                }
            }
            fn rem(self, other: Self) -> Self {
                self % other
            }
            fn mul_add_r(self, mul: Self, add: Self, rm: Round) -> Option<StatusAnd<Self>> {
                match rm {
                    Round::NearestTiesToEven => Some(Status::OK.and(self.mul_add(mul, add))),
                    _ => None,
                }
            }

            /* float->int casts are toward zero */
            fn to_i128_r(self, rm: Round) -> Option<StatusAnd<i128>> {
                match rm {
                    Round::TowardZero => Some(Status::OK.and(self as i128)),
                    _ => None,
                }
            }
            fn to_u128_r(self, rm: Round) -> Option<StatusAnd<u128>> {
                match rm {
                    Round::TowardZero => Some(Status::OK.and(self as u128)),
                    _ => None,
                }
            }

            fn from_i128_r(x: i128, rm: Round) -> Option<StatusAnd<Self>> {
                match rm {
                    Round::NearestTiesToEven => Some(Status::OK.and(x as Self)),
                    _ => None,
                }
            }
            fn from_u128_r(x: u128, rm: Round) -> Option<StatusAnd<Self>> {
                match rm {
                    Round::NearestTiesToEven => Some(Status::OK.and(x as Self)),
                    _ => None,
                }
            }
            fn to_double_r(self, rm: Round) -> Option<StatusAnd<f64>> {
                match rm {
                    Round::NearestTiesToEven => Some(Status::OK.and(self as f64)),
                    _ => None,
                }
            }
            fn from_double_r(x: f64, rm: Round) -> Option<StatusAnd<Self>> {
                match rm {
                    Round::NearestTiesToEven => Some(Status::OK.and(x as Self)),
                    _ => None,
                }
            }
            fn to_single_r(self, rm: Round) -> Option<StatusAnd<f32>> {
                match rm {
                    Round::NearestTiesToEven => Some(Status::OK.and(self as f32)),
                    _ => None,
                }
            }
            fn from_single_r(x: f32, rm: Round) -> Option<StatusAnd<Self>> {
                match rm {
                    Round::NearestTiesToEven => Some(Status::OK.and(x as Self)),
                    _ => None,
                }
            }
        }
    };
}

#[cfg(target_has_reliable_f16)]
impl_host_float!(f16, u16);
impl_host_float!(f32, u32);
impl_host_float!(f64, u64);
#[cfg(target_has_reliable_f128)]
impl_host_float!(f128, u128);

struct FloatPrintHelper<F: FloatRepr>(F);
impl<F: FloatRepr> fmt::Debug for FloatPrintHelper<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{cy}0x{hex:0hex_width$x}{rst} {gry}/* {repr} */{rst}",
            hex = self.0.to_bits_u128(),
            repr = self.0,
            hex_width = F::BYTE_LEN * 2,
            rst = term().rst,
            cy = term().cy,
            gry = term().gry
        )
    }
}

/// Decode a rounding mode.
fn round_from_u8(tag: u8) -> Option<Round> {
    let v = match tag {
        x if x == round_to_u8(Round::NearestTiesToEven) => Round::NearestTiesToEven,
        x if x == round_to_u8(Round::TowardZero) => Round::TowardZero,
        x if x == round_to_u8(Round::TowardPositive) => Round::TowardPositive,
        x if x == round_to_u8(Round::TowardNegative) => Round::TowardNegative,
        x if x == round_to_u8(Round::NearestTiesToAway) => Round::NearestTiesToAway,
        _ => return None,
    };
    Some(v)
}

/// Encode a rounding mode.
fn round_to_u8(rm: Round) -> u8 {
    match rm {
        Round::NearestTiesToEven => 0,
        Round::TowardZero => 1,
        Round::TowardPositive => 2,
        Round::TowardNegative => 3,
        Round::NearestTiesToAway => 4,
    }
}

/// Masks for bitwise operations.
#[derive(Clone, Copy, Debug)]
struct Masks {
    sign_bit_mask: u128,
    exp_mask: u128,
    sig_mask: u128,
    qnan_bit_mask: u128,
}

impl Masks {
    fn for_float<F: FloatRepr>() -> Self {
        // HACK(eddyb) to avoid putting this behind a `HasHardFloat` bound,
        // we hardcode some aspects of the IEEE binary float representation,
        // relying on `rustc_apfloat`-provided constants as a source of truth.
        let sign_bit_mask = 1 << (F::BIT_WIDTH - 1);
        let exp_mask = F::RustcApFloat::INFINITY.to_bits();
        let sig_mask = (1 << exp_mask.trailing_zeros()) - 1;
        let qnan_bit_mask = (sig_mask + 1) >> 1;

        Self {
            sign_bit_mask,
            exp_mask,
            sig_mask,
            qnan_bit_mask,
        }
    }
}

/// Helper for printing with color.
#[allow(dead_code)]
#[derive(Clone, Copy, Default)]
struct Colors {
    bold: &'static str,
    dim: &'static str,
    gry: &'static str,
    red_b: &'static str,
    grn: &'static str,
    yel: &'static str,
    yel_b: &'static str,
    cy: &'static str,
    rst: &'static str,
}

thread_local! {
    static COLORS: Colors = {
        if io::stdout().is_terminal() {
            Colors {
                bold: "\x1b[1m",
                dim: "\x1b[2m",
                gry: "\x1b[90m",
                red_b: "\x1b[1m\x1b[91m",
                grn: "\x1b[92m",
                yel: "\x1b[93m",
                yel_b: "\x1b[1m\x1b[93m",
                cy: "\x1b[96m",
                rst: "\x1b[m",
            }
        } else {
            Colors::default()
        }
    }
}

fn term() -> Colors {
    COLORS.with(|x| *x)
}

/// Interop for items that cross the FFI bounary.
mod cxx {
    use super::*;
    use std::ffi::{CStr, c_char};

    macro_rules! make_extern {
        ($F:ty, $name:ident) => {
            pub safe fn $name(
                opcode: u8,
                round: u8,
                ai: <$F as FloatRepr>::Repr,
                bi: <$F as FloatRepr>::Repr,
                ci: <$F as FloatRepr>::Repr,
                out: &mut <$F as FloatRepr>::Repr,
            ) -> i32;
        };
    }

    // SAFETY: matches definition in `ap_fuzz.cpp`
    unsafe extern "C" {
        fn check_error() -> *const c_char;

        make_extern!(BrainF16, cxx_apf_eval_op_brainf16);
        make_extern!(Ieee16, cxx_apf_eval_op_ieee16);
        make_extern!(Ieee32, cxx_apf_eval_op_ieee32);
        make_extern!(Ieee64, cxx_apf_eval_op_ieee64);
        make_extern!(Ieee128, cxx_apf_eval_op_ieee128);
        // Not defined here
        // make_extern!(PPCDoubleDouble, cxx_apf_eval_op_ppcdoubledouble);
        make_extern!(F8E5M2, cxx_apf_eval_op_f8e5m2);
        make_extern!(F8E4M3FN, cxx_apf_eval_op_f8e4m3fn);
        make_extern!(X87_F80, cxx_apf_eval_op_x87_f80);
    }

    /// Return the caught exception's error if present.
    fn error() -> Option<String> {
        unsafe {
            let p = check_error();
            if p.is_null() {
                return None;
            }

            let s = CStr::from_ptr(p);
            Some(s.to_string_lossy().into_owned())
        }
    }

    /// Turn a status integer into our `Status`.
    pub fn decode_status(mut status: i32) -> Status {
        if status < 0 {
            panic!("error from c++: {}", error().unwrap())
        }

        let mut res = Status::OK;

        let invalid = 0x01;
        let divby0 = 0x02;
        let oflow = 0x04;
        let uflow = 0x08;
        let inexact = 0x10;

        if status & invalid != 0 {
            res |= Status::INVALID_OP;
            status &= !invalid;
        }
        if status & divby0 != 0 {
            res |= Status::DIV_BY_ZERO;
            status &= !divby0;
        }
        if status & oflow != 0 {
            res |= Status::OVERFLOW;
            status &= !oflow;
        }
        if status & uflow != 0 {
            res |= Status::UNDERFLOW;
            status &= !uflow;
        }
        if status & inexact != 0 {
            res |= Status::INEXACT;
            status &= !inexact;
        }
        assert_eq!(status, 0, "uncleared status flags: {status:#010x}");
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_masks() {
        // Ensure our masks are correct
        mask_assertions::<Ieee16>();
        mask_assertions::<Ieee32>();
        mask_assertions::<Ieee64>();
        mask_assertions::<Ieee128>();
        mask_assertions::<F8E5M2>();
        mask_assertions::<F8E4M3FN>();
        mask_assertions::<BrainF16>();
        mask_assertions::<X87_F80>();
    }

    fn mask_assertions<F: FloatRepr>() {
        let masks = Masks::for_float::<F>();
        println!("{} masks: {masks:#?}", F::NAME);
        let Masks {
            sign_bit_mask,
            exp_mask,
            sig_mask,
            qnan_bit_mask,
        } = masks;

        // Sanity Checks
        assert_eq!(
            sign_bit_mask | exp_mask | sig_mask,
            !0 >> (128 - F::BIT_WIDTH)
        );
        assert!((sig_mask + 1).is_power_of_two());
        assert!(((exp_mask | sig_mask) + 1).is_power_of_two());
        assert_eq!(
            sign_bit_mask.count_ones() + exp_mask.count_ones() + sig_mask.count_ones(),
            F::BIT_WIDTH as u32
        );
        if F::KIND == FpKind::F8E4M3FN {
            // No infinity or NaN for this version
            assert_eq!(qnan_bit_mask, 0);
        } else {
            assert!(qnan_bit_mask.is_power_of_two());
        }
        assert_eq!(exp_mask | qnan_bit_mask, F::RustcApFloat::NAN.to_bits());
    }
}
