#![feature(f16)]
#![feature(f128)]
#![allow(internal_features)] // for the below config
#![feature(cfg_target_has_reliable_f16_f128)]

mod exhaustive;
mod host;

use io::IsTerminal;
use io::Read;
use std::io;
use std::path::PathBuf;
use std::{fmt, fs};

use clap::{CommandFactory, Parser, Subcommand};
use num_derive::{FromPrimitive, ToPrimitive};
use num_traits::{FromPrimitive, ToPrimitive};
use rustc_apfloat::ieee::{Double, Single};
use rustc_apfloat::{Float, FloatConvert, Round, Status, StatusAnd, ieee};

use crate::host::HostFloat;

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

fn main() {
    let cli_args = Args::parse();

    // Check and panic as the fuzzer needs to do.
    let fuzz_check =
        |buf: &[u8], always_print| match decode_eval_check(&buf, &cli_args, always_print) {
            Ok(()) => (),
            // Discard decoding errors; we don't want the fuzzer to think this is a failure)
            Err(Error::Decode(e)) => println!("decode error: {e} (no panic raised)"),
            Err(Error::Check(e)) => panic!("check error: {e}"),
        };

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
                fuzz_check(&buf, true);
            }
            Commands::Decode { files } => run_decode_subcmd(files, &cli_args),
            Commands::Bruteforce { .. } => exhaustive::run_for_all_floats(&cli_args),
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
            fuzz_check(&buf, false);
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

    const KIND: FpKind;

    fn short_lowercase_name() -> String {
        Self::NAME.to_ascii_lowercase().replace("ieee", "f")
    }

    fn to_ap(self) -> Self::RustcApFloat;
    fn from_ap(x: Self::RustcApFloat) -> Self;

    // FIXME(const) `[u8; Self::BYTE_LEN]` would be better but requires MGCA.
    fn from_le_bytes(bytes: &[u8]) -> Self;

    fn to_bits_u128(self) -> u128;
    fn from_bits_u128(bits: u128) -> Self;

    fn cxx_apf_eval_fuzz_op(op: Op, rm: Round, a: Self, b: Self, c: Self) -> StatusAnd<Self>;
    fn host_eval_fuzz_op_if_supported(
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
        extern fn = $cxx_apf_eval_fuzz_op:ident;
        $(type HardFloat = $hard_float_ty:ty;)?
    })+) => {
        macro_rules! for_each_repr {
            (for $ty_var:ident in all_floats!() $block:block) => {
                $({
                   type $ty_var = $crate::$name;
                   $block
                })+
            };
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

                fn to_bits_u128(self) -> u128 {
                    self.0.into()
                }
                fn from_bits_u128(bits: u128) -> Self {
                    Self(bits.try_into().unwrap())
                }

                fn cxx_apf_eval_fuzz_op(
                    op: Op, rm: Round, a: Self, b: Self, c: Self
                ) -> StatusAnd<Self>
                {

                    let rm = round_to_u8(rm);
                    let mut out = 0;
                    let status = cxx::$cxx_apf_eval_fuzz_op(
                        op.to_u8().unwrap(), rm, a.0, b.0, c.0, &mut out,
                    );
                    let status = cxx::decode_status(status);

                    status.and(Self(out))
                }

                #[allow(unused_variables)]
                fn host_eval_fuzz_op_if_supported(
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
        extern fn = cxx_apf_eval_op_ieee16;
    }
    Ieee32(u32) {
        type RustcApFloat = rustc_apfloat::ieee::Single;
        extern fn = cxx_apf_eval_op_ieee32;
        type HardFloat = f32;
    }
    Ieee64(u64) {
        type RustcApFloat = rustc_apfloat::ieee::Double;
        extern fn = cxx_apf_eval_op_ieee64;
        type HardFloat = f64;
    }
    Ieee128(u128) {
        type RustcApFloat = rustc_apfloat::ieee::Quad;
        extern fn = cxx_apf_eval_op_ieee128;
    }

    // Non-standard IEEE-like formats.
    F8E5M2(u8) {
        type RustcApFloat = rustc_apfloat::ieee::Float8E5M2;
        extern fn = cxx_apf_eval_op_f8e5m2;
    }
    F8E4M3FN(u8) {
        type RustcApFloat = rustc_apfloat::ieee::Float8E4M3FN;
        extern fn = cxx_apf_eval_op_f8e4m3fn;
    }
    BrainF16(u16) {
        type RustcApFloat = rustc_apfloat::ieee::BFloat;
        extern fn = cxx_apf_eval_op_brainf16;
    }
    X87_F80(u128) {
        type RustcApFloat = rustc_apfloat::ieee::X87DoubleExtended;
        extern fn = cxx_apf_eval_op_x87_f80;
    }
}

pub(crate) use for_each_repr;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, FromPrimitive)]
pub enum FpKind {
    // The tag is based on the bit count. These are specified so corpus inputs are stable.
    Ieee16 = 16,
    Ieee32 = 32,
    Ieee64 = 64,
    Ieee128 = 128,
    F8E5M2 = 8,
    F8E4M3FN = 8 + 1,
    BrainF16 = 16 + 1,
    #[allow(non_camel_case_types)]
    X87_F80 = 80,
}

impl FpKind {
    #[cfg_attr(not(test), expect(unused))]
    const ALL: &[Self] = &[
        Self::Ieee16,
        Self::Ieee32,
        Self::Ieee64,
        Self::Ieee128,
        Self::F8E5M2,
        Self::F8E4M3FN,
        Self::BrainF16,
        Self::X87_F80,
    ];

    pub fn to_u8(self) -> u8 {
        self as u8
    }
}

/// A testable operation, which can be encoded as a byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, FromPrimitive, ToPrimitive)]
pub enum Op {
    Neg = 0,
    Add = 1,
    Sub = 2,
    Mul = 3,
    Div = 4,
    Rem = 5,
    MulAdd = 6,
    FToI128ToF = 7,
    FToU128ToF = 8,
    FToSingleToF = 9,
    FToDoubleToF = 10,
}

impl Op {
    pub const ALL: &[Self] = &[
        Self::Neg,
        Self::Add,
        Self::Sub,
        Self::Mul,
        Self::Div,
        Self::Rem,
        Self::MulAdd,
        Self::FToI128ToF,
        Self::FToU128ToF,
        Self::FToSingleToF,
        Self::FToDoubleToF,
    ];

    pub fn airity(self) -> Arity {
        match self {
            Op::Neg => Arity::Unary,
            Op::Add => Arity::Binary,
            Op::Sub => Arity::Binary,
            Op::Mul => Arity::Binary,
            Op::Div => Arity::Binary,
            Op::Rem => Arity::Binary,
            Op::MulAdd => Arity::Ternary,
            Op::FToI128ToF => Arity::Unary,
            Op::FToU128ToF => Arity::Unary,
            Op::FToSingleToF => Arity::Unary,
            Op::FToDoubleToF => Arity::Unary,
        }
    }
}

/// Number of inputs to an operation.
#[derive(Copy, Clone, Debug)]
pub enum Arity {
    Unary = 1,
    Binary = 2,
    Ternary = 3,
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

/// Context for when a check fails.
#[derive(Clone, Debug)]
struct CheckError(Box<(EvalCfg, ResultSummary)>);

impl fmt::Display for CheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cfg {:?} results {:?}", self.0.0, self.0.1)
    }
}

#[derive(Clone, Debug)]
enum Error {
    Decode(DecodeError),
    Check(CheckError),
}

impl From<DecodeError> for Error {
    fn from(value: DecodeError) -> Self {
        Self::Decode(value)
    }
}

impl From<CheckError> for Error {
    fn from(value: CheckError) -> Self {
        Self::Check(value)
    }
}

/// Configuration for the current operation.
#[derive(Debug, Clone)]
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
            if op == Op::MulAdd {
                // Don't even run for FMA which crashes in LLVM
                ret.run_cxx = false;
            }
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

/// Decode and evaluate all passed files without exiting on mismatches.
fn run_decode_subcmd(files: &[PathBuf], cli_args: &Args) {
    let mut buf = Vec::new();
    for path in files {
        println!("{}{}:{}", term().dim, path.display(), term().rst);

        buf.clear();
        let mut f = fs::File::open(path).unwrap();
        f.read_to_end(&mut buf).unwrap();

        match decode_eval_check(&buf, cli_args, true) {
            Ok(()) => (),
            Err(Error::Decode(e)) => println!("error decoding file: {e}"),
            Err(Error::Check(e)) => println!("check error: {e:?}"),
        }
    }
}

/// Main runner: decode a config, inputs based on that config, and then evaluate the results
/// for Rust, LLVM APFloat, and the host.
fn decode_eval_check(data: &[u8], cli_args: &Args, always_print: bool) -> Result<(), Error> {
    let (cfg, data) = EvalCfg::decode(data, cli_args)?;
    match cfg.kind {
        FpKind::Ieee16 => {
            let (a, b, c, r) = decode_for_ty_eval::<Ieee16>(&cfg, data)?;
            r.check_all(&cfg, a, b, c, always_print)?;
        }
        FpKind::Ieee32 => {
            let (a, b, c, r) = decode_for_ty_eval::<Ieee32>(&cfg, data)?;
            r.check_all(&cfg, a, b, c, always_print)?;
        }
        FpKind::Ieee64 => {
            let (a, b, c, r) = decode_for_ty_eval::<Ieee64>(&cfg, data)?;
            r.check_all(&cfg, a, b, c, always_print)?;
        }
        FpKind::Ieee128 => {
            let (a, b, c, r) = decode_for_ty_eval::<Ieee128>(&cfg, data)?;
            r.check_all(&cfg, a, b, c, always_print)?;
        }
        FpKind::F8E5M2 => {
            let (a, b, c, r) = decode_for_ty_eval::<F8E5M2>(&cfg, data)?;
            r.check_all(&cfg, a, b, c, always_print)?;
        }
        FpKind::F8E4M3FN => {
            let (a, b, c, r) = decode_for_ty_eval::<F8E4M3FN>(&cfg, data)?;
            r.check_all(&cfg, a, b, c, always_print)?;
        }
        FpKind::BrainF16 => {
            let (a, b, c, r) = decode_for_ty_eval::<BrainF16>(&cfg, data)?;
            r.check_all(&cfg, a, b, c, always_print)?;
        }
        FpKind::X87_F80 => {
            let (a, b, c, r) = decode_for_ty_eval::<X87_F80>(&cfg, data)?;
            r.check_all(&cfg, a, b, c, always_print)?;
        }
    }

    Ok(())
}

/// Decode operands for a given type and operation, then evaluate.
fn decode_for_ty_eval<F: FloatRepr>(
    cfg: &EvalCfg,
    data: &[u8],
) -> Result<(F, F, F, FuzzOpEvalOutputs<F>), DecodeError>
where
    Single: FloatConvert<<F as FloatRepr>::RustcApFloat>,
    Double: FloatConvert<<F as FloatRepr>::RustcApFloat>,
{
    let (a, b, c) = decode_operands::<F>(cfg.op, data)?;
    let r = eval_all(cfg, a, b, c);
    Ok((a, b, c, r))
}

#[must_use]
fn eval_all<F: FloatRepr>(cfg: &EvalCfg, a: F, b: F, c: F) -> FuzzOpEvalOutputs<F>
where
    Single: FloatConvert<<F as FloatRepr>::RustcApFloat>,
    Double: FloatConvert<<F as FloatRepr>::RustcApFloat>,
{
    // Evaluate the APFloat version as well as all possible references.
    FuzzOpEvalOutputs {
        rs_apf: eval_rust_ap(cfg.op, cfg.rm, a, b, c),
        cxx_apf: cfg
            .run_cxx
            .then(|| F::cxx_apf_eval_fuzz_op(cfg.op, cfg.rm, a, b, c)),
        host: cfg
            .run_host
            .then(|| F::host_eval_fuzz_op_if_supported(cfg.op, cfg.rm, a, b, c))
            .flatten(),
    }
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
struct FuzzOpEvalOutputs<F> {
    rs_apf: StatusAnd<F>,
    cxx_apf: Option<StatusAnd<F>>,
    host: Option<StatusAnd<F>>,
}

#[derive(Clone, Debug)]
struct ResultSummary {
    cxx_error: bool,
    cxx_ignore: Option<&'static str>,
    cxx_stat_error: bool,
    cxx_stat_ignore: Option<&'static str>,
    host_error: bool,
    host_ignore: Option<&'static str>,
    host_stat_error: bool,
    host_stat_ignore: Option<&'static str>,
}

impl<F: FloatRepr> FuzzOpEvalOutputs<F> {
    /// Validate that outputs are correct. Pass `always_print` for decoding use where we want to
    /// see the output regardless of whether or not it is a failure.
    fn check_all(
        &self,
        cfg: &EvalCfg,
        a: F,
        b: F,
        c: F,
        always_print: bool,
    ) -> Result<(), CheckError> {
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

        let mut res = ResultSummary {
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

        if failure {
            Err(CheckError(Box::new((cfg.clone(), res))))
        } else {
            Ok(())
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

    /* Check that `ALL` actually contains all variants. */

    #[test]
    fn op_all_list() {
        let mut computed = (0u8..=u8::MAX).filter_map(Op::from_u8).collect::<Vec<_>>();
        let mut listed = Op::ALL.to_vec();
        computed.sort_unstable();
        listed.sort_unstable();
        assert_eq!(computed, listed);
    }

    #[test]
    fn fpkind_all_list() {
        let mut computed = (0u8..=u8::MAX)
            .filter_map(FpKind::from_u8)
            .collect::<Vec<_>>();
        let mut listed = FpKind::ALL.to_vec();
        computed.sort_unstable();
        listed.sort_unstable();
        assert_eq!(computed, listed);
    }
}
