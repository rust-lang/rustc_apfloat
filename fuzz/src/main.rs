use clap::{CommandFactory, Parser, Subcommand};
use rustc_apfloat::Float as _;
use std::io::Write;
use std::mem::MaybeUninit;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::{fmt, mem};

// See `build.rs` and `ops.rs` for how `FuzzOp` is generated.
include!(concat!(env!("OUT_DIR"), "/generated_fuzz_ops.rs"));

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
trait FloatRepr: Copy + Default + Eq + fmt::Display {
    type RustcApFloat: rustc_apfloat::Float
        + rustc_apfloat::Float
        + rustc_apfloat::FloatConvert<rustc_apfloat::ieee::Single>
        + rustc_apfloat::FloatConvert<rustc_apfloat::ieee::Double>;

    const BIT_WIDTH: usize = Self::RustcApFloat::BITS;
    const BYTE_LEN: usize = (Self::BIT_WIDTH + 7) / 8;

    const NAME: &'static str;

    // HACK(eddyb) this has to be overwritable because we have more than one
    // format with the same `BIT_WIDTH`, so it's not unambiguous on its own.
    const REPR_TAG: u8 = Self::BIT_WIDTH as u8;

    fn short_lowercase_name() -> String {
        Self::NAME.to_ascii_lowercase().replace("ieee", "f")
    }

    // FIXME(eddyb) these should ideally be using `[u8; Self::BYTE_LEN]`.
    fn from_le_bytes(bytes: &[u8]) -> Self;
    fn write_as_le_bytes_into(self, out_bytes: &mut Vec<u8>);

    fn to_bits_u128(self) -> u128;
    fn from_bits_u128(bits: u128) -> Self;

    // HACK(eddyb) this avoids needing another trait (or an `enum` of all formats).
    fn cxx_apf_eval_fuzz_op(op: FuzzOp<Self>) -> Self;
    // HACK(eddyb) this avoids dealing with separate traits and other complications.
    fn hard_eval_fuzz_op_if_supported(op: FuzzOp<Self>) -> Option<Self>;
}

macro_rules! float_reprs {
    ($($name:ident($repr:ty) {
        type RustcApFloat = $rs_apf_ty:ty;
        $(const REPR_TAG = $repr_tag:expr;)?
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

        $(
            // HACK(eddyb) `packed` is here becasue `u128` alignment can differ
            // from `__uint128_t` alignment, on some platforms, and it's better
            // to just get rid of alignment sources entirely.
            #[repr(C, packed)]
            #[derive(Copy, Clone, Default, PartialEq, Eq)]
            struct $name($repr);

            impl FloatRepr for $name {
                type RustcApFloat = $rs_apf_ty;

                const NAME: &'static str = stringify!($name);

                $(const REPR_TAG: u8 = $repr_tag;)?

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

                fn cxx_apf_eval_fuzz_op(op: FuzzOp<Self>) -> Self {
                    extern "C" {
                        // HACK(eddyb) the warning is about `u128` ABI issues,
                        // which is also why indirection is used.
                        #[allow(improper_ctypes)]
                        fn $cxx_apf_eval_fuzz_op(out: &mut MaybeUninit<$name>, op: &FuzzOp<$name>);
                    }
                    unsafe {
                        let mut out = MaybeUninit::uninit();
                        $cxx_apf_eval_fuzz_op(&mut out, &op);
                        out.assume_init()
                    }
                }

                fn hard_eval_fuzz_op_if_supported(_op: FuzzOp<Self>) -> Option<Self> {
                    None $(.or(Some(
                        Self(_op.map(|Self(x)| <$hard_float_ty>::from_bits(x)).eval_hard().to_bits())
                    )))?
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
                <Self as FloatRepr>::RustcApFloat::from_bits(self.to_bits_u128()).fmt(f)
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

float_reprs! {
    Ieee16(u16) {
        type RustcApFloat = rustc_apfloat::ieee::Half;
        extern fn = cxx_apf_fuzz_eval_op_ieee16;
    }
    Ieee32(u32) {
        type RustcApFloat = rustc_apfloat::ieee::Single;
        extern fn = cxx_apf_fuzz_eval_op_ieee32;
        type HardFloat = f32;
    }
    Ieee64(u64) {
        type RustcApFloat = rustc_apfloat::ieee::Double;
        extern fn = cxx_apf_fuzz_eval_op_ieee64;
        type HardFloat = f64;
    }
    Ieee128(u128) {
        type RustcApFloat = rustc_apfloat::ieee::Quad;
        extern fn = cxx_apf_fuzz_eval_op_ieee128;
    }

    // Non-standard IEEE-like formats.
    F8E5M2(u8) {
        type RustcApFloat = rustc_apfloat::ieee::Float8E5M2;
        const REPR_TAG = 8 + 0;
        extern fn = cxx_apf_fuzz_eval_op_f8e5m2;
    }
    F8E5M2FNUZ(u8) {
        type RustcApFloat = rustc_apfloat::ieee::Float8E5M2FNUZ;
        const REPR_TAG = 8 + 2;
        extern fn = cxx_apf_fuzz_eval_op_f8e5m2fnuz;
    }
    F8E4M3FN(u8) {
        type RustcApFloat = rustc_apfloat::ieee::Float8E4M3FN;
        const REPR_TAG = 8 + 1;
        extern fn = cxx_apf_fuzz_eval_op_f8e4m3fn;
    }
    F8E4M3FNUZ(u8) {
        type RustcApFloat = rustc_apfloat::ieee::Float8E4M3FNUZ;
        const REPR_TAG = 8 + 3;
        extern fn = cxx_apf_fuzz_eval_op_f8e4m3fnuz;
    }
    F8E4M3B11FNUZ(u8) {
        type RustcApFloat = rustc_apfloat::ieee::Float8E4M3B11FNUZ;
        const REPR_TAG = 8 + 4;
        extern fn = cxx_apf_fuzz_eval_op_f8e4m3b11fnuz;
    }
    BrainF16(u16) {
        type RustcApFloat = rustc_apfloat::ieee::BFloat;
        const REPR_TAG = 16 + 1;
        extern fn = cxx_apf_fuzz_eval_op_brainf16;
    }
    X87_F80(u128) {
        type RustcApFloat = rustc_apfloat::ieee::X87DoubleExtended;
        extern fn = cxx_apf_fuzz_eval_op_x87_f80;
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
    rustc_apfloat::ieee::Single: rustc_apfloat::FloatConvert<F::RustcApFloat>,
    rustc_apfloat::ieee::Double: rustc_apfloat::FloatConvert<F::RustcApFloat>,
{
    fn try_decode(data: &[u8]) -> Result<Self, ()> {
        let (&tag, inputs) = data.split_first().ok_or(())?;
        if inputs.len() % F::BYTE_LEN != 0 {
            return Err(());
        }

        let mut inputs = inputs.chunks(F::BYTE_LEN).map(F::from_le_bytes);
        let mut too_few_inputs = false;
        let op = FuzzOp::from_tag(tag).ok_or(())?.map(|()| {
            inputs.next().unwrap_or_else(|| {
                too_few_inputs = true;
                F::default()
            })
        });
        if too_few_inputs || inputs.next().is_some() {
            return Err(());
        }
        Ok(op)
    }

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
        fn if_terminal<T: Default>(x: T) -> T {
            use std::io::IsTerminal;
            thread_local! {
                static STDOUT_IS_TERMINAL: bool = std::io::stdout().is_terminal();
            }
            if STDOUT_IS_TERMINAL.with(|&t| t) {
                x
            } else {
                T::default()
            }
        }

        struct FloatPrintHelper<F: FloatRepr>(F);
        impl<F: FloatRepr> fmt::Debug for FloatPrintHelper<F> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    f,
                    "{c_yellow}0x{:0hex_width$x}{c_normal} {c_grey}/* {} */{c_normal}",
                    self.0.to_bits_u128(),
                    self.0,
                    hex_width = F::BYTE_LEN * 2,
                    c_normal = if_terminal("\x1b[m"),
                    c_yellow = if_terminal("\x1b[93m"),
                    c_grey = if_terminal("\x1b[90m"),
                )
            }
        }

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
                print!(
                    " <- {c_bold_red}!!! MISMATCH !!!{c_normal}",
                    c_normal = if_terminal("\x1b[m"),
                    c_bold_red = if_terminal("\x1b[1m\x1b[91m"),
                )
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
        let Some(Commands::Bruteforce {
            min_width,
            max_width,
            verbose,
            only_non_trivial_fma,
        }) = cli_args.command
        else {
            unreachable!("bruteforce({cli_args:?}): subcommand not `Commands::Bruteforce`");
        };

        if !(min_width..=max_width).contains(&F::BIT_WIDTH) {
            return Ok(());
        }

        // HACK(eddyb) there is a good chance C++ will also fail, so avoid the
        // (more fatal) C++ assertion failure, via `print_op_and_eval_outputs`.
        let cli_args_plus_ignore_cxx = Args {
            ignore_cxx: true,
            ..cli_args.clone()
        };

        let all_ops = (0..)
            .map(FuzzOp::from_tag)
            .take_while(|op| op.is_some())
            .map(|op| op.unwrap())
            .filter(move |op| {
                if only_non_trivial_fma {
                    matches!(op, FuzzOp::MulAdd(..))
                } else {
                    true
                }
            });

        let op_to_combined_input_bits_range = move |op: FuzzOp<()>| {
            let mut total_bit_width = 0;
            op.map(|()| total_bit_width += F::BIT_WIDTH);

            // HACK(eddyb) the highest `F::BIT_WIDTH` bits are the last input,
            // i.e. the addend for FMA (see also `Commands::Bruteforce` docs).
            let start_combined_input_bits = if only_non_trivial_fma {
                1 << (total_bit_width - F::BIT_WIDTH)
            } else {
                0
            };

            start_combined_input_bits..u128::checked_shl(1, total_bit_width as u32).unwrap()
        };
        let op_to_exhaustive_cases = move |op: FuzzOp<()>| {
            op_to_combined_input_bits_range(op).map(move |i| -> Self {
                let mut combined_input_bits = i;
                let op_with_inputs = op.map(|()| {
                    let x = combined_input_bits & ((1 << F::BIT_WIDTH) - 1);
                    combined_input_bits >>= F::BIT_WIDTH;
                    F::from_bits_u128(x)
                });
                assert_eq!(combined_input_bits, 0);
                op_with_inputs
            })
        };

        let num_total_cases = all_ops
            .clone()
            .map(|op| {
                let range = op_to_combined_input_bits_range(op);
                range.end.checked_sub(range.start).unwrap()
            })
            .try_fold(0, u128::checked_add)
            .unwrap();

        let float_name = F::short_lowercase_name();
        println!("Exhaustively checking {num_total_cases} cases for {float_name}:");

        // HACK(eddyb) show some indication of progress at least every few seconds,
        // but also don't show verbose progress as often, with fewer testcases.
        let num_dots = usize::try_from(num_total_cases >> 23)
            .unwrap_or(usize::MAX)
            .max(if verbose { 10 } else { 40 });
        let cases_per_dot =
            usize::try_from(num_total_cases / u128::try_from(num_dots).unwrap()).unwrap();

        // Spawn worker threads and only report back from them once in a while
        // (in large batches of successes), or in case of any failure.
        let num_threads = std::thread::available_parallelism().unwrap();
        let successes_batch_size = (cases_per_dot / num_threads).next_power_of_two();

        struct Update<T> {
            successes: usize,
            mismatch_or_panic: Option<(T, Option<Box<dyn std::any::Any + Send>>)>,
        }
        impl<T> Default for Update<T> {
            fn default() -> Self {
                Update {
                    successes: 0,
                    mismatch_or_panic: None,
                }
            }
        }
        let (updates_tx, updates_rx) = std::sync::mpsc::channel();

        // HACK(eddyb) avoid reporting panics while iterating.
        std::panic::set_hook(Box::new(|_| {}));

        let worker_threads: Vec<_> = (0..num_threads.get())
            .map(|thread_idx| {
                let cli_args = cli_args.clone();
                let updates_tx = updates_tx.clone();
                let cases_per_thread = all_ops
                    .clone()
                    .flat_map(op_to_exhaustive_cases)
                    .skip(thread_idx)
                    .step_by(num_threads.get());
                std::thread::spawn(move || {
                    let mut update = Update::default();
                    for op_with_inputs in cases_per_thread {
                        // HACK(eddyb) there are still panics we need to account for,
                        // e.g. https://github.com/llvm/llvm-project/issues/63895, and
                        // even if the Rust code didn't panic, LLVM asserts would trip.
                        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            op_with_inputs.eval(&cli_args)
                        })) {
                            Ok(out) => {
                                if out.all_match() {
                                    update.successes += 1;
                                } else {
                                    update.mismatch_or_panic = Some((op_with_inputs, None));
                                }
                            }
                            Err(panic) => {
                                update.mismatch_or_panic = Some((op_with_inputs, Some(panic)));
                            }
                        }

                        if update.successes >= successes_batch_size
                            || update.mismatch_or_panic.is_some()
                        {
                            updates_tx.send(mem::take(&mut update)).unwrap();
                        }
                    }
                    updates_tx.send(update).unwrap();
                })
            })
            .collect();

        // HACK(eddyb) ensure that `Sender`s are only tied to active threads,
        // allowing the `for` loop below to exit, once all worker threads finish.
        drop(updates_tx);

        let mut case_idx = 0;
        let mut current_dot_first_case_idx = 0;
        let mut last_mismatch_case_idx = None;
        let mut last_panic_case_idx = None;
        let mut all_mismatches = vec![];
        let mut all_panics = vec![];
        let mut verbose_failed_to_show_some_panics = false;
        for update in updates_rx {
            let Update {
                successes,
                mismatch_or_panic,
            } = update;
            let successes_and_failures = [
                Some(successes).filter(|&n| n > 0).map(Ok),
                mismatch_or_panic.map(Err),
            ]
            .into_iter()
            .flatten();

            for success_or_failure in successes_and_failures {
                match success_or_failure {
                    Ok(successes) => case_idx += successes,

                    Err((op_with_inputs, None)) => {
                        if verbose {
                            op_with_inputs.print_op_and_eval_outputs(cli_args);
                        }

                        last_mismatch_case_idx = Some(case_idx);
                        all_mismatches.push(op_with_inputs);

                        case_idx += 1;
                    }

                    Err((op_with_inputs, Some(panic))) => {
                        if verbose {
                            op_with_inputs.print_op_and_eval_outputs(&cli_args_plus_ignore_cxx);
                            if let Ok(msg) = panic.downcast::<String>() {
                                eprintln!("panicked with: {msg}");
                            } else {
                                verbose_failed_to_show_some_panics = true;
                            }
                        }

                        last_panic_case_idx = Some(case_idx);
                        all_panics.push(op_with_inputs);

                        case_idx += 1;
                    }
                }

                loop {
                    let next_dot_first_case_idx = current_dot_first_case_idx + cases_per_dot;
                    if case_idx < next_dot_first_case_idx {
                        break;
                    }
                    if verbose {
                        println!(
                            "  {:3.1}% done ({case_idx} / {num_total_cases}), \
                           found {} mismatches and {} panics",
                            (case_idx as f64) / (num_total_cases as f64) * 100.0,
                            all_mismatches.len(),
                            all_panics.len()
                        );
                    } else {
                        print!(
                            "{}",
                            if last_panic_case_idx.is_some_and(|i| i >= current_dot_first_case_idx)
                            {
                                '🕱'
                            } else if last_mismatch_case_idx
                                .is_some_and(|i| i >= current_dot_first_case_idx)
                            {
                                '≠'
                            } else {
                                '.'
                            }
                        );
                        // HACK(eddyb) get around `stdout` line buffering.
                        std::io::stdout().flush().unwrap();
                    }
                    current_dot_first_case_idx = next_dot_first_case_idx;
                }
            }
        }
        println!();

        // HACK(eddyb) undo what we did just before spawning worker threads.
        let _ = std::panic::take_hook();

        for worker_thread in worker_threads {
            worker_thread.join().unwrap();
        }

        // HACK(eddyb) keep only one mismatch per `FuzzOp` variant.
        // FIXME(eddyb) consider sorting these (and panics?) due to parallelism.
        let num_mismatches = all_mismatches.len();
        let mut select_mismatches = all_mismatches;
        select_mismatches.dedup_by_key(|op_with_inputs| op_with_inputs.tag());

        if num_mismatches > 0 {
            println!();
            println!(
                "⚠ found {num_mismatches} ({:.1}%) mismatches for {float_name}, showing {} of them:",
                (num_mismatches as f64) / (num_total_cases as f64) * 100.0,
                select_mismatches.len(),
            );
            for mismatch in select_mismatches {
                mismatch.print_op_and_eval_outputs(cli_args);
            }
        }

        if !all_panics.is_empty() {
            println!();
            println!(
                "⚠ found {} panics for {float_name}, {}",
                all_panics.len(),
                if verbose && !verbose_failed_to_show_some_panics {
                    "shown above"
                } else {
                    "showing them (without trying C++):"
                },
            );
            if !verbose || verbose_failed_to_show_some_panics {
                for &panicking_case in &all_panics {
                    panicking_case.print_op_and_eval_outputs(&cli_args_plus_ignore_cxx);
                }
            }
        }

        if num_mismatches == 0 && all_panics.is_empty() {
            println!("✔️ all {num_total_cases} cases match");
        }
        println!();

        NonZeroUsize::new(num_mismatches + all_panics.len()).map_or(Ok(()), Err)
    }
}

fn main() {
    let cli_args = Args::parse();

    if let Some(cmd) = &cli_args.command {
        match cmd {
            Commands::Decode { files } => {
                for file in files {
                    println!("{}:", file.display());
                    let data = std::fs::read(file).unwrap();

                    data.split_first()
                        .ok_or("empty file")
                        .and_then(|(&repr_tag, data)| {
                            dispatch_any_float_repr_by_repr_tag!(match repr_tag {
                                for<F: FloatRepr> => return Ok(
                                    FuzzOp::<F>::try_decode(data)
                                        .ok()
                                        .ok_or(std::any::type_name::<FuzzOp<F>>())?
                                        .print_op_and_eval_outputs(&cli_args)
                                )
                            });
                            Err("first byte not valid `FloatRepr::REPR_TAG`")
                        })
                        .unwrap_or_else(|e| println!("  invalid data ({e})"));
                }
            }
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

    #[cfg_attr(not(fuzzing), allow(unused))]
    let fuzz_one_op = |data: &[u8]| {
        data.split_first().and_then(|(&repr_tag, data)| {
            dispatch_any_float_repr_by_repr_tag!(match repr_tag {
                for<F: FloatRepr> => return Some(
                    assert!(FuzzOp::<F>::try_decode(data).ok()?.eval(&cli_args).all_match())
                )
            });
            None
        });
    };

    // HACK(eddyb) `#[cfg(fuzzing)] {...}` used instead of `if cfg!(fuzzing) {...}`
    // because the latter can still cause the `afl` crate to be linked, and it
    // depends on native libraries that are only available under `cargo afl ...`.
    #[cfg(fuzzing)]
    if true {
        // FIXME(eddyb) make the first argument (panic hook choice) a CLI toggle.
        afl::fuzz(true, fuzz_one_op);

        return;
    }

    // FIXME(eddyb) add better docs for all of this.
    Args::command().print_long_help().unwrap();
    eprintln!();
    eprintln!("To fuzz `rustc_apfloat`, you must use `cargo afl`:");
    eprintln!(" - `cargo install afl`");
    eprintln!(" - build with `cargo afl build -p rustc_apfloat-fuzz --release`");
    // FIXME(eddyb) add `seed` subcommand using `FuzzOp::encode_into`, and a set
    // of basic examples, e.g. every `FuzzOp` variant with `0.0` for all inputs
    // (and/or maybe testcases from known and/or fixed bugs, too).
    eprintln!(" - seed with `mkdir fuzz/in-foo && echo > fuzz/in-foo/empty`");
    eprintln!(" - run with `cargo afl fuzz -i fuzz/in-foo -o fuzz/out-foo target/release/rustc_apfloat-fuzz`");
    std::process::exit(1);
}
