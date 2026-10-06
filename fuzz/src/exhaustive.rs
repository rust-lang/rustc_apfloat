use std::io::Write;
use std::mem;
use std::num::NonZero;
use std::num::NonZeroUsize;

use rustc_apfloat::FloatConvert;
use rustc_apfloat::Round;
use rustc_apfloat::ieee::Double;
use rustc_apfloat::ieee::Single;

use crate::Args;
use crate::Commands;
use crate::EvalCfg;
use crate::FloatRepr;
use crate::Op;
use crate::eval_all;
use crate::for_each_repr;

pub fn run_for_all_floats(cli_args: &Args) {
    let mut any_mismatches = false;
    for_each_repr!(for F in all_reprs!() {
        any_mismatches |= run_exhaustive::<F>(&cli_args).is_err();
    });

    if any_mismatches {
        std::process::exit(1);
    }
}

pub fn run_exhaustive<F: FloatRepr>(cli_args: &Args) -> Result<(), NonZero<usize>>
where
    F: Send + 'static,
    Single: FloatConvert<F::RustcApFloat>,
    Double: FloatConvert<F::RustcApFloat>,
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

    let all_ops = if only_non_trivial_fma {
        &[Op::MulAdd]
    } else {
        Op::ALL
    };

    // This currently only tests round to nearest.
    let make_cfg =
        |op: Op, cli_args: &Args| EvalCfg::new(F::KIND, op, Round::NearestTiesToEven, cli_args);

    let op_to_combined_input_bits_range = move |op: Op| {
        let total_bit_width = F::BIT_WIDTH * (op.airity() as usize);

        // HACK(eddyb) the highest `F::BIT_WIDTH` bits are the last input,
        // i.e. the addend for FMA (see also `Commands::Bruteforce` docs).
        let start_combined_input_bits = if only_non_trivial_fma {
            1 << (total_bit_width - F::BIT_WIDTH)
        } else {
            0
        };

        start_combined_input_bits..1_u128.strict_shl(total_bit_width as u32)
    };
    let op_to_exhaustive_cases = move |op: Op| {
        op_to_combined_input_bits_range(op).map(move |i| -> (F, F, F) {
            let mask = (1 << F::BIT_WIDTH) - 1;
            let a = (i >> (0 * F::BIT_WIDTH)) & mask;
            let b = (i >> (1 * F::BIT_WIDTH)) & mask;
            let c = (i >> (2 * F::BIT_WIDTH)) & mask;
            assert_eq!(i >> (3 * F::BIT_WIDTH), 0);
            (
                F::from_bits_u128(a),
                F::from_bits_u128(b),
                F::from_bits_u128(c),
            )
        })
    };

    let num_total_cases = all_ops
        .iter()
        .map(|op| {
            let range = op_to_combined_input_bits_range(*op);
            range.end.strict_sub(range.start)
        })
        .fold(0, u128::strict_add);

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
                .iter()
                .flat_map(move |op| op_to_exhaustive_cases(*op).map(|(a, b, c)| (*op, a, b, c)))
                .skip(thread_idx)
                .step_by(num_threads.get());

            std::thread::spawn(move || {
                let mut update = Update::default();

                for (op, a, b, c) in cases_per_thread {
                    let cfg = make_cfg(op, &cli_args);
                    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        eval_all(&cfg, a, b, c)
                    }));
                    match res {
                        Ok(out) => {
                            if out.check_all(&cfg, a, b, c, false).is_ok() {
                                update.successes += 1;
                            } else {
                                update.mismatch_or_panic = Some(((op, a, b, c), None));
                            }
                        }
                        Err(panic) => {
                            update.mismatch_or_panic = Some(((op, a, b, c), Some(panic)));
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
                        let (op, a, b, c) = op_with_inputs;
                        let cfg = make_cfg(op, &cli_args);
                        let _ = eval_all(&cfg, a, b, c).check_all(&cfg, a, b, c, true);
                    }

                    last_mismatch_case_idx = Some(case_idx);
                    all_mismatches.push(op_with_inputs);

                    case_idx += 1;
                }

                Err((op_with_inputs, Some(panic))) => {
                    if verbose {
                        let (op, a, b, c) = op_with_inputs;
                        let cfg = make_cfg(op, &cli_args_plus_ignore_cxx);
                        let _ = eval_all(&cfg, a, b, c).check_all(&cfg, a, b, c, true);

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
                        if last_panic_case_idx.is_some_and(|i| i >= current_dot_first_case_idx) {
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
    select_mismatches.dedup_by_key(|(op, _, _, _)| *op);

    if num_mismatches > 0 {
        println!();
        println!(
            "⚠ found {num_mismatches} ({:.1}%) mismatches for {float_name}, showing {} of them:",
            (num_mismatches as f64) / (num_total_cases as f64) * 100.0,
            select_mismatches.len(),
        );
        for mismatch in select_mismatches {
            let (op, a, b, c) = mismatch;
            let cfg = make_cfg(op, &cli_args);
            let _ = eval_all(&cfg, a, b, c).check_all(&cfg, a, b, c, true);
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
                let (op, a, b, c) = panicking_case;
                let cfg = make_cfg(op, &cli_args_plus_ignore_cxx);
                let _ = eval_all(&cfg, a, b, c).check_all(&cfg, a, b, c, true);
            }
        }
    }

    if num_mismatches == 0 && all_panics.is_empty() {
        println!("✔️ all {num_total_cases} cases match");
    }
    println!();

    NonZeroUsize::new(num_mismatches + all_panics.len()).map_or(Ok(()), Err)
}
