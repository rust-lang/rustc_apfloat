use std::fmt::{self, Write as _};
use std::fs;
use std::io::Write;
use std::path::Path;

use num_traits::ToPrimitive;
use rustc_apfloat::{Float, Round};

use crate::{Args, Error, FloatRepr, Op, decode_eval_check, for_each_repr, round_to_u8};

const ROUND_ALL: &[Round] = &[
    Round::NearestTiesToEven,
    Round::TowardPositive,
    Round::TowardNegative,
    Round::TowardZero,
    Round::NearestTiesToAway,
];

/// Create baseline inputs for the fuzzer to start with. It will start applying mutations to these
/// inputs.
///
/// This creates the cartesian product of the following:
///
/// * All float types
/// * All operations
/// * All rounding modes
/// * A small list of possible inputs, applied to each argument.
///
/// This creates a _lot_ of files, so running `cmin` after helps give the fuzzer less input to
/// work with.
pub fn generate(dir: &Path) {
    fs::create_dir_all(&dir).unwrap();
    let mut total = 0;

    for_each_repr!(for F in all_reprs!() {
        let count = gen_for_f::<F>(dir);
        total += count;
    });

    eprintln!("wrote {total} total files to `{}`", dir.display());
    eprintln!("note that it is recommended to run `cargo afl cmin` (`just gen` handles this)");
}

fn gen_for_f<F: FloatRepr>(dir: &Path) -> u64 {
    let mut buf = Vec::new();
    let mut name = String::new();

    // There is no `ONE` constant, so this works.
    let one = (F::RustcApFloat::SMALLEST / F::RustcApFloat::SMALLEST).value;
    let inputs = [
        F::RustcApFloat::ZERO,
        F::RustcApFloat::INFINITY,
        -F::RustcApFloat::ZERO,
        -F::RustcApFloat::INFINITY,
        F::RustcApFloat::qnan(None),
        F::RustcApFloat::snan(None),
        F::RustcApFloat::largest(),
        F::RustcApFloat::SMALLEST,
        F::RustcApFloat::smallest_normalized(),
        one,
    ];
    let mut count = 0;
    let flt_name = F::short_lowercase_name();

    // We don't need to test anything here, just use `cli_args` for config.
    let mut cli_args = Args::default();
    cli_args.ignore_cxx = true;
    cli_args.ignore_hard = true;

    for op in Op::ALL.iter().copied() {
        for rm in ROUND_ALL.iter().copied() {
            let mut write_one = |a: F, b: F, c: F, input_desc: fmt::Arguments| {
                buf.clear();
                name.clear();

                write!(name, "{flt_name}-{op:?}-{rm:?}-{input_desc}").unwrap();

                buf.push(F::KIND.to_u8().unwrap());
                buf.push(op.to_u8().unwrap());
                buf.push(round_to_u8(rm));

                for arg in [a, b, c].iter().take(op.airity() as usize) {
                    arg.write_as_le_bytes_into(&mut buf);
                }

                // Verify that the created input parses correctly. We don't need to do the
                // evaluation check, running the fuzzer will handle that.
                match decode_eval_check(&buf, &cli_args, false) {
                    Ok(()) | Err(Error::Check(_)) => (),
                    Err(Error::Decode(e)) => panic!("error decoding: {e}"),
                }

                let mut f = fs::OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(dir.join(&name))
                    .unwrap();
                f.write_all(&mut buf).unwrap();
                count += 1;
            };

            let airity = op.airity() as u8;
            for (ai, a) in inputs.iter().enumerate() {
                if airity == 1 {
                    write_one(
                        F::from_ap(*a),
                        F::from_bits_u128(0),
                        F::from_bits_u128(0),
                        format_args!("{ai}"),
                    )
                } else {
                    for (bi, b) in inputs.iter().enumerate() {
                        if airity == 2 {
                            write_one(
                                F::from_ap(*a),
                                F::from_ap(*b),
                                F::from_bits_u128(0),
                                format_args!("{ai}-{bi}"),
                            )
                        } else {
                            assert_eq!(airity, 3);
                            for (ci, c) in inputs.iter().enumerate() {
                                write_one(
                                    F::from_ap(*a),
                                    F::from_ap(*b),
                                    F::from_ap(*c),
                                    format_args!("{ai}-{bi}-{ci}"),
                                )
                            }
                        }
                    }
                }
            }
        }
    }

    eprintln!("{flt_name}: wrote {count} files");
    count
}
