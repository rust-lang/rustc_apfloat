//! Benchmarks for converting from/to (decimal) strings, the only operations
//! that (may) need to allocate, and also some of the few that aren't `O(1)`
//! (alongside e.g. div/mod, but even those likely have a better bound).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::fmt::{self, Write as _};

struct Sample {
    name: &'static str,
    decimal_str: &'static str,
}

impl fmt::Display for Sample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // HACK(eddyb) this is mostly to sort criterion's output correctly.
        write!(f, "[len={:02}] ", self.decimal_str.len())?;
        f.write_str(self.decimal_str)?;
        if !self.name.is_empty() {
            write!(f, " aka {}", self.name)?;
        }
        Ok(())
    }
}

impl Sample {
    const fn new(decimal_str: &'static str) -> Self {
        Self { name: "", decimal_str }
    }

    const fn named(self, name: &'static str) -> Self {
        Self { name, ..self }
    }
}

const DOUBLE_SAMPLES: &[Sample] = &[
    Sample::new("0.0"),
    Sample::new("1.0"),
    Sample::new("1234.56789"),
    Sample::new("3.14159265358979323846264338327950288").named("π"),
    Sample::new("0.693147180559945309417232121458176568").named("ln(2)"),
];

fn double_from_str(c: &mut Criterion) {
    let mut group = c.benchmark_group("Double::from_str");
    for sample in DOUBLE_SAMPLES {
        group.bench_with_input(BenchmarkId::from_parameter(sample), sample.decimal_str, |b, s| {
            b.iter(|| s.parse::<rustc_apfloat::ieee::Double>().unwrap());
        });
    }
    group.finish();
}

/// `fmt::Write` implementation that does not need to allocate at all,
/// but instead asserts that what's written matches a known string exactly.
struct CheckerFmtSink<'a> {
    remaining: &'a str,
}

impl fmt::Write for CheckerFmtSink<'_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.remaining = self.remaining.strip_prefix(s).ok_or(fmt::Error)?;
        Ok(())
    }
}

impl CheckerFmtSink<'_> {
    fn finish(self) -> fmt::Result {
        if self.remaining.is_empty() {
            Ok(())
        } else {
            Err(fmt::Error)
        }
    }
}

fn double_to_str(c: &mut Criterion) {
    let mut group = c.benchmark_group("Double::to_str");
    for sample in DOUBLE_SAMPLES {
        let value = sample.decimal_str.parse::<rustc_apfloat::ieee::Double>().unwrap();

        // `CheckerFmtSink` is used later to ensure the formatting doesn't get
        // optimized away, but without allocating - we can, however, allocate
        // the expected output here, ahead of time, and also sanity-check it
        // in a more convenient (and user-friendly) way, ensuring that benching
        // itself never panics (though not in a way the optimizer would know of).
        let value_to_string = &value.to_string();

        // NOTE(eddyb) we only check that we get back the same floating-point
        // `value`, without comparing `value_to_string` and `sample.decimal_str`,
        // because `rustc_apfloat` (correctly) considers "natural precision" can
        // be shorter than our samples, and also it always strips trailing `.0`
        // (outside of scientific notation) - while it is possible to approximate
        // "is this plausibly close enough", it's an irrelevant complication here.
        assert_eq!(value_to_string.parse::<rustc_apfloat::ieee::Double>().unwrap(), value);

        group.bench_with_input(
            BenchmarkId::from_parameter(sample),
            &(value, value_to_string),
            |b, &(value, sample_to_string)| {
                b.iter(|| {
                    let mut checker = CheckerFmtSink {
                        remaining: sample_to_string,
                    };
                    write!(checker, "{value}").unwrap();
                    checker.finish().unwrap();
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, double_from_str, double_to_str);
criterion_main!(benches);
