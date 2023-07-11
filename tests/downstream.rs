//! Tests added to `rustc_apfloat`, that were not ported from the C++ code.

use rustc_apfloat::ieee::{Double, Single};
use rustc_apfloat::Float;

// `f32 -> i128 -> f32` previously-crashing bit-patterns (found by fuzzing).
pub const FUZZ_IEEE32_ROUNDTRIP_THROUGH_I128_CASES: &[u32] = &[
    0xff000000, // -1.7014118e+38
    0xff00e203, // -1.713147e+38
    0xff00e900, // -1.7135099e+38
    0xff7fffff, // -3.4028235e+38
    0xff800000, // -inf
];

// `f64 -> i128 -> f64` previously-crashing bit-patterns (found by fuzzing).
pub const FUZZ_IEEE64_ROUNDTRIP_THROUGH_I128_CASES: &[u64] = &[
    0xc7e5d58020ffedff, // -2.3217876724230413e+38
    0xc7e7fffefefeff00, // -2.5521161229511617e+38
    0xc7e8030000653636, // -2.5533639056589687e+38
    0xea3501e2e8950007, // -4.116505897277026e+203
    0xf3ff0620ca000600, // -5.553072340247723e+250
    0xffc909842600d4ff, // -3.516340112093497e+307
    0xfff0000000000000, // -inf
];

#[test]
fn fuzz_roundtrip_through_i128() {
    for &bits in FUZZ_IEEE32_ROUNDTRIP_THROUGH_I128_CASES {
        assert_eq!(
            Single::from_i128(Single::from_bits(bits.into()).to_i128(128).value)
                .value
                .to_bits(),
            (f32::from_bits(bits) as i128 as f32).to_bits().into()
        );
    }
    for &bits in FUZZ_IEEE64_ROUNDTRIP_THROUGH_I128_CASES {
        assert_eq!(
            Double::from_i128(Double::from_bits(bits.into()).to_i128(128).value)
                .value
                .to_bits(),
            (f64::from_bits(bits) as i128 as f64).to_bits().into()
        );
    }
}
