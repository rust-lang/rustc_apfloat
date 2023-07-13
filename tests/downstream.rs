//! Tests added to `rustc_apfloat`, that were not ported from the C++ code.

use rustc_apfloat::ieee::{Double, Single, X87DoubleExtended};
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

// `f32` FMA bit-patterns which used to produce the wrong output (found by fuzzing).
pub const FUZZ_IEEE32_FMA_CASES_WITH_EXPECTED_OUTPUTS: &[((u32, u32, u32), u32)] = &[
    ((0x00001000 /* 5.74e-42 */, 0x0000001a /* 3.6e-44 */, 0xffff1a00 /* NaN */), 0xffff1a00 /* NaN */),
    ((0x000080aa /* 4.6156e-41 */, 0xaaff0000 /* -4.52971e-13 */, 0xff9e007f /* NaN */), 0xffde007f /* NaN */),
    ((0x0000843f /* 4.7441e-41 */, 0x0084ff80 /* 1.2213942e-38 */, 0xffff8000 /* NaN */), 0xffff8000 /* NaN */),
    ((0x00009eaa /* 5.6918e-41 */, 0x201d7f1e /* 1.3340477e-19 */, 0xffff0001 /* NaN */), 0xffff0001 /* NaN */),
    ((0x020400ff /* 9.698114e-38 */, 0x7f7f2200 /* 3.3912968e+38 */, 0xffffffff /* NaN */), 0xffffffff /* NaN */),
    ((0x02060320 /* 9.845662e-38 */, 0x20002521 /* 1.0854307e-19 */, 0x7f800000 /* inf */), 0x7f800000 /* inf */),
    (
        (0x04000080 /* 1.5046557e-36 */, 0xff7fff00 /* -3.4027717e+38 */, 0xff800000 /* -inf */),
        0xff800000, /* -inf */
    ),
    (
        (0x04007faa /* 1.5104948e-36 */, 0xff200000 /* -2.1267648e+38 */, 0xffff0000 /* NaN */),
        0xffff0000, /* NaN */
    ),
    ((0x1e0603ff /* 7.094727e-21 */, 0x00100000 /* 1.469368e-39 */, 0xffffff4f /* NaN */), 0xffffff4f /* NaN */),
    ((0x200004aa /* 1.0843565e-19 */, 0x00202020 /* 2.95026e-39 */, 0x7fff00ff /* NaN */), 0x7fff00ff /* NaN */),
    (
        (0x20005eaa /* 1.0873343e-19 */, 0x9e9e9e3a /* -1.6794342e-20 */, 0xff9e009e /* NaN */),
        0xffde009e, /* NaN */
    ),
    ((0x20007faa /* 1.0884262e-19 */, 0x9e00611e /* -6.796347e-21 */, 0x7faa0600 /* NaN */), 0x7fea0600 /* NaN */),
    (
        (0x20007faa /* 1.0884262e-19 */, 0xaa069e1e /* -1.1956449e-13 */, 0xffffecff /* NaN */),
        0xffffecff, /* NaN */
    ),
    ((0x20025eaa /* 1.104275e-19 */, 0x9e01033a /* -6.82987e-21 */, 0xff9e009e /* NaN */), 0xffde009e /* NaN */),
    ((0x3314f400 /* 3.4680852e-8 */, 0x00ff7903 /* 2.3461462e-38 */, 0xffffffdb /* NaN */), 0xffffffdb /* NaN */),
    ((0x3314f400 /* 3.4680852e-8 */, 0x00ff7903 /* 2.3461462e-38 */, 0xfffffff6 /* NaN */), 0xfffffff6 /* NaN */),
    ((0x3a218275 /* 0.0006161102 */, 0x3a3a3a3a /* 0.00071040133 */, 0x7f8a063a /* NaN */), 0x7fca063a /* NaN */),
    ((0x40000001 /* 2.0000002 */, 0xfefffffe /* -1.7014116e+38 */, 0xfffe40ff /* NaN */), 0xfffe40ff /* NaN */),
    ((0x50007faa /* 8623401000 */, 0x000011fb /* 6.45e-42 */, 0xff800000 /* -inf */), 0xff800000 /* -inf */),
    ((0x64007f8b /* 9.481495e+21 */, 0xfa9a8702 /* -4.01176e+35 */, 0xff820000 /* NaN */), 0xffc20000 /* NaN */),
    ((0x6a017faa /* 3.9138577e+25 */, 0x00000070 /* 1.57e-43 */, 0xff80db03 /* NaN */), 0xffc0db03 /* NaN */),
    ((0x6a017faa /* 3.9138577e+25 */, 0x00000070 /* 1.57e-43 */, 0xff80db23 /* NaN */), 0xffc0db23 /* NaN */),
    (
        (0x6e000000 /* 9.9035203e+27 */, 0xdf008000 /* -9259401000000000000 */, 0x7f800000 /* inf */),
        0x7f800000, /* inf */
    ),
    ((0x7f7fff00 /* 3.4027717e+38 */, 0x02000080 /* 9.404098e-38 */, 0x7fc00000 /* NaN */), 0x7fc00000 /* NaN */),
    (
        (0xb3eb00ff /* -1.09432214e-7 */, 0x00ffefe2 /* 2.3504105e-38 */, 0xfffffee9 /* NaN */),
        0xfffffee9, /* NaN */
    ),
    (
        (0xdf0603ff /* -9656842000000000000 */, 0x808000ff /* -1.1755301e-38 */, 0xff9b0000 /* NaN */),
        0xffdb0000, /* NaN */
    ),
    (
        (
            0xf1001101, /* -634154200000000000000000000000 */
            0x7f400000, /* 255211780000000000000000000000000000000 */
            0x7f800000, /* inf */
        ),
        0x7f800000, /* inf */
    ),
    ((0xf5000080 /* -1.6226175e+32 */, 0xc9ffff00 /* -2097120 */, 0xffff7fff /* NaN */), 0xffff7fff /* NaN */),
    (
        (0xf5ffffff /* -6.4903707e+32 */, 0xff000b09 /* -1.7019848e+38 */, 0xff800000 /* -inf */),
        0xff800000, /* -inf */
    ),
    (
        (0xf70029e8 /* -2.5994686e+33 */, 0xf7ffff7f /* -1.0384514e+34 */, 0xffff7fff /* NaN */),
        0xffff7fff, /* NaN */
    ),
    (
        (0xff007faa /* -1.7080405e+38 */, 0xd3fface5 /* -2196234700000 */, 0xffff7f00 /* NaN */),
        0xffff7f00, /* NaN */
    ),
    (
        (0xff200000 /* -2.1267648e+38 */, 0xe380ffff /* -4.7592594e+21 */, 0xff800000 /* -inf */),
        0xff800000, /* -inf */
    ),
    ((0xff6d0000 /* -3.1502704e+38 */, 0xc12005ff /* -10.001464 */, 0xff800000 /* -inf */), 0xff800000 /* -inf */),
];

// `f64` FMA bit-patterns which used to produce the wrong output (found by fuzzing).
pub const FUZZ_IEEE64_FMA_CASES_WITH_EXPECTED_OUTPUTS: &[((u64, u64, u64), u64)] = &[
    (
        (
            0x000000000000001e, /* 1.5e-322 */
            0x00000000ffdf0000, /* 2.120927281e-314 */
            0xffffff8000000000, /* NaN */
        ),
        0xffffff8000000000, /* NaN */
    ),
    (
        (
            0x000000007fffffff, /* 1.060997895e-314 */
            0xff00000000200000, /* -5.486124071348364e+303 */
            0xfffd0000000000e9, /* NaN */
        ),
        0xfffd0000000000e9, /* NaN */
    ),
    (
        (
            0x0000020000e30000, /* 1.086469195027e-311 */
            0xff00000011000000, /* -5.48612441622957e+303 */
            0xfffd00000000e0e9, /* NaN */
        ),
        0xfffd00000000e0e9, /* NaN */
    ),
    (
        (
            0x0000040000006400, /* 2.1729237025965e-311 */
            0x000000e5ff000000, /* 4.88050742876e-312 */
            0xffffffe300000000, /* NaN */
        ),
        0xffffffe300000000, /* NaN */
    ),
    (
        (
            0x00006a0000000000, /* 5.75824777836336e-310 */
            0x005015000018f9f1, /* 3.5783707339010265e-307 */
            0x7fffffde00000000, /* NaN */
        ),
        0x7fffffde00000000, /* NaN */
    ),
    (
        (
            0x00007ffa01000373, /* 6.95208343930866e-310 */
            0x0005000000ff107f, /* 6.95335589042254e-309 */
            0xffffffff00000005, /* NaN */
        ),
        0xffffffff00000005, /* NaN */
    ),
    (
        (
            0x0000ff8000000000, /* 1.387955006954565e-309 */
            0x0000000001000000, /* 8.289046e-317 */
            0xfff0000000000000, /* -inf */
        ),
        0xfff0000000000000, /* -inf */
    ),
    (
        (
            0x0002a000f6290000, /* 3.650532203442106e-309 */
            0x400013fffd000000, /* 2.009765602648258 */
            0xfffdfe0000ff9aff, /* NaN */
        ),
        0xfffdfe0000ff9aff, /* NaN */
    ),
    (
        (
            0x0006000000001700, /* 8.344026969431096e-309 */
            0xd9000000da080000, /* -5.164503950933907e+120 */
            0xfffffee5000000fd, /* NaN */
        ),
        0xfffffee5000000fd, /* NaN */
    ),
    (
        (
            0x0006000040000013, /* 8.344032274391576e-309 */
            0xfafe036500061100, /* -2.7893890583525793e+284 */
            0xffff7fff00001011, /* NaN */
        ),
        0xffff7fff00001011, /* NaN */
    ),
    (
        (
            0x00f1000000640000, /* 3.873408578194326e-304 */
            0xffe6005e00000000, /* -1.2359946076651026e+308 */
            0xfffd007000000000, /* NaN */
        ),
        0xfffd007000000000, /* NaN */
    ),
    (
        (
            0x05203a0080ff0513, /* 5.456081264530354e-284 */
            0xf90000000000f7ff, /* -6.924462078599005e+274 */
            0xfff0000000000000, /* -inf */
        ),
        0xfff0000000000000, /* -inf */
    ),
    (
        (
            0x0540400001000513, /* 2.1855837639726535e-283 */
            0xee05130640000100, /* -9.522265158052987e+221 */
            0x7fff00001004fa01, /* NaN */
        ),
        0x7fff00001004fa01, /* NaN */
    ),
    (
        (
            0x0540400001000513, /* 2.1855837639726535e-283 */
            0xffd8000000000000, /* -6.741349255733685e+307 */
            0xfff0001000000000, /* NaN */
        ),
        0xfff8001000000000, /* NaN */
    ),
    (
        (
            0x054040000100e213, /* 2.1855837639996873e-283 */
            0xfbd8000000000000, /* -3.6544927542749997e+288 */
            0xfff0ff1000000000, /* NaN */
        ),
        0xfff8ff1000000000, /* NaN */
    ),
    (
        (
            0x060000000000ff04, /* 8.814425663530262e-280 */
            0x00000020ffff0606, /* 7.00258294846e-313 */
            0xffffffde00001300, /* NaN */
        ),
        0xffffffde00001300, /* NaN */
    ),
    (
        (
            0x1306400001000513, /* 5.042468007014986e-217 */
            0x00001004fa03ee05, /* 8.7022551317144e-311 */
            0xfffc80f7ffff7fff, /* NaN */
        ),
        0xfffc80f7ffff7fff, /* NaN */
    ),
    (
        (
            0x1306400001000513, /* 5.042468007014986e-217 */
            0xa5001004fa01ee05, /* -1.810368898568446e-130 */
            0xfffa80f7ff1b7fff, /* NaN */
        ),
        0xfffa80f7ff1b7fff, /* NaN */
    ),
    (
        (
            0x4006400005130100, /* 2.7812500378059895 */
            0x0000ff4000000000, /* 1.38659692964835e-309 */
            0x7fffffec4200044b, /* NaN */
        ),
        0x7fffffec4200044b, /* NaN */
    ),
    (
        (
            0x4100000001000000, /* 131072.00048828125 */
            0x0000fffffff00000, /* 1.390671156386347e-309 */
            0xfffffe00000040ff, /* NaN */
        ),
        0xfffffe00000040ff, /* NaN */
    ),
    (
        (
            0x7a7a7a7a7a7a0000, /* 9.61276249042562e+281 */
            0xff7a7a7a7a7a7a7a, /* -1.1621116772547446e+306 */
            0xfffd007000ef0000, /* NaN */
        ),
        0xfffd007000ef0000, /* NaN */
    ),
    (
        (
            0x7f000012007ff010, /* 5.4862182545686e+303 */
            0x7f0000120091f010, /* 5.486218256005604e+303 */
            0xfff0000000000000, /* -inf */
        ),
        0xfff0000000000000, /* -inf */
    ),
    (
        (
            0x7f0022000a8000f6, /* 5.531663399192155e+303 */
            0xff00ebfef0800300, /* -5.802213559159178e+303 */
            0x7ff0000000000000, /* inf */
        ),
        0x7ff0000000000000, /* inf */
    ),
    (
        (
            0x7f06400001000513, /* 7.62914130360521e+303 */
            0xff001004fb88f7ff, /* -5.507580309563204e+303 */
            0xfffa01ee0513ffff, /* NaN */
        ),
        0xfffa01ee0513ffff, /* NaN */
    ),
    (
        (
            0xbbbbbb7f01000513, /* -5.872565540268704e-21 */
            0x0100bbbbbbbbbbbb, /* 7.625298445452731e-304 */
            0xffffff4000004000, /* NaN */
        ),
        0xffffff4000004000, /* NaN */
    ),
    (
        (
            0xbc00000000400000, /* -1.0842021734952464e-19 */
            0x00bc000000004000, /* 3.987332354453194e-305 */
            0xfff0000000e20000, /* NaN */
        ),
        0xfff8000000e20000, /* NaN */
    ),
    (
        (
            0xddff000004000000, /* -6.048387862754913e+144 */
            0xff00000000000000, /* -5.486124068793689e+303 */
            0xffffff0000000000, /* NaN */
        ),
        0xffffff0000000000, /* NaN */
    ),
    (
        (
            0xe100051b060c0513, /* -1.759578741202065e+159 */
            0xfbfeee0513064110, /* -1.8838766970066999e+289 */
            0xffff7fdf00001004, /* NaN */
        ),
        0xffff7fdf00001004, /* NaN */
    ),
    (
        (
            0xf0000000007ff010, /* -3.1050361903821855e+231 */
            0x7f06010800180000, /* 7.54480183807128e+303 */
            0x7ff0000000000000, /* inf */
        ),
        0x7ff0000000000000, /* inf */
    ),
    (
        (
            0xf4ffff05021d7d12, /* -3.753309156386366e+255 */
            0xfd100000e8030000, /* -2.5546778042386733e+294 */
            0xfff0000000000000, /* -inf */
        ),
        0xfff0000000000000, /* -inf */
    ),
    (
        (
            0xff0000fff05f0001, /* -5.48746313513839e+303 */
            0xff0000fff0800000, /* -5.487463137772898e+303 */
            0xfff0000000000000, /* -inf */
        ),
        0xfff0000000000000, /* -inf */
    ),
];

#[test]
fn fuzz_fma_with_expected_outputs() {
    for &((a_bits, b_bits, c_bits), expected_bits) in FUZZ_IEEE32_FMA_CASES_WITH_EXPECTED_OUTPUTS {
        let (a, b, c) =
            (Single::from_bits(a_bits.into()), Single::from_bits(b_bits.into()), Single::from_bits(c_bits.into()));
        assert_eq!(a.mul_add(b, c).value.to_bits(), expected_bits.into());
    }
    for &((a_bits, b_bits, c_bits), expected_bits) in FUZZ_IEEE64_FMA_CASES_WITH_EXPECTED_OUTPUTS {
        let (a, b, c) =
            (Double::from_bits(a_bits.into()), Double::from_bits(b_bits.into()), Double::from_bits(c_bits.into()));
        assert_eq!(a.mul_add(b, c).value.to_bits(), expected_bits.into());
    }
}

// x87 80-bit "extended precision"/`long double` bit-patterns which used to
// produce the wrong output when negated (found by fuzzing - though fuzzing also
// found many examples in all ops, as the root issue was the handling of the
// bit-level encoding itself, but negation was the easiest op to test here).
pub const FUZZ_X87_F80_NEG_CASES_WITH_EXPECTED_OUTPUTS: &[(u128, u128)] = &[
    (0x01010101010100000000 /* NaN */, 0xffff0101010100000000 /* NaN */),
    (
        0x0000ff7f2300ff000000, /* 6.71098449692300485303E-4932 */
        0x8001ff7f2300ff000000, /* -6.71098449692300485303E-4932 */
    ),
    (
        0x00008000000000000000, /* 3.36210314311209350626E-4932 */
        0x80018000000000000000, /* -3.36210314311209350626E-4932 */
    ),
];

#[test]
fn fuzz_x87_f80_neg_with_expected_outputs() {
    for &(bits, expected_bits) in FUZZ_X87_F80_NEG_CASES_WITH_EXPECTED_OUTPUTS {
        assert_eq!((-X87DoubleExtended::from_bits(bits)).to_bits(), expected_bits);
    }
}
