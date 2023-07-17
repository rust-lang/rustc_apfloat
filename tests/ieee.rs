#[macro_use]
extern crate rustc_apfloat;

use core::cmp::Ordering;
use rustc_apfloat::ieee::{BFloat, Double, Float8E4M3FN, Float8E5M2, Half, Quad, Single, X87DoubleExtended};
use rustc_apfloat::{Category, ExpInt, IEK_INF, IEK_NAN, IEK_ZERO};
use rustc_apfloat::{Float, FloatConvert, Round, Status};

// FIXME(eddyb) maybe include this in `rustc_apfloat` itself?
macro_rules! define_for_each_float_type {
    ($($ty:ty),+ $(,)?) => {
        macro_rules! for_each_float_type {
            // FIXME(eddyb) use generic closures if they're ever added to Rust.
            (for<$ty_var:ident: Float> $e:expr) => {{
                $({
                    type $ty_var = $ty;
                    $e;
                })+
            }}
        }
    }
}
define_for_each_float_type! {
    Half,
    Single,
    Double,
    Quad,

    BFloat,
    Float8E5M2,
    Float8E4M3FN,
    X87DoubleExtended,

    // NOTE(eddyb) tests for this are usually in `ppc.rs` but this works too.
    rustc_apfloat::ppc::DoubleDouble,
}

trait SingleExt {
    fn from_f32(input: f32) -> Self;
    fn to_f32(self) -> f32;
}

impl SingleExt for Single {
    fn from_f32(input: f32) -> Self {
        Self::from_bits(input.to_bits() as u128)
    }

    fn to_f32(self) -> f32 {
        f32::from_bits(self.to_bits() as u32)
    }
}

trait DoubleExt {
    fn from_f64(input: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl DoubleExt for Double {
    fn from_f64(input: f64) -> Self {
        Self::from_bits(input.to_bits() as u128)
    }

    fn to_f64(self) -> f64 {
        f64::from_bits(self.to_bits() as u64)
    }
}

// NOTE(eddyb) these match the C++ `convertToFloat`/`convertToDouble` methods,
// after their generalization to allow an optional lossless conversion to their
// expected semantics (from e.g. `IEEEhalf`/`BFloat`, for `convertToSingle`).
// FIXME(eddyb) should the methods have e.g. `_lossless_via_convert` in their names?
fn assert_lossless_conversion<S: FloatConvert<T>, T: Float>(src: S) -> T {
    let mut loses_info = false;
    let status;
    let r = unpack!(status=, src.convert(&mut loses_info));
    assert!(!status.intersects(Status::INEXACT) && !loses_info, "Unexpected imprecision");
    r
}

trait ToF32LosslessViaConvertToSingle: FloatConvert<Single> {
    fn to_f32(self) -> f32 {
        assert_lossless_conversion(self).to_f32()
    }
}
impl ToF32LosslessViaConvertToSingle for Half {}
impl ToF32LosslessViaConvertToSingle for BFloat {}
impl ToF32LosslessViaConvertToSingle for Float8E5M2 {}
impl ToF32LosslessViaConvertToSingle for Float8E4M3FN {}

trait ToF64LosslessViaConvertToDouble: FloatConvert<Double> {
    fn to_f64(self) -> f64 {
        assert_lossless_conversion(self).to_f64()
    }
}
impl ToF64LosslessViaConvertToDouble for Single {}
// HACK(eddyb) take advantage of the transitivity of "are conversions lossless".
impl<T: ToF32LosslessViaConvertToSingle + FloatConvert<Double>> ToF64LosslessViaConvertToDouble for T {}

#[test]
fn is_signaling() {
    // We test qNaN, -qNaN, +sNaN, -sNaN with and without payloads.
    let payload = 4;
    assert!(!Single::qnan(None).is_signaling());
    assert!(!(-Single::qnan(None)).is_signaling());
    assert!(!Single::qnan(Some(payload)).is_signaling());
    assert!(!(-Single::qnan(Some(payload))).is_signaling());
    assert!(Single::snan(None).is_signaling());
    assert!((-Single::snan(None)).is_signaling());
    assert!(Single::snan(Some(payload)).is_signaling());
    assert!((-Single::snan(Some(payload))).is_signaling());
}

#[test]
fn next() {
    // 1. Test Special Cases Values.
    //
    // Test all special values for nextUp and nextDown perscribed by IEEE-754R
    // 2008. These are:
    //   1. +inf
    //   2. -inf
    //   3. largest
    //   4. -largest
    //   5. smallest
    //   6. -smallest
    //   7. qNaN
    //   8. sNaN
    //   9. +0
    //   10. -0

    let mut status;

    // nextUp(+inf) = +inf.
    let test = unpack!(status=, Quad::INFINITY.next_up());
    let expected = Quad::INFINITY;
    assert_eq!(status, Status::OK);
    assert!(test.is_infinite());
    assert!(!test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextDown(+inf) = -nextUp(-inf) = -(-largest) = largest
    let test = unpack!(status=, Quad::INFINITY.next_down());
    let expected = Quad::largest();
    assert_eq!(status, Status::OK);
    assert!(!test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextUp(-inf) = -largest
    let test = unpack!(status=, (-Quad::INFINITY).next_up());
    let expected = -Quad::largest();
    assert_eq!(status, Status::OK);
    assert!(test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextDown(-inf) = -nextUp(+inf) = -(+inf) = -inf.
    let test = unpack!(status=, (-Quad::INFINITY).next_down());
    let expected = -Quad::INFINITY;
    assert_eq!(status, Status::OK);
    assert!(test.is_infinite() && test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextUp(largest) = +inf
    let test = unpack!(status=, Quad::largest().next_up());
    let expected = Quad::INFINITY;
    assert_eq!(status, Status::OK);
    assert!(test.is_infinite() && !test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextDown(largest) = -nextUp(-largest)
    //                        = -(-largest + inc)
    //                        = largest - inc.
    let test = unpack!(status=, Quad::largest().next_down());
    let expected = "0x1.fffffffffffffffffffffffffffep+16383".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(!test.is_infinite() && !test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextUp(-largest) = -largest + inc.
    let test = unpack!(status=, (-Quad::largest()).next_up());
    let expected = "-0x1.fffffffffffffffffffffffffffep+16383".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // nextDown(-largest) = -nextUp(largest) = -(inf) = -inf.
    let test = unpack!(status=, (-Quad::largest()).next_down());
    let expected = -Quad::INFINITY;
    assert_eq!(status, Status::OK);
    assert!(test.is_infinite() && test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextUp(smallest) = smallest + inc.
    let test = unpack!(status=, "0x0.0000000000000000000000000001p-16382"
        .parse::<Quad>()
        .unwrap()
        .next_up());
    let expected = "0x0.0000000000000000000000000002p-16382".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // nextDown(smallest) = -nextUp(-smallest) = -(-0) = +0.
    let test = unpack!(status=, "0x0.0000000000000000000000000001p-16382"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = Quad::ZERO;
    assert_eq!(status, Status::OK);
    assert!(test.is_pos_zero());
    assert!(test.bitwise_eq(expected));

    // nextUp(-smallest) = -0.
    let test = unpack!(status=, "-0x0.0000000000000000000000000001p-16382"
        .parse::<Quad>()
        .unwrap()
        .next_up());
    let expected = -Quad::ZERO;
    assert_eq!(status, Status::OK);
    assert!(test.is_neg_zero());
    assert!(test.bitwise_eq(expected));

    // nextDown(-smallest) = -nextUp(smallest) = -smallest - inc.
    let test = unpack!(status=, "-0x0.0000000000000000000000000001p-16382"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = "-0x0.0000000000000000000000000002p-16382".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // nextUp(qNaN) = qNaN
    let test = unpack!(status=, Quad::qnan(None).next_up());
    let expected = Quad::qnan(None);
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // nextDown(qNaN) = qNaN
    let test = unpack!(status=, Quad::qnan(None).next_down());
    let expected = Quad::qnan(None);
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // nextUp(sNaN) = qNaN
    let test = unpack!(status=, Quad::snan(None).next_up());
    let expected = Quad::qnan(None);
    assert_eq!(status, Status::INVALID_OP);
    assert!(test.bitwise_eq(expected));

    // nextDown(sNaN) = qNaN
    let test = unpack!(status=, Quad::snan(None).next_down());
    let expected = Quad::qnan(None);
    assert_eq!(status, Status::INVALID_OP);
    assert!(test.bitwise_eq(expected));

    // nextUp(+0) = +smallest
    let test = unpack!(status=, Quad::ZERO.next_up());
    let expected = Quad::SMALLEST;
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // nextDown(+0) = -nextUp(-0) = -smallest
    let test = unpack!(status=, Quad::ZERO.next_down());
    let expected = -Quad::SMALLEST;
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // nextUp(-0) = +smallest
    let test = unpack!(status=, (-Quad::ZERO).next_up());
    let expected = Quad::SMALLEST;
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // nextDown(-0) = -nextUp(0) = -smallest
    let test = unpack!(status=, (-Quad::ZERO).next_down());
    let expected = -Quad::SMALLEST;
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // 2. Binade Boundary Tests.

    // 2a. Test denormal <-> normal binade boundaries.
    //     * nextUp(+Largest Denormal) -> +Smallest Normal.
    //     * nextDown(-Largest Denormal) -> -Smallest Normal.
    //     * nextUp(-Smallest Normal) -> -Largest Denormal.
    //     * nextDown(+Smallest Normal) -> +Largest Denormal.

    // nextUp(+Largest Denormal) -> +Smallest Normal.
    let test = unpack!(status=, "0x0.ffffffffffffffffffffffffffffp-16382"
        .parse::<Quad>()
        .unwrap()
        .next_up());
    let expected = "0x1.0000000000000000000000000000p-16382".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(!test.is_denormal());
    assert!(test.bitwise_eq(expected));

    // nextDown(-Largest Denormal) -> -Smallest Normal.
    let test = unpack!(status=, "-0x0.ffffffffffffffffffffffffffffp-16382"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = "-0x1.0000000000000000000000000000p-16382".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(!test.is_denormal());
    assert!(test.bitwise_eq(expected));

    // nextUp(-Smallest Normal) -> -Largest Denormal.
    let test = unpack!(status=, "-0x1.0000000000000000000000000000p-16382"
        .parse::<Quad>()
        .unwrap()
        .next_up());
    let expected = "-0x0.ffffffffffffffffffffffffffffp-16382".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.is_denormal());
    assert!(test.bitwise_eq(expected));

    // nextDown(+Smallest Normal) -> +Largest Denormal.
    let test = unpack!(status=, "+0x1.0000000000000000000000000000p-16382"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = "+0x0.ffffffffffffffffffffffffffffp-16382".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.is_denormal());
    assert!(test.bitwise_eq(expected));

    // 2b. Test normal <-> normal binade boundaries.
    //     * nextUp(-Normal Binade Boundary) -> -Normal Binade Boundary + 1.
    //     * nextDown(+Normal Binade Boundary) -> +Normal Binade Boundary - 1.
    //     * nextUp(+Normal Binade Boundary - 1) -> +Normal Binade Boundary.
    //     * nextDown(-Normal Binade Boundary + 1) -> -Normal Binade Boundary.

    // nextUp(-Normal Binade Boundary) -> -Normal Binade Boundary + 1.
    let test = unpack!(status=, "-0x1p+1".parse::<Quad>().unwrap().next_up());
    let expected = "-0x1.ffffffffffffffffffffffffffffp+0".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // nextDown(+Normal Binade Boundary) -> +Normal Binade Boundary - 1.
    let test = unpack!(status=, "0x1p+1".parse::<Quad>().unwrap().next_down());
    let expected = "0x1.ffffffffffffffffffffffffffffp+0".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // nextUp(+Normal Binade Boundary - 1) -> +Normal Binade Boundary.
    let test = unpack!(status=, "0x1.ffffffffffffffffffffffffffffp+0"
        .parse::<Quad>()
        .unwrap()
        .next_up());
    let expected = "0x1p+1".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // nextDown(-Normal Binade Boundary + 1) -> -Normal Binade Boundary.
    let test = unpack!(status=, "-0x1.ffffffffffffffffffffffffffffp+0"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = "-0x1p+1".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // 2c. Test using next at binade boundaries with a direction away from the
    // binade boundary. Away from denormal <-> normal boundaries.
    //
    // This is to make sure that even though we are at a binade boundary, since
    // we are rounding away, we do not trigger the binade boundary code. Thus we
    // test:
    //   * nextUp(-Largest Denormal) -> -Largest Denormal + inc.
    //   * nextDown(+Largest Denormal) -> +Largest Denormal - inc.
    //   * nextUp(+Smallest Normal) -> +Smallest Normal + inc.
    //   * nextDown(-Smallest Normal) -> -Smallest Normal - inc.

    // nextUp(-Largest Denormal) -> -Largest Denormal + inc.
    let test = unpack!(status=, "-0x0.ffffffffffffffffffffffffffffp-16382"
        .parse::<Quad>()
        .unwrap()
        .next_up());
    let expected = "-0x0.fffffffffffffffffffffffffffep-16382".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.is_denormal());
    assert!(test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextDown(+Largest Denormal) -> +Largest Denormal - inc.
    let test = unpack!(status=, "0x0.ffffffffffffffffffffffffffffp-16382"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = "0x0.fffffffffffffffffffffffffffep-16382".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.is_denormal());
    assert!(!test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextUp(+Smallest Normal) -> +Smallest Normal + inc.
    let test = unpack!(status=, "0x1.0000000000000000000000000000p-16382"
        .parse::<Quad>()
        .unwrap()
        .next_up());
    let expected = "0x1.0000000000000000000000000001p-16382".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(!test.is_denormal());
    assert!(!test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextDown(-Smallest Normal) -> -Smallest Normal - inc.
    let test = unpack!(status=, "-0x1.0000000000000000000000000000p-16382"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = "-0x1.0000000000000000000000000001p-16382".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(!test.is_denormal());
    assert!(test.is_negative());
    assert!(test.bitwise_eq(expected));

    // 2d. Test values which cause our exponent to go to min exponent. This
    // is to ensure that guards in the code to check for min exponent
    // trigger properly.
    //     * nextUp(-0x1p-16381) -> -0x1.ffffffffffffffffffffffffffffp-16382
    //     * nextDown(-0x1.ffffffffffffffffffffffffffffp-16382) ->
    //         -0x1p-16381
    //     * nextUp(0x1.ffffffffffffffffffffffffffffp-16382) -> 0x1p-16382
    //     * nextDown(0x1p-16382) -> 0x1.ffffffffffffffffffffffffffffp-16382

    // nextUp(-0x1p-16381) -> -0x1.ffffffffffffffffffffffffffffp-16382
    let test = unpack!(status=, "-0x1p-16381".parse::<Quad>().unwrap().next_up());
    let expected = "-0x1.ffffffffffffffffffffffffffffp-16382".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // nextDown(-0x1.ffffffffffffffffffffffffffffp-16382) ->
    //         -0x1p-16381
    let test = unpack!(status=, "-0x1.ffffffffffffffffffffffffffffp-16382"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = "-0x1p-16381".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // nextUp(0x1.ffffffffffffffffffffffffffffp-16382) -> 0x1p-16381
    let test = unpack!(status=, "0x1.ffffffffffffffffffffffffffffp-16382"
        .parse::<Quad>()
        .unwrap()
        .next_up());
    let expected = "0x1p-16381".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // nextDown(0x1p-16381) -> 0x1.ffffffffffffffffffffffffffffp-16382
    let test = unpack!(status=, "0x1p-16381".parse::<Quad>().unwrap().next_down());
    let expected = "0x1.ffffffffffffffffffffffffffffp-16382".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // 3. Now we test both denormal/normal computation which will not cause us
    // to go across binade boundaries. Specifically we test:
    //   * nextUp(+Denormal) -> +Denormal.
    //   * nextDown(+Denormal) -> +Denormal.
    //   * nextUp(-Denormal) -> -Denormal.
    //   * nextDown(-Denormal) -> -Denormal.
    //   * nextUp(+Normal) -> +Normal.
    //   * nextDown(+Normal) -> +Normal.
    //   * nextUp(-Normal) -> -Normal.
    //   * nextDown(-Normal) -> -Normal.

    // nextUp(+Denormal) -> +Denormal.
    let test = unpack!(status=, "0x0.ffffffffffffffffffffffff000cp-16382"
        .parse::<Quad>()
        .unwrap()
        .next_up());
    let expected = "0x0.ffffffffffffffffffffffff000dp-16382".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.is_denormal());
    assert!(!test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextDown(+Denormal) -> +Denormal.
    let test = unpack!(status=, "0x0.ffffffffffffffffffffffff000cp-16382"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = "0x0.ffffffffffffffffffffffff000bp-16382".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.is_denormal());
    assert!(!test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextUp(-Denormal) -> -Denormal.
    let test = unpack!(status=, "-0x0.ffffffffffffffffffffffff000cp-16382"
        .parse::<Quad>()
        .unwrap()
        .next_up());
    let expected = "-0x0.ffffffffffffffffffffffff000bp-16382".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.is_denormal());
    assert!(test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextDown(-Denormal) -> -Denormal
    let test = unpack!(status=, "-0x0.ffffffffffffffffffffffff000cp-16382"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = "-0x0.ffffffffffffffffffffffff000dp-16382".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.is_denormal());
    assert!(test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextUp(+Normal) -> +Normal.
    let test = unpack!(status=, "0x1.ffffffffffffffffffffffff000cp-16000"
        .parse::<Quad>()
        .unwrap()
        .next_up());
    let expected = "0x1.ffffffffffffffffffffffff000dp-16000".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(!test.is_denormal());
    assert!(!test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextDown(+Normal) -> +Normal.
    let test = unpack!(status=, "0x1.ffffffffffffffffffffffff000cp-16000"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = "0x1.ffffffffffffffffffffffff000bp-16000".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(!test.is_denormal());
    assert!(!test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextUp(-Normal) -> -Normal.
    let test = unpack!(status=, "-0x1.ffffffffffffffffffffffff000cp-16000"
        .parse::<Quad>()
        .unwrap()
        .next_up());
    let expected = "-0x1.ffffffffffffffffffffffff000bp-16000".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(!test.is_denormal());
    assert!(test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextDown(-Normal) -> -Normal.
    let test = unpack!(status=, "-0x1.ffffffffffffffffffffffff000cp-16000"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = "-0x1.ffffffffffffffffffffffff000dp-16000".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(!test.is_denormal());
    assert!(test.is_negative());
    assert!(test.bitwise_eq(expected));
}

#[test]
fn fma() {
    {
        let mut f1 = Single::from_f32(14.5);
        let f2 = Single::from_f32(-14.5);
        let f3 = Single::from_f32(225.0);
        f1 = f1.mul_add(f2, f3).value;
        assert_eq!(14.75, f1.to_f32());
    }

    {
        let val2 = Single::from_f32(2.0);
        let mut f1 = Single::from_f32(1.17549435e-38);
        let mut f2 = Single::from_f32(1.17549435e-38);
        f1 /= val2;
        f2 /= val2;
        let f3 = Single::from_f32(12.0);
        f1 = f1.mul_add(f2, f3).value;
        assert_eq!(12.0, f1.to_f32());
    }

    // Test for correct zero sign when answer is exactly zero.
    // fma(1.0, -1.0, 1.0) -> +ve 0.
    {
        let mut f1 = Double::from_f64(1.0);
        let f2 = Double::from_f64(-1.0);
        let f3 = Double::from_f64(1.0);
        f1 = f1.mul_add(f2, f3).value;
        assert!(!f1.is_negative() && f1.is_zero());
    }

    // Test for correct zero sign when answer is exactly zero and rounding towards
    // negative.
    // fma(1.0, -1.0, 1.0) -> +ve 0.
    {
        let mut f1 = Double::from_f64(1.0);
        let f2 = Double::from_f64(-1.0);
        let f3 = Double::from_f64(1.0);
        f1 = f1.mul_add_r(f2, f3, Round::TowardNegative).value;
        assert!(f1.is_negative() && f1.is_zero());
    }

    // Test for correct (in this case -ve) sign when adding like signed zeros.
    // Test fma(0.0, -0.0, -0.0) -> -ve 0.
    {
        let mut f1 = Double::from_f64(0.0);
        let f2 = Double::from_f64(-0.0);
        let f3 = Double::from_f64(-0.0);
        f1 = f1.mul_add(f2, f3).value;
        assert!(f1.is_negative() && f1.is_zero());
    }

    // Test -ve sign preservation when small negative results underflow.
    {
        let mut f1 = "-0x1p-1074".parse::<Double>().unwrap();
        let f2 = "+0x1p-1074".parse::<Double>().unwrap();
        let f3 = Double::from_f64(0.0);
        f1 = f1.mul_add(f2, f3).value;
        assert!(f1.is_negative() && f1.is_zero());
    }

    // Test x87 extended precision case from http://llvm.org/PR20728.
    {
        let mut m1 = X87DoubleExtended::from_u128(1).value;
        let m2 = X87DoubleExtended::from_u128(1).value;
        let a = X87DoubleExtended::from_u128(3).value;

        let mut loses_info = false;
        m1 = m1.mul_add(m2, a).value;
        let r: Single = m1.convert(&mut loses_info).value;
        assert!(!loses_info);
        assert_eq!(4.0, r.to_f32());
    }

    // Regression test that failed an assertion.
    {
        let mut f1 = Single::from_f32(-8.85242279E-41);
        let f2 = Single::from_f32(2.0);
        let f3 = Single::from_f32(8.85242279E-41);
        f1 = f1.mul_add(f2, f3).value;
        assert_eq!(-8.85242279E-41, f1.to_f32());
    }

    // Test using only a single instance of APFloat.
    {
        let mut f = Double::from_f64(1.5);

        f = f.mul_add(f, f).value;
        assert_eq!(3.75, f.to_f64());
    }
}

#[test]
fn min_num() {
    let f1 = Double::from_f64(1.0);
    let f2 = Double::from_f64(2.0);
    let nan = Double::NAN;

    assert_eq!(1.0, f1.min(f2).to_f64());
    assert_eq!(1.0, f2.min(f1).to_f64());
    assert_eq!(1.0, f1.min(nan).to_f64());
    assert_eq!(1.0, nan.min(f1).to_f64());
}

#[test]
fn max_num() {
    let f1 = Double::from_f64(1.0);
    let f2 = Double::from_f64(2.0);
    let nan = Double::NAN;

    assert_eq!(2.0, f1.max(f2).to_f64());
    assert_eq!(2.0, f2.max(f1).to_f64());
    assert_eq!(1.0, f1.max(nan).to_f64());
    assert_eq!(1.0, nan.max(f1).to_f64());
}

#[test]
fn minimum() {
    let f1 = Double::from_f64(1.0);
    let f2 = Double::from_f64(2.0);
    let zp = Double::from_f64(0.0);
    let zn = Double::from_f64(-0.0);
    let nan = Double::NAN;

    assert_eq!(1.0, f1.minimum(f2).to_f64());
    assert_eq!(1.0, f2.minimum(f1).to_f64());
    assert_eq!(-0.0, zp.minimum(zn).to_f64());
    assert_eq!(-0.0, zn.minimum(zp).to_f64());
    assert!(f1.minimum(nan).to_f64().is_nan());
    assert!(nan.minimum(f1).to_f64().is_nan());
}

#[test]
fn maximum() {
    let f1 = Double::from_f64(1.0);
    let f2 = Double::from_f64(2.0);
    let zp = Double::from_f64(0.0);
    let zn = Double::from_f64(-0.0);
    let nan = Double::NAN;

    assert_eq!(2.0, f1.maximum(f2).to_f64());
    assert_eq!(2.0, f2.maximum(f1).to_f64());
    assert_eq!(0.0, zp.maximum(zn).to_f64());
    assert_eq!(0.0, zn.maximum(zp).to_f64());
    assert!(f1.maximum(nan).to_f64().is_nan());
    assert!(nan.maximum(f1).to_f64().is_nan());
}

#[test]
fn denormal() {
    // Test single precision
    {
        assert!(!Single::from_u128(0).value.is_denormal());

        let mut t = "1.17549435082228750797e-38".parse::<Single>().unwrap();
        assert!(!t.is_denormal());

        t /= Single::from_u128(2).value;
        assert!(t.is_denormal());
    }

    // Test double precision
    {
        assert!(!Double::from_u128(0).value.is_denormal());

        let mut t = "2.22507385850720138309e-308".parse::<Double>().unwrap();
        assert!(!t.is_denormal());

        t /= Double::from_u128(2).value;
        assert!(t.is_denormal());
    }

    // Test Intel double-ext
    {
        assert!(!X87DoubleExtended::from_u128(0).value.is_denormal());

        let mut t = "3.36210314311209350626e-4932".parse::<X87DoubleExtended>().unwrap();
        assert!(!t.is_denormal());

        t /= X87DoubleExtended::from_u128(2).value;
        assert!(t.is_denormal());
    }

    // Test quadruple precision
    {
        assert!(!Quad::from_u128(0).value.is_denormal());

        let mut t = "3.36210314311209350626267781732175260e-4932".parse::<Quad>().unwrap();
        assert!(!t.is_denormal());

        t /= Quad::from_u128(2).value;
        assert!(t.is_denormal());
    }
}

#[test]
fn is_smallest_normalized() {
    for_each_float_type!(for<F: Float> test::<F>());
    fn test<F: Float>() {
        assert!(!F::ZERO.is_smallest_normalized());
        assert!(!(-F::ZERO).is_smallest_normalized());

        assert!(!F::INFINITY.is_smallest_normalized());
        assert!(!(-F::INFINITY).is_smallest_normalized());

        assert!(!F::qnan(None).is_smallest_normalized());
        assert!(!F::snan(None).is_smallest_normalized());

        assert!(!F::largest().is_smallest_normalized());
        assert!(!(-F::largest()).is_smallest_normalized());

        assert!(!F::SMALLEST.is_smallest_normalized());
        assert!(!(-F::SMALLEST).is_smallest_normalized());

        assert!(!F::from_bits(!0u128 >> (128 - F::BITS)).is_smallest_normalized());

        let pos_smallest_normalized = F::smallest_normalized();
        let neg_smallest_normalized = -F::smallest_normalized();
        assert!(pos_smallest_normalized.is_smallest_normalized());
        assert!(neg_smallest_normalized.is_smallest_normalized());

        for mut val in [pos_smallest_normalized, neg_smallest_normalized] {
            let old_sign = val.is_negative();

            let mut status;

            // Step down, make sure it's still not smallest normalized.
            val = unpack!(status=, val.next_down());
            assert_eq!(Status::OK, status);
            assert_eq!(old_sign, val.is_negative());
            assert!(!val.is_smallest_normalized());
            assert_eq!(old_sign, val.is_negative());

            // Step back up should restore it to being smallest normalized.
            val = unpack!(status=, val.next_up());
            assert_eq!(Status::OK, status);
            assert!(val.is_smallest_normalized());
            assert_eq!(old_sign, val.is_negative());

            // Step beyond should no longer smallest normalized.
            val = unpack!(status=, val.next_up());
            assert_eq!(Status::OK, status);
            assert!(!val.is_smallest_normalized());
            assert_eq!(old_sign, val.is_negative());
        }
    }
}

#[test]
fn decimal_strings_without_null_terminators() {
    // Make sure that we can parse strings without null terminators.
    // rdar://14323230.
    assert_eq!("0.00"[..3].parse::<Double>().unwrap().to_f64(), 0.0);
    assert_eq!("0.01"[..3].parse::<Double>().unwrap().to_f64(), 0.0);
    assert_eq!("0.09"[..3].parse::<Double>().unwrap().to_f64(), 0.0);
    assert_eq!("0.095"[..4].parse::<Double>().unwrap().to_f64(), 0.09);
    assert_eq!("0.00e+3"[..7].parse::<Double>().unwrap().to_f64(), 0.00);
    assert_eq!("0e+3"[..4].parse::<Double>().unwrap().to_f64(), 0.00);
}

#[test]
fn from_zero_decimal_string() {
    assert_eq!(0.0, "0".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0.".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0.".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0.".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, ".0".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+.0".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-.0".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0.0".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0.0".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0.0".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "00000.".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+00000.".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-00000.".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, ".00000".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+.00000".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-.00000".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0000.00000".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0000.00000".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0000.00000".parse::<Double>().unwrap().to_f64());
}

#[test]
fn from_zero_decimal_single_exponent_string() {
    assert_eq!(0.0, "0e1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0e1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0e1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0e+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0e+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0e+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0e-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0e-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0e-1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0.e1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0.e1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0.e1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0.e+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0.e+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0.e+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0.e-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0.e-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0.e-1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, ".0e1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+.0e1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-.0e1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, ".0e+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+.0e+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-.0e+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, ".0e-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+.0e-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-.0e-1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0.0e1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0.0e1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0.0e1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0.0e+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0.0e+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0.0e+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0.0e-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0.0e-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0.0e-1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "000.0000e1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+000.0000e+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-000.0000e+1".parse::<Double>().unwrap().to_f64());
}

#[test]
fn from_zero_decimal_large_exponent_string() {
    assert_eq!(0.0, "0e1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0e1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0e1234".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0e+1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0e+1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0e+1234".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0e-1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0e-1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0e-1234".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "000.0000e1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "000.0000e-1234".parse::<Double>().unwrap().to_f64());
}

#[test]
fn from_zero_hexadecimal_string() {
    assert_eq!(0.0, "0x0p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x0p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x0p1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x0p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x0p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x0p+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x0p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x0p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x0p-1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x0.p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x0.p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x0.p1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x0.p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x0.p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x0.p+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x0.p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x0.p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x0.p-1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x.0p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x.0p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x.0p1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x.0p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x.0p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x.0p+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x.0p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x.0p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x.0p-1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x0.0p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x0.0p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x0.0p1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x0.0p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x0.0p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x0.0p+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x0.0p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x0.0p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x0.0p-1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x00000.p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "0x0000.00000p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "0x.00000p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "0x0.p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "0x0p1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x0p1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "0x00000.p1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "0x0000.00000p1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "0x.00000p1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "0x0.p1234".parse::<Double>().unwrap().to_f64());
}

#[test]
fn from_decimal_string() {
    assert_eq!(1.0, "1".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.0, "2.".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.5, ".5".parse::<Double>().unwrap().to_f64());
    assert_eq!(1.0, "1.0".parse::<Double>().unwrap().to_f64());
    assert_eq!(-2.0, "-2".parse::<Double>().unwrap().to_f64());
    assert_eq!(-4.0, "-4.".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.5, "-.5".parse::<Double>().unwrap().to_f64());
    assert_eq!(-1.5, "-1.5".parse::<Double>().unwrap().to_f64());
    assert_eq!(1.25e12, "1.25e12".parse::<Double>().unwrap().to_f64());
    assert_eq!(1.25e+12, "1.25e+12".parse::<Double>().unwrap().to_f64());
    assert_eq!(1.25e-12, "1.25e-12".parse::<Double>().unwrap().to_f64());
    assert_eq!(1024.0, "1024.".parse::<Double>().unwrap().to_f64());
    assert_eq!(1024.05, "1024.05000".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.05, ".05000".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.0, "2.".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.0e2, "2.e2".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.0e+2, "2.e+2".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.0e-2, "2.e-2".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.05e2, "002.05000e2".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.05e+2, "002.05000e+2".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.05e-2, "002.05000e-2".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.05e12, "002.05000e12".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.05e+12, "002.05000e+12".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.05e-12, "002.05000e-12".parse::<Double>().unwrap().to_f64());

    assert_eq!(1.0, "1e".parse::<Double>().unwrap().to_f64());
    assert_eq!(1.0, "+1e".parse::<Double>().unwrap().to_f64());
    assert_eq!(-1.0, "-1e".parse::<Double>().unwrap().to_f64());

    assert_eq!(1.0, "1.e".parse::<Double>().unwrap().to_f64());
    assert_eq!(1.0, "+1.e".parse::<Double>().unwrap().to_f64());
    assert_eq!(-1.0, "-1.e".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.1, ".1e".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.1, "+.1e".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.1, "-.1e".parse::<Double>().unwrap().to_f64());

    assert_eq!(1.1, "1.1e".parse::<Double>().unwrap().to_f64());
    assert_eq!(1.1, "+1.1e".parse::<Double>().unwrap().to_f64());
    assert_eq!(-1.1, "-1.1e".parse::<Double>().unwrap().to_f64());

    assert_eq!(1.0, "1e+".parse::<Double>().unwrap().to_f64());
    assert_eq!(1.0, "1e-".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.1, ".1e".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.1, ".1e+".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.1, ".1e-".parse::<Double>().unwrap().to_f64());

    assert_eq!(1.0, "1.0e".parse::<Double>().unwrap().to_f64());
    assert_eq!(1.0, "1.0e+".parse::<Double>().unwrap().to_f64());
    assert_eq!(1.0, "1.0e-".parse::<Double>().unwrap().to_f64());

    // These are "carefully selected" to overflow the fast log-base
    // calculations in the implementation.
    assert!("99e99999".parse::<Double>().unwrap().is_infinite());
    assert!("-99e99999".parse::<Double>().unwrap().is_infinite());
    assert!("1e-99999".parse::<Double>().unwrap().is_pos_zero());
    assert!("-1e-99999".parse::<Double>().unwrap().is_neg_zero());

    assert_eq!(2.71828, "2.71828".parse::<Double>().unwrap().to_f64());
}

#[test]
fn from_string_specials() {
    let precision = 53;
    let payload_bits = precision - 2;
    let payload_mask = (1 << payload_bits) - 1;

    let mut nan_payloads = [
        0,
        1,
        123,
        0xDEADBEEF,
        -2i32 as u128,
        1 << payload_bits,       // overflow bit
        1 << (payload_bits - 1), // signaling bit
        1 << (payload_bits - 2), // highest possible bit
    ];

    // Convert payload integer to decimal string representation.
    let nan_payload_dec_strings: Vec<_> = nan_payloads.iter().map(|payload| format!("{payload}")).collect();

    // Convert payload integer to hexadecimal string representation.
    let nan_payload_hex_strings: Vec<_> = nan_payloads.iter().map(|payload| format!("{payload:#x}")).collect();

    // Fix payloads to expected result.
    for payload in &mut nan_payloads {
        *payload &= payload_mask;
    }

    // Signaling NaN must have a non-zero payload. In case a zero payload is
    // requested, a default arbitrary payload is set instead. Save this payload
    // for testing.
    let snan_default_payload = Double::snan(None).to_bits() & payload_mask;

    // Negative sign prefix (or none - for positive).
    let signs = ["", "-"];

    // "Signaling" prefix (or none - for "Quiet").
    let nan_types = ["", "s", "S"];

    let nan_strings = ["nan", "NaN"];
    for nan_str in nan_strings {
        for type_str in nan_types {
            let signaling = matches!(type_str, "s" | "S");

            for j in 0..nan_payloads.len() {
                let payload = if signaling && nan_payloads[j] == 0 {
                    snan_default_payload
                } else {
                    nan_payloads[j]
                };
                let payload_dec = &nan_payload_dec_strings[j];
                let payload_hex = &nan_payload_hex_strings[j];

                for sign_str in signs {
                    let negative = sign_str == "-";

                    let prefix = format!("{sign_str}{type_str}{nan_str}");

                    let test_strings = [
                        // Test without any paylod.
                        (payload == 0).then(|| prefix.clone()),
                        // Test with the payload as a suffix.
                        Some(format!("{prefix}{payload_dec}")),
                        Some(format!("{prefix}{payload_hex}")),
                        // Test with the payload inside parentheses.
                        Some(format!("{prefix}({payload_dec})")),
                        Some(format!("{prefix}({payload_hex})")),
                    ]
                    .into_iter()
                    .flatten();

                    for test_str in test_strings {
                        let f = test_str
                            .parse::<Double>()
                            .map_err(|e| format!("{test_str:?}: {e:?}"))
                            .unwrap();
                        assert!(f.is_nan());
                        assert_eq!(signaling, f.is_signaling());
                        assert_eq!(negative, f.is_negative());
                        assert_eq!(payload, f.to_bits() & payload_mask);
                    }
                }
            }
        }
    }

    let inf_strings = ["inf", "INFINITY", "+Inf", "-inf", "-INFINITY", "-Inf"];
    for &inf_str in &inf_strings {
        let negative = inf_str.starts_with('-');

        let f = inf_str.parse::<Double>().unwrap();
        assert!(f.is_infinite());
        assert_eq!(negative, f.is_negative());
        assert_eq!(0, f.to_bits() & payload_mask);
    }
}

#[test]
fn from_to_string_specials() {
    assert_eq!("+Inf", "+Inf".parse::<Double>().unwrap().to_string());
    assert_eq!("+Inf", "INFINITY".parse::<Double>().unwrap().to_string());
    assert_eq!("+Inf", "inf".parse::<Double>().unwrap().to_string());
    assert_eq!("-Inf", "-Inf".parse::<Double>().unwrap().to_string());
    assert_eq!("-Inf", "-INFINITY".parse::<Double>().unwrap().to_string());
    assert_eq!("-Inf", "-inf".parse::<Double>().unwrap().to_string());
    assert_eq!("NaN", "NaN".parse::<Double>().unwrap().to_string());
    assert_eq!("NaN", "nan".parse::<Double>().unwrap().to_string());
    assert_eq!("NaN", "-NaN".parse::<Double>().unwrap().to_string());
    assert_eq!("NaN", "-nan".parse::<Double>().unwrap().to_string());
}

#[test]
fn from_hexadecimal_string() {
    assert_eq!(1.0, "0x1p0".parse::<Double>().unwrap().to_f64());
    assert_eq!(1.0, "+0x1p0".parse::<Double>().unwrap().to_f64());
    assert_eq!(-1.0, "-0x1p0".parse::<Double>().unwrap().to_f64());

    assert_eq!(1.0, "0x1p+0".parse::<Double>().unwrap().to_f64());
    assert_eq!(1.0, "+0x1p+0".parse::<Double>().unwrap().to_f64());
    assert_eq!(-1.0, "-0x1p+0".parse::<Double>().unwrap().to_f64());

    assert_eq!(1.0, "0x1p-0".parse::<Double>().unwrap().to_f64());
    assert_eq!(1.0, "+0x1p-0".parse::<Double>().unwrap().to_f64());
    assert_eq!(-1.0, "-0x1p-0".parse::<Double>().unwrap().to_f64());

    assert_eq!(2.0, "0x1p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.0, "+0x1p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-2.0, "-0x1p1".parse::<Double>().unwrap().to_f64());

    assert_eq!(2.0, "0x1p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.0, "+0x1p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-2.0, "-0x1p+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.5, "0x1p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.5, "+0x1p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.5, "-0x1p-1".parse::<Double>().unwrap().to_f64());

    assert_eq!(3.0, "0x1.8p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(3.0, "+0x1.8p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-3.0, "-0x1.8p1".parse::<Double>().unwrap().to_f64());

    assert_eq!(3.0, "0x1.8p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(3.0, "+0x1.8p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-3.0, "-0x1.8p+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.75, "0x1.8p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.75, "+0x1.8p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.75, "-0x1.8p-1".parse::<Double>().unwrap().to_f64());

    assert_eq!(8192.0, "0x1000.000p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(8192.0, "+0x1000.000p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-8192.0, "-0x1000.000p1".parse::<Double>().unwrap().to_f64());

    assert_eq!(8192.0, "0x1000.000p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(8192.0, "+0x1000.000p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-8192.0, "-0x1000.000p+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(2048.0, "0x1000.000p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(2048.0, "+0x1000.000p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-2048.0, "-0x1000.000p-1".parse::<Double>().unwrap().to_f64());

    assert_eq!(8192.0, "0x1000p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(8192.0, "+0x1000p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-8192.0, "-0x1000p1".parse::<Double>().unwrap().to_f64());

    assert_eq!(8192.0, "0x1000p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(8192.0, "+0x1000p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-8192.0, "-0x1000p+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(2048.0, "0x1000p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(2048.0, "+0x1000p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-2048.0, "-0x1000p-1".parse::<Double>().unwrap().to_f64());

    assert_eq!(16384.0, "0x10p10".parse::<Double>().unwrap().to_f64());
    assert_eq!(16384.0, "+0x10p10".parse::<Double>().unwrap().to_f64());
    assert_eq!(-16384.0, "-0x10p10".parse::<Double>().unwrap().to_f64());

    assert_eq!(16384.0, "0x10p+10".parse::<Double>().unwrap().to_f64());
    assert_eq!(16384.0, "+0x10p+10".parse::<Double>().unwrap().to_f64());
    assert_eq!(-16384.0, "-0x10p+10".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.015625, "0x10p-10".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.015625, "+0x10p-10".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.015625, "-0x10p-10".parse::<Double>().unwrap().to_f64());

    assert_eq!(1.0625, "0x1.1p0".parse::<Double>().unwrap().to_f64());
    assert_eq!(1.0, "0x1p0".parse::<Double>().unwrap().to_f64());

    assert_eq!(
        "0x1p-150".parse::<Double>().unwrap().to_f64(),
        "+0x800000000000000001.p-221".parse::<Double>().unwrap().to_f64()
    );
    assert_eq!(2251799813685248.5, "0x80000000000004000000.010p-28".parse::<Double>().unwrap().to_f64());
}

#[test]
fn to_string() {
    let to_string = |d: f64, precision: usize, width: usize| {
        let x = Double::from_f64(d);
        if precision == 0 {
            format!("{:1$}", x, width)
        } else {
            format!("{:2$.1$}", x, precision, width)
        }
    };
    assert_eq!("10", to_string(10.0, 6, 3));
    assert_eq!("1.0E+1", to_string(10.0, 6, 0));
    assert_eq!("10100", to_string(1.01E+4, 5, 2));
    assert_eq!("1.01E+4", to_string(1.01E+4, 4, 2));
    assert_eq!("1.01E+4", to_string(1.01E+4, 5, 1));
    assert_eq!("0.0101", to_string(1.01E-2, 5, 2));
    assert_eq!("0.0101", to_string(1.01E-2, 4, 2));
    assert_eq!("1.01E-2", to_string(1.01E-2, 5, 1));
    assert_eq!("0.78539816339744828", to_string(0.78539816339744830961, 0, 3));
    assert_eq!("4.9406564584124654E-324", to_string(4.9406564584124654e-324, 0, 3));
    assert_eq!("873.18340000000001", to_string(873.1834, 0, 1));
    assert_eq!("8.7318340000000001E+2", to_string(873.1834, 0, 0));
    assert_eq!("1.7976931348623157E+308", to_string(1.7976931348623157E+308, 0, 0));

    let to_string = |d: f64, precision: usize, width: usize| {
        let x = Double::from_f64(d);
        if precision == 0 {
            format!("{:#1$}", x, width)
        } else {
            format!("{:#2$.1$}", x, precision, width)
        }
    };
    assert_eq!("10", to_string(10.0, 6, 3));
    assert_eq!("1.000000e+01", to_string(10.0, 6, 0));
    assert_eq!("10100", to_string(1.01E+4, 5, 2));
    assert_eq!("1.0100e+04", to_string(1.01E+4, 4, 2));
    assert_eq!("1.01000e+04", to_string(1.01E+4, 5, 1));
    assert_eq!("0.0101", to_string(1.01E-2, 5, 2));
    assert_eq!("0.0101", to_string(1.01E-2, 4, 2));
    assert_eq!("1.01000e-02", to_string(1.01E-2, 5, 1));
    assert_eq!("0.78539816339744828", to_string(0.78539816339744830961, 0, 3));
    assert_eq!("4.94065645841246540e-324", to_string(4.9406564584124654e-324, 0, 3));
    assert_eq!("873.18340000000001", to_string(873.1834, 0, 1));
    assert_eq!("8.73183400000000010e+02", to_string(873.1834, 0, 0));
    assert_eq!("1.79769313486231570e+308", to_string(1.7976931348623157E+308, 0, 0));
    assert_eq!("NaN", X87DoubleExtended::from_bits(1 << 64).to_string());
}

#[test]
fn to_integer() {
    let mut is_exact = false;

    assert_eq!(
        Status::OK.and(10),
        "10".parse::<Double>()
            .unwrap()
            .to_u128_r(5, Round::TowardZero, &mut is_exact,)
    );
    assert!(is_exact);

    assert_eq!(
        Status::INVALID_OP.and(0),
        "-10"
            .parse::<Double>()
            .unwrap()
            .to_u128_r(5, Round::TowardZero, &mut is_exact,)
    );
    assert!(!is_exact);

    assert_eq!(
        Status::INVALID_OP.and(31),
        "32".parse::<Double>()
            .unwrap()
            .to_u128_r(5, Round::TowardZero, &mut is_exact,)
    );
    assert!(!is_exact);

    assert_eq!(
        Status::INEXACT.and(7),
        "7.9"
            .parse::<Double>()
            .unwrap()
            .to_u128_r(5, Round::TowardZero, &mut is_exact,)
    );
    assert!(!is_exact);

    assert_eq!(
        Status::OK.and(-10),
        "-10"
            .parse::<Double>()
            .unwrap()
            .to_i128_r(5, Round::TowardZero, &mut is_exact,)
    );
    assert!(is_exact);

    assert_eq!(
        Status::INVALID_OP.and(-16),
        "-17"
            .parse::<Double>()
            .unwrap()
            .to_i128_r(5, Round::TowardZero, &mut is_exact,)
    );
    assert!(!is_exact);

    assert_eq!(
        Status::INVALID_OP.and(15),
        "16".parse::<Double>()
            .unwrap()
            .to_i128_r(5, Round::TowardZero, &mut is_exact,)
    );
    assert!(!is_exact);
}

#[test]
fn nan() {
    fn nanbits_from_u128<F: Float>(signaling: bool, negative: bool, payload: u128) -> u128 {
        let x = if signaling {
            F::snan(Some(payload))
        } else {
            F::qnan(Some(payload))
        };
        if negative {
            (-x).to_bits()
        } else {
            x.to_bits()
        }
    }

    let tests_single = [
        // expected   SNaN    Neg     payload
        (0x7fc00000, false, false, 0x00000000),
        (0xffc00000, false, true, 0x00000000),
        (0x7fc0ae72, false, false, 0x0000ae72),
        (0x7fffae72, false, false, 0xffffae72),
        (0x7fdaae72, false, false, 0x00daae72),
        (0x7fa00000, true, false, 0x00000000),
        (0xffa00000, true, true, 0x00000000),
        (0x7f80ae72, true, false, 0x0000ae72),
        (0x7fbfae72, true, false, 0xffffae72),
        (0x7f9aae72, true, false, 0x001aae72),
    ];
    let tests_double = [
        //         expected   SNaN    Neg             payload
        (0x7ff8000000000000, false, false, 0x0000000000000000),
        (0xfff8000000000000, false, true, 0x0000000000000000),
        (0x7ff800000000ae72, false, false, 0x000000000000ae72),
        (0x7fffffffffffae72, false, false, 0xffffffffffffae72),
        (0x7ffdaaaaaaaaae72, false, false, 0x000daaaaaaaaae72),
        (0x7ff4000000000000, true, false, 0x0000000000000000),
        (0xfff4000000000000, true, true, 0x0000000000000000),
        (0x7ff000000000ae72, true, false, 0x000000000000ae72),
        (0x7ff7ffffffffae72, true, false, 0xffffffffffffae72),
        (0x7ff1aaaaaaaaae72, true, false, 0x0001aaaaaaaaae72),
    ];
    for (expected, signaling, negative, payload) in tests_single {
        assert_eq!(expected, nanbits_from_u128::<Single>(signaling, negative, payload));
    }
    for (expected, signaling, negative, payload) in tests_double {
        assert_eq!(expected, nanbits_from_u128::<Double>(signaling, negative, payload));
    }
}

#[test]
fn string_decimal_error() {
    assert_eq!("Invalid string length", "".parse::<Double>().unwrap_err().0);
    assert_eq!("String has no digits", "+".parse::<Double>().unwrap_err().0);
    assert_eq!("String has no digits", "-".parse::<Double>().unwrap_err().0);

    assert_eq!("Invalid character in significand", "\0".parse::<Double>().unwrap_err().0);
    assert_eq!("Invalid character in significand", "1\0".parse::<Double>().unwrap_err().0);
    assert_eq!("Invalid character in significand", "1\02".parse::<Double>().unwrap_err().0);
    assert_eq!("Invalid character in significand", "1\02e1".parse::<Double>().unwrap_err().0);
    assert_eq!("Invalid character in exponent", "1e\0".parse::<Double>().unwrap_err().0);
    assert_eq!("Invalid character in exponent", "1e1\0".parse::<Double>().unwrap_err().0);
    assert_eq!("Invalid character in exponent", "1e1\02".parse::<Double>().unwrap_err().0);

    assert_eq!("Invalid character in significand", "1.0f".parse::<Double>().unwrap_err().0);

    assert_eq!("String contains multiple dots", "..".parse::<Double>().unwrap_err().0);
    assert_eq!("String contains multiple dots", "..0".parse::<Double>().unwrap_err().0);
    assert_eq!("String contains multiple dots", "1.0.0".parse::<Double>().unwrap_err().0);
}

#[test]
fn string_decimal_significand_error() {
    assert_eq!("Significand has no digits", ".".parse::<Double>().unwrap_err().0);
    assert_eq!("Significand has no digits", "+.".parse::<Double>().unwrap_err().0);
    assert_eq!("Significand has no digits", "-.".parse::<Double>().unwrap_err().0);

    assert_eq!("Significand has no digits", "e".parse::<Double>().unwrap_err().0);
    assert_eq!("Significand has no digits", "+e".parse::<Double>().unwrap_err().0);
    assert_eq!("Significand has no digits", "-e".parse::<Double>().unwrap_err().0);

    assert_eq!("Significand has no digits", "e1".parse::<Double>().unwrap_err().0);
    assert_eq!("Significand has no digits", "+e1".parse::<Double>().unwrap_err().0);
    assert_eq!("Significand has no digits", "-e1".parse::<Double>().unwrap_err().0);

    assert_eq!("Significand has no digits", ".e1".parse::<Double>().unwrap_err().0);
    assert_eq!("Significand has no digits", "+.e1".parse::<Double>().unwrap_err().0);
    assert_eq!("Significand has no digits", "-.e1".parse::<Double>().unwrap_err().0);

    assert_eq!("Significand has no digits", ".e".parse::<Double>().unwrap_err().0);
    assert_eq!("Significand has no digits", "+.e".parse::<Double>().unwrap_err().0);
    assert_eq!("Significand has no digits", "-.e".parse::<Double>().unwrap_err().0);
}

#[test]
fn string_hexadecimal_error() {
    assert_eq!("Invalid string", "0x".parse::<Double>().unwrap_err().0);
    assert_eq!("Invalid string", "+0x".parse::<Double>().unwrap_err().0);
    assert_eq!("Invalid string", "-0x".parse::<Double>().unwrap_err().0);

    assert_eq!("Hex strings require an exponent", "0x0".parse::<Double>().unwrap_err().0);
    assert_eq!("Hex strings require an exponent", "+0x0".parse::<Double>().unwrap_err().0);
    assert_eq!("Hex strings require an exponent", "-0x0".parse::<Double>().unwrap_err().0);

    assert_eq!("Hex strings require an exponent", "0x0.".parse::<Double>().unwrap_err().0);
    assert_eq!("Hex strings require an exponent", "+0x0.".parse::<Double>().unwrap_err().0);
    assert_eq!("Hex strings require an exponent", "-0x0.".parse::<Double>().unwrap_err().0);

    assert_eq!("Hex strings require an exponent", "0x.0".parse::<Double>().unwrap_err().0);
    assert_eq!("Hex strings require an exponent", "+0x.0".parse::<Double>().unwrap_err().0);
    assert_eq!("Hex strings require an exponent", "-0x.0".parse::<Double>().unwrap_err().0);

    assert_eq!("Hex strings require an exponent", "0x0.0".parse::<Double>().unwrap_err().0);
    assert_eq!("Hex strings require an exponent", "+0x0.0".parse::<Double>().unwrap_err().0);
    assert_eq!("Hex strings require an exponent", "-0x0.0".parse::<Double>().unwrap_err().0);

    assert_eq!("Invalid character in significand", "0x\0".parse::<Double>().unwrap_err().0);
    assert_eq!("Invalid character in significand", "0x1\0".parse::<Double>().unwrap_err().0);
    assert_eq!("Invalid character in significand", "0x1\02".parse::<Double>().unwrap_err().0);
    assert_eq!("Invalid character in significand", "0x1\02p1".parse::<Double>().unwrap_err().0);
    assert_eq!("Invalid character in exponent", "0x1p\0".parse::<Double>().unwrap_err().0);
    assert_eq!("Invalid character in exponent", "0x1p1\0".parse::<Double>().unwrap_err().0);
    assert_eq!("Invalid character in exponent", "0x1p1\02".parse::<Double>().unwrap_err().0);

    assert_eq!("Invalid character in exponent", "0x1p0f".parse::<Double>().unwrap_err().0);

    assert_eq!("String contains multiple dots", "0x..p1".parse::<Double>().unwrap_err().0);
    assert_eq!("String contains multiple dots", "0x..0p1".parse::<Double>().unwrap_err().0);
    assert_eq!("String contains multiple dots", "0x1.0.0p1".parse::<Double>().unwrap_err().0);
}

#[test]
fn string_hexadecimal_significand_error() {
    assert_eq!("Significand has no digits", "0x.".parse::<Double>().unwrap_err().0);
    assert_eq!("Significand has no digits", "+0x.".parse::<Double>().unwrap_err().0);
    assert_eq!("Significand has no digits", "-0x.".parse::<Double>().unwrap_err().0);

    assert_eq!("Significand has no digits", "0xp".parse::<Double>().unwrap_err().0);
    assert_eq!("Significand has no digits", "+0xp".parse::<Double>().unwrap_err().0);
    assert_eq!("Significand has no digits", "-0xp".parse::<Double>().unwrap_err().0);

    assert_eq!("Significand has no digits", "0xp+".parse::<Double>().unwrap_err().0);
    assert_eq!("Significand has no digits", "+0xp+".parse::<Double>().unwrap_err().0);
    assert_eq!("Significand has no digits", "-0xp+".parse::<Double>().unwrap_err().0);

    assert_eq!("Significand has no digits", "0xp-".parse::<Double>().unwrap_err().0);
    assert_eq!("Significand has no digits", "+0xp-".parse::<Double>().unwrap_err().0);
    assert_eq!("Significand has no digits", "-0xp-".parse::<Double>().unwrap_err().0);

    assert_eq!("Significand has no digits", "0x.p".parse::<Double>().unwrap_err().0);
    assert_eq!("Significand has no digits", "+0x.p".parse::<Double>().unwrap_err().0);
    assert_eq!("Significand has no digits", "-0x.p".parse::<Double>().unwrap_err().0);

    assert_eq!("Significand has no digits", "0x.p+".parse::<Double>().unwrap_err().0);
    assert_eq!("Significand has no digits", "+0x.p+".parse::<Double>().unwrap_err().0);
    assert_eq!("Significand has no digits", "-0x.p+".parse::<Double>().unwrap_err().0);

    assert_eq!("Significand has no digits", "0x.p-".parse::<Double>().unwrap_err().0);
    assert_eq!("Significand has no digits", "+0x.p-".parse::<Double>().unwrap_err().0);
    assert_eq!("Significand has no digits", "-0x.p-".parse::<Double>().unwrap_err().0);
}

#[test]
fn string_hexadecimal_exponent_error() {
    assert_eq!("Exponent has no digits", "0x1p".parse::<Double>().unwrap_err().0);
    assert_eq!("Exponent has no digits", "+0x1p".parse::<Double>().unwrap_err().0);
    assert_eq!("Exponent has no digits", "-0x1p".parse::<Double>().unwrap_err().0);

    assert_eq!("Exponent has no digits", "0x1p+".parse::<Double>().unwrap_err().0);
    assert_eq!("Exponent has no digits", "+0x1p+".parse::<Double>().unwrap_err().0);
    assert_eq!("Exponent has no digits", "-0x1p+".parse::<Double>().unwrap_err().0);

    assert_eq!("Exponent has no digits", "0x1p-".parse::<Double>().unwrap_err().0);
    assert_eq!("Exponent has no digits", "+0x1p-".parse::<Double>().unwrap_err().0);
    assert_eq!("Exponent has no digits", "-0x1p-".parse::<Double>().unwrap_err().0);

    assert_eq!("Exponent has no digits", "0x1.p".parse::<Double>().unwrap_err().0);
    assert_eq!("Exponent has no digits", "+0x1.p".parse::<Double>().unwrap_err().0);
    assert_eq!("Exponent has no digits", "-0x1.p".parse::<Double>().unwrap_err().0);

    assert_eq!("Exponent has no digits", "0x1.p+".parse::<Double>().unwrap_err().0);
    assert_eq!("Exponent has no digits", "+0x1.p+".parse::<Double>().unwrap_err().0);
    assert_eq!("Exponent has no digits", "-0x1.p+".parse::<Double>().unwrap_err().0);

    assert_eq!("Exponent has no digits", "0x1.p-".parse::<Double>().unwrap_err().0);
    assert_eq!("Exponent has no digits", "+0x1.p-".parse::<Double>().unwrap_err().0);
    assert_eq!("Exponent has no digits", "-0x1.p-".parse::<Double>().unwrap_err().0);

    assert_eq!("Exponent has no digits", "0x.1p".parse::<Double>().unwrap_err().0);
    assert_eq!("Exponent has no digits", "+0x.1p".parse::<Double>().unwrap_err().0);
    assert_eq!("Exponent has no digits", "-0x.1p".parse::<Double>().unwrap_err().0);

    assert_eq!("Exponent has no digits", "0x.1p+".parse::<Double>().unwrap_err().0);
    assert_eq!("Exponent has no digits", "+0x.1p+".parse::<Double>().unwrap_err().0);
    assert_eq!("Exponent has no digits", "-0x.1p+".parse::<Double>().unwrap_err().0);

    assert_eq!("Exponent has no digits", "0x.1p-".parse::<Double>().unwrap_err().0);
    assert_eq!("Exponent has no digits", "+0x.1p-".parse::<Double>().unwrap_err().0);
    assert_eq!("Exponent has no digits", "-0x.1p-".parse::<Double>().unwrap_err().0);

    assert_eq!("Exponent has no digits", "0x1.1p".parse::<Double>().unwrap_err().0);
    assert_eq!("Exponent has no digits", "+0x1.1p".parse::<Double>().unwrap_err().0);
    assert_eq!("Exponent has no digits", "-0x1.1p".parse::<Double>().unwrap_err().0);

    assert_eq!("Exponent has no digits", "0x1.1p+".parse::<Double>().unwrap_err().0);
    assert_eq!("Exponent has no digits", "+0x1.1p+".parse::<Double>().unwrap_err().0);
    assert_eq!("Exponent has no digits", "-0x1.1p+".parse::<Double>().unwrap_err().0);

    assert_eq!("Exponent has no digits", "0x1.1p-".parse::<Double>().unwrap_err().0);
    assert_eq!("Exponent has no digits", "+0x1.1p-".parse::<Double>().unwrap_err().0);
    assert_eq!("Exponent has no digits", "-0x1.1p-".parse::<Double>().unwrap_err().0);
}

#[test]
fn exact_inverse() {
    // Trivial operation.
    assert!(Double::from_f64(2.0)
        .get_exact_inverse()
        .unwrap()
        .bitwise_eq(Double::from_f64(0.5)));
    assert!(Single::from_f32(2.0)
        .get_exact_inverse()
        .unwrap()
        .bitwise_eq(Single::from_f32(0.5)));
    assert!("2.0"
        .parse::<Quad>()
        .unwrap()
        .get_exact_inverse()
        .unwrap()
        .bitwise_eq("0.5".parse::<Quad>().unwrap()));
    assert!("2.0"
        .parse::<X87DoubleExtended>()
        .unwrap()
        .get_exact_inverse()
        .unwrap()
        .bitwise_eq("0.5".parse::<X87DoubleExtended>().unwrap()));

    // FLT_MIN
    assert!(Single::from_f32(1.17549435e-38)
        .get_exact_inverse()
        .unwrap()
        .bitwise_eq(Single::from_f32(8.5070592e+37)));

    // Large float, inverse is a denormal.
    assert!(Single::from_f32(1.7014118e38).get_exact_inverse().is_none());
    // Zero
    assert!(Double::from_f64(0.0).get_exact_inverse().is_none());
    // Denormalized float
    assert!(Single::from_f32(1.40129846e-45).get_exact_inverse().is_none());
}

#[test]
fn round_to_integral() {
    let t = Double::from_f64(-0.5);
    assert_eq!(-0.0, t.round_to_integral(Round::TowardZero).value.to_f64());
    assert_eq!(-1.0, t.round_to_integral(Round::TowardNegative).value.to_f64());
    assert_eq!(-0.0, t.round_to_integral(Round::TowardPositive).value.to_f64());
    assert_eq!(-0.0, t.round_to_integral(Round::NearestTiesToEven).value.to_f64());

    let s = Double::from_f64(3.14);
    assert_eq!(3.0, s.round_to_integral(Round::TowardZero).value.to_f64());
    assert_eq!(3.0, s.round_to_integral(Round::TowardNegative).value.to_f64());
    assert_eq!(4.0, s.round_to_integral(Round::TowardPositive).value.to_f64());
    assert_eq!(3.0, s.round_to_integral(Round::NearestTiesToEven).value.to_f64());

    let r = Double::largest();
    assert_eq!(r.to_f64(), r.round_to_integral(Round::TowardZero).value.to_f64());
    assert_eq!(r.to_f64(), r.round_to_integral(Round::TowardNegative).value.to_f64());
    assert_eq!(r.to_f64(), r.round_to_integral(Round::TowardPositive).value.to_f64());
    assert_eq!(r.to_f64(), r.round_to_integral(Round::NearestTiesToEven).value.to_f64());

    let p = Double::ZERO.round_to_integral(Round::TowardZero).value;
    assert_eq!(0.0, p.to_f64());
    let p = (-Double::ZERO).round_to_integral(Round::TowardZero).value;
    assert_eq!(-0.0, p.to_f64());
    let p = Double::NAN.round_to_integral(Round::TowardZero).value;
    assert!(p.to_f64().is_nan());
    let p = Double::INFINITY.round_to_integral(Round::TowardZero).value;
    assert!(p.to_f64().is_infinite() && p.to_f64() > 0.0);
    let p = (-Double::INFINITY).round_to_integral(Round::TowardZero).value;
    assert!(p.to_f64().is_infinite() && p.to_f64() < 0.0);

    let mut status;

    let p = unpack!(status=, Double::NAN.round_to_integral(Round::TowardZero));
    assert!(p.is_nan());
    assert!(!p.is_negative());
    assert_eq!(Status::OK, status);

    let p = unpack!(status=, (-Double::NAN).round_to_integral(Round::TowardZero));
    assert!(p.is_nan());
    assert!(p.is_negative());
    assert_eq!(Status::OK, status);

    let p = unpack!(status=, Double::snan(None).round_to_integral(Round::TowardZero));
    assert!(p.is_nan());
    assert!(!p.is_signaling());
    assert!(!p.is_negative());
    assert_eq!(Status::INVALID_OP, status);

    let p = unpack!(status=, (-Double::snan(None)).round_to_integral(Round::TowardZero));
    assert!(p.is_nan());
    assert!(!p.is_signaling());
    assert!(p.is_negative());
    assert_eq!(Status::INVALID_OP, status);

    let p = unpack!(status=, Double::INFINITY.round_to_integral(Round::TowardZero));
    assert!(p.is_infinite());
    assert!(!p.is_negative());
    assert_eq!(Status::OK, status);

    let p = unpack!(status=, (-Double::INFINITY).round_to_integral(Round::TowardZero));
    assert!(p.is_infinite());
    assert!(p.is_negative());
    assert_eq!(Status::OK, status);

    let p = unpack!(status=, Double::ZERO.round_to_integral(Round::TowardZero));
    assert!(p.is_zero());
    assert!(!p.is_negative());
    assert_eq!(Status::OK, status);

    let p = unpack!(status=, Double::ZERO.round_to_integral(Round::TowardNegative));
    assert!(p.is_zero());
    assert!(!p.is_negative());
    assert_eq!(Status::OK, status);

    let p = unpack!(status=, (-Double::ZERO).round_to_integral(Round::TowardZero));
    assert!(p.is_zero());
    assert!(p.is_negative());
    assert_eq!(Status::OK, status);

    let p = unpack!(status=, (-Double::ZERO).round_to_integral(Round::TowardNegative));
    assert!(p.is_zero());
    assert!(p.is_negative());
    assert_eq!(Status::OK, status);

    let p = unpack!(status=, Double::from_f64(1E-100).round_to_integral(Round::TowardNegative));
    assert!(p.is_zero());
    assert!(!p.is_negative());
    assert_eq!(Status::INEXACT, status);

    let p = unpack!(status=, Double::from_f64(1E-100).round_to_integral(Round::TowardPositive));
    assert_eq!(1.0, p.to_f64());
    assert!(!p.is_negative());
    assert_eq!(Status::INEXACT, status);

    let p = unpack!(status=, Double::from_f64(-1E-100).round_to_integral(Round::TowardNegative));
    assert!(p.is_negative());
    assert_eq!(-1.0, p.to_f64());
    assert_eq!(Status::INEXACT, status);

    let p = unpack!(status=, Double::from_f64(-1E-100).round_to_integral(Round::TowardPositive));
    assert!(p.is_zero());
    assert!(p.is_negative());
    assert_eq!(Status::INEXACT, status);

    let p = unpack!(status=, Double::from_f64(10.0).round_to_integral(Round::TowardZero));
    assert_eq!(10.0, p.to_f64());
    assert_eq!(Status::OK, status);

    let p = unpack!(status=, Double::from_f64(10.5).round_to_integral(Round::TowardZero));
    assert_eq!(10.0, p.to_f64());
    assert_eq!(Status::INEXACT, status);

    let p = unpack!(status=, Double::from_f64(10.5).round_to_integral(Round::TowardPositive));
    assert_eq!(11.0, p.to_f64());
    assert_eq!(Status::INEXACT, status);

    let p = unpack!(status=, Double::from_f64(10.5).round_to_integral(Round::TowardNegative));
    assert_eq!(10.0, p.to_f64());
    assert_eq!(Status::INEXACT, status);

    let p = unpack!(status=, Double::from_f64(10.5).round_to_integral(Round::NearestTiesToAway));
    assert_eq!(11.0, p.to_f64());
    assert_eq!(Status::INEXACT, status);

    let p = unpack!(status=, Double::from_f64(10.5).round_to_integral(Round::NearestTiesToEven));
    assert_eq!(10.0, p.to_f64());
    assert_eq!(Status::INEXACT, status);
}

#[test]
fn is_integer() {
    let t = Double::from_f64(-0.0);
    assert!(t.is_integer());
    let t = Double::from_f64(3.14159);
    assert!(!t.is_integer());
    let t = Double::NAN;
    assert!(!t.is_integer());
    let t = Double::INFINITY;
    assert!(!t.is_integer());
    let t = -Double::INFINITY;
    assert!(!t.is_integer());
    let t = Double::largest();
    assert!(t.is_integer());
}

#[test]
fn largest() {
    assert_eq!(3.402823466e+38, Single::largest().to_f32());
    assert_eq!(1.7976931348623158e+308, Double::largest().to_f64());
    assert_eq!(448.0, Float8E4M3FN::largest().to_f64());
}

#[test]
fn smallest() {
    let test = Single::SMALLEST;
    let expected = "0x0.000002p-126".parse::<Single>().unwrap();
    assert!(!test.is_negative());
    assert!(test.is_finite_non_zero());
    assert!(test.is_denormal());
    assert!(test.bitwise_eq(expected));

    let test = -Single::SMALLEST;
    let expected = "-0x0.000002p-126".parse::<Single>().unwrap();
    assert!(test.is_negative());
    assert!(test.is_finite_non_zero());
    assert!(test.is_denormal());
    assert!(test.bitwise_eq(expected));

    let test = Quad::SMALLEST;
    let expected = "0x0.0000000000000000000000000001p-16382".parse::<Quad>().unwrap();
    assert!(!test.is_negative());
    assert!(test.is_finite_non_zero());
    assert!(test.is_denormal());
    assert!(test.bitwise_eq(expected));

    let test = -Quad::SMALLEST;
    let expected = "-0x0.0000000000000000000000000001p-16382".parse::<Quad>().unwrap();
    assert!(test.is_negative());
    assert!(test.is_finite_non_zero());
    assert!(test.is_denormal());
    assert!(test.bitwise_eq(expected));
}

#[test]
fn smallest_normalized() {
    let test = Single::smallest_normalized();
    let expected = "0x1p-126".parse::<Single>().unwrap();
    assert!(!test.is_negative());
    assert!(test.is_finite_non_zero());
    assert!(!test.is_denormal());
    assert!(test.bitwise_eq(expected));
    assert!(test.is_smallest_normalized());

    let test = -Single::smallest_normalized();
    let expected = "-0x1p-126".parse::<Single>().unwrap();
    assert!(test.is_negative());
    assert!(test.is_finite_non_zero());
    assert!(!test.is_denormal());
    assert!(test.bitwise_eq(expected));
    assert!(test.is_smallest_normalized());

    let test = Double::smallest_normalized();
    let expected = "0x1p-1022".parse::<Double>().unwrap();
    assert!(!test.is_negative());
    assert!(test.is_finite_non_zero());
    assert!(!test.is_denormal());
    assert!(test.bitwise_eq(expected));
    assert!(test.is_smallest_normalized());

    let test = -Double::smallest_normalized();
    let expected = "-0x1p-1022".parse::<Double>().unwrap();
    assert!(test.is_negative());
    assert!(test.is_finite_non_zero());
    assert!(!test.is_denormal());
    assert!(test.bitwise_eq(expected));
    assert!(test.is_smallest_normalized());

    let test = Quad::smallest_normalized();
    let expected = "0x1p-16382".parse::<Quad>().unwrap();
    assert!(!test.is_negative());
    assert!(test.is_finite_non_zero());
    assert!(!test.is_denormal());
    assert!(test.bitwise_eq(expected));
    assert!(test.is_smallest_normalized());

    let test = -Quad::smallest_normalized();
    let expected = "-0x1p-16382".parse::<Quad>().unwrap();
    assert!(test.is_negative());
    assert!(test.is_finite_non_zero());
    assert!(!test.is_denormal());
    assert!(test.bitwise_eq(expected));
    assert!(test.is_smallest_normalized());
}

#[test]
fn zero() {
    assert_eq!(0.0, Single::from_f32(0.0).to_f32());
    assert_eq!(-0.0, Single::from_f32(-0.0).to_f32());
    assert!(Single::from_f32(-0.0).is_negative());

    assert_eq!(0.0, Double::from_f64(0.0).to_f64());
    assert_eq!(-0.0, Double::from_f64(-0.0).to_f64());
    assert!(Double::from_f64(-0.0).is_negative());

    fn test<F: Float>(sign: bool, bits: u128) {
        let test = if sign { -F::ZERO } else { F::ZERO };
        let pattern = if sign { "-0x0p+0" } else { "0x0p+0" };
        let expected = pattern.parse::<F>().unwrap();
        assert!(test.is_zero());
        assert_eq!(sign, test.is_negative());
        assert!(test.bitwise_eq(expected));
        assert_eq!(bits, test.to_bits());
    }
    test::<Half>(false, 0);
    test::<Half>(true, 0x8000);
    test::<Single>(false, 0);
    test::<Single>(true, 0x80000000);
    test::<Double>(false, 0);
    test::<Double>(true, 0x8000000000000000);
    test::<Quad>(false, 0);
    test::<Quad>(true, 0x8000000000000000_0000000000000000);
    test::<X87DoubleExtended>(false, 0);
    test::<X87DoubleExtended>(true, 0x8000_0000000000000000);
    test::<Float8E5M2>(false, 0);
    test::<Float8E5M2>(true, 0x80);
    test::<Float8E4M3FN>(false, 0);
    test::<Float8E4M3FN>(true, 0x80);
}

#[test]
fn copy_sign() {
    assert!(Double::from_f64(-42.0).bitwise_eq(Double::from_f64(42.0).copy_sign(Double::from_f64(-1.0),),));
    assert!(Double::from_f64(42.0).bitwise_eq(Double::from_f64(-42.0).copy_sign(Double::from_f64(1.0),),));
    assert!(Double::from_f64(-42.0).bitwise_eq(Double::from_f64(-42.0).copy_sign(Double::from_f64(-1.0),),));
    assert!(Double::from_f64(42.0).bitwise_eq(Double::from_f64(42.0).copy_sign(Double::from_f64(1.0),),));
}

#[test]
fn convert() {
    let mut loses_info = false;
    let mut status;

    let test = "1.0".parse::<Double>().unwrap();
    let test: Single = test.convert(&mut loses_info).value;
    assert_eq!(1.0, test.to_f32());
    assert!(!loses_info);

    let mut test = "0x1p-53".parse::<X87DoubleExtended>().unwrap();
    let one = "1.0".parse::<X87DoubleExtended>().unwrap();
    test += one;
    let test: Double = test.convert(&mut loses_info).value;
    assert_eq!(1.0, test.to_f64());
    assert!(loses_info);

    let mut test = "0x1p-53".parse::<Quad>().unwrap();
    let one = "1.0".parse::<Quad>().unwrap();
    test += one;
    let test: Double = test.convert(&mut loses_info).value;
    assert_eq!(1.0, test.to_f64());
    assert!(loses_info);

    let test = "0xf.fffffffp+28".parse::<X87DoubleExtended>().unwrap();
    let test: Double = test.convert(&mut loses_info).value;
    assert_eq!(4294967295.0, test.to_f64());
    assert!(!loses_info);

    let test = Single::snan(None);
    let test: X87DoubleExtended = unpack!(status=, test.convert(&mut loses_info));
    // Conversion quiets the SNAN, so now 2 bits of the 64-bit significand should be set.
    assert!(test.bitwise_eq(X87DoubleExtended::qnan(Some(0x6000000000000000))));
    assert!(!loses_info);
    assert_eq!(status, Status::INVALID_OP);

    let test = Single::qnan(None);
    let x87_qnan = X87DoubleExtended::qnan(None);
    let test: X87DoubleExtended = test.convert(&mut loses_info).value;
    assert!(test.bitwise_eq(x87_qnan));
    assert!(!loses_info);

    // NOTE(eddyb) these were mistakenly noops upstream, here they're already
    // fixed (by instead converting from `Double` to `X87DoubleExtended`),
    // see also upstream issue https://github.com/llvm/llvm-project/issues/63842.
    let test = Double::snan(None);
    let test: X87DoubleExtended = test.convert(&mut loses_info).value;
    // Conversion quiets the SNAN, so now 2 bits of the 64-bit significand should be set.
    assert!(test.bitwise_eq(X87DoubleExtended::qnan(Some(0x6000000000000000))));
    assert!(!loses_info);

    let test = Double::qnan(None);
    let test: X87DoubleExtended = test.convert(&mut loses_info).value;
    assert!(test.bitwise_eq(x87_qnan));
    assert!(!loses_info);

    // The payload is lost in truncation, but we retain NaN by setting the quiet bit.
    let test = Double::snan(Some(1));
    let test: Single = unpack!(status=, test.convert(&mut loses_info));
    assert_eq!(0x7fc00000, test.to_bits());
    assert!(loses_info);
    assert_eq!(status, Status::INVALID_OP);

    // The payload is lost in truncation. QNaN remains QNaN.
    let test = Double::qnan(Some(1));
    let test: Single = unpack!(status=, test.convert(&mut loses_info));
    assert_eq!(0x7fc00000, test.to_bits());
    assert!(loses_info);
    assert_eq!(status, Status::OK);

    // Test that subnormals are handled correctly in double to float conversion
    let test = "0x0.0000010000000p-1022".parse::<Double>().unwrap();
    let test: Single = unpack!(status=, test.convert(&mut loses_info));
    assert_eq!(0.0, test.to_f32());
    assert!(loses_info);

    let test = "0x0.0000010000001p-1022".parse::<Double>().unwrap();
    let test: Single = unpack!(status=, test.convert(&mut loses_info));
    assert_eq!(0.0, test.to_f32());
    assert!(loses_info);

    let test = "-0x0.0000010000001p-1022".parse::<Double>().unwrap();
    let test: Single = unpack!(status=, test.convert(&mut loses_info));
    assert_eq!(0.0, test.to_f32());
    assert!(loses_info);

    let test = "0x0.0000020000000p-1022".parse::<Double>().unwrap();
    let test: Single = unpack!(status=, test.convert(&mut loses_info));
    assert_eq!(0.0, test.to_f32());
    assert!(loses_info);

    let test = "0x0.0000020000001p-1022".parse::<Double>().unwrap();
    let test: Single = unpack!(status=, test.convert(&mut loses_info));
    assert_eq!(0.0, test.to_f32());
    assert!(loses_info);

    // Test subnormal conversion to bfloat
    let test = "0x0.01p-126".parse::<Single>().unwrap();
    let test: BFloat = unpack!(status=, test.convert(&mut loses_info));
    assert_eq!(0.0, test.to_f32());
    assert!(loses_info);

    let test = "0x0.02p-126".parse::<Single>().unwrap();
    let test: BFloat = unpack!(status=, test.convert(&mut loses_info));
    assert_eq!(0x01, test.to_bits());
    assert!(!loses_info);

    let test = "0x0.01p-126".parse::<Single>().unwrap();
    let test: BFloat = unpack!(status=, test.convert_r(Round::NearestTiesToAway, &mut loses_info));
    assert_eq!(0x01, test.to_bits());
    assert!(loses_info);
}

#[test]
fn is_negative() {
    let t = "0x1p+0".parse::<Single>().unwrap();
    assert!(!t.is_negative());
    let t = "-0x1p+0".parse::<Single>().unwrap();
    assert!(t.is_negative());

    assert!(!Single::INFINITY.is_negative());
    assert!((-Single::INFINITY).is_negative());

    assert!(!Single::ZERO.is_negative());
    assert!((-Single::ZERO).is_negative());

    assert!(!Single::NAN.is_negative());
    assert!((-Single::NAN).is_negative());

    assert!(!Single::snan(None).is_negative());
    assert!((-Single::snan(None)).is_negative());
}

#[test]
fn is_normal() {
    let t = "0x1p+0".parse::<Single>().unwrap();
    assert!(t.is_normal());

    assert!(!Single::INFINITY.is_normal());
    assert!(!Single::ZERO.is_normal());
    assert!(!Single::NAN.is_normal());
    assert!(!Single::snan(None).is_normal());
    assert!(!"0x1p-149".parse::<Single>().unwrap().is_normal());
}

#[test]
fn is_finite() {
    let t = "0x1p+0".parse::<Single>().unwrap();
    assert!(t.is_finite());
    assert!(!Single::INFINITY.is_finite());
    assert!(Single::ZERO.is_finite());
    assert!(!Single::NAN.is_finite());
    assert!(!Single::snan(None).is_finite());
    assert!("0x1p-149".parse::<Single>().unwrap().is_finite());
}

#[test]
fn is_infinite() {
    let t = "0x1p+0".parse::<Single>().unwrap();
    assert!(!t.is_infinite());

    let pos_inf = Single::INFINITY;
    let neg_inf = -Single::INFINITY;

    assert!(pos_inf.is_infinite());
    assert!(pos_inf.is_pos_infinity());
    assert!(!pos_inf.is_neg_infinity());
    assert!(neg_inf.is_infinite());
    assert!(!neg_inf.is_pos_infinity());
    assert!(neg_inf.is_neg_infinity());

    assert!(!Single::ZERO.is_infinite());
    assert!(!Single::NAN.is_infinite());
    assert!(!Single::snan(None).is_infinite());
    assert!(!"0x1p-149".parse::<Single>().unwrap().is_infinite());
}

#[test]
fn is_nan() {
    let t = "0x1p+0".parse::<Single>().unwrap();
    assert!(!t.is_nan());
    assert!(!Single::INFINITY.is_nan());
    assert!(!Single::ZERO.is_nan());
    assert!(Single::NAN.is_nan());
    assert!(Single::snan(None).is_nan());
    assert!(!"0x1p-149".parse::<Single>().unwrap().is_nan());
}

#[test]
fn is_finite_non_zero() {
    // Test positive/negative normal value.
    assert!("0x1p+0".parse::<Single>().unwrap().is_finite_non_zero());
    assert!("-0x1p+0".parse::<Single>().unwrap().is_finite_non_zero());

    // Test positive/negative denormal value.
    assert!("0x1p-149".parse::<Single>().unwrap().is_finite_non_zero());
    assert!("-0x1p-149".parse::<Single>().unwrap().is_finite_non_zero());

    // Test +/- Infinity.
    assert!(!Single::INFINITY.is_finite_non_zero());
    assert!(!(-Single::INFINITY).is_finite_non_zero());

    // Test +/- Zero.
    assert!(!Single::ZERO.is_finite_non_zero());
    assert!(!(-Single::ZERO).is_finite_non_zero());

    // Test +/- qNaN. +/- dont mean anything with qNaN but paranoia can't hurt in
    // this instance.
    assert!(!Single::NAN.is_finite_non_zero());
    assert!(!(-Single::NAN).is_finite_non_zero());

    // Test +/- sNaN. +/- dont mean anything with sNaN but paranoia can't hurt in
    // this instance.
    assert!(!Single::snan(None).is_finite_non_zero());
    assert!(!(-Single::snan(None)).is_finite_non_zero());
}

#[test]
fn add() {
    // Test Special Cases against each other and normal values.

    let p_inf = Single::INFINITY;
    let m_inf = -Single::INFINITY;
    let p_zero = Single::ZERO;
    let m_zero = -Single::ZERO;
    let qnan = Single::NAN;
    let snan = "snan123".parse::<Single>().unwrap();
    let p_normal_value = "0x1p+0".parse::<Single>().unwrap();
    let m_normal_value = "-0x1p+0".parse::<Single>().unwrap();
    let p_largest_value = Single::largest();
    let m_largest_value = -Single::largest();
    let p_smallest_value = Single::SMALLEST;
    let m_smallest_value = -Single::SMALLEST;
    let p_smallest_normalized = Single::smallest_normalized();
    let m_smallest_normalized = -Single::smallest_normalized();

    let overflow_status = Status::OVERFLOW | Status::INEXACT;

    let special_cases = [
        (p_inf, p_inf, "inf", Status::OK, Category::Infinity),
        (p_inf, m_inf, "nan", Status::INVALID_OP, Category::NaN),
        (p_inf, p_zero, "inf", Status::OK, Category::Infinity),
        (p_inf, m_zero, "inf", Status::OK, Category::Infinity),
        (p_inf, qnan, "nan", Status::OK, Category::NaN),
        (p_inf, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (p_inf, p_normal_value, "inf", Status::OK, Category::Infinity),
        (p_inf, m_normal_value, "inf", Status::OK, Category::Infinity),
        (p_inf, p_largest_value, "inf", Status::OK, Category::Infinity),
        (p_inf, m_largest_value, "inf", Status::OK, Category::Infinity),
        (p_inf, p_smallest_value, "inf", Status::OK, Category::Infinity),
        (p_inf, m_smallest_value, "inf", Status::OK, Category::Infinity),
        (p_inf, p_smallest_normalized, "inf", Status::OK, Category::Infinity),
        (p_inf, m_smallest_normalized, "inf", Status::OK, Category::Infinity),
        (m_inf, p_inf, "nan", Status::INVALID_OP, Category::NaN),
        (m_inf, m_inf, "-inf", Status::OK, Category::Infinity),
        (m_inf, p_zero, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_zero, "-inf", Status::OK, Category::Infinity),
        (m_inf, qnan, "nan", Status::OK, Category::NaN),
        (m_inf, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (m_inf, p_normal_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_normal_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, p_largest_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_largest_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, p_smallest_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_smallest_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, p_smallest_normalized, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_smallest_normalized, "-inf", Status::OK, Category::Infinity),
        (p_zero, p_inf, "inf", Status::OK, Category::Infinity),
        (p_zero, m_inf, "-inf", Status::OK, Category::Infinity),
        (p_zero, p_zero, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_zero, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, qnan, "nan", Status::OK, Category::NaN),
        (p_zero, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (p_zero, p_normal_value, "0x1p+0", Status::OK, Category::Normal),
        (p_zero, m_normal_value, "-0x1p+0", Status::OK, Category::Normal),
        (p_zero, p_largest_value, "0x1.fffffep+127", Status::OK, Category::Normal),
        (p_zero, m_largest_value, "-0x1.fffffep+127", Status::OK, Category::Normal),
        (p_zero, p_smallest_value, "0x1p-149", Status::OK, Category::Normal),
        (p_zero, m_smallest_value, "-0x1p-149", Status::OK, Category::Normal),
        (p_zero, p_smallest_normalized, "0x1p-126", Status::OK, Category::Normal),
        (p_zero, m_smallest_normalized, "-0x1p-126", Status::OK, Category::Normal),
        (m_zero, p_inf, "inf", Status::OK, Category::Infinity),
        (m_zero, m_inf, "-inf", Status::OK, Category::Infinity),
        (m_zero, p_zero, "0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_zero, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, qnan, "nan", Status::OK, Category::NaN),
        (m_zero, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (m_zero, p_normal_value, "0x1p+0", Status::OK, Category::Normal),
        (m_zero, m_normal_value, "-0x1p+0", Status::OK, Category::Normal),
        (m_zero, p_largest_value, "0x1.fffffep+127", Status::OK, Category::Normal),
        (m_zero, m_largest_value, "-0x1.fffffep+127", Status::OK, Category::Normal),
        (m_zero, p_smallest_value, "0x1p-149", Status::OK, Category::Normal),
        (m_zero, m_smallest_value, "-0x1p-149", Status::OK, Category::Normal),
        (m_zero, p_smallest_normalized, "0x1p-126", Status::OK, Category::Normal),
        (m_zero, m_smallest_normalized, "-0x1p-126", Status::OK, Category::Normal),
        (qnan, p_inf, "nan", Status::OK, Category::NaN),
        (qnan, m_inf, "nan", Status::OK, Category::NaN),
        (qnan, p_zero, "nan", Status::OK, Category::NaN),
        (qnan, m_zero, "nan", Status::OK, Category::NaN),
        (qnan, qnan, "nan", Status::OK, Category::NaN),
        (qnan, snan, "nan", Status::INVALID_OP, Category::NaN),
        (qnan, p_normal_value, "nan", Status::OK, Category::NaN),
        (qnan, m_normal_value, "nan", Status::OK, Category::NaN),
        (qnan, p_largest_value, "nan", Status::OK, Category::NaN),
        (qnan, m_largest_value, "nan", Status::OK, Category::NaN),
        (qnan, p_smallest_value, "nan", Status::OK, Category::NaN),
        (qnan, m_smallest_value, "nan", Status::OK, Category::NaN),
        (qnan, p_smallest_normalized, "nan", Status::OK, Category::NaN),
        (qnan, m_smallest_normalized, "nan", Status::OK, Category::NaN),
        (snan, p_inf, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, m_inf, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, p_zero, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, m_zero, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, qnan, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, p_normal_value, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, m_normal_value, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, p_largest_value, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, m_largest_value, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, p_smallest_value, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, m_smallest_value, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, p_smallest_normalized, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, m_smallest_normalized, "nan123", Status::INVALID_OP, Category::NaN),
        (p_normal_value, p_inf, "inf", Status::OK, Category::Infinity),
        (p_normal_value, m_inf, "-inf", Status::OK, Category::Infinity),
        (p_normal_value, p_zero, "0x1p+0", Status::OK, Category::Normal),
        (p_normal_value, m_zero, "0x1p+0", Status::OK, Category::Normal),
        (p_normal_value, qnan, "nan", Status::OK, Category::NaN),
        (p_normal_value, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (p_normal_value, p_normal_value, "0x1p+1", Status::OK, Category::Normal),
        (p_normal_value, m_normal_value, "0x0p+0", Status::OK, Category::Zero),
        (p_normal_value, p_largest_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_normal_value, m_largest_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_normal_value, p_smallest_value, "0x1p+0", Status::INEXACT, Category::Normal),
        (p_normal_value, m_smallest_value, "0x1p+0", Status::INEXACT, Category::Normal),
        (p_normal_value, p_smallest_normalized, "0x1p+0", Status::INEXACT, Category::Normal),
        (p_normal_value, m_smallest_normalized, "0x1p+0", Status::INEXACT, Category::Normal),
        (m_normal_value, p_inf, "inf", Status::OK, Category::Infinity),
        (m_normal_value, m_inf, "-inf", Status::OK, Category::Infinity),
        (m_normal_value, p_zero, "-0x1p+0", Status::OK, Category::Normal),
        (m_normal_value, m_zero, "-0x1p+0", Status::OK, Category::Normal),
        (m_normal_value, qnan, "nan", Status::OK, Category::NaN),
        (m_normal_value, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (m_normal_value, p_normal_value, "0x0p+0", Status::OK, Category::Zero),
        (m_normal_value, m_normal_value, "-0x1p+1", Status::OK, Category::Normal),
        (m_normal_value, p_largest_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_normal_value, m_largest_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_normal_value, p_smallest_value, "-0x1p+0", Status::INEXACT, Category::Normal),
        (m_normal_value, m_smallest_value, "-0x1p+0", Status::INEXACT, Category::Normal),
        (m_normal_value, p_smallest_normalized, "-0x1p+0", Status::INEXACT, Category::Normal),
        (m_normal_value, m_smallest_normalized, "-0x1p+0", Status::INEXACT, Category::Normal),
        (p_largest_value, p_inf, "inf", Status::OK, Category::Infinity),
        (p_largest_value, m_inf, "-inf", Status::OK, Category::Infinity),
        (p_largest_value, p_zero, "0x1.fffffep+127", Status::OK, Category::Normal),
        (p_largest_value, m_zero, "0x1.fffffep+127", Status::OK, Category::Normal),
        (p_largest_value, qnan, "nan", Status::OK, Category::NaN),
        (p_largest_value, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (p_largest_value, p_normal_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_largest_value, m_normal_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_largest_value, p_largest_value, "inf", overflow_status, Category::Infinity),
        (p_largest_value, m_largest_value, "0x0p+0", Status::OK, Category::Zero),
        (p_largest_value, p_smallest_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_largest_value, m_smallest_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_largest_value, p_smallest_normalized, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_largest_value, m_smallest_normalized, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_largest_value, p_inf, "inf", Status::OK, Category::Infinity),
        (m_largest_value, m_inf, "-inf", Status::OK, Category::Infinity),
        (m_largest_value, p_zero, "-0x1.fffffep+127", Status::OK, Category::Normal),
        (m_largest_value, m_zero, "-0x1.fffffep+127", Status::OK, Category::Normal),
        (m_largest_value, qnan, "nan", Status::OK, Category::NaN),
        (m_largest_value, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (m_largest_value, p_normal_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_largest_value, m_normal_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_largest_value, p_largest_value, "0x0p+0", Status::OK, Category::Zero),
        (m_largest_value, m_largest_value, "-inf", overflow_status, Category::Infinity),
        (m_largest_value, p_smallest_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_largest_value, m_smallest_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_largest_value, p_smallest_normalized, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_largest_value, m_smallest_normalized, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_smallest_value, p_inf, "inf", Status::OK, Category::Infinity),
        (p_smallest_value, m_inf, "-inf", Status::OK, Category::Infinity),
        (p_smallest_value, p_zero, "0x1p-149", Status::OK, Category::Normal),
        (p_smallest_value, m_zero, "0x1p-149", Status::OK, Category::Normal),
        (p_smallest_value, qnan, "nan", Status::OK, Category::NaN),
        (p_smallest_value, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (p_smallest_value, p_normal_value, "0x1p+0", Status::INEXACT, Category::Normal),
        (p_smallest_value, m_normal_value, "-0x1p+0", Status::INEXACT, Category::Normal),
        (p_smallest_value, p_largest_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_smallest_value, m_largest_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_smallest_value, p_smallest_value, "0x1p-148", Status::OK, Category::Normal),
        (p_smallest_value, m_smallest_value, "0x0p+0", Status::OK, Category::Zero),
        (p_smallest_value, p_smallest_normalized, "0x1.000002p-126", Status::OK, Category::Normal),
        (p_smallest_value, m_smallest_normalized, "-0x1.fffffcp-127", Status::OK, Category::Normal),
        (m_smallest_value, p_inf, "inf", Status::OK, Category::Infinity),
        (m_smallest_value, m_inf, "-inf", Status::OK, Category::Infinity),
        (m_smallest_value, p_zero, "-0x1p-149", Status::OK, Category::Normal),
        (m_smallest_value, m_zero, "-0x1p-149", Status::OK, Category::Normal),
        (m_smallest_value, qnan, "nan", Status::OK, Category::NaN),
        (m_smallest_value, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (m_smallest_value, p_normal_value, "0x1p+0", Status::INEXACT, Category::Normal),
        (m_smallest_value, m_normal_value, "-0x1p+0", Status::INEXACT, Category::Normal),
        (m_smallest_value, p_largest_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_smallest_value, m_largest_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_smallest_value, p_smallest_value, "0x0p+0", Status::OK, Category::Zero),
        (m_smallest_value, m_smallest_value, "-0x1p-148", Status::OK, Category::Normal),
        (m_smallest_value, p_smallest_normalized, "0x1.fffffcp-127", Status::OK, Category::Normal),
        (m_smallest_value, m_smallest_normalized, "-0x1.000002p-126", Status::OK, Category::Normal),
        (p_smallest_normalized, p_inf, "inf", Status::OK, Category::Infinity),
        (p_smallest_normalized, m_inf, "-inf", Status::OK, Category::Infinity),
        (p_smallest_normalized, p_zero, "0x1p-126", Status::OK, Category::Normal),
        (p_smallest_normalized, m_zero, "0x1p-126", Status::OK, Category::Normal),
        (p_smallest_normalized, qnan, "nan", Status::OK, Category::NaN),
        (p_smallest_normalized, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (p_smallest_normalized, p_normal_value, "0x1p+0", Status::INEXACT, Category::Normal),
        (p_smallest_normalized, m_normal_value, "-0x1p+0", Status::INEXACT, Category::Normal),
        (p_smallest_normalized, p_largest_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_smallest_normalized, m_largest_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_smallest_normalized, p_smallest_value, "0x1.000002p-126", Status::OK, Category::Normal),
        (p_smallest_normalized, m_smallest_value, "0x1.fffffcp-127", Status::OK, Category::Normal),
        (p_smallest_normalized, p_smallest_normalized, "0x1p-125", Status::OK, Category::Normal),
        (p_smallest_normalized, m_smallest_normalized, "0x0p+0", Status::OK, Category::Zero),
        (m_smallest_normalized, p_inf, "inf", Status::OK, Category::Infinity),
        (m_smallest_normalized, m_inf, "-inf", Status::OK, Category::Infinity),
        (m_smallest_normalized, p_zero, "-0x1p-126", Status::OK, Category::Normal),
        (m_smallest_normalized, m_zero, "-0x1p-126", Status::OK, Category::Normal),
        (m_smallest_normalized, qnan, "nan", Status::OK, Category::NaN),
        (m_smallest_normalized, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (m_smallest_normalized, p_normal_value, "0x1p+0", Status::INEXACT, Category::Normal),
        (m_smallest_normalized, m_normal_value, "-0x1p+0", Status::INEXACT, Category::Normal),
        (m_smallest_normalized, p_largest_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_smallest_normalized, m_largest_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_smallest_normalized, p_smallest_value, "-0x1.fffffcp-127", Status::OK, Category::Normal),
        (m_smallest_normalized, m_smallest_value, "-0x1.000002p-126", Status::OK, Category::Normal),
        (m_smallest_normalized, p_smallest_normalized, "0x0p+0", Status::OK, Category::Zero),
        (m_smallest_normalized, m_smallest_normalized, "-0x1p-125", Status::OK, Category::Normal),
    ];

    for case @ &(x, y, e_result, e_status, e_category) in &special_cases[..] {
        let status;
        let result = unpack!(status=, x + y);
        assert_eq!(e_status, status);
        assert_eq!(e_category, result.category());
        assert!(result.bitwise_eq(e_result.parse::<Single>().unwrap()), "result = {result:?}, case = {case:?}");
    }
}

#[test]
fn subtract() {
    // Test Special Cases against each other and normal values.

    let p_inf = Single::INFINITY;
    let m_inf = -Single::INFINITY;
    let p_zero = Single::ZERO;
    let m_zero = -Single::ZERO;
    let qnan = Single::NAN;
    let snan = "snan123".parse::<Single>().unwrap();
    let p_normal_value = "0x1p+0".parse::<Single>().unwrap();
    let m_normal_value = "-0x1p+0".parse::<Single>().unwrap();
    let p_largest_value = Single::largest();
    let m_largest_value = -Single::largest();
    let p_smallest_value = Single::SMALLEST;
    let m_smallest_value = -Single::SMALLEST;
    let p_smallest_normalized = Single::smallest_normalized();
    let m_smallest_normalized = -Single::smallest_normalized();

    let overflow_status = Status::OVERFLOW | Status::INEXACT;

    let special_cases = [
        (p_inf, p_inf, "nan", Status::INVALID_OP, Category::NaN),
        (p_inf, m_inf, "inf", Status::OK, Category::Infinity),
        (p_inf, p_zero, "inf", Status::OK, Category::Infinity),
        (p_inf, m_zero, "inf", Status::OK, Category::Infinity),
        (p_inf, qnan, "nan", Status::OK, Category::NaN),
        (p_inf, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (p_inf, p_normal_value, "inf", Status::OK, Category::Infinity),
        (p_inf, m_normal_value, "inf", Status::OK, Category::Infinity),
        (p_inf, p_largest_value, "inf", Status::OK, Category::Infinity),
        (p_inf, m_largest_value, "inf", Status::OK, Category::Infinity),
        (p_inf, p_smallest_value, "inf", Status::OK, Category::Infinity),
        (p_inf, m_smallest_value, "inf", Status::OK, Category::Infinity),
        (p_inf, p_smallest_normalized, "inf", Status::OK, Category::Infinity),
        (p_inf, m_smallest_normalized, "inf", Status::OK, Category::Infinity),
        (m_inf, p_inf, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_inf, "nan", Status::INVALID_OP, Category::NaN),
        (m_inf, p_zero, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_zero, "-inf", Status::OK, Category::Infinity),
        (m_inf, qnan, "nan", Status::OK, Category::NaN),
        (m_inf, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (m_inf, p_normal_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_normal_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, p_largest_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_largest_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, p_smallest_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_smallest_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, p_smallest_normalized, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_smallest_normalized, "-inf", Status::OK, Category::Infinity),
        (p_zero, p_inf, "-inf", Status::OK, Category::Infinity),
        (p_zero, m_inf, "inf", Status::OK, Category::Infinity),
        (p_zero, p_zero, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_zero, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, qnan, "nan", Status::OK, Category::NaN),
        (p_zero, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (p_zero, p_normal_value, "-0x1p+0", Status::OK, Category::Normal),
        (p_zero, m_normal_value, "0x1p+0", Status::OK, Category::Normal),
        (p_zero, p_largest_value, "-0x1.fffffep+127", Status::OK, Category::Normal),
        (p_zero, m_largest_value, "0x1.fffffep+127", Status::OK, Category::Normal),
        (p_zero, p_smallest_value, "-0x1p-149", Status::OK, Category::Normal),
        (p_zero, m_smallest_value, "0x1p-149", Status::OK, Category::Normal),
        (p_zero, p_smallest_normalized, "-0x1p-126", Status::OK, Category::Normal),
        (p_zero, m_smallest_normalized, "0x1p-126", Status::OK, Category::Normal),
        (m_zero, p_inf, "-inf", Status::OK, Category::Infinity),
        (m_zero, m_inf, "inf", Status::OK, Category::Infinity),
        (m_zero, p_zero, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_zero, "0x0p+0", Status::OK, Category::Zero),
        (m_zero, qnan, "nan", Status::OK, Category::NaN),
        (m_zero, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (m_zero, p_normal_value, "-0x1p+0", Status::OK, Category::Normal),
        (m_zero, m_normal_value, "0x1p+0", Status::OK, Category::Normal),
        (m_zero, p_largest_value, "-0x1.fffffep+127", Status::OK, Category::Normal),
        (m_zero, m_largest_value, "0x1.fffffep+127", Status::OK, Category::Normal),
        (m_zero, p_smallest_value, "-0x1p-149", Status::OK, Category::Normal),
        (m_zero, m_smallest_value, "0x1p-149", Status::OK, Category::Normal),
        (m_zero, p_smallest_normalized, "-0x1p-126", Status::OK, Category::Normal),
        (m_zero, m_smallest_normalized, "0x1p-126", Status::OK, Category::Normal),
        (qnan, p_inf, "nan", Status::OK, Category::NaN),
        (qnan, m_inf, "nan", Status::OK, Category::NaN),
        (qnan, p_zero, "nan", Status::OK, Category::NaN),
        (qnan, m_zero, "nan", Status::OK, Category::NaN),
        (qnan, qnan, "nan", Status::OK, Category::NaN),
        (qnan, snan, "nan", Status::INVALID_OP, Category::NaN),
        (qnan, p_normal_value, "nan", Status::OK, Category::NaN),
        (qnan, m_normal_value, "nan", Status::OK, Category::NaN),
        (qnan, p_largest_value, "nan", Status::OK, Category::NaN),
        (qnan, m_largest_value, "nan", Status::OK, Category::NaN),
        (qnan, p_smallest_value, "nan", Status::OK, Category::NaN),
        (qnan, m_smallest_value, "nan", Status::OK, Category::NaN),
        (qnan, p_smallest_normalized, "nan", Status::OK, Category::NaN),
        (qnan, m_smallest_normalized, "nan", Status::OK, Category::NaN),
        (snan, p_inf, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, m_inf, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, p_zero, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, m_zero, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, qnan, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, p_normal_value, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, m_normal_value, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, p_largest_value, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, m_largest_value, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, p_smallest_value, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, m_smallest_value, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, p_smallest_normalized, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, m_smallest_normalized, "nan123", Status::INVALID_OP, Category::NaN),
        (p_normal_value, p_inf, "-inf", Status::OK, Category::Infinity),
        (p_normal_value, m_inf, "inf", Status::OK, Category::Infinity),
        (p_normal_value, p_zero, "0x1p+0", Status::OK, Category::Normal),
        (p_normal_value, m_zero, "0x1p+0", Status::OK, Category::Normal),
        (p_normal_value, qnan, "nan", Status::OK, Category::NaN),
        (p_normal_value, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (p_normal_value, p_normal_value, "0x0p+0", Status::OK, Category::Zero),
        (p_normal_value, m_normal_value, "0x1p+1", Status::OK, Category::Normal),
        (p_normal_value, p_largest_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_normal_value, m_largest_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_normal_value, p_smallest_value, "0x1p+0", Status::INEXACT, Category::Normal),
        (p_normal_value, m_smallest_value, "0x1p+0", Status::INEXACT, Category::Normal),
        (p_normal_value, p_smallest_normalized, "0x1p+0", Status::INEXACT, Category::Normal),
        (p_normal_value, m_smallest_normalized, "0x1p+0", Status::INEXACT, Category::Normal),
        (m_normal_value, p_inf, "-inf", Status::OK, Category::Infinity),
        (m_normal_value, m_inf, "inf", Status::OK, Category::Infinity),
        (m_normal_value, p_zero, "-0x1p+0", Status::OK, Category::Normal),
        (m_normal_value, m_zero, "-0x1p+0", Status::OK, Category::Normal),
        (m_normal_value, qnan, "nan", Status::OK, Category::NaN),
        (m_normal_value, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (m_normal_value, p_normal_value, "-0x1p+1", Status::OK, Category::Normal),
        (m_normal_value, m_normal_value, "0x0p+0", Status::OK, Category::Zero),
        (m_normal_value, p_largest_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_normal_value, m_largest_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_normal_value, p_smallest_value, "-0x1p+0", Status::INEXACT, Category::Normal),
        (m_normal_value, m_smallest_value, "-0x1p+0", Status::INEXACT, Category::Normal),
        (m_normal_value, p_smallest_normalized, "-0x1p+0", Status::INEXACT, Category::Normal),
        (m_normal_value, m_smallest_normalized, "-0x1p+0", Status::INEXACT, Category::Normal),
        (p_largest_value, p_inf, "-inf", Status::OK, Category::Infinity),
        (p_largest_value, m_inf, "inf", Status::OK, Category::Infinity),
        (p_largest_value, p_zero, "0x1.fffffep+127", Status::OK, Category::Normal),
        (p_largest_value, m_zero, "0x1.fffffep+127", Status::OK, Category::Normal),
        (p_largest_value, qnan, "nan", Status::OK, Category::NaN),
        (p_largest_value, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (p_largest_value, p_normal_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_largest_value, m_normal_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_largest_value, p_largest_value, "0x0p+0", Status::OK, Category::Zero),
        (p_largest_value, m_largest_value, "inf", overflow_status, Category::Infinity),
        (p_largest_value, p_smallest_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_largest_value, m_smallest_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_largest_value, p_smallest_normalized, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_largest_value, m_smallest_normalized, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_largest_value, p_inf, "-inf", Status::OK, Category::Infinity),
        (m_largest_value, m_inf, "inf", Status::OK, Category::Infinity),
        (m_largest_value, p_zero, "-0x1.fffffep+127", Status::OK, Category::Normal),
        (m_largest_value, m_zero, "-0x1.fffffep+127", Status::OK, Category::Normal),
        (m_largest_value, qnan, "nan", Status::OK, Category::NaN),
        (m_largest_value, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (m_largest_value, p_normal_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_largest_value, m_normal_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_largest_value, p_largest_value, "-inf", overflow_status, Category::Infinity),
        (m_largest_value, m_largest_value, "0x0p+0", Status::OK, Category::Zero),
        (m_largest_value, p_smallest_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_largest_value, m_smallest_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_largest_value, p_smallest_normalized, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_largest_value, m_smallest_normalized, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_smallest_value, p_inf, "-inf", Status::OK, Category::Infinity),
        (p_smallest_value, m_inf, "inf", Status::OK, Category::Infinity),
        (p_smallest_value, p_zero, "0x1p-149", Status::OK, Category::Normal),
        (p_smallest_value, m_zero, "0x1p-149", Status::OK, Category::Normal),
        (p_smallest_value, qnan, "nan", Status::OK, Category::NaN),
        (p_smallest_value, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (p_smallest_value, p_normal_value, "-0x1p+0", Status::INEXACT, Category::Normal),
        (p_smallest_value, m_normal_value, "0x1p+0", Status::INEXACT, Category::Normal),
        (p_smallest_value, p_largest_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_smallest_value, m_largest_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_smallest_value, p_smallest_value, "0x0p+0", Status::OK, Category::Zero),
        (p_smallest_value, m_smallest_value, "0x1p-148", Status::OK, Category::Normal),
        (p_smallest_value, p_smallest_normalized, "-0x1.fffffcp-127", Status::OK, Category::Normal),
        (p_smallest_value, m_smallest_normalized, "0x1.000002p-126", Status::OK, Category::Normal),
        (m_smallest_value, p_inf, "-inf", Status::OK, Category::Infinity),
        (m_smallest_value, m_inf, "inf", Status::OK, Category::Infinity),
        (m_smallest_value, p_zero, "-0x1p-149", Status::OK, Category::Normal),
        (m_smallest_value, m_zero, "-0x1p-149", Status::OK, Category::Normal),
        (m_smallest_value, qnan, "nan", Status::OK, Category::NaN),
        (m_smallest_value, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (m_smallest_value, p_normal_value, "-0x1p+0", Status::INEXACT, Category::Normal),
        (m_smallest_value, m_normal_value, "0x1p+0", Status::INEXACT, Category::Normal),
        (m_smallest_value, p_largest_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_smallest_value, m_largest_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_smallest_value, p_smallest_value, "-0x1p-148", Status::OK, Category::Normal),
        (m_smallest_value, m_smallest_value, "0x0p+0", Status::OK, Category::Zero),
        (m_smallest_value, p_smallest_normalized, "-0x1.000002p-126", Status::OK, Category::Normal),
        (m_smallest_value, m_smallest_normalized, "0x1.fffffcp-127", Status::OK, Category::Normal),
        (p_smallest_normalized, p_inf, "-inf", Status::OK, Category::Infinity),
        (p_smallest_normalized, m_inf, "inf", Status::OK, Category::Infinity),
        (p_smallest_normalized, p_zero, "0x1p-126", Status::OK, Category::Normal),
        (p_smallest_normalized, m_zero, "0x1p-126", Status::OK, Category::Normal),
        (p_smallest_normalized, qnan, "nan", Status::OK, Category::NaN),
        (p_smallest_normalized, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (p_smallest_normalized, p_normal_value, "-0x1p+0", Status::INEXACT, Category::Normal),
        (p_smallest_normalized, m_normal_value, "0x1p+0", Status::INEXACT, Category::Normal),
        (p_smallest_normalized, p_largest_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_smallest_normalized, m_largest_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (p_smallest_normalized, p_smallest_value, "0x1.fffffcp-127", Status::OK, Category::Normal),
        (p_smallest_normalized, m_smallest_value, "0x1.000002p-126", Status::OK, Category::Normal),
        (p_smallest_normalized, p_smallest_normalized, "0x0p+0", Status::OK, Category::Zero),
        (p_smallest_normalized, m_smallest_normalized, "0x1p-125", Status::OK, Category::Normal),
        (m_smallest_normalized, p_inf, "-inf", Status::OK, Category::Infinity),
        (m_smallest_normalized, m_inf, "inf", Status::OK, Category::Infinity),
        (m_smallest_normalized, p_zero, "-0x1p-126", Status::OK, Category::Normal),
        (m_smallest_normalized, m_zero, "-0x1p-126", Status::OK, Category::Normal),
        (m_smallest_normalized, qnan, "nan", Status::OK, Category::NaN),
        (m_smallest_normalized, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (m_smallest_normalized, p_normal_value, "-0x1p+0", Status::INEXACT, Category::Normal),
        (m_smallest_normalized, m_normal_value, "0x1p+0", Status::INEXACT, Category::Normal),
        (m_smallest_normalized, p_largest_value, "-0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_smallest_normalized, m_largest_value, "0x1.fffffep+127", Status::INEXACT, Category::Normal),
        (m_smallest_normalized, p_smallest_value, "-0x1.000002p-126", Status::OK, Category::Normal),
        (m_smallest_normalized, m_smallest_value, "-0x1.fffffcp-127", Status::OK, Category::Normal),
        (m_smallest_normalized, p_smallest_normalized, "-0x1p-125", Status::OK, Category::Normal),
        (m_smallest_normalized, m_smallest_normalized, "0x0p+0", Status::OK, Category::Zero),
    ];

    for case @ &(x, y, e_result, e_status, e_category) in &special_cases[..] {
        let status;
        let result = unpack!(status=, x - y);
        assert_eq!(e_status, status);
        assert_eq!(e_category, result.category());
        assert!(result.bitwise_eq(e_result.parse::<Single>().unwrap()), "result = {result:?}, case = {case:?}");
    }
}

#[test]
fn multiply() {
    // Test Special Cases against each other and normal values.

    let p_inf = Single::INFINITY;
    let m_inf = -Single::INFINITY;
    let p_zero = Single::ZERO;
    let m_zero = -Single::ZERO;
    let qnan = Single::NAN;
    let snan = "snan123".parse::<Single>().unwrap();
    let p_normal_value = "0x1p+0".parse::<Single>().unwrap();
    let m_normal_value = "-0x1p+0".parse::<Single>().unwrap();
    let p_largest_value = Single::largest();
    let m_largest_value = -Single::largest();
    let p_smallest_value = Single::SMALLEST;
    let m_smallest_value = -Single::SMALLEST;
    let p_smallest_normalized = Single::smallest_normalized();
    let m_smallest_normalized = -Single::smallest_normalized();

    let max_quad = "0x1.ffffffffffffffffffffffffffffp+16383".parse::<Quad>().unwrap();
    let min_quad = "0x0.0000000000000000000000000001p-16382".parse::<Quad>().unwrap();
    let n_min_quad = "-0x0.0000000000000000000000000001p-16382".parse::<Quad>().unwrap();

    let overflow_status = Status::OVERFLOW | Status::INEXACT;
    let underflow_status = Status::UNDERFLOW | Status::INEXACT;

    let single_special_cases = [
        (p_inf, p_inf, "inf", Status::OK, Category::Infinity),
        (p_inf, m_inf, "-inf", Status::OK, Category::Infinity),
        (p_inf, p_zero, "nan", Status::INVALID_OP, Category::NaN),
        (p_inf, m_zero, "nan", Status::INVALID_OP, Category::NaN),
        (p_inf, qnan, "nan", Status::OK, Category::NaN),
        (p_inf, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (p_inf, p_normal_value, "inf", Status::OK, Category::Infinity),
        (p_inf, m_normal_value, "-inf", Status::OK, Category::Infinity),
        (p_inf, p_largest_value, "inf", Status::OK, Category::Infinity),
        (p_inf, m_largest_value, "-inf", Status::OK, Category::Infinity),
        (p_inf, p_smallest_value, "inf", Status::OK, Category::Infinity),
        (p_inf, m_smallest_value, "-inf", Status::OK, Category::Infinity),
        (p_inf, p_smallest_normalized, "inf", Status::OK, Category::Infinity),
        (p_inf, m_smallest_normalized, "-inf", Status::OK, Category::Infinity),
        (m_inf, p_inf, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_inf, "inf", Status::OK, Category::Infinity),
        (m_inf, p_zero, "nan", Status::INVALID_OP, Category::NaN),
        (m_inf, m_zero, "nan", Status::INVALID_OP, Category::NaN),
        (m_inf, qnan, "nan", Status::OK, Category::NaN),
        (m_inf, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (m_inf, p_normal_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_normal_value, "inf", Status::OK, Category::Infinity),
        (m_inf, p_largest_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_largest_value, "inf", Status::OK, Category::Infinity),
        (m_inf, p_smallest_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_smallest_value, "inf", Status::OK, Category::Infinity),
        (m_inf, p_smallest_normalized, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_smallest_normalized, "inf", Status::OK, Category::Infinity),
        (p_zero, p_inf, "nan", Status::INVALID_OP, Category::NaN),
        (p_zero, m_inf, "nan", Status::INVALID_OP, Category::NaN),
        (p_zero, p_zero, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_zero, "-0x0p+0", Status::OK, Category::Zero),
        (p_zero, qnan, "nan", Status::OK, Category::NaN),
        (p_zero, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (p_zero, p_normal_value, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_normal_value, "-0x0p+0", Status::OK, Category::Zero),
        (p_zero, p_largest_value, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_largest_value, "-0x0p+0", Status::OK, Category::Zero),
        (p_zero, p_smallest_value, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_smallest_value, "-0x0p+0", Status::OK, Category::Zero),
        (p_zero, p_smallest_normalized, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_smallest_normalized, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, p_inf, "nan", Status::INVALID_OP, Category::NaN),
        (m_zero, m_inf, "nan", Status::INVALID_OP, Category::NaN),
        (m_zero, p_zero, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_zero, "0x0p+0", Status::OK, Category::Zero),
        (m_zero, qnan, "nan", Status::OK, Category::NaN),
        (m_zero, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (m_zero, p_normal_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_normal_value, "0x0p+0", Status::OK, Category::Zero),
        (m_zero, p_largest_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_largest_value, "0x0p+0", Status::OK, Category::Zero),
        (m_zero, p_smallest_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_smallest_value, "0x0p+0", Status::OK, Category::Zero),
        (m_zero, p_smallest_normalized, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_smallest_normalized, "0x0p+0", Status::OK, Category::Zero),
        (qnan, p_inf, "nan", Status::OK, Category::NaN),
        (qnan, m_inf, "nan", Status::OK, Category::NaN),
        (qnan, p_zero, "nan", Status::OK, Category::NaN),
        (qnan, m_zero, "nan", Status::OK, Category::NaN),
        (qnan, qnan, "nan", Status::OK, Category::NaN),
        (qnan, snan, "nan", Status::INVALID_OP, Category::NaN),
        (qnan, p_normal_value, "nan", Status::OK, Category::NaN),
        (qnan, m_normal_value, "nan", Status::OK, Category::NaN),
        (qnan, p_largest_value, "nan", Status::OK, Category::NaN),
        (qnan, m_largest_value, "nan", Status::OK, Category::NaN),
        (qnan, p_smallest_value, "nan", Status::OK, Category::NaN),
        (qnan, m_smallest_value, "nan", Status::OK, Category::NaN),
        (qnan, p_smallest_normalized, "nan", Status::OK, Category::NaN),
        (qnan, m_smallest_normalized, "nan", Status::OK, Category::NaN),
        (snan, p_inf, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, m_inf, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, p_zero, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, m_zero, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, qnan, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, p_normal_value, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, m_normal_value, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, p_largest_value, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, m_largest_value, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, p_smallest_value, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, m_smallest_value, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, p_smallest_normalized, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, m_smallest_normalized, "nan123", Status::INVALID_OP, Category::NaN),
        (p_normal_value, p_inf, "inf", Status::OK, Category::Infinity),
        (p_normal_value, m_inf, "-inf", Status::OK, Category::Infinity),
        (p_normal_value, p_zero, "0x0p+0", Status::OK, Category::Zero),
        (p_normal_value, m_zero, "-0x0p+0", Status::OK, Category::Zero),
        (p_normal_value, qnan, "nan", Status::OK, Category::NaN),
        (p_normal_value, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (p_normal_value, p_normal_value, "0x1p+0", Status::OK, Category::Normal),
        (p_normal_value, m_normal_value, "-0x1p+0", Status::OK, Category::Normal),
        (p_normal_value, p_largest_value, "0x1.fffffep+127", Status::OK, Category::Normal),
        (p_normal_value, m_largest_value, "-0x1.fffffep+127", Status::OK, Category::Normal),
        (p_normal_value, p_smallest_value, "0x1p-149", Status::OK, Category::Normal),
        (p_normal_value, m_smallest_value, "-0x1p-149", Status::OK, Category::Normal),
        (p_normal_value, p_smallest_normalized, "0x1p-126", Status::OK, Category::Normal),
        (p_normal_value, m_smallest_normalized, "-0x1p-126", Status::OK, Category::Normal),
        (m_normal_value, p_inf, "-inf", Status::OK, Category::Infinity),
        (m_normal_value, m_inf, "inf", Status::OK, Category::Infinity),
        (m_normal_value, p_zero, "-0x0p+0", Status::OK, Category::Zero),
        (m_normal_value, m_zero, "0x0p+0", Status::OK, Category::Zero),
        (m_normal_value, qnan, "nan", Status::OK, Category::NaN),
        (m_normal_value, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (m_normal_value, p_normal_value, "-0x1p+0", Status::OK, Category::Normal),
        (m_normal_value, m_normal_value, "0x1p+0", Status::OK, Category::Normal),
        (m_normal_value, p_largest_value, "-0x1.fffffep+127", Status::OK, Category::Normal),
        (m_normal_value, m_largest_value, "0x1.fffffep+127", Status::OK, Category::Normal),
        (m_normal_value, p_smallest_value, "-0x1p-149", Status::OK, Category::Normal),
        (m_normal_value, m_smallest_value, "0x1p-149", Status::OK, Category::Normal),
        (m_normal_value, p_smallest_normalized, "-0x1p-126", Status::OK, Category::Normal),
        (m_normal_value, m_smallest_normalized, "0x1p-126", Status::OK, Category::Normal),
        (p_largest_value, p_inf, "inf", Status::OK, Category::Infinity),
        (p_largest_value, m_inf, "-inf", Status::OK, Category::Infinity),
        (p_largest_value, p_zero, "0x0p+0", Status::OK, Category::Zero),
        (p_largest_value, m_zero, "-0x0p+0", Status::OK, Category::Zero),
        (p_largest_value, qnan, "nan", Status::OK, Category::NaN),
        (p_largest_value, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (p_largest_value, p_normal_value, "0x1.fffffep+127", Status::OK, Category::Normal),
        (p_largest_value, m_normal_value, "-0x1.fffffep+127", Status::OK, Category::Normal),
        (p_largest_value, p_largest_value, "inf", overflow_status, Category::Infinity),
        (p_largest_value, m_largest_value, "-inf", overflow_status, Category::Infinity),
        (p_largest_value, p_smallest_value, "0x1.fffffep-22", Status::OK, Category::Normal),
        (p_largest_value, m_smallest_value, "-0x1.fffffep-22", Status::OK, Category::Normal),
        (p_largest_value, p_smallest_normalized, "0x1.fffffep+1", Status::OK, Category::Normal),
        (p_largest_value, m_smallest_normalized, "-0x1.fffffep+1", Status::OK, Category::Normal),
        (m_largest_value, p_inf, "-inf", Status::OK, Category::Infinity),
        (m_largest_value, m_inf, "inf", Status::OK, Category::Infinity),
        (m_largest_value, p_zero, "-0x0p+0", Status::OK, Category::Zero),
        (m_largest_value, m_zero, "0x0p+0", Status::OK, Category::Zero),
        (m_largest_value, qnan, "nan", Status::OK, Category::NaN),
        (m_largest_value, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (m_largest_value, p_normal_value, "-0x1.fffffep+127", Status::OK, Category::Normal),
        (m_largest_value, m_normal_value, "0x1.fffffep+127", Status::OK, Category::Normal),
        (m_largest_value, p_largest_value, "-inf", overflow_status, Category::Infinity),
        (m_largest_value, m_largest_value, "inf", overflow_status, Category::Infinity),
        (m_largest_value, p_smallest_value, "-0x1.fffffep-22", Status::OK, Category::Normal),
        (m_largest_value, m_smallest_value, "0x1.fffffep-22", Status::OK, Category::Normal),
        (m_largest_value, p_smallest_normalized, "-0x1.fffffep+1", Status::OK, Category::Normal),
        (m_largest_value, m_smallest_normalized, "0x1.fffffep+1", Status::OK, Category::Normal),
        (p_smallest_value, p_inf, "inf", Status::OK, Category::Infinity),
        (p_smallest_value, m_inf, "-inf", Status::OK, Category::Infinity),
        (p_smallest_value, p_zero, "0x0p+0", Status::OK, Category::Zero),
        (p_smallest_value, m_zero, "-0x0p+0", Status::OK, Category::Zero),
        (p_smallest_value, qnan, "nan", Status::OK, Category::NaN),
        (p_smallest_value, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (p_smallest_value, p_normal_value, "0x1p-149", Status::OK, Category::Normal),
        (p_smallest_value, m_normal_value, "-0x1p-149", Status::OK, Category::Normal),
        (p_smallest_value, p_largest_value, "0x1.fffffep-22", Status::OK, Category::Normal),
        (p_smallest_value, m_largest_value, "-0x1.fffffep-22", Status::OK, Category::Normal),
        (p_smallest_value, p_smallest_value, "0x0p+0", underflow_status, Category::Zero),
        (p_smallest_value, m_smallest_value, "-0x0p+0", underflow_status, Category::Zero),
        (p_smallest_value, p_smallest_normalized, "0x0p+0", underflow_status, Category::Zero),
        (p_smallest_value, m_smallest_normalized, "-0x0p+0", underflow_status, Category::Zero),
        (m_smallest_value, p_inf, "-inf", Status::OK, Category::Infinity),
        (m_smallest_value, m_inf, "inf", Status::OK, Category::Infinity),
        (m_smallest_value, p_zero, "-0x0p+0", Status::OK, Category::Zero),
        (m_smallest_value, m_zero, "0x0p+0", Status::OK, Category::Zero),
        (m_smallest_value, qnan, "nan", Status::OK, Category::NaN),
        (m_smallest_value, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (m_smallest_value, p_normal_value, "-0x1p-149", Status::OK, Category::Normal),
        (m_smallest_value, m_normal_value, "0x1p-149", Status::OK, Category::Normal),
        (m_smallest_value, p_largest_value, "-0x1.fffffep-22", Status::OK, Category::Normal),
        (m_smallest_value, m_largest_value, "0x1.fffffep-22", Status::OK, Category::Normal),
        (m_smallest_value, p_smallest_value, "-0x0p+0", underflow_status, Category::Zero),
        (m_smallest_value, m_smallest_value, "0x0p+0", underflow_status, Category::Zero),
        (m_smallest_value, p_smallest_normalized, "-0x0p+0", underflow_status, Category::Zero),
        (m_smallest_value, m_smallest_normalized, "0x0p+0", underflow_status, Category::Zero),
        (p_smallest_normalized, p_inf, "inf", Status::OK, Category::Infinity),
        (p_smallest_normalized, m_inf, "-inf", Status::OK, Category::Infinity),
        (p_smallest_normalized, p_zero, "0x0p+0", Status::OK, Category::Zero),
        (p_smallest_normalized, m_zero, "-0x0p+0", Status::OK, Category::Zero),
        (p_smallest_normalized, qnan, "nan", Status::OK, Category::NaN),
        (p_smallest_normalized, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (p_smallest_normalized, p_normal_value, "0x1p-126", Status::OK, Category::Normal),
        (p_smallest_normalized, m_normal_value, "-0x1p-126", Status::OK, Category::Normal),
        (p_smallest_normalized, p_largest_value, "0x1.fffffep+1", Status::OK, Category::Normal),
        (p_smallest_normalized, m_largest_value, "-0x1.fffffep+1", Status::OK, Category::Normal),
        (p_smallest_normalized, p_smallest_value, "0x0p+0", underflow_status, Category::Zero),
        (p_smallest_normalized, m_smallest_value, "-0x0p+0", underflow_status, Category::Zero),
        (p_smallest_normalized, p_smallest_normalized, "0x0p+0", underflow_status, Category::Zero),
        (p_smallest_normalized, m_smallest_normalized, "-0x0p+0", underflow_status, Category::Zero),
        (m_smallest_normalized, p_inf, "-inf", Status::OK, Category::Infinity),
        (m_smallest_normalized, m_inf, "inf", Status::OK, Category::Infinity),
        (m_smallest_normalized, p_zero, "-0x0p+0", Status::OK, Category::Zero),
        (m_smallest_normalized, m_zero, "0x0p+0", Status::OK, Category::Zero),
        (m_smallest_normalized, qnan, "nan", Status::OK, Category::NaN),
        (m_smallest_normalized, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (m_smallest_normalized, p_normal_value, "-0x1p-126", Status::OK, Category::Normal),
        (m_smallest_normalized, m_normal_value, "0x1p-126", Status::OK, Category::Normal),
        (m_smallest_normalized, p_largest_value, "-0x1.fffffep+1", Status::OK, Category::Normal),
        (m_smallest_normalized, m_largest_value, "0x1.fffffep+1", Status::OK, Category::Normal),
        (m_smallest_normalized, p_smallest_value, "-0x0p+0", underflow_status, Category::Zero),
        (m_smallest_normalized, m_smallest_value, "0x0p+0", underflow_status, Category::Zero),
        (m_smallest_normalized, p_smallest_normalized, "-0x0p+0", underflow_status, Category::Zero),
        (m_smallest_normalized, m_smallest_normalized, "0x0p+0", underflow_status, Category::Zero),
    ];
    let quad_special_cases = [
        (
            max_quad,
            min_quad,
            "0x1.ffffffffffffffffffffffffffffp-111",
            Status::OK,
            Category::Normal,
            Round::NearestTiesToEven,
        ),
        (
            max_quad,
            min_quad,
            "0x1.ffffffffffffffffffffffffffffp-111",
            Status::OK,
            Category::Normal,
            Round::TowardPositive,
        ),
        (
            max_quad,
            min_quad,
            "0x1.ffffffffffffffffffffffffffffp-111",
            Status::OK,
            Category::Normal,
            Round::TowardNegative,
        ),
        (max_quad, min_quad, "0x1.ffffffffffffffffffffffffffffp-111", Status::OK, Category::Normal, Round::TowardZero),
        (
            max_quad,
            min_quad,
            "0x1.ffffffffffffffffffffffffffffp-111",
            Status::OK,
            Category::Normal,
            Round::NearestTiesToAway,
        ),
        (
            max_quad,
            n_min_quad,
            "-0x1.ffffffffffffffffffffffffffffp-111",
            Status::OK,
            Category::Normal,
            Round::NearestTiesToEven,
        ),
        (
            max_quad,
            n_min_quad,
            "-0x1.ffffffffffffffffffffffffffffp-111",
            Status::OK,
            Category::Normal,
            Round::TowardPositive,
        ),
        (
            max_quad,
            n_min_quad,
            "-0x1.ffffffffffffffffffffffffffffp-111",
            Status::OK,
            Category::Normal,
            Round::TowardNegative,
        ),
        (
            max_quad,
            n_min_quad,
            "-0x1.ffffffffffffffffffffffffffffp-111",
            Status::OK,
            Category::Normal,
            Round::TowardZero,
        ),
        (
            max_quad,
            n_min_quad,
            "-0x1.ffffffffffffffffffffffffffffp-111",
            Status::OK,
            Category::Normal,
            Round::NearestTiesToAway,
        ),
        (max_quad, max_quad, "inf", overflow_status, Category::Infinity, Round::NearestTiesToEven),
        (max_quad, max_quad, "inf", overflow_status, Category::Infinity, Round::TowardPositive),
        (
            max_quad,
            max_quad,
            "0x1.ffffffffffffffffffffffffffffp+16383",
            Status::INEXACT,
            Category::Normal,
            Round::TowardNegative,
        ),
        (
            max_quad,
            max_quad,
            "0x1.ffffffffffffffffffffffffffffp+16383",
            Status::INEXACT,
            Category::Normal,
            Round::TowardZero,
        ),
        (max_quad, max_quad, "inf", overflow_status, Category::Infinity, Round::NearestTiesToAway),
        (min_quad, min_quad, "0", underflow_status, Category::Zero, Round::NearestTiesToEven),
        (
            min_quad,
            min_quad,
            "0x0.0000000000000000000000000001p-16382",
            underflow_status,
            Category::Normal,
            Round::TowardPositive,
        ),
        (min_quad, min_quad, "0", underflow_status, Category::Zero, Round::TowardNegative),
        (min_quad, min_quad, "0", underflow_status, Category::Zero, Round::TowardZero),
        (min_quad, min_quad, "0", underflow_status, Category::Zero, Round::NearestTiesToAway),
        (min_quad, n_min_quad, "-0", underflow_status, Category::Zero, Round::NearestTiesToEven),
        (min_quad, n_min_quad, "-0", underflow_status, Category::Zero, Round::TowardPositive),
        (
            min_quad,
            n_min_quad,
            "-0x0.0000000000000000000000000001p-16382",
            underflow_status,
            Category::Normal,
            Round::TowardNegative,
        ),
        (min_quad, n_min_quad, "-0", underflow_status, Category::Zero, Round::TowardZero),
        (min_quad, n_min_quad, "-0", underflow_status, Category::Zero, Round::NearestTiesToAway),
    ];

    for case @ &(x, y, e_result, e_status, e_category) in &single_special_cases {
        let status;
        let result = unpack!(status=, x * y);
        assert_eq!(e_status, status);
        assert_eq!(e_category, result.category());
        assert!(result.bitwise_eq(e_result.parse::<Single>().unwrap()), "result = {result:?}, case = {case:?}");
    }
    for case @ &(x, y, e_result, e_status, e_category, round) in &quad_special_cases {
        let status;
        let result = unpack!(status=, x.mul_r(y, round));
        assert_eq!(e_status, status);
        assert_eq!(e_category, result.category());
        assert!(result.bitwise_eq(e_result.parse::<Quad>().unwrap()), "result = {result:?}, case = {case:?}");
    }
}

#[test]
fn divide() {
    // Test Special Cases against each other and normal values.

    let p_inf = Single::INFINITY;
    let m_inf = -Single::INFINITY;
    let p_zero = Single::ZERO;
    let m_zero = -Single::ZERO;
    let qnan = Single::NAN;
    let snan = "snan123".parse::<Single>().unwrap();
    let p_normal_value = "0x1p+0".parse::<Single>().unwrap();
    let m_normal_value = "-0x1p+0".parse::<Single>().unwrap();
    let p_largest_value = Single::largest();
    let m_largest_value = -Single::largest();
    let p_smallest_value = Single::SMALLEST;
    let m_smallest_value = -Single::SMALLEST;
    let p_smallest_normalized = Single::smallest_normalized();
    let m_smallest_normalized = -Single::smallest_normalized();

    let max_quad = "0x1.ffffffffffffffffffffffffffffp+16383".parse::<Quad>().unwrap();
    let min_quad = "0x0.0000000000000000000000000001p-16382".parse::<Quad>().unwrap();
    let n_min_quad = "-0x0.0000000000000000000000000001p-16382".parse::<Quad>().unwrap();

    let overflow_status = Status::OVERFLOW | Status::INEXACT;
    let underflow_status = Status::UNDERFLOW | Status::INEXACT;

    let single_special_cases = [
        (p_inf, p_inf, "nan", Status::INVALID_OP, Category::NaN),
        (p_inf, m_inf, "nan", Status::INVALID_OP, Category::NaN),
        (p_inf, p_zero, "inf", Status::OK, Category::Infinity),
        (p_inf, m_zero, "-inf", Status::OK, Category::Infinity),
        (p_inf, qnan, "nan", Status::OK, Category::NaN),
        (p_inf, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (p_inf, p_normal_value, "inf", Status::OK, Category::Infinity),
        (p_inf, m_normal_value, "-inf", Status::OK, Category::Infinity),
        (p_inf, p_largest_value, "inf", Status::OK, Category::Infinity),
        (p_inf, m_largest_value, "-inf", Status::OK, Category::Infinity),
        (p_inf, p_smallest_value, "inf", Status::OK, Category::Infinity),
        (p_inf, m_smallest_value, "-inf", Status::OK, Category::Infinity),
        (p_inf, p_smallest_normalized, "inf", Status::OK, Category::Infinity),
        (p_inf, m_smallest_normalized, "-inf", Status::OK, Category::Infinity),
        (m_inf, p_inf, "nan", Status::INVALID_OP, Category::NaN),
        (m_inf, m_inf, "nan", Status::INVALID_OP, Category::NaN),
        (m_inf, p_zero, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_zero, "inf", Status::OK, Category::Infinity),
        (m_inf, qnan, "nan", Status::OK, Category::NaN),
        (m_inf, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (m_inf, p_normal_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_normal_value, "inf", Status::OK, Category::Infinity),
        (m_inf, p_largest_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_largest_value, "inf", Status::OK, Category::Infinity),
        (m_inf, p_smallest_value, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_smallest_value, "inf", Status::OK, Category::Infinity),
        (m_inf, p_smallest_normalized, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_smallest_normalized, "inf", Status::OK, Category::Infinity),
        (p_zero, p_inf, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_inf, "-0x0p+0", Status::OK, Category::Zero),
        (p_zero, p_zero, "nan", Status::INVALID_OP, Category::NaN),
        (p_zero, m_zero, "nan", Status::INVALID_OP, Category::NaN),
        (p_zero, qnan, "nan", Status::OK, Category::NaN),
        (p_zero, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (p_zero, p_normal_value, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_normal_value, "-0x0p+0", Status::OK, Category::Zero),
        (p_zero, p_largest_value, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_largest_value, "-0x0p+0", Status::OK, Category::Zero),
        (p_zero, p_smallest_value, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_smallest_value, "-0x0p+0", Status::OK, Category::Zero),
        (p_zero, p_smallest_normalized, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_smallest_normalized, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, p_inf, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_inf, "0x0p+0", Status::OK, Category::Zero),
        (m_zero, p_zero, "nan", Status::INVALID_OP, Category::NaN),
        (m_zero, m_zero, "nan", Status::INVALID_OP, Category::NaN),
        (m_zero, qnan, "nan", Status::OK, Category::NaN),
        (m_zero, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (m_zero, p_normal_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_normal_value, "0x0p+0", Status::OK, Category::Zero),
        (m_zero, p_largest_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_largest_value, "0x0p+0", Status::OK, Category::Zero),
        (m_zero, p_smallest_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_smallest_value, "0x0p+0", Status::OK, Category::Zero),
        (m_zero, p_smallest_normalized, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_smallest_normalized, "0x0p+0", Status::OK, Category::Zero),
        (qnan, p_inf, "nan", Status::OK, Category::NaN),
        (qnan, m_inf, "nan", Status::OK, Category::NaN),
        (qnan, p_zero, "nan", Status::OK, Category::NaN),
        (qnan, m_zero, "nan", Status::OK, Category::NaN),
        (qnan, qnan, "nan", Status::OK, Category::NaN),
        (qnan, snan, "nan", Status::INVALID_OP, Category::NaN),
        (qnan, p_normal_value, "nan", Status::OK, Category::NaN),
        (qnan, m_normal_value, "nan", Status::OK, Category::NaN),
        (qnan, p_largest_value, "nan", Status::OK, Category::NaN),
        (qnan, m_largest_value, "nan", Status::OK, Category::NaN),
        (qnan, p_smallest_value, "nan", Status::OK, Category::NaN),
        (qnan, m_smallest_value, "nan", Status::OK, Category::NaN),
        (qnan, p_smallest_normalized, "nan", Status::OK, Category::NaN),
        (qnan, m_smallest_normalized, "nan", Status::OK, Category::NaN),
        (snan, p_inf, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, m_inf, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, p_zero, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, m_zero, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, qnan, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, p_normal_value, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, m_normal_value, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, p_largest_value, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, m_largest_value, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, p_smallest_value, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, m_smallest_value, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, p_smallest_normalized, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, m_smallest_normalized, "nan123", Status::INVALID_OP, Category::NaN),
        (p_normal_value, p_inf, "0x0p+0", Status::OK, Category::Zero),
        (p_normal_value, m_inf, "-0x0p+0", Status::OK, Category::Zero),
        (p_normal_value, p_zero, "inf", Status::DIV_BY_ZERO, Category::Infinity),
        (p_normal_value, m_zero, "-inf", Status::DIV_BY_ZERO, Category::Infinity),
        (p_normal_value, qnan, "nan", Status::OK, Category::NaN),
        (p_normal_value, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (p_normal_value, p_normal_value, "0x1p+0", Status::OK, Category::Normal),
        (p_normal_value, m_normal_value, "-0x1p+0", Status::OK, Category::Normal),
        (p_normal_value, p_largest_value, "0x1p-128", underflow_status, Category::Normal),
        (p_normal_value, m_largest_value, "-0x1p-128", underflow_status, Category::Normal),
        (p_normal_value, p_smallest_value, "inf", overflow_status, Category::Infinity),
        (p_normal_value, m_smallest_value, "-inf", overflow_status, Category::Infinity),
        (p_normal_value, p_smallest_normalized, "0x1p+126", Status::OK, Category::Normal),
        (p_normal_value, m_smallest_normalized, "-0x1p+126", Status::OK, Category::Normal),
        (m_normal_value, p_inf, "-0x0p+0", Status::OK, Category::Zero),
        (m_normal_value, m_inf, "0x0p+0", Status::OK, Category::Zero),
        (m_normal_value, p_zero, "-inf", Status::DIV_BY_ZERO, Category::Infinity),
        (m_normal_value, m_zero, "inf", Status::DIV_BY_ZERO, Category::Infinity),
        (m_normal_value, qnan, "nan", Status::OK, Category::NaN),
        (m_normal_value, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (m_normal_value, p_normal_value, "-0x1p+0", Status::OK, Category::Normal),
        (m_normal_value, m_normal_value, "0x1p+0", Status::OK, Category::Normal),
        (m_normal_value, p_largest_value, "-0x1p-128", underflow_status, Category::Normal),
        (m_normal_value, m_largest_value, "0x1p-128", underflow_status, Category::Normal),
        (m_normal_value, p_smallest_value, "-inf", overflow_status, Category::Infinity),
        (m_normal_value, m_smallest_value, "inf", overflow_status, Category::Infinity),
        (m_normal_value, p_smallest_normalized, "-0x1p+126", Status::OK, Category::Normal),
        (m_normal_value, m_smallest_normalized, "0x1p+126", Status::OK, Category::Normal),
        (p_largest_value, p_inf, "0x0p+0", Status::OK, Category::Zero),
        (p_largest_value, m_inf, "-0x0p+0", Status::OK, Category::Zero),
        (p_largest_value, p_zero, "inf", Status::DIV_BY_ZERO, Category::Infinity),
        (p_largest_value, m_zero, "-inf", Status::DIV_BY_ZERO, Category::Infinity),
        (p_largest_value, qnan, "nan", Status::OK, Category::NaN),
        (p_largest_value, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (p_largest_value, p_normal_value, "0x1.fffffep+127", Status::OK, Category::Normal),
        (p_largest_value, m_normal_value, "-0x1.fffffep+127", Status::OK, Category::Normal),
        (p_largest_value, p_largest_value, "0x1p+0", Status::OK, Category::Normal),
        (p_largest_value, m_largest_value, "-0x1p+0", Status::OK, Category::Normal),
        (p_largest_value, p_smallest_value, "inf", overflow_status, Category::Infinity),
        (p_largest_value, m_smallest_value, "-inf", overflow_status, Category::Infinity),
        (p_largest_value, p_smallest_normalized, "inf", overflow_status, Category::Infinity),
        (p_largest_value, m_smallest_normalized, "-inf", overflow_status, Category::Infinity),
        (m_largest_value, p_inf, "-0x0p+0", Status::OK, Category::Zero),
        (m_largest_value, m_inf, "0x0p+0", Status::OK, Category::Zero),
        (m_largest_value, p_zero, "-inf", Status::DIV_BY_ZERO, Category::Infinity),
        (m_largest_value, m_zero, "inf", Status::DIV_BY_ZERO, Category::Infinity),
        (m_largest_value, qnan, "nan", Status::OK, Category::NaN),
        (m_largest_value, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (m_largest_value, p_normal_value, "-0x1.fffffep+127", Status::OK, Category::Normal),
        (m_largest_value, m_normal_value, "0x1.fffffep+127", Status::OK, Category::Normal),
        (m_largest_value, p_largest_value, "-0x1p+0", Status::OK, Category::Normal),
        (m_largest_value, m_largest_value, "0x1p+0", Status::OK, Category::Normal),
        (m_largest_value, p_smallest_value, "-inf", overflow_status, Category::Infinity),
        (m_largest_value, m_smallest_value, "inf", overflow_status, Category::Infinity),
        (m_largest_value, p_smallest_normalized, "-inf", overflow_status, Category::Infinity),
        (m_largest_value, m_smallest_normalized, "inf", overflow_status, Category::Infinity),
        (p_smallest_value, p_inf, "0x0p+0", Status::OK, Category::Zero),
        (p_smallest_value, m_inf, "-0x0p+0", Status::OK, Category::Zero),
        (p_smallest_value, p_zero, "inf", Status::DIV_BY_ZERO, Category::Infinity),
        (p_smallest_value, m_zero, "-inf", Status::DIV_BY_ZERO, Category::Infinity),
        (p_smallest_value, qnan, "nan", Status::OK, Category::NaN),
        (p_smallest_value, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (p_smallest_value, p_normal_value, "0x1p-149", Status::OK, Category::Normal),
        (p_smallest_value, m_normal_value, "-0x1p-149", Status::OK, Category::Normal),
        (p_smallest_value, p_largest_value, "0x0p+0", underflow_status, Category::Zero),
        (p_smallest_value, m_largest_value, "-0x0p+0", underflow_status, Category::Zero),
        (p_smallest_value, p_smallest_value, "0x1p+0", Status::OK, Category::Normal),
        (p_smallest_value, m_smallest_value, "-0x1p+0", Status::OK, Category::Normal),
        (p_smallest_value, p_smallest_normalized, "0x1p-23", Status::OK, Category::Normal),
        (p_smallest_value, m_smallest_normalized, "-0x1p-23", Status::OK, Category::Normal),
        (m_smallest_value, p_inf, "-0x0p+0", Status::OK, Category::Zero),
        (m_smallest_value, m_inf, "0x0p+0", Status::OK, Category::Zero),
        (m_smallest_value, p_zero, "-inf", Status::DIV_BY_ZERO, Category::Infinity),
        (m_smallest_value, m_zero, "inf", Status::DIV_BY_ZERO, Category::Infinity),
        (m_smallest_value, qnan, "nan", Status::OK, Category::NaN),
        (m_smallest_value, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (m_smallest_value, p_normal_value, "-0x1p-149", Status::OK, Category::Normal),
        (m_smallest_value, m_normal_value, "0x1p-149", Status::OK, Category::Normal),
        (m_smallest_value, p_largest_value, "-0x0p+0", underflow_status, Category::Zero),
        (m_smallest_value, m_largest_value, "0x0p+0", underflow_status, Category::Zero),
        (m_smallest_value, p_smallest_value, "-0x1p+0", Status::OK, Category::Normal),
        (m_smallest_value, m_smallest_value, "0x1p+0", Status::OK, Category::Normal),
        (m_smallest_value, p_smallest_normalized, "-0x1p-23", Status::OK, Category::Normal),
        (m_smallest_value, m_smallest_normalized, "0x1p-23", Status::OK, Category::Normal),
        (p_smallest_normalized, p_inf, "0x0p+0", Status::OK, Category::Zero),
        (p_smallest_normalized, m_inf, "-0x0p+0", Status::OK, Category::Zero),
        (p_smallest_normalized, p_zero, "inf", Status::DIV_BY_ZERO, Category::Infinity),
        (p_smallest_normalized, m_zero, "-inf", Status::DIV_BY_ZERO, Category::Infinity),
        (p_smallest_normalized, qnan, "nan", Status::OK, Category::NaN),
        (p_smallest_normalized, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (p_smallest_normalized, p_normal_value, "0x1p-126", Status::OK, Category::Normal),
        (p_smallest_normalized, m_normal_value, "-0x1p-126", Status::OK, Category::Normal),
        (p_smallest_normalized, p_largest_value, "0x0p+0", underflow_status, Category::Zero),
        (p_smallest_normalized, m_largest_value, "-0x0p+0", underflow_status, Category::Zero),
        (p_smallest_normalized, p_smallest_value, "0x1p+23", Status::OK, Category::Normal),
        (p_smallest_normalized, m_smallest_value, "-0x1p+23", Status::OK, Category::Normal),
        (p_smallest_normalized, p_smallest_normalized, "0x1p+0", Status::OK, Category::Normal),
        (p_smallest_normalized, m_smallest_normalized, "-0x1p+0", Status::OK, Category::Normal),
        (m_smallest_normalized, p_inf, "-0x0p+0", Status::OK, Category::Zero),
        (m_smallest_normalized, m_inf, "0x0p+0", Status::OK, Category::Zero),
        (m_smallest_normalized, p_zero, "-inf", Status::DIV_BY_ZERO, Category::Infinity),
        (m_smallest_normalized, m_zero, "inf", Status::DIV_BY_ZERO, Category::Infinity),
        (m_smallest_normalized, qnan, "nan", Status::OK, Category::NaN),
        (m_smallest_normalized, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (m_smallest_normalized, p_normal_value, "-0x1p-126", Status::OK, Category::Normal),
        (m_smallest_normalized, m_normal_value, "0x1p-126", Status::OK, Category::Normal),
        (m_smallest_normalized, p_largest_value, "-0x0p+0", underflow_status, Category::Zero),
        (m_smallest_normalized, m_largest_value, "0x0p+0", underflow_status, Category::Zero),
        (m_smallest_normalized, p_smallest_value, "-0x1p+23", Status::OK, Category::Normal),
        (m_smallest_normalized, m_smallest_value, "0x1p+23", Status::OK, Category::Normal),
        (m_smallest_normalized, p_smallest_normalized, "-0x1p+0", Status::OK, Category::Normal),
        (m_smallest_normalized, m_smallest_normalized, "0x1p+0", Status::OK, Category::Normal),
    ];
    let quad_special_cases = [
        (max_quad, n_min_quad, "-inf", overflow_status, Category::Infinity, Round::NearestTiesToEven),
        (
            max_quad,
            n_min_quad,
            "-0x1.ffffffffffffffffffffffffffffp+16383",
            Status::INEXACT,
            Category::Normal,
            Round::TowardPositive,
        ),
        (max_quad, n_min_quad, "-inf", overflow_status, Category::Infinity, Round::TowardNegative),
        (
            max_quad,
            n_min_quad,
            "-0x1.ffffffffffffffffffffffffffffp+16383",
            Status::INEXACT,
            Category::Normal,
            Round::TowardZero,
        ),
        (max_quad, n_min_quad, "-inf", overflow_status, Category::Infinity, Round::NearestTiesToAway),
        (min_quad, max_quad, "0", underflow_status, Category::Zero, Round::NearestTiesToEven),
        (
            min_quad,
            max_quad,
            "0x0.0000000000000000000000000001p-16382",
            underflow_status,
            Category::Normal,
            Round::TowardPositive,
        ),
        (min_quad, max_quad, "0", underflow_status, Category::Zero, Round::TowardNegative),
        (min_quad, max_quad, "0", underflow_status, Category::Zero, Round::TowardZero),
        (min_quad, max_quad, "0", underflow_status, Category::Zero, Round::NearestTiesToAway),
        (n_min_quad, max_quad, "-0", underflow_status, Category::Zero, Round::NearestTiesToEven),
        (n_min_quad, max_quad, "-0", underflow_status, Category::Zero, Round::TowardPositive),
        (
            n_min_quad,
            max_quad,
            "-0x0.0000000000000000000000000001p-16382",
            underflow_status,
            Category::Normal,
            Round::TowardNegative,
        ),
        (n_min_quad, max_quad, "-0", underflow_status, Category::Zero, Round::TowardZero),
        (n_min_quad, max_quad, "-0", underflow_status, Category::Zero, Round::NearestTiesToAway),
    ];

    for case @ &(x, y, e_result, e_status, e_category) in &single_special_cases {
        let status;
        let result = unpack!(status=, x / y);
        assert_eq!(e_status, status);
        assert_eq!(e_category, result.category());
        assert!(result.bitwise_eq(e_result.parse::<Single>().unwrap()), "result = {result:?}, case = {case:?}");
    }
    for case @ &(x, y, e_result, e_status, e_category, round) in &quad_special_cases {
        let status;
        let result = unpack!(status=, x.div_r(y, round));
        assert_eq!(e_status, status);
        assert_eq!(e_category, result.category());
        assert!(result.bitwise_eq(e_result.parse::<Quad>().unwrap()), "result = {result:?}, case = {case:?}");
    }
}

#[test]
fn operator_overloads() {
    // This is mostly testing that these operator overloads compile.
    let one = "0x1p+0".parse::<Single>().unwrap();
    let two = "0x2p+0".parse::<Single>().unwrap();
    assert!(two.bitwise_eq((one + one).value));
    assert!(one.bitwise_eq((two - one).value));
    assert!(two.bitwise_eq((one * two).value));
    assert!(one.bitwise_eq((two / two).value));
}

#[test]
fn comparisons() {
    let vals = [
        /* MNan */ -Single::NAN,
        /* MInf */ -Single::INFINITY,
        /* MBig */ -Single::largest(),
        /* MOne */ "-0x1p+0".parse::<Single>().unwrap(),
        /* MZer */ -Single::ZERO,
        /* PZer */ Single::ZERO,
        /* POne */ "0x1p+0".parse::<Single>().unwrap(),
        /* PBig */ Single::largest(),
        /* PInf */ Single::INFINITY,
        /* PNan */ Single::NAN,
    ];

    const LT: Option<Ordering> = Some(Ordering::Less);
    const EQ: Option<Ordering> = Some(Ordering::Equal);
    const GT: Option<Ordering> = Some(Ordering::Greater);
    const UN: Option<Ordering> = None;

    // HACK(eddyb) for some reason the first row (MNan) gets formatted differently.
    #[rustfmt::skip]
    let relations = [
        //          -N  -I  -B  -1  -0  +0  +1  +B  +I  +N
        /* MNan */ [UN, UN, UN, UN, UN, UN, UN, UN, UN, UN],
        /* MInf */ [UN, EQ, LT, LT, LT, LT, LT, LT, LT, UN],
        /* MBig */ [UN, GT, EQ, LT, LT, LT, LT, LT, LT, UN],
        /* MOne */ [UN, GT, GT, EQ, LT, LT, LT, LT, LT, UN],
        /* MZer */ [UN, GT, GT, GT, EQ, EQ, LT, LT, LT, UN],
        /* PZer */ [UN, GT, GT, GT, EQ, EQ, LT, LT, LT, UN],
        /* POne */ [UN, GT, GT, GT, GT, GT, EQ, LT, LT, UN],
        /* PBig */ [UN, GT, GT, GT, GT, GT, GT, EQ, LT, UN],
        /* PInf */ [UN, GT, GT, GT, GT, GT, GT, GT, EQ, UN],
        /* PNan */ [UN, UN, UN, UN, UN, UN, UN, UN, UN, UN],
    ];
    for (i, &lhs) in vals.iter().enumerate() {
        for (j, &rhs) in vals.iter().enumerate() {
            let relation = lhs.partial_cmp(&rhs);
            assert_eq!(relation, relations[i][j]);

            // NOTE(eddyb) these checks have been kept from the C++ code which didn't
            // appear to have a concept like `Option<Ordering>`, but in Rust they
            // should be entirely redundant with the single `assert_eq!` above.
            match relation {
                LT => {
                    assert!(!(lhs == rhs));
                    assert!(lhs != rhs);
                    assert!(lhs < rhs);
                    assert!(!(lhs > rhs));
                    assert!(lhs <= rhs);
                    assert!(!(lhs >= rhs));
                }
                EQ => {
                    assert!(lhs == rhs);
                    assert!(!(lhs != rhs));
                    assert!(!(lhs < rhs));
                    assert!(!(lhs > rhs));
                    assert!(lhs <= rhs);
                    assert!(lhs >= rhs);
                }
                GT => {
                    assert!(!(lhs == rhs));
                    assert!(lhs != rhs);
                    assert!(!(lhs < rhs));
                    assert!(lhs > rhs);
                    assert!(!(lhs <= rhs));
                    assert!(lhs >= rhs);
                }
                UN => {
                    assert!(!(lhs == rhs));
                    assert!(lhs != rhs);
                    assert!(!(lhs < rhs));
                    assert!(!(lhs > rhs));
                    assert!(!(lhs <= rhs));
                    assert!(!(lhs >= rhs));
                }
            }
        }
    }
}

#[test]
fn abs() {
    let p_inf = Single::INFINITY;
    let m_inf = -Single::INFINITY;
    let p_zero = Single::ZERO;
    let m_zero = -Single::ZERO;
    let p_qnan = Single::NAN;
    let m_qnan = -Single::NAN;
    let p_snan = Single::snan(None);
    let m_snan = -Single::snan(None);
    let p_normal_value = "0x1p+0".parse::<Single>().unwrap();
    let m_normal_value = "-0x1p+0".parse::<Single>().unwrap();
    let p_largest_value = Single::largest();
    let m_largest_value = -Single::largest();
    let p_smallest_value = Single::SMALLEST;
    let m_smallest_value = -Single::SMALLEST;
    let p_smallest_normalized = Single::smallest_normalized();
    let m_smallest_normalized = -Single::smallest_normalized();

    assert!(p_inf.bitwise_eq(p_inf.abs()));
    assert!(p_inf.bitwise_eq(m_inf.abs()));
    assert!(p_zero.bitwise_eq(p_zero.abs()));
    assert!(p_zero.bitwise_eq(m_zero.abs()));
    assert!(p_qnan.bitwise_eq(p_qnan.abs()));
    assert!(p_qnan.bitwise_eq(m_qnan.abs()));
    assert!(p_snan.bitwise_eq(p_snan.abs()));
    assert!(p_snan.bitwise_eq(m_snan.abs()));
    assert!(p_normal_value.bitwise_eq(p_normal_value.abs()));
    assert!(p_normal_value.bitwise_eq(m_normal_value.abs()));
    assert!(p_largest_value.bitwise_eq(p_largest_value.abs()));
    assert!(p_largest_value.bitwise_eq(m_largest_value.abs()));
    assert!(p_smallest_value.bitwise_eq(p_smallest_value.abs()));
    assert!(p_smallest_value.bitwise_eq(m_smallest_value.abs()));
    assert!(p_smallest_normalized.bitwise_eq(p_smallest_normalized.abs(),));
    assert!(p_smallest_normalized.bitwise_eq(m_smallest_normalized.abs(),));
}

#[test]
fn neg() {
    let one = "1.0".parse::<Single>().unwrap();
    let neg_one = "-1.0".parse::<Single>().unwrap();
    let zero = Single::ZERO;
    let neg_zero = -Single::ZERO;
    let inf = Single::INFINITY;
    let neg_inf = -Single::INFINITY;
    let qnan = Single::NAN;
    let neg_qnan = -Single::NAN;

    assert!(neg_one.bitwise_eq(-one));
    assert!(one.bitwise_eq(-neg_one));
    assert!(neg_zero.bitwise_eq(-zero));
    assert!(zero.bitwise_eq(-neg_zero));
    assert!(neg_inf.bitwise_eq(-inf));
    assert!(inf.bitwise_eq(-neg_inf));
    assert!(neg_inf.bitwise_eq(-inf));
    assert!(inf.bitwise_eq(-neg_inf));
    assert!(neg_qnan.bitwise_eq(-qnan));
    assert!(qnan.bitwise_eq(-neg_qnan));
}

#[test]
fn ilogb() {
    assert_eq!(-1074, Double::SMALLEST.ilogb());
    assert_eq!(-1074, (-Double::SMALLEST).ilogb());
    assert_eq!(-1023, "0x1.ffffffffffffep-1024".parse::<Double>().unwrap().ilogb());
    assert_eq!(-1023, "0x1.ffffffffffffep-1023".parse::<Double>().unwrap().ilogb());
    assert_eq!(-1023, "-0x1.ffffffffffffep-1023".parse::<Double>().unwrap().ilogb());
    assert_eq!(-51, "0x1p-51".parse::<Double>().unwrap().ilogb());
    assert_eq!(-1023, "0x1.c60f120d9f87cp-1023".parse::<Double>().unwrap().ilogb());
    assert_eq!(-2, "0x0.ffffp-1".parse::<Double>().unwrap().ilogb());
    assert_eq!(-1023, "0x1.fffep-1023".parse::<Double>().unwrap().ilogb());
    assert_eq!(1023, Double::largest().ilogb());
    assert_eq!(1023, (-Double::largest()).ilogb());

    assert_eq!(0, "0x1p+0".parse::<Single>().unwrap().ilogb());
    assert_eq!(0, "-0x1p+0".parse::<Single>().unwrap().ilogb());
    assert_eq!(42, "0x1p+42".parse::<Single>().unwrap().ilogb());
    assert_eq!(-42, "0x1p-42".parse::<Single>().unwrap().ilogb());

    assert_eq!(IEK_INF, Single::INFINITY.ilogb());
    assert_eq!(IEK_INF, (-Single::INFINITY).ilogb());
    assert_eq!(IEK_ZERO, Single::ZERO.ilogb());
    assert_eq!(IEK_ZERO, (-Single::ZERO).ilogb());
    assert_eq!(IEK_NAN, Single::NAN.ilogb());
    assert_eq!(IEK_NAN, Single::snan(None).ilogb());

    assert_eq!(127, Single::largest().ilogb());
    assert_eq!(127, (-Single::largest()).ilogb());

    assert_eq!(-149, Single::SMALLEST.ilogb());
    assert_eq!(-149, (-Single::SMALLEST).ilogb());
    assert_eq!(-126, Single::smallest_normalized().ilogb());
    assert_eq!(-126, (-Single::smallest_normalized()).ilogb());
}

#[test]
fn scalbn() {
    assert!("0x1p+0"
        .parse::<Single>()
        .unwrap()
        .bitwise_eq("0x1p+0".parse::<Single>().unwrap().scalbn(0),));
    assert!("0x1p+42"
        .parse::<Single>()
        .unwrap()
        .bitwise_eq("0x1p+0".parse::<Single>().unwrap().scalbn(42),));
    assert!("0x1p-42"
        .parse::<Single>()
        .unwrap()
        .bitwise_eq("0x1p+0".parse::<Single>().unwrap().scalbn(-42),));

    let p_inf = Single::INFINITY;
    let m_inf = -Single::INFINITY;
    let p_zero = Single::ZERO;
    let m_zero = -Single::ZERO;
    let p_qnan = Single::NAN;
    let m_qnan = -Single::NAN;
    let snan = Single::snan(None);

    assert!(p_inf.bitwise_eq(p_inf.scalbn(0)));
    assert!(m_inf.bitwise_eq(m_inf.scalbn(0)));
    assert!(p_zero.bitwise_eq(p_zero.scalbn(0)));
    assert!(m_zero.bitwise_eq(m_zero.scalbn(0)));
    assert!(p_qnan.bitwise_eq(p_qnan.scalbn(0)));
    assert!(m_qnan.bitwise_eq(m_qnan.scalbn(0)));
    assert!(!snan.scalbn(0).is_signaling());

    let scalbn_snan = snan.scalbn(1);
    assert!(scalbn_snan.is_nan() && !scalbn_snan.is_signaling());

    // Make sure highest bit of payload is preserved.
    let payload = (1 << 50) | (1 << 49) | (1234 << 32) | 1;

    let snan_with_payload = Double::snan(Some(payload));
    let quiet_payload = snan_with_payload.scalbn(1);
    assert!(quiet_payload.is_nan() && !quiet_payload.is_signaling());
    assert_eq!(payload, quiet_payload.to_bits() & ((1 << 51) - 1));

    assert!(p_inf.bitwise_eq("0x1p+0".parse::<Single>().unwrap().scalbn(128),));
    assert!(m_inf.bitwise_eq("-0x1p+0".parse::<Single>().unwrap().scalbn(128),));
    assert!(p_inf.bitwise_eq("0x1p+127".parse::<Single>().unwrap().scalbn(1),));
    assert!(p_zero.bitwise_eq("0x1p-127".parse::<Single>().unwrap().scalbn(-127),));
    assert!(m_zero.bitwise_eq("-0x1p-127".parse::<Single>().unwrap().scalbn(-127),));
    assert!("-0x1p-149"
        .parse::<Single>()
        .unwrap()
        .bitwise_eq("-0x1p-127".parse::<Single>().unwrap().scalbn(-22),));
    assert!(p_zero.bitwise_eq("0x1p-126".parse::<Single>().unwrap().scalbn(-24),));

    let smallest_f64 = Double::SMALLEST;
    let neg_smallest_f64 = -Double::SMALLEST;

    let largest_f64 = Double::largest();
    let neg_largest_f64 = -Double::largest();

    let largest_denormal_f64 = "0x1.ffffffffffffep-1023".parse::<Double>().unwrap();
    let neg_largest_denormal_f64 = "-0x1.ffffffffffffep-1023".parse::<Double>().unwrap();

    assert!(smallest_f64.bitwise_eq("0x1p-1074".parse::<Double>().unwrap().scalbn(0),));
    assert!(neg_smallest_f64.bitwise_eq("-0x1p-1074".parse::<Double>().unwrap().scalbn(0),));

    assert!("0x1p+1023"
        .parse::<Double>()
        .unwrap()
        .bitwise_eq(smallest_f64.scalbn(2097,),));

    assert!(smallest_f64.scalbn(-2097).is_pos_zero());
    assert!(smallest_f64.scalbn(-2098).is_pos_zero());
    assert!(smallest_f64.scalbn(-2099).is_pos_zero());
    assert!("0x1p+1022"
        .parse::<Double>()
        .unwrap()
        .bitwise_eq(smallest_f64.scalbn(2096,),));
    assert!("0x1p+1023"
        .parse::<Double>()
        .unwrap()
        .bitwise_eq(smallest_f64.scalbn(2097,),));
    assert!(smallest_f64.scalbn(2098).is_infinite());
    assert!(smallest_f64.scalbn(2099).is_infinite());

    // Test for integer overflows when adding to exponent.
    assert!(smallest_f64.scalbn(-ExpInt::max_value()).is_pos_zero());
    assert!(largest_f64.scalbn(ExpInt::max_value()).is_infinite());

    assert!(largest_denormal_f64.bitwise_eq(largest_denormal_f64.scalbn(0),));
    assert!(neg_largest_denormal_f64.bitwise_eq(neg_largest_denormal_f64.scalbn(0),));

    assert!("0x1.ffffffffffffep-1022"
        .parse::<Double>()
        .unwrap()
        .bitwise_eq(largest_denormal_f64.scalbn(1)));
    assert!("-0x1.ffffffffffffep-1021"
        .parse::<Double>()
        .unwrap()
        .bitwise_eq(neg_largest_denormal_f64.scalbn(2)));

    assert!("0x1.ffffffffffffep+1"
        .parse::<Double>()
        .unwrap()
        .bitwise_eq(largest_denormal_f64.scalbn(1024)));
    assert!(largest_denormal_f64.scalbn(-1023).is_pos_zero());
    assert!(largest_denormal_f64.scalbn(-1024).is_pos_zero());
    assert!(largest_denormal_f64.scalbn(-2048).is_pos_zero());
    assert!(largest_denormal_f64.scalbn(2047).is_infinite());
    assert!(largest_denormal_f64.scalbn(2098).is_infinite());
    assert!(largest_denormal_f64.scalbn(2099).is_infinite());

    assert!("0x1.ffffffffffffep-2"
        .parse::<Double>()
        .unwrap()
        .bitwise_eq(largest_denormal_f64.scalbn(1021)));
    assert!("0x1.ffffffffffffep-1"
        .parse::<Double>()
        .unwrap()
        .bitwise_eq(largest_denormal_f64.scalbn(1022)));
    assert!("0x1.ffffffffffffep+0"
        .parse::<Double>()
        .unwrap()
        .bitwise_eq(largest_denormal_f64.scalbn(1023)));
    assert!("0x1.ffffffffffffep+1023"
        .parse::<Double>()
        .unwrap()
        .bitwise_eq(largest_denormal_f64.scalbn(2046)));
    assert!("0x1p+974"
        .parse::<Double>()
        .unwrap()
        .bitwise_eq(smallest_f64.scalbn(2048,),));

    let random_denormal_f64 = "0x1.c60f120d9f87cp+51".parse::<Double>().unwrap();
    assert!("0x1.c60f120d9f87cp-972"
        .parse::<Double>()
        .unwrap()
        .bitwise_eq(random_denormal_f64.scalbn(-1023)));
    assert!("0x1.c60f120d9f87cp-1"
        .parse::<Double>()
        .unwrap()
        .bitwise_eq(random_denormal_f64.scalbn(-52)));
    assert!("0x1.c60f120d9f87cp-2"
        .parse::<Double>()
        .unwrap()
        .bitwise_eq(random_denormal_f64.scalbn(-53)));
    assert!("0x1.c60f120d9f87cp+0"
        .parse::<Double>()
        .unwrap()
        .bitwise_eq(random_denormal_f64.scalbn(-51)));

    assert!(random_denormal_f64.scalbn(-2097).is_pos_zero());
    assert!(random_denormal_f64.scalbn(-2090).is_pos_zero());

    assert!("-0x1p-1073"
        .parse::<Double>()
        .unwrap()
        .bitwise_eq(neg_largest_f64.scalbn(-2097),));

    assert!("-0x1p-1024"
        .parse::<Double>()
        .unwrap()
        .bitwise_eq(neg_largest_f64.scalbn(-2048),));

    assert!("0x1p-1073"
        .parse::<Double>()
        .unwrap()
        .bitwise_eq(largest_f64.scalbn(-2097,),));

    assert!("0x1p-1074"
        .parse::<Double>()
        .unwrap()
        .bitwise_eq(largest_f64.scalbn(-2098,),));
    assert!("-0x1p-1074"
        .parse::<Double>()
        .unwrap()
        .bitwise_eq(neg_largest_f64.scalbn(-2098),));
    assert!(neg_largest_f64.scalbn(-2099).is_neg_zero());
    assert!(largest_f64.scalbn(1).is_infinite());

    assert!("0x1p+0"
        .parse::<Double>()
        .unwrap()
        .bitwise_eq("0x1p+52".parse::<Double>().unwrap().scalbn(-52),));

    assert!("0x1p-103"
        .parse::<Double>()
        .unwrap()
        .bitwise_eq("0x1p-51".parse::<Double>().unwrap().scalbn(-52),));
}

#[test]
fn frexp() {
    let p_zero = Double::ZERO;
    let m_zero = -Double::ZERO;
    let one = Double::from_f64(1.0);
    let m_one = Double::from_f64(-1.0);

    let largest_denormal = "0x1.ffffffffffffep-1023".parse::<Double>().unwrap();
    let neg_largest_denormal = "-0x1.ffffffffffffep-1023".parse::<Double>().unwrap();

    let smallest = Double::SMALLEST;
    let neg_smallest = -Double::SMALLEST;

    let largest = Double::largest();
    let neg_largest = -Double::largest();

    let p_inf = Double::INFINITY;
    let m_inf = -Double::INFINITY;

    let p_qnan = Double::NAN;
    let m_qnan = -Double::NAN;
    let snan = Double::snan(None);

    // Make sure highest bit of payload is preserved.
    let payload = (1 << 50) | (1 << 49) | (1234 << 32) | 1;

    let snan_with_payload = Double::snan(Some(payload));

    let mut exp = 0;

    let frac = p_zero.frexp(&mut exp);
    assert_eq!(0, exp);
    assert!(frac.is_pos_zero());

    let frac = m_zero.frexp(&mut exp);
    assert_eq!(0, exp);
    assert!(frac.is_neg_zero());

    let frac = one.frexp(&mut exp);
    assert_eq!(1, exp);
    assert!("0x1p-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = m_one.frexp(&mut exp);
    assert_eq!(1, exp);
    assert!("-0x1p-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = largest_denormal.frexp(&mut exp);
    assert_eq!(-1022, exp);
    assert!("0x1.ffffffffffffep-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = neg_largest_denormal.frexp(&mut exp);
    assert_eq!(-1022, exp);
    assert!("-0x1.ffffffffffffep-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = smallest.frexp(&mut exp);
    assert_eq!(-1073, exp);
    assert!("0x1p-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = neg_smallest.frexp(&mut exp);
    assert_eq!(-1073, exp);
    assert!("-0x1p-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = largest.frexp(&mut exp);
    assert_eq!(1024, exp);
    assert!("0x1.fffffffffffffp-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = neg_largest.frexp(&mut exp);
    assert_eq!(1024, exp);
    assert!("-0x1.fffffffffffffp-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = p_inf.frexp(&mut exp);
    assert_eq!(IEK_INF, exp);
    assert!(frac.is_infinite() && !frac.is_negative());

    let frac = m_inf.frexp(&mut exp);
    assert_eq!(IEK_INF, exp);
    assert!(frac.is_infinite() && frac.is_negative());

    let frac = p_qnan.frexp(&mut exp);
    assert_eq!(IEK_NAN, exp);
    assert!(frac.is_nan());

    let frac = m_qnan.frexp(&mut exp);
    assert_eq!(IEK_NAN, exp);
    assert!(frac.is_nan());

    let frac = snan.frexp(&mut exp);
    assert_eq!(IEK_NAN, exp);
    assert!(frac.is_nan() && !frac.is_signaling());

    let frac = snan_with_payload.frexp(&mut exp);
    assert_eq!(IEK_NAN, exp);
    assert!(frac.is_nan() && !frac.is_signaling());
    assert_eq!(payload, frac.to_bits() & ((1 << 51) - 1));

    let frac = "0x0.ffffp-1".parse::<Double>().unwrap().frexp(&mut exp);
    assert_eq!(-1, exp);
    assert!("0x1.fffep-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = "0x1p-51".parse::<Double>().unwrap().frexp(&mut exp);
    assert_eq!(-50, exp);
    assert!("0x1p-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = "0x1.c60f120d9f87cp+51".parse::<Double>().unwrap().frexp(&mut exp);
    assert_eq!(52, exp);
    assert!("0x1.c60f120d9f87cp-1".parse::<Double>().unwrap().bitwise_eq(frac));
}

#[test]
fn modulo() {
    let mut status;
    {
        let f1 = "1.5".parse::<Double>().unwrap();
        let f2 = "1.0".parse::<Double>().unwrap();
        let expected = "0.5".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1 % f2).bitwise_eq(expected));
        assert_eq!(status, Status::OK);
    }
    {
        let f1 = "0.5".parse::<Double>().unwrap();
        let f2 = "1.0".parse::<Double>().unwrap();
        let expected = "0.5".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1 % f2).bitwise_eq(expected));
        assert_eq!(status, Status::OK);
    }
    {
        let f1 = "0x1.3333333333333p-2".parse::<Double>().unwrap(); // 0.3
        let f2 = "0x1.47ae147ae147bp-7".parse::<Double>().unwrap(); // 0.01
                                                                    // 0.009999999999999983
        let expected = "0x1.47ae147ae1471p-7".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1 % f2).bitwise_eq(expected));
        assert_eq!(status, Status::OK);
    }
    {
        let f1 = "0x1p64".parse::<Double>().unwrap(); // 1.8446744073709552e19
        let f2 = "1.5".parse::<Double>().unwrap();
        let expected = "1.0".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1 % f2).bitwise_eq(expected));
        assert_eq!(status, Status::OK);
    }
    {
        let f1 = "0x1p1000".parse::<Double>().unwrap();
        let f2 = "0x1p-1000".parse::<Double>().unwrap();
        let expected = "0.0".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1 % f2).bitwise_eq(expected));
        assert_eq!(status, Status::OK);
    }
    {
        let f1 = "0.0".parse::<Double>().unwrap();
        let f2 = "1.0".parse::<Double>().unwrap();
        let expected = "0.0".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1 % f2).bitwise_eq(expected));
        assert_eq!(status, Status::OK);
    }
    {
        let f1 = "1.0".parse::<Double>().unwrap();
        let f2 = "0.0".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1 % f2).is_nan());
        assert_eq!(status, Status::INVALID_OP);
    }
    {
        let f1 = "0.0".parse::<Double>().unwrap();
        let f2 = "0.0".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1 % f2).is_nan());
        assert_eq!(status, Status::INVALID_OP);
    }
    {
        let f1 = Double::INFINITY;
        let f2 = "1.0".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1 % f2).is_nan());
        assert_eq!(status, Status::INVALID_OP);
    }
    {
        let f1 = "-4.0".parse::<Double>().unwrap();
        let f2 = "-2.0".parse::<Double>().unwrap();
        let expected = "-0.0".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1 % f2).bitwise_eq(expected));
        assert_eq!(status, Status::OK);
    }
    {
        let f1 = "-4.0".parse::<Double>().unwrap();
        let f2 = "2.0".parse::<Double>().unwrap();
        let expected = "-0.0".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1 % f2).bitwise_eq(expected));
        assert_eq!(status, Status::OK);
    }
    {
        // Test E4M3FN mod where the LHS exponent is maxExponent (8) and the RHS is
        // the max value whose exponent is minExponent (-6). This requires special
        // logic in the mod implementation to prevent overflow to NaN.
        let f1 = "0x1p8".parse::<Float8E4M3FN>().unwrap(); // 256
        let f2 = "0x1.ep-6".parse::<Float8E4M3FN>().unwrap(); // 0.029296875
        let expected = "0x1p-8".parse::<Float8E4M3FN>().unwrap(); // 0.00390625
        assert!(unpack!(status=, f1 % f2).bitwise_eq(expected));
        assert_eq!(status, Status::OK);
    }
}

#[test]
fn remainder() {
    // Test Special Cases against each other and normal values.

    let p_inf = Single::INFINITY;
    let m_inf = -Single::INFINITY;
    let p_zero = Single::ZERO;
    let m_zero = -Single::ZERO;
    let qnan = Single::NAN;
    let snan = "snan123".parse::<Single>().unwrap();
    let p_normal_value = "0x1p+0".parse::<Single>().unwrap();
    let m_normal_value = "-0x1p+0".parse::<Single>().unwrap();
    let p_largest_value = Single::largest();
    let m_largest_value = -Single::largest();
    let p_smallest_value = Single::SMALLEST;
    let m_smallest_value = -Single::SMALLEST;
    let p_smallest_normalized = Single::smallest_normalized();
    let m_smallest_normalized = -Single::smallest_normalized();

    let p_val1 = "0x1.fffffep+126".parse::<Single>().unwrap();
    let m_val1 = "-0x1.fffffep+126".parse::<Single>().unwrap();
    let p_val2 = "0x1.fffffep-126".parse::<Single>().unwrap();
    let m_val2 = "-0x1.fffffep-126".parse::<Single>().unwrap();
    let p_val3 = "0x1p-125".parse::<Single>().unwrap();
    let m_val3 = "-0x1p-125".parse::<Single>().unwrap();
    let p_val4 = "0x1p+127".parse::<Single>().unwrap();
    let m_val4 = "-0x1p+127".parse::<Single>().unwrap();
    let p_val5 = "1.5".parse::<Single>().unwrap();
    let m_val5 = "-1.5".parse::<Single>().unwrap();
    let p_val6 = "1".parse::<Single>().unwrap();
    let m_val6 = "-1".parse::<Single>().unwrap();

    let special_cases = [
        (p_inf, p_inf, "nan", Status::INVALID_OP, Category::NaN),
        (p_inf, m_inf, "nan", Status::INVALID_OP, Category::NaN),
        (p_inf, p_zero, "nan", Status::INVALID_OP, Category::NaN),
        (p_inf, m_zero, "nan", Status::INVALID_OP, Category::NaN),
        (p_inf, qnan, "nan", Status::OK, Category::NaN),
        (p_inf, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (p_inf, p_normal_value, "nan", Status::INVALID_OP, Category::NaN),
        (p_inf, m_normal_value, "nan", Status::INVALID_OP, Category::NaN),
        (p_inf, p_largest_value, "nan", Status::INVALID_OP, Category::NaN),
        (p_inf, m_largest_value, "nan", Status::INVALID_OP, Category::NaN),
        (p_inf, p_smallest_value, "nan", Status::INVALID_OP, Category::NaN),
        (p_inf, m_smallest_value, "nan", Status::INVALID_OP, Category::NaN),
        (p_inf, p_smallest_normalized, "nan", Status::INVALID_OP, Category::NaN),
        (p_inf, m_smallest_normalized, "nan", Status::INVALID_OP, Category::NaN),
        (m_inf, p_inf, "nan", Status::INVALID_OP, Category::NaN),
        (m_inf, m_inf, "nan", Status::INVALID_OP, Category::NaN),
        (m_inf, p_zero, "nan", Status::INVALID_OP, Category::NaN),
        (m_inf, m_zero, "nan", Status::INVALID_OP, Category::NaN),
        (m_inf, qnan, "nan", Status::OK, Category::NaN),
        (m_inf, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (m_inf, p_normal_value, "nan", Status::INVALID_OP, Category::NaN),
        (m_inf, m_normal_value, "nan", Status::INVALID_OP, Category::NaN),
        (m_inf, p_largest_value, "nan", Status::INVALID_OP, Category::NaN),
        (m_inf, m_largest_value, "nan", Status::INVALID_OP, Category::NaN),
        (m_inf, p_smallest_value, "nan", Status::INVALID_OP, Category::NaN),
        (m_inf, m_smallest_value, "nan", Status::INVALID_OP, Category::NaN),
        (m_inf, p_smallest_normalized, "nan", Status::INVALID_OP, Category::NaN),
        (m_inf, m_smallest_normalized, "nan", Status::INVALID_OP, Category::NaN),
        (p_zero, p_inf, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_inf, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, p_zero, "nan", Status::INVALID_OP, Category::NaN),
        (p_zero, m_zero, "nan", Status::INVALID_OP, Category::NaN),
        (p_zero, qnan, "nan", Status::OK, Category::NaN),
        (p_zero, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (p_zero, p_normal_value, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_normal_value, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, p_largest_value, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_largest_value, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, p_smallest_value, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_smallest_value, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, p_smallest_normalized, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_smallest_normalized, "0x0p+0", Status::OK, Category::Zero),
        (m_zero, p_inf, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_inf, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, p_zero, "nan", Status::INVALID_OP, Category::NaN),
        (m_zero, m_zero, "nan", Status::INVALID_OP, Category::NaN),
        (m_zero, qnan, "nan", Status::OK, Category::NaN),
        (m_zero, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (m_zero, p_normal_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_normal_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, p_largest_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_largest_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, p_smallest_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_smallest_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, p_smallest_normalized, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_smallest_normalized, "-0x0p+0", Status::OK, Category::Zero),
        (qnan, p_inf, "nan", Status::OK, Category::NaN),
        (qnan, m_inf, "nan", Status::OK, Category::NaN),
        (qnan, p_zero, "nan", Status::OK, Category::NaN),
        (qnan, m_zero, "nan", Status::OK, Category::NaN),
        (qnan, qnan, "nan", Status::OK, Category::NaN),
        (qnan, snan, "nan", Status::INVALID_OP, Category::NaN),
        (qnan, p_normal_value, "nan", Status::OK, Category::NaN),
        (qnan, m_normal_value, "nan", Status::OK, Category::NaN),
        (qnan, p_largest_value, "nan", Status::OK, Category::NaN),
        (qnan, m_largest_value, "nan", Status::OK, Category::NaN),
        (qnan, p_smallest_value, "nan", Status::OK, Category::NaN),
        (qnan, m_smallest_value, "nan", Status::OK, Category::NaN),
        (qnan, p_smallest_normalized, "nan", Status::OK, Category::NaN),
        (qnan, m_smallest_normalized, "nan", Status::OK, Category::NaN),
        (snan, p_inf, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, m_inf, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, p_zero, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, m_zero, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, qnan, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, p_normal_value, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, m_normal_value, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, p_largest_value, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, m_largest_value, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, p_smallest_value, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, m_smallest_value, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, p_smallest_normalized, "nan123", Status::INVALID_OP, Category::NaN),
        (snan, m_smallest_normalized, "nan123", Status::INVALID_OP, Category::NaN),
        (p_normal_value, p_inf, "0x1p+0", Status::OK, Category::Normal),
        (p_normal_value, m_inf, "0x1p+0", Status::OK, Category::Normal),
        (p_normal_value, p_zero, "nan", Status::INVALID_OP, Category::NaN),
        (p_normal_value, m_zero, "nan", Status::INVALID_OP, Category::NaN),
        (p_normal_value, qnan, "nan", Status::OK, Category::NaN),
        (p_normal_value, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (p_normal_value, p_normal_value, "0x0p+0", Status::OK, Category::Zero),
        (p_normal_value, m_normal_value, "0x0p+0", Status::OK, Category::Zero),
        (p_normal_value, p_largest_value, "0x1p+0", Status::OK, Category::Normal),
        (p_normal_value, m_largest_value, "0x1p+0", Status::OK, Category::Normal),
        (p_normal_value, p_smallest_value, "0x0p+0", Status::OK, Category::Zero),
        (p_normal_value, m_smallest_value, "0x0p+0", Status::OK, Category::Zero),
        (p_normal_value, p_smallest_normalized, "0x0p+0", Status::OK, Category::Zero),
        (p_normal_value, m_smallest_normalized, "0x0p+0", Status::OK, Category::Zero),
        (m_normal_value, p_inf, "-0x1p+0", Status::OK, Category::Normal),
        (m_normal_value, m_inf, "-0x1p+0", Status::OK, Category::Normal),
        (m_normal_value, p_zero, "nan", Status::INVALID_OP, Category::NaN),
        (m_normal_value, m_zero, "nan", Status::INVALID_OP, Category::NaN),
        (m_normal_value, qnan, "nan", Status::OK, Category::NaN),
        (m_normal_value, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (m_normal_value, p_normal_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_normal_value, m_normal_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_normal_value, p_largest_value, "-0x1p+0", Status::OK, Category::Normal),
        (m_normal_value, m_largest_value, "-0x1p+0", Status::OK, Category::Normal),
        (m_normal_value, p_smallest_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_normal_value, m_smallest_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_normal_value, p_smallest_normalized, "-0x0p+0", Status::OK, Category::Zero),
        (m_normal_value, m_smallest_normalized, "-0x0p+0", Status::OK, Category::Zero),
        (p_largest_value, p_inf, "0x1.fffffep+127", Status::OK, Category::Normal),
        (p_largest_value, m_inf, "0x1.fffffep+127", Status::OK, Category::Normal),
        (p_largest_value, p_zero, "nan", Status::INVALID_OP, Category::NaN),
        (p_largest_value, m_zero, "nan", Status::INVALID_OP, Category::NaN),
        (p_largest_value, qnan, "nan", Status::OK, Category::NaN),
        (p_largest_value, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (p_largest_value, p_normal_value, "0x0p+0", Status::OK, Category::Zero),
        (p_largest_value, m_normal_value, "0x0p+0", Status::OK, Category::Zero),
        (p_largest_value, p_largest_value, "0x0p+0", Status::OK, Category::Zero),
        (p_largest_value, m_largest_value, "0x0p+0", Status::OK, Category::Zero),
        (p_largest_value, p_smallest_value, "0x0p+0", Status::OK, Category::Zero),
        (p_largest_value, m_smallest_value, "0x0p+0", Status::OK, Category::Zero),
        (p_largest_value, p_smallest_normalized, "0x0p+0", Status::OK, Category::Zero),
        (p_largest_value, m_smallest_normalized, "0x0p+0", Status::OK, Category::Zero),
        (m_largest_value, p_inf, "-0x1.fffffep+127", Status::OK, Category::Normal),
        (m_largest_value, m_inf, "-0x1.fffffep+127", Status::OK, Category::Normal),
        (m_largest_value, p_zero, "nan", Status::INVALID_OP, Category::NaN),
        (m_largest_value, m_zero, "nan", Status::INVALID_OP, Category::NaN),
        (m_largest_value, qnan, "nan", Status::OK, Category::NaN),
        (m_largest_value, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (m_largest_value, p_normal_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_largest_value, m_normal_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_largest_value, p_largest_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_largest_value, m_largest_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_largest_value, p_smallest_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_largest_value, m_smallest_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_largest_value, p_smallest_normalized, "-0x0p+0", Status::OK, Category::Zero),
        (m_largest_value, m_smallest_normalized, "-0x0p+0", Status::OK, Category::Zero),
        (p_smallest_value, p_inf, "0x1p-149", Status::OK, Category::Normal),
        (p_smallest_value, m_inf, "0x1p-149", Status::OK, Category::Normal),
        (p_smallest_value, p_zero, "nan", Status::INVALID_OP, Category::NaN),
        (p_smallest_value, m_zero, "nan", Status::INVALID_OP, Category::NaN),
        (p_smallest_value, qnan, "nan", Status::OK, Category::NaN),
        (p_smallest_value, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (p_smallest_value, p_normal_value, "0x1p-149", Status::OK, Category::Normal),
        (p_smallest_value, m_normal_value, "0x1p-149", Status::OK, Category::Normal),
        (p_smallest_value, p_largest_value, "0x1p-149", Status::OK, Category::Normal),
        (p_smallest_value, m_largest_value, "0x1p-149", Status::OK, Category::Normal),
        (p_smallest_value, p_smallest_value, "0x0p+0", Status::OK, Category::Zero),
        (p_smallest_value, m_smallest_value, "0x0p+0", Status::OK, Category::Zero),
        (p_smallest_value, p_smallest_normalized, "0x1p-149", Status::OK, Category::Normal),
        (p_smallest_value, m_smallest_normalized, "0x1p-149", Status::OK, Category::Normal),
        (m_smallest_value, p_inf, "-0x1p-149", Status::OK, Category::Normal),
        (m_smallest_value, m_inf, "-0x1p-149", Status::OK, Category::Normal),
        (m_smallest_value, p_zero, "nan", Status::INVALID_OP, Category::NaN),
        (m_smallest_value, m_zero, "nan", Status::INVALID_OP, Category::NaN),
        (m_smallest_value, qnan, "nan", Status::OK, Category::NaN),
        (m_smallest_value, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (m_smallest_value, p_normal_value, "-0x1p-149", Status::OK, Category::Normal),
        (m_smallest_value, m_normal_value, "-0x1p-149", Status::OK, Category::Normal),
        (m_smallest_value, p_largest_value, "-0x1p-149", Status::OK, Category::Normal),
        (m_smallest_value, m_largest_value, "-0x1p-149", Status::OK, Category::Normal),
        (m_smallest_value, p_smallest_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_smallest_value, m_smallest_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_smallest_value, p_smallest_normalized, "-0x1p-149", Status::OK, Category::Normal),
        (m_smallest_value, m_smallest_normalized, "-0x1p-149", Status::OK, Category::Normal),
        (p_smallest_normalized, p_inf, "0x1p-126", Status::OK, Category::Normal),
        (p_smallest_normalized, m_inf, "0x1p-126", Status::OK, Category::Normal),
        (p_smallest_normalized, p_zero, "nan", Status::INVALID_OP, Category::NaN),
        (p_smallest_normalized, m_zero, "nan", Status::INVALID_OP, Category::NaN),
        (p_smallest_normalized, qnan, "nan", Status::OK, Category::NaN),
        (p_smallest_normalized, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (p_smallest_normalized, p_normal_value, "0x1p-126", Status::OK, Category::Normal),
        (p_smallest_normalized, m_normal_value, "0x1p-126", Status::OK, Category::Normal),
        (p_smallest_normalized, p_largest_value, "0x1p-126", Status::OK, Category::Normal),
        (p_smallest_normalized, m_largest_value, "0x1p-126", Status::OK, Category::Normal),
        (p_smallest_normalized, p_smallest_value, "0x0p+0", Status::OK, Category::Zero),
        (p_smallest_normalized, m_smallest_value, "0x0p+0", Status::OK, Category::Zero),
        (p_smallest_normalized, p_smallest_normalized, "0x0p+0", Status::OK, Category::Zero),
        (p_smallest_normalized, m_smallest_normalized, "0x0p+0", Status::OK, Category::Zero),
        (m_smallest_normalized, p_inf, "-0x1p-126", Status::OK, Category::Normal),
        (m_smallest_normalized, m_inf, "-0x1p-126", Status::OK, Category::Normal),
        (m_smallest_normalized, p_zero, "nan", Status::INVALID_OP, Category::NaN),
        (m_smallest_normalized, m_zero, "nan", Status::INVALID_OP, Category::NaN),
        (m_smallest_normalized, qnan, "nan", Status::OK, Category::NaN),
        (m_smallest_normalized, snan, "nan123", Status::INVALID_OP, Category::NaN),
        (m_smallest_normalized, p_normal_value, "-0x1p-126", Status::OK, Category::Normal),
        (m_smallest_normalized, m_normal_value, "-0x1p-126", Status::OK, Category::Normal),
        (m_smallest_normalized, p_largest_value, "-0x1p-126", Status::OK, Category::Normal),
        (m_smallest_normalized, m_largest_value, "-0x1p-126", Status::OK, Category::Normal),
        (m_smallest_normalized, p_smallest_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_smallest_normalized, m_smallest_value, "-0x0p+0", Status::OK, Category::Zero),
        (m_smallest_normalized, p_smallest_normalized, "-0x0p+0", Status::OK, Category::Zero),
        (m_smallest_normalized, m_smallest_normalized, "-0x0p+0", Status::OK, Category::Zero),
        (p_val1, p_val1, "0x0p+0", Status::OK, Category::Zero),
        (p_val1, m_val1, "0x0p+0", Status::OK, Category::Zero),
        (p_val1, p_val2, "0x0p+0", Status::OK, Category::Zero),
        (p_val1, m_val2, "0x0p+0", Status::OK, Category::Zero),
        (p_val1, p_val3, "0x0p+0", Status::OK, Category::Zero),
        (p_val1, m_val3, "0x0p+0", Status::OK, Category::Zero),
        (p_val1, p_val4, "-0x1p+103", Status::OK, Category::Normal),
        (p_val1, m_val4, "-0x1p+103", Status::OK, Category::Normal),
        (p_val1, p_val5, "0x0p+0", Status::OK, Category::Zero),
        (p_val1, m_val5, "0x0p+0", Status::OK, Category::Zero),
        (p_val1, p_val6, "0x0p+0", Status::OK, Category::Zero),
        (p_val1, m_val6, "0x0p+0", Status::OK, Category::Zero),
        (m_val1, p_val1, "-0x0p+0", Status::OK, Category::Zero),
        (m_val1, m_val1, "-0x0p+0", Status::OK, Category::Zero),
        (m_val1, p_val2, "-0x0p+0", Status::OK, Category::Zero),
        (m_val1, m_val2, "-0x0p+0", Status::OK, Category::Zero),
        (m_val1, p_val3, "-0x0p+0", Status::OK, Category::Zero),
        (m_val1, m_val3, "-0x0p+0", Status::OK, Category::Zero),
        (m_val1, p_val4, "0x1p+103", Status::OK, Category::Normal),
        (m_val1, m_val4, "0x1p+103", Status::OK, Category::Normal),
        (m_val1, p_val5, "-0x0p+0", Status::OK, Category::Zero),
        (m_val1, m_val5, "-0x0p+0", Status::OK, Category::Zero),
        (m_val1, p_val6, "-0x0p+0", Status::OK, Category::Zero),
        (m_val1, m_val6, "-0x0p+0", Status::OK, Category::Zero),
        (p_val2, p_val1, "0x1.fffffep-126", Status::OK, Category::Normal),
        (p_val2, m_val1, "0x1.fffffep-126", Status::OK, Category::Normal),
        (p_val2, p_val2, "0x0p+0", Status::OK, Category::Zero),
        (p_val2, m_val2, "0x0p+0", Status::OK, Category::Zero),
        (p_val2, p_val3, "-0x0.000002p-126", Status::OK, Category::Normal),
        (p_val2, m_val3, "-0x0.000002p-126", Status::OK, Category::Normal),
        (p_val2, p_val4, "0x1.fffffep-126", Status::OK, Category::Normal),
        (p_val2, m_val4, "0x1.fffffep-126", Status::OK, Category::Normal),
        (p_val2, p_val5, "0x1.fffffep-126", Status::OK, Category::Normal),
        (p_val2, m_val5, "0x1.fffffep-126", Status::OK, Category::Normal),
        (p_val2, p_val6, "0x1.fffffep-126", Status::OK, Category::Normal),
        (p_val2, m_val6, "0x1.fffffep-126", Status::OK, Category::Normal),
        (m_val2, p_val1, "-0x1.fffffep-126", Status::OK, Category::Normal),
        (m_val2, m_val1, "-0x1.fffffep-126", Status::OK, Category::Normal),
        (m_val2, p_val2, "-0x0p+0", Status::OK, Category::Zero),
        (m_val2, m_val2, "-0x0p+0", Status::OK, Category::Zero),
        (m_val2, p_val3, "0x0.000002p-126", Status::OK, Category::Normal),
        (m_val2, m_val3, "0x0.000002p-126", Status::OK, Category::Normal),
        (m_val2, p_val4, "-0x1.fffffep-126", Status::OK, Category::Normal),
        (m_val2, m_val4, "-0x1.fffffep-126", Status::OK, Category::Normal),
        (m_val2, p_val5, "-0x1.fffffep-126", Status::OK, Category::Normal),
        (m_val2, m_val5, "-0x1.fffffep-126", Status::OK, Category::Normal),
        (m_val2, p_val6, "-0x1.fffffep-126", Status::OK, Category::Normal),
        (m_val2, m_val6, "-0x1.fffffep-126", Status::OK, Category::Normal),
        (p_val3, p_val1, "0x1p-125", Status::OK, Category::Normal),
        (p_val3, m_val1, "0x1p-125", Status::OK, Category::Normal),
        (p_val3, p_val2, "0x0.000002p-126", Status::OK, Category::Normal),
        (p_val3, m_val2, "0x0.000002p-126", Status::OK, Category::Normal),
        (p_val3, p_val3, "0x0p+0", Status::OK, Category::Zero),
        (p_val3, m_val3, "0x0p+0", Status::OK, Category::Zero),
        (p_val3, p_val4, "0x1p-125", Status::OK, Category::Normal),
        (p_val3, m_val4, "0x1p-125", Status::OK, Category::Normal),
        (p_val3, p_val5, "0x1p-125", Status::OK, Category::Normal),
        (p_val3, m_val5, "0x1p-125", Status::OK, Category::Normal),
        (p_val3, p_val6, "0x1p-125", Status::OK, Category::Normal),
        (p_val3, m_val6, "0x1p-125", Status::OK, Category::Normal),
        (m_val3, p_val1, "-0x1p-125", Status::OK, Category::Normal),
        (m_val3, m_val1, "-0x1p-125", Status::OK, Category::Normal),
        (m_val3, p_val2, "-0x0.000002p-126", Status::OK, Category::Normal),
        (m_val3, m_val2, "-0x0.000002p-126", Status::OK, Category::Normal),
        (m_val3, p_val3, "-0x0p+0", Status::OK, Category::Zero),
        (m_val3, m_val3, "-0x0p+0", Status::OK, Category::Zero),
        (m_val3, p_val4, "-0x1p-125", Status::OK, Category::Normal),
        (m_val3, m_val4, "-0x1p-125", Status::OK, Category::Normal),
        (m_val3, p_val5, "-0x1p-125", Status::OK, Category::Normal),
        (m_val3, m_val5, "-0x1p-125", Status::OK, Category::Normal),
        (m_val3, p_val6, "-0x1p-125", Status::OK, Category::Normal),
        (m_val3, m_val6, "-0x1p-125", Status::OK, Category::Normal),
        (p_val4, p_val1, "0x1p+103", Status::OK, Category::Normal),
        (p_val4, m_val1, "0x1p+103", Status::OK, Category::Normal),
        (p_val4, p_val2, "0x0.002p-126", Status::OK, Category::Normal),
        (p_val4, m_val2, "0x0.002p-126", Status::OK, Category::Normal),
        (p_val4, p_val3, "0x0p+0", Status::OK, Category::Zero),
        (p_val4, m_val3, "0x0p+0", Status::OK, Category::Zero),
        (p_val4, p_val4, "0x0p+0", Status::OK, Category::Zero),
        (p_val4, m_val4, "0x0p+0", Status::OK, Category::Zero),
        (p_val4, p_val5, "0.5", Status::OK, Category::Normal),
        (p_val4, m_val5, "0.5", Status::OK, Category::Normal),
        (p_val4, p_val6, "0x0p+0", Status::OK, Category::Zero),
        (p_val4, m_val6, "0x0p+0", Status::OK, Category::Zero),
        (m_val4, p_val1, "-0x1p+103", Status::OK, Category::Normal),
        (m_val4, m_val1, "-0x1p+103", Status::OK, Category::Normal),
        (m_val4, p_val2, "-0x0.002p-126", Status::OK, Category::Normal),
        (m_val4, m_val2, "-0x0.002p-126", Status::OK, Category::Normal),
        (m_val4, p_val3, "-0x0p+0", Status::OK, Category::Zero),
        (m_val4, m_val3, "-0x0p+0", Status::OK, Category::Zero),
        (m_val4, p_val4, "-0x0p+0", Status::OK, Category::Zero),
        (m_val4, m_val4, "-0x0p+0", Status::OK, Category::Zero),
        (m_val4, p_val5, "-0.5", Status::OK, Category::Normal),
        (m_val4, m_val5, "-0.5", Status::OK, Category::Normal),
        (m_val4, p_val6, "-0x0p+0", Status::OK, Category::Zero),
        (m_val4, m_val6, "-0x0p+0", Status::OK, Category::Zero),
        (p_val5, p_val1, "1.5", Status::OK, Category::Normal),
        (p_val5, m_val1, "1.5", Status::OK, Category::Normal),
        (p_val5, p_val2, "0x0.00006p-126", Status::OK, Category::Normal),
        (p_val5, m_val2, "0x0.00006p-126", Status::OK, Category::Normal),
        (p_val5, p_val3, "0x0p+0", Status::OK, Category::Zero),
        (p_val5, m_val3, "0x0p+0", Status::OK, Category::Zero),
        (p_val5, p_val4, "1.5", Status::OK, Category::Normal),
        (p_val5, m_val4, "1.5", Status::OK, Category::Normal),
        (p_val5, p_val5, "0x0p+0", Status::OK, Category::Zero),
        (p_val5, m_val5, "0x0p+0", Status::OK, Category::Zero),
        (p_val5, p_val6, "-0.5", Status::OK, Category::Normal),
        (p_val5, m_val6, "-0.5", Status::OK, Category::Normal),
        (m_val5, p_val1, "-1.5", Status::OK, Category::Normal),
        (m_val5, m_val1, "-1.5", Status::OK, Category::Normal),
        (m_val5, p_val2, "-0x0.00006p-126", Status::OK, Category::Normal),
        (m_val5, m_val2, "-0x0.00006p-126", Status::OK, Category::Normal),
        (m_val5, p_val3, "-0x0p+0", Status::OK, Category::Zero),
        (m_val5, m_val3, "-0x0p+0", Status::OK, Category::Zero),
        (m_val5, p_val4, "-1.5", Status::OK, Category::Normal),
        (m_val5, m_val4, "-1.5", Status::OK, Category::Normal),
        (m_val5, p_val5, "-0x0p+0", Status::OK, Category::Zero),
        (m_val5, m_val5, "-0x0p+0", Status::OK, Category::Zero),
        (m_val5, p_val6, "0.5", Status::OK, Category::Normal),
        (m_val5, m_val6, "0.5", Status::OK, Category::Normal),
        (p_val6, p_val1, "0x1p+0", Status::OK, Category::Normal),
        (p_val6, m_val1, "0x1p+0", Status::OK, Category::Normal),
        (p_val6, p_val2, "0x0.00004p-126", Status::OK, Category::Normal),
        (p_val6, m_val2, "0x0.00004p-126", Status::OK, Category::Normal),
        (p_val6, p_val3, "0x0p+0", Status::OK, Category::Zero),
        (p_val6, m_val3, "0x0p+0", Status::OK, Category::Zero),
        (p_val6, p_val4, "0x1p+0", Status::OK, Category::Normal),
        (p_val6, m_val4, "0x1p+0", Status::OK, Category::Normal),
        (p_val6, p_val5, "-0.5", Status::OK, Category::Normal),
        (p_val6, m_val5, "-0.5", Status::OK, Category::Normal),
        (p_val6, p_val6, "0x0p+0", Status::OK, Category::Zero),
        (p_val6, m_val6, "0x0p+0", Status::OK, Category::Zero),
        (m_val6, p_val1, "-0x1p+0", Status::OK, Category::Normal),
        (m_val6, m_val1, "-0x1p+0", Status::OK, Category::Normal),
        (m_val6, p_val2, "-0x0.00004p-126", Status::OK, Category::Normal),
        (m_val6, m_val2, "-0x0.00004p-126", Status::OK, Category::Normal),
        (m_val6, p_val3, "-0x0p+0", Status::OK, Category::Zero),
        (m_val6, m_val3, "-0x0p+0", Status::OK, Category::Zero),
        (m_val6, p_val4, "-0x1p+0", Status::OK, Category::Normal),
        (m_val6, m_val4, "-0x1p+0", Status::OK, Category::Normal),
        (m_val6, p_val5, "0.5", Status::OK, Category::Normal),
        (m_val6, m_val5, "0.5", Status::OK, Category::Normal),
        (m_val6, p_val6, "-0x0p+0", Status::OK, Category::Zero),
        (m_val6, m_val6, "-0x0p+0", Status::OK, Category::Zero),
    ];

    for case @ &(x, y, e_result, e_status, e_category) in &special_cases {
        let status;
        let result = unpack!(status=, x.ieee_rem(y));
        assert_eq!(e_status, status);
        assert_eq!(e_category, result.category());
        assert!(result.bitwise_eq(e_result.parse::<Single>().unwrap()), "result = {result:?}, case = {case:?}");
    }

    let mut status;
    {
        let f1 = "0x1.3333333333333p-2".parse::<Double>().unwrap(); // 0.3
        let f2 = "0x1.47ae147ae147bp-7".parse::<Double>().unwrap(); // 0.01
        let expected = "-0x1.4p-56".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1.ieee_rem(f2)).bitwise_eq(expected));
        assert_eq!(status, Status::OK);
    }
    {
        let f1 = "0x1p64".parse::<Double>().unwrap(); // 1.8446744073709552e19
        let f2 = "1.5".parse::<Double>().unwrap();
        let expected = "-0.5".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1.ieee_rem(f2)).bitwise_eq(expected));
        assert_eq!(status, Status::OK);
    }
    {
        let f1 = "0x1p1000".parse::<Double>().unwrap();
        let f2 = "0x1p-1000".parse::<Double>().unwrap();
        let expected = "0.0".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1.ieee_rem(f2)).bitwise_eq(expected));
        assert_eq!(status, Status::OK);
    }
    {
        let f1 = Double::INFINITY;
        let f2 = "1.0".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1.ieee_rem(f2)).is_nan());
        assert_eq!(status, Status::INVALID_OP);
    }
    {
        let f1 = "-4.0".parse::<Double>().unwrap();
        let f2 = "-2.0".parse::<Double>().unwrap();
        let expected = "-0.0".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1.ieee_rem(f2)).bitwise_eq(expected));
        assert_eq!(status, Status::OK);
    }
    {
        let f1 = "-4.0".parse::<Double>().unwrap();
        let f2 = "2.0".parse::<Double>().unwrap();
        let expected = "-0.0".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1.ieee_rem(f2)).bitwise_eq(expected));
        assert_eq!(status, Status::OK);
    }
}

#[test]
fn x87_largest() {
    assert!(X87DoubleExtended::largest().is_largest());
}

#[test]
fn x87_next() {
    assert_eq!("-1.0".parse::<X87DoubleExtended>().unwrap().next_up().value.ilogb(), -1);
}

#[test]
fn convert_e4m3fn_to_e5m2() {
    let mut status;
    let mut loses_info = false;

    let test = "1.0".parse::<Float8E4M3FN>().unwrap();
    let test: Float8E5M2 = unpack!(status=, test.convert(&mut loses_info));
    assert_eq!(1.0, test.to_f32());
    assert!(!loses_info);
    assert_eq!(status, Status::OK);

    let test = "0.0".parse::<Float8E4M3FN>().unwrap();
    let test: Float8E5M2 = unpack!(status=, test.convert(&mut loses_info));
    assert_eq!(0.0, test.to_f32());
    assert!(!loses_info);
    assert_eq!(status, Status::OK);

    let test = "0x1.2p0".parse::<Float8E4M3FN>().unwrap(); // 1.125
    let test: Float8E5M2 = unpack!(status=, test.convert(&mut loses_info));
    assert_eq!(/* 0x1.0p0 */ 1.0, test.to_f32());
    assert!(loses_info);
    assert_eq!(status, Status::INEXACT);

    let test = "0x1.6p0".parse::<Float8E4M3FN>().unwrap(); // 1.375
    let test: Float8E5M2 = unpack!(status=, test.convert(&mut loses_info));
    assert_eq!(/* 0x1.8p0 */ 1.5, test.to_f32());
    assert!(loses_info);
    assert_eq!(status, Status::INEXACT);

    // Convert E4M3 denormal to E5M2 normal. Should not be truncated, despite the
    // destination format having one fewer significand bit
    let test = "0x1.Cp-7".parse::<Float8E4M3FN>().unwrap();
    let test: Float8E5M2 = unpack!(status=, test.convert(&mut loses_info));
    assert_eq!(/* 0x1.Cp-7 */ 0.013671875, test.to_f32());
    assert!(!loses_info);
    assert_eq!(status, Status::OK);

    // Test convert from NaN
    let test = "nan".parse::<Float8E4M3FN>().unwrap();
    let test: Float8E5M2 = unpack!(status=, test.convert(&mut loses_info));
    assert!(test.to_f32().is_nan());
    assert!(!loses_info);
    assert_eq!(status, Status::OK);
}

#[test]
fn convert_e5m2_to_e4m3fn() {
    let mut status;
    let mut loses_info = false;

    let test = "1.0".parse::<Float8E5M2>().unwrap();
    let test: Float8E4M3FN = unpack!(status=, test.convert(&mut loses_info));
    assert_eq!(1.0, test.to_f32());
    assert!(!loses_info);
    assert_eq!(status, Status::OK);

    let test = "0.0".parse::<Float8E5M2>().unwrap();
    let test: Float8E4M3FN = unpack!(status=, test.convert(&mut loses_info));
    assert_eq!(0.0, test.to_f32());
    assert!(!loses_info);
    assert_eq!(status, Status::OK);

    let test = "0x1.Cp8".parse::<Float8E5M2>().unwrap(); // 448
    let test: Float8E4M3FN = unpack!(status=, test.convert(&mut loses_info));
    assert_eq!(/* 0x1.Cp8 */ 448.0, test.to_f32());
    assert!(!loses_info);
    assert_eq!(status, Status::OK);

    // Test overflow
    let test = "0x1.0p9".parse::<Float8E5M2>().unwrap(); // 512
    let test: Float8E4M3FN = unpack!(status=, test.convert(&mut loses_info));
    assert!(test.to_f32().is_nan());
    assert!(loses_info);
    assert_eq!(status, Status::OVERFLOW | Status::INEXACT);

    // Test underflow
    let test = "0x1.0p-10".parse::<Float8E5M2>().unwrap();
    let test: Float8E4M3FN = unpack!(status=, test.convert(&mut loses_info));
    assert_eq!(0., test.to_f32());
    assert!(loses_info);
    assert_eq!(status, Status::UNDERFLOW | Status::INEXACT);

    // Test rounding up to smallest denormal number
    let test = "0x1.8p-10".parse::<Float8E5M2>().unwrap();
    let test: Float8E4M3FN = unpack!(status=, test.convert(&mut loses_info));
    assert_eq!(/* 0x1.0p-9 */ 0.001953125, test.to_f32());
    assert!(loses_info);
    assert_eq!(status, Status::UNDERFLOW | Status::INEXACT);

    // Testing inexact rounding to denormal number
    let test = "0x1.8p-9".parse::<Float8E5M2>().unwrap();
    let test: Float8E4M3FN = unpack!(status=, test.convert(&mut loses_info));
    assert_eq!(/* 0x1.0p-8 */ 0.00390625, test.to_f32());
    assert!(loses_info);
    assert_eq!(status, Status::UNDERFLOW | Status::INEXACT);

    let nan = "nan".parse::<Float8E4M3FN>().unwrap();

    // Testing convert from Inf
    let test = "inf".parse::<Float8E5M2>().unwrap();
    let test: Float8E4M3FN = unpack!(status=, test.convert(&mut loses_info));
    assert!(test.to_f32().is_nan());
    assert!(loses_info);
    assert_eq!(status, Status::INEXACT);
    assert!(test.bitwise_eq(nan));

    // Testing convert from quiet NaN
    let test = "nan".parse::<Float8E5M2>().unwrap();
    let test: Float8E4M3FN = unpack!(status=, test.convert(&mut loses_info));
    assert!(test.to_f32().is_nan());
    assert!(loses_info);
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(nan));

    // Testing convert from signaling NaN
    let test = "snan".parse::<Float8E5M2>().unwrap();
    let test: Float8E4M3FN = unpack!(status=, test.convert(&mut loses_info));
    assert!(test.to_f32().is_nan());
    assert!(loses_info);
    assert_eq!(status, Status::INVALID_OP);
    assert!(test.bitwise_eq(nan));
}

#[test]
fn float8e4m3fn_infinity() {
    let t = Float8E4M3FN::INFINITY;
    assert!(t.is_nan());
    assert!(!t.is_infinite());
}

#[test]
fn float8e4m3fn_from_string() {
    // Exactly representable
    assert_eq!(448.0, "448".parse::<Float8E4M3FN>().unwrap().to_f64());
    // Round down to maximum value
    assert_eq!(448.0, "464".parse::<Float8E4M3FN>().unwrap().to_f64());
    // Round up, causing overflow to NaN
    assert!("465".parse::<Float8E4M3FN>().unwrap().is_nan());
    // Overflow without rounding
    assert!("480".parse::<Float8E4M3FN>().unwrap().is_nan());
    // Inf converted to NaN
    assert!("inf".parse::<Float8E4M3FN>().unwrap().is_nan());
    // NaN converted to NaN
    assert!("nan".parse::<Float8E4M3FN>().unwrap().is_nan());
}

#[test]
fn float8e4m3fn_add() {
    let qnan = Float8E4M3FN::NAN;

    let from_str = |s: &str| s.parse::<Float8E4M3FN>().unwrap();

    let addition_tests = [
        // Test addition operations involving NaN, overflow, and the max E4M3
        // value (448) because E4M3 differs from IEEE-754 types in these regards
        (from_str("448"), from_str("16"), "448", Status::INEXACT, Category::Normal, Round::NearestTiesToEven),
        (
            from_str("448"),
            from_str("18"),
            "NaN",
            Status::OVERFLOW | Status::INEXACT,
            Category::NaN,
            Round::NearestTiesToEven,
        ),
        (
            from_str("448"),
            from_str("32"),
            "NaN",
            Status::OVERFLOW | Status::INEXACT,
            Category::NaN,
            Round::NearestTiesToEven,
        ),
        (
            from_str("-448"),
            from_str("-32"),
            "-NaN",
            Status::OVERFLOW | Status::INEXACT,
            Category::NaN,
            Round::NearestTiesToEven,
        ),
        (qnan, from_str("-448"), "NaN", Status::OK, Category::NaN, Round::NearestTiesToEven),
        (from_str("448"), from_str("-32"), "416", Status::OK, Category::Normal, Round::NearestTiesToEven),
        (from_str("448"), from_str("0"), "448", Status::OK, Category::Normal, Round::NearestTiesToEven),
        (from_str("448"), from_str("32"), "448", Status::INEXACT, Category::Normal, Round::TowardZero),
        (from_str("448"), from_str("448"), "448", Status::INEXACT, Category::Normal, Round::TowardZero),
    ];

    for case @ &(x, y, e_result, e_status, e_category, round) in &addition_tests {
        let status;
        let result = unpack!(status=, x.add_r(y, round));
        assert_eq!(e_status, status);
        assert_eq!(e_category, result.category());
        assert!(result.bitwise_eq(e_result.parse::<Float8E4M3FN>().unwrap()), "result = {result:?}, case = {case:?}");
    }
}

#[test]
fn float8e4m3fn_divide_by_zero() {
    let x = "1".parse::<Float8E4M3FN>().unwrap();
    let zero = "0".parse::<Float8E4M3FN>().unwrap();
    let status;
    assert!(unpack!(status=, x / zero).is_nan());
    assert_eq!(status, Status::DIV_BY_ZERO);
}

#[test]
fn float8e4m3fn_next() {
    let mut status;

    // nextUp on positive numbers
    for i in 0..127 {
        let test = Float8E4M3FN::from_bits(i);
        let expected = Float8E4M3FN::from_bits(i + 1);
        assert!(unpack!(status=, test.next_up()).bitwise_eq(expected));
        assert_eq!(status, Status::OK);
    }

    // nextUp on negative zero
    let test = -Float8E4M3FN::ZERO;
    let expected = Float8E4M3FN::SMALLEST;
    assert!(unpack!(status=, test.next_up()).bitwise_eq(expected));
    assert_eq!(status, Status::OK);

    // nextUp on negative nonzero numbers
    for i in 129..255 {
        let test = Float8E4M3FN::from_bits(i);
        let expected = Float8E4M3FN::from_bits(i - 1);
        assert!(unpack!(status=, test.next_up()).bitwise_eq(expected));
        assert_eq!(status, Status::OK);
    }

    // nextUp on NaN
    let test = Float8E4M3FN::qnan(None);
    let expected = Float8E4M3FN::qnan(None);
    assert!(unpack!(status=, test.next_up()).bitwise_eq(expected));
    assert_eq!(status, Status::OK);

    // nextDown on positive nonzero finite numbers
    for i in 1..127 {
        let test = Float8E4M3FN::from_bits(i);
        let expected = Float8E4M3FN::from_bits(i - 1);
        assert!(unpack!(status=, test.next_down()).bitwise_eq(expected));
        assert_eq!(status, Status::OK);
    }

    // nextDown on positive zero
    let test = -Float8E4M3FN::ZERO;
    let expected = -Float8E4M3FN::SMALLEST;
    assert!(unpack!(status=, test.next_down()).bitwise_eq(expected));
    assert_eq!(status, Status::OK);

    // nextDown on negative finite numbers
    for i in 128..255 {
        let test = Float8E4M3FN::from_bits(i);
        let expected = Float8E4M3FN::from_bits(i + 1);
        assert!(unpack!(status=, test.next_down()).bitwise_eq(expected));
        assert_eq!(status, Status::OK);
    }

    // nextDown on NaN
    let test = Float8E4M3FN::qnan(None);
    let expected = Float8E4M3FN::qnan(None);
    assert!(unpack!(status=, test.next_down()).bitwise_eq(expected));
    assert_eq!(status, Status::OK);
}

#[test]
fn float8e4m3fn_exhaustive() {
    // Test each of the 256 Float8E4M3FN values.
    for i in 0..=u8::MAX {
        let test = Float8E4M3FN::from_bits(i.into());

        // isLargest
        if i == 126 || i == 254 {
            assert!(test.is_largest());
            assert_eq!(test.abs().to_f64(), 448.);
        } else {
            assert!(!test.is_largest());
        }

        // isSmallest
        if i == 1 || i == 129 {
            assert!(test.is_smallest());
            assert_eq!(test.abs().to_f64(), /* 0x1p-9 */ 0.001953125);
        } else {
            assert!(!test.is_smallest());
        }

        // convert to BFloat
        let status;
        let mut loses_info = false;
        let test2: BFloat = unpack!(status=, test.convert(&mut loses_info));
        assert_eq!(status, Status::OK);
        assert!(!loses_info);
        if i == 127 || i == 255 {
            assert!(test2.is_nan());
        } else {
            assert_eq!(test.to_f32(), test2.to_f32());
        }

        // bitcastToAPInt
        assert_eq!(u128::from(i), test.to_bits());
    }
}

#[test]
fn float8e4m3fn_exhaustive_pair() {
    // Test each pair of Float8E4M3FN values.
    for i in 0..=u8::MAX {
        for j in 0..=u8::MAX {
            let x = Float8E4M3FN::from_bits(i.into());
            let y = Float8E4M3FN::from_bits(j.into());

            let mut loses_info = false;
            let x16: Half = x.convert(&mut loses_info).value;
            assert!(!loses_info);
            let y16: Half = y.convert(&mut loses_info).value;
            assert!(!loses_info);

            // Add
            let z = (x + y).value;
            let z16 = (x16 + y16).value;
            assert!(z.bitwise_eq(z16.convert(&mut loses_info).value), "i={i}, j={j}");

            // Subtract
            let z = (x - y).value;
            let z16 = (x16 - y16).value;
            assert!(z.bitwise_eq(z16.convert(&mut loses_info).value), "i={i}, j={j}");

            // Multiply
            let z = (x * y).value;
            let z16 = (x16 * y16).value;
            assert!(z.bitwise_eq(z16.convert(&mut loses_info).value), "i={i}, j={j}");

            // Divide
            let z = (x / y).value;
            let z16 = (x16 / y16).value;
            assert!(z.bitwise_eq(z16.convert(&mut loses_info).value), "i={i}, j={j}");

            // Mod
            let z = (x % y).value;
            let z16 = (x16 % y16).value;
            assert!(z.bitwise_eq(z16.convert(&mut loses_info).value), "i={i}, j={j}");

            // Remainder
            let z = x.ieee_rem(y).value;
            let z16 = x16.ieee_rem(y16).value;
            assert!(z.bitwise_eq(z16.convert(&mut loses_info).value), "i={i}, j={j}");
        }
    }
}

#[test]
fn f8_to_string() {
    for_each_float_type!(for<F: Float> test::<F>());
    fn test<F: Float>() {
        if F::BITS != 8 {
            return;
        }

        // NOTE(eddyb) this was buggy upstream as it didn't test `F` but `Float8E5M2`,
        // https://github.com/llvm/llvm-project/commit/6109e70c72fc5171d25c4467fc3cfe6eb2029f50
        // fixed it upstream so we've effectively backported that commit.
        for i in 0..=u8::MAX {
            let test = F::from_bits(i.into());
            let str = test.to_string();

            if test.is_nan() {
                assert_eq!(str, "NaN");
            } else {
                assert!(test.bitwise_eq(str.parse::<F>().unwrap()));
            }
        }
    }
}

// HACK(eddyb) C`{FLT,DBL}_TRUE_MIN` / C++ `std::numeric_limits<T>::denorm_min`
// equivalents, for the two tests below, as Rust seems to lack anything like them,
// but their bit-patterns are thankfuly trivial, with the main caveat that they
// can't be `const` (subnormals and NaNs are banned from CTFE `{to,from}_bits`).
fn f64_smallest_subnormal() -> f64 {
    f64::from_bits(1)
}
fn f32_smallest_subnormal() -> f32 {
    f32::from_bits(1)
}

#[test]
fn double_to_f64() {
    let d_pos_zero = Double::from_f64(0.0);
    assert!(Double::from_f64(d_pos_zero.to_f64()).is_pos_zero());
    let d_neg_zero = Double::from_f64(-0.0);
    assert!(Double::from_f64(d_neg_zero.to_f64()).is_neg_zero());

    let d_one = Double::from_f64(1.0);
    assert_eq!(1.0, d_one.to_f64());
    let d_pos_largest = Double::largest();
    assert_eq!(f64::MAX, d_pos_largest.to_f64());
    let d_neg_largest = -Double::largest();
    assert_eq!(-f64::MAX, d_neg_largest.to_f64());
    let d_pos_smallest = Double::smallest_normalized();
    assert_eq!(f64::MIN_POSITIVE, d_pos_smallest.to_f64());
    let d_neg_smallest = -Double::smallest_normalized();
    assert_eq!(-f64::MIN_POSITIVE, d_neg_smallest.to_f64());

    let d_smallest_denorm = Double::SMALLEST;
    assert_eq!(f64_smallest_subnormal(), d_smallest_denorm.to_f64());
    let d_largest_denorm = "0x0.FFFFFFFFFFFFFp-1022".parse::<Double>().unwrap();
    assert_eq!(/*0x0.FFFFFFFFFFFFFp-1022*/ 2.225073858507201e-308, d_largest_denorm.to_f64());

    let d_pos_inf = Double::INFINITY;
    assert_eq!(f64::INFINITY, d_pos_inf.to_f64());
    let d_neg_inf = -Double::INFINITY;
    assert_eq!(-f64::INFINITY, d_neg_inf.to_f64());
    let d_qnan = Double::qnan(None);
    assert!(d_qnan.to_f64().is_nan());
}

#[test]
fn single_to_f64() {
    let f_pos_zero = Single::from_f32(0.0);
    assert!(Double::from_f64(f_pos_zero.to_f64()).is_pos_zero());
    let f_neg_zero = Single::from_f32(-0.0);
    assert!(Double::from_f64(f_neg_zero.to_f64()).is_neg_zero());

    let f_one = Single::from_f32(1.0);
    assert_eq!(1.0, f_one.to_f64());
    let f_pos_largest = Single::largest();
    assert_eq!(f32::MAX as f64, f_pos_largest.to_f64());
    let f_neg_largest = -Single::largest();
    assert_eq!(-f32::MAX as f64, f_neg_largest.to_f64());
    let f_pos_smallest = Single::smallest_normalized();
    assert_eq!(f32::MIN_POSITIVE as f64, f_pos_smallest.to_f64());
    let f_neg_smallest = -Single::smallest_normalized();
    assert_eq!(-f32::MIN_POSITIVE as f64, f_neg_smallest.to_f64());

    let f_smallest_denorm = Single::SMALLEST;
    assert_eq!(f32_smallest_subnormal() as f64, f_smallest_denorm.to_f64());
    let f_largest_denorm = "0x0.FFFFFEp-126".parse::<Double>().unwrap();
    assert_eq!(/*0x0.FFFFFEp-126*/ 1.1754942106924411e-38, f_largest_denorm.to_f64());

    let f_pos_inf = Single::INFINITY;
    assert_eq!(f64::INFINITY, f_pos_inf.to_f64());
    let f_neg_inf = -Single::INFINITY;
    assert_eq!(-f64::INFINITY, f_neg_inf.to_f64());
    let f_qnan = Single::qnan(None);
    assert!(f_qnan.to_f64().is_nan());

    let h_pos_zero = Half::ZERO;
    assert!(Double::from_f64(h_pos_zero.to_f64()).is_pos_zero());
    let h_neg_zero = -Half::ZERO;
    assert!(Double::from_f64(h_neg_zero.to_f64()).is_neg_zero());
}

#[test]
fn half_to_f64() {
    let h_one = "1.0".parse::<Half>().unwrap();
    assert_eq!(1.0, h_one.to_f64());
    let h_pos_largest = Half::largest();
    assert_eq!(65504.0, h_pos_largest.to_f64());
    let h_neg_largest = -Half::largest();
    assert_eq!(-65504.0, h_neg_largest.to_f64());
    let h_pos_smallest = Half::smallest_normalized();
    assert_eq!(/*0x1.p-14*/ 6.103515625e-05, h_pos_smallest.to_f64());
    let h_neg_smallest = -Half::smallest_normalized();
    assert_eq!(/*-0x1.p-14*/ -6.103515625e-05, h_neg_smallest.to_f64());

    let h_smallest_denorm = Half::SMALLEST;
    assert_eq!(/*0x1.p-24*/ 5.960464477539063e-08, h_smallest_denorm.to_f64());
    let h_largest_denorm = "0x1.FFCp-14".parse::<Half>().unwrap();
    assert_eq!(/*0x1.FFCp-14*/ 0.00012201070785522461, h_largest_denorm.to_f64());

    let h_pos_inf = Half::INFINITY;
    assert_eq!(f64::INFINITY, h_pos_inf.to_f64());
    let h_neg_inf = -Half::INFINITY;
    assert_eq!(-f64::INFINITY, h_neg_inf.to_f64());
    let h_qnan = Half::qnan(None);
    assert!(h_qnan.to_f64().is_nan());
}

#[test]
fn bfloat_to_f64() {
    let b_pos_zero = Half::ZERO;
    assert!(Double::from_f64(b_pos_zero.to_f64()).is_pos_zero());
    let b_neg_zero = -Half::ZERO;
    assert!(Double::from_f64(b_neg_zero.to_f64()).is_neg_zero());

    let b_one = "1.0".parse::<BFloat>().unwrap();
    assert_eq!(1.0, b_one.to_f64());
    let b_pos_largest = BFloat::largest();
    assert_eq!(/*0x1.FEp127*/ 3.3895313892515355e+38, b_pos_largest.to_f64());
    let b_neg_largest = -BFloat::largest();
    assert_eq!(/*-0x1.FEp127*/ -3.3895313892515355e+38, b_neg_largest.to_f64());
    let b_pos_smallest = BFloat::smallest_normalized();
    assert_eq!(/*0x1.p-126*/ 1.1754943508222875e-38, b_pos_smallest.to_f64());
    let b_neg_smallest = -BFloat::smallest_normalized();
    assert_eq!(/*-0x1.p-126*/ -1.1754943508222875e-38, b_neg_smallest.to_f64());

    let b_smallest_denorm = BFloat::SMALLEST;
    assert_eq!(/*0x1.p-133*/ 9.183549615799121e-41, b_smallest_denorm.to_f64());
    let b_largest_denorm = "0x1.FCp-127".parse::<BFloat>().unwrap();
    assert_eq!(/*0x1.FCp-127*/ 1.1663108012064884e-38, b_largest_denorm.to_f64());

    let b_pos_inf = BFloat::INFINITY;
    assert_eq!(f64::INFINITY, b_pos_inf.to_f64());
    let b_neg_inf = -BFloat::INFINITY;
    assert_eq!(-f64::INFINITY, b_neg_inf.to_f64());
    let b_qnan = BFloat::qnan(None);
    assert!(b_qnan.to_f64().is_nan());
}

#[test]
fn float8e5m2_to_f64() {
    let one = "1.0".parse::<Float8E5M2>().unwrap();
    assert_eq!(1.0, one.to_f64());
    let two = "2.0".parse::<Float8E5M2>().unwrap();
    assert_eq!(2.0, two.to_f64());
    let pos_largest = Float8E5M2::largest();
    assert_eq!(5.734400e+04, pos_largest.to_f64());
    let neg_largest = -Float8E5M2::largest();
    assert_eq!(-5.734400e+04, neg_largest.to_f64());
    let pos_smallest = Float8E5M2::smallest_normalized();
    assert_eq!(/* 0x1.p-14 */ 6.103515625e-05, pos_smallest.to_f64());
    let neg_smallest = -Float8E5M2::smallest_normalized();
    assert_eq!(/* -0x1.p-14 */ -6.103515625e-05, neg_smallest.to_f64());

    let smallest_denorm = Float8E5M2::SMALLEST;
    assert!(smallest_denorm.is_denormal());
    assert_eq!(/* 0x1p-16 */ 0.0000152587890625, smallest_denorm.to_f64());

    let pos_inf = Float8E5M2::INFINITY;
    assert_eq!(f64::INFINITY, pos_inf.to_f64());
    let neg_inf = -Float8E5M2::INFINITY;
    assert_eq!(-f64::INFINITY, neg_inf.to_f64());
    let qnan = Float8E5M2::qnan(None);
    assert!(qnan.to_f64().is_nan());
}

#[test]
fn float8e4m3fn_to_f64() {
    let one = "1.0".parse::<Float8E4M3FN>().unwrap();
    assert_eq!(1.0, one.to_f64());
    let two = "2.0".parse::<Float8E4M3FN>().unwrap();
    assert_eq!(2.0, two.to_f64());
    let pos_largest = Float8E4M3FN::largest();
    assert_eq!(448., pos_largest.to_f64());
    let neg_largest = -Float8E4M3FN::largest();
    assert_eq!(-448., neg_largest.to_f64());
    let pos_smallest = Float8E4M3FN::smallest_normalized();
    assert_eq!(/* 0x1.p-6 */ 0.015625, pos_smallest.to_f64());
    let neg_smallest = -Float8E4M3FN::smallest_normalized();
    assert_eq!(/* -0x1.p-6 */ -0.015625, neg_smallest.to_f64());

    let smallest_denorm = Float8E4M3FN::SMALLEST;
    assert!(smallest_denorm.is_denormal());
    assert_eq!(/* 0x1p-9 */ 0.001953125, smallest_denorm.to_f64());

    let qnan = Float8E4M3FN::qnan(None);
    assert!(qnan.to_f64().is_nan());
}

#[test]
fn single_to_f32() {
    let f_pos_zero = Single::from_f32(0.0);
    assert!(Single::from_f32(f_pos_zero.to_f32()).is_pos_zero());
    let f_neg_zero = Single::from_f32(-0.0);
    assert!(Single::from_f32(f_neg_zero.to_f32()).is_neg_zero());

    let f_one = Single::from_f32(1.0);
    assert_eq!(1.0, f_one.to_f32());
    let f_pos_largest = Single::largest();
    assert_eq!(f32::MAX, f_pos_largest.to_f32());
    let f_neg_largest = -Single::largest();
    assert_eq!(-f32::MAX, f_neg_largest.to_f32());
    let f_pos_smallest = Single::smallest_normalized();
    assert_eq!(f32::MIN_POSITIVE, f_pos_smallest.to_f32());
    let f_neg_smallest = -Single::smallest_normalized();
    assert_eq!(-f32::MIN_POSITIVE, f_neg_smallest.to_f32());

    let f_smallest_denorm = Single::SMALLEST;
    assert_eq!(f32_smallest_subnormal(), f_smallest_denorm.to_f32());
    let f_largest_denorm = "0x1.FFFFFEp-126".parse::<Single>().unwrap();
    assert_eq!(/*0x1.FFFFFEp-126*/ 2.3509885615147286e-38, f_largest_denorm.to_f32());

    let f_pos_inf = Single::INFINITY;
    assert_eq!(f32::INFINITY, f_pos_inf.to_f32());
    let f_neg_inf = -Single::INFINITY;
    assert_eq!(-f32::INFINITY, f_neg_inf.to_f32());
    let f_qnan = Single::qnan(None);
    assert!(f_qnan.to_f32().is_nan());
}

#[test]
fn half_to_f32() {
    let h_pos_zero = Half::ZERO;
    assert!(Single::from_f32(h_pos_zero.to_f32()).is_pos_zero());
    let h_neg_zero = -Half::ZERO;
    assert!(Single::from_f32(h_neg_zero.to_f32()).is_neg_zero());

    let h_one = "1.0".parse::<Half>().unwrap();
    assert_eq!(1.0, h_one.to_f32());
    let h_pos_largest = Half::largest();
    assert_eq!(/*0x1.FFCp15*/ 65504.0, h_pos_largest.to_f32());
    let h_neg_largest = -Half::largest();
    assert_eq!(/*-0x1.FFCp15*/ -65504.0, h_neg_largest.to_f32());
    let h_pos_smallest = Half::smallest_normalized();
    assert_eq!(/*0x1.p-14*/ 6.103515625e-05, h_pos_smallest.to_f32());
    let h_neg_smallest = -Half::smallest_normalized();
    assert_eq!(/*-0x1.p-14*/ -6.103515625e-05, h_neg_smallest.to_f32());

    let h_smallest_denorm = Half::SMALLEST;
    assert_eq!(/*0x1.p-24*/ 5.960464477539063e-08, h_smallest_denorm.to_f32());
    let h_largest_denorm = "0x1.FFCp-14".parse::<Half>().unwrap();
    assert_eq!(/*0x1.FFCp-14*/ 0.00012201070785522461, h_largest_denorm.to_f32());

    let h_pos_inf = Half::INFINITY;
    assert_eq!(f32::INFINITY, h_pos_inf.to_f32());
    let h_neg_inf = -Half::INFINITY;
    assert_eq!(-f32::INFINITY, h_neg_inf.to_f32());
    let h_qnan = Half::qnan(None);
    assert!(h_qnan.to_f32().is_nan());
}

#[test]
fn bfloat_to_f32() {
    let b_pos_zero = BFloat::ZERO;
    assert!(Single::from_f32(b_pos_zero.to_f32()).is_pos_zero());
    let b_neg_zero = -BFloat::ZERO;
    assert!(Single::from_f32(b_neg_zero.to_f32()).is_neg_zero());

    let b_one = "1.0".parse::<BFloat>().unwrap();
    assert_eq!(1.0, b_one.to_f32());
    let b_pos_largest = BFloat::largest();
    assert_eq!(/*0x1.FEp127*/ 3.3895313892515355e+38, b_pos_largest.to_f32());
    let b_neg_largest = -BFloat::largest();
    assert_eq!(/*-0x1.FEp127*/ -3.3895313892515355e+38, b_neg_largest.to_f32());
    let b_pos_smallest = BFloat::smallest_normalized();
    assert_eq!(/*0x1.p-126*/ 1.1754943508222875e-38, b_pos_smallest.to_f32());
    let b_neg_smallest = -BFloat::smallest_normalized();
    assert_eq!(/*-0x1.p-126*/ -1.1754943508222875e-38, b_neg_smallest.to_f32());

    let b_smallest_denorm = BFloat::SMALLEST;
    assert_eq!(/*0x1.p-133*/ 9.183549615799121e-41, b_smallest_denorm.to_f32());
    let b_largest_denorm = "0x1.FCp-127".parse::<BFloat>().unwrap();
    assert_eq!(/*0x1.FCp-127*/ 1.1663108012064884e-38, b_largest_denorm.to_f32());

    let b_pos_inf = BFloat::INFINITY;
    assert_eq!(f32::INFINITY, b_pos_inf.to_f32());
    let b_neg_inf = -BFloat::INFINITY;
    assert_eq!(-f32::INFINITY, b_neg_inf.to_f32());
    let b_qnan = BFloat::qnan(None);
    assert!(b_qnan.to_f32().is_nan());
}

#[test]
fn float8e5m2_to_f32() {
    let pos_zero = Float8E5M2::ZERO;
    assert!(Single::from_f32(pos_zero.to_f32()).is_pos_zero());
    let neg_zero = -Float8E5M2::ZERO;
    assert!(Single::from_f32(neg_zero.to_f32()).is_neg_zero());

    let one = "1.0".parse::<Float8E5M2>().unwrap();
    assert_eq!(1.0, one.to_f32());
    let two = "2.0".parse::<Float8E5M2>().unwrap();
    assert_eq!(2.0, two.to_f32());

    let pos_largest = Float8E5M2::largest();
    assert_eq!(5.734400e+04, pos_largest.to_f32());
    let neg_largest = -Float8E5M2::largest();
    assert_eq!(-5.734400e+04, neg_largest.to_f32());
    let pos_smallest = Float8E5M2::smallest_normalized();
    assert_eq!(/* 0x1.p-14 */ 6.103515625e-05, pos_smallest.to_f32());
    let neg_smallest = -Float8E5M2::smallest_normalized();
    assert_eq!(/* -0x1.p-14 */ -6.103515625e-05, neg_smallest.to_f32());

    let smallest_denorm = Float8E5M2::SMALLEST;
    assert!(smallest_denorm.is_denormal());
    assert_eq!(/* 0x1.p-16 */ 0.0000152587890625, smallest_denorm.to_f32());

    let pos_inf = Float8E5M2::INFINITY;
    assert_eq!(f32::INFINITY, pos_inf.to_f32());
    let neg_inf = -Float8E5M2::INFINITY;
    assert_eq!(-f32::INFINITY, neg_inf.to_f32());
    let qnan = Float8E5M2::qnan(None);
    assert!(qnan.to_f32().is_nan());
}

#[test]
fn float8e4m3fn_to_f32() {
    let pos_zero = Float8E4M3FN::ZERO;
    assert!(Single::from_f32(pos_zero.to_f32()).is_pos_zero());
    let neg_zero = -Float8E4M3FN::ZERO;
    assert!(Single::from_f32(neg_zero.to_f32()).is_neg_zero());

    let one = "1.0".parse::<Float8E4M3FN>().unwrap();
    assert_eq!(1.0, one.to_f32());
    let two = "2.0".parse::<Float8E4M3FN>().unwrap();
    assert_eq!(2.0, two.to_f32());

    let pos_largest = Float8E4M3FN::largest();
    assert_eq!(448., pos_largest.to_f32());
    let neg_largest = -Float8E4M3FN::largest();
    assert_eq!(-448.0, neg_largest.to_f32());
    let pos_smallest = Float8E4M3FN::smallest_normalized();
    assert_eq!(/* 0x1.p-6 */ 0.015625, pos_smallest.to_f32());
    let neg_smallest = -Float8E4M3FN::smallest_normalized();
    assert_eq!(/* -0x1.p-6 */ -0.015625, neg_smallest.to_f32());

    let smallest_denorm = Float8E4M3FN::SMALLEST;
    assert!(smallest_denorm.is_denormal());
    assert_eq!(/* 0x1.p-9 */ 0.001953125, smallest_denorm.to_f32());

    let qnan = Float8E4M3FN::qnan(None);
    assert!(qnan.to_f32().is_nan());
}
