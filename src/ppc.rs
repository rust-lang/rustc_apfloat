//! Support for floating point types that use PowerPC semantics.

use crate::ieee;
use crate::{Category, ExpInt, Float, FloatConvert, ParseError, Round, Status, StatusAnd};

use core::cmp::Ordering;
use core::fmt;
use core::ops::Neg;

/// A larger floating point number represented by two smaller floats.
#[must_use]
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct DoubleFloat<F>(F, F);

/// 128-bit floating point number comprised of two IEEE [`Double`](ieee::Double) values.
///
/// This is the "IBM Extended Double" format, described at
/// <https://www.ibm.com/docs/en/aix/7.3?topic=sepl-128-bit-long-double-floating-point-data-type>.
pub type DoubleDouble = DoubleFloat<ieee::Double>;

// These are legacy semantics for the Fallback, inaccurate implementation of
// IBM double-double, if the accurate DoubleDouble doesn't handle the
// operation. It's equivalent to having an IEEE number with consecutive 106
// bits of mantissa and 11 bits of exponent.
//
// It's not equivalent to IBM double-double. For example, a legit IBM
// double-double, 1 + epsilon:
//
//   1 + epsilon = 1 + (1 >> 1076)
//
// is not representable by a consecutive 106 bits of mantissa.
//
// Currently, these semantics are used in the following way:
//
//   DoubleDouble -> (Double, Double) ->
//   DoubleDouble's Fallback -> IEEE operations
//
// FIXME: Implement all operations in DoubleDouble, and delete these
// semantics.
// FIXME(eddyb) This shouldn't need to be `pub`, it's only used in bounds.
pub struct FallbackS<F>(F);
type Fallback<F> = ieee::IeeeFloat<FallbackS<F>>;
impl<F: Float> ieee::Semantics for FallbackS<F> {
    // Forbid any conversion to/from bits.
    const BITS: usize = 0;
    const EXP_BITS: usize = 0;

    const PRECISION: usize = F::PRECISION * 2;
    const MAX_EXP: ExpInt = F::MAX_EXP as ExpInt;
    const MIN_EXP: ExpInt = F::MIN_EXP as ExpInt + F::PRECISION as ExpInt;
}

// Convert number to F. To avoid spurious underflows, we re-
// normalize against the F exponent range first, and only *then*
// truncate the mantissa. The result of that second conversion
// may be inexact, but should never underflow.
// FIXME(eddyb) This shouldn't need to be `pub`, it's only used in bounds.
pub struct FallbackExtendedS<F>(F);
type FallbackExtended<F> = ieee::IeeeFloat<FallbackExtendedS<F>>;
impl<F: Float> ieee::Semantics for FallbackExtendedS<F> {
    // Forbid any conversion to/from bits.
    const BITS: usize = 0;
    const EXP_BITS: usize = 0;

    const PRECISION: usize = Fallback::<F>::PRECISION;
    const MAX_EXP: ExpInt = F::MAX_EXP as ExpInt;
    const MIN_EXP: ExpInt = F::MIN_EXP as ExpInt;
}

impl<F: Float> From<Fallback<F>> for DoubleFloat<F>
where
    F: FloatConvert<FallbackExtended<F>>,
    FallbackExtended<F>: FloatConvert<F>,
{
    fn from(x: Fallback<F>) -> Self {
        let mut status;
        let mut loses_info = false;

        let extended: FallbackExtended<F> = unpack!(status=, x.convert(&mut loses_info));
        assert_eq!((status, loses_info), (Status::OK, false));

        let a = unpack!(status=, extended.convert(&mut loses_info));
        assert_eq!(status - Status::INEXACT, Status::OK);

        // If conversion was exact or resulted in a special case, we're done;
        // just set the second double to zero. Otherwise, re-convert back to
        // the extended format and compute the difference. This now should
        // convert exactly to double.
        let b = if a.is_finite_non_zero() && loses_info {
            let u: FallbackExtended<F> = unpack!(status=, a.convert(&mut loses_info));
            assert_eq!((status, loses_info), (Status::OK, false));
            let v = unpack!(status=, extended - u);
            assert_eq!(status, Status::OK);
            let v = unpack!(status=, v.convert(&mut loses_info));
            assert_eq!((status, loses_info), (Status::OK, false));
            v
        } else {
            F::ZERO
        };

        DoubleFloat(a, b)
    }
}

impl<F: FloatConvert<Self>> From<DoubleFloat<F>> for Fallback<F> {
    fn from(DoubleFloat(a, b): DoubleFloat<F>) -> Self {
        let mut status;
        let mut loses_info = false;

        // Get the first F and convert to our format.
        let a = unpack!(status=, a.convert(&mut loses_info));
        assert_eq!((status, loses_info), (Status::OK, false));

        // Unless we have a special case, add in second F.
        if a.is_finite_non_zero() {
            let b = unpack!(status=, b.convert(&mut loses_info));
            assert_eq!((status, loses_info), (Status::OK, false));

            (a + b).value
        } else {
            a
        }
    }
}

float_common_impls!(DoubleFloat<F>);

impl<F: Float> Neg for DoubleFloat<F> {
    type Output = Self;
    fn neg(self) -> Self {
        if self.1.is_finite_non_zero() {
            DoubleFloat(-self.0, -self.1)
        } else {
            DoubleFloat(-self.0, self.1)
        }
    }
}

impl<F: FloatConvert<Fallback<F>>> fmt::Display for DoubleFloat<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&Fallback::from(*self), f)
    }
}

/// Returns a tuple (hi, lo) such that:
///
/// 1. lo.abs() <= ulp(hi)/2
/// 2. hi == (hi + lo).round_to_integral(Round::NearestTiesToEven)
/// 3. hi + lo == x + y
///
/// Requires that log2(x) >= log2(y).
fn fast_two_sum<F: Float>(x: F, y: F) -> (F, F) {
    if !x.is_finite() {
        return (x, F::ZERO);
    }

    let hi = (x + y).value;
    let delta = (hi - x).value;
    let lo = (y - delta).value;

    (hi, lo)
}

fn is_power_of_two<F: Float>(x: F) -> bool {
    x.is_finite_non_zero() && x.abs().scalbn(-x.ilogb()) == F::from_u128(1).value
}

/// Compute the ULP of the input using a definition from:
/// Jean-Michel Muller. On the definition of ulp(x). [Research Report] RR-5504,
/// LIP RR-2005-09, INRIA, LIP. 2005, pp.16. inria-00070503
fn harrison_ulp<F: Float>(x: F) -> F {
    match x.category() {
        Category::NaN => return F::qnan(None),
        Category::Infinity => return F::INFINITY,
        Category::Zero => return F::SMALLEST,
        Category::Normal => { /* fall through */ }
    }

    if x.is_denormal() || x.is_smallest_normalized() {
        return F::SMALLEST;
    }

    // Match LLVM in not considering negative powers of two.
    let mut exp = x.ilogb();
    if !x.is_negative() && is_power_of_two(x) {
        exp -= 1;
    }

    F::from_u128(1).value.scalbn(exp - (F::PRECISION as i32 - 1))
}

impl<F: FloatConvert<Fallback<F>>> Float for DoubleFloat<F>
where
    Self: From<Fallback<F>>,
{
    const BITS: usize = F::BITS * 2;
    const PRECISION: usize = Fallback::<F>::PRECISION;
    const MAX_EXP: ExpInt = Fallback::<F>::MAX_EXP;
    const MIN_EXP: ExpInt = Fallback::<F>::MIN_EXP;

    const ZERO: Self = DoubleFloat(F::ZERO, F::ZERO);

    const INFINITY: Self = DoubleFloat(F::INFINITY, F::ZERO);

    // FIXME(eddyb) remove when qnan becomes const fn.
    const NAN: Self = DoubleFloat(F::NAN, F::ZERO);

    fn qnan(payload: Option<u128>) -> Self {
        DoubleFloat(F::qnan(payload), F::ZERO)
    }

    fn snan(payload: Option<u128>) -> Self {
        DoubleFloat(F::snan(payload), F::ZERO)
    }

    fn largest() -> Self {
        let status;
        let mut r = DoubleFloat(F::largest(), F::largest());
        r.1 = r.1.scalbn(-(F::PRECISION as ExpInt + 1));
        r.1 = unpack!(status=, r.1.next_down());
        assert_eq!(status, Status::OK);
        r
    }

    const SMALLEST: Self = DoubleFloat(F::SMALLEST, F::ZERO);

    fn smallest_normalized() -> Self {
        DoubleFloat(F::smallest_normalized().scalbn(F::PRECISION as ExpInt), F::ZERO)
    }

    // Implement addition, subtraction, multiplication and division based on:
    // "Software for Doubled-Precision Floating-Point Computations",
    // by Seppo Linnainmaa, ACM TOMS vol 7 no 3, September 1981, pages 272-283.

    fn add_r(mut self, rhs: Self, round: Round) -> StatusAnd<Self> {
        match (self.category(), rhs.category()) {
            (Category::Infinity, Category::Infinity) => {
                if self.is_negative() != rhs.is_negative() {
                    Status::INVALID_OP.and(Self::NAN.copy_sign(self))
                } else {
                    Status::OK.and(self)
                }
            }

            (_, Category::Zero) | (Category::NaN, _) | (Category::Infinity, Category::Normal) => Status::OK.and(self),

            (Category::Zero, _) | (_, Category::NaN) | (_, Category::Infinity) => Status::OK.and(rhs),

            (Category::Normal, Category::Normal) => {
                let mut status = Status::OK;
                let (a, aa, c, cc) = (self.0, self.1, rhs.0, rhs.1);
                let mut z = a;
                z = unpack!(status|=, z.add_r(c, round));
                if !z.is_finite() {
                    if !z.is_infinite() {
                        return status.and(DoubleFloat(z, F::ZERO));
                    }
                    status = Status::OK;
                    let a_cmp_c = a.cmp_abs_normal(c);
                    z = cc;
                    z = unpack!(status|=, z.add_r(aa, round));
                    if a_cmp_c == Ordering::Greater {
                        // z = cc + aa + c + a;
                        z = unpack!(status|=, z.add_r(c, round));
                        z = unpack!(status|=, z.add_r(a, round));
                    } else {
                        // z = cc + aa + a + c;
                        z = unpack!(status|=, z.add_r(a, round));
                        z = unpack!(status|=, z.add_r(c, round));
                    }
                    if !z.is_finite() {
                        return status.and(DoubleFloat(z, F::ZERO));
                    }
                    self.0 = z;
                    let mut zz = aa;
                    zz = unpack!(status|=, zz.add_r(cc, round));
                    if a_cmp_c == Ordering::Greater {
                        // self.1 = a - z + c + zz;
                        self.1 = a;
                        self.1 = unpack!(status|=, self.1.sub_r(z, round));
                        self.1 = unpack!(status|=, self.1.add_r(c, round));
                        self.1 = unpack!(status|=, self.1.add_r(zz, round));
                    } else {
                        // self.1 = c - z + a + zz;
                        self.1 = c;
                        self.1 = unpack!(status|=, self.1.sub_r(z, round));
                        self.1 = unpack!(status|=, self.1.add_r(a, round));
                        self.1 = unpack!(status|=, self.1.add_r(zz, round));
                    }
                } else {
                    // q = a - z;
                    let mut q = a;
                    q = unpack!(status|=, q.sub_r(z, round));

                    // zz = q + c + (a - (q + z)) + aa + cc;
                    // Compute a - (q + z) as -((q + z) - a) to avoid temporary copies.
                    let mut zz = q;
                    zz = unpack!(status|=, zz.add_r(c, round));
                    q = unpack!(status|=, q.add_r(z, round));
                    q = unpack!(status|=, q.sub_r(a, round));
                    q = -q;
                    zz = unpack!(status|=, zz.add_r(q, round));
                    zz = unpack!(status|=, zz.add_r(aa, round));
                    zz = unpack!(status|=, zz.add_r(cc, round));
                    if zz.is_zero() && !zz.is_negative() {
                        return Status::OK.and(DoubleFloat(z, F::ZERO));
                    }
                    self.0 = z;
                    self.0 = unpack!(status|=, self.0.add_r(zz, round));
                    if !self.0.is_finite() {
                        self.1 = F::ZERO;
                        return status.and(self);
                    }
                    self.1 = z;
                    self.1 = unpack!(status|=, self.1.sub_r(self.0, round));
                    self.1 = unpack!(status|=, self.1.add_r(zz, round));
                }
                status.and(self)
            }
        }
    }

    fn mul_r(mut self, rhs: Self, round: Round) -> StatusAnd<Self> {
        // Interesting observation: For special categories, finding the lowest
        // common ancestor of the following layered graph gives the correct
        // return category:
        //
        //    NaN
        //   /   \
        // Zero  Inf
        //   \   /
        //   Normal
        //
        // e.g. NaN * NaN = NaN
        //      Zero * Inf = NaN
        //      Normal * Zero = Zero
        //      Normal * Inf = Inf
        match (self.category(), rhs.category()) {
            (Category::NaN, _) => Status::OK.and(self),

            (_, Category::NaN) => Status::OK.and(rhs),

            (Category::Zero, Category::Infinity) | (Category::Infinity, Category::Zero) => Status::OK.and(Self::NAN),

            (Category::Zero, _) | (Category::Infinity, _) => Status::OK.and(self),

            (_, Category::Zero) | (_, Category::Infinity) => Status::OK.and(rhs),

            (Category::Normal, Category::Normal) => {
                let mut status = Status::OK;
                let (a, b, c, d) = (self.0, self.1, rhs.0, rhs.1);
                // t = a * c
                let mut t = a;
                t = unpack!(status|=, t.mul_r(c, round));
                if !t.is_finite_non_zero() {
                    return status.and(DoubleFloat(t, F::ZERO));
                }

                // tau = fmsub(a, c, t), that is -fmadd(-a, c, t).
                let mut tau = a;
                tau = unpack!(status|=, tau.mul_add_r(c, -t, round));
                // v = a * d
                let mut v = a;
                v = unpack!(status|=, v.mul_r(d, round));
                // w = b * c
                let mut w = b;
                w = unpack!(status|=, w.mul_r(c, round));
                v = unpack!(status|=, v.add_r(w, round));
                // tau += v + w
                tau = unpack!(status|=, tau.add_r(v, round));
                // u = t + tau
                let mut u = t;
                u = unpack!(status|=, u.add_r(tau, round));

                self.0 = u;
                if !u.is_finite() {
                    self.1 = F::ZERO;
                } else {
                    // self.1 = (t - u) + tau
                    t = unpack!(status|=, t.sub_r(u, round));
                    t = unpack!(status|=, t.add_r(tau, round));
                    self.1 = t;
                }
                status.and(self)
            }
        }
    }

    fn mul_add_r(self, multiplicand: Self, addend: Self, round: Round) -> StatusAnd<Self> {
        Fallback::from(self)
            .mul_add_r(Fallback::from(multiplicand), Fallback::from(addend), round)
            .map(Self::from)
    }

    fn div_r(self, rhs: Self, round: Round) -> StatusAnd<Self> {
        Fallback::from(self).div_r(Fallback::from(rhs), round).map(Self::from)
    }

    fn ieee_rem(self, rhs: Self) -> StatusAnd<Self> {
        Fallback::from(self).ieee_rem(Fallback::from(rhs)).map(Self::from)
    }

    fn c_fmod(self, rhs: Self) -> StatusAnd<Self> {
        Fallback::from(self).c_fmod(Fallback::from(rhs)).map(Self::from)
    }

    fn round_to_integral(self, round: Round) -> StatusAnd<Self> {
        let Self(hi, lo) = self;

        let StatusAnd {
            status: hi_status,
            value: mut rounded_hi,
        } = hi.round_to_integral(round);

        // We can reduce the problem to just the high part if the input:
        // 1. Represents a non-finite value.
        // 2. Has a component which is zero.
        if !hi.is_finite_non_zero() || lo.is_zero() {
            return hi_status.and(Self(rounded_hi, F::ZERO));
        }

        // Adjust `rounded` in the direction of `tie_breaker` if `to_round` was at a halfway point.
        let round_to_nearest_helper = |to_round: F, mut rounded: F, tie_breaker: F| {
            // rounding_error tells us which direction we rounded:
            //   - rounding_error > 0: we rounded up.
            //   - rounding_error < 0: we rounded down.
            // Sterbenz' lemma ensures that rounding_error is exact.
            let rounding_error = (rounded - to_round).value;

            // The constant `0.5`.
            let half = F::from_u128(1).value.scalbn(-1);

            if tie_breaker.is_non_zero()
                && tie_breaker.is_negative() != rounding_error.is_negative()
                && rounding_error.abs() == half
            {
                let sign = F::from_u128(1).value.copy_sign(tie_breaker);
                let add = rounded.add_r(sign, Round::NearestTiesToEven);
                assert_eq!(add.status, Status::OK);
                rounded = add.value;
            }

            rounded
        };

        // Case 1: hi is not an integer.
        // Special cases are for rounding modes that are sensitive to ties.
        if rounded_hi != hi {
            // We need to consider the case where hi was between two integers and the
            // rounding mode broke the tie when, in fact, lo may have had a different
            // sign than hi.
            if matches!(round, Round::NearestTiesToAway | Round::NearestTiesToEven) {
                rounded_hi = round_to_nearest_helper(hi, rounded_hi, lo);
            }

            return hi_status.and(Self(rounded_hi, F::ZERO));
        }

        // Case 2: hi is an integer.
        // Special cases are for rounding modes which are rounding towards or away from zero.
        let lo_rounding_mode = if round == Round::TowardZero {
            // When our input is positive, we want the lo component rounded toward
            // negative infinity to get the smallest result magnitude. Likewise,
            // negative inputs want the lo component rounded toward positive infinity.
            if self.is_negative() {
                Round::TowardPositive
            } else {
                Round::TowardNegative
            }
        } else {
            round
        };

        let StatusAnd {
            status: lo_status,
            value: mut rounded_lo,
        } = lo.round_to_integral(lo_rounding_mode);

        if lo_rounding_mode == Round::NearestTiesToAway {
            // We need to consider the case where lo was between two integers and the
            // rounding mode broke the tie when, in fact, hi may have had a different
            // sign than lo.
            rounded_lo = round_to_nearest_helper(lo, rounded_lo, hi);
        }

        // We must ensure that the final result has no overlap between the two Float values.
        let (rounded_hi, rounded_lo) = fast_two_sum(rounded_hi, rounded_lo);

        lo_status.and(Self(rounded_hi, rounded_lo))
    }

    fn next_up(self) -> StatusAnd<Self> {
        Fallback::from(self).next_up().map(Self::from)
    }

    fn from_bits(input: u128) -> Self {
        let (a, b) = (input, input >> F::BITS);
        DoubleFloat(F::from_bits(a & ((1 << F::BITS) - 1)), F::from_bits(b & ((1 << F::BITS) - 1)))
    }

    fn from_u128_r(input: u128, round: Round) -> StatusAnd<Self> {
        Fallback::from_u128_r(input, round).map(Self::from)
    }

    fn from_str_r(s: &str, round: Round) -> Result<StatusAnd<Self>, ParseError> {
        Fallback::from_str_r(s, round).map(|r| r.map(Self::from))
    }

    fn to_bits(self) -> u128 {
        self.0.to_bits() | (self.1.to_bits() << F::BITS)
    }

    fn to_u128_r(self, width: usize, round: Round, is_exact: &mut bool) -> StatusAnd<u128> {
        Fallback::from(self).to_u128_r(width, round, is_exact)
    }

    fn cmp_abs_normal(self, rhs: Self) -> Ordering {
        self.0.cmp_abs_normal(rhs.0).then_with(|| {
            let result = self.1.cmp_abs_normal(rhs.1);
            if result != Ordering::Equal {
                let against = self.0.is_negative() ^ self.1.is_negative();
                let rhs_against = rhs.0.is_negative() ^ rhs.1.is_negative();
                (!against)
                    .cmp(&!rhs_against)
                    .then_with(|| if against { result.reverse() } else { result })
            } else {
                result
            }
        })
    }

    fn bitwise_eq(self, rhs: Self) -> bool {
        self.0.bitwise_eq(rhs.0) && self.1.bitwise_eq(rhs.1)
    }

    fn is_sign_negative(self) -> bool {
        self.0.is_negative()
    }

    fn is_denormal(self) -> bool {
        self.category() == Category::Normal
            && (self.0.is_denormal() || self.0.is_denormal() ||
          // (double)(Hi + Lo) == Hi defines a normal number.
          self.0 !=  (self.0 + self.1).value)
    }

    fn is_signaling(self) -> bool {
        self.0.is_signaling()
    }

    fn category(self) -> Category {
        self.0.category()
    }

    fn is_integer(self) -> bool {
        self.0.is_integer() && self.1.is_integer()
    }

    fn make_quiet(self) -> Self {
        Self(self.0.make_quiet(), self.1)
    }

    fn get_exact_inverse(self) -> Option<Self> {
        Fallback::from(self).get_exact_inverse().map(Self::from)
    }

    fn ilogb(self) -> ExpInt {
        let DoubleFloat(hi, lo) = self;

        if self.category() != Category::Normal
            || (lo.is_zero() || hi.is_negative() == lo.is_negative())
            || !is_power_of_two(hi)
        {
            hi.ilogb()
        } else {
            // Numbers of the form 2^a - 2^b or -2^a + 2^b are almost powers of two but
            // get nudged out of the binade by the low component.
            hi.ilogb() - 1
        }
    }

    fn scalbn_r(self, exp: ExpInt, round: Round) -> Self {
        DoubleFloat(self.0.scalbn_r(exp, round), self.1.scalbn_r(exp, round))
    }

    fn frexp_r(self, exp: &mut ExpInt, round: Round) -> Self {
        // Get the unbiased exponent e of the number, where |self| = m * 2^e for m in [1.0, 2.0).
        *exp = self.ilogb();

        // For NaNs, quiet any signaling NaN and return the result, as per standard practice.
        if *exp == crate::IEK_NAN {
            let mut quiet = self;
            quiet.0 = quiet.0.add_r(F::ZERO, Round::NearestTiesToEven).value;
            return quiet;
        }

        // For infinity, return it unchanged. The exponent remains IEK_Inf.
        if *exp == crate::IEK_INF {
            return self;
        }

        // For zero, the fraction is zero and the standard requires the exponent be 0.
        if *exp == crate::IEK_ZERO {
            *exp = 0;
            return self;
        }

        let DoubleFloat(hi, lo) = self;

        // frexp requires the fraction's absolute value to be in [0.5, 1.0).
        // ilogb provides an exponent for an absolute value in [1.0, 2.0).
        // Increment the exponent to ensure the fraction is in the correct range.
        *exp += 1;

        let signs_disagree = hi.is_negative() != lo.is_negative();
        let mut second = lo;
        if self.category() == Category::Normal && lo.is_finite_non_zero() {
            // The interpretation of Round::TowardZero depends on the sign of the combined
            // self rather than the sign of the component.
            let lo_rounding_mode = if round == Round::TowardZero {
                if self.is_negative() {
                    Round::TowardPositive
                } else {
                    Round::TowardNegative
                }
            } else if round == Round::NearestTiesToAway && signs_disagree && *exp > 0 {
                // For Round::NearestTiesToAway, we face a similar problem. If signs disagree,
                // Lo is a correction *toward* zero relative to Hi. Rounding Lo
                // "away from zero" based on its own sign would move the value in the
                // wrong direction. As a safe proxy, we use Round::NearestTiesToEven, which is
                // direction-agnostic. We only need to bother with this if Lo is scaled
                // down.
                Round::NearestTiesToEven
            } else {
                round
            };

            second = lo.scalbn_r(-*exp, lo_rounding_mode);

            // The Round::NearestTiesToEven proxy is correct most of the time, but it
            // differs from Round::NearestTiesToAway when the scaled value of Lo is an
            // exact midpoint.
            // NOTE: This is morally equivalent to roundTiesTowardZero.
            if round == Round::NearestTiesToAway && lo_rounding_mode == Round::NearestTiesToEven {
                // Re-scale the result back to check if rounding occurred.
                let recomposed_lo = second.scalbn_r(*exp, Round::NearestTiesToEven);
                if recomposed_lo != lo {
                    // RoundingError tells us which direction we rounded:
                    //   - RoundingError > 0: we rounded up.
                    //   - RoundingError < 0: we down up.
                    let rounding_error = (recomposed_lo - lo).value;
                    // Determine if scalbn(Lo, -Exp) landed exactly on a midpoint.
                    // We do this by checking if the absolute rounding error is exactly
                    // half a ULP of the result.
                    let ulp_of_second = harrison_ulp(second);
                    let scaled_ulp_of_second = ulp_of_second.scalbn_r(*exp - 1, Round::NearestTiesToEven);
                    let is_midpoint = rounding_error.abs() == scaled_ulp_of_second;
                    let rounded_lo_away = second.is_negative() == rounding_error.is_negative();
                    // The sign of Hi and Lo disagree and we rounded Lo away: we must
                    // decrease the magnitude of Second to increase the magnitude
                    // First+Second.
                    if is_midpoint && rounded_lo_away {
                        second = if second.is_negative() {
                            second.next_up().value
                        } else {
                            second.next_down().value
                        };
                    }
                }
            }

            // Handle a tricky edge case where self is slightly less than a power of two
            // (e.g., self = 2^k - epsilon). In this situation:
            // 1. Hi is 2^k, and Lo is a small negative value -epsilon.
            // 2. ilogb(self) correctly returns k-1.
            // 3. Our initial Exp becomes (k-1) + 1 = k.
            // 4. Scaling Hi (2^k) by 2^-k would yield a magnitude of 1.0 and
            //    scaling Lo by 2^-k would yield zero. This would make the result 1.0
            //    which is an invalid fraction, as the required interval is [0.5, 1.0).
            // We detect this specific case by checking if Hi is a power of two and if
            // the scaled Lo underflowed to zero. The fix: Increment Exp to k+1. This
            // adjusts the scale factor, causing Hi to be scaled to 0.5, which is a
            // valid fraction.
            if second.is_zero() && signs_disagree && is_power_of_two(hi) {
                *exp += 1;
            }
        }

        let first = hi.scalbn_r(-*exp, round);

        DoubleFloat(first, second)
    }
}

// HACK(eddyb) this is here instead of in `tests/ppc.rs` because `DoubleFloat`
// has private fields, and it's not worth it to make them public just for testing.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_integer() {
        let double_from_f64 = |f: f64| ieee::Double::from_bits(f.to_bits().into());
        assert!(DoubleFloat(double_from_f64(-0.0), double_from_f64(-0.0)).is_integer());
        assert!(!DoubleFloat(double_from_f64(3.14159), double_from_f64(-0.0)).is_integer());
        assert!(!DoubleFloat(double_from_f64(-0.0), double_from_f64(3.14159)).is_integer());
    }

    #[derive(Copy, Clone)]
    struct TestCase<I, T> {
        input: I,

        nearest_ties_to_even: Option<T>,
        toward_positive: Option<T>,
        toward_negative: Option<T>,
        toward_zero: Option<T>,
        nearest_ties_to_away: Option<T>,
    }

    impl<I, T: Copy> TestCase<I, T> {
        fn new(input: I) -> Self {
            Self {
                input,
                nearest_ties_to_even: None,
                toward_positive: None,
                toward_negative: None,
                toward_zero: None,
                nearest_ties_to_away: None,
            }
        }

        fn with_all(self, expected: T) -> Self {
            Self {
                input: self.input,
                nearest_ties_to_even: Some(expected),
                toward_positive: Some(expected),
                toward_negative: Some(expected),
                toward_zero: Some(expected),
                nearest_ties_to_away: Some(expected),
            }
        }

        fn with(mut self, round: Round, expected: T) -> Self {
            match round {
                Round::NearestTiesToEven => self.nearest_ties_to_even = Some(expected),
                Round::TowardPositive => self.toward_positive = Some(expected),
                Round::TowardNegative => self.toward_negative = Some(expected),
                Round::TowardZero => self.toward_zero = Some(expected),
                Round::NearestTiesToAway => self.nearest_ties_to_away = Some(expected),
            }

            self
        }
    }

    fn dd(hi: f64, lo: f64) -> DoubleDouble {
        let double_from_f64 = |f: f64| ieee::Double::from_bits(f.to_bits().into());
        DoubleFloat(double_from_f64(hi), double_from_f64(lo))
    }

    #[test]
    fn ppc_double_double_round_to_integral() {
        let eps = f64::EPSILON;
        let half_eps = eps / 2.0;
        let quarter_eps = eps / 4.0;
        let smallest_normal = f64::MIN_POSITIVE;
        let even_integer_threshold = (1u64 << f64::MANTISSA_DIGITS) as f64;

        let round_to_integral_test_cases = [
            // 1. Zeros and Basic Integers
            TestCase::new(dd(0.0, 0.0)).with_all(dd(0.0, 0.0)),
            TestCase::new(dd(2.0, 0.0)).with_all(dd(2.0, 0.0)),
            TestCase::new(dd(3.0, 0.0)).with_all(dd(3.0, 0.0)),
            // 2. General Fractions (Non-Ties)
            TestCase::new(dd(2.3, 0.0))
                .with_all(dd(2.0, 0.0))
                .with(Round::TowardPositive, dd(3.0, 0.0)),
            TestCase::new(dd(2.7, 0.0))
                .with_all(dd(3.0, 0.0))
                .with(Round::TowardZero, dd(2.0, 0.0))
                .with(Round::TowardNegative, dd(2.0, 0.0)),
            TestCase::new(dd(2.3, smallest_normal))
                .with_all(dd(2.0, 0.0))
                .with(Round::TowardPositive, dd(3.0, 0.0)),
            // 3. Exact Midpoints (Ties at N.5)
            TestCase::new(dd(0.5, 0.0))
                .with_all(dd(0.0, 0.0))
                .with(Round::TowardPositive, dd(1.0, 0.0))
                .with(Round::NearestTiesToAway, dd(1.0, 0.0)),
            TestCase::new(dd(1.5, 0.0))
                .with_all(dd(2.0, 0.0))
                .with(Round::TowardZero, dd(1.0, 0.0))
                .with(Round::TowardNegative, dd(1.0, 0.0)),
            TestCase::new(dd(2.5, 0.0))
                .with_all(dd(2.0, 0.0))
                .with(Round::TowardPositive, dd(3.0, 0.0))
                .with(Round::NearestTiesToAway, dd(3.0, 0.0)),
            // 4. Near Midpoints (lo breaks the tie)
            TestCase::new(dd(2.5, smallest_normal))
                .with_all(dd(3.0, 0.0))
                .with(Round::TowardZero, dd(2.0, 0.0))
                .with(Round::TowardNegative, dd(2.0, 0.0)),
            TestCase::new(dd(2.5, -smallest_normal))
                .with_all(dd(2.0, 0.0))
                .with(Round::TowardPositive, dd(3.0, 0.0)),
            TestCase::new(dd(1.5, smallest_normal))
                .with_all(dd(2.0, 0.0))
                .with(Round::TowardZero, dd(1.0, 0.0))
                .with(Round::TowardNegative, dd(1.0, 0.0)),
            TestCase::new(dd(1.5, -smallest_normal))
                .with_all(dd(1.0, 0.0))
                .with(Round::TowardPositive, dd(2.0, 0.0)),
            // 5. Near Integers (lo crosses the integer boundary)
            TestCase::new(dd(2.0, smallest_normal))
                .with_all(dd(2.0, 0.0))
                .with(Round::TowardPositive, dd(3.0, 0.0)),
            TestCase::new(dd(2.0, -smallest_normal))
                .with_all(dd(2.0, 0.0))
                .with(Round::TowardZero, dd(1.0, 0.0))
                .with(Round::TowardNegative, dd(1.0, 0.0)),
            TestCase::new(dd(smallest_normal, 0.0))
                .with_all(dd(0.0, 0.0))
                .with(Round::TowardPositive, dd(1.0, 0.0)),
            // 6. Boundary of Canonicalization (Maximum lo)
            TestCase::new(dd(1.0, half_eps))
                .with_all(dd(1.0, 0.0))
                .with(Round::TowardPositive, dd(2.0, 0.0)),
            TestCase::new(dd(1.0, -quarter_eps))
                .with_all(dd(1.0, 0.0))
                .with(Round::TowardZero, dd(0.0, 0.0))
                .with(Round::TowardNegative, dd(0.0, 0.0)),
            // 7. Large Magnitudes (Beyond 2^53). N = EvenIntegerThreshold (Even)
            TestCase::new(dd(even_integer_threshold, 0.0)).with_all(dd(even_integer_threshold, 0.0)),
            TestCase::new(dd(even_integer_threshold, 1.0)).with_all(dd(even_integer_threshold, 1.0)),
            // Fractions
            TestCase::new(dd(even_integer_threshold, 0.25))
                .with_all(dd(even_integer_threshold, 0.0))
                .with(Round::TowardPositive, dd(even_integer_threshold, 1.0)),
            TestCase::new(dd(even_integer_threshold, 0.75))
                .with_all(dd(even_integer_threshold, 1.0))
                .with(Round::TowardZero, dd(even_integer_threshold, 0.0))
                .with(Round::TowardNegative, dd(even_integer_threshold, 0.0)),
            // Ties (Midpoints)
            TestCase::new(dd(even_integer_threshold, 0.5))
                .with_all(dd(even_integer_threshold, 0.0))
                .with(Round::TowardPositive, dd(even_integer_threshold, 1.0))
                .with(Round::NearestTiesToAway, dd(even_integer_threshold, 1.0)),
            TestCase::new(dd(even_integer_threshold + 2.0, 0.5))
                .with_all(dd(even_integer_threshold + 2.0, 0.0))
                .with(Round::TowardPositive, dd(even_integer_threshold + 4.0, -1.0))
                .with(Round::NearestTiesToAway, dd(even_integer_threshold + 4.0, -1.0)),
            // Near Ties
            TestCase::new(dd(even_integer_threshold, 0.5 + half_eps))
                .with_all(dd(even_integer_threshold, 1.0))
                .with(Round::TowardZero, dd(even_integer_threshold, 0.0))
                .with(Round::TowardNegative, dd(even_integer_threshold, 0.0)),
            TestCase::new(dd(even_integer_threshold, 0.5 - quarter_eps))
                .with_all(dd(even_integer_threshold, 0.0))
                .with(Round::TowardPositive, dd(even_integer_threshold, 1.0)),
            // Canonical Boundary (Max lo for EvenIntegerThreshold is 1.0)
            TestCase::new(dd(even_integer_threshold, 1.0)).with_all(dd(even_integer_threshold, 1.0)),
            // 8. Special Values
            TestCase::new(dd(f64::INFINITY, 0.0)).with_all(dd(f64::INFINITY, 0.0)),
        ];

        let negate = |test_case: TestCase<DoubleDouble, DoubleDouble>| TestCase {
            input: -test_case.input,
            nearest_ties_to_even: test_case.nearest_ties_to_even.map(|v| -v),
            toward_positive: test_case.toward_negative.map(|v| -v),
            toward_negative: test_case.toward_positive.map(|v| -v),
            toward_zero: test_case.toward_zero.map(|v| -v),
            nearest_ties_to_away: test_case.nearest_ties_to_away.map(|v| -v),
        };

        for case in round_to_integral_test_cases.iter().flat_map(|v| [*v, negate(*v)]) {
            if let Some(expected) = case.nearest_ties_to_even {
                assert_eq!(case.input.round_to_integral(Round::NearestTiesToEven).value, expected);
            }

            if let Some(expected) = case.nearest_ties_to_away {
                assert_eq!(case.input.round_to_integral(Round::NearestTiesToAway).value, expected);
            }

            if let Some(expected) = case.toward_positive {
                assert_eq!(case.input.round_to_integral(Round::TowardPositive).value, expected);
            }

            if let Some(expected) = case.toward_negative {
                assert_eq!(case.input.round_to_integral(Round::TowardNegative).value, expected);
            }

            if let Some(expected) = case.toward_zero {
                assert_eq!(case.input.round_to_integral(Round::TowardZero).value, expected);
            }
        }
    }

    #[test]
    #[rustfmt::skip]
    fn ppc_double_double_round_to_integral_quiet_nan() {
        let quiet_nan = dd(f64::NAN, 0.0);

        assert!(quiet_nan.round_to_integral(Round::TowardZero).value.bitwise_eq(quiet_nan));
        assert!(quiet_nan.round_to_integral(Round::TowardNegative).value.bitwise_eq(quiet_nan));
        assert!(quiet_nan.round_to_integral(Round::TowardPositive).value.bitwise_eq(quiet_nan));
        assert!(quiet_nan.round_to_integral(Round::NearestTiesToAway).value.bitwise_eq(quiet_nan));
        assert!(quiet_nan.round_to_integral(Round::NearestTiesToEven).value.bitwise_eq(quiet_nan));
    }

    #[test]
    fn ppc_double_double_frexp() {
        let double_from_f64 = |f: f64| ieee::Double::from_bits(f.to_bits().into());
        let dd = |hi, lo| DoubleFloat(double_from_f64(hi), double_from_f64(lo));

        let frexp_test_cases = [
            // Input: +infinity
            TestCase::new(dd(f64::INFINITY, 0.0)).with_all((dd(f64::INFINITY, 0.0), ExpInt::MAX)),
            // Input: -infinity
            TestCase::new(dd(f64::NEG_INFINITY, 0.0)).with_all((dd(f64::NEG_INFINITY, 0.0), ExpInt::MAX)),
            // Input: 2^-1074
            TestCase::new(dd(f64::from_bits(1), 0.0)).with_all((dd(0.5, 0.0), -1073)),
            // Input: (2^1, -2^-1073 + -2^-1074)
            TestCase::new(dd(2.0, -3.0 * f64::from_bits(1)))
                .with_all((dd(1.0, -2.0 * f64::from_bits(1)), 1))
                .with(Round::NearestTiesToAway, (dd(1.0, -f64::from_bits(1)), 1))
                .with(Round::TowardPositive, (dd(1.0, -f64::from_bits(1)), 1)),
            // Input: (2^1, -2^-1073)
            TestCase::new(dd(2.0, -2.0 * f64::from_bits(1))).with_all((dd(1.0, -f64::from_bits(1)), 1)),
            // Input: (2^1, -2^-1074)
            TestCase::new(dd(2.0, -f64::from_bits(1)))
                .with_all((dd(0.5, -0.0), 2))
                .with(Round::TowardNegative, (dd(1.0, -f64::from_bits(1)), 1))
                .with(Round::TowardZero, (dd(1.0, -f64::from_bits(1)), 1)),
            // Input: (2^2, -2^-1072 + -2^-1073 + -2^-1074)
            TestCase::new(dd(4.0, -7.0 * f64::from_bits(1)))
                .with_all((dd(1.0, -2.0 * f64::from_bits(1)), 2))
                .with(Round::TowardPositive, (dd(1.0, -f64::from_bits(1)), 2)),
            // Input: (2^2, -2^-1072 + -2^-1073)
            TestCase::new(dd(4.0, -6.0 * f64::from_bits(1)))
                .with_all((dd(1.0, -2.0 * f64::from_bits(1)), 2))
                .with(Round::NearestTiesToAway, (dd(1.0, -f64::from_bits(1)), 2))
                .with(Round::TowardPositive, (dd(1.0, -f64::from_bits(1)), 2)),
            // Input: (2^2, -2^-1072 + -2^-1074)
            TestCase::new(dd(4.0, -5.0 * f64::from_bits(1)))
                .with_all((dd(1.0, -f64::from_bits(1)), 2))
                .with(Round::TowardNegative, (dd(1.0, -2.0 * f64::from_bits(1)), 2))
                .with(Round::TowardZero, (dd(1.0, -2.0 * f64::from_bits(1)), 2)),
            // Input: (2^2, -2^-1072)
            TestCase::new(dd(4.0, -4.0 * f64::from_bits(1))).with_all((dd(1.0, -f64::from_bits(1)), 2)),
            // Input: (2^2, -2^-1073 + -2^-1074)
            TestCase::new(dd(4.0, -3.0 * f64::from_bits(1)))
                .with_all((dd(1.0, -f64::from_bits(1)), 2))
                .with(Round::TowardPositive, (dd(0.5, -0.0), 3)),
            // Input: (2^2, -2^-1073)
            TestCase::new(dd(4.0, -2.0 * f64::from_bits(1)))
                .with_all((dd(0.5, -0.0), 3))
                .with(Round::TowardNegative, (dd(1.0, -f64::from_bits(1)), 2))
                .with(Round::TowardZero, (dd(1.0, -f64::from_bits(1)), 2)),
            // Input: (2^2, -2^-1074)
            TestCase::new(dd(4.0, -f64::from_bits(1)))
                .with_all((dd(0.5, -0.0), 3))
                .with(Round::TowardNegative, (dd(1.0, -f64::from_bits(1)), 2))
                .with(Round::TowardZero, (dd(1.0, -f64::from_bits(1)), 2)),
            // Input: 3+3*2^-53 canonicalized to (3+2^-51, -2^-53)
            // Output: 0.75+0.75*2^-53 canonicalized to (.75+2^-53, -2^-55)
            TestCase::new(dd(f64::from_bits(0x4008_0000_0000_0001), -2.0f64.powi(-53)))
                .with_all((dd(f64::from_bits(0x3fe8_0000_0000_0001), -2.0f64.powi(-55)), 2)),
            // Input: (2^1021+2^969, 2^968-2^915)
            TestCase::new(dd(f64::from_bits(0x7fc0_0000_0000_0001), f64::from_bits(0x7c6f_ffff_ffff_ffff)))
                .with_all((dd(f64::from_bits(0x3fe0_0000_0000_0001), f64::from_bits(0x3c8f_ffff_ffff_ffff)), 1022)),
            // Input: (2^1023, -2^-1)
            TestCase::new(dd(2.0f64.powi(1023), -0.5)).with_all((dd(1.0, -f64::from_bits(1 << 50)), 1023)),
            // Input: (2^1023, -2^-51)
            TestCase::new(dd(2.0f64.powi(1023), -2.0f64.powi(-51))).with_all((dd(1.0, -f64::from_bits(1)), 1023)),
            // Input: (2^1023, -2^-52)
            TestCase::new(dd(2.0f64.powi(1023), -2.0f64.powi(-52)))
                .with_all((dd(0.5, -0.0), 1024))
                .with(Round::TowardNegative, (dd(1.0, -f64::from_bits(1)), 1023))
                .with(Round::TowardZero, (dd(1.0, -f64::from_bits(1)), 1023)),
            // Input: (2^1023, 2^-1074)
            TestCase::new(dd(2.0f64.powi(1023), f64::from_bits(1)))
                .with_all((dd(0.5, 0.0), 1024))
                .with(Round::TowardPositive, (dd(0.5, f64::from_bits(1)), 1024)),
            // Input: (2^1024-2^971, 2^970-2^918)
            TestCase::new(DoubleDouble::largest())
                .with_all((dd(f64::from_bits(0x3fef_ffff_ffff_ffff), f64::from_bits(0x3c8f_ffff_ffff_fffe)), 1024)),
        ];

        let negate = |test_case: TestCase<DoubleDouble, (DoubleDouble, ExpInt)>| TestCase {
            input: -test_case.input,
            nearest_ties_to_even: test_case.nearest_ties_to_even.map(|(v, e)| (-v, e)),
            toward_positive: test_case.toward_negative.map(|(v, e)| (-v, e)),
            toward_negative: test_case.toward_positive.map(|(v, e)| (-v, e)),
            toward_zero: test_case.toward_zero.map(|(v, e)| (-v, e)),
            nearest_ties_to_away: test_case.nearest_ties_to_away.map(|(v, e)| (-v, e)),
        };

        let mut actual_exp = 0;

        for case in frexp_test_cases.iter().flat_map(|v| [*v, negate(*v)]) {
            if let Some((expected, expected_exp)) = case.nearest_ties_to_even {
                assert_eq!(case.input.frexp_r(&mut actual_exp, Round::NearestTiesToEven), expected);
                assert_eq!(expected_exp, actual_exp);
            }

            if let Some((expected, expected_exp)) = case.nearest_ties_to_away {
                assert_eq!(case.input.frexp_r(&mut actual_exp, Round::NearestTiesToAway), expected);
                assert_eq!(expected_exp, actual_exp);
            }

            if let Some((expected, expected_exp)) = case.toward_positive {
                assert_eq!(case.input.frexp_r(&mut actual_exp, Round::TowardPositive), expected);
                assert_eq!(expected_exp, actual_exp);
            }

            if let Some((expected, expected_exp)) = case.toward_negative {
                assert_eq!(case.input.frexp_r(&mut actual_exp, Round::TowardNegative), expected);
                assert_eq!(expected_exp, actual_exp);
            }

            if let Some((expected, expected_exp)) = case.toward_zero {
                assert_eq!(case.input.frexp_r(&mut actual_exp, Round::TowardZero), expected);
                assert_eq!(expected_exp, actual_exp);
            }
        }
    }
}
