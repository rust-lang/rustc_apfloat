//! Support for floating point types compatible with IEEE 754.

use crate::{Category, ExpInt, IEK_INF, IEK_NAN, IEK_ZERO};
use crate::{Float, FloatConvert, ParseError, Round, Status, StatusAnd};

use core::cmp::{self, Ordering};
use core::convert::TryFrom;
use core::fmt::{self, Write};
use core::marker::PhantomData;
use core::mem;
use core::ops::Neg;

/// A floating point number that uses IEEE semantics.
///
/// Usually you will want to use the available type aliases of this type
/// (e.g., [`Single`], [`Double`]) rather than referencing it directly.
///
/// If `S` implements [`Semantics`], this type will implement [`Float`].
#[must_use]
pub struct IeeeFloat<S> {
    /// Absolute significand value (including the integer bit).
    sig: [Limb; 1],

    /// The signed unbiased exponent of the value.
    exp: ExpInt,

    /// What kind of floating point number this is.
    //
    // HACK(eddyb) because mutating this without accounting for `exp`/`sig`
    // can break some subtle edge cases, it should be only read through the
    // `.category()` method, and only set during initialization, either for
    // one of the special value constants, or for conversion from bits.
    read_only_category_do_not_mutate: Category,

    /// Sign bit of the number.
    //
    // HACK(eddyb) because mutating this without accounting for `category`
    // can break some subtle edge cases, it should be only read through the
    // `.is_negative()` method, and only set through negation (which can be
    // more easily used through e.g. `copy_sign`/`negate_if`/`with_sign`).
    read_only_sign_do_not_mutate: bool,

    marker: PhantomData<S>,
}

/// Fundamental unit of big integer arithmetic, but also
/// large to store the largest significands by itself.
type Limb = u128;
const LIMB_BITS: usize = 128;
#[inline(always)]
fn limbs_for_bits(bits: usize) -> usize {
    (bits + LIMB_BITS - 1) / LIMB_BITS
}

/// Growable `[Limb]` (i.e. heap-allocated and typically `Vec`/`SmallVec`/etc.),
/// used only by algorithms that may require dynamically arbitrary precision,
/// i.e. conversions from/to decimal strings.
///
/// Note: the specific type was chosen by starting with `SmallVec<[_; 1]>` and
/// increasing the inline length as long as benchmarks were showing improvements,
/// or at least the `Double::from_str` ones, which roughly had these behaviors:
/// * `Vec<_>` -> `SmallVec<[_; 1]>`: ~15% speedup, but only for shorter inputs
/// * `SmallVec<[_; 1]>` -> `SmallVec<[_; 2]>`: ~10% speedup for longer inputs
/// * `SmallVec<[_; 2]>` -> `SmallVec<[_; 3]>`: noise and/or diminishing returns
///
/// Note: the choice of type described above, and the factors in its decision,
/// are specific to `Limb` being `u128`, so if `Limb` changes, this should too.
type DynPrecisionLimbVec = smallvec::SmallVec<[Limb; 2]>;

/// Enum that represents what fraction of the LSB truncated bits of an fp number
/// represent.
///
/// This essentially combines the roles of guard and sticky bits.
#[must_use]
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Loss {
    // Example of truncated bits:
    ExactlyZero,  // 000000
    LessThanHalf, // 0xxxxx  x's not all zero
    ExactlyHalf,  // 100000
    MoreThanHalf, // 1xxxxx  x's not all zero
}

/// How the nonfinite values Inf and NaN are represented.
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum NonfiniteBehavior {
    /// Represents standard IEEE 754 behavior. A value is nonfinite if the
    /// exponent field is all 1s. In such cases, a value is Inf if the
    /// significand bits are all zero, and NaN otherwise
    IEEE754,

    /// Only the Float8E5M2 has this behavior. There is no Inf representation. A
    /// value is NaN if the exponent field and the mantissa field are all 1s.
    /// This behavior matches the FP8 E4M3 type described in
    /// <https://arxiv.org/abs/2209.05433>. We treat both signed and unsigned NaNs
    /// as non-signalling, although the paper does not state whether the NaN
    /// values are signalling or not.
    NanOnly,
}

// HACK(eddyb) extension method flipping/changing the sign based on `bool`s.
trait NegExt: Neg<Output = Self> + Sized {
    fn negate_if(self, negate: bool) -> Self {
        if negate {
            -self
        } else {
            self
        }
    }

    fn with_sign(self, sign: bool) -> Self
    where
        Self: Float,
    {
        self.negate_if(self.is_negative() != sign)
    }
}
impl<T: Neg<Output = Self>> NegExt for T {}

/// Represents floating point arithmetic semantics.
pub trait Semantics: Sized {
    /// Total number of bits in the interchange format.
    const BITS: usize;

    /// Number of exponent bits in the interchange format.
    const EXP_BITS: usize;

    /// Number of bits in the significand. This includes the integer bit.
    const PRECISION: usize = (Self::BITS - 1 - Self::EXP_BITS) + 1;

    /// How the nonfinite values Inf and NaN are represented.
    const NONFINITE_BEHAVIOR: NonfiniteBehavior = NonfiniteBehavior::IEEE754;

    /// The largest E such that 2^E is representable; this matches the
    /// definition of IEEE 754.
    const MAX_EXP: ExpInt = {
        let ieee_inf_and_nan_replaced_with_extra_normals = match Self::NONFINITE_BEHAVIOR {
            NonfiniteBehavior::IEEE754 => false,
            NonfiniteBehavior::NanOnly => true,
        };
        Self::IEEE_MAX_EXP
            + (Self::MIN_EXP - Self::IEEE_MIN_EXP)
            + (ieee_inf_and_nan_replaced_with_extra_normals as ExpInt)
    };
    const IEEE_MAX_EXP: ExpInt = -Self::IEEE_MIN_EXP + 1;

    /// The smallest E such that 2^E is a normalized number; this
    /// matches the definition of IEEE 754.
    const MIN_EXP: ExpInt = Self::IEEE_MIN_EXP;
    const IEEE_MIN_EXP: ExpInt = -(1 << (Self::EXP_BITS - 1)) + 2;

    /// The base significand bitpattern of NaNs, i.e. the bits that must always
    /// be set in all NaNs, with other significand bits being either used for
    /// payload bits (if `NAN_PAYLOAD_MASK` covers them) or always unset.
    const NAN_SIGNIFICAND_BASE: Limb = match Self::NONFINITE_BEHAVIOR {
        NonfiniteBehavior::IEEE754 => 0,
        NonfiniteBehavior::NanOnly => (1 << (Self::PRECISION - 1)) - 1,
    };

    /// The significand bitmask for the payload of a NaN (if supported),
    /// including the "quiet bit" (telling QNaNs apart from SNaNs).
    const NAN_PAYLOAD_MASK: Limb = match Self::NONFINITE_BEHAVIOR {
        NonfiniteBehavior::IEEE754 => (1 << (Self::PRECISION - 1)) - 1,
        NonfiniteBehavior::NanOnly => 0,
    };

    /// The significand bitpattern to mark a NaN as quiet (if supported).
    ///
    /// NOTE: for X87DoubleExtended we need to set two bits instead of one.
    ///
    /// NOTE: all NaNs are quiet if unsupported (see `NonfiniteBehavior::NanOnly`).
    const QNAN_SIGNIFICAND: Limb = match Self::NONFINITE_BEHAVIOR {
        NonfiniteBehavior::IEEE754 => 1 << (Self::PRECISION - 2),
        NonfiniteBehavior::NanOnly => 0,
    };

    fn from_bits(bits: u128) -> IeeeFloat<Self> {
        assert!(Self::BITS > Self::PRECISION);

        let sign = bits & (1 << (Self::BITS - 1));
        let exponent = ((bits & !sign) >> (Self::BITS - 1 - Self::EXP_BITS)) & ((1 << Self::EXP_BITS) - 1);
        let mut r = IeeeFloat {
            sig: [bits & ((1 << (Self::PRECISION - 1)) - 1)],
            // Convert the exponent from its bias representation to a signed integer.
            exp: (exponent as ExpInt) + (Self::MIN_EXP - 1),
            read_only_category_do_not_mutate: Category::Zero,
            read_only_sign_do_not_mutate: sign != 0,
            marker: PhantomData,
        };

        // NOTE(eddyb) unlike the original C++ code, this doesn't check for
        // specific exponent/significand combinations, but instead relies on
        // being able to construct known-good special values to compare to.
        let try_category_from_special = |mut special: IeeeFloat<Self>| {
            special = special.copy_sign(r);

            // Ignore NaN payload to avoid needing a separate NaN check.
            let sig_mask = if special.is_nan() { !Self::NAN_PAYLOAD_MASK } else { !0 };

            if special.is_negative() == r.is_negative()
                && special.exp == r.exp
                && special.sig[0] & sig_mask == r.sig[0] & sig_mask
            {
                Some(special.category())
            } else {
                None
            }
        };

        // NOTE(eddyb) the order here matters, i.e. `NAN` needs to be last, as
        // its relaxed check (see above) overlaps `INFINITY`, for IEEE NaNs.
        let specials = [
            IeeeFloat::<Self>::ZERO,
            IeeeFloat::<Self>::INFINITY,
            IeeeFloat::<Self>::NAN,
        ];

        let category = specials
            .into_iter()
            .find_map(try_category_from_special)
            .unwrap_or_else(|| {
                if r.exp == Self::MIN_EXP - 1 {
                    // Denormal.
                    r.exp = Self::MIN_EXP;
                } else {
                    // Set integer bit.
                    sig::set_bit(&mut r.sig, Self::PRECISION - 1);
                }
                Category::Normal
            });

        r.read_only_category_do_not_mutate = category;

        r
    }

    fn to_bits(x: IeeeFloat<Self>) -> u128 {
        assert!(Self::BITS > Self::PRECISION);

        // Split integer bit from significand.
        let integer_bit = sig::get_bit(&x.sig, Self::PRECISION - 1);
        let mut significand = x.sig[0] & ((1 << (Self::PRECISION - 1)) - 1);
        let mut exponent = x.exp;
        match x.category() {
            Category::Normal => {
                if exponent == Self::MIN_EXP && !integer_bit {
                    // Denormal.
                    exponent -= 1;
                }
            }
            Category::Zero => {
                // FIXME(eddyb) Maybe we should guarantee an invariant instead?
                IeeeFloat::<Self> {
                    sig: [significand],
                    exp: exponent,
                    ..
                } = Float::ZERO;
            }
            Category::Infinity => {
                // FIXME(eddyb) Maybe we should guarantee an invariant instead?
                IeeeFloat::<Self> {
                    sig: [significand],
                    exp: exponent,
                    ..
                } = Float::INFINITY;
            }
            Category::NaN => {
                IeeeFloat::<Self> { exp: exponent, .. } = Float::NAN;
            }
        }

        // Convert the exponent from a signed integer to its bias representation.
        let exponent = (exponent - (Self::MIN_EXP - 1)) as u128;

        ((x.is_negative() as u128) << (Self::BITS - 1)) | (exponent << (Self::PRECISION - 1)) | significand
    }
}

impl<S> Copy for IeeeFloat<S> {}
impl<S> Clone for IeeeFloat<S> {
    fn clone(&self) -> Self {
        *self
    }
}

macro_rules! ieee_semantics {
    ($(
        $(#[$meta:meta])*
        $name:ident = $sem:ident($bits:tt : $exp_bits:tt) $({ $($extra:tt)* })?
    ),* $(,)?) => {
        $(
            #[doc = concat!("Floating point semantics for [`", stringify!($name), "`].")]
            ///
            /// See that type for more details.
            pub struct $sem;

            $(#[$meta])*
            pub type $name = IeeeFloat<$sem>;

            impl Semantics for $sem {
                const BITS: usize = $bits;
                const EXP_BITS: usize = $exp_bits;

                $($($extra)*)?
            }
        )*
    }
}

ieee_semantics! {
    /// IEEE binary16 half-precision (16-bit) floating point number.
    Half = HalfS(16:5),

    /// IEEE binary32 single-precision (32-bit) floating point number.
    Single = SingleS(32:8),

    /// IEEE binary64 double-precision (64-bit) floating point number.
    Double = DoubleS(64:11),

    /// IEEE binary128 quadruple-precision (128-bit) floating point number.
    Quad = QuadS(128:15),

    /// 16-bit brain floating point number.
    ///
    /// This is not an IEEE kind but uses the same semantics.
    BFloat = BFloatS(16:8),

    /// 8-bit floating point number with S1E5M2 bit layout.
    ///
    /// Follows IEEE-754 conventions with S1E5M2 bit layout as described in
    /// <https://arxiv.org/abs/2209.05433>.
    Float8E5M2 = Float8E5M2S(8:5),

    /// 8-bit floating point number with S1E4M3 bit layout.
    ///
    /// This type mostly follows IEEE-754 conventions with a
    /// bit layout S1E4M3 as described in <https://arxiv.org/abs/2209.05433>.
    /// Unlike IEEE-754 types, there are no infinity values, and NaN is
    /// represented with the exponent and mantissa bits set to all 1s.
    Float8E4M3FN = Float8E4M3FNS(8:4) {
        const NONFINITE_BEHAVIOR: NonfiniteBehavior = NonfiniteBehavior::NanOnly;
    },
}

// FIXME(eddyb) consider moving X87-specific logic to a "has explicit integer bit"
// associated `const` on `Semantics` itself.
/// Floating point semantics for [`X87DoubleExtended`].
///
/// See that type for more details.
pub struct X87DoubleExtendedS;

/// 80-bit floating point number that uses IEEE extended precision semantics, as used
/// by x87 `long double`.
pub type X87DoubleExtended = IeeeFloat<X87DoubleExtendedS>;

impl Semantics for X87DoubleExtendedS {
    const BITS: usize = 80;
    const EXP_BITS: usize = 15;

    // HACK(eddyb) overwriting `EXP_BITS` because its default is incorrect.
    // FIMXE(eddyb) get rid of this by allowing `Semantics` to generically have
    // the concept of "explicitly included integer bit", which is the main way
    // in which the 80-bit X87 format differs from standard IEEE encodings.
    const PRECISION: usize = 64;

    /// For x87 extended precision, we want to make a NaN, not a
    /// pseudo-NaN. Maybe we should expose the ability to make
    /// pseudo-NaNs?
    const QNAN_SIGNIFICAND: Limb = 0b11 << (Self::PRECISION - 2);

    /// Integer bit is explicit in this format. Intel hardware (387 and later)
    /// does not support these bit patterns:
    ///  exponent = all 1's, integer bit 0, significand 0 ("pseudoinfinity")
    ///  exponent = all 1's, integer bit 0, significand nonzero ("pseudoNaN")
    ///  exponent!=0 nor all 1's, integer bit 0 ("unnormal")
    ///  exponent = 0, integer bit 1 ("pseudodenormal")
    /// At the moment, the first three are treated as NaNs, the last one as Normal.
    #[inline]
    fn from_bits(bits: u128) -> IeeeFloat<Self> {
        let sign = bits & (1 << (Self::BITS - 1));
        let exponent = ((bits & !sign) >> (Self::BITS - 1 - Self::EXP_BITS)) & ((1 << Self::EXP_BITS) - 1);
        let mut r = IeeeFloat {
            sig: [bits & ((1 << Self::PRECISION) - 1)],
            // Convert the exponent from its bias representation to a signed integer.
            exp: (exponent as ExpInt) + (Self::MIN_EXP - 1),
            read_only_category_do_not_mutate: Category::Zero,
            read_only_sign_do_not_mutate: sign != 0,
            marker: PhantomData,
        };

        let integer_bit = r.sig[0] >> (Self::PRECISION - 1);

        let category = if r.exp == Self::MIN_EXP - 1 && r.sig == [0] {
            Category::Zero
        } else if r.exp == Self::MAX_EXP + 1 && r.sig == [1 << (Self::PRECISION - 1)] {
            Category::Infinity
        } else if r.exp == Self::MAX_EXP + 1 && r.sig != [1 << (Self::PRECISION - 1)]
            || r.exp != Self::MAX_EXP + 1 && r.exp != Self::MIN_EXP - 1 && integer_bit == 0
        {
            r.exp = Self::MAX_EXP + 1;
            Category::NaN
        } else {
            if r.exp == Self::MIN_EXP - 1 {
                // Denormal.
                r.exp = Self::MIN_EXP;
            }
            Category::Normal
        };

        r.read_only_category_do_not_mutate = category;

        r
    }

    #[inline]
    fn to_bits(x: IeeeFloat<Self>) -> u128 {
        // Get integer bit from significand.
        let integer_bit = sig::get_bit(&x.sig, Self::PRECISION - 1);
        let mut significand = x.sig[0] & ((1 << Self::PRECISION) - 1);
        let exponent = match x.category() {
            Category::Normal => {
                if x.exp == Self::MIN_EXP && !integer_bit {
                    // Denormal.
                    Self::MIN_EXP - 1
                } else {
                    x.exp
                }
            }
            Category::Zero => {
                // FIXME(eddyb) Maybe we should guarantee an invariant instead?
                significand = 0;
                Self::MIN_EXP - 1
            }
            Category::Infinity => {
                // FIXME(eddyb) Maybe we should guarantee an invariant instead?
                significand = 1 << (Self::PRECISION - 1);
                Self::MAX_EXP + 1
            }
            Category::NaN => Self::MAX_EXP + 1,
        };

        // Convert the exponent from a signed integer to its bias representation.
        let exponent = (exponent - (Self::MIN_EXP - 1)) as u128;

        ((x.is_negative() as u128) << (Self::BITS - 1)) | (exponent << Self::PRECISION) | significand
    }
}

float_common_impls!(IeeeFloat<S>);

impl<S: Semantics> PartialEq for IeeeFloat<S> {
    fn eq(&self, rhs: &Self) -> bool {
        self.partial_cmp(rhs) == Some(Ordering::Equal)
    }
}

impl<S: Semantics> PartialOrd for IeeeFloat<S> {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        match (self.category(), rhs.category()) {
            (Category::NaN, _) | (_, Category::NaN) => None,

            (Category::Infinity, Category::Infinity) => Some((!self.is_negative()).cmp(&(!rhs.is_negative()))),

            (Category::Zero, Category::Zero) => Some(Ordering::Equal),

            (Category::Infinity, _) | (Category::Normal, Category::Zero) => {
                Some((!self.is_negative()).cmp(&self.is_negative()))
            }

            (_, Category::Infinity) | (Category::Zero, Category::Normal) => {
                Some(rhs.is_negative().cmp(&(!rhs.is_negative())))
            }

            (Category::Normal, Category::Normal) => {
                // Two normal numbers. Do they have the same sign?
                Some((!self.is_negative()).cmp(&(!rhs.is_negative())).then_with(|| {
                    // Compare absolute values; invert result if negative.
                    let result = self.cmp_abs_normal(*rhs);

                    if self.is_negative() {
                        result.reverse()
                    } else {
                        result
                    }
                }))
            }
        }
    }
}

impl<S: Semantics> Neg for IeeeFloat<S> {
    type Output = Self;
    fn neg(mut self) -> Self {
        self.read_only_sign_do_not_mutate = !self.is_negative();
        self
    }
}

/// Prints this value as a decimal string.
///
/// \param precision The maximum number of digits of
///   precision to output. If there are fewer digits available,
///   zero padding will not be used unless the value is
///   integral and small enough to be expressed in
///   precision digits. 0 means to use the natural
///   precision of the number.
/// \param width The maximum number of zeros to
///   consider inserting before falling back to scientific
///   notation. 0 means to always use scientific notation.
///
/// \param alternate Indicate whether to remove the trailing zero in
///   fraction part or not. Also setting this parameter to true forces
///   producing of output more similar to default printf behavior.
///   Specifically the lower e is used as exponent delimiter and exponent
///   always contains no less than two digits.
///
/// Number       precision    width      Result
/// ------       ---------    -----      ------
/// 1.01E+4              5        2       10100
/// 1.01E+4              4        2       1.01E+4
/// 1.01E+4              5        1       1.01E+4
/// 1.01E-2              5        2       0.0101
/// 1.01E-2              4        2       0.0101
/// 1.01E-2              4        1       1.01E-2
impl<S: Semantics> fmt::Display for IeeeFloat<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let width = f.width().unwrap_or(3);
        let alternate = f.alternate();

        match self.category() {
            Category::Infinity => {
                if self.is_negative() {
                    return f.write_str("-Inf");
                } else {
                    return f.write_str("+Inf");
                }
            }

            Category::NaN => return f.write_str("NaN"),

            Category::Zero => {
                if self.is_negative() {
                    f.write_char('-')?;
                }

                if width == 0 {
                    if alternate {
                        f.write_str("0.0")?;
                        if let Some(n) = f.precision() {
                            for _ in 1..n {
                                f.write_char('0')?;
                            }
                        }
                        f.write_str("e+00")?;
                    } else {
                        f.write_str("0.0E+0")?;
                    }
                } else {
                    f.write_char('0')?;
                }
                return Ok(());
            }

            Category::Normal => {}
        }

        if self.is_negative() {
            f.write_char('-')?;
        }

        // We use enough digits so the number can be round-tripped back to an
        // APFloat. The formula comes from "How to Print Floating-Point Numbers
        // Accurately" by Steele and White.
        // FIXME: Using a formula based purely on the precision is conservative;
        // we can print fewer digits depending on the actual value being printed.

        // precision = 2 + floor(S::PRECISION / lg_2(10))
        let precision = f.precision().unwrap_or(2 + S::PRECISION * 59 / 196);

        // Decompose the number into an APInt and an exponent.
        let mut exp = self.exp - (S::PRECISION as ExpInt - 1);
        let mut sig: DynPrecisionLimbVec = [self.sig[0]].into_iter().collect();

        // Ignore trailing binary zeros.
        let trailing_zeros = sig[0].trailing_zeros();
        let _: Loss = sig::shift_right(&mut sig, &mut exp, trailing_zeros as usize);

        // Change the exponent from 2^e to 10^e.
        if exp == 0 {
            // Nothing to do.
        } else if exp > 0 {
            // Just shift left.
            let shift = exp as usize;
            sig.resize(limbs_for_bits(S::PRECISION + shift), 0);
            sig::shift_left(&mut sig, &mut exp, shift);
        } else {
            // exp < 0
            let mut texp = -exp as usize;

            // We transform this using the identity:
            //   (N)(2^-e) == (N)(5^e)(10^-e)

            // Multiply significand by 5^e.
            //   N * 5^0101 == N * 5^(1*1) * 5^(0*2) * 5^(1*4) * 5^(0*8)
            let mut sig_scratch = DynPrecisionLimbVec::new();
            let mut p5 = DynPrecisionLimbVec::new();
            let mut p5_scratch = DynPrecisionLimbVec::new();
            while texp != 0 {
                if p5.is_empty() {
                    p5.push(5);
                } else {
                    p5_scratch.resize(p5.len() * 2, 0);
                    let _: Loss = sig::mul(&mut p5_scratch, &mut 0, &p5, &p5, p5.len() * 2 * LIMB_BITS);
                    while p5_scratch.last() == Some(&0) {
                        p5_scratch.pop();
                    }
                    mem::swap(&mut p5, &mut p5_scratch);
                }
                if texp & 1 != 0 {
                    sig_scratch.resize(sig.len() + p5.len(), 0);
                    let _: Loss = sig::mul(&mut sig_scratch, &mut 0, &sig, &p5, (sig.len() + p5.len()) * LIMB_BITS);
                    while sig_scratch.last() == Some(&0) {
                        sig_scratch.pop();
                    }
                    mem::swap(&mut sig, &mut sig_scratch);
                }
                texp >>= 1;
            }
        }

        // Fill the buffer.
        let mut buffer = smallvec::SmallVec::<[u8; 64]>::new();

        // Ignore digits from the significand until it is no more
        // precise than is required for the desired precision.
        // 196/59 is a very slight overestimate of lg_2(10).
        let required = (precision * 196 + 58) / 59;
        let mut discard_digits = sig::omsb(&sig).saturating_sub(required) * 59 / 196;
        let mut in_trail = true;
        while !sig.is_empty() {
            // Perform short division by 10 to extract the rightmost digit.
            // rem <- sig % 10
            // sig <- sig / 10
            let mut rem = 0;

            // Use 64-bit division and remainder, with 32-bit chunks from sig.
            sig::each_chunk(&mut sig, 32, |chunk| {
                let chunk = chunk as u32;
                let combined = ((rem as u64) << 32) | (chunk as u64);
                rem = (combined % 10) as u8;
                (combined / 10) as u32 as Limb
            });

            // Reduce the sigificand to avoid wasting time dividing 0's.
            while sig.last() == Some(&0) {
                sig.pop();
            }

            let digit = rem;

            // Ignore digits we don't need.
            if discard_digits > 0 {
                discard_digits -= 1;
                exp += 1;
                continue;
            }

            // Drop trailing zeros.
            if in_trail && digit == 0 {
                exp += 1;
            } else {
                in_trail = false;
                buffer.push(b'0' + digit);
            }
        }

        assert!(!buffer.is_empty(), "no characters in buffer!");

        // Drop down to precision.
        // FIXME: don't do more precise calculations above than are required.
        if buffer.len() > precision {
            // The most significant figures are the last ones in the buffer.
            let mut first_sig = buffer.len() - precision;

            // Round.
            // FIXME: this probably shouldn't use 'round half up'.

            // Rounding down is just a truncation, except we also want to drop
            // trailing zeros from the new result.
            if buffer[first_sig - 1] < b'5' {
                while first_sig < buffer.len() && buffer[first_sig] == b'0' {
                    first_sig += 1;
                }
            } else {
                // Rounding up requires a decimal add-with-carry. If we continue
                // the carry, the newly-introduced zeros will just be truncated.
                for x in &mut buffer[first_sig..] {
                    if *x == b'9' {
                        first_sig += 1;
                    } else {
                        *x += 1;
                        break;
                    }
                }
            }

            exp += first_sig as ExpInt;
            buffer.drain(..first_sig);

            // If we carried through, we have exactly one digit of precision.
            if buffer.is_empty() {
                buffer.push(b'1');
            }
        }

        let digits = buffer.len();

        // Check whether we should use scientific notation.
        let scientific = if width == 0 {
            true
        } else {
            if exp >= 0 {
                // 765e3 --> 765000
                //              ^^^
                // But we shouldn't make the number look more precise than it is.
                exp as usize > width || digits + exp as usize > precision
            } else {
                // Power of the most significant digit.
                let msd = exp + (digits - 1) as ExpInt;
                if msd >= 0 {
                    // 765e-2 == 7.65
                    false
                } else {
                    // 765e-5 == 0.00765
                    //           ^ ^^
                    -msd as usize > width
                }
            }
        };

        // Scientific formatting is pretty straightforward.
        if scientific {
            exp += digits as ExpInt - 1;

            f.write_char(buffer[digits - 1] as char)?;
            f.write_char('.')?;
            let truncate_zero = !alternate;
            if digits == 1 && truncate_zero {
                f.write_char('0')?;
            } else {
                for &d in buffer[..digits - 1].iter().rev() {
                    f.write_char(d as char)?;
                }
            }
            // Fill with zeros up to precision.
            if !truncate_zero && precision > digits - 1 {
                for _ in 0..precision - digits + 1 {
                    f.write_char('0')?;
                }
            }
            // For alternate we use lower 'e'.
            f.write_char(if alternate { 'e' } else { 'E' })?;

            // Exponent always at least two digits if we do not truncate zeros.
            if truncate_zero {
                write!(f, "{:+}", exp)?;
            } else {
                write!(f, "{:+03}", exp)?;
            }

            return Ok(());
        }

        // Non-scientific, positive exponents.
        if exp >= 0 {
            for &d in buffer.iter().rev() {
                f.write_char(d as char)?;
            }
            for _ in 0..exp {
                f.write_char('0')?;
            }
            return Ok(());
        }

        // Non-scientific, negative exponents.
        let unit_place = -exp as usize;
        if unit_place < digits {
            for &d in buffer[unit_place..].iter().rev() {
                f.write_char(d as char)?;
            }
            f.write_char('.')?;
            for &d in buffer[..unit_place].iter().rev() {
                f.write_char(d as char)?;
            }
        } else {
            f.write_str("0.")?;
            for _ in digits..unit_place {
                f.write_char('0')?;
            }
            for &d in buffer.iter().rev() {
                f.write_char(d as char)?;
            }
        }

        Ok(())
    }
}

impl<S: Semantics> fmt::Debug for IeeeFloat<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}({:?} | {}{:?} * 2^{})",
            self,
            self.category(),
            if self.is_negative() { "-" } else { "+" },
            self.sig,
            self.exp
        )
    }
}

// HACK(eddyb) this logic is duplicated throughout the original C++ code,
// but it's a bit too long to keep repeating in the Rust port for all ops.
// FIXME(eddyb) find a better name/organization for all of this functionality
// (`IeeeDefaultExceptionHandling` doesn't have a counterpart in the C++ code).
struct IeeeDefaultExceptionHandling;
impl IeeeDefaultExceptionHandling {
    fn result_from_nan<S: Semantics>(mut r: IeeeFloat<S>) -> StatusAnd<IeeeFloat<S>> {
        assert!(r.is_nan());

        let status = if r.is_signaling() {
            // [IEEE Std 754-2008 6.2]:
            // Under default exception handling, any operation signaling an invalid
            // operation exception and for which a floating-point result is to be
            // delivered shall deliver a quiet NaN.
            let [sig] = &mut r.sig;
            *sig |= if S::QNAN_SIGNIFICAND == X87DoubleExtendedS::QNAN_SIGNIFICAND {
                // HACK(eddyb) remain bug-compatible with the original C++ code
                // which doesn't appear to attempt avoiding creating pseudo-NaNs
                // (see https://github.com/llvm/llvm-project/issues/63938).
                S::QNAN_SIGNIFICAND & S::NAN_PAYLOAD_MASK
            } else {
                S::QNAN_SIGNIFICAND
            };

            // [IEEE Std 754-2008 6.2]:
            // Signaling NaNs shall be reserved operands that, under default exception
            // handling, signal the invalid operation exception(see 7.2) for every
            // general-computational and signaling-computational operation except for
            // the conversions described in 5.12.
            Status::INVALID_OP
        } else {
            // [IEEE Std 754-2008 6.2]:
            // For an operation with quiet NaN inputs, other than maximum and minimum
            // operations, if a floating-point result is to be delivered the result
            // shall be a quiet NaN which should be one of the input NaNs.
            // ...
            // Every general-computational and quiet-computational operation involving
            // one or more input NaNs, none of them signaling, shall signal no
            // exception, except fusedMultiplyAdd might signal the invalid operation
            // exception(see 7.2).
            Status::OK
        };
        status.and(r)
    }

    fn binop_result_from_either_nan<S: Semantics>(a: IeeeFloat<S>, b: IeeeFloat<S>) -> StatusAnd<IeeeFloat<S>> {
        let r = match (a.category(), b.category()) {
            (Category::NaN, _) => a,
            (_, Category::NaN) => b,
            _ => unreachable!(),
        };
        let mut status_and_r = Self::result_from_nan(r);
        if b.is_signaling() {
            status_and_r.status |= Status::INVALID_OP;
        }
        status_and_r
    }
}

impl<S: Semantics> IeeeFloat<S> {
    // HACK(eddyb) allow `Self::qnan` to be used from `IeeeFloat::NAN`.
    // FIXME(eddyb) move back to the trait impl when that can be `const fn`.
    const fn qnan(payload: Option<u128>) -> Self {
        let sig = [S::NAN_SIGNIFICAND_BASE
            | S::QNAN_SIGNIFICAND
            | match payload {
                // Zero out the excess bits of the significand.
                Some(payload) => payload & S::NAN_PAYLOAD_MASK,
                None => 0,
            }];

        let exp = match S::NONFINITE_BEHAVIOR {
            NonfiniteBehavior::IEEE754 => S::MAX_EXP + 1,
            NonfiniteBehavior::NanOnly => S::MAX_EXP,
        };

        IeeeFloat {
            sig,
            exp,
            read_only_category_do_not_mutate: Category::NaN,
            read_only_sign_do_not_mutate: false,
            marker: PhantomData,
        }
    }
}

impl<S: Semantics> Float for IeeeFloat<S> {
    const BITS: usize = S::BITS;
    const PRECISION: usize = S::PRECISION;
    const MAX_EXP: ExpInt = S::MAX_EXP;
    const MIN_EXP: ExpInt = S::MIN_EXP;

    const ZERO: Self = IeeeFloat {
        sig: [0],
        exp: S::MIN_EXP - 1,
        read_only_category_do_not_mutate: Category::Zero,
        read_only_sign_do_not_mutate: false,
        marker: PhantomData,
    };

    const INFINITY: Self = match S::NONFINITE_BEHAVIOR {
        NonfiniteBehavior::IEEE754 => IeeeFloat {
            sig: [0],
            exp: S::MAX_EXP + 1,
            read_only_category_do_not_mutate: Category::Infinity,
            read_only_sign_do_not_mutate: false,
            marker: PhantomData,
        },

        // There is no Inf, so make NaN instead.
        NonfiniteBehavior::NanOnly => Self::NAN,
    };

    const NAN: Self = Self::qnan(None);

    fn qnan(payload: Option<u128>) -> Self {
        // NOTE(eddyb) this is not a recursive self-call, but rather it calls
        // the `const fn` inherent mehtod (see above).
        Self::qnan(payload)
    }

    fn snan(payload: Option<u128>) -> Self {
        let mut snan = Self::qnan(payload);

        let [sig] = &mut snan.sig;

        // We always have to clear the QNaN bit to make it an SNaN.
        *sig &= !(S::QNAN_SIGNIFICAND & S::NAN_PAYLOAD_MASK);

        // If there are no bits set in the payload, we have to set
        // *something* to make it a NaN instead of an infinity;
        // conventionally, this is the next bit down from the QNaN bit.
        if *sig & S::NAN_PAYLOAD_MASK == 0 {
            *sig |= (S::QNAN_SIGNIFICAND & S::NAN_PAYLOAD_MASK) >> 1;
        }

        snan
    }

    fn largest() -> Self {
        // We want (in interchange format):
        //   exponent = 1..10
        //   significand = 1..1
        IeeeFloat {
            sig: [((1 << S::PRECISION) - 1)
                & match S::NONFINITE_BEHAVIOR {
                    // The largest number by magnitude in our format will be the floating point
                    // number with maximum exponent and with significand that is all ones.
                    NonfiniteBehavior::IEEE754 => !0,

                    // The largest number by magnitude in our format will be the floating point
                    // number with maximum exponent and with significand that is all ones except
                    // the LSB.
                    NonfiniteBehavior::NanOnly => !1,
                }],
            exp: S::MAX_EXP,
            read_only_category_do_not_mutate: Category::Normal,
            read_only_sign_do_not_mutate: false,
            marker: PhantomData,
        }
    }

    // We want (in interchange format):
    //   exponent = 0..0
    //   significand = 0..01
    const SMALLEST: Self = IeeeFloat {
        sig: [1],
        exp: S::MIN_EXP,
        read_only_category_do_not_mutate: Category::Normal,
        read_only_sign_do_not_mutate: false,
        marker: PhantomData,
    };

    fn smallest_normalized() -> Self {
        // We want (in interchange format):
        //   exponent = 0..0
        //   significand = 10..0
        IeeeFloat {
            sig: [1 << (S::PRECISION - 1)],
            exp: S::MIN_EXP,
            read_only_category_do_not_mutate: Category::Normal,
            read_only_sign_do_not_mutate: false,
            marker: PhantomData,
        }
    }

    fn add_r(mut self, rhs: Self, round: Round) -> StatusAnd<Self> {
        let status = match (self.category(), rhs.category()) {
            (Category::NaN, _) | (_, Category::NaN) => {
                return IeeeDefaultExceptionHandling::binop_result_from_either_nan(self, rhs);
            }

            (Category::Infinity, Category::Infinity) => {
                // Differently signed infinities can only be validly
                // subtracted.
                if self.is_negative() != rhs.is_negative() {
                    self = Self::NAN;
                    Status::INVALID_OP
                } else {
                    Status::OK
                }
            }

            // Sign may depend on rounding mode; handled below.
            (_, Category::Zero) | (Category::Infinity, Category::Normal) => Status::OK,

            (Category::Zero, _) | (_, Category::Infinity) => {
                self = rhs;
                Status::OK
            }

            (Category::Normal, Category::Normal) => {
                let mut sign = self.is_negative();
                let loss = sig::add_or_sub(
                    &mut self.sig,
                    &mut self.exp,
                    &mut sign,
                    &mut [rhs.sig[0]],
                    rhs.exp,
                    rhs.is_negative(),
                );
                self = self.with_sign(sign);

                let status;
                self = unpack!(status=, self.normalize(round, loss));

                // Can only be zero if we lost no fraction.
                assert!(self.category() != Category::Zero || loss == Loss::ExactlyZero);

                status
            }
        };

        // If two numbers add (exactly) to zero, IEEE 754 decrees it is a
        // positive zero unless rounding to minus infinity, except that
        // adding two like-signed zeroes gives that zero.
        if self.category() == Category::Zero
            && (rhs.category() != Category::Zero || self.is_negative() != rhs.is_negative())
        {
            self = self.with_sign(round == Round::TowardNegative);
        }

        status.and(self)
    }

    // NOTE(eddyb) we can't rely on the `sub_r` method default implementation
    // because NaN handling needs the original `rhs` (i.e. without negation).
    fn sub_r(self, rhs: Self, round: Round) -> StatusAnd<Self> {
        match (self.category(), rhs.category()) {
            (Category::NaN, _) | (_, Category::NaN) => {
                IeeeDefaultExceptionHandling::binop_result_from_either_nan(self, rhs)
            }

            _ => self.add_r(-rhs, round),
        }
    }

    fn mul_r(mut self, rhs: Self, round: Round) -> StatusAnd<Self> {
        self = self.negate_if(rhs.is_negative());

        match (self.category(), rhs.category()) {
            (Category::NaN, _) | (_, Category::NaN) => {
                self = self.negate_if(rhs.is_negative()); // restore the original sign

                IeeeDefaultExceptionHandling::binop_result_from_either_nan(self, rhs)
            }

            (Category::Zero, Category::Infinity) | (Category::Infinity, Category::Zero) => {
                Status::INVALID_OP.and(Self::NAN)
            }

            (Category::Infinity, _) | (_, Category::Infinity) => Status::OK.and(Self::INFINITY.copy_sign(self)),

            (Category::Zero, _) | (_, Category::Zero) => Status::OK.and(Self::ZERO.copy_sign(self)),

            (Category::Normal, Category::Normal) => {
                self.exp += rhs.exp;
                let mut wide_sig = [0; 2];
                let loss = sig::mul(&mut wide_sig, &mut self.exp, &self.sig, &rhs.sig, S::PRECISION);
                self.sig = [wide_sig[0]];
                let mut status;
                self = unpack!(status=, self.normalize(round, loss));
                if loss != Loss::ExactlyZero {
                    status |= Status::INEXACT;
                }
                status.and(self)
            }
        }
    }

    fn mul_add_r(mut self, multiplicand: Self, addend: Self, round: Round) -> StatusAnd<Self> {
        // If and only if all arguments are normal do we need to do an
        // extended-precision calculation.
        if !self.is_finite_non_zero() || !multiplicand.is_finite_non_zero() || !addend.is_finite() {
            let mut status;
            if self.is_finite_non_zero() && multiplicand.is_finite_non_zero() {
                // HACK(eddyb) this corresponds to the case where the C++ code
                // (`multiplySpecials`) is effectively a noop: no multiplication
                // is actually performed, because we're really only here to handle
                // the "infinite/NaN `addend`" special-case, which needs to ignore
                // the "finite * finite" multiplication entirely, instead of letting
                // it e.g. overflow into infinity (and trample over `addend`).
                assert!(!addend.is_finite());
                status = Status::OK;
            } else {
                self = unpack!(status=, self.mul_r(multiplicand, round));
            }

            // FS can only be Status::OK or Status::INVALID_OP. There is no more work
            // to do in the latter case. The IEEE-754R standard says it is
            // implementation-defined in this case whether, if ADDEND is a
            // quiet NaN, we raise invalid op; this implementation does so.
            //
            // If we need to do the addition we can do so with normal
            // precision.
            if status == Status::OK {
                self = unpack!(status=, self.add_r(addend, round));
            }
            return status.and(self);
        }

        // Post-multiplication sign, before addition.
        self = self.negate_if(multiplicand.is_negative());

        // Allocate space for twice as many bits as the original significand, plus one
        // extra bit for the addition to overflow into.
        assert!(limbs_for_bits(S::PRECISION * 2 + 1) <= 2);
        let mut wide_sig = sig::widening_mul(self.sig[0], multiplicand.sig[0]);

        let mut loss = Loss::ExactlyZero;
        let mut omsb = sig::omsb(&wide_sig);
        self.exp += multiplicand.exp;

        // Assume the operands involved in the multiplication are single-precision
        // FP, and the two multiplicants are:
        //     lhs = a23 . a22 ... a0 * 2^e1
        //     rhs = b23 . b22 ... b0 * 2^e2
        // the result of multiplication is:
        //     lhs = c48 c47 c46 . c45 ... c0 * 2^(e1+e2)
        // Note that there are three significant bits at the left-hand side of the
        // radix point: two for the multiplication, and an overflow bit for the
        // addition (that will always be zero at this point). Move the radix point
        // toward left by two bits, and adjust exponent accordingly.
        self.exp += 2;

        if addend.is_non_zero() {
            // Normalize our MSB to one below the top bit to allow for overflow.
            let ext_precision = 2 * S::PRECISION + 1;
            if omsb != ext_precision - 1 {
                assert!(ext_precision > omsb);
                sig::shift_left(&mut wide_sig, &mut self.exp, (ext_precision - 1) - omsb);
            }

            // The intermediate result of the multiplication has "2 * S::PRECISION"
            // signicant bit; adjust the addend to be consistent with mul result.
            let mut ext_addend_sig = [addend.sig[0], 0];

            // Extend the addend significand to ext_precision - 1. This guarantees
            // that the high bit of the significand is zero (same as wide_sig),
            // so the addition will overflow (if it does overflow at all) into the top bit.
            sig::shift_left(&mut ext_addend_sig, &mut 0, ext_precision - 1 - S::PRECISION);

            let mut sign = self.is_negative();
            loss = sig::add_or_sub(
                &mut wide_sig,
                &mut self.exp,
                &mut sign,
                &mut ext_addend_sig,
                addend.exp + 1,
                addend.is_negative(),
            );
            self = self.with_sign(sign);

            omsb = sig::omsb(&wide_sig);
        }

        // Convert the result having "2 * S::PRECISION" significant-bits back to the one
        // having "S::PRECISION" significant-bits. First, move the radix point from
        // poision "2*S::PRECISION - 1" to "S::PRECISION - 1". The exponent need to be
        // adjusted by "2*S::PRECISION - 1" - "S::PRECISION - 1" = "S::PRECISION".
        self.exp -= S::PRECISION as ExpInt + 1;

        // In case MSB resides at the left-hand side of radix point, shift the
        // mantissa right by some amount to make sure the MSB reside right before
        // the radix point (i.e. "MSB . rest-significant-bits").
        if omsb > S::PRECISION {
            let bits = omsb - S::PRECISION;
            loss = sig::shift_right(&mut wide_sig, &mut self.exp, bits).combine(loss);
        }

        self.sig[0] = wide_sig[0];

        let mut status;
        self = unpack!(status=, self.normalize(round, loss));
        if loss != Loss::ExactlyZero {
            status |= Status::INEXACT;
        }

        // If two numbers add (exactly) to zero, IEEE 754 decrees it is a
        // positive zero unless rounding to minus infinity, except that
        // adding two like-signed zeroes gives that zero.
        if self.category() == Category::Zero
            && !status.intersects(Status::UNDERFLOW)
            && self.is_negative() != addend.is_negative()
        {
            self = self.with_sign(round == Round::TowardNegative);
        }

        status.and(self)
    }

    fn div_r(mut self, rhs: Self, round: Round) -> StatusAnd<Self> {
        self = self.negate_if(rhs.is_negative());

        match (self.category(), rhs.category()) {
            (Category::NaN, _) | (_, Category::NaN) => {
                self = self.negate_if(rhs.is_negative()); // restore the original sign

                IeeeDefaultExceptionHandling::binop_result_from_either_nan(self, rhs)
            }

            (Category::Infinity, Category::Infinity) | (Category::Zero, Category::Zero) => {
                Status::INVALID_OP.and(Self::NAN)
            }

            (Category::Infinity | Category::Zero, _) => Status::OK.and(self),

            (_, Category::Infinity) => Status::OK.and(Self::ZERO.copy_sign(self)),

            (_, Category::Zero) => Status::DIV_BY_ZERO.and(Self::INFINITY.copy_sign(self)),

            (Category::Normal, Category::Normal) => {
                self.exp -= rhs.exp;
                let dividend = self.sig[0];
                let loss = sig::div(&mut self.sig, &mut self.exp, &mut [dividend], &mut [rhs.sig[0]], S::PRECISION);
                let mut status;
                self = unpack!(status=, self.normalize(round, loss));
                if loss != Loss::ExactlyZero {
                    status |= Status::INEXACT;
                }
                status.and(self)
            }
        }
    }

    fn ieee_rem(self, rhs: Self) -> StatusAnd<Self> {
        match (self.category(), rhs.category()) {
            (Category::NaN, _) | (_, Category::NaN) => {
                IeeeDefaultExceptionHandling::binop_result_from_either_nan(self, rhs)
            }

            (Category::Infinity, _) | (_, Category::Zero) => Status::INVALID_OP.and(Self::NAN),

            (Category::Zero, _) | (_, Category::Infinity) => Status::OK.and(self),

            (Category::Normal, Category::Normal) => {
                let mut status;

                let mut x = self;
                let mut p = rhs;

                // Make sure the current value is less than twice the denom. If the addition
                // did not succeed (an overflow has happened), which means that the finite
                // value we currently posses must be less than twice the denom (as we are
                // using the same semantics).
                let p2 = unpack!(status=, p + p);
                if status == Status::OK {
                    x = unpack!(status=, x.c_fmod(p2));
                    assert_eq!(status, Status::OK);
                }

                // Lets work with absolute numbers.
                p = p.abs();
                x = x.abs();

                //
                // To calculate the remainder we use the following scheme.
                //
                // The remainder is defained as follows:
                //
                // remainder = numer - rquot * denom = x - r * p
                //
                // Where r is the result of: x/p, rounded toward the nearest integral value
                // (with halfway cases rounded toward the even number).
                //
                // Currently, (after x mod 2p):
                // r is the number of 2p's present inside x, which is inherently, an even
                // number of p's.
                //
                // We may split the remaining calculation into 4 options:
                // - if x < 0.5p then we round to the nearest number with is 0, and are done.
                // - if x == 0.5p then we round to the nearest even number which is 0, and we
                //   are done as well.
                // - if 0.5p < x < p then we round to nearest number which is 1, and we have
                //   to subtract 1p at least once.
                // - if x >= p then we must subtract p at least once, as x must be a
                //   remainder.
                //
                // By now, we were done, or we added 1 to r, which in turn, now an odd number.
                //
                // We can now split the remaining calculation to the following 3 options:
                // - if x < 0.5p then we round to the nearest number with is 0, and are done.
                // - if x == 0.5p then we round to the nearest even number. As r is odd, we
                //   must round up to the next even number. so we must subtract p once more.
                // - if x > 0.5p (and inherently x < p) then we must round r up to the next
                //   integral, and subtract p once more.
                //

                // Return `x * 2` at no loss of precision (i.e. no overflow).
                //
                // HACK(eddyb) this may seem a bit sketchy because it can return
                // values that `normalize` would've replaced with `overflow_result`
                // (e.g. overflowing to infinity), but the result is only used for
                // comparisons, where both sides of such comparison can be seen
                // as transiently having a larger *effective* exponent range.
                let lossless_2x = |mut x: Self| {
                    x.exp += 1;

                    if x.exp >= Self::MAX_EXP {
                        // HACK(eddyb) skip lossy `normalize` (see above).
                    } else {
                        let status;
                        x = unpack!(status=, x.normalize(Round::NearestTiesToEven, Loss::ExactlyZero));
                        assert_eq!(status, Status::OK);
                    }

                    x
                };

                if lossless_2x(x) > p {
                    x = unpack!(status=, x - p);
                    assert_eq!(status, Status::OK);

                    if lossless_2x(x) >= p {
                        x = unpack!(status=, x - p);
                        assert_eq!(status, Status::OK);
                    }
                }

                if x.is_zero() {
                    Status::OK.and(x.copy_sign(self)) // IEEE754 requires this
                } else {
                    Status::OK.and(x.negate_if(self.is_negative()))
                }
            }
        }
    }

    fn c_fmod(mut self, rhs: Self) -> StatusAnd<Self> {
        match (self.category(), rhs.category()) {
            (Category::NaN, _) | (_, Category::NaN) => {
                IeeeDefaultExceptionHandling::binop_result_from_either_nan(self, rhs)
            }

            (Category::Infinity, _) | (_, Category::Zero) => Status::INVALID_OP.and(Self::NAN),

            (Category::Zero, _) | (_, Category::Infinity) => Status::OK.and(self),

            (Category::Normal, Category::Normal) => {
                let orig = self;

                while self.is_finite_non_zero()
                    && rhs.is_finite_non_zero()
                    && self.cmp_abs_normal(rhs) != Ordering::Less
                {
                    let exp = self.ilogb() - rhs.ilogb();
                    let mut v = rhs.scalbn(exp);
                    // `v` can overflow to NaN with `NonfiniteBehavior::NanOnly`, so explicitly
                    // check for it.
                    if v.is_nan() || self.cmp_abs_normal(v) == Ordering::Less {
                        v = rhs.scalbn(exp - 1);
                    }
                    v = v.copy_sign(self);

                    let status;
                    self = unpack!(status=, self - v);
                    assert_eq!(status, Status::OK);
                }
                if self.is_zero() {
                    self = self.copy_sign(orig);
                }
                Status::OK.and(self)
            }
        }
    }

    fn round_to_integral(self, round: Round) -> StatusAnd<Self> {
        match self.category() {
            Category::NaN => IeeeDefaultExceptionHandling::result_from_nan(self),

            // [IEEE Std 754-2008 6.1]:
            // The behavior of infinity in floating-point arithmetic is derived from the
            // limiting cases of real arithmetic with operands of arbitrarily
            // large magnitude, when such a limit exists.
            // ...
            // Operations on infinite operands are usually exact and therefore signal no
            // exceptions ...
            Category::Infinity => Status::OK.and(self),

            // [IEEE Std 754-2008 6.3]:
            // ... the sign of the result of conversions, the quantize operation, the
            // roundToIntegral operations, and the roundToIntegralExact(see 5.3.1) is
            // the sign of the first or only operand.
            Category::Zero => Status::OK.and(self),

            Category::Normal => {
                // If the exponent is large enough, we know that this value is already
                // integral, and the arithmetic below would potentially cause it to saturate
                // to +/-Inf. Bail out early instead.
                if self.exp + 1 >= S::PRECISION as ExpInt {
                    return Status::OK.and(self);
                }

                // The algorithm here is quite simple: we add 2^(p-1), where p is the
                // precision of our format, and then subtract it back off again. The choice
                // of rounding modes for the addition/subtraction determines the rounding mode
                // for our integral rounding as well.
                // NOTE: When the input value is negative, we do subtraction followed by
                // addition instead.
                assert!(S::PRECISION <= 128);
                let mut status;
                let magic_const = unpack!(status=, Self::from_u128(1 << (S::PRECISION - 1)));
                assert_eq!(status, Status::OK);
                let magic_const = magic_const.copy_sign(self);

                let mut r = self;
                r = unpack!(status=, r.add_r(magic_const, round));

                // Current value and 'MagicConstant' are both integers, so the result of the
                // subtraction is always exact according to Sterbenz' lemma.
                r = r.sub_r(magic_const, round).value;

                // Restore the input sign to handle the case of zero result
                // correctly.
                status.and(r.copy_sign(self))
            }
        }
    }

    fn next_up(mut self) -> StatusAnd<Self> {
        // Compute nextUp(x), handling each float category separately.
        match self.category() {
            Category::Infinity => {
                if self.is_negative() {
                    // nextUp(-inf) = -largest
                    Status::OK.and(-Self::largest())
                } else {
                    // nextUp(+inf) = +inf
                    Status::OK.and(self)
                }
            }
            Category::NaN => {
                // IEEE-754R 2008 6.2 Par 2: nextUp(sNaN) = qNaN. Set Invalid flag.
                // IEEE-754R 2008 6.2: nextUp(qNaN) = qNaN. Must be identity so we do not
                //                     change the payload.
                if self.is_signaling() {
                    // For consistency, propagate the sign of the sNaN to the qNaN.
                    Status::INVALID_OP.and(Self::NAN.copy_sign(self))
                } else {
                    Status::OK.and(self)
                }
            }
            Category::Zero => {
                // nextUp(pm 0) = +smallest
                Status::OK.and(Self::SMALLEST)
            }
            Category::Normal => {
                // nextUp(-smallest) = -0
                if self.is_smallest() && self.is_negative() {
                    return Status::OK.and(-Self::ZERO);
                }

                // nextUp(largest) == INFINITY
                if self.is_largest() && !self.is_negative() {
                    return Status::OK.and(Self::INFINITY);
                }

                // Excluding the integral bit. This allows us to test for binade boundaries.
                let sig_mask = (1 << (S::PRECISION - 1)) - 1;

                // nextUp(normal) == normal + inc.
                if self.is_negative() {
                    // If we are negative, we need to decrement the significand.

                    // We only cross a binade boundary that requires adjusting the exponent
                    // if:
                    //   1. exponent != S::MIN_EXP. This implies we are not in the
                    //   smallest binade or are dealing with denormals.
                    //   2. Our significand excluding the integral bit is all zeros.
                    let crossing_binade_boundary = self.exp != S::MIN_EXP && self.sig[0] & sig_mask == 0;

                    // Decrement the significand.
                    //
                    // We always do this since:
                    //   1. If we are dealing with a non-binade decrement, by definition we
                    //   just decrement the significand.
                    //   2. If we are dealing with a normal -> normal binade decrement, since
                    //   we have an explicit integral bit the fact that all bits but the
                    //   integral bit are zero implies that subtracting one will yield a
                    //   significand with 0 integral bit and 1 in all other spots. Thus we
                    //   must just adjust the exponent and set the integral bit to 1.
                    //   3. If we are dealing with a normal -> denormal binade decrement,
                    //   since we set the integral bit to 0 when we represent denormals, we
                    //   just decrement the significand.
                    sig::decrement(&mut self.sig);

                    if crossing_binade_boundary {
                        // Our result is a normal number. Do the following:
                        // 1. Set the integral bit to 1.
                        // 2. Decrement the exponent.
                        sig::set_bit(&mut self.sig, S::PRECISION - 1);
                        self.exp -= 1;
                    }
                } else {
                    // If we are positive, we need to increment the significand.

                    // We only cross a binade boundary that requires adjusting the exponent if
                    // the input is not a denormal and all of said input's significand bits
                    // are set. If all of said conditions are true: clear the significand, set
                    // the integral bit to 1, and increment the exponent. If we have a
                    // denormal always increment since moving denormals and the numbers in the
                    // smallest normal binade have the same exponent in our representation.
                    let crossing_binade_boundary = !self.is_denormal() && self.sig[0] & sig_mask == sig_mask;

                    if crossing_binade_boundary {
                        self.sig = [0];
                        sig::set_bit(&mut self.sig, S::PRECISION - 1);
                        assert_ne!(
                            self.exp,
                            S::MAX_EXP,
                            "We can not increment an exponent beyond the MAX_EXP \
                             allowed by the given floating point semantics."
                        );
                        self.exp += 1;
                    } else {
                        sig::increment(&mut self.sig);
                    }
                }
                Status::OK.and(self)
            }
        }
    }

    fn from_bits(input: u128) -> Self {
        // Dispatch to semantics.
        S::from_bits(input)
    }

    fn from_u128_r(input: u128, round: Round) -> StatusAnd<Self> {
        IeeeFloat {
            sig: [input],
            exp: S::PRECISION as ExpInt - 1,
            read_only_category_do_not_mutate: Category::Normal,
            read_only_sign_do_not_mutate: false,
            marker: PhantomData,
        }
        .normalize(round, Loss::ExactlyZero)
    }

    fn from_str_r(s: &str, mut round: Round) -> Result<StatusAnd<Self>, ParseError> {
        if s.is_empty() {
            return Err(ParseError("Invalid string length"));
        }

        // Handle a leading minus sign.
        let (minus, s) = s.strip_prefix("-").map(|s| (true, s)).unwrap_or((false, s));
        let from_abs = |r: Self| r.negate_if(minus);

        // Handle a leading plus sign (mutually exclusive with minus).
        let (explicit_plus, s) = s
            .strip_prefix("+")
            .filter(|_| !minus)
            .map(|s| (true, s))
            .unwrap_or((false, s));

        // Handle special cases.
        let special = match s {
            "Inf" if minus || explicit_plus => Some(Self::INFINITY),

            "inf" | "INFINITY" if !explicit_plus => Some(Self::INFINITY),

            _ if !explicit_plus => {
                // If we have a 's' (or 'S') prefix, then this is a Signaling NaN.
                let (is_signaling, s) = s.strip_prefix(['s', 'S']).map_or((false, s), |s| (true, s));

                s.strip_prefix("nan").or_else(|| s.strip_prefix("NaN")).and_then(|s| {
                    // Allow the payload to be inside parentheses.
                    let s = s
                        .strip_prefix("(")
                        .and_then(|s| {
                            // Parentheses should be balanced (and not empty).
                            s.strip_suffix(")").filter(|s| !s.is_empty())
                        })
                        .unwrap_or(s);

                    let payload = if s.is_empty() {
                        // A NaN without payload.
                        None
                    } else {
                        // Determine the payload number's radix.
                        let (radix, s) = s
                            .strip_prefix("0")
                            .filter(|s| !s.is_empty())
                            .map(|s| s.strip_prefix(['x', 'X']).map(|s| (16, s)).unwrap_or((8, s)))
                            .unwrap_or((10, s));

                        // Parse the payload and make the NaN.
                        Some(u128::from_str_radix(s, radix).ok()?)
                    };

                    Some(if is_signaling {
                        Self::snan(payload)
                    } else {
                        Self::qnan(payload)
                    })
                })
            }

            _ => None,
        };
        if let Some(r) = special {
            return Ok(Status::OK.and(from_abs(r)));
        }

        if s.is_empty() {
            return Err(ParseError("String has no digits"));
        }

        // Adjust the rounding mode for the absolute value below.
        round = round.negate_if(minus);

        let (is_hex, s) = s
            .strip_prefix("0")
            .and_then(|s| s.strip_prefix(['x', 'X']))
            .map(|s| (true, s))
            .unwrap_or((false, s));

        let r = if is_hex {
            if s.is_empty() {
                return Err(ParseError("Invalid string"));
            }
            Self::from_hexadecimal_string(s, round)?
        } else {
            Self::from_decimal_string(s, round)?
        };

        Ok(r.map(from_abs))
    }

    fn to_bits(self) -> u128 {
        // Dispatch to semantics.
        S::to_bits(self)
    }

    fn to_u128_r(self, width: usize, round: Round, is_exact: &mut bool) -> StatusAnd<u128> {
        // The result of trying to convert a number too large.
        let overflow = if self.is_negative() {
            // Negative numbers cannot be represented as unsigned.
            0
        } else {
            // Largest unsigned integer of the given width.
            !0 >> (128 - width)
        };

        *is_exact = false;

        match self.category() {
            Category::NaN => Status::INVALID_OP.and(0),

            Category::Infinity => Status::INVALID_OP.and(overflow),

            Category::Zero => {
                // Negative zero can't be represented as an int.
                *is_exact = !self.is_negative();
                Status::OK.and(0)
            }

            Category::Normal => {
                let mut r = 0;

                // Step 1: place our absolute value, with any fraction truncated, in
                // the destination.
                let truncated_bits = if self.exp < 0 {
                    // Our absolute value is less than one; truncate everything.
                    // For exponent -1 the integer bit represents .5, look at that.
                    // For smaller exponents leftmost truncated bit is 0.
                    S::PRECISION - 1 + (-self.exp) as usize
                } else {
                    // We want the most significant (exponent + 1) bits; the rest are
                    // truncated.
                    let bits = self.exp as usize + 1;

                    // Hopelessly large in magnitude?
                    if bits > width {
                        return Status::INVALID_OP.and(overflow);
                    }

                    if bits < S::PRECISION {
                        // We truncate (S::PRECISION - bits) bits.
                        r = self.sig[0] >> (S::PRECISION - bits);
                        S::PRECISION - bits
                    } else {
                        // We want at least as many bits as are available.
                        r = self.sig[0] << (bits - S::PRECISION);
                        0
                    }
                };

                // Step 2: work out any lost fraction, and increment the absolute
                // value if we would round away from zero.
                let mut loss = Loss::ExactlyZero;
                if truncated_bits > 0 {
                    loss = Loss::through_truncation(&self.sig, truncated_bits);
                    if loss != Loss::ExactlyZero && self.round_away_from_zero(round, loss, truncated_bits) {
                        r = r.wrapping_add(1);
                        if r == 0 {
                            return Status::INVALID_OP.and(overflow); // Overflow.
                        }
                    }
                }

                // Step 3: check if we fit in the destination.
                if r > overflow {
                    return Status::INVALID_OP.and(overflow);
                }

                if loss == Loss::ExactlyZero {
                    *is_exact = true;
                    Status::OK.and(r)
                } else {
                    Status::INEXACT.and(r)
                }
            }
        }
    }

    fn cmp_abs_normal(self, rhs: Self) -> Ordering {
        assert!(self.is_finite_non_zero());
        assert!(rhs.is_finite_non_zero());

        // If exponents are equal, do an unsigned comparison of the significands.
        self.exp.cmp(&rhs.exp).then_with(|| sig::cmp(&self.sig, &rhs.sig))
    }

    fn bitwise_eq(self, rhs: Self) -> bool {
        if self.category() != rhs.category() || self.is_negative() != rhs.is_negative() {
            return false;
        }

        if self.category() == Category::Zero || self.category() == Category::Infinity {
            return true;
        }

        if self.is_finite_non_zero() && self.exp != rhs.exp {
            return false;
        }

        self.sig == rhs.sig
    }

    fn is_negative(self) -> bool {
        self.read_only_sign_do_not_mutate
    }

    fn is_denormal(self) -> bool {
        self.is_finite_non_zero() && self.exp == S::MIN_EXP && !sig::get_bit(&self.sig, S::PRECISION - 1)
    }

    fn is_signaling(self) -> bool {
        // IEEE-754R 2008 6.2.1: A signaling NaN bit string should be encoded with the
        // first bit of the trailing significand being 0.
        self.is_nan() && self.sig[0] & S::QNAN_SIGNIFICAND != S::QNAN_SIGNIFICAND
    }

    fn category(self) -> Category {
        self.read_only_category_do_not_mutate
    }

    fn get_exact_inverse(self) -> Option<Self> {
        // Special floats and denormals have no exact inverse.
        if !self.is_finite_non_zero() {
            return None;
        }

        // Check that the number is a power of two by making sure that only the
        // integer bit is set in the significand.
        if self.sig != [1 << (S::PRECISION - 1)] {
            return None;
        }

        // Get the inverse.
        let mut reciprocal = Self::from_u128(1).value;
        let status;
        reciprocal = unpack!(status=, reciprocal / self);
        if status != Status::OK {
            return None;
        }

        // Avoid multiplication with a denormal, it is not safe on all platforms and
        // may be slower than a normal division.
        if reciprocal.is_denormal() {
            return None;
        }

        assert!(reciprocal.is_finite_non_zero());
        assert_eq!(reciprocal.sig, [1 << (S::PRECISION - 1)]);

        Some(reciprocal)
    }

    fn ilogb(mut self) -> ExpInt {
        if self.is_nan() {
            return IEK_NAN;
        }
        if self.is_zero() {
            return IEK_ZERO;
        }
        if self.is_infinite() {
            return IEK_INF;
        }
        if !self.is_denormal() {
            return self.exp;
        }

        let sig_bits = (S::PRECISION - 1) as ExpInt;
        self.exp += sig_bits;
        self = self.normalize(Round::NearestTiesToEven, Loss::ExactlyZero).value;
        self.exp - sig_bits
    }

    fn scalbn_r(mut self, exp: ExpInt, round: Round) -> Self {
        // If exp is wildly out-of-scale, simply adding it to self.exp will
        // overflow; clamp it to a safe range before adding, but ensure that the range
        // is large enough that the clamp does not change the result. The range we
        // need to support is the difference between the largest possible exponent and
        // the normalized exponent of half the smallest denormal.

        let sig_bits = (S::PRECISION - 1) as i32;
        let max_change = S::MAX_EXP as i32 - (S::MIN_EXP as i32 - sig_bits) + 1;

        // Clamp to one past the range ends to let normalize handle overflow.
        let exp_change = cmp::min(cmp::max(exp as i32, -max_change - 1), max_change);
        self.exp = self.exp.saturating_add(exp_change as ExpInt);
        self = self.normalize(round, Loss::ExactlyZero).value;
        if self.is_nan() {
            self = IeeeDefaultExceptionHandling::result_from_nan(self).value;
        }
        self
    }

    fn frexp_r(mut self, exp: &mut ExpInt, round: Round) -> Self {
        *exp = self.ilogb();

        // Quiet signalling nans.
        if *exp == IEK_NAN {
            self = IeeeDefaultExceptionHandling::result_from_nan(self).value;
            return self;
        }

        if *exp == IEK_INF {
            return self;
        }

        // 1 is added because frexp is defined to return a normalized fraction in
        // +/-[0.5, 1.0), rather than the usual +/-[1.0, 2.0).
        if *exp == IEK_ZERO {
            *exp = 0;
        } else {
            *exp += 1;
        }
        self.scalbn_r(-*exp, round)
    }
}

impl<S: Semantics, T: Semantics> FloatConvert<IeeeFloat<T>> for IeeeFloat<S> {
    fn convert_r(mut self, round: Round, loses_info: &mut bool) -> StatusAnd<IeeeFloat<T>> {
        // FIXME(eddyb) move this into the return result.
        *loses_info = false;

        // x86 has some unusual NaNs which cannot be represented in any other
        // format; note them here.
        fn is_x87_double_extended<S: Semantics>() -> bool {
            S::QNAN_SIGNIFICAND == X87DoubleExtendedS::QNAN_SIGNIFICAND
        }
        let loses_x87_pseudo_nan = is_x87_double_extended::<S>()
            && !is_x87_double_extended::<T>()
            && self.category() == Category::NaN
            && (self.sig[0] & S::QNAN_SIGNIFICAND) != S::QNAN_SIGNIFICAND;

        // NOTE(eddyb) this is done early because the target semantics may not
        // actually support encoding the distinction between SNaN and QNaN.
        //
        // Convert of sNaN creates qNaN and raises an exception (invalid op).
        // This also guarantees that a sNaN does not become Inf on a truncation
        // that loses all payload bits.
        let mut status = Status::OK;
        if self.is_nan() {
            self = unpack!(status|=, IeeeDefaultExceptionHandling::result_from_nan(self));
        }

        let Self { mut sig, mut exp, .. } = self;

        // If this is a truncation of a denormal number, and the target semantics
        // has larger exponent range than the source semantics (this can happen
        // when truncating from PowerPC double-double to double format), the
        // right shift could lose result mantissa bits. Adjust exponent instead
        // of performing excessive shift.
        // Also do a similar trick in case shifting denormal would produce zero
        // significand as this case isn't handled correctly by normalize.
        let mut shift = T::PRECISION as ExpInt - S::PRECISION as ExpInt;
        if shift < 0 && self.is_finite_non_zero() {
            let omsb = sig::omsb(&sig) as ExpInt;
            let mut exp_change = omsb - S::PRECISION as ExpInt;
            if exp + exp_change < T::MIN_EXP {
                exp_change = T::MIN_EXP - exp;
            }
            if exp_change < shift {
                exp_change = shift;
            }
            if exp_change < 0 {
                shift -= exp_change;
                exp += exp_change;
            } else if omsb <= -shift {
                exp_change = omsb + shift - 1; // leave at least one bit set
                shift -= exp_change;
                exp += exp_change;
            }
        }

        // If this is a truncation, perform the shift.
        let mut loss = Loss::ExactlyZero;
        if shift < 0 && (self.is_finite_non_zero() || self.category() == Category::NaN && S::NAN_PAYLOAD_MASK != 0) {
            loss = sig::shift_right(&mut sig, &mut 0, -shift as usize);
        }

        // If this is an extension, perform the shift.
        if shift > 0 && (self.is_finite_non_zero() || self.category() == Category::NaN) {
            sig::shift_left(&mut sig, &mut 0, shift as usize);
        }

        let r = match self.category() {
            Category::Normal => {
                let r = IeeeFloat::<T> {
                    sig,
                    exp,
                    read_only_category_do_not_mutate: self.category(),
                    read_only_sign_do_not_mutate: self.is_negative(),
                    marker: PhantomData,
                };
                unpack!(status|=, r.normalize(round, loss))
            }
            Category::NaN => {
                *loses_info = loss != Loss::ExactlyZero
                    || loses_x87_pseudo_nan
                    || S::NAN_PAYLOAD_MASK != 0 && T::NAN_PAYLOAD_MASK == 0;

                IeeeFloat::<T>::qnan(Some(sig[0])).with_sign(self.is_negative())
            }
            Category::Infinity => IeeeFloat::<T>::INFINITY.with_sign(self.is_negative()),
            Category::Zero => IeeeFloat::<T>::ZERO.with_sign(self.is_negative()),
        };

        // NOTE(eddyb) this catches all cases of e.g. ±Inf turning into NaN,
        // because of `T::NONFINITE_BEHAVIOR` not being `IEEE754`.
        if matches!(self.category(), Category::Infinity | Category::Zero)
            && (r.category() != self.category() || r.is_negative() != self.is_negative())
        {
            status |= Status::INEXACT;
        }

        *loses_info |= (status - Status::INVALID_OP) != Status::OK;

        status.and(r)
    }
}

impl<S: Semantics> IeeeFloat<S> {
    /// Handle positive overflow. We either return infinity or
    /// the largest finite number. For negative overflow,
    /// negate the `round` argument before calling.
    fn overflow_result(round: Round) -> StatusAnd<Self> {
        match round {
            // Infinity?
            Round::NearestTiesToEven | Round::NearestTiesToAway | Round::TowardPositive => {
                (Status::OVERFLOW | Status::INEXACT).and(Self::INFINITY)
            }
            // Otherwise we become the largest finite number.
            Round::TowardNegative | Round::TowardZero => Status::INEXACT.and(Self::largest()),
        }
    }

    /// Returns TRUE if, when truncating the current number, with BIT the
    /// new LSB, with the given lost fraction and rounding mode, the result
    /// would need to be rounded away from zero (i.e., by increasing the
    /// signficand). This routine must work for Category::Zero of both signs, and
    /// Category::Normal numbers.
    fn round_away_from_zero(&self, round: Round, loss: Loss, bit: usize) -> bool {
        // NaNs and infinities should not have lost fractions.
        assert!(self.is_finite_non_zero() || self.is_zero());

        // Current callers never pass this so we don't handle it.
        assert_ne!(loss, Loss::ExactlyZero);

        match round {
            Round::NearestTiesToAway => loss == Loss::ExactlyHalf || loss == Loss::MoreThanHalf,
            Round::NearestTiesToEven => {
                if loss == Loss::MoreThanHalf {
                    return true;
                }

                // Our zeros don't have a significand to test.
                if loss == Loss::ExactlyHalf && self.category() != Category::Zero {
                    return sig::get_bit(&self.sig, bit);
                }

                false
            }
            Round::TowardZero => false,
            Round::TowardPositive => !self.is_negative(),
            Round::TowardNegative => self.is_negative(),
        }
    }

    fn normalize(mut self, round: Round, mut loss: Loss) -> StatusAnd<Self> {
        if !self.is_finite_non_zero() {
            return Status::OK.and(self);
        }

        // Before rounding normalize the exponent of Category::Normal numbers.
        let mut omsb = sig::omsb(&self.sig);

        // Only skip this `if` if the value is exactly zero.
        if omsb > 0 || loss != Loss::ExactlyZero {
            // OMSB is numbered from 1. We want to place it in the integer
            // bit numbered PRECISION if possible, with a compensating change in
            // the exponent.
            let mut final_exp = self.exp.saturating_add(omsb as ExpInt - S::PRECISION as ExpInt);

            // If the resulting exponent is too high, overflow according to
            // the rounding mode.
            if final_exp > S::MAX_EXP {
                let round = round.negate_if(self.is_negative());
                return Self::overflow_result(round).map(|r| r.copy_sign(self));
            }

            // Subnormal numbers have exponent MIN_EXP, and their MSB
            // is forced based on that.
            if final_exp < S::MIN_EXP {
                final_exp = S::MIN_EXP;
            }

            // Shifting left is easy as we don't lose precision.
            if final_exp < self.exp {
                assert_eq!(loss, Loss::ExactlyZero);

                let exp_change = (self.exp - final_exp) as usize;
                sig::shift_left(&mut self.sig, &mut self.exp, exp_change);

                return Status::OK.and(self);
            }

            // Shift right and capture any new lost fraction.
            if final_exp > self.exp {
                let exp_change = (final_exp - self.exp) as usize;
                loss = sig::shift_right(&mut self.sig, &mut self.exp, exp_change).combine(loss);

                // Keep OMSB up-to-date.
                omsb = omsb.saturating_sub(exp_change);
            }
        }

        // NOTE(eddyb) for `NonfiniteBehavior::NanOnly`, the unique `NAN` takes up
        // the largest significand of `MAX_EXP` (which also has normals), though
        // comparing significands needs to ignore the integer bit `NAN` lacks.
        if S::NONFINITE_BEHAVIOR == NonfiniteBehavior::NanOnly
            && self.exp == Self::NAN.exp
            && [self.sig[0] & S::NAN_SIGNIFICAND_BASE] == Self::NAN.sig
        {
            return Self::overflow_result(round).map(|r| r.copy_sign(self));
        }

        // Now round the number according to round given the lost
        // fraction.

        // As specified in IEEE 754, since we do not trap we do not report
        // underflow for exact results.
        if loss == Loss::ExactlyZero {
            // Canonicalize zeros.
            if omsb == 0 {
                self = Self::ZERO.copy_sign(self);
            }

            return Status::OK.and(self);
        }

        // Increment the significand if we're rounding away from zero.
        if self.round_away_from_zero(round, loss, 0) {
            if omsb == 0 {
                self.exp = S::MIN_EXP;
            }

            // We should never overflow.
            assert_eq!(sig::increment(&mut self.sig), 0);
            omsb = sig::omsb(&self.sig);

            // Did the significand increment overflow?
            if omsb == S::PRECISION + 1 {
                // Renormalize by incrementing the exponent and shifting our
                // significand right one. However if we already have the
                // maximum exponent we overflow to infinity.
                if self.exp == S::MAX_EXP {
                    return (Status::OVERFLOW | Status::INEXACT).and(Self::INFINITY.copy_sign(self));
                }

                let _: Loss = sig::shift_right(&mut self.sig, &mut self.exp, 1);

                return Status::INEXACT.and(self);
            }

            // NOTE(eddyb) for `NonfiniteBehavior::NanOnly`, the unique `NAN` takes up
            // the largest significand of `MAX_EXP` (which also has normals), though
            // comparing significands needs to ignore the integer bit `NAN` lacks.
            if S::NONFINITE_BEHAVIOR == NonfiniteBehavior::NanOnly
                && self.exp == Self::NAN.exp
                && [self.sig[0] & S::NAN_SIGNIFICAND_BASE] == Self::NAN.sig
            {
                return Self::overflow_result(round).map(|r| r.copy_sign(self));
            }
        }

        // The normal case - we were and are not denormal, and any
        // significand increment above didn't overflow.
        if omsb == S::PRECISION {
            return Status::INEXACT.and(self);
        }

        // We have a non-zero denormal.
        assert!(omsb < S::PRECISION);

        // Canonicalize zeros.
        if omsb == 0 {
            self = Self::ZERO.copy_sign(self);
        }

        // The Category::Zero case is a denormal that underflowed to zero.
        (Status::UNDERFLOW | Status::INEXACT).and(self)
    }

    fn from_hexadecimal_string(s: &str, round: Round) -> Result<StatusAnd<Self>, ParseError> {
        let mut r = IeeeFloat {
            sig: [0],
            exp: 0,
            read_only_category_do_not_mutate: Category::Normal,
            read_only_sign_do_not_mutate: false,
            marker: PhantomData,
        };

        let mut any_digits = false;
        let mut has_exp = false;
        let mut bit_pos = LIMB_BITS as isize;
        let mut loss = None;

        // Without leading or trailing zeros, irrespective of the dot.
        let mut first_sig_digit = None;
        let mut dot = s.len();

        for (p, c) in s.char_indices() {
            // Skip leading zeros and any (hexa)decimal point.
            if c == '.' {
                if dot != s.len() {
                    return Err(ParseError("String contains multiple dots"));
                }
                dot = p;
            } else if let Some(hex_value) = c.to_digit(16) {
                any_digits = true;

                if first_sig_digit.is_none() {
                    if hex_value == 0 {
                        continue;
                    }
                    first_sig_digit = Some(p);
                }

                // Store the number while we have space.
                bit_pos -= 4;
                if bit_pos >= 0 {
                    r.sig[0] |= (hex_value as Limb) << bit_pos;
                } else {
                    // If zero or one-half (the hexadecimal digit 8) are followed
                    // by non-zero, they're a little more than zero or one-half.
                    if let Some(ref mut loss) = loss {
                        if hex_value != 0 {
                            if *loss == Loss::ExactlyZero {
                                *loss = Loss::LessThanHalf;
                            }
                            if *loss == Loss::ExactlyHalf {
                                *loss = Loss::MoreThanHalf;
                            }
                        }
                    } else {
                        loss = Some(match hex_value {
                            0 => Loss::ExactlyZero,
                            1..=7 => Loss::LessThanHalf,
                            8 => Loss::ExactlyHalf,
                            9..=15 => Loss::MoreThanHalf,
                            _ => unreachable!(),
                        });
                    }
                }
            } else if c == 'p' || c == 'P' {
                if !any_digits {
                    return Err(ParseError("Significand has no digits"));
                }

                if dot == s.len() {
                    dot = p;
                }

                let mut chars = s[p + 1..].chars().peekable();

                // Adjust for the given exponent.
                let exp_minus = chars.peek() == Some(&'-');
                if exp_minus || chars.peek() == Some(&'+') {
                    chars.next();
                }

                for c in chars {
                    if let Some(value) = c.to_digit(10) {
                        has_exp = true;
                        r.exp = r.exp.saturating_mul(10).saturating_add(value as ExpInt);
                    } else {
                        return Err(ParseError("Invalid character in exponent"));
                    }
                }
                if !has_exp {
                    return Err(ParseError("Exponent has no digits"));
                }

                r.exp = r.exp.negate_if(exp_minus);

                break;
            } else {
                return Err(ParseError("Invalid character in significand"));
            }
        }
        if !any_digits {
            return Err(ParseError("Significand has no digits"));
        }

        // Hex floats require an exponent but not a hexadecimal point.
        if !has_exp {
            return Err(ParseError("Hex strings require an exponent"));
        }

        // Ignore the exponent if we are zero.
        let first_sig_digit = match first_sig_digit {
            Some(p) => p,
            None => return Ok(Status::OK.and(Self::ZERO)),
        };

        // Calculate the exponent adjustment implicit in the number of
        // significant digits and adjust for writing the significand starting
        // at the most significant nibble.
        let exp_adjustment = if dot > first_sig_digit {
            ExpInt::try_from(dot - first_sig_digit).unwrap()
        } else {
            -ExpInt::try_from(first_sig_digit - dot - 1).unwrap()
        };
        let exp_adjustment = exp_adjustment
            .saturating_mul(4)
            .saturating_sub(1)
            .saturating_add(S::PRECISION as ExpInt)
            .saturating_sub(LIMB_BITS as ExpInt);
        r.exp = r.exp.saturating_add(exp_adjustment);

        Ok(r.normalize(round, loss.unwrap_or(Loss::ExactlyZero)))
    }

    fn from_decimal_string(s: &str, round: Round) -> Result<StatusAnd<Self>, ParseError> {
        // Given a normal decimal floating point number of the form
        //
        //   dddd.dddd[eE][+-]ddd
        //
        // where the decimal point and exponent are optional, fill out the
        // variables below. Exponent is appropriate if the significand is
        // treated as an integer, and normalized_exp if the significand
        // is taken to have the decimal point after a single leading
        // non-zero digit.
        //
        // If the value is zero, first_sig_digit is None.

        let mut any_digits = false;
        let mut dec_exp = 0i32;

        // Without leading or trailing zeros, irrespective of the dot.
        let mut first_sig_digit = None;
        let mut last_sig_digit = 0;
        let mut dot = s.len();

        for (p, c) in s.char_indices() {
            if c == '.' {
                if dot != s.len() {
                    return Err(ParseError("String contains multiple dots"));
                }
                dot = p;
            } else if let Some(dec_value) = c.to_digit(10) {
                any_digits = true;

                if dec_value != 0 {
                    if first_sig_digit.is_none() {
                        first_sig_digit = Some(p);
                    }
                    last_sig_digit = p;
                }
            } else if c == 'e' || c == 'E' {
                if !any_digits {
                    return Err(ParseError("Significand has no digits"));
                }

                if dot == s.len() {
                    dot = p;
                }

                let mut chars = s[p + 1..].chars().peekable();

                // Adjust for the given exponent.
                let exp_minus = chars.peek() == Some(&'-');
                if exp_minus || chars.peek() == Some(&'+') {
                    chars.next();
                }

                let mut any_exp_digits = false;
                for c in chars {
                    if let Some(value) = c.to_digit(10) {
                        any_exp_digits = true;
                        dec_exp = dec_exp.saturating_mul(10).saturating_add(value as i32);
                    } else {
                        return Err(ParseError("Invalid character in exponent"));
                    }
                }
                // Treat no exponent as 0 to match binutils
                if !any_exp_digits {
                    assert_eq!(dec_exp, 0);
                }

                dec_exp = dec_exp.negate_if(exp_minus);

                break;
            } else {
                return Err(ParseError("Invalid character in significand"));
            }
        }
        if !any_digits {
            return Err(ParseError("Significand has no digits"));
        }

        // Test if we have a zero number allowing for non-zero exponents.
        let first_sig_digit = match first_sig_digit {
            Some(p) => p,
            None => return Ok(Status::OK.and(Self::ZERO)),
        };

        // Adjust the exponents for any decimal point.
        if dot > last_sig_digit {
            dec_exp = dec_exp.saturating_add((dot - last_sig_digit - 1) as i32);
        } else {
            dec_exp = dec_exp.saturating_sub((last_sig_digit - dot) as i32);
        }
        let significand_digits =
            last_sig_digit - first_sig_digit + 1 - (dot > first_sig_digit && dot < last_sig_digit) as usize;
        let normalized_exp = dec_exp.saturating_add(significand_digits as i32 - 1);

        // Handle the cases where exponents are obviously too large or too
        // small. Writing L for log 10 / log 2, a number d.ddddd*10^dec_exp
        // definitely overflows if
        //
        //       (dec_exp - 1) * L >= MAX_EXP
        //
        // and definitely underflows to zero where
        //
        //       (dec_exp + 1) * L <= MIN_EXP - PRECISION
        //
        // With integer arithmetic the tightest bounds for L are
        //
        //       93/28 < L < 196/59            [ numerator <= 256 ]
        //       42039/12655 < L < 28738/8651  [ numerator <= 65536 ]

        // Check for MAX_EXP.
        if normalized_exp.saturating_sub(1).saturating_mul(42039) >= 12655 * S::MAX_EXP as i32 {
            // Overflow and round.
            return Ok(Self::overflow_result(round));
        }

        // Check for MIN_EXP.
        if normalized_exp.saturating_add(1).saturating_mul(28738) <= 8651 * (S::MIN_EXP as i32 - S::PRECISION as i32) {
            // Underflow to zero and round.
            let r = if round == Round::TowardPositive {
                IeeeFloat::SMALLEST
            } else {
                IeeeFloat::ZERO
            };
            return Ok((Status::UNDERFLOW | Status::INEXACT).and(r));
        }

        // A tight upper bound on number of bits required to hold an
        // N-digit decimal integer is N * 196 / 59. Allocate enough space
        // to hold the full significand, and an extra limb required by
        // tcMultiplyPart.
        let max_limbs = limbs_for_bits(1 + 196 * significand_digits / 59);
        let mut dec_sig = DynPrecisionLimbVec::with_capacity(max_limbs);

        // Convert to binary efficiently - we do almost all multiplication
        // in a Limb. When this would overflow do we do a single
        // bignum multiplication, and then revert again to multiplication
        // in a Limb.
        let mut chars = s[first_sig_digit..last_sig_digit + 1].chars();
        loop {
            let mut val = 0;
            let mut multiplier = 1;

            loop {
                let dec_value = match chars.next() {
                    Some('.') => continue,
                    Some(c) => c.to_digit(10).unwrap(),
                    None => break,
                };

                multiplier *= 10;
                val = val * 10 + dec_value as Limb;

                // The maximum number that can be multiplied by ten with any
                // digit added without overflowing a Limb.
                if multiplier > (!0 - 9) / 10 {
                    break;
                }
            }

            // If we've consumed no digits, we're done.
            if multiplier == 1 {
                break;
            }

            // Multiply out the current limb.
            let mut carry = val;
            for x in &mut dec_sig {
                let [low, mut high] = sig::widening_mul(*x, multiplier);

                // Now add carry.
                let (low, overflow) = low.overflowing_add(carry);
                high += overflow as Limb;

                *x = low;
                carry = high;
            }

            // If we had carry, we need another limb (likely but not guaranteed).
            if carry > 0 {
                dec_sig.push(carry);
            }
        }

        // Calculate pow(5, abs(dec_exp)) into `pow5_full`.
        // The *_calc Vec's are reused scratch space, as an optimization.
        let (pow5_full, mut pow5_calc, mut sig_calc, mut sig_scratch_calc) = {
            let mut power = dec_exp.abs() as usize;

            const FIRST_EIGHT_POWERS: [Limb; 8] = [1, 5, 25, 125, 625, 3125, 15625, 78125];

            let mut p5_scratch = DynPrecisionLimbVec::new();
            let mut p5: DynPrecisionLimbVec = [FIRST_EIGHT_POWERS[4]].into_iter().collect();

            let mut r_scratch = DynPrecisionLimbVec::new();
            let mut r: DynPrecisionLimbVec = [FIRST_EIGHT_POWERS[power & 7]].into_iter().collect();
            power >>= 3;

            while power > 0 {
                // Calculate pow(5,pow(2,n+3)).
                p5_scratch.resize(p5.len() * 2, 0);
                let _: Loss = sig::mul(&mut p5_scratch, &mut 0, &p5, &p5, p5.len() * 2 * LIMB_BITS);
                while p5_scratch.last() == Some(&0) {
                    p5_scratch.pop();
                }
                mem::swap(&mut p5, &mut p5_scratch);

                if power & 1 != 0 {
                    r_scratch.resize(r.len() + p5.len(), 0);
                    let _: Loss = sig::mul(&mut r_scratch, &mut 0, &r, &p5, (r.len() + p5.len()) * LIMB_BITS);
                    while r_scratch.last() == Some(&0) {
                        r_scratch.pop();
                    }
                    mem::swap(&mut r, &mut r_scratch);
                }

                power >>= 1;
            }

            (r, r_scratch, p5, p5_scratch)
        };

        // Attempt dec_sig * 10^dec_exp with increasing precision.
        let mut attempt = 0;
        loop {
            let calc_precision = (LIMB_BITS << attempt) - 1;
            attempt += 1;

            let calc_normal_from_limbs = |sig: &mut DynPrecisionLimbVec, limbs: &[Limb]| -> StatusAnd<ExpInt> {
                sig.resize(limbs_for_bits(calc_precision), 0);
                let (mut loss, mut exp) = sig::from_limbs(sig, limbs, calc_precision);

                // Before rounding normalize the exponent of Category::Normal numbers.
                let mut omsb = sig::omsb(sig);

                assert_ne!(omsb, 0);

                // OMSB is numbered from 1. We want to place it in the integer
                // bit numbered PRECISION if possible, with a compensating change in
                // the exponent.
                let final_exp = exp.saturating_add(omsb as ExpInt - calc_precision as ExpInt);

                // Shifting left is easy as we don't lose precision.
                if final_exp < exp {
                    assert_eq!(loss, Loss::ExactlyZero);

                    let exp_change = (exp - final_exp) as usize;
                    sig::shift_left(sig, &mut exp, exp_change);

                    return Status::OK.and(exp);
                }

                // Shift right and capture any new lost fraction.
                if final_exp > exp {
                    let exp_change = (final_exp - exp) as usize;
                    loss = sig::shift_right(sig, &mut exp, exp_change).combine(loss);

                    // Keep OMSB up-to-date.
                    omsb = omsb.saturating_sub(exp_change);
                }

                assert_eq!(omsb, calc_precision);

                // Now round the number according to round given the lost
                // fraction.

                // As specified in IEEE 754, since we do not trap we do not report
                // underflow for exact results.
                if loss == Loss::ExactlyZero {
                    return Status::OK.and(exp);
                }

                // Increment the significand if we're rounding away from zero.
                if loss == Loss::MoreThanHalf || loss == Loss::ExactlyHalf && sig::get_bit(sig, 0) {
                    // We should never overflow.
                    assert_eq!(sig::increment(sig), 0);
                    omsb = sig::omsb(sig);

                    // Did the significand increment overflow?
                    if omsb == calc_precision + 1 {
                        let _: Loss = sig::shift_right(sig, &mut exp, 1);

                        return Status::INEXACT.and(exp);
                    }
                }

                // The normal case - we were and are not denormal, and any
                // significand increment above didn't overflow.
                Status::INEXACT.and(exp)
            };

            let status;
            let mut exp = unpack!(status=,
                calc_normal_from_limbs(&mut sig_calc, &dec_sig));
            let pow5_status;
            let pow5_exp = unpack!(pow5_status=,
                calc_normal_from_limbs(&mut pow5_calc, &pow5_full));

            // Add dec_exp, as 10^n = 5^n * 2^n.
            exp += dec_exp as ExpInt;

            let mut used_bits = S::PRECISION;
            let mut truncated_bits = calc_precision - used_bits;

            let half_ulp_err1 = (status != Status::OK) as Limb;
            let (calc_loss, half_ulp_err2);
            if dec_exp >= 0 {
                exp += pow5_exp;

                sig_scratch_calc.resize(sig_calc.len() + pow5_calc.len(), 0);
                calc_loss = sig::mul(&mut sig_scratch_calc, &mut exp, &sig_calc, &pow5_calc, calc_precision);
                mem::swap(&mut sig_calc, &mut sig_scratch_calc);

                half_ulp_err2 = (pow5_status != Status::OK) as Limb;
            } else {
                exp -= pow5_exp;

                sig_scratch_calc.resize(sig_calc.len(), 0);
                calc_loss = sig::div(&mut sig_scratch_calc, &mut exp, &mut sig_calc, &mut pow5_calc, calc_precision);
                mem::swap(&mut sig_calc, &mut sig_scratch_calc);

                // Denormal numbers have less precision.
                if exp < S::MIN_EXP {
                    truncated_bits += (S::MIN_EXP - exp) as usize;
                    used_bits = calc_precision.saturating_sub(truncated_bits);
                }
                // Extra half-ulp lost in reciprocal of exponent.
                half_ulp_err2 = 2 * (pow5_status != Status::OK || calc_loss != Loss::ExactlyZero) as Limb;
            }

            // Both sig::mul and sig::div return the
            // result with the integer bit set.
            assert!(sig::get_bit(&sig_calc, calc_precision - 1));

            // The error from the true value, in half-ulps, on multiplying two
            // floating point numbers, which differ from the value they
            // approximate by at most half_ulp_err1 and half_ulp_err2 half-ulps, is strictly less
            // than the returned value.
            //
            // See "How to Read Floating Point Numbers Accurately" by William D Clinger.
            assert!(half_ulp_err1 < 2 || half_ulp_err2 < 2 || (half_ulp_err1 + half_ulp_err2 < 8));

            let inexact = (calc_loss != Loss::ExactlyZero) as Limb;
            let half_ulp_err = if half_ulp_err1 + half_ulp_err2 == 0 {
                inexact * 2 // <= inexact half-ulps.
            } else {
                inexact + 2 * (half_ulp_err1 + half_ulp_err2)
            };

            let ulps_from_boundary = {
                let bits = calc_precision - used_bits - 1;

                let i = bits / LIMB_BITS;
                let limb = sig_calc[i] & (!0 >> (LIMB_BITS - 1 - bits % LIMB_BITS));
                let boundary = match round {
                    Round::NearestTiesToEven | Round::NearestTiesToAway => 1 << (bits % LIMB_BITS),
                    _ => 0,
                };
                if i == 0 {
                    let delta = limb.wrapping_sub(boundary);
                    cmp::min(delta, delta.wrapping_neg())
                } else if limb == boundary {
                    if !sig::is_all_zeros(&sig_calc[1..i]) {
                        !0 // A lot.
                    } else {
                        sig_calc[0]
                    }
                } else if limb == boundary.wrapping_sub(1) {
                    if sig_calc[1..i].iter().any(|&x| x.wrapping_neg() != 1) {
                        !0 // A lot.
                    } else {
                        sig_calc[0].wrapping_neg()
                    }
                } else {
                    !0 // A lot.
                }
            };

            // Are we guaranteed to round correctly if we truncate?
            if ulps_from_boundary.saturating_mul(2) >= half_ulp_err {
                let mut r = IeeeFloat {
                    sig: [0],
                    exp,
                    read_only_category_do_not_mutate: Category::Normal,
                    read_only_sign_do_not_mutate: false,
                    marker: PhantomData,
                };
                sig::extract(&mut r.sig, &sig_calc, used_bits, calc_precision - used_bits);
                // If we extracted less bits above we must adjust our exponent
                // to compensate for the implicit right shift.
                r.exp += (S::PRECISION - used_bits) as ExpInt;
                let loss = Loss::through_truncation(&sig_calc, truncated_bits);
                return Ok(r.normalize(round, loss));
            }
        }
    }
}

impl Loss {
    /// Combine the effect of two lost fractions.
    #[inline]
    fn combine(self, less_significant: Loss) -> Loss {
        let mut more_significant = self;
        if less_significant != Loss::ExactlyZero {
            if more_significant == Loss::ExactlyZero {
                more_significant = Loss::LessThanHalf;
            } else if more_significant == Loss::ExactlyHalf {
                more_significant = Loss::MoreThanHalf;
            }
        }

        more_significant
    }

    /// Return the fraction lost were a bignum truncated losing the least
    /// significant `bits` bits.
    #[inline]
    fn through_truncation(limbs: &[Limb], bits: usize) -> Loss {
        if bits == 0 {
            return Loss::ExactlyZero;
        }

        let half_bit = bits - 1;
        let half_limb = half_bit / LIMB_BITS;
        let (half_limb, rest) = if half_limb < limbs.len() {
            (limbs[half_limb], &limbs[..half_limb])
        } else {
            (0, limbs)
        };
        let half = 1 << (half_bit % LIMB_BITS);
        let has_half = half_limb & half != 0;
        let has_rest = half_limb & (half - 1) != 0 || !sig::is_all_zeros(rest);

        match (has_half, has_rest) {
            (false, false) => Loss::ExactlyZero,
            (false, true) => Loss::LessThanHalf,
            (true, false) => Loss::ExactlyHalf,
            (true, true) => Loss::MoreThanHalf,
        }
    }
}

/// Implementation details of IeeeFloat significands, such as big integer arithmetic.
/// As a rule of thumb, no functions in this module should dynamically allocate.
mod sig {
    use super::{limbs_for_bits, ExpInt, Limb, Loss, LIMB_BITS};
    use core::cmp::Ordering;
    use core::mem;

    #[inline]
    pub(super) fn is_all_zeros(limbs: &[Limb]) -> bool {
        limbs.iter().all(|&l| l == 0)
    }

    /// One, not zero, based LSB. That is, returns 0 for a zeroed significand.
    #[inline]
    pub(super) fn olsb(limbs: &[Limb]) -> usize {
        for i in 0..limbs.len() {
            if limbs[i] != 0 {
                return i * LIMB_BITS + limbs[i].trailing_zeros() as usize + 1;
            }
        }

        0
    }

    /// One, not zero, based MSB. That is, returns 0 for a zeroed significand.
    #[inline]
    pub(super) fn omsb(limbs: &[Limb]) -> usize {
        for i in (0..limbs.len()).rev() {
            if limbs[i] != 0 {
                return (i + 1) * LIMB_BITS - limbs[i].leading_zeros() as usize;
            }
        }

        0
    }

    /// Comparison (unsigned) of two significands.
    #[inline]
    pub(super) fn cmp(a: &[Limb], b: &[Limb]) -> Ordering {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b).rev() {
            match a.cmp(b) {
                Ordering::Equal => {}
                o => return o,
            }
        }

        Ordering::Equal
    }

    /// Extract the given bit.
    #[inline]
    pub(super) fn get_bit(limbs: &[Limb], bit: usize) -> bool {
        limbs[bit / LIMB_BITS] & (1 << (bit % LIMB_BITS)) != 0
    }

    /// Set the given bit.
    #[inline]
    pub(super) fn set_bit(limbs: &mut [Limb], bit: usize) {
        limbs[bit / LIMB_BITS] |= 1 << (bit % LIMB_BITS);
    }

    /// Shift `dst` left `bits` bits, subtract `bits` from its exponent.
    #[inline]
    pub(super) fn shift_left(dst: &mut [Limb], exp: &mut ExpInt, bits: usize) {
        if bits > 0 {
            // Our exponent should not underflow.
            *exp = exp.checked_sub(bits as ExpInt).unwrap();

            // Jump is the inter-limb jump; shift is is intra-limb shift.
            let jump = bits / LIMB_BITS;
            let shift = bits % LIMB_BITS;

            for i in (0..dst.len()).rev() {
                let mut limb;

                if i < jump {
                    limb = 0;
                } else {
                    // dst[i] comes from the two limbs src[i - jump] and, if we have
                    // an intra-limb shift, src[i - jump - 1].
                    limb = dst[i - jump];
                    if shift > 0 {
                        limb <<= shift;
                        if i >= jump + 1 {
                            limb |= dst[i - jump - 1] >> (LIMB_BITS - shift);
                        }
                    }
                }

                dst[i] = limb;
            }
        }
    }

    /// Shift `dst` right `bits` bits noting lost fraction.
    #[inline]
    pub(super) fn shift_right(dst: &mut [Limb], exp: &mut ExpInt, bits: usize) -> Loss {
        let loss = Loss::through_truncation(dst, bits);

        if bits > 0 {
            // Our exponent should not overflow.
            *exp = exp.checked_add(bits as ExpInt).unwrap();

            // Jump is the inter-limb jump; shift is is intra-limb shift.
            let jump = bits / LIMB_BITS;
            let shift = bits % LIMB_BITS;

            // Perform the shift. This leaves the most significant `bits` bits
            // of the result at zero.
            for i in 0..dst.len() {
                let mut limb;

                if i + jump >= dst.len() {
                    limb = 0;
                } else {
                    limb = dst[i + jump];
                    if shift > 0 {
                        limb >>= shift;
                        if i + jump + 1 < dst.len() {
                            limb |= dst[i + jump + 1] << (LIMB_BITS - shift);
                        }
                    }
                }

                dst[i] = limb;
            }
        }

        loss
    }

    /// Copy the bit vector of width `src_bits` from `src`, starting at bit SRC_LSB,
    /// to `dst`, such that the bit SRC_LSB becomes the least significant bit of `dst`.
    /// All high bits above `src_bits` in `dst` are zero-filled.
    #[inline]
    pub(super) fn extract(dst: &mut [Limb], src: &[Limb], src_bits: usize, src_lsb: usize) {
        if src_bits == 0 {
            return;
        }

        let dst_limbs = limbs_for_bits(src_bits);
        assert!(dst_limbs <= dst.len());

        let src = &src[src_lsb / LIMB_BITS..];
        dst[..dst_limbs].copy_from_slice(&src[..dst_limbs]);

        let shift = src_lsb % LIMB_BITS;
        let _: Loss = shift_right(&mut dst[..dst_limbs], &mut 0, shift);

        // We now have (dst_limbs * LIMB_BITS - shift) bits from `src`
        // in `dst`.  If this is less that src_bits, append the rest, else
        // clear the high bits.
        let n = dst_limbs * LIMB_BITS - shift;
        if n < src_bits {
            let mask = (1 << (src_bits - n)) - 1;
            dst[dst_limbs - 1] |= (src[dst_limbs] & mask) << n % LIMB_BITS;
        } else if n > src_bits && src_bits % LIMB_BITS > 0 {
            dst[dst_limbs - 1] &= (1 << (src_bits % LIMB_BITS)) - 1;
        }

        // Clear high limbs.
        for x in &mut dst[dst_limbs..] {
            *x = 0;
        }
    }

    /// We want the most significant PRECISION bits of `src`. There may not
    /// be that many; extract what we can.
    #[inline]
    pub(super) fn from_limbs(dst: &mut [Limb], src: &[Limb], precision: usize) -> (Loss, ExpInt) {
        let omsb = omsb(src);

        if precision <= omsb {
            extract(dst, src, precision, omsb - precision);
            (Loss::through_truncation(src, omsb - precision), omsb as ExpInt - 1)
        } else {
            extract(dst, src, omsb, 0);
            (Loss::ExactlyZero, precision as ExpInt - 1)
        }
    }

    /// For every consecutive chunk of `bits` bits from `limbs`,
    /// going from most significant to the least significant bits,
    /// call `f` to transform those bits and store the result back.
    #[inline]
    pub(super) fn each_chunk<F: FnMut(Limb) -> Limb>(limbs: &mut [Limb], bits: usize, mut f: F) {
        assert_eq!(LIMB_BITS % bits, 0);
        for limb in limbs.iter_mut().rev() {
            let mut r = 0;
            for i in (0..LIMB_BITS / bits).rev() {
                r |= f((*limb >> (i * bits)) & ((1 << bits) - 1)) << (i * bits);
            }
            *limb = r;
        }
    }

    /// Increment in-place, return the carry flag.
    #[inline]
    pub(super) fn increment(dst: &mut [Limb]) -> Limb {
        for x in dst {
            *x = x.wrapping_add(1);
            if *x != 0 {
                return 0;
            }
        }

        1
    }

    /// Decrement in-place, return the borrow flag.
    #[inline]
    pub(super) fn decrement(dst: &mut [Limb]) -> Limb {
        for x in dst {
            *x = x.wrapping_sub(1);
            if *x != !0 {
                return 0;
            }
        }

        1
    }

    /// `a += b + c` where `c` is zero or one. Returns the carry flag.
    #[inline]
    pub(super) fn add(a: &mut [Limb], b: &[Limb], mut c: Limb) -> Limb {
        assert!(c <= 1);

        for (a, &b) in a.iter_mut().zip(b) {
            let (r, overflow) = a.overflowing_add(b);
            let (r, overflow2) = r.overflowing_add(c);
            *a = r;
            c = (overflow | overflow2) as Limb;
        }

        c
    }

    /// `a -= b + c` where `c` is zero or one. Returns the borrow flag.
    #[inline]
    pub(super) fn sub(a: &mut [Limb], b: &[Limb], mut c: Limb) -> Limb {
        assert!(c <= 1);

        for (a, &b) in a.iter_mut().zip(b) {
            let (r, overflow) = a.overflowing_sub(b);
            let (r, overflow2) = r.overflowing_sub(c);
            *a = r;
            c = (overflow | overflow2) as Limb;
        }

        c
    }

    /// `a += b` or `a -= b`. Does not preserve `b`.
    #[inline]
    pub(super) fn add_or_sub(
        a_sig: &mut [Limb],
        a_exp: &mut ExpInt,
        a_sign: &mut bool,
        b_sig: &mut [Limb],
        b_exp: ExpInt,
        b_sign: bool,
    ) -> Loss {
        // Are we bigger exponent-wise than the RHS?
        let bits = *a_exp - b_exp;

        // Determine if the operation on the absolute values is effectively
        // an addition or subtraction.
        // Subtraction is more subtle than one might naively expect.
        if *a_sign ^ b_sign {
            let (mut loss, loss_is_from_b) = if bits == 0 {
                (Loss::ExactlyZero, false)
            } else if bits > 0 {
                shift_left(a_sig, a_exp, 1);
                (shift_right(b_sig, &mut 0, (bits - 1) as usize), true)
            } else {
                shift_left(b_sig, &mut 0, 1);
                (shift_right(a_sig, a_exp, (-bits - 1) as usize), false)
            };

            let invert_loss = |loss| match loss {
                Loss::LessThanHalf => Loss::MoreThanHalf,
                Loss::MoreThanHalf => Loss::LessThanHalf,
                _ => loss,
            };

            // Should we reverse the subtraction.
            match cmp(a_sig, b_sig) {
                Ordering::Less => {
                    let borrow = if loss != Loss::ExactlyZero && !loss_is_from_b {
                        // The loss is being subtracted, borrow from the significand and invert
                        // `loss`.
                        loss = invert_loss(loss);
                        1
                    } else {
                        0
                    };
                    // The code above is intended to ensure that no borrow is necessary.
                    assert_eq!(sub(b_sig, a_sig, borrow), 0);
                    a_sig.copy_from_slice(b_sig);
                    *a_sign = !*a_sign;
                }
                Ordering::Greater => {
                    let borrow = if loss != Loss::ExactlyZero && loss_is_from_b {
                        // The loss is being subtracted, borrow from the significand and invert
                        // `loss`.
                        loss = invert_loss(loss);
                        1
                    } else {
                        0
                    };
                    // The code above is intended to ensure that no borrow is necessary.
                    assert_eq!(sub(a_sig, b_sig, borrow), 0);
                }
                Ordering::Equal => {
                    a_sig.fill(0);
                    if loss != Loss::ExactlyZero && loss_is_from_b {
                        // b is slightly larger due to the loss, flip the sign.
                        *a_sign = !*a_sign;
                    }
                }
            }

            loss
        } else {
            let loss = if bits > 0 {
                shift_right(b_sig, &mut 0, bits as usize)
            } else {
                shift_right(a_sig, a_exp, -bits as usize)
            };
            // We have a guard bit; generating a carry cannot happen.
            assert_eq!(add(a_sig, b_sig, 0), 0);
            loss
        }
    }

    #[test]
    fn test_add_or_sub() {
        #[track_caller]
        fn run_test(
            subtract: bool,
            mut lhs_sign: bool,
            mut lhs_exponent: ExpInt,
            mut lhs_significand: Limb,
            rhs_sign: bool,
            rhs_exponent: ExpInt,
            mut rhs_significand: Limb,
            expected_sign: bool,
            expected_exponent: ExpInt,
            expected_significand: Limb,
            expected_loss: Loss,
        ) {
            let loss = add_or_sub(
                core::array::from_mut(&mut lhs_significand),
                &mut lhs_exponent,
                &mut lhs_sign,
                core::array::from_mut(&mut rhs_significand),
                rhs_exponent,
                rhs_sign ^ subtract,
            );
            assert_eq!(loss, expected_loss);
            assert_eq!(lhs_sign, expected_sign);
            assert_eq!(lhs_exponent, expected_exponent);
            assert_eq!(lhs_significand, expected_significand);
        }

        // Test cases are all combinations of:
        // {equal exponents, LHS larger exponent, RHS larger exponent}
        // {equal significands, LHS larger significand, RHS larger significand}
        // {no loss, loss}

        // Equal exponents (loss cannot occur as their is no shifting)
        run_test(true, false, 1, 0x10, false, 1, 0x5, false, 1, 0xb, Loss::ExactlyZero);
        run_test(false, false, -2, 0x20, true, -2, 0x20, false, -2, 0, Loss::ExactlyZero);
        run_test(false, true, 3, 0x20, false, 3, 0x30, false, 3, 0x10, Loss::ExactlyZero);

        // LHS larger exponent
        // LHS significand greater after shitfing
        run_test(true, false, 7, 0x100, false, 3, 0x100, false, 6, 0x1e0, Loss::ExactlyZero);
        run_test(true, false, 7, 0x100, false, 3, 0x101, false, 6, 0x1df, Loss::MoreThanHalf);
        // Significands equal after shitfing
        run_test(true, false, 7, 0x100, false, 3, 0x1000, false, 6, 0, Loss::ExactlyZero);
        run_test(true, false, 7, 0x100, false, 3, 0x1001, true, 6, 0, Loss::LessThanHalf);
        // RHS significand greater after shitfing
        run_test(true, false, 7, 0x100, false, 3, 0x10000, true, 6, 0x1e00, Loss::ExactlyZero);
        run_test(true, false, 7, 0x100, false, 3, 0x10001, true, 6, 0x1e00, Loss::LessThanHalf);

        // RHS larger exponent
        // RHS significand greater after shitfing
        run_test(true, false, 3, 0x100, false, 7, 0x100, true, 6, 0x1e0, Loss::ExactlyZero);
        run_test(true, false, 3, 0x101, false, 7, 0x100, true, 6, 0x1df, Loss::MoreThanHalf);
        // Significands equal after shitfing
        run_test(true, false, 3, 0x1000, false, 7, 0x100, false, 6, 0, Loss::ExactlyZero);
        run_test(true, false, 3, 0x1001, false, 7, 0x100, false, 6, 0, Loss::LessThanHalf);
        // LHS significand greater after shitfing
        run_test(true, false, 3, 0x10000, false, 7, 0x100, false, 6, 0x1e00, Loss::ExactlyZero);
        run_test(true, false, 3, 0x10001, false, 7, 0x100, false, 6, 0x1e00, Loss::LessThanHalf);
    }

    /// `[low, high] = a * b`.
    ///
    /// This cannot overflow, because
    ///
    /// `(n - 1) * (n - 1) + 2 * (n - 1) == (n - 1) * (n + 1)`
    ///
    /// which is less than n^2.
    #[inline]
    pub(super) fn widening_mul(a: Limb, b: Limb) -> [Limb; 2] {
        let mut wide = [0, 0];

        if a == 0 || b == 0 {
            return wide;
        }

        const HALF_BITS: usize = LIMB_BITS / 2;

        let select = |limb, i| (limb >> (i * HALF_BITS)) & ((1 << HALF_BITS) - 1);
        for i in 0..2 {
            for j in 0..2 {
                let mut x = [select(a, i) * select(b, j), 0];
                shift_left(&mut x, &mut 0, (i + j) * HALF_BITS);
                assert_eq!(add(&mut wide, &x, 0), 0);
            }
        }

        wide
    }

    /// `dst = a * b` (for normal `a` and `b`). Returns the lost fraction.
    #[inline]
    pub(super) fn mul<'a>(
        dst: &mut [Limb],
        exp: &mut ExpInt,
        mut a: &'a [Limb],
        mut b: &'a [Limb],
        precision: usize,
    ) -> Loss {
        // Put the narrower number on the `a` for less loops below.
        if a.len() > b.len() {
            mem::swap(&mut a, &mut b);
        }

        for x in &mut dst[..b.len()] {
            *x = 0;
        }

        for i in 0..a.len() {
            let mut carry = 0;
            for j in 0..b.len() {
                let [low, mut high] = widening_mul(a[i], b[j]);

                // Now add carry.
                let (low, overflow) = low.overflowing_add(carry);
                high += overflow as Limb;

                // And now `dst[i + j]`, and store the new low part there.
                let (low, overflow) = low.overflowing_add(dst[i + j]);
                high += overflow as Limb;

                dst[i + j] = low;
                carry = high;
            }
            dst[i + b.len()] = carry;
        }

        // Assume the operands involved in the multiplication are single-precision
        // FP, and the two multiplicants are:
        //     a = a23 . a22 ... a0 * 2^e1
        //     b = b23 . b22 ... b0 * 2^e2
        // the result of multiplication is:
        //     dst = c48 c47 c46 . c45 ... c0 * 2^(e1+e2)
        // Note that there are three significant bits at the left-hand side of the
        // radix point: two for the multiplication, and an overflow bit for the
        // addition (that will always be zero at this point). Move the radix point
        // toward left by two bits, and adjust exponent accordingly.
        *exp += 2;

        // Convert the result having "2 * precision" significant-bits back to the one
        // having "precision" significant-bits. First, move the radix point from
        // poision "2*precision - 1" to "precision - 1". The exponent need to be
        // adjusted by "2*precision - 1" - "precision - 1" = "precision".
        *exp -= precision as ExpInt + 1;

        // In case MSB resides at the left-hand side of radix point, shift the
        // mantissa right by some amount to make sure the MSB reside right before
        // the radix point (i.e. "MSB . rest-significant-bits").
        //
        // Note that the result is not normalized when "omsb < precision". So, the
        // caller needs to call IeeeFloat::normalize() if normalized value is
        // expected.
        let omsb = omsb(dst);
        if omsb <= precision {
            Loss::ExactlyZero
        } else {
            shift_right(dst, exp, omsb - precision)
        }
    }

    /// `quotient = dividend / divisor`. Returns the lost fraction.
    /// Does not preserve `dividend` or `divisor`.
    #[inline]
    pub(super) fn div(
        quotient: &mut [Limb],
        exp: &mut ExpInt,
        dividend: &mut [Limb],
        divisor: &mut [Limb],
        precision: usize,
    ) -> Loss {
        // Normalize the divisor.
        let bits = precision - omsb(divisor);
        shift_left(divisor, &mut 0, bits);
        *exp += bits as ExpInt;

        // Normalize the dividend.
        let bits = precision - omsb(dividend);
        shift_left(dividend, exp, bits);

        // Division by 1.
        let olsb_divisor = olsb(divisor);
        if olsb_divisor == precision {
            quotient.copy_from_slice(dividend);
            return Loss::ExactlyZero;
        }

        // Ensure the dividend >= divisor initially for the loop below.
        // Incidentally, this means that the division loop below is
        // guaranteed to set the integer bit to one.
        if cmp(dividend, divisor) == Ordering::Less {
            shift_left(dividend, exp, 1);
            assert_ne!(cmp(dividend, divisor), Ordering::Less)
        }

        // Helper for figuring out the lost fraction.
        let lost_fraction = |dividend: &[Limb], divisor: &[Limb]| match cmp(dividend, divisor) {
            Ordering::Greater => Loss::MoreThanHalf,
            Ordering::Equal => Loss::ExactlyHalf,
            Ordering::Less => {
                if is_all_zeros(dividend) {
                    Loss::ExactlyZero
                } else {
                    Loss::LessThanHalf
                }
            }
        };

        // Try to perform a (much faster) short division for small divisors.
        let divisor_bits = precision - (olsb_divisor - 1);
        macro_rules! try_short_div {
            ($W:ty, $H:ty, $half:expr) => {
                if divisor_bits * 2 <= $half {
                    // Extract the small divisor.
                    let _: Loss = shift_right(divisor, &mut 0, olsb_divisor - 1);
                    let divisor = divisor[0] as $H as $W;

                    // Shift the dividend to produce a quotient with the unit bit set.
                    let top_limb = *dividend.last().unwrap();
                    let mut rem = (top_limb >> (LIMB_BITS - (divisor_bits - 1))) as $H;
                    shift_left(dividend, &mut 0, divisor_bits - 1);

                    // Apply short division in place on $H (of $half bits) chunks.
                    each_chunk(dividend, $half, |chunk| {
                        let chunk = chunk as $H;
                        let combined = ((rem as $W) << $half) | (chunk as $W);
                        rem = (combined % divisor) as $H;
                        (combined / divisor) as $H as Limb
                    });
                    quotient.copy_from_slice(dividend);

                    return lost_fraction(&[(rem as Limb) << 1], &[divisor as Limb]);
                }
            };
        }

        try_short_div!(u32, u16, 16);
        try_short_div!(u64, u32, 32);
        try_short_div!(u128, u64, 64);

        // Zero the quotient before setting bits in it.
        for x in &mut quotient[..limbs_for_bits(precision)] {
            *x = 0;
        }

        // Long division.
        for bit in (0..precision).rev() {
            if cmp(dividend, divisor) != Ordering::Less {
                sub(dividend, divisor, 0);
                set_bit(quotient, bit);
            }
            shift_left(dividend, &mut 0, 1);
        }

        lost_fraction(dividend, divisor)
    }
}
