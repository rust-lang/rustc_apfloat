use crate::{
    ieee::{IeeeDefaultExceptionHandling, IeeeFloat, Semantics},
    Category, Float, Round, Status, StatusAnd,
};

impl<S: Semantics> IeeeFloat<S> {
    /// This is a spec conformant implementation of the IEEE Float sqrt function
    /// This is put in downstream.rs because this function hasn't been implemented in the upstream C++ version yet.
    pub(crate) fn ieee_sqrt(self, round: Round) -> StatusAnd<Self> {
        match self.category() {
            // preserve zero sign
            Category::Zero => return Status::OK.and(self),
            // propagate NaN
            // If the input is a signalling NaN, then IEEE 754 requires the result to be converted to a quiet NaN.
            // On most CPUs that means the most significant bit of the significand field is 0 for signalling NaNs and 1 for quiet NaNs.
            // On most CPUs they quiet a NaN by setting that bit to a 1, RISC-V instead returns the canonical NaN with positive sign,
            // the most significant significand bit set and all other significand bits cleared.
            // However, Rust and LLVM allow input NaNs to be returned unmodified as well as a few other options -- see Rust's rules for NaNs.
            // https://doc.rust-lang.org/std/primitive.f32.html#nan-bit-patterns
            // (Thanks @programmerjake for the comment)
            Category::NaN => return IeeeDefaultExceptionHandling::result_from_nan(self),
            // sqrt of negative number is NaN
            _ if self.is_negative() => return Status::INVALID_OP.and(Self::NAN),
            // sqrt(inf) = inf
            Category::Infinity => return Status::OK.and(Self::INFINITY),
            Category::Normal => (),
        }

        // Floating point precision, excluding the integer bit.
        let prec = i32::try_from(Self::PRECISION).unwrap() - 1;

        // x = 2^(exp - prec) * mant
        // where mant is an integer with prec+1 bits.
        // mant is a u128, which is large enough for the largest prec (112 for f128).
        let mut exp = self.ilogb();
        let mut mant = self.scalbn(prec - exp).to_u128(128).value;

        if exp % 2 != 0 {
            // Make exponent even, so it can be divided by 2.
            exp -= 1;
            mant <<= 1;
        }

        // Bit-by-bit (base-2 digit-by-digit) sqrt of mant.
        // mant is treated here as a fixed point number with prec fractional bits.
        // mant will be shifted left by one bit to have an extra fractional bit, which
        // will be used to determine the rounding direction.

        // res is the truncated sqrt of mant, where one bit is added at each iteration.
        let mut res = 0u128;
        // rem is the remainder with the current res
        // rem_i = 2^i * ((mant<<1) - res_i^2)
        // starting with res = 0, rem = mant<<1
        let mut rem = mant << 1;
        // s_i = 2*res_i
        let mut s = 0u128;
        // d is used to iterate over bits, from high to low (d_i = 2^(-i))
        let mut d = 1u128 << (prec + 1);

        // For iteration j=i+1, we need to find largest b_j = 0 or 1 such that
        //  (res_i + b_j * 2^(-j))^2 <= mant<<1
        // Expanding (a + b)^2 = a^2 + b^2 + 2*a*b:
        //  res_i^2 + (b_j * 2^(-j))^2 + 2 * res_i * b_j * 2^(-j) <= mant<<1
        // And rearranging the terms:
        //  b_j^2 * 2^(-j) + 2 * res_i * b_j <= 2^j * (mant<<1 - res_i^2)
        //  b_j^2 * 2^(-j) + 2 * res_i * b_j <= rem_i

        while d != 0 {
            // Probe b_j^2 * 2^(-j) + 2 * res_i * b_j <= rem_i with b_j = 1:
            // t = 2*res_i + 2^(-j)
            let t = s + d;
            if rem >= t {
                // b_j should be 1, so make res_j = res_i + 2^(-j) and adjust rem
                res += d;
                s += d + d;
                rem -= t;
            }
            // Adjust rem for next iteration
            rem <<= 1;
            // Shift iterator
            d >>= 1;
        }

        let mut status = Status::OK;

        // A nonzero remainder indicates that we could continue processing sqrt if we had
        // more precision, potentially indefinitely. We don't because we have enough bits
        // to fill our significand already, and only need the one extra bit to determine
        // rounding.
        if rem != 0 {
            status = Status::INEXACT;

            match round {
                // If the LSB is 0, we should round down and this 1 gets cut off. If the LSB
                // is 1, it is either a tie (if all remaining bits would be 0) or something
                // that should be rounded up.
                //
                // Square roots are either exact or irrational, so a `1` in the extra bit
                // already implies an irrational result with more `1`s in the infinite
                // precision tail that should be rounded up, which this does. We are in a
                // `rem != 0` block but could technically add the `1` unconditionally, given
                // that a 0 in the extra bit would imply an exact result to be rounded down
                // (and the extra bit is just shifted out).
                Round::NearestTiesToEven => res += 1,
                // We know we have an inexact result that needs rounding up. If the round
                // bit is 1, adding 1 is sufficient and adding 2 does nothing extra (the
                // new LSB will get truncated). If the round bit is 0, we need to add
                // two anyway to affect the significand.
                Round::TowardPositive => res += 2,
                // By default, shifting will round down.
                Round::TowardNegative => (),
                // Same as negative since the result of sqrt is positive.
                Round::TowardZero => (),
                Round::NearestTiesToAway => unimplemented!("unsupported rounding mode"),
            };
        }

        // Remove the extra fractional bit.
        res >>= 1;

        // Build resulting value with res as mantissa and exp/2 as exponent
        status.and(Self::from_u128(res).value.scalbn(exp / 2 - prec))
    }
}
