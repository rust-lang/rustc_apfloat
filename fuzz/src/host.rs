use std::fmt;

use rustc_apfloat::{Round, Status, StatusAnd};

/// Abstraction over host float operations. If the requested rounding mode is not supported,
/// return `None`.
pub trait HostFloat: Copy + Sized + fmt::Debug {
    type UInt: Copy + fmt::LowerHex;
    fn from_bits(bits: Self::UInt) -> Self;
    fn to_bits(self) -> Self::UInt;
    fn neg(self) -> Self;
    fn add_r(self, other: Self, rm: Round) -> Option<StatusAnd<Self>>;
    fn sub_r(self, other: Self, rm: Round) -> Option<StatusAnd<Self>>;
    fn mul_r(self, other: Self, rm: Round) -> Option<StatusAnd<Self>>;
    fn div_r(self, other: Self, rm: Round) -> Option<StatusAnd<Self>>;
    fn rem(self, other: Self) -> Self;
    fn mul_add_r(self, mul: Self, add: Self, rm: Round) -> Option<StatusAnd<Self>>;
    fn to_i128_r(self, rm: Round) -> Option<StatusAnd<i128>>;
    fn to_u128_r(self, rm: Round) -> Option<StatusAnd<u128>>;
    fn from_i128_r(x: i128, rm: Round) -> Option<StatusAnd<Self>>;
    fn from_u128_r(x: u128, rm: Round) -> Option<StatusAnd<Self>>;
    fn to_double_r(self, rm: Round) -> Option<StatusAnd<f64>>;
    fn from_double_r(x: f64, rm: Round) -> Option<StatusAnd<Self>>;
    fn to_single_r(self, rm: Round) -> Option<StatusAnd<f32>>;
    fn from_single_r(x: f32, rm: Round) -> Option<StatusAnd<Self>>;
}

macro_rules! impl_host_float {
    ($ty:ty, $ity:ty) => {
        impl HostFloat for $ty {
            type UInt = $ity;
            fn from_bits(bits: Self::UInt) -> Self {
                Self::from_bits(bits)
            }
            fn to_bits(self) -> Self::UInt {
                self.to_bits()
            }
            fn neg(self) -> Self {
                -self
            }
            fn add_r(self, other: Self, rm: Round) -> Option<StatusAnd<Self>> {
                no_fp_env(rm, || self + other)
            }
            fn sub_r(self, other: Self, rm: Round) -> Option<StatusAnd<Self>> {
                no_fp_env(rm, || self - other)
            }
            fn mul_r(self, other: Self, rm: Round) -> Option<StatusAnd<Self>> {
                no_fp_env(rm, || self * other)
            }
            fn div_r(self, other: Self, rm: Round) -> Option<StatusAnd<Self>> {
                no_fp_env(rm, || self / other)
            }
            fn rem(self, other: Self) -> Self {
                self % other
            }
            fn mul_add_r(self, mul: Self, add: Self, rm: Round) -> Option<StatusAnd<Self>> {
                no_fp_env(rm, || self.mul_add(mul, add))
            }

            /* float->int casts are toward zero */
            fn to_i128_r(self, rm: Round) -> Option<StatusAnd<i128>> {
                no_fp_env_toward_zero(rm, || self as i128)
            }
            fn to_u128_r(self, rm: Round) -> Option<StatusAnd<u128>> {
                no_fp_env_toward_zero(rm, || self as u128)
            }

            fn from_i128_r(x: i128, rm: Round) -> Option<StatusAnd<Self>> {
                no_fp_env(rm, || x as Self)
            }
            fn from_u128_r(x: u128, rm: Round) -> Option<StatusAnd<Self>> {
                no_fp_env(rm, || x as Self)
            }
            fn to_double_r(self, rm: Round) -> Option<StatusAnd<f64>> {
                no_fp_env(rm, || self as f64)
            }
            fn from_double_r(x: f64, rm: Round) -> Option<StatusAnd<Self>> {
                no_fp_env(rm, || x as Self)
            }
            fn to_single_r(self, rm: Round) -> Option<StatusAnd<f32>> {
                no_fp_env(rm, || self as f32)
            }
            fn from_single_r(x: f32, rm: Round) -> Option<StatusAnd<Self>> {
                no_fp_env(rm, || x as Self)
            }
        }
    };
}

/// Run `f` and turn it into OK status if the rounding mode is nearest-even, `None` otherwise.
fn no_fp_env<T>(rm: Round, f: impl Fn() -> T) -> Option<StatusAnd<T>> {
    match rm {
        Round::NearestTiesToEven => Some(Status::OK.and(f())),
        _ => None,
    }
}

/// The same as [`no_fp_env`] but checks for rounding toward zero.
fn no_fp_env_toward_zero<T>(rm: Round, f: impl Fn() -> T) -> Option<StatusAnd<T>> {
    match rm {
        Round::TowardZero => Some(Status::OK.and(f())),
        _ => None,
    }
}

#[cfg(not(x86_sse2))]
#[cfg(target_has_reliable_f16)]
impl_host_float!(f16, u16);
#[cfg(not(x86_sse2))]
impl_host_float!(f32, u32);
#[cfg(not(x86_sse2))]
impl_host_float!(f64, u64);
#[cfg(target_has_reliable_f128)]
impl_host_float!(f128, u128);

/// Assembly implementations on x86 which respect rounding mode.
#[cfg(x86_sse2)]
#[cfg(target_has_reliable_f16)]
mod x86 {
    use super::*;

    /// Given a rounding mode, assembly operation, and assembly configuration,
    macro_rules! with_fp_env {
        (
            $rm:ident,
            $op:literal,
            $( $name:ident = $dir:ident($kind:ident) $val:expr $(=> $dst:ident)?, )+
        ) => {{
            let mut csr_stash = 0u32;
            let mut csr = make_mxcsr_cw($rm)?;

            core::arch::asm!(
                // stash the current control state
                "stmxcsr [{csr_stash}]",
                // set the control state we want, clears flags
                "ldmxcsr [{csr}]",
                $op,
                // get the new control state
                "stmxcsr [{csr}]",
                // restore the original control state
                "ldmxcsr [{csr_stash}]",
                csr_stash = in(reg) &mut csr_stash,
                csr = in(reg) &mut csr,
                $( $name = $dir($kind) $val $(=> $dst)?, )+
                options(nostack),
            );

            check_exceptions(csr)
        }}
    }

    impl HostFloat for f16 {
        type UInt = u16;

        fn from_bits(bits: Self::UInt) -> Self {
            Self::from_bits(bits)
        }

        fn to_bits(self) -> Self::UInt {
            self.to_bits()
        }

        fn neg(self) -> Self {
            -self
        }

        fn add_r(mut self, other: Self, rm: Round) -> Option<StatusAnd<Self>> {
            if is_x86_feature_detected!("avx512fp16") {
                unsafe {
                    let status = with_fp_env!(
                        rm,
                        "addsh {x}, {y}",
                        x = inout(xmm_reg) self,
                        y = in(xmm_reg) other,
                    );
                    Some(status.and(self))
                }
            } else {
                no_fp_env(rm, || self + other)
            }
        }

        fn sub_r(mut self, other: Self, rm: Round) -> Option<StatusAnd<Self>> {
            if is_x86_feature_detected!("avx512fp16") {
                unsafe {
                    let status = with_fp_env!(
                        rm,
                        "subsh {x}, {y}",
                        x = inout(xmm_reg) self,
                        y = in(xmm_reg) other,
                    );
                    Some(status.and(self))
                }
            } else {
                no_fp_env(rm, || self - other)
            }
        }

        fn mul_r(mut self, other: Self, rm: Round) -> Option<StatusAnd<Self>> {
            if is_x86_feature_detected!("avx512fp16") {
                unsafe {
                    let status = with_fp_env!(
                        rm,
                        "mulsh {x}, {y}",
                        x = inout(xmm_reg) self,
                        y = in(xmm_reg) other,
                    );
                    Some(status.and(self))
                }
            } else {
                no_fp_env(rm, || self * other)
            }
        }

        fn div_r(mut self, other: Self, rm: Round) -> Option<StatusAnd<Self>> {
            if is_x86_feature_detected!("avx512fp16") {
                unsafe {
                    let status = with_fp_env!(
                        rm,
                        "divsh {x}, {y}",
                        x = inout(xmm_reg) self,
                        y = in(xmm_reg) other,
                    );
                    Some(status.and(self))
                }
            } else {
                no_fp_env(rm, || self / other)
            }
        }

        fn rem(self, other: Self) -> Self {
            self % other
        }

        fn mul_add_r(mut self, mul: Self, add: Self, rm: Round) -> Option<StatusAnd<Self>> {
            if is_x86_feature_detected!("avx512fp16") {
                unsafe {
                    let status = with_fp_env!(
                        rm,
                        "vfmadd213ss {x}, {y}, {z}",
                        x = inout(xmm_reg) self,
                        y = in(xmm_reg) mul,
                        z = in(xmm_reg) add,
                    );
                    Some(status.and(self))
                }
            } else {
                no_fp_env(rm, || self.mul_add(mul, add))
            }
        }

        fn to_i128_r(self, rm: Round) -> Option<StatusAnd<i128>> {
            no_fp_env_toward_zero(rm, || self as i128)
        }

        fn to_u128_r(self, rm: Round) -> Option<StatusAnd<u128>> {
            no_fp_env_toward_zero(rm, || self as u128)
        }

        fn from_i128_r(x: i128, rm: Round) -> Option<StatusAnd<Self>> {
            no_fp_env(rm, || x as Self)
        }

        fn from_u128_r(x: u128, rm: Round) -> Option<StatusAnd<Self>> {
            no_fp_env(rm, || x as Self)
        }

        fn to_double_r(self, rm: Round) -> Option<StatusAnd<f64>> {
            if is_x86_feature_detected!("avx512fp16") {
                unsafe {
                    let dst: f64;
                    let status = with_fp_env!(
                        rm,
                        "vcvtsh2sd {y}, {x}",
                        x = in(xmm_reg) self,
                        y = out(xmm_reg) dst,
                    );
                    Some(status.and(dst))
                }
            } else {
                no_fp_env(rm, || self as f64)
            }
        }

        fn from_double_r(x: f64, rm: Round) -> Option<StatusAnd<Self>> {
            if is_x86_feature_detected!("avx512fp16") {
                unsafe {
                    let dst: f16;
                    let status = with_fp_env!(
                        rm,
                        "vcvtsd2sh {y}, {x}",
                        x = in(xmm_reg) x,
                        y = out(xmm_reg) dst,
                    );
                    Some(status.and(dst))
                }
            } else {
                no_fp_env(rm, || x as Self)
            }
        }

        fn to_single_r(self, rm: Round) -> Option<StatusAnd<f32>> {
            if is_x86_feature_detected!("avx512fp16") {
                unsafe {
                    let dst: f32;
                    let status = with_fp_env!(
                        rm,
                        "vcvtsh2ss {y}, {x}",
                        x = in(xmm_reg) self,
                        y = out(xmm_reg) dst,
                    );
                    Some(status.and(dst))
                }
            } else {
                no_fp_env(rm, || self as f32)
            }
        }

        fn from_single_r(x: f32, rm: Round) -> Option<StatusAnd<Self>> {
            if is_x86_feature_detected!("avx512fp16") {
                unsafe {
                    let dst: f16;
                    let status = with_fp_env!(
                        rm,
                        "vcvtss2sh {y}, {x}",
                        x = in(xmm_reg) x,
                        y = out(xmm_reg) dst,
                    );
                    Some(status.and(dst))
                }
            } else {
                no_fp_env(rm, || x as Self)
            }
        }
    }

    impl HostFloat for f32 {
        type UInt = u32;

        fn from_bits(bits: Self::UInt) -> Self {
            Self::from_bits(bits)
        }

        fn to_bits(self) -> Self::UInt {
            self.to_bits()
        }

        fn neg(self) -> Self {
            -self
        }

        fn add_r(mut self, other: Self, rm: Round) -> Option<StatusAnd<Self>> {
            unsafe {
                let status = with_fp_env!(
                    rm,
                    "addss {x}, {y}",
                    x = inout(xmm_reg) self,
                    y = in(xmm_reg) other,
                );
                Some(status.and(self))
            }
        }

        fn sub_r(mut self, other: Self, rm: Round) -> Option<StatusAnd<Self>> {
            unsafe {
                let status = with_fp_env!(
                    rm,
                    "subss {x}, {y}",
                    x = inout(xmm_reg) self,
                    y = in(xmm_reg) other,
                );
                Some(status.and(self))
            }
        }

        fn mul_r(mut self, other: Self, rm: Round) -> Option<StatusAnd<Self>> {
            unsafe {
                let status = with_fp_env!(
                    rm,
                    "mulss {x}, {y}",
                    x = inout(xmm_reg) self,
                    y = in(xmm_reg) other,
                );
                Some(status.and(self))
            }
        }

        fn div_r(mut self, other: Self, rm: Round) -> Option<StatusAnd<Self>> {
            unsafe {
                let status = with_fp_env!(
                    rm,
                    "divss {x}, {y}",
                    x = inout(xmm_reg) self,
                    y = in(xmm_reg) other,
                );
                Some(status.and(self))
            }
        }

        fn rem(self, other: Self) -> Self {
            self % other
        }

        fn mul_add_r(mut self, mul: Self, add: Self, rm: Round) -> Option<StatusAnd<Self>> {
            if is_x86_feature_detected!("fma") {
                unsafe {
                    let status = with_fp_env!(
                        rm,
                        "vfmadd213ss {x}, {y}, {z}",
                        x = inout(xmm_reg) self,
                        y = in(xmm_reg) mul,
                        z = in(xmm_reg) add,
                    );
                    Some(status.and(self))
                }
            } else {
                no_fp_env(rm, || self.mul_add(mul, add))
            }
        }

        fn to_i128_r(self, rm: Round) -> Option<StatusAnd<i128>> {
            no_fp_env_toward_zero(rm, || self as i128)
        }

        fn to_u128_r(self, rm: Round) -> Option<StatusAnd<u128>> {
            no_fp_env_toward_zero(rm, || self as u128)
        }

        fn from_i128_r(x: i128, rm: Round) -> Option<StatusAnd<Self>> {
            no_fp_env(rm, || x as Self)
        }

        fn from_u128_r(x: u128, rm: Round) -> Option<StatusAnd<Self>> {
            no_fp_env(rm, || x as Self)
        }

        fn to_double_r(self, rm: Round) -> Option<StatusAnd<f64>> {
            unsafe {
                let dst: f64;
                let status = with_fp_env!(
                    rm,
                    "cvtss2sd {y}, {x}",
                    x = in(xmm_reg) self,
                    y = out(xmm_reg) dst,
                );
                Some(status.and(dst))
            }
        }

        fn from_double_r(x: f64, rm: Round) -> Option<StatusAnd<Self>> {
            unsafe {
                let dst: f32;
                let status = with_fp_env!(
                    rm,
                    "cvtsd2ss {y}, {x}",
                    x = in(xmm_reg) x,
                    y = out(xmm_reg) dst,
                );
                Some(status.and(dst))
            }
        }

        fn to_single_r(self, _rm: Round) -> Option<StatusAnd<f32>> {
            Some(Status::OK.and(self))
        }

        fn from_single_r(x: f32, _rm: Round) -> Option<StatusAnd<Self>> {
            Some(Status::OK.and(x))
        }
    }

    impl HostFloat for f64 {
        type UInt = u64;

        fn from_bits(bits: Self::UInt) -> Self {
            Self::from_bits(bits)
        }

        fn to_bits(self) -> Self::UInt {
            self.to_bits()
        }

        fn neg(self) -> Self {
            -self
        }

        fn add_r(mut self, other: Self, rm: Round) -> Option<StatusAnd<Self>> {
            unsafe {
                let status = with_fp_env!(
                    rm,
                    "addsd {x}, {y}",
                    x = inout(xmm_reg) self,
                    y = in(xmm_reg) other,
                );
                Some(status.and(self))
            }
        }

        fn sub_r(mut self, other: Self, rm: Round) -> Option<StatusAnd<Self>> {
            unsafe {
                let status = with_fp_env!(
                    rm,
                    "subsd {x}, {y}",
                    x = inout(xmm_reg) self,
                    y = in(xmm_reg) other,
                );
                Some(status.and(self))
            }
        }

        fn mul_r(mut self, other: Self, rm: Round) -> Option<StatusAnd<Self>> {
            unsafe {
                let status = with_fp_env!(
                    rm,
                    "mulsd {x}, {y}",
                    x = inout(xmm_reg) self,
                    y = in(xmm_reg) other,
                );
                Some(status.and(self))
            }
        }

        fn div_r(mut self, other: Self, rm: Round) -> Option<StatusAnd<Self>> {
            unsafe {
                let status = with_fp_env!(
                    rm,
                    "divsd {x}, {y}",
                    x = inout(xmm_reg) self,
                    y = in(xmm_reg) other,
                );
                Some(status.and(self))
            }
        }

        fn rem(self, other: Self) -> Self {
            self % other
        }

        fn mul_add_r(mut self, mul: Self, add: Self, rm: Round) -> Option<StatusAnd<Self>> {
            if is_x86_feature_detected!("fma") {
                unsafe {
                    let status = with_fp_env!(
                        rm,
                        "vfmadd213sd {x}, {y}, {z}",
                        x = inout(xmm_reg) self,
                        y = in(xmm_reg) mul,
                        z = in(xmm_reg) add,
                    );
                    Some(status.and(self))
                }
            } else {
                no_fp_env(rm, || self.mul_add(mul, add))
            }
        }

        fn to_i128_r(self, rm: Round) -> Option<StatusAnd<i128>> {
            no_fp_env_toward_zero(rm, || self as i128)
        }

        fn to_u128_r(self, rm: Round) -> Option<StatusAnd<u128>> {
            no_fp_env_toward_zero(rm, || self as u128)
        }

        fn from_i128_r(x: i128, rm: Round) -> Option<StatusAnd<Self>> {
            no_fp_env(rm, || x as Self)
        }

        fn from_u128_r(x: u128, rm: Round) -> Option<StatusAnd<Self>> {
            no_fp_env(rm, || x as Self)
        }

        fn to_double_r(self, _rm: Round) -> Option<StatusAnd<f64>> {
            Some(Status::OK.and(self))
        }

        fn from_double_r(x: f64, _rm: Round) -> Option<StatusAnd<Self>> {
            Some(Status::OK.and(x))
        }

        fn to_single_r(self, rm: Round) -> Option<StatusAnd<f32>> {
            unsafe {
                let dst: f32;
                let status = with_fp_env!(
                    rm,
                    "cvtsd2ss {y}, {x}",
                    x = in(xmm_reg) self,
                    y = out(xmm_reg) dst,
                );
                Some(status.and(dst))
            }
        }

        fn from_single_r(x: f32, rm: Round) -> Option<StatusAnd<Self>> {
            unsafe {
                let dst: f64;
                let status = with_fp_env!(
                    rm,
                    "cvtss2sd {y}, {x}",
                    x = in(xmm_reg) x,
                    y = out(xmm_reg) dst,
                );
                Some(status.and(dst))
            }
        }
    }

    /// Make a control word or return `None` if the rounding mode is not supported.
    fn make_mxcsr_cw(round: Round) -> Option<u32> {
        // Default: Clear exception flags, no DAZ, no FTZ
        let mut csr = 0u32;
        // Set all masks so fp status doesn't turn into SIGFPE
        csr |= 0b111111 << 7;

        let rc = match round {
            Round::NearestTiesToEven => 0b00,
            Round::TowardPositive => 0b10,
            Round::TowardNegative => 0b01,
            Round::TowardZero => 0b11,
            Round::NearestTiesToAway => return None,
        };

        csr |= rc << 13;
        Some(csr)
    }

    fn check_exceptions(csr: u32) -> Status {
        let mut status = Status::OK;

        if csr & (1 << 0) != 0 {
            status |= Status::INVALID_OP;
        }
        if csr & (1 << 1) != 0 {
            // denormal flag, not part of status
        }
        if csr & (1 << 2) != 0 {
            status |= Status::DIV_BY_ZERO;
        }
        if csr & (1 << 3) != 0 {
            status |= Status::OVERFLOW;
        }
        if csr & (1 << 4) != 0 {
            status |= Status::UNDERFLOW;
        }
        if csr & (1 << 5) != 0 {
            status |= Status::INEXACT;
        }

        status
    }
}
