//! Fuzzing "ops" are a small set of floating-point operations (available both
//! natively and via `llvm::APFloat`/`rustc_apfloat`), represented in code as a
//! generic `FuzzOp` `enum` (with each variant also carrying that op's inputs),
//! with a straight-forward binary serialization (for fuzzing to operate on),
//! and a defined ABI (which Rust code can use to call into the C++ wrapper).

#[derive(Copy, Clone, Debug)]
#[repr(u8)]
pub enum FuzzOp<T> {
    Neg(T) = 0,
    Add(T, T) = 1,
    Sub(T, T) = 2,
    Mul(T, T) = 3,
    Div(T, T) = 4,
    Rem(T, T) = 5,
    MulAdd(T, T, T) = 6,
    FToI128ToF(T) = 7,
    FToU128ToF(T) = 8,
    FToSingleToF(T) = 9,
    FToDoubleToF(T) = 10,
}

impl FuzzOp<()> {
    pub fn from_tag(tag: u8) -> Option<Self> {
        Some(match tag {
            0 => FuzzOp::Neg(()),
            1 => FuzzOp::Add((), ()),
            2 => FuzzOp::Sub((), ()),
            3 => FuzzOp::Mul((), ()),
            4 => FuzzOp::Div((), ()),
            5 => FuzzOp::Rem((), ()),
            6 => FuzzOp::MulAdd((), (), ()),
            7 => FuzzOp::FToI128ToF(()),
            8 => FuzzOp::FToU128ToF(()),
            9 => FuzzOp::FToSingleToF(()),
            10 => FuzzOp::FToDoubleToF(()),
            _ => return None,
        })
    }
}

impl<T> FuzzOp<T> {
    pub fn tag(self) -> u8 {
        match self {
            FuzzOp::Neg(..) => 0,
            FuzzOp::Add(..) => 1,
            FuzzOp::Sub(..) => 2,
            FuzzOp::Mul(..) => 3,
            FuzzOp::Div(..) => 4,
            FuzzOp::Rem(..) => 5,
            FuzzOp::MulAdd(..) => 6,
            FuzzOp::FToI128ToF(..) => 7,
            FuzzOp::FToU128ToF(..) => 8,
            FuzzOp::FToSingleToF(..) => 9,
            FuzzOp::FToDoubleToF(..) => 10,
        }
    }

    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> FuzzOp<U> {
        match self {
            FuzzOp::Neg(a) => FuzzOp::Neg(f(a)),
            FuzzOp::Add(a, b) => FuzzOp::Add(f(a), f(b)),
            FuzzOp::Sub(a, b) => FuzzOp::Sub(f(a), f(b)),
            FuzzOp::Mul(a, b) => FuzzOp::Mul(f(a), f(b)),
            FuzzOp::Div(a, b) => FuzzOp::Div(f(a), f(b)),
            FuzzOp::Rem(a, b) => FuzzOp::Rem(f(a), f(b)),
            FuzzOp::MulAdd(a, b, c) => FuzzOp::MulAdd(f(a), f(b), f(c)),
            FuzzOp::FToI128ToF(a) => FuzzOp::FToI128ToF(f(a)),
            FuzzOp::FToU128ToF(a) => FuzzOp::FToU128ToF(f(a)),
            FuzzOp::FToSingleToF(a) => FuzzOp::FToSingleToF(f(a)),
            FuzzOp::FToDoubleToF(a) => FuzzOp::FToDoubleToF(f(a)),
        }
    }
}

/// A testable operation, which can be encoded as a byte.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Op {
    Neg = 0,
    Add = 1,
    Sub = 2,
    Mul = 3,
    Div = 4,
    Rem = 5,
    MulAdd = 6,
    FToI128ToF = 7,
    FToU128ToF = 8,
    FToSingleToF = 9,
    FToDoubleToF = 10,
}

impl Op {
    pub fn from_u8(tag: u8) -> Option<Self> {
        let v = match tag {
            x if x == Self::Neg.to_u8() => Self::Neg,
            x if x == Self::Add.to_u8() => Self::Add,
            x if x == Self::Sub.to_u8() => Self::Sub,
            x if x == Self::Mul.to_u8() => Self::Mul,
            x if x == Self::Div.to_u8() => Self::Div,
            x if x == Self::Rem.to_u8() => Self::Rem,
            x if x == Self::MulAdd.to_u8() => Self::MulAdd,
            x if x == Self::FToI128ToF.to_u8() => Self::FToI128ToF,
            x if x == Self::FToU128ToF.to_u8() => Self::FToU128ToF,
            x if x == Self::FToSingleToF.to_u8() => Self::FToSingleToF,
            x if x == Self::FToDoubleToF.to_u8() => Self::FToDoubleToF,
            _ => return None,
        };
        Some(v)
    }

    pub fn to_u8(self) -> u8 {
        self as u8
    }

    pub fn airity(self) -> Arity {
        match self {
            Op::Neg => Arity::Unary,
            Op::Add => Arity::Binary,
            Op::Sub => Arity::Binary,
            Op::Mul => Arity::Binary,
            Op::Div => Arity::Binary,
            Op::Rem => Arity::Binary,
            Op::MulAdd => Arity::Ternary,
            Op::FToI128ToF => Arity::Unary,
            Op::FToU128ToF => Arity::Unary,
            Op::FToSingleToF => Arity::Unary,
            Op::FToDoubleToF => Arity::Unary,
        }
    }
}

/// Number of inputs to an operation.
#[derive(Copy, Clone, Debug)]
pub enum Arity {
    Unary = 1,
    Binary = 2,
    Ternary = 3,
}

impl<HF> FuzzOp<HF>
where
    HF: num_traits::Float
        + num_traits::AsPrimitive<i128>
        + num_traits::AsPrimitive<u128>
        + num_traits::AsPrimitive<f32>
        + num_traits::AsPrimitive<f64>,
    i128: num_traits::AsPrimitive<HF>,
    u128: num_traits::AsPrimitive<HF>,
    f32: num_traits::AsPrimitive<HF>,
    f64: num_traits::AsPrimitive<HF>,
{
    pub fn eval_hard(self) -> HF {
        match self {
            FuzzOp::Neg(a) => -a,
            FuzzOp::Add(a, b) => a + b,
            FuzzOp::Sub(a, b) => a - b,
            FuzzOp::Mul(a, b) => a * b,
            FuzzOp::Div(a, b) => a / b,
            FuzzOp::Rem(a, b) => a % b,
            FuzzOp::MulAdd(a, b, c) => a.mul_add(b, c),
            FuzzOp::FToI128ToF(a) => <i128 as num_traits::AsPrimitive<HF>>::as_(
                <HF as num_traits::AsPrimitive<i128>>::as_(a),
            ),
            FuzzOp::FToU128ToF(a) => <u128 as num_traits::AsPrimitive<HF>>::as_(
                <HF as num_traits::AsPrimitive<u128>>::as_(a),
            ),
            FuzzOp::FToSingleToF(a) => <f32 as num_traits::AsPrimitive<HF>>::as_(
                <HF as num_traits::AsPrimitive<f32>>::as_(a),
            ),
            FuzzOp::FToDoubleToF(a) => <f64 as num_traits::AsPrimitive<HF>>::as_(
                <HF as num_traits::AsPrimitive<f64>>::as_(a),
            ),
        }
    }
}

impl<F> FuzzOp<F>
where
    F: rustc_apfloat::Float
        + rustc_apfloat::FloatConvert<rustc_apfloat::ieee::Single>
        + rustc_apfloat::FloatConvert<rustc_apfloat::ieee::Double>,
    rustc_apfloat::ieee::Single: rustc_apfloat::FloatConvert<F>,
    rustc_apfloat::ieee::Double: rustc_apfloat::FloatConvert<F>,
{
    pub fn eval_rs_apf(self) -> F {
        match self {
            FuzzOp::Neg(a) => -a,
            FuzzOp::Add(a, b) => (a + b).value,
            FuzzOp::Sub(a, b) => (a - b).value,
            FuzzOp::Mul(a, b) => (a * b).value,
            FuzzOp::Div(a, b) => (a / b).value,
            FuzzOp::Rem(a, b) => (a % b).value,
            FuzzOp::MulAdd(a, b, c) => a.mul_add(b, c).value,
            FuzzOp::FToI128ToF(a) => F::from_i128(a.to_i128(128).value).value,
            FuzzOp::FToU128ToF(a) => F::from_u128(a.to_u128(128).value).value,
            FuzzOp::FToSingleToF(a) => {
                rustc_apfloat::FloatConvert::convert(
                    rustc_apfloat::FloatConvert::<rustc_apfloat::ieee::Single>::convert(
                        a, &mut false,
                    )
                    .value,
                    &mut false,
                )
                .value
            }
            FuzzOp::FToDoubleToF(a) => {
                rustc_apfloat::FloatConvert::convert(
                    rustc_apfloat::FloatConvert::<rustc_apfloat::ieee::Double>::convert(
                        a, &mut false,
                    )
                    .value,
                    &mut false,
                )
                .value
            }
        }
    }
}
