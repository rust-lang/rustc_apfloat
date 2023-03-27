///! Fuzzing "ops" are a small set of floating-point operations (available both
/// natively and via `llvm::APFloat`/`rustc_apfloat`), represented in code as a
/// generic `FuzzOp` `enum` (with each variant also carrying that op's inputs),
/// with a straight-forward binary serialization (for fuzzing to operate on),
/// and a defined ABI (which Rust code can use to call into the C++ wrapper).
///
/// This file contains the definitions used to generate both Rust and C++ code.

// HACK(eddyb) newtypes to make it easy to tell apart Rust vs C++ specifics.
struct Rust<T>(T);
struct Cxx<T>(T);

use self::OpKind::*;
enum OpKind {
    Unary(Rust<char>, Cxx<&'static str>),
    Binary(Rust<char>, Cxx<&'static str>),
    Ternary(Rust<&'static str>, Cxx<&'static str>),

    // HACK(eddyb) all other ops have floating-point inputs *and* outputs, so
    // the easiest way to fuzz conversions from/to other types, even if it won't
    // cover *all possible* inputs, is to do a round-trip through the other type.
    Roundtrip(Type),
}

enum Type {
    SInt(usize),
    UInt(usize),
}

impl Type {
    fn rust_type(&self) -> String {
        match self {
            Type::SInt(w) => format!("i{w}"),
            Type::UInt(w) => format!("u{w}"),
        }
    }
}

impl OpKind {
    fn inputs<'a, T>(&self, all_inputs: &'a [T; 3]) -> &'a [T] {
        match self {
            Unary(..) | Roundtrip(_) => &all_inputs[..1],
            Binary(..) => &all_inputs[..2],
            Ternary(..) => &all_inputs[..3],
        }
    }
}

const OPS: &[(&str, OpKind)] = &[
    // Unary (`F -> F`) ops.
    ("Neg", Unary(Rust('-'), Cxx("changeSign"))),
    // Binary (`(F, F) -> F`) ops.
    ("Add", Binary(Rust('+'), Cxx("add"))),
    ("Sub", Binary(Rust('-'), Cxx("subtract"))),
    ("Mul", Binary(Rust('*'), Cxx("multiply"))),
    ("Div", Binary(Rust('/'), Cxx("divide"))),
    ("Rem", Binary(Rust('%'), Cxx("mod"))),
    // Ternary (`(F, F) -> F`) ops.
    ("MulAdd", Ternary(Rust("mul_add"), Cxx("fusedMultiplyAdd"))),
    // Roundtrip (`F -> T -> F`) ops.
    ("FToI128ToF", Roundtrip(Type::SInt(128))),
    ("FToU128ToF", Roundtrip(Type::UInt(128))),
];

fn all_ops_map_concat(f: impl Fn(usize, &'static str, &OpKind) -> String) -> String {
    OPS.iter()
        .enumerate()
        .map(|(tag, (name, kind))| f(tag, name, kind))
        .collect()
}

pub fn generate_rust() -> String {
    String::new()
        + "
#[derive(Copy, Clone, Debug)]
#[repr(u8)]
enum FuzzOp<T> {"
        + &all_ops_map_concat(|tag, name, kind| {
            format!(
                "
    {name}({input_types}) = {tag},",
                input_types = kind.inputs(&["T", "T", "T"]).join(", ")
            )
        })
        + "
}

impl FuzzOp<()> {
    fn from_tag(tag: u8) -> Option<Self> {
        Some(match tag {"
        + &all_ops_map_concat(|tag, name, kind| {
            format!(
                "
            {tag} => FuzzOp::{name}({inputs}),",
                inputs = kind.inputs(&["()", "()", "()"]).join(", ")
            )
        })
        + "
            _ => return None,
        })
    }
}

impl<T> FuzzOp<T> {
    fn tag(self) -> u8 {
        match self {"
        + &all_ops_map_concat(|tag, name, _op| {
            format!(
                "
            FuzzOp::{name}(..) => {tag},",
            )
        })
        + "
        }
    }

    fn map<U>(self, mut f: impl FnMut(T) -> U) -> FuzzOp<U> {
        match self {
" + &all_ops_map_concat(|_tag, name, kind| {
        format!(
            "
            FuzzOp::{name}({inputs}) => FuzzOp::{name}({f_inputs}),",
            inputs = kind.inputs(&["a", "b", "c"]).join(", "),
            f_inputs = kind.inputs(&["f(a)", "f(b)", "f(c)"]).join(", "),
        )
    }) + "
        }
    }
}

impl<HF> FuzzOp<HF>
    where
        HF: num_traits::Float
            + num_traits::AsPrimitive<i128>
            + num_traits::AsPrimitive<u128>,
        i128: num_traits::AsPrimitive<HF>,
        u128: num_traits::AsPrimitive<HF>,
{
    fn eval_hard(self) -> HF {
        match self {
" + &all_ops_map_concat(|_tag, name, kind| {
        let inputs = kind.inputs(&["a", "b", "c"]);
        let expr = match kind {
            Unary(Rust(op), _) => format!("{op}{}", inputs[0]),
            Binary(Rust(op), _) => format!("{} {op} {}", inputs[0], inputs[1]),
            Ternary(Rust(method), _) => {
                format!("{}.{method}({}, {})", inputs[0], inputs[1], inputs[2])
            }
            Roundtrip(ty) => format!(
                "<{ty} as num_traits::AsPrimitive::<HF>>::as_(
                    <HF as num_traits::AsPrimitive::<{ty}>>::as_({}))",
                inputs[0],
                ty = ty.rust_type()
            ),
        };
        format!(
            "
            FuzzOp::{name}({inputs}) => {expr},",
            inputs = inputs.join(", "),
        )
    }) + "
        }
    }
}

impl<RustcApFloat: rustc_apfloat::Float> FuzzOp<RustcApFloat> {
    fn eval_rs_apf(self) -> RustcApFloat {
        match self {
" + &all_ops_map_concat(|_tag, name, kind| {
        let inputs = kind.inputs(&["a", "b", "c"]);
        let expr = match kind {
            Unary(Rust(op), _) => format!("{op}{}", inputs[0]),
            Binary(Rust(op), _) => format!("({} {op} {}).value", inputs[0], inputs[1]),
            Ternary(Rust(method), _) => {
                format!("{}.{method}({}).value", inputs[0], inputs[1..].join(", "))
            }
            Roundtrip(ty @ (Type::SInt(_) | Type::UInt(_))) => {
                let (w, i_or_u) = match ty {
                    Type::SInt(w) => (w, "i"),
                    Type::UInt(w) => (w, "u"),
                };
                format!(
                    "RustcApFloat::from_{i_or_u}128({}.to_{i_or_u}128({w}).value).value",
                    inputs[0],
                )
            }
        };
        format!(
            "
            FuzzOp::{name}({inputs}) => {expr},",
            inputs = inputs.join(", "),
        )
    }) + "
        }
    }
}"
}

pub fn generate_cxx(exported_symbols: &mut Vec<String>) -> String {
    String::new()
        + r#"
#include <array>
#include <llvm/ADT/APFloat.h>

using namespace llvm;

#pragma clang diagnostic error "-Wall"
#pragma clang diagnostic error "-Wextra"
#pragma clang diagnostic error "-Wunknown-attributes"

// HACK(eddyb) allow code below to assume `uintN_t` maps to Rust `uN`.
// FIXME(eddyb) make sure this doesn't run into `u128` ABI issues.
using uint128_t = __uint128_t;

template<typename F>
struct FuzzOp {
    enum : uint8_t {"#
        + &all_ops_map_concat(|tag, name, _kind| {
            format!(
                "
        {name} = {tag},"
            )
        })
        + "
    } tag;
    F a, b, c;

    F eval() const {
        switch(tag) {
            "
        + &all_ops_map_concat(|_tag, name, kind| {
            let inputs = kind.inputs(&["a.to_apf()", "b.to_apf()", "c.to_apf()"]);
            let (this, args) = inputs.split_first().unwrap();
            let args = args.join(", ");
            let stmt = match kind {
                // HACK(eddyb) `mod` doesn't take a rounding mode.
                Unary(_, Cxx(method)) | Binary(_, Cxx(method @ "mod")) => {
                    format!("r.{method}({args})")
                }

                Binary(_, Cxx(method)) | Ternary(_, Cxx(method)) => {
                    format!("r.{method}({args}, APFloat::rmNearestTiesToEven)")
                }

                Roundtrip(ty @ (Type::SInt(_) | Type::UInt(_))) => {
                    let (w, signed) = match ty {
                        Type::SInt(w) => (w, true),
                        Type::UInt(w) => (w, false),
                    };
                    format!(
                        "
                APSInt i({w}, !{signed});
                bool isExact;
                r.convertToInteger(i, APFloat::rmTowardZero, &isExact);
                r.convertFromAPInt(i, {signed}, APFloat::rmNearestTiesToEven)"
                    )
                }
            };
            format!(
                "
            case {name}: {{
                APFloat r = {this};
                {stmt};
                return F::from_apf(r);
            }}",
            )
        })
        + "
        }
    }
};
" + &[
        (16, "APFloat::IEEEhalf()"),
        (32, "APFloat::IEEEsingle()"),
        (64, "APFloat::IEEEdouble()"),
        (128, "APFloat::IEEEquad()"),
        (80, "APFloat::x87DoubleExtended()"),
    ]
    .into_iter()
    .map(|(w, cxx_apf_semantics)| {
        let (name_prefix, uint_width) = match w {
            80 => ("X87_F", 128),
            _ => ("IEEE", w),
        };
        let name = format!("{name_prefix}{w}");
        let exported_symbol = format!(
            "cxx_apf_fuzz_eval_op_{}{w}",
            name_prefix.to_ascii_lowercase()
        );
        exported_symbols.push(exported_symbol.clone());
        let uint = format!("uint{uint_width}_t");
        format!(
            r#"
struct __attribute__((packed)) {name} {{
    {uint} bits;

    // HACK(eddyb) these work around `APInt` only being convenient up to 64 bits.

    static {name} from_apf(APFloat apf) {{
        auto ap_bits = apf.bitcastToAPInt();
        assert(ap_bits.getBitWidth() == {w});

        {uint} bits = 0;
        for(int i = 0; i < {w}; i += APInt::APINT_BITS_PER_WORD)
            bits |= static_cast<{uint}>(
                ap_bits.getRawData()[i / APInt::APINT_BITS_PER_WORD]
            ) << i;
        return {{ bits }};
    }}

    APFloat to_apf() const {{
        std::array<APInt::WordType, ({w} + APInt::APINT_BITS_PER_WORD - 1) / APInt::APINT_BITS_PER_WORD>
            words;
        for(int i = 0; i < {w}; i += APInt::APINT_BITS_PER_WORD)
            words[i / APInt::APINT_BITS_PER_WORD] = bits >> i;
        return APFloat({cxx_apf_semantics}, APInt({w}, words));
    }}
}};
extern "C" {{
    void {exported_symbol}({name} *out, const FuzzOp<{name}> &op) {{
        *out = op.eval();
    }}
}}"#
        )
    })
    .collect::<String>()
}
