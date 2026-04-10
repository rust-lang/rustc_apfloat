
#include <array>
#include <cstdint>
#include <stdint.h>
#include <stdio.h>
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/FloatingPointMode.h"

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define rassert(condition,  ...) \
    if(!__builtin_expect(condition, 1)) { \
        fprintf(stderr, "assertion `%s` failed at %d: ", STRINGIFY(condition), __LINE__); \
        fprintf(stderr,  __VA_ARGS__); \
        fprintf(stderr, "\n"); \
        exit(1); \
    }

using namespace llvm;

#pragma clang diagnostic error "-Wall"
#pragma clang diagnostic error "-Wextra"
#pragma clang diagnostic error "-Wunknown-attributes"

using uint128_t = __uint128_t;

/** The operation to perform is encoded as a u8 for passing across FFI */
enum class OpCode: uint8_t {
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
} tag;


/** Similarly, rounding mode is passed as a u8 */
enum class Round: uint8_t {
  NearestTiesToEven = 0,
  TowardZero        = 1,
  TowardPositive    = 2,
  TowardNegative    = 3,
  NearestTiesToAway = 4,
};

/* LLVM uses the following values:
 * opOK        = 0x00,
 * opInvalidOp = 0x01,
 * opDivByZero = 0x02,
 * opOverflow  = 0x04,
 * opUnderflow = 0x08,
 * opInexact   = 0x10
 */
using StatusFlags = unsigned;

/** Common operations for a given semantics are grouped in this class */
template<APFloat::Semantics S, typename U, const unsigned BITS = sizeof(U) * 8>
class FloatEval {
public:
    /** The integer size for passing in FFI (need not be the same size as BITS) */
    using UInt = U;

    /** Turn an APFloat back into its bitwise representation */
    static UInt apf_to_bits(APFloat apf) {
        APInt ap_bits = apf.bitcastToAPInt();
        unsigned bit_width = ap_bits.getBitWidth();
        rassert(bit_width == BITS, "%d != %d", bit_width, BITS);

        UInt repr = 0;
        for (unsigned i = 0; i < BITS; i += APInt::APINT_BITS_PER_WORD) {
            uint64_t word = ap_bits.getRawData()[i / APInt::APINT_BITS_PER_WORD];
            repr |= static_cast<UInt>(word) << i;
        }

        return repr;
    }

    /** Construct an APFloat from a bitwise representation */
    static APFloat bits_to_apf(UInt bits) {
        constexpr int ARR_LEN = (BITS + APInt::APINT_BITS_PER_WORD - 1) /
                                APInt::APINT_BITS_PER_WORD;
        std::array<APInt::WordType, ARR_LEN> words;
        for(unsigned i = 0; i < BITS; i += APInt::APINT_BITS_PER_WORD)
            words[i / APInt::APINT_BITS_PER_WORD] = bits >> i;
        APInt i(BITS, words);
        unsigned int_bits = i.getBitWidth();
        unsigned sem_size = APFloat::semanticsSizeInBits(getSemantics());
        rassert(int_bits == sem_size, "%d != %d", int_bits, sem_size);
        return APFloat(getSemantics(), APInt(BITS, words));
    }

    /** Evaluate a dynamically specified operation with the given configuration */
    static StatusFlags eval(OpCode op, Round round, UInt ai, UInt bi,
                            UInt ci, UInt &out)
    {
        APFloat a = bits_to_apf(ai);
        APFloat b = bits_to_apf(bi);
        APFloat c = bits_to_apf(ci);
        APFloat::opStatus status = APFloat::opOK;
        RoundingMode rm = APFloat::rmNearestTiesToEven;
        APSInt i;
        bool cvt_exact = false;

        const fltSemantics& sem = getSemantics();
        unsigned sem_size = APFloat::semanticsSizeInBits(sem);
        rassert(sem_size == BITS, "%d != %d", sem_size, BITS);

        switch(round) {
            case Round::NearestTiesToEven: break;
            case Round::TowardPositive:
                rm = APFloat::rmTowardPositive;
                break;
            case Round::TowardNegative:
                rm = APFloat::rmTowardNegative;
                break;
            case Round::TowardZero:
                rm = APFloat::rmTowardZero;
                break;
            case Round::NearestTiesToAway:
                rm = APFloat::rmNearestTiesToAway;
                break;
            default:
                printf("unsupported rounding mode %d\n", static_cast<int8_t>(rm));
                exit(1);
        }

        switch(op) {
            case OpCode::Neg:
                a.changeSign();
                break;
            case OpCode::Add:
                status = a.add(b, rm);
                break;
            case OpCode::Sub:
                status = a.subtract(b, rm);
                break;
            case OpCode::Mul:
                status = a.multiply(b, rm);
                break;
            case OpCode::Div:
                status = a.divide(b, rm);
                break;
            case OpCode::Rem:
                status = a.mod(b);
                break;
            case OpCode::MulAdd:
                status = a.fusedMultiplyAdd(b, c, rm);
                break;
            /* FIXME: the below operations could be incorrect and are discarding a
               status, and (though unlikely) could have mistkes that cancel. It would
               be better to make `out` a u128 and only do a single conversion. */
            case OpCode::FToI128ToF:
                i = APSInt(128, false);
                status = a.convertToInteger(i, rm, &cvt_exact);
                status = a.convertFromAPInt(i, true, rm);
                break;
            case OpCode::FToU128ToF:
                i = APSInt(128, true);
                status = a.convertToInteger(i, rm, &cvt_exact);
                status = a.convertFromAPInt(i, false, rm);
                break;
            case OpCode::FToSingleToF:
                a.convert(APFloat::IEEEsingle(), rm, &cvt_exact);
                a.convert(sem, rm, &cvt_exact);
                break;
            case OpCode::FToDoubleToF:
                a.convert(APFloat::IEEEsingle(), rm, &cvt_exact);
                a.convert(sem, rm, &cvt_exact);
                break;
            default:
                printf("unrecognized op tag %d", static_cast<int8_t>(tag));
                exit(1);
        }

        out = apf_to_bits(a);
        return (StatusFlags)status;
    }

    static const fltSemantics &getSemantics() {
        return APFloat::EnumToSemantics(S);
    }
};

/* Use the template to produce concrete classes */
class EvalBrainF16: public FloatEval<APFloat::Semantics::S_BFloat, uint16_t> {};
class EvalIeee16: public FloatEval<APFloat::Semantics::S_IEEEhalf, uint16_t> {};
class EvalIeee32: public FloatEval<APFloat::Semantics::S_IEEEsingle, uint32_t> {};
class EvalIeee64: public FloatEval<APFloat::Semantics::S_IEEEdouble, uint64_t> {};
class EvalIeee128: public FloatEval<APFloat::Semantics::S_IEEEquad, uint128_t> {};
class EvalPpcDoubleDouble: public FloatEval<APFloat::Semantics::S_PPCDoubleDouble, uint128_t> {};
class EvalF8E5M2: public FloatEval<APFloat::Semantics::S_Float8E5M2, uint8_t> {};
class EvalF8E4M3FN: public FloatEval<APFloat::Semantics::S_Float8E4M3FN, uint8_t> {};
class EvalX87F80: public FloatEval<APFloat::Semantics::S_x87DoubleExtended, uint128_t, 80> {};

/* And define the ways to invoke them */
#define MAKE_EXTERN(Ty, name) \
    StatusFlags name(OpCode op, Round round, Ty::UInt ai, Ty::UInt bi, \
                     Ty::UInt ci, Ty::UInt &out) { \
        return Ty::eval(op, round, ai, bi, ci, out); \
    }

extern "C" {
    /* NB: Every symbol defined here also needs to be in the list in build.rs */
    MAKE_EXTERN(EvalBrainF16, cxx_apf_eval_op_brainf16);
    MAKE_EXTERN(EvalIeee16, cxx_apf_eval_op_ieee16);
    MAKE_EXTERN(EvalIeee32, cxx_apf_eval_op_ieee32);
    MAKE_EXTERN(EvalIeee64, cxx_apf_eval_op_ieee64);
    MAKE_EXTERN(EvalIeee128, cxx_apf_eval_op_ieee128);
    MAKE_EXTERN(EvalPpcDoubleDouble, cxx_apf_eval_op_ppcdoubledouble);
    MAKE_EXTERN(EvalF8E5M2, cxx_apf_eval_op_f8e5m2);
    MAKE_EXTERN(EvalF8E4M3FN, cxx_apf_eval_op_f8e4m3fn);
    MAKE_EXTERN(EvalX87F80, cxx_apf_eval_op_x87_f80);
}
