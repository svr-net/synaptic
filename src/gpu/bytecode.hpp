#pragma once

// Bytecode of the software GPU, shared by the WGSL compiler and the virtual machine.
//
// Every value lives in 32-bit words of an invocation's memory M (or of the workgroup's memory W,
// or of a buffer). Vectors, arrays and structures are consecutive words; booleans are 0 or 1.
// The compiler expands vector operations into scalar instructions, so the machine only knows
// scalars. The scalar semantics below are used both to fold constants at compile time and to
// execute, so a folded constant is exactly what the machine would have computed.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace syn::gpu::bc {

// The machine applies these per lane in its inner loops: they must be inlined there.
#if defined(__GNUC__) || defined(__clang__)
#define SYN_GPU_INLINE inline __attribute__((always_inline))
#else
#define SYN_GPU_INLINE inline
#endif

enum Op : std::uint32_t {
  // control
  END, BARRIER, JMP, JZ, JNZ,
  // copies
  MOV, MOVN,
  // f32
  FADD, FSUB, FMUL, FDIV, FMOD, FMIN, FMAX, FATAN2, FPOW,
  FLT, FLE, FGT, FGE, FEQ, FNE,
  FNEG, FABS, FSQRT, FINVSQRT, FLOG, FLOG2, FEXP, FEXP2, FSIN, FCOS, FTAN, FASIN, FACOS, FATAN,
  FSINH, FCOSH, FTANH, FFLOOR, FCEIL, FROUND, FTRUNC, FFRACT, FSIGN,
  // integers (two's complement words; IADD/ISUB/IMUL serve i32 and u32)
  IADD, ISUB, IMUL, IDIV, IMOD, IMIN, IMAX, ILT, ILE, IGT, IGE, INEG, IABS,
  UDIV, UMOD, UMIN, UMAX, ULT, ULE, UGT, UGE,
  EQ, NE, AND, OR, XOR, SHL, SHR, SAR, NOT, LNOT,
  // conversions
  F2I, F2U, I2F, U2F, B2F, F2B, I2B,
  // select(f, t, c)
  SEL,
  // memory with a dynamic word offset
  LDM, STM, LDW, STW, LDB, STB,
  // workgroup atomics
  ATADD, ATLD, ATST,
  OP_COUNT
};

inline bool isUnary(std::uint32_t op) {
  return (op >= FNEG && op <= FSIGN) || op == INEG || op == IABS || op == NOT || op == LNOT || (op >= F2I && op <= I2B);
}
inline bool isBinary(std::uint32_t op) { return op >= FADD && op <= LNOT && !isUnary(op); }

// Operand roles, one character per operand a..e:
//   'w' writes M,  'r' reads M (may be a constant),  '-' anything else (jump target, count, ...).
inline const char* roles(std::uint32_t op) {
  switch (op) {
    case END: case BARRIER: return "-----";
    case JMP: return "-----";
    case JZ: case JNZ: return "r----";
    case MOV: return "wr---";
    case MOVN: return "wr---";
    case SEL: return "wrrr-";
    case LDM: return "wrr--";
    case STM: return "-rr--";
    case LDW: return "w-r--";
    case STW: return "-rr--";
    case LDB: return "w-r--";
    case STB: return "-rr--";
    case ATADD: return "w-rr-";
    case ATLD: return "w-r--";
    case ATST: return "-rr--";
    default: break;
  }
  return isUnary(op) ? "wr---" : "wrr--";
}


SYN_GPU_INLINE float F(std::uint32_t w) {
  float f;
  std::memcpy(&f, &w, 4);
  return f;
}
SYN_GPU_INLINE std::uint32_t W(float f) {
  std::uint32_t w;
  std::memcpy(&w, &f, 4);
  return w;
}
SYN_GPU_INLINE std::int32_t I(std::uint32_t w) { return static_cast<std::int32_t>(w); }

SYN_GPU_INLINE std::uint32_t unary(std::uint32_t op, std::uint32_t x) {
  const float f = F(x);
  switch (op) {
    case FNEG: return W(-f);
    case FABS: return W(std::fabs(f));
    case FSQRT: return W(std::sqrt(f));
    case FINVSQRT: return W(1.0f / std::sqrt(f));
    case FLOG: return W(std::log(f));
    case FLOG2: return W(std::log2(f));
    case FEXP: return W(std::exp(f));
    case FEXP2: return W(std::exp2(f));
    case FSIN: return W(std::sin(f));
    case FCOS: return W(std::cos(f));
    case FTAN: return W(std::tan(f));
    case FASIN: return W(std::asin(f));
    case FACOS: return W(std::acos(f));
    case FATAN: return W(std::atan(f));
    case FSINH: return W(std::sinh(f));
    case FCOSH: return W(std::cosh(f));
    case FTANH: return W(std::tanh(f));
    case FFLOOR: return W(std::floor(f));
    case FCEIL: return W(std::ceil(f));
    case FROUND: return W(std::nearbyint(f));
    case FTRUNC: return W(std::trunc(f));
    case FFRACT: return W(f - std::floor(f));
    case FSIGN: return W(f > 0.0f ? 1.0f : (f < 0.0f ? -1.0f : 0.0f));
    case INEG: return 0u - x;
    case IABS: return I(x) < 0 ? 0u - x : x;
    case NOT: return ~x;
    case LNOT: return x == 0u ? 1u : 0u;
    case F2I: {
      if (std::isnan(f)) return 0u;
      const float c = f < -2147483648.0f ? -2147483648.0f : (f > 2147483520.0f ? 2147483520.0f : f);
      return static_cast<std::uint32_t>(static_cast<std::int32_t>(c));
    }
    case F2U: {
      if (std::isnan(f) || f <= 0.0f) return 0u;
      const float c = f > 4294967040.0f ? 4294967040.0f : f;
      return static_cast<std::uint32_t>(c);
    }
    case I2F: return W(static_cast<float>(I(x)));
    case U2F: return W(static_cast<float>(x));
    case B2F: return W(x != 0u ? 1.0f : 0.0f);
    case F2B: return f != 0.0f ? 1u : 0u;
    case I2B: return x != 0u ? 1u : 0u;
    default: return 0u;
  }
}

SYN_GPU_INLINE std::uint32_t binary(std::uint32_t op, std::uint32_t x, std::uint32_t y) {
  switch (op) {
    case FADD: return W(F(x) + F(y));
    case FSUB: return W(F(x) - F(y));
    case FMUL: return W(F(x) * F(y));
    case FDIV: return W(F(x) / F(y));
    case FMOD: return W(F(x) - F(y) * std::trunc(F(x) / F(y)));
    case FMIN: return W(F(y) < F(x) ? F(y) : F(x));
    case FMAX: return W(F(y) > F(x) ? F(y) : F(x));
    case FATAN2: return W(std::atan2(F(x), F(y)));
    case FPOW: return W(std::pow(F(x), F(y)));
    case FLT: return F(x) < F(y);
    case FLE: return F(x) <= F(y);
    case FGT: return F(x) > F(y);
    case FGE: return F(x) >= F(y);
    case FEQ: return F(x) == F(y);
    case FNE: return F(x) != F(y);
    case IADD: return x + y;
    case ISUB: return x - y;
    case IMUL: return x * y;
    case IDIV:
      if (y == 0u || (x == 0x80000000u && y == 0xffffffffu)) return x;
      return static_cast<std::uint32_t>(I(x) / I(y));
    case IMOD:
      if (y == 0u || (x == 0x80000000u && y == 0xffffffffu)) return 0u;
      return static_cast<std::uint32_t>(I(x) % I(y));
    case IMIN: return I(y) < I(x) ? y : x;
    case IMAX: return I(y) > I(x) ? y : x;
    case ILT: return I(x) < I(y);
    case ILE: return I(x) <= I(y);
    case IGT: return I(x) > I(y);
    case IGE: return I(x) >= I(y);
    case UDIV: return y == 0u ? x : x / y;
    case UMOD: return y == 0u ? 0u : x % y;
    case UMIN: return y < x ? y : x;
    case UMAX: return y > x ? y : x;
    case ULT: return x < y;
    case ULE: return x <= y;
    case UGT: return x > y;
    case UGE: return x >= y;
    case EQ: return x == y;
    case NE: return x != y;
    case AND: return x & y;
    case OR: return x | y;
    case XOR: return x ^ y;
    case SHL: return x << (y & 31u);
    case SHR: return x >> (y & 31u);
    case SAR: return static_cast<std::uint32_t>(I(x) >> (y & 31u));
    default: return 0u;
  }
}

}  // namespace syn::gpu::bc
