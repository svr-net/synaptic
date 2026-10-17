// The virtual machine of the software GPU: executes compiled WGSL compute shaders the way a GPU
// does, as SIMT. A workgroup's invocations are the lanes of one wide machine: every instruction
// runs for all lanes at once over lane-major memory ([word][lane]), so the interpretation cost is
// shared by the whole workgroup and the per-lane loops vectorise. When a branch splits the
// lanes, each lane keeps its own program counter and the machine runs the lowest one first with
// the lanes that are there (execution masking); the lanes meet again where the paths join (the
// end of an if, a loop or an inlined function), and the machine is uniform again.
// Barriers are where all lanes are anyway.
#include <algorithm>
#include <array>
#include <cstring>
#include <utility>
#include <stdexcept>
#include <string>

#include "bytecode.hpp"
#include "syn/gpu/software_gpu.hpp"

namespace syn::gpu {

using namespace bc;

Buffer Buffer::ofFloats(const std::vector<float>& v) {
  Buffer b;
  b.words.resize(v.size());
  if (!v.empty()) std::memcpy(b.words.data(), v.data(), v.size() * 4);
  return b;
}

Buffer Buffer::ofWords(const std::vector<std::uint32_t>& v) {
  Buffer b;
  b.words = v;
  return b;
}

Buffer Buffer::zeros(std::size_t words, bool trackWrites) {
  Buffer b;
  b.words.assign(words, 0u);
  if (trackWrites) b.written.assign(words, 0u);
  return b;
}

std::vector<float> Buffer::floats() const {
  std::vector<float> v(words.size());
  if (!v.empty()) std::memcpy(v.data(), words.data(), v.size() * 4);
  return v;
}

namespace {

[[noreturn]] void fault(const char* what) { throw std::runtime_error(std::string("software GPU: ") + what); }

constexpr std::uint32_t kDone = 0xffffffffu;

class Machine {
 public:
  Machine(const Shader& s, const std::vector<Buffer*>& buffers)
      : s_(s), bufs_(buffers.data()), L_(s.workgroupSize[0] * s.workgroupSize[1] * s.workgroupSize[2]) {
    mem_.assign(static_cast<std::size_t>(s.memoryWords) * L_, 0u);
    for (std::size_t k = 0; k < s.constants.size(); ++k)
      std::fill_n(mem_.begin() + static_cast<std::ptrdiff_t>((s.constantBase + k) * L_), L_, s.constants[k]);
    wg_.assign(std::max<std::uint32_t>(1, s.workgroupWords), 0u);
    pc_.assign(L_, 0u);
    act_.resize(L_);
  }

  void group(std::uint32_t gx, std::uint32_t gy, std::uint32_t gz, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    const std::uint32_t wx = s_.workgroupSize[0], wy = s_.workgroupSize[1], wz = s_.workgroupSize[2];
    std::fill(wg_.begin(), wg_.end(), 0u);
    std::fill_n(mem_.begin(), static_cast<std::size_t>(s_.privateWords) * L_, 0u);
    for (const auto& b : s_.builtins)
      for (std::uint32_t i = 0; i < L_; ++i) {
        const std::uint32_t lx = i % wx, ly = (i / wx) % wy, lz = i / (wx * wy);
        std::uint32_t v[3] = {0, 0, 0};
        switch (b.which) {
          case Shader::LocalId: v[0] = lx, v[1] = ly, v[2] = lz; break;
          case Shader::WorkgroupId: v[0] = x, v[1] = y, v[2] = z; break;
          case Shader::GlobalId: v[0] = x * wx + lx, v[1] = y * wy + ly, v[2] = z * wz + lz; break;
          case Shader::NumWorkgroups: v[0] = gx, v[1] = gy, v[2] = gz; break;
          case Shader::LocalIndex: v[0] = i; break;
        }
        const std::uint32_t n = b.which == Shader::LocalIndex ? 1 : 3;
        for (std::uint32_t k = 0; k < n; ++k) mem_[(b.slot + k) * L_ + i] = v[k];
      }
    run();
  }

 private:
  const Shader& s_;
  Buffer* const* bufs_;
  const std::uint32_t L_;
  std::vector<std::uint32_t> mem_, wg_, pc_, act_;

  std::uint32_t* row(std::uint32_t slot) { return mem_.data() + static_cast<std::size_t>(slot) * L_; }

  // One handler per operation: small functions, so that a WebAssembly engine compiles and
  // optimises the hot ones quickly (one large function would run unoptimised for long while
  // every worker competes for the compiler's threads).
  using Handler = void (Machine::*)(const Ins&, std::uint32_t);

  template <bool all, std::size_t... Ops>
  static std::array<Handler, sizeof...(Ops)> table(std::index_sequence<Ops...>) {
    return {{&Machine::exec<static_cast<std::uint32_t>(Ops), all>...}};
  }

  // Executes operation `Code` for the lanes act_[0..n) (all lanes when `all`).
  template <std::uint32_t Code, bool all>
  void exec(const Ins& I, std::uint32_t n) {
    const std::uint32_t L = L_;
    auto lanes = [&](auto&& f) {
      if constexpr (all) {
        for (std::uint32_t l = 0; l < L; ++l) f(l);
      } else {
        for (std::uint32_t k = 0; k < n; ++k) f(act_[k]);
      }
    };
    switch (Code) {
      case MOV: {
        std::uint32_t *d = row(I.a), *a = row(I.b);
        lanes([&](std::uint32_t l) { d[l] = a[l]; });
        return;
      }
      case MOVN: {
        if constexpr (all) {
          std::memmove(row(I.a), row(I.b), static_cast<std::size_t>(I.c) * L * 4);
        } else {
          if (I.a <= I.b)
            for (std::uint32_t k = 0; k < I.c; ++k) {
              std::uint32_t *d = row(I.a + k), *a = row(I.b + k);
              lanes([&](std::uint32_t l) { d[l] = a[l]; });
            }
          else
            for (std::uint32_t k = I.c; k-- > 0;) {
              std::uint32_t *d = row(I.a + k), *a = row(I.b + k);
              lanes([&](std::uint32_t l) { d[l] = a[l]; });
            }
        }
        return;
      }
      case SEL: {
        std::uint32_t *d = row(I.a), *f = row(I.b), *t = row(I.c), *c = row(I.d);
        lanes([&](std::uint32_t l) { d[l] = c[l] != 0u ? t[l] : f[l]; });
        return;
      }
      case LDM: {
        const std::uint32_t* off = row(I.c);
        lanes([&](std::uint32_t l) {
          const std::size_t at = static_cast<std::size_t>(I.b) + off[l];
          if (at + I.d > s_.memoryWords) fault("private read out of range");
          for (std::uint32_t k = 0; k < I.d; ++k) mem_[(I.a + k) * L + l] = mem_[(at + k) * L + l];
        });
        return;
      }
      case STM: {
        const std::uint32_t* off = row(I.b);
        lanes([&](std::uint32_t l) {
          const std::size_t at = static_cast<std::size_t>(I.a) + off[l];
          if (at + I.d > s_.memoryWords) fault("private write out of range");
          for (std::uint32_t k = 0; k < I.d; ++k) mem_[(at + k) * L + l] = mem_[(I.c + k) * L + l];
        });
        return;
      }
      case LDW: {
        const std::uint32_t* off = row(I.c);
        lanes([&](std::uint32_t l) {
          const std::size_t at = static_cast<std::size_t>(I.b) + off[l];
          if (at + I.d > wg_.size()) fault("workgroup read out of range");
          for (std::uint32_t k = 0; k < I.d; ++k) mem_[(I.a + k) * L + l] = wg_[at + k];
        });
        return;
      }
      case STW: {
        const std::uint32_t* off = row(I.b);
        lanes([&](std::uint32_t l) {
          const std::size_t at = static_cast<std::size_t>(I.a) + off[l];
          if (at + I.d > wg_.size()) fault("workgroup write out of range");
          for (std::uint32_t k = 0; k < I.d; ++k) wg_[at + k] = mem_[(I.c + k) * L + l];
        });
        return;
      }
      case LDB: {
        const Buffer& b = *bufs_[I.b];
        const std::size_t size = b.words.size();
        const std::uint32_t* off = row(I.c);
        lanes([&](std::uint32_t l) {
          const std::size_t at = static_cast<std::size_t>(I.e) + off[l];
          for (std::uint32_t k = 0; k < I.d; ++k) mem_[(I.a + k) * L + l] = at + k < size ? b.words[at + k] : 0u;
        });
        return;
      }
      case STB: {
        Buffer& b = *bufs_[I.a];
        const std::size_t size = b.words.size();
        const bool track = !b.written.empty();
        const std::uint32_t* off = row(I.b);
        lanes([&](std::uint32_t l) {
          const std::size_t at = static_cast<std::size_t>(I.e) + off[l];
          for (std::uint32_t k = 0; k < I.d; ++k)
            if (at + k < size) {
              b.words[at + k] = mem_[(I.c + k) * L + l];
              if (track) b.written[at + k] = 1u;
            }
        });
        return;
      }
      case ATADD: {
        const std::uint32_t *off = row(I.c), *v = row(I.d);
        std::uint32_t* d = row(I.a);
        lanes([&](std::uint32_t l) {
          const std::size_t at = static_cast<std::size_t>(I.b) + off[l];
          if (at >= wg_.size()) fault("atomic out of range");
          d[l] = wg_[at];
          wg_[at] += v[l];
        });
        return;
      }
      case ATLD: {
        const std::uint32_t* off = row(I.c);
        std::uint32_t* d = row(I.a);
        lanes([&](std::uint32_t l) {
          const std::size_t at = static_cast<std::size_t>(I.b) + off[l];
          if (at >= wg_.size()) fault("atomic out of range");
          d[l] = wg_[at];
        });
        return;
      }
      case ATST: {
        const std::uint32_t *off = row(I.b), *v = row(I.c);
        lanes([&](std::uint32_t l) {
          const std::size_t at = static_cast<std::size_t>(I.a) + off[l];
          if (at >= wg_.size()) fault("atomic out of range");
          wg_[at] = v[l];
        });
        return;
      }
#define SYN_BIN(OP)                                                                \
  case OP: {                                                                       \
    std::uint32_t *d = row(I.a), *a = row(I.b), *b = row(I.c);                     \
    lanes([&](std::uint32_t l) { d[l] = binary(OP, a[l], b[l]); });               \
    return;                                                                        \
  }
#define SYN_UN(OP)                                                                 \
  case OP: {                                                                       \
    std::uint32_t *d = row(I.a), *a = row(I.b);                                    \
    lanes([&](std::uint32_t l) { d[l] = unary(OP, a[l]); });                       \
    return;                                                                        \
  }
      SYN_BIN(FADD) SYN_BIN(FSUB) SYN_BIN(FMUL) SYN_BIN(FDIV) SYN_BIN(FMOD) SYN_BIN(FMIN) SYN_BIN(FMAX) SYN_BIN(FATAN2) SYN_BIN(FPOW)
      SYN_BIN(FLT) SYN_BIN(FLE) SYN_BIN(FGT) SYN_BIN(FGE) SYN_BIN(FEQ) SYN_BIN(FNE)
      SYN_BIN(IADD) SYN_BIN(ISUB) SYN_BIN(IMUL) SYN_BIN(IDIV) SYN_BIN(IMOD) SYN_BIN(IMIN) SYN_BIN(IMAX)
      SYN_BIN(ILT) SYN_BIN(ILE) SYN_BIN(IGT) SYN_BIN(IGE)
      SYN_BIN(UDIV) SYN_BIN(UMOD) SYN_BIN(UMIN) SYN_BIN(UMAX) SYN_BIN(ULT) SYN_BIN(ULE) SYN_BIN(UGT) SYN_BIN(UGE)
      SYN_BIN(EQ) SYN_BIN(NE) SYN_BIN(AND) SYN_BIN(OR) SYN_BIN(XOR) SYN_BIN(SHL) SYN_BIN(SHR) SYN_BIN(SAR)
      SYN_UN(FNEG) SYN_UN(FABS) SYN_UN(FSQRT) SYN_UN(FINVSQRT) SYN_UN(FLOG) SYN_UN(FLOG2) SYN_UN(FEXP) SYN_UN(FEXP2)
      SYN_UN(FSIN) SYN_UN(FCOS) SYN_UN(FTAN) SYN_UN(FASIN) SYN_UN(FACOS) SYN_UN(FATAN) SYN_UN(FSINH) SYN_UN(FCOSH)
      SYN_UN(FTANH) SYN_UN(FFLOOR) SYN_UN(FCEIL) SYN_UN(FROUND) SYN_UN(FTRUNC) SYN_UN(FFRACT) SYN_UN(FSIGN)
      SYN_UN(INEG) SYN_UN(IABS) SYN_UN(NOT) SYN_UN(LNOT)
      SYN_UN(F2I) SYN_UN(F2U) SYN_UN(I2F) SYN_UN(U2F) SYN_UN(B2F) SYN_UN(F2B) SYN_UN(I2B)
#undef SYN_BIN
#undef SYN_UN
      default: fault("bad instruction");
    }
  }

  void run() {
    static const auto kAllLanes = table<true>(std::make_index_sequence<OP_COUNT>{});
    static const auto kSomeLanes = table<false>(std::make_index_sequence<OP_COUNT>{});
    const Ins* code = s_.code.data();
    const std::uint32_t L = L_;
    std::uint32_t upc = 0;  // the program counter while all lanes are together
    bool uniform = true;
    for (;;) {
      if (uniform) {
        const Ins& I = code[upc];
        switch (I.op) {
          case END: return;
          case BARRIER: ++upc; continue;
          case JMP: upc = I.a; continue;
          case JZ:
          case JNZ: {
            const std::uint32_t* c = row(I.a);
            std::uint32_t taken = 0;
            for (std::uint32_t l = 0; l < L; ++l) taken += (c[l] == 0u) == (I.op == JZ) ? 1u : 0u;
            if (taken == L) {
              upc = I.b;
            } else if (taken == 0) {
              ++upc;
            } else {
              for (std::uint32_t l = 0; l < L; ++l) pc_[l] = (c[l] == 0u) == (I.op == JZ) ? I.b : upc + 1;
              uniform = false;
            }
            continue;
          }
          default:
            (this->*kAllLanes[I.op])(I, L);
            ++upc;
            continue;
        }
      }
      // Divergent: the lanes at the lowest program counter run together until they reach the
      // next waiting lane (where they may join it), split at a branch, or end.
      std::uint32_t lo = kDone, wait = kDone;
      for (std::uint32_t l = 0; l < L; ++l) lo = std::min(lo, pc_[l]);
      if (lo == kDone) return;
      std::uint32_t n = 0;
      for (std::uint32_t l = 0; l < L; ++l) {
        if (pc_[l] == lo) act_[n++] = l;
        else wait = std::min(wait, pc_[l]);
      }
      if (n == L) {  // together again
        uniform = true;
        upc = lo;
        continue;
      }
      std::uint32_t pc = lo;
      for (;;) {
        if (pc >= wait) {
          for (std::uint32_t k = 0; k < n; ++k) pc_[act_[k]] = pc;
          break;
        }
        const Ins& I = code[pc];
        if (I.op == END) {
          for (std::uint32_t k = 0; k < n; ++k) pc_[act_[k]] = kDone;
          break;
        }
        if (I.op == BARRIER) {
          ++pc;
          continue;
        }
        if (I.op == JMP) {
          pc = I.a;
          continue;
        }
        if (I.op == JZ || I.op == JNZ) {
          const std::uint32_t* c = row(I.a);
          std::uint32_t taken = 0;
          for (std::uint32_t k = 0; k < n; ++k) taken += (c[act_[k]] == 0u) == (I.op == JZ) ? 1u : 0u;
          if (taken == n) {
            pc = I.b;
          } else if (taken == 0) {
            ++pc;
          } else {
            for (std::uint32_t k = 0; k < n; ++k) {
              const std::uint32_t l = act_[k];
              pc_[l] = (c[l] == 0u) == (I.op == JZ) ? I.b : pc + 1;
            }
            break;
          }
          continue;
        }
        (this->*kSomeLanes[I.op])(I, n);
        ++pc;
      }
    }
  }
};

}  // namespace

void dispatch(const Shader& s, const std::vector<Buffer*>& buffers, std::uint32_t gx, std::uint32_t gy, std::uint32_t gz, std::size_t first,
              std::size_t last) {
  for (const auto& b : s.bindings)
    if (b.binding >= buffers.size() || buffers[b.binding] == nullptr) fault(("no buffer bound for " + b.name).c_str());
  const std::size_t total = static_cast<std::size_t>(gx) * gy * gz;
  last = std::min(last, total);
  if (first >= last) return;
  Machine m(s, buffers);
  for (std::size_t g = first; g < last; ++g) {
    const auto x = static_cast<std::uint32_t>(g % gx), y = static_cast<std::uint32_t>((g / gx) % gy),
               z = static_cast<std::uint32_t>(g / (static_cast<std::size_t>(gx) * gy));
    m.group(gx, gy, gz, x, y, z);
  }
}

}  // namespace syn::gpu
