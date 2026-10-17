#pragma once

// The software GPU: runs WGSL compute shaders on the CPU, exactly as a GPU would.
//
// When the browser has no WebGPU adapter (and in the native build and its tests), the library's
// kernels still run as WGSL: Shader::compile translates the WGSL source into bytecode and
// dispatch() executes it with the GPU's execution model:
//   - a grid of workgroups; each workgroup runs its invocations in lockstep between barriers
//     (workgroupBarrier), with its own var<workgroup> memory;
//   - every invocation has its own var<private> and function memory;
//   - storage and uniform buffers are arrays of 32-bit words shared by the whole dispatch;
//     out-of-bounds reads return 0 and out-of-bounds writes are dropped;
//   - f32, i32 and u32 arithmetic in 32 bits: integers wrap, float-to-integer conversions
//     saturate, shifts use the low five bits of the shift amount.
// There is no other implementation of the kernels: the WGSL that WebGPU runs is the WGSL this
// runs. The supported language is the part of WGSL the kernels use (see wgsl_compiler.cpp).

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace syn::gpu {

/** The float stored in a 32-bit word, and back. */
inline float asFloat(std::uint32_t w) {
  float f;
  std::memcpy(&f, &w, 4);
  return f;
}
inline std::uint32_t asWord(float f) {
  std::uint32_t w;
  std::memcpy(&w, &f, 4);
  return w;
}

/** One bytecode instruction: an operation and up to five operands (see bytecode.hpp). */
struct Ins {
  std::uint32_t op = 0, a = 0, b = 0, c = 0, d = 0, e = 0;
};

/** A resource binding of a shader (@group(0) only). */
struct BindingInfo {
  std::uint32_t binding = 0;
  bool uniform = false;
  bool writable = false;
  std::string name;
};

/** A compiled compute shader. */
class Shader {
 public:
  /** Compiles the entry point `entry` of a WGSL module. Throws std::runtime_error on errors. */
  static std::shared_ptr<const Shader> compile(const std::string& wgsl, const std::string& entry = "main");

  std::uint32_t workgroupSize[3] = {1, 1, 1};
  std::vector<BindingInfo> bindings;

  // ---- the compiled program (used by dispatch) ----
  std::vector<Ins> code;
  std::uint32_t memoryWords = 0;      ///< per-invocation memory: [private | function slots | constants]
  std::uint32_t privateWords = 0;     ///< zeroed at the start of every invocation
  std::vector<std::uint32_t> constants;  ///< copied to [constantBase, constantBase + size)
  std::uint32_t constantBase = 0;
  std::uint32_t workgroupWords = 0;   ///< var<workgroup> memory, zeroed for every workgroup
  enum Builtin : std::uint32_t { LocalId, WorkgroupId, GlobalId, NumWorkgroups, LocalIndex };
  struct BuiltinSlot {
    Builtin which;
    std::uint32_t slot;
  };
  std::vector<BuiltinSlot> builtins;
  bool hasBarriers = false;
};

/** A buffer of 32-bit words. `written` (if not empty) records which words a dispatch stored. */
struct Buffer {
  std::vector<std::uint32_t> words;
  std::vector<std::uint8_t> written;

  static Buffer ofFloats(const std::vector<float>& v);
  static Buffer ofWords(const std::vector<std::uint32_t>& v);
  static Buffer zeros(std::size_t words, bool trackWrites = false);
  std::vector<float> floats() const;
};

/**
 * Runs workgroups [first, last) of the grid (gx, gy, gz) — linear index x + gx·(y + gy·z) — with
 * `buffers[i]` bound at @binding(i). A dispatch split into ranges across workers gives the same
 * result as one dispatch, for kernels whose workgroups write disjoint words (as on a GPU).
 */
void dispatch(const Shader& shader, const std::vector<Buffer*>& buffers, std::uint32_t gx, std::uint32_t gy, std::uint32_t gz,
              std::size_t first = 0, std::size_t last = static_cast<std::size_t>(-1));

}  // namespace syn::gpu
