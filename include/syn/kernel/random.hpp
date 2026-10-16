#pragma once

// Counter-based random numbers shared by every engine: draw k of agent a under seed s is a PCG
// hash of (s, a, k), the same on the CPU and in WGSL, so every engine sees the same luck and two
// jobs with the same seed see the same learners (common random numbers).

#include <cstdint>

namespace syn::kernel {

inline std::uint32_t pcg(std::uint32_t v) {
  const std::uint32_t state = v * 747796405u + 2891336453u;
  const std::uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

/** The k-th uniform in (0, 1) of agent a under seed s (24-bit resolution, as in WGSL). */
template <class T>
inline T uniform(std::uint32_t s, std::uint32_t a, std::uint32_t k) {
  const std::uint32_t h = pcg(s ^ pcg((a * 2654435769u) ^ pcg(k)));
  return (static_cast<T>(h >> 8u) + T(0.5)) / T(16777216.0);
}

}  // namespace syn::kernel
