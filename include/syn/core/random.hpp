#pragma once

// Reproducible random numbers: xoshiro256** seeded through splitmix64, with independent
// streams per learner so that simulations can use common random numbers.

#include <cmath>
#include <cstdint>

namespace syn {

/** splitmix64 step: a good mixer for turning (seed, stream) pairs into generator states. */
inline std::uint64_t splitmix64(std::uint64_t& x) {
  std::uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

/** Seed of an independent stream: the same (seed, stream) always gives the same numbers. */
inline std::uint64_t streamSeed(std::uint64_t seed, std::uint64_t stream) {
  std::uint64_t x = seed ^ (0xd1b54a32d192ed03ULL * (stream + 1));
  return splitmix64(x);
}

class Rng {
 public:
  explicit Rng(std::uint64_t seed = 0x5eedULL) {
    std::uint64_t x = seed;
    for (auto& s : s_) s = splitmix64(x);
  }

  /** Generator for stream `stream` of `seed` (one stream per learner, per experiment...). */
  static Rng stream(std::uint64_t seed, std::uint64_t stream) { return Rng(streamSeed(seed, stream)); }

  std::uint64_t next() {
    const std::uint64_t result = rotl(s_[1] * 5, 7) * 9;
    const std::uint64_t t = s_[1] << 17;
    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3] = rotl(s_[3], 45);
    return result;
  }

  /** Uniform on the open interval (0, 1). */
  double uniform() { return (static_cast<double>(next() >> 11) + 0.5) * 0x1.0p-53; }

  double uniform(double lo, double hi) { return lo + (hi - lo) * uniform(); }

  /** Standard normal by the polar-free Box–Muller transform (pairs are cached). */
  double normal() {
    if (hasSpare_) {
      hasSpare_ = false;
      return spare_;
    }
    const double u1 = uniform(), u2 = uniform();
    const double r = std::sqrt(-2.0 * std::log(u1));
    const double a = 6.283185307179586 * u2;
    spare_ = r * std::sin(a);
    hasSpare_ = true;
    return r * std::cos(a);
  }

  double normal(double mean, double sd) { return mean + sd * normal(); }

  bool bernoulli(double p) { return uniform() < p; }

  /** Uniform integer in [0, n). */
  int uniformInt(int n) { return n <= 1 ? 0 : static_cast<int>(uniform() * n) % n; }

 private:
  static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  std::uint64_t s_[4]{};
  bool hasSpare_ = false;
  double spare_ = 0.0;
};

}  // namespace syn
