#pragma once

// Small numerical helpers shared by every model: logistic link, clamping, sample
// statistics and quantiles.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace syn {

inline double logistic(double x) { return 1.0 / (1.0 + std::exp(-x)); }

inline double logit(double p) {
  if (!(p > 0.0 && p < 1.0)) throw std::invalid_argument("logit: p must lie in (0, 1)");
  return std::log(p / (1.0 - p));
}

inline double clamp(double x, double lo, double hi) { return std::min(hi, std::max(lo, x)); }
inline double clamp01(double x) { return clamp(x, 0.0, 1.0); }

inline double mean(const std::vector<double>& v) {
  if (v.empty()) return 0.0;
  double s = 0.0;
  for (double x : v) s += x;
  return s / static_cast<double>(v.size());
}

/** Sample standard deviation (n − 1 denominator); zero for fewer than two values. */
inline double stdev(const std::vector<double>& v) {
  if (v.size() < 2) return 0.0;
  const double m = mean(v);
  double s = 0.0;
  for (double x : v) s += (x - m) * (x - m);
  return std::sqrt(s / static_cast<double>(v.size() - 1));
}

/** Quantile with linear interpolation between order statistics (Hyndman–Fan type 7). */
inline double quantile(std::vector<double> v, double q) {
  if (v.empty()) throw std::invalid_argument("quantile of an empty sample");
  if (!(q >= 0.0 && q <= 1.0)) throw std::invalid_argument("quantile level must lie in [0, 1]");
  std::sort(v.begin(), v.end());
  const double h = q * static_cast<double>(v.size() - 1);
  const auto lo = static_cast<std::size_t>(std::floor(h));
  const auto hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (h - static_cast<double>(lo)) * (v[hi] - v[lo]);
}

/** n equally spaced points from a to b inclusive. */
inline std::vector<double> linspace(double a, double b, std::size_t n) {
  std::vector<double> out(n);
  for (std::size_t i = 0; i < n; ++i) out[i] = n == 1 ? a : a + (b - a) * static_cast<double>(i) / static_cast<double>(n - 1);
  return out;
}

/** Summary of a sample: mean, standard deviation and the 10/25/50/75/90% quantiles. */
struct Summary {
  double mean = 0.0, sd = 0.0, q10 = 0.0, q25 = 0.0, median = 0.0, q75 = 0.0, q90 = 0.0;
};

inline Summary summarise(const std::vector<double>& v) {
  Summary s;
  if (v.empty()) return s;
  s.mean = mean(v);
  s.sd = stdev(v);
  s.q10 = quantile(v, 0.10);
  s.q25 = quantile(v, 0.25);
  s.median = quantile(v, 0.50);
  s.q75 = quantile(v, 0.75);
  s.q90 = quantile(v, 0.90);
  return s;
}

/** Spearman-free rank agreement: share of pairs (i, j) ordered the same way by a and b. */
inline double pairwiseConcordance(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) throw std::invalid_argument("pairwiseConcordance: size mismatch");
  std::size_t agree = 0, total = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    for (std::size_t j = i + 1; j < a.size(); ++j) {
      if (a[i] == a[j] || b[i] == b[j]) continue;
      ++total;
      if ((a[i] < a[j]) == (b[i] < b[j])) ++agree;
    }
  return total ? static_cast<double>(agree) / static_cast<double>(total) : 0.5;
}

}  // namespace syn
