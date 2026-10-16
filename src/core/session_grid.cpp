#include "syn/core/session_grid.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace syn {

SessionGrid::SessionGrid(std::vector<double> days) : days_(std::move(days)) {
  for (double d : days_)
    if (!std::isfinite(d)) throw std::invalid_argument("SessionGrid: non-finite day");
  std::sort(days_.begin(), days_.end());
}

SessionGrid SessionGrid::uniform(double start, double gap, std::size_t count) {
  if (gap < 0.0) throw std::invalid_argument("SessionGrid::uniform: negative gap");
  std::vector<double> d(count);
  for (std::size_t i = 0; i < count; ++i) d[i] = start + gap * static_cast<double>(i);
  return SessionGrid(std::move(d));
}

SessionGrid SessionGrid::expanding(double start, double firstGap, double ratio, std::size_t count) {
  if (firstGap <= 0.0 || ratio < 1.0) throw std::invalid_argument("SessionGrid::expanding: need firstGap > 0 and ratio >= 1");
  std::vector<double> d;
  double t = start, gap = firstGap;
  for (std::size_t i = 0; i < count; ++i) {
    d.push_back(t);
    t += gap;
    gap *= ratio;
  }
  return SessionGrid(std::move(d));
}

SessionGrid SessionGrid::massed(double day, std::size_t count) {
  std::vector<double> d(count);
  for (std::size_t i = 0; i < count; ++i) d[i] = day + 0.01 * static_cast<double>(i);  // minutes apart
  return SessionGrid(std::move(d));
}

SessionGrid SessionGrid::reporting(double horizon, double step) {
  if (horizon < 0.0 || step <= 0.0) throw std::invalid_argument("SessionGrid::reporting: need horizon >= 0, step > 0");
  std::vector<double> d;
  const auto n = static_cast<std::size_t>(std::floor(horizon / step + 1e-9));
  for (std::size_t i = 0; i <= n; ++i) d.push_back(step * static_cast<double>(i));
  if (horizon - d.back() > 1e-9) d.push_back(horizon);
  return SessionGrid(std::move(d));
}

SessionGrid SessionGrid::merged(const SessionGrid& other) const {
  std::vector<double> d = days_;
  d.insert(d.end(), other.days_.begin(), other.days_.end());
  std::sort(d.begin(), d.end());
  std::vector<double> out;
  for (double x : d)
    if (out.empty() || x - out.back() > 1e-9) out.push_back(x);
  return SessionGrid(std::move(out));
}

double SessionGrid::last() const {
  if (days_.empty()) throw std::logic_error("SessionGrid::last on an empty grid");
  return days_.back();
}

long SessionGrid::indexAtOrBefore(double t) const {
  const auto it = std::upper_bound(days_.begin(), days_.end(), t + 1e-12);
  return static_cast<long>(it - days_.begin()) - 1;
}

}  // namespace syn
