#pragma once

// Calendars of instruction: the days on which sessions, reviews and tests happen.
// Times are in days from the first session (day 0).

#include <cstddef>
#include <vector>

namespace syn {

class SessionGrid {
 public:
  SessionGrid() = default;
  explicit SessionGrid(std::vector<double> days);

  /** count sessions every `gap` days from `start`. */
  static SessionGrid uniform(double start, double gap, std::size_t count);
  /** Expanding gaps: start, start + g, start + g + g·r, ... (the classic spaced-review calendar). */
  static SessionGrid expanding(double start, double firstGap, double ratio, std::size_t count);
  /** count sessions crammed together on one day (a tiny offset keeps them ordered). */
  static SessionGrid massed(double day, std::size_t count);
  /** Reporting grid 0, step, 2·step, ..., horizon. */
  static SessionGrid reporting(double horizon, double step);

  /** Sorted union of two grids (duplicates within 1e-9 days removed). */
  SessionGrid merged(const SessionGrid& other) const;

  const std::vector<double>& days() const { return days_; }
  std::size_t size() const { return days_.size(); }
  double operator[](std::size_t i) const { return days_[i]; }
  double last() const;
  /** Index of the last grid day not after t, or -1 if t precedes the grid. */
  long indexAtOrBefore(double t) const;

 private:
  std::vector<double> days_;
};

}  // namespace syn
