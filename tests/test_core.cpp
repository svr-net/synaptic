#include <cmath>

#include "syn/core/math.hpp"
#include "syn/core/random.hpp"
#include "syn/core/session_grid.hpp"
#include "test_framework.hpp"

using namespace syn;

TEST(rng_is_reproducible_and_streams_differ) {
  Rng a(42), b(42), c = Rng::stream(42, 1), d = Rng::stream(42, 2);
  for (int i = 0; i < 100; ++i) CHECK(a.next() == b.next());
  int same = 0;
  for (int i = 0; i < 100; ++i) same += c.next() == d.next() ? 1 : 0;
  CHECK(same == 0);
}

TEST(rng_moments) {
  Rng rng(7);
  const int n = 200000;
  double su = 0, sn = 0, sn2 = 0;
  for (int i = 0; i < n; ++i) {
    const double u = rng.uniform();
    CHECK(u > 0.0 && u < 1.0);
    su += u;
    const double z = rng.normal();
    sn += z;
    sn2 += z * z;
  }
  CHECK_NEAR(su / n, 0.5, 0.005);
  CHECK_NEAR(sn / n, 0.0, 0.01);
  CHECK_NEAR(sn2 / n, 1.0, 0.015);
  int hits[5] = {0, 0, 0, 0, 0};
  for (int i = 0; i < 50000; ++i) ++hits[rng.uniformInt(5)];
  for (int h : hits) CHECK_NEAR(h / 50000.0, 0.2, 0.01);
}

TEST(logistic_and_logit_are_inverse) {
  for (double p : {0.01, 0.2, 0.5, 0.9, 0.999}) CHECK_NEAR(logistic(logit(p)), p, 1e-12);
  CHECK_THROWS(logit(0.0));
  CHECK_THROWS(logit(1.0));
}

TEST(quantiles_and_summary) {
  const std::vector<double> v = {5, 1, 4, 2, 3};
  CHECK_NEAR(quantile(v, 0.0), 1.0, 1e-12);
  CHECK_NEAR(quantile(v, 0.5), 3.0, 1e-12);
  CHECK_NEAR(quantile(v, 1.0), 5.0, 1e-12);
  CHECK_NEAR(quantile(v, 0.25), 2.0, 1e-12);
  CHECK_NEAR(quantile(v, 0.1), 1.4, 1e-12);
  const Summary s = summarise(v);
  CHECK_NEAR(s.mean, 3.0, 1e-12);
  CHECK_NEAR(s.sd, std::sqrt(2.5), 1e-12);
  CHECK(s.q10 <= s.q25 && s.q25 <= s.median && s.median <= s.q75 && s.q75 <= s.q90);
  CHECK_THROWS(quantile({}, 0.5));
  CHECK_THROWS(quantile(v, 1.5));
}

TEST(pairwise_concordance) {
  CHECK_NEAR(pairwiseConcordance({1, 2, 3, 4}, {10, 20, 30, 40}), 1.0, 1e-12);
  CHECK_NEAR(pairwiseConcordance({1, 2, 3, 4}, {4, 3, 2, 1}), 0.0, 1e-12);
  CHECK_NEAR(pairwiseConcordance({1, 2, 3}, {1, 3, 2}), 2.0 / 3.0, 1e-12);
}

TEST(session_grids) {
  const auto u = SessionGrid::uniform(0, 7, 4);
  CHECK(u.size() == 4);
  CHECK_NEAR(u.last(), 21.0, 1e-12);
  const auto e = SessionGrid::expanding(0, 1, 2, 5);
  CHECK_NEAR(e[1], 1.0, 1e-12);
  CHECK_NEAR(e[4], 15.0, 1e-12);  // 0, 1, 3, 7, 15
  const auto m = SessionGrid::massed(10, 3);
  CHECK(m.size() == 3 && m[2] > m[1] && m[2] - 10 < 0.1);
  const auto r = SessionGrid::reporting(10, 3);
  CHECK(r.size() == 5);  // 0 3 6 9 10
  CHECK_NEAR(r.last(), 10.0, 1e-12);
  const auto merged = u.merged(SessionGrid({3.0, 7.0, 30.0}));
  CHECK(merged.size() == 6);  // 0 3 7 14 21 30
  CHECK(merged.indexAtOrBefore(8.0) == 2);
  CHECK(merged.indexAtOrBefore(-1.0) == -1);
  CHECK_THROWS(SessionGrid::expanding(0, 1, 0.5, 3));
}
