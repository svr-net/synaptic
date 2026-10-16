#include "syn/assessment/rubric.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "syn/core/math.hpp"
#include "syn/core/random.hpp"

namespace syn {

const char* rubricKindName(RubricKind k) {
  switch (k) {
    case RubricKind::None: return "no rubric";
    case RubricKind::Vague: return "vague rubric";
    case RubricKind::Quantified: return "quantified rubric";
    case RubricKind::Precise: return "precise rubric";
  }
  return "?";
}

namespace {

struct RaterModel {
  double severity, noise, boundaryJitter, qualityGain, countGain;
};

RaterModel raterModel(RubricKind k) {
  switch (k) {
    case RubricKind::None: return {0.5, 0.6, 0.4, 0.0, 0.0};
    case RubricKind::Vague: return {0.3, 0.45, 0.35, 0.1, 0.0};
    case RubricKind::Quantified: return {0.0, 0.05, 0.0, 0.0, 1.5};
    case RubricKind::Precise: return {0.1, 0.25, 0.08, 0.3, 0.0};
  }
  return {0, 0, 0, 0, 0};
}

double correlation(const std::vector<double>& x, const std::vector<double>& y) {
  const double mx = mean(x), my = mean(y);
  double sxy = 0.0, sxx = 0.0, syy = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    sxy += (x[i] - mx) * (y[i] - my);
    sxx += (x[i] - mx) * (x[i] - mx);
    syy += (y[i] - my) * (y[i] - my);
  }
  return sxx > 0.0 && syy > 0.0 ? sxy / std::sqrt(sxx * syy) : 0.0;
}

}  // namespace

RubricResult simulateRubric(const RubricParams& p, std::uint64_t seed) {
  if (p.works < 2 || p.raters < 2 || p.levels < 2 || p.levels > 10) throw std::invalid_argument("simulateRubric: bad sizes");
  const RaterModel model = raterModel(p.kind);
  const int W = p.works, R = p.raters, L = p.levels;
  // Level boundaries: equally spaced on the quality scale.
  std::vector<double> boundary(L - 1);
  for (int k = 1; k < L; ++k) boundary[k - 1] = -1.2 + 2.4 * k / L;
  // Each rater's private reading of the boundaries, and severity.
  std::vector<std::vector<double>> bounds(R);
  std::vector<double> severity(R);
  Rng raterRng = Rng::stream(seed, 0xBA7E);
  for (int r = 0; r < R; ++r) {
    severity[r] = model.severity * raterRng.normal();
    bounds[r] = boundary;
    for (double& b : bounds[r]) b += model.boundaryJitter * raterRng.normal();
    std::sort(bounds[r].begin(), bounds[r].end());
  }
  std::vector<double> quality(W), count(W);
  std::vector<std::vector<int>> level(R, std::vector<int>(W));
  for (int w = 0; w < W; ++w) {
    Rng rng = Rng::stream(seed, static_cast<std::uint64_t>(w));
    const double q = rng.normal() + (p.shared ? model.qualityGain : 0.0);
    // References: weakly related to quality; a shared count-based rubric adds references.
    const double c = std::max(0.0, std::round(4.0 + 1.2 * (0.35 * q + 0.94 * rng.normal()) + (p.shared ? model.countGain : 0.0)));
    quality[w] = q;
    count[w] = c;
    for (int r = 0; r < R; ++r) {
      const double seen = p.kind == RubricKind::Quantified ? (c - 4.0) / 1.2 + model.noise * rng.normal()
                                                          : q + severity[r] + model.noise * rng.normal();
      int lv = 0;
      for (double b : bounds[r]) lv += seen > b ? 1 : 0;
      level[r][w] = lv;
    }
  }
  RubricResult out;
  out.kind = rubricKindName(p.kind);
  int pairs = 0;
  for (int a = 0; a < R; ++a)
    for (int b = a + 1; b < R; ++b) {
      std::vector<double> ma(L, 0.0), mb(L, 0.0);
      double agree = 0.0;
      for (int w = 0; w < W; ++w) {
        agree += level[a][w] == level[b][w] ? 1.0 : 0.0;
        ma[level[a][w]] += 1.0 / W;
        mb[level[b][w]] += 1.0 / W;
      }
      const double po = agree / W;
      double pe = 0.0;
      for (int k = 0; k < L; ++k) pe += ma[k] * mb[k];
      out.exactAgreement += po;
      out.kappa += pe < 1.0 ? (po - pe) / (1.0 - pe) : 1.0;
      ++pairs;
    }
  out.exactAgreement /= pairs;
  out.kappa /= pairs;
  // Validity of one rater's judgement: does a level mean the same whoever assigns it?
  for (int r = 0; r < R; ++r) {
    std::vector<double> lv(level[r].begin(), level[r].end());
    out.validity += correlation(lv, quality) / R;
  }
  out.meanQuality = mean(quality);
  out.meanCount = mean(count);
  return out;
}

SelfAssessmentResult simulateSelfAssessment(const SelfAssessmentParams& p, std::uint64_t seed) {
  if (p.learners < 8 || p.rounds < 1) throw std::invalid_argument("simulateSelfAssessment: bad sizes");
  const int N = p.learners;
  std::vector<double> skill(N);
  for (int i = 0; i < N; ++i) {
    Rng rng = Rng::stream(seed, static_cast<std::uint64_t>(i));
    skill[i] = logistic(1.2 * rng.normal());
  }
  // Quartiles by starting skill (fixed membership, as in the classic plot).
  std::vector<int> order(N);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) { return skill[a] < skill[b]; });
  std::vector<int> quartile(N);
  for (int k = 0; k < N; ++k) quartile[order[k]] = std::min(3, 4 * k / N);
  double beta = p.regression, flattery = p.flattery;
  SelfAssessmentResult out;
  for (int round = 0; round < p.rounds; ++round) {
    std::array<double, 4> act{}, per{}, cnt{};
    double err = 0.0, total = 0.0;
    for (int i = 0; i < N; ++i) {
      Rng rng = Rng::stream(seed ^ 0x5E1Full, static_cast<std::uint64_t>(round) * 100003ull + static_cast<std::uint64_t>(i));
      const double a = skill[i];
      const double s = clamp01(a + beta * (0.6 - a) + flattery + p.noise * rng.normal());
      act[quartile[i]] += a;
      per[quartile[i]] += s;
      cnt[quartile[i]] += 1.0;
      err += std::fabs(s - a) / N;
      total += a / N;
    }
    for (int q = 0; q < 4; ++q) {
      act[q] /= std::max(1.0, cnt[q]);
      per[q] /= std::max(1.0, cnt[q]);
    }
    out.actual.push_back(act);
    out.perceived.push_back(per);
    out.calibrationError.push_back(err);
    out.meanSkill.push_back(total);
    // Reflection, and feedback on the self-assessment.
    const double gain = p.reflection * (p.feedback ? 2.0 : 1.0);
    for (double& a : skill) a += gain * (1.0 - a);
    if (p.feedback) {
      beta *= 1.0 - p.feedbackShrink;
      flattery *= 1.0 - p.feedbackShrink;
    }
  }
  return out;
}

}  // namespace syn
