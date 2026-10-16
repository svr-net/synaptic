// The fused classroom kernels executed on the CPU in single precision, mirroring the WGSL in
// fused_kernels.cpp statement for statement (same counter-based random numbers, same order of
// operations), so the kernel algorithm can be tested natively and run without a GPU.
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include "syn/gpu/fused_classroom.hpp"

namespace syn::gpu {

namespace {

using f32 = float;
using u32 = std::uint32_t;

u32 pcg(u32 v) {
  const u32 state = v * 747796405u + 2891336453u;
  const u32 word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

f32 draw(u32 s, u32 l, u32 k) {
  const u32 h = pcg(s ^ pcg((l * 2654435769u) ^ pcg(k)));
  return (static_cast<f32>(h >> 8u) + 0.5f) / 16777216.0f;
}

std::array<f32, 2> gauss2(u32 s, u32 l, u32 k) {
  const f32 u1 = draw(s, l, k), u2 = draw(s, l, k + 1u);
  const f32 a = 6.2831853f * u2;
  const f32 r = std::sqrt(-2.0f * std::log(u1));
  return {r * std::cos(a), r * std::sin(a)};
}

f32 ncdf(f32 x) {
  const f32 z = std::fabs(x) * 0.70710678f;
  const f32 t = 1.0f / (1.0f + 0.3275911f * z);
  const f32 poly = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
  const f32 erf = 1.0f - poly * std::exp(-z * z);
  return x >= 0.0f ? 0.5f * (1.0f + erf) : 0.5f * (1.0f - erf);
}

f32 sigma(f32 x) { return 1.0f / (1.0f + std::exp(-x)); }
f32 c01(f32 x) { return std::min(1.0f, std::max(0.0f, x)); }

// Pure elliptic CGA on the circle of tasks, as in the kernel: vectors (n, m, e), e² = −1.
struct V3 {
  f32 x, y, z;
};
f32 cgaDot(V3 a, V3 b) { return a.x * b.x + a.y * b.y - a.z * b.z; }
V3 taskPoint(f32 delta) {
  const f32 phi = 2.0f * std::atan(std::exp(0.5f * delta));
  return {std::cos(phi), std::sin(phi), 1.0f};
}
V3 dilate(V3 x, f32 beta) {
  const f32 ch = std::cosh(beta), sh = std::sinh(beta);
  const V3 y{x.x * ch - x.z * sh, x.y, x.z * ch - x.x * sh};
  return {y.x / y.z, y.y / y.z, 1.0f};
}
const V3 kLearner{1.0f, 0.0f, 1.0f};
f32 success(V3 t) { return 1.0f + 0.5f * cgaDot(t, kLearner); }
bool insideCap(V3 t, f32 p) { return cgaDot(t, V3{1.0f, 0.0f, 1.0f - 2.0f * (1.0f - p)}) >= 0.0f; }

f32 retrievability(f32 t, f32 stability, f32 last) {
  if (t <= last) return 1.0f;
  return 1.0f / std::sqrt(1.0f + (19.0f / 81.0f) * (t - last) / stability);
}

// One learner of one job: writes its statistics into `out` (statsPerJob floats).
void runLearner(const FusedPlan& p, u32 job, u32 l, f32* out) {
  const auto& C = p.params;
  auto P = [&](u32 i) { return C[i]; };
  const u32 nS = static_cast<u32>(p.sessions), nD = static_cast<u32>(p.numDays());
  const u32 seed = p.seeds[job];
  const f32* J = p.jobs.data() + job * kJobStride;
  const f32 reinforcement = J[0], direction = J[1], empathy = J[2], choice = J[3], relevance = J[4], transparency = J[5],
            prior = J[6], retrieval = J[7], spacing = J[8], interleaving = J[9], feedback = J[10], problemFirst = J[11],
            scaffolding = J[12], dialogue = J[13], inclusivity = J[14], growthFeedback = J[15];

  const auto z01 = gauss2(seed, l, 0u), z23 = gauss2(seed, l, 2u), z45 = gauss2(seed, l, 4u), z67 = gauss2(seed, l, 6u);
  const f32 age = std::max(4.0f, P(0) + P(1) * z01[0]);
  const f32 capital = c01(P(12) + P(13) * z01[1]);
  f32 theta = P(2) + P(3) * z23[0] + P(14) * (capital - P(12));
  f32 anxiety = c01(P(4) + P(5) * z23[1]);
  f32 growth = c01(P(6) + P(7) * z45[0]);
  f32 efficacy = c01(P(8) + P(9) * z45[1]);
  f32 intrinsic = c01(P(10) + P(11) * z67[0]);
  const f32 physiological = c01(0.85f + 0.1f * z67[1]);
  const f32 start = theta;
  const f32 adult = c01((age - 12.0f) / 10.0f);
  const f32 formal = ncdf((age - P(35)) / P(34));
  const f32 abstractFit = 1.0f - P(18) * (1.0f - formal) * (1.0f - 0.5f * scaffolding);
  const f32 access = 1.0f - 0.5f * (1.0f - capital) * (1.0f - inclusivity);
  f32 habit = 0.0f, discrimination = 0.0f, persistence = 0.0f;
  bool engaged = true;
  std::array<f32, kMaxSessions> learned{}, stab{}, last{};
  u32 units = 0, nextDay = 0;
  f32 endScore = 0.0f, delayed = 0.0f;

  for (u32 s = 0; s <= nS; ++s) {
    const f32 t = s == nS ? 1e30f : P(15) * static_cast<f32>(s);
    while (nextDay < nD) {
      const f32 day = p.dayTable[nextDay];
      if (!(day < t - 1e-4f)) break;
      f32 rec = 0.0f;
      for (u32 k = 0; k < units; ++k) rec = rec + learned[k] * retrievability(day, stab[k], last[k]);
      rec = rec / static_cast<f32>(nS);
      const f32 skill = sigma(theta - P(20));
      const f32 pc = 1.0f / 3.0f + 2.0f / 3.0f * (1.0f - std::exp(-(0.6f + discrimination)));
      const f32 score = rec * pc * (0.5f + 0.5f * skill);
      f32* o = out + nextDay * 4u;
      o[0] = score;
      o[1] = score * score;
      o[2] = rec;
      o[3] = score < P(19) ? 1.0f : 0.0f;
      if (nextDay == p.endDayIndex) endScore = score;
      if (nextDay == p.testDayIndex) delayed = score;
      ++nextDay;
    }
    if (s == nS) break;

    const u32 k0 = 16u + s * 24u;
    std::array<f32, 15> u{};
    for (u32 j = 0; j < 15u; ++j) u[j] = draw(seed, l, k0 + j);
    const f32 noise = gauss2(seed, l, k0 + 16u)[0];

    const f32 safety0 = c01(0.45f + 0.25f * capital - 0.2f * anxiety);
    const f32 belonging0 = c01(0.35f + 0.35f * capital);
    const f32 safety = safety0 + c01(0.8f * empathy) * (1.0f - safety0);
    const f32 belonging = belonging0 + c01(0.5f * inclusivity + 0.3f * dialogue + 0.3f * empathy) * (1.0f - belonging0);
    const f32 readiness = sigma((physiological - P(29)) / P(30)) * sigma((safety - P(29)) / P(30)) * sigma((belonging - P(29)) / P(30));

    anxiety = c01(anxiety + 0.08f * problemFirst * (1.0f - growth) * (1.0f - empathy));
    anxiety = anxiety * (1.0f - 0.25f * empathy);
    const f32 clarity = 1.0f / (1.0f + 0.5f * anxiety);

    intrinsic = std::max(0.0f, intrinsic - 0.01f * reinforcement * intrinsic);
    const f32 value = std::min(1.2f, std::max(0.0f, 0.2f + 0.35f * intrinsic + relevance * (0.15f + 0.2f * adult) +
                                                         transparency * (0.05f + 0.15f * adult) + choice * (0.05f + 0.15f * adult) +
                                                         reinforcement * (0.05f + 0.3f * (1.0f - adult)) - 0.1f * adult * direction));
    const f32 motivation = value * efficacy;
    if (engaged) habit = habit + 0.1f * reinforcement * (1.0f - habit);
    const f32 effort = engaged ? c01(0.35f + motivation + 0.15f * habit + persistence) : 0.1f;

    const f32 level = P(16) + P(17) * static_cast<f32>(s);
    const f32 b = (1.0f - scaffolding) * level + scaffolding * (theta + 0.75f);
    const f32 support = 1.5f * scaffolding + 0.4f * direction + 0.8f * dialogue * sigma(-theta + 0.5f);
    const V3 task = taskPoint(b - theta);
    const V3 helped = dilate(task, -0.5f * support);
    const f32 pAlone = success(task), pAssisted = success(helped);
    const bool inZpd = !insideCap(task, P(31)) && insideCap(helped, P(32));

    const f32 struggle = 1.0f + problemFirst * (0.5f * (growth + empathy) - 0.35f);
    const f32 drive = effort * clarity * (0.6f + 0.4f * readiness) * access * abstractFit;
    theta = theta + 0.9f * drive * pAssisted * (1.0f - pAlone) * (1.0f + 0.3f * dialogue) * (1.0f + 0.2f * feedback) * struggle *
                        std::max(0.0f, 1.0f + 0.15f * noise);

    learned[s] = c01(2.2f * drive * (0.5f + 0.5f * pAssisted) * (0.75f + 0.25f * direction) * (1.0f + 0.1f * transparency) *
                     (1.0f + 0.15f * feedback));
    stab[s] = P(21) * (1.0f + 0.6f * prior + P(28) * problemFirst + 0.4f * relevance + 0.3f * dialogue);
    last[s] = P(15) * static_cast<f32>(s);
    units = s + 1u;
    for (u32 k = 0; k < 3u; ++k) {
      const f32 uTarget = u[3 + 4 * k], uMode = u[4 + 4 * k], uRecall = u[5 + 4 * k], uFeedback = u[6 + 4 * k];
      u32 target = s;
      if (s > 0u && uTarget < spacing) target = std::min(s - 1u, static_cast<u32>(uTarget / spacing * static_cast<f32>(s)));
      const f32 at = P(15) * static_cast<f32>(s) + 0.01f * static_cast<f32>(k + 1u);
      const f32 r = retrievability(at, stab[target], last[target]);
      if (uMode < retrieval) {
        if (uRecall < r) {
          stab[target] = stab[target] * (1.0f + P(24) + P(25) * (1.0f - r));
          last[target] = at;
        } else if (uFeedback < feedback) {
          stab[target] = std::max(P(27), P(26) * stab[target]) * (1.0f + P(22) + P(23) * (1.0f - r));
          last[target] = at;
          learned[target] = c01(learned[target] + 0.15f * (1.0f - learned[target]));
        }
      } else {
        stab[target] = stab[target] * (1.0f + P(22) + P(23) * (1.0f - r));
        last[target] = at;
      }
      if (effort > 0.3f) learned[target] = c01(learned[target] + 0.03f * (1.0f - learned[target]));
    }
    discrimination = discrimination + 0.03f + 0.25f * interleaving;

    const bool succeeded = u[0] < pAssisted;
    efficacy = c01(efficacy + (0.1f + 0.1f * feedback) * ((succeeded ? 1.0f : 0.0f) - efficacy));
    if (!succeeded) {
      persistence = std::min(0.2f, std::max(-0.3f, persistence + 0.06f * growth - 0.08f * (1.0f - growth)));
      efficacy = c01(efficacy - 0.04f * (1.0f - growth) * (1.0f - growthFeedback));
      anxiety = c01(anxiety + 0.05f * (1.0f - growth) * (1.0f - empathy));
    } else {
      anxiety = c01(anxiety - 0.03f);
    }
    growth = c01(growth + 0.05f * growthFeedback * (1.0f - growth));
    if (engaged && !succeeded && effort < 0.45f && growth < 0.45f && u[1] < 0.25f * (1.0f - empathy)) engaged = false;
    else if (!engaged && u[2] < 0.3f * empathy + 0.2f * relevance) engaged = true;

    if (p.profiles) {
      f32* o = out + p.sessionOffset() + s * kSessionStats;
      o[0] = theta - start;
      o[1] = effort;
      o[2] = motivation;
      o[3] = value;
      o[4] = efficacy;
      o[5] = anxiety;
      o[6] = readiness;
      o[7] = growth;
      o[8] = engaged ? 1.0f : 0.0f;
      o[9] = inZpd ? 1.0f : 0.0f;
    }
  }
  const f32 high = capital > P(33) ? 1.0f : 0.0f;
  f32* o = out + p.jobOffset();
  o[0] = endScore;
  o[1] = delayed;
  o[2] = theta - start;
  o[3] = delayed * high;
  o[4] = high;
}

}  // namespace

FusedSlice runFusedTiles(const FusedPlan& p, std::size_t tileBegin, std::size_t tileEnd) {
  const std::size_t tiles = p.numJobs * p.numWorkgroups();
  if (tileBegin > tileEnd || tileEnd > tiles) throw std::invalid_argument("runFusedTiles: tile range out of bounds");
  const std::size_t stride = p.statsPerJob();
  FusedSlice slice;
  slice.tileBegin = tileBegin;
  slice.tileEnd = tileEnd;
  slice.partials.assign((tileEnd - tileBegin) * stride, 0.0f);
  if (p.quantiles) slice.scores.assign((tileEnd - tileBegin) * kWorkgroupSize * p.numDays(), 0.0f);
  std::vector<f32> stats(stride);
  for (std::size_t tile = tileBegin; tile < tileEnd; ++tile) {
    const u32 job = static_cast<u32>(tile / p.numWorkgroups());
    const std::size_t wg = tile % p.numWorkgroups();
    // The workgroup reduction sums learner by learner in index order, as the kernel does.
    std::vector<std::vector<f32>> learners;
    for (std::size_t m = 0; m < kWorkgroupSize; ++m) {
      const std::size_t l = wg * kWorkgroupSize + m;
      if (l >= p.numLearners) break;
      std::fill(stats.begin(), stats.end(), 0.0f);
      runLearner(p, job, static_cast<u32>(l), stats.data());
      learners.push_back(stats);
      if (p.quantiles)
        for (std::size_t d = 0; d < p.numDays(); ++d)
          slice.scores[((tile - tileBegin) * kWorkgroupSize + m) * p.numDays() + d] = stats[d * kDayStats];
    }
    f32* dst = slice.partials.data() + (tile - tileBegin) * stride;
    for (std::size_t j = 0; j < stride; ++j) {
      f32 acc = 0.0f;
      for (const auto& x : learners) acc = acc + x[j];
      dst[j] = acc;
    }
  }
  return slice;
}

FusedOutput finishFused(const FusedPlan& p, const std::vector<FusedSlice>& slices) {
  const std::size_t tiles = p.numJobs * p.numWorkgroups(), stride = p.statsPerJob();
  std::size_t covered = 0;
  for (const auto& s : slices) {
    if (s.tileBegin != covered) throw std::invalid_argument("finishFused: slices must cover the tiles in order");
    covered = s.tileEnd;
  }
  if (covered != tiles) throw std::invalid_argument("finishFused: slices do not cover every tile");
  auto partial = [&](std::size_t tile) -> const f32* {
    for (const auto& s : slices)
      if (tile >= s.tileBegin && tile < s.tileEnd) return s.partials.data() + (tile - s.tileBegin) * stride;
    return nullptr;
  };
  FusedOutput out;
  // reduce-partials: per (job, stat), sum the workgroups in order and divide by the learners.
  out.sums.assign(p.numJobs * stride, 0.0f);
  for (std::size_t job = 0; job < p.numJobs; ++job)
    for (std::size_t j = 0; j < stride; ++j) {
      f32 acc = 0.0f;
      for (std::size_t w = 0; w < p.numWorkgroups(); ++w) acc = acc + partial(job * p.numWorkgroups() + w)[j];
      out.sums[job * stride + j] = acc / static_cast<f32>(p.numLearners);
    }
  // score-quantiles: histogram of each (job, day) and the bin centre of the type-7 rank.
  if (p.quantiles) {
    out.quantiles.assign(p.numJobs * p.numDays() * 3, 0.0f);
    std::vector<u32> hist(kHistogramBins);
    for (std::size_t job = 0; job < p.numJobs; ++job)
      for (std::size_t d = 0; d < p.numDays(); ++d) {
        std::fill(hist.begin(), hist.end(), 0u);
        for (std::size_t l = 0; l < p.numLearners; ++l) {
          const std::size_t tile = job * p.numWorkgroups() + l / kWorkgroupSize;
          for (const auto& s : slices)
            if (tile >= s.tileBegin && tile < s.tileEnd) {
              const f32 score = std::min(1.0f, std::max(0.0f, s.scores[((tile - s.tileBegin) * kWorkgroupSize + l % kWorkgroupSize) * p.numDays() + d]));
              ++hist[std::min<u32>(kHistogramBins - 1, static_cast<u32>(score * static_cast<f32>(kHistogramBins)))];
            }
        }
        const f32 qs[3] = {0.1f, 0.5f, 0.9f};
        for (int i = 0; i < 3; ++i) {
          const u32 rank = static_cast<u32>(std::floor(qs[i] * static_cast<f32>(p.numLearners - 1)));
          u32 cum = 0, bin = 0;
          for (u32 b = 0; b < kHistogramBins; ++b) {
            cum += hist[b];
            if (cum > rank) {
              bin = b;
              break;
            }
          }
          out.quantiles[(job * p.numDays() + d) * 3 + i] = (static_cast<f32>(bin) + 0.5f) / static_cast<f32>(kHistogramBins);
        }
      }
  }
  return out;
}

FusedOutput runFusedReference(const FusedPlan& p) {
  return finishFused(p, {runFusedTiles(p, 0, p.numJobs * p.numWorkgroups())});
}

}  // namespace syn::gpu
