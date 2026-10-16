#pragma once

// A classroom point system (token economy) and its gamified variant: students earn points for
// the target behaviour (e.g. coming prepared) and trade them for rewards. Motivation in this
// model is external: engagement on day d is
//   P(engage) = σ(β₀ + I_i + H_i + w·R_d),
// with I_i the student's intrinsic interest, H_i a habit strength built by reinforcement (and
// lost by extinction) and R_d the incentive value of the points while the system runs.
// Rewarding behaviour that is already intrinsically motivated can erode I_i (the
// overjustification effect); the size of that effect is a parameter and defaults to small.

#include <cstdint>
#include <vector>

namespace syn {

struct TokenEconomyParams {
  int students = 30;
  int days = 60;
  int rewardStart = 15;           // first day of the point system
  int rewardEnd = 40;             // first day after it is withdrawn
  double baseline = -0.6;         // β₀
  double intrinsicMean = 0.0;     // I_i ~ N(mean, sd)
  double intrinsicSd = 1.0;
  double incentive = 1.2;         // w·R while rewards run
  double levelBonus = 0.0;        // gamification: extra incentive per level reached (every levelPoints)
  int levelPoints = 10;
  double habitRate = 0.04;        // habit gain per reinforced day
  double habitMax = 1.0;
  double habitDecay = 0.03;       // extinction of habit per unreinforced engaged day
  double undermining = 0.004;     // loss of intrinsic interest per rewarded day (scaled by I_i > 0)
};

struct TokenEconomyRun {
  std::vector<double> engagement;      // mean P(engage) per day
  std::vector<double> engagementLow;   // students with below-median intrinsic interest
  std::vector<double> engagementHigh;  // students with above-median intrinsic interest
  std::vector<double> points;          // mean cumulative points
  std::vector<double> intrinsic;       // mean intrinsic interest
  double before = 0.0, during = 0.0, after = 0.0;  // mean engagement in the three phases
};

TokenEconomyRun simulateTokenEconomy(const TokenEconomyParams& params, std::uint64_t seed);

}  // namespace syn
