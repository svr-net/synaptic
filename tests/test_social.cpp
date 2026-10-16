#include "syn/core/math.hpp"
#include "syn/social/dialogue.hpp"
#include "syn/social/zpd.hpp"
#include "test_framework.hpp"

using namespace syn;

TEST(zones_of_proximal_development) {
  ZpdParams p;
  CHECK(zoneOf(1.0, -2.0, p) == Zone::Independent);
  CHECK(zoneOf(1.0, 1.5, p) == Zone::Proximal);
  CHECK(zoneOf(1.0, 5.0, p) == Zone::Beyond);
  const auto band = zpdBand(1.0, p);
  CHECK(zoneOf(1.0, band.first - 0.01, p) == Zone::Independent);
  CHECK(zoneOf(1.0, band.first + 0.01, p) == Zone::Proximal);
  CHECK(zoneOf(1.0, band.second - 0.01, p) == Zone::Proximal);
  CHECK(zoneOf(1.0, band.second + 0.01, p) == Zone::Beyond);
  CHECK(successAssisted(0.0, 1.0, p.scaffold) > successAlone(0.0, 1.0));
}

TEST(learning_gain_peaks_inside_the_zpd) {
  ZpdParams p;
  const double theta = 0.0;
  const auto band = zpdBand(theta, p);
  double best = -1.0, bestB = 0.0;
  for (double b = -6.0; b <= 8.0; b += 0.01) {
    const double g = expectedGain(theta, b, p);
    if (g > best) {
      best = g;
      bestB = b;
    }
  }
  CHECK(bestB > band.first && bestB < band.second);
  CHECK_NEAR(bestB, theta + p.scaffold / 2.0, 0.01);  // σ(x + s)·σ(−x) peaks at x = −s/2
  CHECK(expectedGain(theta, band.first - 2.0, p) < 0.5 * best);   // boredom
  CHECK(expectedGain(theta, band.second + 3.0, p) < 0.2 * best);  // frustration
}

TEST(tutoring_in_the_zpd_beats_too_easy_or_too_hard) {
  ZpdParams z;
  TutoringParams t;
  const auto zpd = simulateTutoring(z, t, TaskPolicy::Zpd, 3);
  const auto easy = simulateTutoring(z, t, TaskPolicy::TooEasy, 3);
  const auto hard = simulateTutoring(z, t, TaskPolicy::TooHard, 3);
  const auto cls = simulateTutoring(z, t, TaskPolicy::ClassAverage, 3);
  CHECK(zpd.finalGain > cls.finalGain);
  CHECK(cls.finalGain > easy.finalGain);
  CHECK(easy.finalGain > hard.finalGain);
  CHECK(zpd.zoneShare[static_cast<int>(Zone::Proximal)][0] > 0.9);
  CHECK(hard.zoneShare[static_cast<int>(Zone::Beyond)][0] > 0.99);
  for (std::size_t s = 0; s < zpd.meanAbility.size(); ++s) {
    double total = 0.0;
    for (const auto& z3 : zpd.zoneShare) total += z3[s];
    CHECK_NEAR(total, 1.0, 1e-9);
  }
}

TEST(peer_dialogue_and_inclusive_teaching) {
  DialogueParams base;
  base.grouping = Grouping::None;
  DialogueParams mixed = base;
  mixed.grouping = Grouping::MixedAbility;
  DialogueParams same = base;
  same.grouping = Grouping::SameAbility;
  DialogueParams inclusive = base;
  inclusive.inclusivity = 1.0;
  const auto a = simulateDialogue(base, 9), b = simulateDialogue(mixed, 9), c = simulateDialogue(same, 9),
             d = simulateDialogue(inclusive, 9);
  CHECK(b.finalMean > a.finalMean + 0.05);  // dialogue helps
  CHECK(b.finalMean > c.finalMean);         // more knowledgeable others in every group
  CHECK(b.finalGap < a.finalGap);
  CHECK(a.highCapital.front() > a.lowCapital.front());  // schools reflect wealthier families' culture
  CHECK(a.finalGap > a.highCapital.front() - a.lowCapital.front());  // ...and the gap widens
  CHECK(d.finalGap < 0.5 * a.finalGap);     // inclusive teaching removes most of it
  for (std::size_t s = 1; s < b.meanUnderstanding.size(); ++s) CHECK(b.meanUnderstanding[s] >= b.meanUnderstanding[s - 1] - 1e-12);
}
