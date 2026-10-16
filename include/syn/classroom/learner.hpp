#pragma once

// A heterogeneous population of learners. Each learner carries the internal states the
// chapter's theories care about: prior knowledge (cognitivism, constructivism), anxiety
// (humanism, library anxiety), mindset (Dweck), self-efficacy (Svinicki), intrinsic interest
// (humanism, andragogy), cultural capital (Bourdieu) and age (Piaget, andragogy).

#include <cstdint>
#include <string>
#include <vector>

namespace syn {

struct PopulationSpec {
  std::string name = "secondary school (16)";
  double age = 16.0;
  double ageSd = 0.5;
  double priorMean = 0.0, priorSd = 1.0;          // ability θ on a logit scale
  double anxietyMean = 0.4, anxietySd = 0.2;      // [0, 1]
  double growthMean = 0.5, growthSd = 0.25;       // mindset, [0, 1]
  double efficacyMean = 0.55, efficacySd = 0.15;  // [0, 1]
  double intrinsicMean = 0.35, intrinsicSd = 0.2; // [0, 1]
  double capitalMean = 0.55, capitalSd = 0.25;    // match of home and school culture, [0, 1]
  double capitalPrior = 0.5;                      // shift of prior ability per unit of capital

  static PopulationSpec children();     // primary school, age 9
  static PopulationSpec secondary();    // secondary school, age 16
  static PopulationSpec adults();       // adult learners, age 35
};

struct Learner {
  double age = 16.0;
  double ability = 0.0;
  double anxiety = 0.4;
  double growth = 0.5;
  double efficacy = 0.5;
  double intrinsic = 0.35;
  double capital = 0.5;
  double physiological = 0.85;  // basic needs met (Maslow's bottom level)
};

/** Draws one learner from the population with the given generator. */
class Rng;
Learner sampleLearner(const PopulationSpec& spec, Rng& rng);

/** n learners, learner i drawn from stream i of the seed. */
std::vector<Learner> samplePopulation(const PopulationSpec& spec, int n, std::uint64_t seed);

/** 0 for a child, rising to 1 for an adult (andragogy's assumptions hold more strongly). */
double adultness(double age);

}  // namespace syn
