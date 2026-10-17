// Every model of the library, run as its fused kernel on the CPU in 64-bit floats: the results
// reproduce what the chapters describe. (test_space checks that the 32-bit engines agree.)
#include <cmath>
#include <string>

#include "syn/core/value.hpp"
#include "syn/kernel/kernel.hpp"
#include "test_framework.hpp"

using namespace syn;

namespace {

Value spec(int learners = 1000, int seed = 7) {
  Value s = Value::object();
  Value sim = Value::object();
  sim.set("learners", learners).set("seed", seed);
  s.set("sim", sim);
  return s;
}

Value run(const std::string& name, const Value& s = spec()) { return kernel::run(name, s); }

double num(const Value& v, const std::string& key) { return v.num(key, std::nan("")); }
double at(const Value& v, const std::string& key, std::size_t i) { return v.at(key)[i].asNumber(); }
const Value& item(const Value& v, const std::string& key, std::size_t i) { return v.at(key).asArray().at(i); }

bool finiteEverywhere(const Value& v) {
  if (v.isNumber()) return std::isfinite(v.asNumber());
  if (v.isNumbers()) {
    for (double x : v.asNumbers())
      if (!std::isfinite(x)) return false;
    return true;
  }
  if (v.isArray()) {
    for (const auto& x : v.asArray())
      if (!finiteEverywhere(x)) return false;
    return true;
  }
  if (v.isObject()) {
    for (const auto& [key, x] : v.asObject())
      if (key != "recall" && !finiteEverywhere(x)) return false;  // recall has no quantiles (NaN)
    return true;
  }
  return true;
}

}  // namespace

TEST(every_model_runs_and_is_finite) {
  CHECK(kernel::models().size() == 39);
  for (const kernel::Model* m : kernel::models()) {
    const Value r = run(m->name, spec(200));
    if (!finiteEverywhere(r)) ::syntest::fail("non-finite value in " + m->name, __FILE__, __LINE__);
  }
}

TEST(every_model_is_reproducible_and_seeded) {
  for (const char* name : {"operant", "classroom", "rubric"}) {
    CHECK(run(name, spec(200, 3)).json() == run(name, spec(200, 3)).json());
    CHECK(run(name, spec(200, 3)).json() != run(name, spec(200, 4)).json());
  }
}

// ------------------------------------------------------------------ behaviourism

TEST(conditioning_pavlov_blocking_and_generalised_fear) {
  const Value r = run("conditioning");
  CHECK(at(r, "pavlov", 29) > 0.95);                       // acquired after 30 pairings
  CHECK(r.at("pavlov").size() == 60);
  CHECK(r.at("pavlov")[59].asNumber() < 0.1);              // extinguished by the bell alone
  CHECK(num(r.sub("blocking"), "blockedTone") < 0.1);      // Kamin: the light blocks the tone
  CHECK(num(r.sub("blocking"), "controlToneFinal") > 0.4);
  double previous = 2;
  for (const auto& probe : r.at("fear").asArray()) {       // fear falls off with distance
    CHECK(num(probe, "fear") <= previous);
    previous = num(probe, "fear");
  }
  for (std::size_t t = 0; t < 60; ++t) CHECK_NEAR(at(r, "pavlov", t), at(r, "closedForm", t), 1e-6);
}

TEST(operant_partial_reinforcement_extinction_effect) {
  const Value r = run("operant");
  const Value& crf = item(r, "runs", 0);
  const Value& vr = item(r, "runs", 2);
  CHECK(num(crf, "acquisitionSteps") < num(vr, "acquisitionSteps"));  // continuous: fastest to learn
  CHECK(num(vr, "resistance") > 2 * num(crf, "resistance"));         // variable ratio: slowest to extinguish
}

TEST(token_economy_undermines_interest_after_withdrawal) {
  const Value r = run("tokenEconomy");
  CHECK(num(r, "during") > num(r, "before") + 0.1);
  CHECK(num(r, "after") < num(r, "during"));
}

// ------------------------------------------------------------------ humanism

TEST(maslow_climate_releases_higher_needs) {
  const Value r = run("maslow");
  CHECK(r.str("dominant", "") == "safety");
  CHECK(r.str("dominantSupported", "") == "self-actualisation");
  CHECK(at(r, "activationSupported", 4) > 5 * at(r, "activation", 4));
}

TEST(information_search_support_helps_finish) {
  const Value r = run("isp");
  CHECK(r.at("isp").size() == 3);
  CHECK(num(item(r, "isp", 2), "completedOnTime") > num(item(r, "isp", 0), "completedOnTime") + 0.2);
  CHECK(at(item(r, "isp", 2), "uncertainty", 0) < at(item(r, "isp", 0), "uncertainty", 0));
}

// ------------------------------------------------------------------ cognitivism

TEST(memory_spaced_retrieval_beats_massed_restudy) {
  const Value r = run("memory");
  const double massed = num(item(r, "plans", 0), "recallAtTest");
  const double spacedRetrieval = num(item(r, "plans", 3), "recallAtTest");
  CHECK(spacedRetrieval > massed + 0.2);
  CHECK(num(item(r, "plans", 1), "recallAtTest") > massed);  // the testing effect
  CHECK(num(item(r, "plans", 2), "recallAtTest") > massed);  // the spacing effect
}

TEST(interleaving_lowers_practice_but_raises_the_test) {
  const Value r = run("interleaving");
  CHECK(num(r.sub("interleaved"), "practiceMean") < num(r.sub("blocked"), "practiceMean"));
  CHECK(num(r.sub("interleaved"), "test") > num(r.sub("blocked"), "test") + 0.2);
}

TEST(schema_prior_knowledge_connects_the_new_concept) {
  const Value r = run("schema");
  CHECK(num(r.sub("schemaPrimed"), "degree") > num(r.sub("schemaPlain"), "degree"));
  CHECK(at(r.sub("schemaPrimed"), "recallFrom", 0) > at(r.sub("schemaPlain"), "recallFrom", 0));
}

// ------------------------------------------------------------------ constructivism and social

TEST(equilibration_feedback_drives_accommodation) {
  const Value r = run("equilibration");
  CHECK(num(item(r, "runs", 1), "finalAccuracy") > num(item(r, "runs", 0), "finalAccuracy") + 0.3);
  CHECK(num(item(r, "runs", 1), "accommodations") > num(item(r, "runs", 0), "accommodations"));
  CHECK(num(item(r, "runs", 3), "finalAccuracy") < num(item(r, "runs", 1), "finalAccuracy"));  // avoiding discomfort
}

TEST(zpd_targeted_tasks_teach_most) {
  const Value r = run("zpd");
  const double zpd = num(item(r, "tutoring", 3), "gain");
  for (std::size_t i = 0; i < 3; ++i) CHECK(zpd > num(item(r, "tutoring", i), "gain"));
  CHECK(num(item(r, "tutoring", 2), "gain") < num(item(r, "tutoring", 0), "gain"));  // too hard is worst
}

TEST(dialogue_and_inclusivity_close_the_gap) {
  const Value r = run("dialogue");
  const Value& alone = item(r, "runs", 0);
  const Value& groups = item(r, "runs", 3);
  CHECK(num(groups, "finalMean") > num(alone, "finalMean"));
  CHECK(num(groups, "finalGap") < num(alone, "finalGap"));
}

TEST(development_stages_and_perry) {
  const Value r = run("development");
  CHECK(r.at("stages").size() == 4);
  CHECK(r.at("perryNames").size() == 4);
  // A challenging course moves more students beyond dualism than an unchallenging one.
  const auto& plain = r.at("perryPlain").asArray(), &challenged = r.at("perryChallenged").asArray();
  const std::size_t last = plain[0].size() - 1;
  CHECK(challenged[2][last].asNumber() > plain[2][last].asNumber());
}

TEST(andragogy_input_improves_the_needs_cycle) {
  const Value r = run("andragogy");
  CHECK(at(item(r, "cycles", 2), "hitRate", 0) > at(item(r, "cycles", 0), "hitRate", 0));
  const Value& cycle = item(r, "cycles", 2);
  CHECK(at(cycle, "gap", 6) < at(cycle, "gap", 0));
}

TEST(motivation_levers_and_growth_feedback) {
  const Value r = run("motivation");
  const double none = num(item(r, "terms", 0), "gain");
  for (std::size_t i : {1, 2, 3, 4}) CHECK(num(item(r, "terms", i), "gain") > none);
  for (std::size_t i = 0; i + 1 < r.at("terms").size(); ++i) CHECK(num(item(r, "terms", 6), "gain") > num(item(r, "terms", i), "gain"));  // all five
  CHECK(num(item(r, "mindset", 2), "persist") > num(item(r, "mindset", 0), "persist"));
  CHECK(num(item(r, "mindset", 2), "gain") > num(item(r, "mindset", 0), "gain"));
}

// ------------------------------------------------------------------ the classroom

TEST(classroom_blended_design_teaches_and_retains) {
  const Value r = run("classroom");
  CHECK(num(r, "endScore") > 0.6);
  CHECK(num(r, "delayedScore") > 0.4);
  CHECK(num(r, "delayedScore") < num(r, "endScore"));
  CHECK(num(r, "equityGap") < 0.05);
}

TEST(theories_blended_beats_the_banking_lecture_everywhere) {
  const Value r = run("theories", spec(400));
  CHECK(r.at("designs").size() == 8);
  CHECK(num(item(r, "designs", 0), "teacherDirection") == 1);
  for (const auto& population : r.at("populations").asArray()) {
    const auto& results = population.at("results").asArray();
    CHECK(num(results[7], "delayedScore") > num(results[0], "delayedScore") + 0.2);
  }
}

TEST(allocation_shapley_shares_add_up) {
  const Value r = run("allocation", spec(400));
  double sum = 0;
  for (const auto& lever : r.at("levers").asArray()) sum += num(lever, "shapley");
  CHECK_NEAR(sum, num(r, "designScore") - num(r, "baselineScore"), 1e-9);
  CHECK(num(r, "designScore") > num(r, "baselineScore"));
}

TEST(sensitivity_common_random_numbers_reduce_variance) {
  const Value r = run("sensitivity", spec(300));
  CHECK(num(r, "varianceReduction") > 10);
  for (const auto& lever : r.at("levers").asArray()) CHECK(num(lever, "crnSe") < num(lever, "independentSe"));
}

TEST(gagne_leaving_out_an_event_costs) {
  const Value r = run("gagne", spec(400));
  const double full = num(r, "full");
  CHECK(full > num(r, "presentOnly"));
  for (const auto& event : r.at("events").asArray()) CHECK(num(event, "withoutIt") <= full + 0.01);
}

// ------------------------------------------------------------------ chapter 7

TEST(audience_pre_assessment_improves_estimates) {
  const Value r = run("audience");
  const Value& none = item(r, "instruments", 0);
  const Value& poll = item(r, "instruments", 2);
  CHECK(at(poll, "rmse", 0) < at(none, "rmse", 0));
  CHECK(num(poll, "teachingMinutes") < num(none, "teachingMinutes"));  // it costs class time
}

TEST(community_labels_explain_little_pre_assessment_much) {
  const Value r = run("community");
  CHECK(num(r, "bestSingleR2") < 0.1);
  CHECK(num(r, "intersectionR2") >= num(r, "bestSingleR2"));
  CHECK(num(r, "preAssessmentR2") > 0.5);
  CHECK(num(r, "combinedR2") >= num(r, "preAssessmentR2") - 1e-9);
}

// ------------------------------------------------------------------ chapter 8

TEST(outcomes_mastery_falls_with_bloom_level) {
  const Value r = run("outcomes");
  CHECK(r.at("levels").size() == 6);
  for (std::size_t i = 1; i < 6; ++i) CHECK(at(r, "levelMastery", i) < at(r, "levelMastery", i - 1));
  CHECK(num(item(r, "plans", 0), "coverage") == 1);
}

// ------------------------------------------------------------------ chapter 9

TEST(assessment_direct_evidence_tracks_learning_best) {
  const Value r = run("assessmentActivities");
  const Value& s = r.sub("signals");
  CHECK(num(s, "direct") > num(s, "grade"));
  CHECK(num(s, "grade") > num(s, "satisfaction"));
  CHECK(num(s, "selfReport") > num(s, "satisfaction"));
}

TEST(assessment_formative_checks_and_the_cycle) {
  const Value r = run("assessmentSession");
  CHECK(num(item(r, "sessions", 1), "estimateError") < num(item(r, "sessions", 0), "estimateError"));
  const Value& direct = item(r, "cycle", 3);
  const Value& none = item(r, "cycle", 0);
  CHECK(at(direct, "mastery", 7) > at(none, "mastery", 7));
}

TEST(rubric_precision_raises_agreement_and_feedback_calibrates) {
  const Value r = run("rubric");
  const Value& none = item(r, "rubrics", 0).sub("private");
  const Value& precise = item(r, "rubrics", 3).sub("private");
  CHECK(num(precise, "kappa") > num(none, "kappa") + 0.2);
  CHECK(num(precise, "validity") > num(none, "validity"));
  const Value& quantified = item(r, "rubrics", 2).sub("private");
  CHECK(num(quantified, "validity") < num(none, "validity"));  // counting is reliable, not valid
  CHECK(at(item(r, "selfAssessment", 1), "calibrationError", 3) < at(item(r, "selfAssessment", 0), "calibrationError", 3));
}

// ------------------------------------------------------------------ chapter 10

TEST(lecture_chunks_hold_attention_and_stories_and_analogies_help) {
  const Value r = run("lecture");
  const Value& straight = item(r, "sessions", 0);
  const Value& twelve = item(r, "sessions", 3);
  const auto& a = straight.at("attention").asNumbers();
  CHECK(a.back() < 0.5 * a.front());                                 // attention fades in a long lecture
  CHECK(num(twelve, "learned") > num(straight, "learned") + 0.05);   // chunks of 12-15 minutes
  CHECK(num(twelve, "retained") > num(straight, "retained"));
  CHECK(num(item(r, "sessions", 5), "retained") > num(twelve, "retained"));  // a story deepens the trace
  CHECK(num(item(r, "sessions", 6), "learned") > num(twelve, "learned"));    // an analogy connects
  const auto& videos = r.at("videos").asArray();
  for (std::size_t k = 1; k < videos.size(); ++k) CHECK(num(videos[k], "completed") < num(videos[k - 1], "completed"));
  CHECK(num(videos[0], "completed") > 0.8);                          // short videos are watched to the end
  CHECK(num(videos.back(), "completed") < 0.1);
}

TEST(discussion_wait_time_structures_and_managing_talkers) {
  const Value r = run("discussion", spec(300));
  const auto& waits = r.at("waitTime").asArray();
  CHECK(num(waits.back(), "voices") > num(waits.front(), "voices") + 0.05);      // silence lets more voices in
  CHECK(num(waits.back(), "moreTimeVoices") > num(waits.front(), "moreTimeVoices") + 0.1);
  const Value& whole = item(r, "designs", 0);
  CHECK(num(whole, "topFiveShare") > 0.75);                                      // a few people do most of the talking
  CHECK(num(item(r, "designs", 1), "articulated") > 0.99);                      // think-pair-share: everyone articulates
  CHECK(num(item(r, "designs", 1), "learned") > num(whole, "learned"));
  CHECK(num(item(r, "designs", 5), "voices") > num(whole, "voices"));          // three chips each
  CHECK(num(item(r, "designs", 5), "topFiveShare") < num(whole, "topFiveShare"));
  CHECK(num(item(r, "designs", 7), "learned") < num(whole, "learned"));        // closed questions
}

TEST(scaffolding_gradual_release_builds_independence) {
  const Value r = run("scaffolding");
  const Value& alone = item(r, "sequences", 0);
  const Value& gradual = item(r, "sequences", 3);
  CHECK(num(gradual, "transfer") > num(alone, "transfer") + 0.05);
  CHECK(num(gradual, "transfer") > num(item(r, "sequences", 2), "transfer"));  // support must fade
  CHECK(num(alone, "gaveUp") > num(gradual, "gaveUp") + 0.2);                  // zone 3 frustrates
  CHECK(num(item(r, "projects", 2), "quality") > 2 * num(item(r, "projects", 0), "quality"));  // stages with feedback
}

TEST(strategies_content_not_learning_styles_and_variety_for_access) {
  const Value r = run("strategies");
  for (const auto& c : r.at("styles").asArray()) {
    const auto& rules = c.at("rules").asArray();
    CHECK(num(rules[2], "knowledge") >= num(rules[1], "knowledge"));            // content beats stated style
    CHECK(num(rules[1], "satisfaction") > num(rules[2], "satisfaction"));       // a preference only pleases
    CHECK(num(rules[3], "barredKnowledge") > num(rules[2], "barredKnowledge"));  // variety reaches everyone
  }
  const auto& f = r.at("flipped").asArray();
  CHECK(num(f[4], "prepared") > num(f[1], "prepared") + 0.3);                   // accountability and embedding
  CHECK(num(f[4], "higherOrder") > num(f[0], "higherOrder"));
  CHECK(num(f[1], "unpreparedHigherOrder") < num(f[0], "higherOrder"));         // unprepared learners fall behind
  const auto& pbl = r.at("pbl").asArray();
  CHECK_NEAR(num(pbl[1], "basics"), num(pbl[0], "basics"), 0.02);               // similar outcomes
  CHECK(num(pbl[1], "interest") > num(pbl[0], "interest"));                     // higher satisfaction
}

TEST(lesson_plan_time_stamps_cut_lists_and_backups) {
  const Value r = run("lessonPlan", spec(1000));
  const auto& d = r.at("designs").asArray();
  CHECK(r.at("plan").size() == 6);
  CHECK(num(d[1], "onTime") > num(d[0], "onTime") + 0.5);    // a timed plan keeps the wrap-up and reflection
  CHECK(num(d[2], "onTime") >= num(d[1], "onTime"));         // a cut list
  CHECK(num(d[2], "learning") > num(d[1], "learning"));      // cutting beats rushing
  CHECK(num(d[8], "spareMinutes") < num(d[6], "spareMinutes"));  // a backup uses spare minutes
  CHECK(num(d[5], "onTime") > num(d[0], "onTime"));          // experience
}

TEST(reflection_structured_evidence_and_one_change_at_a_time) {
  const Value r = run("reflection", spec(1000));
  const auto& d = r.at("designs").asArray();
  CHECK(num(d[1], "final") > num(d[0], "final"));                        // even informal reflection helps a little
  CHECK(num(d[2], "final") > num(d[1], "final") + 0.1);                  // the four-step process, applied
  for (int k : {3, 4, 5}) CHECK(num(d[k], "blindSpots") > num(d[2], "blindSpots"));  // evidence sees what you cannot
  CHECK(num(d[5], "rightTarget") > num(d[2], "rightTarget") + 0.2);      // the learners' perspective, not guesses
  CHECK(num(d[8], "final") < num(d[2], "final") - 0.05);                 // change one or two things, not everything
  CHECK(num(d[9], "final") < num(d[2], "final"));                        // reflect soon after the session
  CHECK(num(d[10], "efficacy") < num(d[2], "efficacy") - 0.3);           // be kind to yourself
  CHECK(num(d[10], "skipped") > 1);
}

TEST(student_feedback_patterns_not_single_comments_and_biased_ratings) {
  const Value r = run("studentFeedback");
  const auto& d = r.at("rules").asArray();
  CHECK(num(d[0], "falseRevisions") > 5 * num(d[1], "falseRevisions"));  // one or two comments are not a pattern
  CHECK(num(d[1], "after") > num(d[0], "after"));
  CHECK(num(d[1], "caught") > 0.9);
  CHECK(num(d[2], "caught") < 0.3);                                      // a rating does not say what to fix
  CHECK(num(d[3], "caught") < num(d[1], "caught"));                      // anonymity
  CHECK(std::abs(num(d[0], "ratingLearning")) < 0.3);                    // ratings are weak indicators of learning
  CHECK(num(d[4], "rating") < num(d[1], "rating") - 0.15);               // bias, with the same teaching
  CHECK_NEAR(num(d[4], "after"), num(d[1], "after"), 1e-9);
  CHECK(num(d[5], "falseRevisions") > num(d[2], "falseRevisions"));
}

TEST(critical_reflection_assumptions_hold_for_some_learners) {
  const Value r = run("criticalReflection", spec(4000));
  const auto& d = r.at("designs").asArray();
  auto group = [&](int j, int g, const char* key) { return num(d[j].at("groups").asArray()[g], key); };
  CHECK(group(1, 0, "comfort") > group(0, 0, "comfort"));               // a circle invites the confident...
  CHECK(group(1, 1, "comfort") < group(0, 1, "comfort"));               // ...and exposes the shy
  CHECK(group(2, 1, "comfort") > group(1, 1, "comfort"));               // explaining the choice
  CHECK(num(d[4], "comfort") > num(d[0], "comfort"));                   // a "pass"
  CHECK(num(d[4], "spoke") > num(d[3], "spoke"));
  CHECK(num(d[6], "questioned") > num(d[5], "questioned") + 0.1);       // openness shown, not just stated
  CHECK(group(7, 0, "learned") > group(0, 0, "learned"));               // maximum choice suits the self-directed
  CHECK(group(7, 4, "learned") < group(0, 4, "learned") - 0.1);         // and loses the novices
  CHECK(num(d[10], "lowestGroup") > num(d[0], "lowestGroup") + 0.1);
}

TEST(professional_growth_communities_adapting_and_beginners_mind) {
  const Value r = run("professionalGrowth");
  const auto& d = r.at("designs").asArray();
  CHECK(num(d[1], "efficacy") > num(d[0], "efficacy") + 0.2);           // a community of practice supports
  CHECK(num(d[1], "drainedMonths") < num(d[0], "drainedMonths"));
  CHECK(num(d[3], "gain") > num(d[2], "gain"));                         // adapt, do not copy
  CHECK(num(d[2], "backfired") > 0.5);
  for (int k : {4, 5}) CHECK(num(d[k], "gain") > num(d[0], "gain"));    // courses and the literature
  CHECK(num(d[6], "blindSpot") < num(d[0], "blindSpot") - 0.3);          // becoming a student again
  CHECK(num(d[7], "gain") > num(d[1], "gain"));
}

TEST(fga_algebra_claims_checked_in_the_space) {
  const Value r = run("fgaAlgebra");
  const auto& claims = r.at("claims").asArray();
  CHECK(claims.size() == 24);
  auto holds = [&](std::size_t i) { return claims[i].at("holds").asBool(); };
  for (std::size_t i : {0u, 1u, 2u, 3u, 5u, 6u, 9u, 11u, 12u, 13u, 14u, 15u, 16u, 17u, 18u, 21u, 22u, 23u}) CHECK(holds(i));  // exact
  for (std::size_t i : {4u, 7u, 8u, 10u, 19u, 20u}) CHECK(!holds(i));  // the paper's claims that fail in Cl(4, 1), and why
  const auto& fga = r.at("fgaRecovery").asNumbers();
  const auto& hrr = r.at("hrrRecovery").asNumbers();
  CHECK(fga[0] == 1.0 && fga[1] == 1.0);  // exact up to the two role directions of <N, e>
  CHECK(fga[2] < 0.5);
  CHECK(hrr[0] > 0.95 && hrr[5] < hrr[0]);  // holographic binding: approximate, degrading
}

TEST(fga_learning_structure_is_an_inductive_bias) {
  const Value r = run("fgaLearning");
  const auto& d = r.at("designs").asArray();
  CHECK(d.size() == 5);
  for (const auto& x : d) CHECK(num(x, "read") > 0.85);                       // every learner learns the text
  CHECK(num(d[0], "reversed") > 0.8);                                           // bivectors: the reverse is false
  CHECK(num(d[4], "reversed") < 0.2);                                           // flat: the product is symmetric
  CHECK(num(d[4], "symmetric") > 0.8);
  CHECK(num(d[0], "symmetric") < 0.2);                                          // no bivector is symmetric
  CHECK(at(d[0], "entityGrades", 1) > 0.999);                                   // typed: entities in grade 1
  CHECK(num(d[3], "reversed") > num(d[2], "reversed"));                         // the grade prior helps
  const Value u = run("fgaLearning", [] {
    Value s = spec(64);
    s.set("fgaLearning", Value::object().set("text", "John loves Mary. Mary does not love John. The book is older than Mary."));
    return s;
  }());
  CHECK(u.at("relations").size() == 2 && u.at("entities").size() == 3 && !u.at("world").asBool());
  CHECK(num(item(u, "designs", 0), "read") > 0.9);
}

TEST(fga_rotors_generalise_from_few_words) {
  const Value r = run("fgaRotors");
  const auto& runs = r.at("runs").asArray();
  CHECK(num(runs[1], "rotorTest") < 0.5 * num(runs[1], "projectionTest"));  // two words: a rotor family generalises
  CHECK(num(runs[1], "plane") > 0.95);                                         // ... and finds the hidden plane
  for (const auto& x : runs) {
    CHECK(num(x, "rotorNorm") < 1e-4);                                         // norm-preserving
    CHECK(num(x, "projectionNorm") > num(x, "rotorNorm"));
    CHECK(num(x, "hyperbolic") < 1e-6);                                        // an alternation inside V_E stays elliptic
  }
  CHECK(num(runs[4], "projectionTest") < num(runs[0], "projectionTest"));  // projections need more words
}

TEST(reading_qa_answers_with_the_sentence_it_believes) {
  const Value r = run("readingQA");  // the built-in example
  const auto& a = r.at("answers").asArray();
  CHECK(a.size() == 3);
  CHECK(a[0].at("reading").asString().find("weight of the ice") != std::string::npos);
  CHECK(a[1].at("reading").asString().find("length of the waggle run") != std::string::npos);
  CHECK(a[2].at("reading").asString().find("pressed into dense ice") != std::string::npos);
  for (const auto& x : a) {
    CHECK(x.at("openBook").asString() == x.at("reading").asString());  // the corpus does not distract it
    CHECK(num(x, "readingBelief") > 0.0 && num(x, "readingBelief") < 1.0);
  }
}

TEST(reading_qa_scores_itself_against_the_embedded_golden_answers) {
  Value s = spec(64);
  s.set("readingQA", Value::object().set("set", "golden").set("questions", 60));
  const Value r = run("readingQA", s);
  CHECK(num(r, "scored") == 60);
  const Value& sc = r.at("scores");
  CHECK(num(sc, "reading") > num(sc, "firstSentence") + 0.1);  // better than the first sentence
  CHECK(num(sc, "reading") > num(sc, "randomSentence") + 0.1);
  CHECK(num(sc, "oracle") >= num(sc, "reading"));               // the ceiling of one-sentence answers
  CHECK(num(sc, "openBookFoundPassage") > 0.9);
}
