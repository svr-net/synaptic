#include "syn/cognitivism/schema.hpp"

#include <algorithm>
#include <stdexcept>

#include "syn/core/math.hpp"

namespace syn {

std::size_t SchemaNetwork::addConcept(const std::string& name) {
  for (const auto& n : names_)
    if (n == name) throw std::invalid_argument("SchemaNetwork: duplicate node " + name);
  names_.push_back(name);
  for (auto& row : w_) row.push_back(0.0);
  w_.emplace_back(names_.size(), 0.0);
  return names_.size() - 1;
}

void SchemaNetwork::connect(std::size_t a, std::size_t b, double weight) {
  if (a >= size() || b >= size() || a == b) throw std::invalid_argument("SchemaNetwork::connect: bad concepts");
  if (weight < 0.0 || weight > 1.0) throw std::invalid_argument("SchemaNetwork::connect: weight must lie in [0, 1]");
  w_[a][b] = w_[b][a] = weight;
}

std::size_t SchemaNetwork::index(const std::string& name) const {
  for (std::size_t i = 0; i < names_.size(); ++i)
    if (names_[i] == name) return i;
  throw std::invalid_argument("SchemaNetwork: unknown node " + name);
}

double SchemaNetwork::weight(std::size_t a, std::size_t b) const { return w_.at(a).at(b); }

std::vector<double> SchemaNetwork::spread(const std::vector<std::size_t>& cues, int steps, double decay) const {
  std::vector<double> a(size(), 0.0);
  for (auto c : cues) a.at(c) = 1.0;
  for (int s = 0; s < steps; ++s) {
    std::vector<double> next = a;
    for (std::size_t i = 0; i < size(); ++i) {
      // Activation arriving at i: strongest incoming route (max keeps it in [0, 1]).
      double in = 0.0;
      for (std::size_t j = 0; j < size(); ++j) in = std::max(in, decay * w_[j][i] * a[j]);
      next[i] = std::max(a[i], in);
    }
    a.swap(next);
  }
  return a;
}

double SchemaNetwork::recallProbability(std::size_t node, const std::vector<std::size_t>& cues, double threshold,
                                        double gain) const {
  return logistic(gain * (spread(cues).at(node) - threshold));
}

std::size_t SchemaNetwork::learn(const std::string& name, const std::vector<std::size_t>& presented,
                                 const std::vector<std::size_t>& primed, double linkThreshold, double strength) {
  std::vector<std::size_t> active = presented;
  active.insert(active.end(), primed.begin(), primed.end());
  const auto a = spread(active, 1);
  const std::size_t id = addConcept(name);
  for (std::size_t j = 0; j + 1 < size(); ++j)
    if (a[j] >= linkThreshold) w_[id][j] = w_[j][id] = clamp01(strength * a[j]);
  return id;
}

std::size_t SchemaNetwork::degree(std::size_t node) const {
  std::size_t n = 0;
  for (double x : w_.at(node)) n += x > 0.0 ? 1 : 0;
  return n;
}

SchemaNetwork SchemaNetwork::librarySearchExample() {
  SchemaNetwork s;
  for (const char* c : {"research question", "keyword", "synonym", "search engine", "Venn diagram", "set", "narrow results",
                        "broaden results", "exclude a term", "relevance"})
    s.addConcept(c);
  auto link = [&](const char* a, const char* b, double w) { s.connect(s.index(a), s.index(b), w); };
  link("research question", "keyword", 0.9);
  link("keyword", "synonym", 0.8);
  link("keyword", "search engine", 0.7);
  link("search engine", "relevance", 0.6);
  link("Venn diagram", "set", 0.9);
  link("set", "narrow results", 0.5);
  link("set", "broaden results", 0.5);
  link("set", "exclude a term", 0.5);
  link("narrow results", "relevance", 0.6);
  link("broaden results", "synonym", 0.6);
  return s;
}

}  // namespace syn
