#pragma once

// Schemas: knowledge in long-term memory organised as concepts connected to one another.
// Retrieval follows the connections (spreading activation): cues activate concepts, and
// activation flows along weighted links, decaying at each step. A node is recalled when its
// activation crosses a threshold. New knowledge is encoded by linking it to whatever is active
// at the time, which is why stimulating recall of prior learning (Gagné's third event) gives
// the new node more retrieval routes.

#include <cstddef>
#include <string>
#include <vector>

namespace syn {

class SchemaNetwork {
 public:
  std::size_t addConcept(const std::string& name);
  void connect(std::size_t a, std::size_t b, double weight);  // symmetric link
  std::size_t index(const std::string& name) const;
  std::size_t size() const { return names_.size(); }
  const std::string& name(std::size_t i) const { return names_.at(i); }
  double weight(std::size_t a, std::size_t b) const;

  /** Activation after `steps` rounds of spreading from the cues (each cue clamped at 1). */
  std::vector<double> spread(const std::vector<std::size_t>& cues, int steps = 3, double decay = 0.6) const;

  /** P(recall) of a node from the cues: σ(gain·(activation − threshold)). */
  double recallProbability(std::size_t node, const std::vector<std::size_t>& cues, double threshold = 0.25,
                           double gain = 12.0) const;

  /**
   * Encodes a new node presented with `presented` cues. It is linked to every node whose
   * activation (spreading from the presented cues and the `primed` prior knowledge) exceeds
   * `linkThreshold`, with weight proportional to that activation.
   */
  std::size_t learn(const std::string& name, const std::vector<std::size_t>& presented,
                    const std::vector<std::size_t>& primed, double linkThreshold = 0.2, double strength = 0.8);

  /** Number of links of a node (its retrieval routes). */
  std::size_t degree(std::size_t node) const;

  /** Small network of library-search knowledge used by the examples. */
  static SchemaNetwork librarySearchExample();

 private:
  std::vector<std::string> names_;
  std::vector<std::vector<double>> w_;
};

}  // namespace syn
