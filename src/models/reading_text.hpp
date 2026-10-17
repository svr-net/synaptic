#pragma once

// Reading text for the reading models (readingQA, readingStudent): words, sentences, SQuAD's token
// F1, and the built-in golden set. Host-side preparation only; the reading itself runs in WGSL.
#include <algorithm>
#include <cctype>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "syn/core/value.hpp"

namespace syn::models::text {

inline const std::set<std::string>& stopwords() {
  static const std::set<std::string> s = {
      "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at", "by", "for", "with", "from", "as", "into", "about", "than",
      "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "has", "have", "had", "it", "its", "this", "that",
      "these", "those", "there", "their", "they", "them", "he", "she", "his", "her", "him", "we", "our", "you", "your", "i", "my", "me",
      "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "can", "could", "would", "should", "will", "may", "might",
      "must", "not", "no", "so", "such", "also", "some", "any", "all", "each", "other", "more", "most", "one", "two", "according",
      "passage", "text", "does", "do", "s", "t"};
  return s;
}

// Lower-case words of letters and digits, with a light stemming of plural and verb endings.
inline std::vector<std::string> words(const std::string& text, bool content) {
  std::vector<std::string> out;
  std::string w;
  auto flush = [&]() {
    if (w.empty()) return;
    if (w.size() > 4 && w.compare(w.size() - 3, 3, "ing") == 0) w.resize(w.size() - 3);
    else if (w.size() > 3 && w.compare(w.size() - 2, 2, "ed") == 0) w.resize(w.size() - 2);
    else if (w.size() > 3 && w.compare(w.size() - 2, 2, "es") == 0) w.resize(w.size() - 2);
    else if (w.size() > 3 && w.back() == 's') w.pop_back();
    if (!content || !stopwords().count(w)) out.push_back(w);
    w.clear();
  };
  for (char ch : text) {
    if (std::isalnum(static_cast<unsigned char>(ch))) w += static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    else flush();
  }
  flush();
  return out;
}

// Sentences: split after . ! ? followed by a space and an upper-case letter, a digit or a quote.
inline std::vector<std::string> sentences(const std::string& text) {
  std::vector<std::string> out;
  std::string cur;
  for (std::size_t i = 0; i < text.size(); ++i) {
    cur += text[i];
    const bool end = (text[i] == '.' || text[i] == '!' || text[i] == '?') && i + 2 < text.size() && text[i + 1] == ' ' &&
                     (std::isupper(static_cast<unsigned char>(text[i + 2])) || std::isdigit(static_cast<unsigned char>(text[i + 2])) || text[i + 2] == '"');
    if (end && words(cur, false).size() >= 4) {
      out.push_back(cur);
      cur.clear();
      ++i;  // the space
    }
  }
  if (!words(cur, false).empty()) {
    if (!out.empty() && words(cur, false).size() < 4) out.back() += cur;
    else out.push_back(cur);
  }
  for (auto& s : out) {
    s.erase(0, s.find_first_not_of(" \t\r\n"));
    s.erase(s.find_last_not_of(" \t\r\n") + 1);
  }
  return out;
}

// SQuAD's token F1: lower case, no punctuation, no articles.
inline std::vector<std::string> squadWords(const std::string& text) {
  std::vector<std::string> out;
  std::string w;
  auto flush = [&]() {
    if (!w.empty() && w != "a" && w != "an" && w != "the") out.push_back(w);
    w.clear();
  };
  for (char ch : text) {
    if (std::isalnum(static_cast<unsigned char>(ch))) w += static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    else if (std::isspace(static_cast<unsigned char>(ch))) flush();
  }
  flush();
  return out;
}

inline double tokenF1(const std::string& prediction, const std::string& golden) {
  const auto p = squadWords(prediction), g = squadWords(golden);
  if (p.empty() || g.empty()) return p == g ? 1.0 : 0.0;
  std::map<std::string, int> count;
  for (const auto& w : g) count[w]++;
  int common = 0;
  for (const auto& w : p)
    if (count[w]-- > 0) ++common;
  if (common == 0) return 0.0;
  const double precision = double(common) / double(p.size()), recall = double(common) / double(g.size());
  return 2 * precision * recall / (precision + recall);
}

/** The first `count` questions of the built-in golden set, with their passages and golden answers. */
Value goldenSet(int count);

}  // namespace syn::models::text
