#pragma once

// The reader as a native tool: what it remembers (memory records), asked from a shell or a server, over the passages
// it is given -- or, asked alone, its stored capsule (the golden passages) -- with the same models and kernels as the
// library (capsuleQA, capsuleIngest), on the software GPU. A server's knowledge is its own store's (svr-net/ui: capsule's).
//
//   reader version                         the reader and its capsule's format
//   reader info                            what it knows (its stored capsule) and remembers: passages, words, memory
//   reader ask QUESTION                    one answer from its stored capsule, as one JSON line
//   reader session                         {"ready":…} then, per line of input (a question), its answer as a JSON line
//   reader from QUESTION                   every line of input a passage, read into a capsule of its own (kept nowhere)
//                                          and the question answered from it alone, as one JSON line
//   reader practise                        every line of input a lesson -- question<TAB>golden answer<TAB>passage... --
//                                          practised as a student (practise below); what it learned as one JSON line
//
// Options: --records FILE (memory records: a JSON array of base64 records, or one per line; repeatable).
#include <iosfwd>
#include <string>
#include <vector>

namespace syn::reader {

/** Run a reader command. `records` are what the caller already holds (read from a store's reader log, say) and come
 *  before those named by --records. Returns the exit code. */
int run(const std::vector<std::string>& args, std::vector<std::string> records, std::istream& in, std::ostream& out,
        std::ostream& err);

/** One question answered from the passages given -- read into a capsule of their own, held only while it answers --
 *  as one JSON line (capsuleQA's answer with found and confident, and "passages"). A server that finds the passages a
 *  question activates (svr-net/ui: capsule's engine) has the reader reason over them. */
std::string answerFrom(const std::vector<std::string>& passages, const std::string& question, const std::vector<std::string>& records);

/** A lesson: a question, the answer it should give, and the passages it may answer from (read for it, kept nowhere). */
struct Lesson {
  std::string question, golden;
  std::vector<std::string> passages;
};
/** Practice as a student (readingStudent, constructing, with every learning method): its constructing habits learned
 *  from three quarters of the lessons and tested on the rest, held out. The study is kept -- as a memory record -- only
 *  when its memory with it folded in answers the held-out lessons at least as well as without it. One JSON line:
 *  {practised, train, test, before, after, study, kept, record (base64, when kept), generation, reason}. `records`: what
 *  it remembers already. A server that ingests knowledge incrementally (svr-net/ui) practises after each segment. */
std::string practise(const std::vector<Lesson>& lessons, const std::vector<std::string>& records, const std::string& source);

/** The base64 blobs of a file: a JSON array of strings, or one per line. */
std::vector<std::string> blobsOf(const std::string& text);

}  // namespace syn::reader
