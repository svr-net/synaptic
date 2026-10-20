#pragma once

// The reader as a native tool: what it knows (capsule segments) and remembers (memory records), asked from a
// shell or a server — the same models and kernels as the library (capsuleQA, capsuleIngest), on the software GPU.
//
//   reader version                         the reader and its capsule's format
//   reader info                            what it knows: passages, words, segments (used or refused), memory
//   reader ask QUESTION                    one answer, as one JSON line
//   reader session                         {"ready":…} then, per line of input (a question), its answer as a JSON line
//   reader batch                           every line of input a question, answered in one run, a JSON line each
//   reader ingest OUT TSV [--stored]       the passages of TSV (text, or title<TAB>text per line) ingested on top of
//                                          what it knows; the new segment written to OUT (base64)
//   reader from QUESTION                   every line of input a passage, read into a capsule of its own (kept nowhere)
//                                          and the question answered from it alone, as one JSON line
//   reader practise                        every line of input a lesson -- question<TAB>golden answer<TAB>passage... --
//                                          practised as a student (practise below); what it learned as one JSON line
//
// Options: --segments FILE (a JSON array of base64 segments, or one per line; repeatable), --records FILE (memory
// records, likewise). With no segments it knows the stored capsule; segments that extend it are read on top of it.
#include <iosfwd>
#include <string>
#include <vector>

namespace syn::reader {

/** Run a reader command. `segments` and `records` are what the caller already holds (read from a knowledge
 *  repo, say) and come before those named by --segments / --records. Returns the exit code. */
int run(const std::vector<std::string>& args, std::vector<std::string> segments, std::vector<std::string> records, std::istream& in,
        std::ostream& out, std::ostream& err);

/** One question answered from the passages given -- read into a capsule of their own, held only while it answers --
 *  as one JSON line (capsuleQA's answer with found and confident, and "passages"). A server that finds the passages a
 *  question activates (svr-net/ui: capsule's engine) has the reader reason over them. */
std::string answerFrom(const std::vector<std::string>& passages, const std::string& question, const std::vector<std::string>& records);
/** One question answered from what it knows (segments) and remembers (records), as one JSON line. */
std::string answerKnown(const std::vector<std::string>& segments, const std::vector<std::string>& records, const std::string& question);

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
