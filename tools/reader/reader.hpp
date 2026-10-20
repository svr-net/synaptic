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

/** The base64 blobs of a file: a JSON array of strings, or one per line. */
std::vector<std::string> blobsOf(const std::string& text);

}  // namespace syn::reader
