#!/bin/sh
# The end-to-end QA golden test: download and sample the golden questions, read the knowledge
# corpus, answer on the software GPU and on WebGPU, score against the golden answers, and fail
# if a score drops below tools/qa_golden/baseline.json.
#
#   tools/qa_golden/run.sh /path/to/knowledge [DATA_DIR]
# Needs: a native build in build/ (cmake -B build && cmake --build build), python3 with pyarrow,
# node and Playwright's Chromium for the WebGPU run.
set -e
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
knowledge=${1:?usage: run.sh /path/to/knowledge [DATA_DIR]}
data=${2:-$repo/build/qa_golden}
python3 "$here/prepare.py" --knowledge "$knowledge" --out "$data"
"$repo/build/qa_golden" "$data" --baseline "$here/baseline.json"
node "$here/webgpu.mjs" "$data"
