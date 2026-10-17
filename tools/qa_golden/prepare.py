#!/usr/bin/env python3
"""Prepares the end-to-end QA golden test.

Downloads the golden question-answer pairs (edededdy/fineweb-edu-grounded-qa: questions about
FineWeb-Edu passages with golden answers, ODC-By), checks the file against its pinned hash, draws a
seeded sample of questions about distinct passages, and writes:

  <out>/qa.tsv      id, question, golden answer, passage   (one question per line)
  <out>/corpus.tsv  id, text                                (the knowledge repository's documents)

Usage: prepare.py --knowledge /path/to/knowledge --out DIR [--questions 200] [--seed 2026]
Needs pyarrow (pip install pyarrow).
"""
import argparse
import hashlib
import os
import random
import re
import sys
import urllib.request

URL = ("https://huggingface.co/datasets/edededdy/fineweb-edu-grounded-qa/resolve/"
       "refs%2Fconvert%2Fparquet/default/train/0000.parquet")
SHA256 = "3ce8e7946e1637a77418260e43d75712d1b62104d1de3ccb761386e465ccd7c8"


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knowledge", required=True, help="checkout of svr-net/knowledge")
    ap.add_argument("--out", required=True)
    ap.add_argument("--questions", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    parquet = os.path.join(args.out, "golden.parquet")
    if not os.path.exists(parquet):
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, parquet)
    digest = hashlib.sha256(open(parquet, "rb").read()).hexdigest()
    if digest != SHA256:
        sys.exit(f"golden set changed: sha256 {digest}, expected {SHA256}")

    import pyarrow.parquet as pq
    rows = pq.read_table(parquet, columns=["question", "answer", "passage", "doc_index", "refers_to_passage"]).to_pylist()
    # Questions that stand on their own (not "according to the passage"), one per passage.
    by_doc = {}
    for r in rows:
        if not r["refers_to_passage"]:
            by_doc.setdefault(r["doc_index"], []).append(r)
    docs = sorted(by_doc)
    rng = random.Random(args.seed)
    rng.shuffle(docs)
    picked = [rng.choice(by_doc[d]) for d in docs[: args.questions]]
    with open(os.path.join(args.out, "qa.tsv"), "w") as f:
        for r in picked:
            f.write("\t".join([str(r["doc_index"]), clean(r["question"]), clean(r["answer"]), clean(r["passage"])]) + "\n")

    with open(os.path.join(args.out, "corpus.tsv"), "w") as f:
        root = os.path.join(args.knowledge, "docs")
        for name in sorted(os.listdir(root)):
            if name.endswith(".txt"):
                f.write(name[:-4] + "\t" + clean(open(os.path.join(root, name)).read()) + "\n")
    print(f"{len(picked)} questions, {len(os.listdir(os.path.join(args.knowledge, 'docs')))} corpus documents -> {args.out}")


if __name__ == "__main__":
    main()
