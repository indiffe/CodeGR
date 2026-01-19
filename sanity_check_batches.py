# sanity_check_batches.py
# --------------------------------------------------
# Purpose:
#   Check whether docid / answer_id is empty or malformed
#   in the training jsonl file (root cause of NaN)
#
# Usage:
#   python sanity_check_batches.py --data cosqa_train.jsonl
# --------------------------------------------------

import argparse
import json
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="cosqa_train.jsonl")
    parser.add_argument("--max-samples", type=int, default=5000)
    args = parser.parse_args()

    empty = 0
    short = 0
    ok = 0
    patterns = Counter()

    with open(args.data, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.max_samples:
                break
            item = json.loads(line)

            docid = item.get("docid") or item.get("answer_id") or ""

            if not isinstance(docid, str) or docid.strip() == "":
                empty += 1
                continue

            if len(docid.strip()) < 8:
                short += 1
                continue

            ok += 1

            # record pattern
            patterns[docid.strip()[:20]] += 1

    print("===== DOCID DATA SANITY CHECK =====")
    print(f"Checked samples : {empty + short + ok}")
    print(f"Empty docid     : {empty}")
    print(f"Too short docid : {short}")
    print(f"Valid docid     : {ok}")

    print("\nTop docid prefixes:")
    for k, v in patterns.most_common(5):
        print(f"  {k}... : {v}")

    print("\nInterpretation:")
    print("- empty > 0     → MUST fix dataset (NaN guaranteed)")
    print("- short > 0     → tokenizer may produce empty labels")
    print("- valid only    → dataset OK, model-side guards sufficient")


if __name__ == "__main__":
    main()
