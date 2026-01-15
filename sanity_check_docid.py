# sanity_check_docid.py
# --------------------------------------------------
# Purpose:
#   Robust sanity check for DSI-style DocID generation
#   - Prefer strict cosqa_XXXXXXXX extraction
#   - Tolerate noisy / trailing tokens
#
# Usage:
#   python sanity_check_docid.py \
#       --ckpt path/to/epoch_1.pth \
#       --data cosqa_valid.jsonl \
#       --num-samples 5
# --------------------------------------------------

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import re
import argparse
import torch

from gemkr.models.gemkr_codebert import GEMKRCodeBERTDSI


# ============================
# Regex patterns (STRICT FIRST)
# ============================
DOCID_STRICT = re.compile(r"cosqa_(\d{8})")
DOCID_LOOSE  = re.compile(r"cosqa_([0-9]+)")
DIGITS_ANY   = re.compile(r"\d+")


def extract_docid(gen_text: str):
    """
    Extract docid from generated text with priority:
      1) cosqa_XXXXXXXX (exact 8 digits)
      2) cosqa_[digits] (loose)
      3) any digit sequence (fallback, diagnostic only)
    """
    # 1) strict
    m = DOCID_STRICT.search(gen_text)
    if m:
        return {
            "type": "strict",
            "docid": f"cosqa_{m.group(1)}",
            "digits": [m.group(1)],
        }

    # 2) loose cosqa_
    m = DOCID_LOOSE.search(gen_text)
    if m:
        return {
            "type": "loose",
            "docid": f"cosqa_{m.group(1)}",
            "digits": [m.group(1)],
        }

    # 3) fallback: any digits
    digits = DIGITS_ANY.findall(gen_text)
    return {
        "type": "digits_only",
        "docid": None,
        "digits": digits,
    }


def load_jsonl(path, n=5):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
            if len(data) >= n:
                break
    return data


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("DocID Sanity Check (Robust)")
    parser.add_argument("--ckpt", required=True, help="checkpoint path (e.g., epoch_1.pth)")
    parser.add_argument("--data", required=True, help="cosqa_valid.jsonl")
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    model = GEMKRCodeBERTDSI(
        encoder_name="/data/lizhen/CodeDSI/codebert-base",
        decoder_name="/data/lizhen/llama-7b",
        hidden_size=4096,
        freeze_encoder=True,
        freeze_decoder=False,   # must be False to allow docid learning
        encoder_max_length=512,
        decoder_max_length=64,
        decoder_dtype="float16",
        enable_gradient_checkpointing=False,
    )

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    print("Loading data...")
    samples = load_jsonl(args.data, n=args.num_samples)

    print("\n===== DOCID SANITY CHECK (ROBUST) =====")

    strict_hits = 0

    for i, item in enumerate(samples):
        gold = item.get("docid", "N/A")

        outputs = model.generate(
            {
                "query": [item["query"]],
                "code":  [item["code"]],
            },
            max_new_tokens=32,
            num_beams=1,
            do_sample=False,
        )

        gen_text = model.decoder_tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        parsed = extract_docid(gen_text)

        if parsed["type"] == "strict":
            strict_hits += 1

        print(f"\n[{i}]")
        print(f"Gold docid : {gold}")
        print(f"Generated  : {gen_text}")
        print(f"Parse type : {parsed['type']}")
        print(f"Parsed id  : {parsed['docid']}")
        print(f"Digits     : {parsed['digits']}")

    print("\n================================")
    print(f"Strict cosqa_XXXXXXXX hits : {strict_hits} / {len(samples)}")
    print("\nInterpretation:")
    print("- >=1 strict hit → training signal is correct, continue training")
    print("- Only loose / digits_only → model started learning structure, but not stabilized")
    print("- No digits at all → stop and fix training setup")


if __name__ == "__main__":
    main()
