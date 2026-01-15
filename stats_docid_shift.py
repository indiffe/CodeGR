# =========================
# MUST be first
# =========================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# =========================
# imports
# =========================
import json
import re
import argparse
import statistics
from collections import Counter

import torch
from tqdm import tqdm

from gemkr.models.gemkr_codebert import GEMKRCodeBERTDSI


# =========================
# utils: robust docid parser
# =========================
def extract_docid_int(text: str):
    """
    Robust docid extractor.
    Extract the FIRST integer appearing in text.
    Return None if no integer found.
    """
    if text is None:
        return None
    m = re.search(r"(\d+)", text)
    return int(m.group(1)) if m else None


def load_cosqa_jsonl(path, max_samples=None):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
            if max_samples is not None and len(data) >= max_samples:
                break
    return data


@torch.no_grad()
def generate_top1_docid(model, query, code):
    samples = {
        "query": [query],
        "code":  [code],
    }

    outputs = model.generate(
        samples,
        max_length=16,
        num_beams=1,
        num_return_sequences=1,
        early_stopping=True,
        use_cache=True,
    )

    text = model.decoder_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return text, extract_docid_int(text)


# =========================
# main
# =========================
def main():
    parser = argparse.ArgumentParser("Robust DocID Shift Diagnostic Tool")
    parser.add_argument("--ckpt", required=True, help="path to checkpoint")
    parser.add_argument("--data", required=True, help="cosqa_valid.jsonl")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--show-errors", type=int, default=10,
                        help="print top-K largest shift samples")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    model = GEMKRCodeBERTDSI(
        encoder_name="/data/lizhen/CodeDSI/codebert-base",
        decoder_name="/data/lizhen/llama-7b",
        hidden_size=4096,
        freeze_encoder=True,
        freeze_decoder=True,
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
    data = load_cosqa_jsonl(args.data, max_samples=args.max_samples)

    gold_ids = []
    pred_ids = []
    shifts = []
    error_cases = []

    cnt_gold_fail = 0
    cnt_pred_fail = 0

    print("Running diagnostics...")
    for item in tqdm(data):
        gold = extract_docid_int(item.get("docid"))
        if gold is None:
            cnt_gold_fail += 1
            continue

        gen_text, pred = generate_top1_docid(
            model,
            item["query"],
            item["code"],
        )

        if pred is None:
            cnt_pred_fail += 1
            continue

        gold_ids.append(gold)
        pred_ids.append(pred)

        diff = pred - gold
        shifts.append(diff)

        error_cases.append({
            "gold": gold,
            "pred": pred,
            "shift": diff,
            "gen_text": gen_text,
            "query": item["query"][:200],
        })

    # =========================
    # summary
    # =========================
    print("\n========== DOCID SHIFT DIAGNOSTICS ==========")
    print(f"Total samples       : {len(data)}")
    print(f"Gold parse failed   : {cnt_gold_fail}")
    print(f"Pred parse failed   : {cnt_pred_fail}")
    print(f"Valid pairs         : {len(gold_ids)}")

    if len(gold_ids) == 0:
        print("\n[ERROR] No valid (gold, pred) docid pairs collected.")
        print("Check:")
        print("  - docid format in dataset")
        print("  - whether model generates any digits at all")
        return

    print("\n----- Distribution -----")
    print(f"Gold mean / std     : {statistics.mean(gold_ids):.1f} / {statistics.pstdev(gold_ids):.1f}")
    print(f"Pred mean / std     : {statistics.mean(pred_ids):.1f} / {statistics.pstdev(pred_ids):.1f}")

    print("\n----- Shift (pred - gold) -----")
    print(f"Mean shift          : {statistics.mean(shifts):.1f}")
    print(f"Median shift        : {statistics.median(shifts):.1f}")
    print(f"Min / Max shift     : {min(shifts)} / {max(shifts)}")

    # =========================
    # collapse / drift hint
    # =========================
    if statistics.pstdev(pred_ids) < 0.1 * statistics.pstdev(gold_ids):
        print("\n[WARN] Possible DOCID COLLAPSE detected (pred variance too small).")

    if abs(statistics.mean(shifts)) > 0.1 * statistics.pstdev(gold_ids):
        print("[WARN] Global DOCID DRIFT detected (mean shift far from 0).")

    # =========================
    # show largest errors
    # =========================
    print("\n----- Top shift samples -----")
    error_cases.sort(key=lambda x: abs(x["shift"]), reverse=True)

    for i, e in enumerate(error_cases[:args.show_errors]):
        print(f"\n[{i}] shift={e['shift']} | gold={e['gold']} | pred={e['pred']}")
        print(f"gen_text: {e['gen_text']}")
        print(f"query   : {e['query']}")

    print("\n============================================")


if __name__ == "__main__":
    main()
