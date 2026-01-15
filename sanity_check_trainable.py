# sanity_check_trainable.py
# --------------------------------------
# Purpose:
#   Verify which parameters are trainable
#   after model construction (LoRA + freeze logic)
#
# Usage:
#   python sanity_check_trainable.py
# --------------------------------------

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch

from gemkr.models.gemkr_codebert import GEMKRCodeBERTDSI


def main():
    # ====== MODIFY THESE PATHS IF NEEDED ======
    encoder_name = "/data/lizhen/CodeDSI/codebert-base"
    decoder_name = "/data/lizhen/llama-7b"

    print("Building model ...")

    model = GEMKRCodeBERTDSI(
        encoder_name=encoder_name,
        decoder_name=decoder_name,
        hidden_size=4096,
        freeze_encoder=True,
        freeze_decoder=False,      # 🔴 must be False
        encoder_max_length=512,
        decoder_max_length=64,
        decoder_dtype="float16",
        enable_gradient_checkpointing=False,
    )

    # ======================================================
    # Inspect trainable parameters
    # ======================================================
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    frozen = [(n, p.numel()) for n, p in model.named_parameters() if not p.requires_grad]

    print("\n===== TRAINABLE PARAMETERS (first 30) =====")
    for n, c in trainable[:30]:
        print(f"{n:80s} {c}")

    print("\n===== FROZEN PARAMETERS (first 10) =====")
    for n, c in frozen[:10]:
        print(f"{n:80s} {c}")

    trainable_cnt = sum(c for _, c in trainable)
    total_cnt = sum(p.numel() for p in model.parameters())

    print("\n===== SUMMARY =====")
    print(f"Trainable params : {trainable_cnt / 1e6:.2f} M")
    print(f"Total params     : {total_cnt / 1e6:.2f} M")
    print(f"Trainable ratio  : {100 * trainable_cnt / total_cnt:.2f} %")

    # ======================================================
    # Hard assertions (fail fast)
    # ======================================================
    lora_params = [n for n, _ in trainable if "lora_" in n]
    assert len(lora_params) > 0, (
        "❌ No LoRA parameters are trainable!\n"
        "Check: get_peft_model() + freeze logic."
    )

    print("\n✅ LoRA parameters detected:")
    for n in lora_params[:10]:
        print("  ", n)

    print("\n🎯 SANITY CHECK PASSED: decoder is trainable via LoRA.")


if __name__ == "__main__":
    main()
