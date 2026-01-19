import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from gemkr.models.gemkr_codebert import GEMKRCodeBERTDSI

def main():
    model = GEMKRCodeBERTDSI(
        encoder_name="/data/lizhen/CodeDSI/codebert-base",
        decoder_name="/data/lizhen/llama-7b",
        hidden_size=4096,
        freeze_encoder=True,
        freeze_decoder=True,
        decoder_dtype="float16",
        enable_gradient_checkpointing=False,
    )

    # check a few lora params
    print("===== LoRA PARAM DTYPES (first 20) =====")
    cnt = 0
    for n, p in model.named_parameters():
        if "lora_" in n:
            print(n, p.dtype)
            cnt += 1
            if cnt >= 20:
                break

    # check decoder embedding dtype (source of truth)
    emb_dtype = model.decoder.get_input_embeddings().weight.dtype
    print("\nDecoder embedding dtype:", emb_dtype)

    # hard check
    for n, p in model.named_parameters():
        if "lora_" in n:
            assert p.dtype == emb_dtype, f"dtype mismatch: {n} {p.dtype} vs emb {emb_dtype}"

    print("\n✅ PASS: all LoRA params match decoder embedding dtype.")

if __name__ == "__main__":
    main()
