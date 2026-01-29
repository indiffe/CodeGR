# =========================
# ⚠️ 必须是第一行（任何 import 之前）
# =========================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"   # 使用物理 GPU 7


# =========================
# 现在才可以 import
# =========================
import json
import re
import torch
from tqdm import tqdm

from gemkr.models.gemkr_codebert import GEMKRCodeBERTDSI


# =========================
# 0. 推理 dtype 修复（避免 Half/Float 报错）
# =========================
def fix_infer_dtypes(model):
    """
    推理阶段统一 dtype，避免：
      - alignment_proj mat1/mat2 dtype mismatch
      - lm_head expected Half but found Float

    推荐推理配置：
      - alignment_proj: FP32（接 CodeBERT 输出通常 FP32）
      - decoder/emb/lm_head: FP16（和 decoder hidden_states 一致）
    """
    # alignment_proj 用 fp32
    model.alignment_proj.to(dtype=torch.float32)

    # decoder 侧统一 fp16
    model.decoder.to(dtype=torch.float16)
    model.decoder.get_input_embeddings().to(dtype=torch.float16)
    model.decoder.get_output_embeddings().to(dtype=torch.float16)
    if hasattr(model.decoder, "lm_head") and model.decoder.lm_head is not None:
        model.decoder.lm_head.to(dtype=torch.float16)

    # 打印确认
    try:
        print(
            "[DTYPE CHECK]",
            "alignment_proj:", model.alignment_proj.weight.dtype,
            "| decoder(any):", next(model.decoder.parameters()).dtype,
            "| emb:", next(model.decoder.get_input_embeddings().parameters()).dtype,
            "| lm_head:", next(model.decoder.get_output_embeddings().parameters()).dtype,
        )
    except Exception as e:
        print("Could not inspect dtypes:", repr(e))


# =========================
# 1. 手动加载 CoSQA jsonl
# =========================
def load_cosqa_jsonl(path, max_samples=None):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
            if max_samples is not None and len(data) >= max_samples:
                break
    return data


# =========================
# 2. docid 标准化（保持你原逻辑）
# =========================
def normalize_docid(pred: str) -> str:
    if pred is None:
        return pred
    pred = str(pred).strip()

    # 1) 优先抽标准形式
    m = re.search(r"(cosqa_\d{8})", pred)
    if m:
        return m.group(1)

    # 2) 否则抽任意 8 位数字，并补回 cosqa_
    m2 = re.search(r"(\d{8})", pred)
    if m2:
        return "cosqa_" + m2.group(1)

    return pred


# =========================
# 3. 手动生成 DocID（最小修改：eos + max_new_tokens）
# =========================
@torch.no_grad()
def generate_docids(model, samples, num_beams=10, num_return_sequences=10):
    """
    samples = {"query":[str], "code":[str]}
    return: List[str]  # normalize 后的 docid
    """
    end_id = model.decoder_tokenizer.convert_tokens_to_ids("</DOCID>")
    if end_id is None or end_id < 0:
        end_id = model.decoder_tokenizer.eos_token_id

    outputs = model.generate(
        samples,
        max_new_tokens=10,
        num_beams=int(num_beams),
        num_return_sequences=int(num_return_sequences),
        early_stopping=True,
        eos_token_id=int(end_id),
        pad_token_id=int(model.decoder_tokenizer.pad_token_id),
        use_cache=True,
        min_new_tokens=8,
    )

    preds = []
    for seq in outputs:
        text = model.decoder_tokenizer.decode(
            seq,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        preds.append(normalize_docid(text))

    return preds


# =========================
# 4. Hit@K / Recall@K + HitCount@K
# =========================
def evaluate_cosqa(model, data, device="cuda", ks=(1, 5, 10)):
    model.to(device)
    model.eval()

    hit_count = {k: 0 for k in ks}
    total = len(data)

    for item in tqdm(data, desc="Evaluating"):
        samples = {
            "query": [item["query"]],
            "code":  [item["code"]],
        }
        gold = item["docid"]

        preds = generate_docids(
            model,
            samples,
            num_beams=max(ks),
            num_return_sequences=max(ks),
        )

        for k in ks:
            if gold in preds[:k]:
                hit_count[k] += 1

    results = {}
    for k in ks:
        results[f"HitCount@{k}"] = hit_count[k]
        results[f"Hit@{k}"] = hit_count[k] / total
        results[f"Recall@{k}"] = hit_count[k] / total  # CoSQA 单一 gold

    results["Total"] = total
    return results


# =========================
# 5. main（手动流程）
# =========================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch sees device_count =", torch.cuda.device_count())
    print("device name =", torch.cuda.get_device_name(0))

    # ---------
    # 5.1 构建模型（注意：和训练保持一致）
    # ---------
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
        prefix_length=16,
        # ✅ 训练开了就保持一致（即使推理我们会 dtype 修复）
        unfreeze_decoder_embed_and_lm_head=True,
    )

    # ---------
    # 5.2 加载 checkpoint
    # ---------
    ckpt_path = "/data/lizhen/CodeGR/output/gemkr_codebert_dsi/20260129113/checkpoint_epoch_4.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")

    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("Loaded checkpoint:", ckpt_path)
    print("Missing keys count:", len(missing))
    print("Unexpected keys count:", len(unexpected))

    model.to(device)
    model.eval()

    # ✅ 推理 dtype 修复
    fix_infer_dtypes(model)

    # ---------
    # 5.3 加载 CoSQA 训练集（先测训练 hit，验证是否学到了）
    # ---------
    data = load_cosqa_jsonl(
        "/data/lizhen/CodeGR/dataset/CoSQA_std/cosqa_train.jsonl",
        max_samples=None,   # 先全量；想快速看就填 200/1000
    )

    # ---------
    # 5.4 SANITY CHECK（看第一个样本）
    # ---------
    item = data[0]
    samples = {"query": [item["query"]], "code": [item["code"]]}

    preds = generate_docids(model, samples, num_beams=10, num_return_sequences=10)

    print("\n===== SANITY CHECK (TRAIN) =====")
    print("Gold docid:", item["docid"])
    print("Generated docids:")
    for p in preds:
        print(f"  [{p}]")
    print("================================\n")

    # ---------
    # 5.5 正式评测（TRAIN）
    # ---------
    results = evaluate_cosqa(model, data, device=device, ks=(1, 5, 10))

    print("\n===== Evaluation Results (TRAIN) =====")
    print(f"Total: {results['Total']}")
    for k in (1, 5, 10):
        print(f"HitCount@{k}: {results[f'HitCount@{k}']}")
        print(f"Hit@{k}: {results[f'Hit@{k}']:.4f}")
        print(f"Recall@{k}: {results[f'Recall@{k}']:.4f}")


if __name__ == "__main__":
    main()
