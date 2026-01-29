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
import types
import torch
from tqdm import tqdm

from gemkr.models.gemkr_codebert import GEMKRCodeBERTDSI


def load_cosqa_jsonl(path, max_samples=None):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
            if max_samples is not None and len(data) >= max_samples:
                break
    return data


def normalize_docid(pred: str) -> str:
    if pred is None:
        return pred
    pred = str(pred).strip()

    m = re.search(r"(cosqa_\d{8})", pred)
    if m:
        return m.group(1)

    m2 = re.search(r"(\d{8})", pred)
    if m2:
        return "cosqa_" + m2.group(1)

    return pred


@torch.no_grad()
def generate_docids(
    model,
    samples,
    num_beams=10,
    num_return_sequences=10,
    do_sample=False,
    temperature=1.0,
    top_p=0.95,
    max_new_tokens=10,
    min_new_tokens=8,
):
    end_id = model.decoder_tokenizer.convert_tokens_to_ids("</DOCID>")
    if end_id is None or end_id < 0:
        end_id = model.decoder_tokenizer.eos_token_id

    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        eos_token_id=int(end_id),
        pad_token_id=int(model.decoder_tokenizer.pad_token_id),
        use_cache=True,
        min_new_tokens=int(min_new_tokens),
    )

    if do_sample:
        gen_kwargs.update(
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            num_return_sequences=int(num_return_sequences),
        )
    else:
        gen_kwargs.update(
            num_beams=int(num_beams),
            num_return_sequences=int(num_return_sequences),
            early_stopping=True,
        )

    outputs = model.generate(samples, **gen_kwargs)

    raw_texts, norm_docids = [], []
    for seq in outputs:
        text = model.decoder_tokenizer.decode(
            seq,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        raw_texts.append(text)
        norm_docids.append(normalize_docid(text))

    return norm_docids, raw_texts


def fix_infer_dtypes(model):
    """
    保证推理不因 dtype 混乱报错：
      - alignment_proj: FP32（匹配 encoder 输出通常 FP32）
      - decoder + emb + lm_head: FP16（匹配 decoder hidden_states 通常 FP16）
    """
    model.alignment_proj.to(dtype=torch.float32)

    model.decoder.to(dtype=torch.float16)
    emb = model.decoder.get_input_embeddings()
    head = model.decoder.get_output_embeddings()
    emb.to(dtype=torch.float16)
    head.to(dtype=torch.float16)
    if hasattr(model.decoder, "lm_head") and model.decoder.lm_head is not None:
        model.decoder.lm_head.to(dtype=torch.float16)

    try:
        print("[DTYPE CHECK]",
              "alignment_proj:", model.alignment_proj.weight.dtype,
              "| decoder(any):", next(model.decoder.parameters()).dtype,
              "| emb:", next(model.decoder.get_input_embeddings().parameters()).dtype,
              "| lm_head:", next(model.decoder.get_output_embeddings().parameters()).dtype)
    except Exception as e:
        print("Could not inspect dtypes:", repr(e))


def attach_generate_probe(model, print_limit=50):
    """
    monkey-patch model.generate：
      - 在真正调用原 generate 前，计算 enc_hidden / encoder_lm_embeds
      - 打印 variance / std / mean 等统计，帮助定位“全一样”是否来自 conditioning 恒定
    """
    original_generate = model.generate

    state = {"count": 0}

    @torch.no_grad()
    def generate_with_probe(self, samples, **gen_kwargs):
        # 只打印前 print_limit 条，避免刷屏
        do_print = state["count"] < int(print_limit)
        state["count"] += 1

        if do_print:
            try:
                device = self.device
                # 1) 复现 gemkr_codebert.py generate() 中的 encoder_texts / tokenizer / encoder 前向
                encoder_texts = [
                    f"Query: {q}\nCode:\n{c}"
                    for q, c in zip(samples["query"], samples["code"])
                ]

                enc = self.encoder_tokenizer(
                    encoder_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.encoder_max_length,
                    return_tensors="pt",
                ).to(device)

                enc_hidden = self.encoder(
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                )

                if enc_hidden.size(1) > self.prefix_length:
                    enc_hidden = enc_hidden[:, : self.prefix_length, :]

                # 2) alignment_proj dtype 对齐（避免 matmul dtype mismatch）
                proj_dtype = self.alignment_proj.weight.dtype
                if enc_hidden.dtype != proj_dtype:
                    enc_hidden_proj = enc_hidden.to(dtype=proj_dtype)
                else:
                    enc_hidden_proj = enc_hidden

                encoder_lm_embeds = self.alignment_proj(enc_hidden_proj)  # dtype = proj_dtype
                # gemkr_codebert.py 里会 .to(decoder.dtype)，但我们先看投影本身是否有差异

                # 3) 统计：enc_hidden 和 encoder_lm_embeds 的 variance / std
                def stats(x: torch.Tensor):
                    x_f = x.float()
                    return {
                        "dtype": str(x.dtype),
                        "shape": tuple(x.shape),
                        "mean": float(x_f.mean().item()),
                        "std": float(x_f.std(unbiased=False).item()),
                        "min": float(x_f.min().item()),
                        "max": float(x_f.max().item()),
                    }

                # token维度方差：先对 hidden 维求均值，再看 token 维方差（反映不同 token 信息量）
                # 这里主要看 encoder_lm_embeds 是否“几乎常量”
                x = encoder_lm_embeds.float()  # [B, L, H]
                token_var = x.var(dim=1, unbiased=False).mean().item()  # avg over H,B
                hidden_var = x.var(dim=2, unbiased=False).mean().item() # avg over L,B
                global_var = x.var(unbiased=False).item()

                # alignment_proj 权重范数（看是否接近 0 或非常小）
                w = self.alignment_proj.weight.detach().float()
                w_norm = w.norm().item()
                w_std = w.std(unbiased=False).item()

                # 打印前两个 token 前8维，直观看是否“每条样本都相同”
                # 只取 batch[0]
                preview = None
                if x.numel() > 0:
                    b0 = x[0]
                    t0 = b0[0, :8].tolist() if b0.size(0) > 0 else []
                    t1 = b0[1, :8].tolist() if b0.size(0) > 1 else []
                    preview = {"token0[:8]": t0, "token1[:8]": t1}

                print("\n========== PROBE ==========")
                print(f"[probe #{state['count']-1}] query_head:", samples["query"][0][:80].replace("\n", " "))
                print("[enc_hidden]", stats(enc_hidden))
                print("[encoder_lm_embeds]", stats(encoder_lm_embeds))
                print(f"[vars] global_var={global_var:.6e} | token_var(avgH,B)={token_var:.6e} | hidden_var(avgL,B)={hidden_var:.6e}")
                print(f"[alignment_proj] weight_norm={w_norm:.6e} | weight_std={w_std:.6e}")
                if preview is not None:
                    print("[preview] token0[:8] =", ["{:+.3e}".format(v) for v in preview["token0[:8]"]])
                    print("[preview] token1[:8] =", ["{:+.3e}".format(v) for v in preview["token1[:8]"]])
                print("===========================\n")

            except Exception as e:
                print("[PROBE ERROR]", repr(e))

        return original_generate(samples, **gen_kwargs)

    model.generate = types.MethodType(generate_with_probe, model)
    return original_generate  # 返回原 generate（需要时可还原）


@torch.no_grad()
def dump_generated_docids(
    model,
    data,
    output_path,
    device="cuda",
    num_beams=10,
    num_return_sequences=10,
    enable_sampling_debug=True,
    temperature=1.0,
    top_p=0.95,
    print_every=20,
):
    model.eval()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 对照：unconstrained（禁用 prefix_allowed_tokens_fn）
    original_generate = model.generate

    def generate_without_constraints(self, samples, **gen_kwargs):
        if "prefix_allowed_tokens_fn" in gen_kwargs:
            gen_kwargs.pop("prefix_allowed_tokens_fn")
        return original_generate(samples, **gen_kwargs)

    patched_generate_unconstrained = types.MethodType(generate_without_constraints, model)

    with open(output_path, "w", encoding="utf-8") as fout:
        for idx, item in enumerate(tqdm(data, desc="Generating DocIDs (constrained vs unconstrained)")):
            samples = {"query": [item["query"]], "code": [item["code"]]}

            record = {
                "index": idx,
                "query": item.get("query", None),
                "gold_docid": item.get("docid", None),
            }

            # A) constrained（带 probe 的 generate）
            constrained_beam_docids, constrained_beam_raw = generate_docids(
                model, samples,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                do_sample=False,
            )
            record["constrained_beam_generated_docids"] = constrained_beam_docids
            record["constrained_beam_raw_texts"] = constrained_beam_raw

            if enable_sampling_debug:
                constrained_samp_docids, constrained_samp_raw = generate_docids(
                    model, samples,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                )
                record["constrained_sample_generated_docids"] = constrained_samp_docids
                record["constrained_sample_raw_texts"] = constrained_samp_raw

            # B) unconstrained（禁用 token 约束，但仍然走 probe 包裹过的 generate）
            model.generate = patched_generate_unconstrained

            unconstrained_beam_docids, unconstrained_beam_raw = generate_docids(
                model, samples,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                do_sample=False,
                max_new_tokens=16,
                min_new_tokens=1,
            )
            record["unconstrained_beam_generated_docids"] = unconstrained_beam_docids
            record["unconstrained_beam_raw_texts"] = unconstrained_beam_raw

            if enable_sampling_debug:
                unconstrained_samp_docids, unconstrained_samp_raw = generate_docids(
                    model, samples,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=16,
                    min_new_tokens=1,
                )
                record["unconstrained_sample_generated_docids"] = unconstrained_samp_docids
                record["unconstrained_sample_raw_texts"] = unconstrained_samp_raw

            # 还原 generate（带 probe 的 constrained 版本）
            model.generate = original_generate

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            if print_every > 0 and (idx % print_every == 0):
                print("\n==============================")
                print(f"[{idx}] Gold:", record["gold_docid"])
                print("[A] Constrained BEAM:", record["constrained_beam_generated_docids"][:5])
                print("[B] Unconstrained BEAM:", record["unconstrained_beam_generated_docids"][:5])
                print("==============================")

    print("\nAll generated docids saved to:")
    print(output_path)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch sees device_count =", torch.cuda.device_count())
    print("device name =", torch.cuda.get_device_name(0))

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
        unfreeze_decoder_embed_and_lm_head=True,
    )

    ckpt_path = "/data/lizhen/CodeGR/output/gemkr_codebert_dsi/20260129113/checkpoint_epoch_4.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("Loaded checkpoint:", ckpt_path)
    print("Missing keys count:", len(missing))
    print("Unexpected keys count:", len(unexpected))

    model.to(device)
    model.eval()

    # 保证推理 dtype 不炸
    fix_infer_dtypes(model)

    # ✅ attach probe：打印 encoder_lm_embeds 的 variance
    _orig = attach_generate_probe(model, print_limit=50)

    data = load_cosqa_jsonl(
        "/data/lizhen/CodeGR/dataset/CoSQA_std/cosqa_valid.jsonl",
        max_samples=None,
    )

    output_path = (
        "/data/lizhen/CodeGR/output/gemkr_codebert_dsi/"
        "cosqa_valid_generated_docids.contrast.debug.jsonl"
    )

    dump_generated_docids(
        model=model,
        data=data,
        output_path=output_path,
        device=device,
        num_beams=10,
        num_return_sequences=10,
        enable_sampling_debug=True,
        temperature=1.0,
        top_p=0.95,
        print_every=20,
    )


if __name__ == "__main__":
    main()

