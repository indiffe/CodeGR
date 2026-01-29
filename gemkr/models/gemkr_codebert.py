import torch
import torch.nn as nn
from transformers import RobertaTokenizer, LlamaTokenizer

from gemkr.common.registry import registry
from gemkr.models.base_model import BaseModel
from gemkr.models.modeling_llama import LlamaForCausalLM
from gemkr.models.codebert_encoder import CodeBERTEncoder


@registry.register_model("gemkr_codebert_dsi")
class GEMKRCodeBERTDSI(BaseModel):
    """
    CodeBERT -> token-level alignment -> LLaMA -> Generative Retrieval (DSI)

    Structured DOCID supervision:
        <DOCID> cosqa_XXXXXXXX </DOCID>

    生成侧关键修复：
      (1) eos_token_id 只用 </DOCID>（单个 int）
      (2) allowed token 集合覆盖 tokenizer 的“数字piece”
      (3) min_new_tokens=8

    ✅ 关键修复（输入不生效 / 全部输出一样）：
      - 不再直接调用 CodeBERTEncoder.forward（它可能只返回 prefix_embedding）
      - 改为从 CodeBERTEncoder 内部拿到底层 RoBERTa，使用 last_hidden_state（依赖输入）
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "gemkr_codebert_dsi": "gemkr/configs/models/gemkr_codebert.yaml"
    }

    def __init__(
        self,
        encoder_name: str,
        decoder_name: str,
        hidden_size: int,
        freeze_encoder: bool = True,
        freeze_decoder: bool = True,
        encoder_max_length: int = 512,
        decoder_max_length: int = 64,
        decoder_dtype: str = "float16",
        enable_gradient_checkpointing: bool = True,
        prefix_length: int = 10,
        docid_prefix: str = "cosqa_",
        # ✅ A方案开关：只训练 embed + lm_head
        unfreeze_decoder_embed_and_lm_head: bool = False,
    ):
        super().__init__()

        # ✅ 固定 prefix 长度
        self.prefix_length = int(prefix_length)

        # =============================== Encoder (CodeBERT) ===============================
        self.encoder = CodeBERTEncoder(
            model_name=encoder_name,
            freeze=freeze_encoder,
            prefix_length=prefix_length,
        )
        self.encoder_tokenizer = RobertaTokenizer.from_pretrained(encoder_name)
        self.encoder_max_length = int(encoder_max_length)

        # =============================== Decoder (LLaMA) ===============================
        self.decoder_max_length = int(decoder_max_length)

        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map.get(str(decoder_dtype).lower(), torch.float16)

        self.decoder = LlamaForCausalLM.from_pretrained(
            decoder_name,
            torch_dtype=torch_dtype,
        )

        self.decoder_tokenizer = LlamaTokenizer.from_pretrained(
            decoder_name, use_fast=False
        )
        if self.decoder_tokenizer.pad_token is None:
            self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token

        # ===== Register DOCID special tokens (CRITICAL) =====
        special_tokens = {"additional_special_tokens": ["<DOCID>", "</DOCID>"]}
        num_added = self.decoder_tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            self.decoder.resize_token_embeddings(len(self.decoder_tokenizer))

        # Training-time memory controls
        try:
            self.decoder.config.use_cache = False
        except Exception:
            pass

        if enable_gradient_checkpointing:
            try:
                self.decoder.gradient_checkpointing_enable()
            except Exception:
                try:
                    self.decoder.model.gradient_checkpointing_enable()
                except Exception:
                    pass

        # =============================== Freeze decoder by default ===============================
        if freeze_decoder:
            for p in self.decoder.parameters():
                p.requires_grad = False

        # =============================== ✅ A方案：解冻 embed + lm_head，并转 FP32 ===============================
        # 解决：GradScaler.step() 报 "Attempting to unscale FP16 gradients."
        if unfreeze_decoder_embed_and_lm_head:
            for p in self.decoder.parameters():
                p.requires_grad = False

            emb = self.decoder.get_input_embeddings()
            emb.to(dtype=torch.float32)
            for p in emb.parameters():
                p.requires_grad = True

            head = self.decoder.get_output_embeddings()
            head.to(dtype=torch.float32)
            for p in head.parameters():
                p.requires_grad = True

        # =============================== Alignment head ===============================
        self.hidden_size = int(hidden_size)
        self.alignment_proj = nn.Linear(
            self.encoder.hidden_size,  # CodeBERT hidden_size (usually 768)
            self.hidden_size,          # LLaMA hidden_size (e.g. 4096)
        )

        # =============================== DOCID prefix utils ===============================
        self.docid_prefix = docid_prefix

        self._fixed_prefix_text = f"<DOCID> {self.docid_prefix}"
        self._fixed_prefix_ids = self.decoder_tokenizer(
            self._fixed_prefix_text,
            add_special_tokens=False,
            return_tensors=None,
        )["input_ids"]  # List[int]

        self._allowed_digit_ids = self._build_allowed_digit_ids()

    @classmethod
    def from_config(cls, cfg):
        return cls(
            encoder_name=cfg.encoder_name,
            decoder_name=cfg.decoder_name,
            hidden_size=cfg.hidden_size,
            freeze_encoder=cfg.get("freeze_encoder", True),
            freeze_decoder=cfg.get("freeze_decoder", True),
            encoder_max_length=cfg.get("encoder_max_length", 512),
            decoder_max_length=cfg.get("decoder_max_length", 64),
            decoder_dtype=cfg.get("decoder_dtype", "float16"),
            enable_gradient_checkpointing=cfg.get("enable_gradient_checkpointing", True),
            prefix_length=cfg.get("prefix_length", 10),
            docid_prefix=cfg.get("docid_prefix", "cosqa_"),
            unfreeze_decoder_embed_and_lm_head=cfg.get("unfreeze_decoder_embed_and_lm_head", False),
        )

    # ------------------------------------------------ Helpers ------------------------------------------------
    def _ensure_docid_prefix(self, raw_id: str) -> str:
        s = str(raw_id).strip()
        if s.startswith("<DOCID>"):
            inner = s.replace("<DOCID>", "").replace("</DOCID>", "").strip()
            if not inner.startswith(self.docid_prefix):
                inner = self.docid_prefix + inner
            return f"<DOCID> {inner} </DOCID>"

        if not s.startswith(self.docid_prefix):
            s = self.docid_prefix + s
        return f"<DOCID> {s} </DOCID>"

    def _tokenize_docids(self, answer_ids):
        structured = [self._ensure_docid_prefix(x) for x in answer_ids]
        return self.decoder_tokenizer(
            structured,
            padding=True,
            truncation=True,
            max_length=self.decoder_max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )

    def _build_allowed_digit_ids(self):
        tok = self.decoder_tokenizer
        allowed = set()

        candidates = []
        for d in "0123456789":
            candidates += [d, " " + d]

        candidates += [
            "00", "000", "0000", "00000000",
            " 00", " 000", " 0000", " 00000000",
        ]

        for s in candidates:
            ids = tok(s, add_special_tokens=False)["input_ids"]
            for tid in ids:
                allowed.add(int(tid))

        end_id = tok.convert_tokens_to_ids("</DOCID>")
        if end_id is None or end_id < 0:
            raise ValueError("Cannot find token id for </DOCID> (did you add_special_tokens?)")
        allowed.add(int(end_id))

        banned = set(getattr(tok, "all_special_ids", []))
        for attr in ["bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"]:
            tid = getattr(tok, attr, None)
            if tid is not None:
                banned.add(int(tid))

        allowed = allowed - banned
        if len(allowed) == 0:
            raise ValueError("Allowed token set is empty after filtering specials.")

        return sorted(list(allowed))

    def _get_underlying_roberta(self):
        """
        ✅ 从 CodeBERTEncoder 中尽力拿到真正的 RoBERTa/CodeBERT backbone。
        你现在的问题大概率是 CodeBERTEncoder.forward() 只返回 prefix_embedding，
        所以我们绕开 forward，直接调用 backbone 得到 last_hidden_state（依赖输入）。
        """
        # 常见字段名尝试
        for attr in ["model", "encoder", "roberta", "backbone", "bert"]:
            if hasattr(self.encoder, attr):
                m = getattr(self.encoder, attr)
                if m is not None:
                    return m
        return None

    def _encode_with_roberta(self, input_ids, attention_mask):
        """
        返回 contextual hidden states: [B, L, 768]
        """
        roberta = self._get_underlying_roberta()
        if roberta is None:
            raise AttributeError(
                "Cannot access underlying RoBERTa from CodeBERTEncoder. "
                "Please expose it as .model or .encoder etc."
            )
        out = roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return out.last_hidden_state

    def _project_prefix(self, enc_hidden):
        """
        ✅ dtype 兜底：Linear(mat1/mat2) dtype 必须一致
        """
        proj_dtype = self.alignment_proj.weight.dtype
        if enc_hidden.dtype != proj_dtype:
            enc_hidden = enc_hidden.to(dtype=proj_dtype)

        encoder_lm_embeds = self.alignment_proj(enc_hidden)

        # decoder 侧通常是 fp16/bf16，prefix embeds 必须跟 decoder dtype 对齐
        encoder_lm_embeds = encoder_lm_embeds.to(dtype=self.decoder.dtype)
        return encoder_lm_embeds

    # ------------------------------------------------ Training forward ------------------------------------------------
    def forward(self, samples):
        device = self.device

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

        # ✅ 关键改动：不再 self.encoder(...)，而是直接 roberta last_hidden_state
        enc_hidden = self._encode_with_roberta(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
        )

        # 固定 prefix 长度
        if enc_hidden.size(1) > self.prefix_length:
            enc_hidden = enc_hidden[:, : self.prefix_length, :]

        # ✅ alignment + dtype 对齐
        encoder_lm_embeds = self._project_prefix(enc_hidden)
        B, L_enc, _ = encoder_lm_embeds.shape

        prefix_attention_mask = torch.ones(
            (B, L_enc),
            device=device,
            dtype=torch.long,
        )

        doc = self._tokenize_docids(samples["answer_id"])
        doc = {k: v.to(device) for k, v in doc.items()}

        doc_input_ids = doc["input_ids"]
        doc_attention_mask = doc.get("attention_mask", torch.ones_like(doc_input_ids))

        embed_layer = self.decoder.get_input_embeddings()
        doc_embeds = embed_layer(doc_input_ids).to(dtype=self.decoder.dtype)

        inputs_embeds = torch.cat([encoder_lm_embeds, doc_embeds], dim=1)
        attention_mask = torch.cat([prefix_attention_mask, doc_attention_mask], dim=1)

        L_doc = doc_input_ids.size(1)
        labels = torch.full(
            (B, L_enc + L_doc),
            fill_value=-100,
            dtype=torch.long,
            device=device,
        )
        labels[:, L_enc:] = doc_input_ids
        labels[labels == self.decoder_tokenizer.pad_token_id] = -100

        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        return outputs

    # ------------------------------------------------ Generation ------------------------------------------------
    @torch.no_grad()
    def generate(self, samples, **gen_kwargs):
        device = self.device
        tok = self.decoder_tokenizer

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

        # ✅ 关键改动：不再 self.encoder(...)，而是直接 roberta last_hidden_state
        enc_hidden = self._encode_with_roberta(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
        )

        if enc_hidden.size(1) > self.prefix_length:
            enc_hidden = enc_hidden[:, : self.prefix_length, :]

        # ✅ alignment + dtype 对齐
        encoder_lm_embeds = self._project_prefix(enc_hidden)
        B, L_enc, _ = encoder_lm_embeds.shape

        encoder_attention_mask = torch.ones((B, L_enc), device=device, dtype=torch.long)

        prompt_ids = torch.tensor(self._fixed_prefix_ids, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        embed_layer = self.decoder.get_input_embeddings()
        prompt_embeds = embed_layer(prompt_ids).to(dtype=self.decoder.dtype)
        prompt_attention_mask = torch.ones((B, prompt_ids.size(1)), device=device, dtype=torch.long)

        inputs_embeds = torch.cat([encoder_lm_embeds, prompt_embeds], dim=1)
        attention_mask = torch.cat([encoder_attention_mask, prompt_attention_mask], dim=1)

        end_id = tok.convert_tokens_to_ids("</DOCID>")
        if end_id is None or end_id < 0:
            raise ValueError("Cannot find token id for </DOCID> (did you add_special_tokens?)")

        gen_kwargs.setdefault("eos_token_id", int(end_id))
        gen_kwargs.setdefault("pad_token_id", int(tok.pad_token_id))
        gen_kwargs.setdefault("min_new_tokens", 8)

        gen_kwargs.setdefault("do_sample", False)
        gen_kwargs.setdefault("num_beams", 5)
        gen_kwargs.setdefault("temperature", 0.0)

        allowed_ids = self._allowed_digit_ids

        def prefix_allowed_tokens_fn(batch_id, input_ids):
            return allowed_ids

        gen_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn

        if "max_new_tokens" not in gen_kwargs and "max_length" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 16

        if "max_length" in gen_kwargs and "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = int(gen_kwargs.pop("max_length"))

        gen_kwargs.setdefault("use_cache", True)

        return self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
