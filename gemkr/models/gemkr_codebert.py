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

    This version is aligned with structured DOCID supervision:
        <DOCID> cosqa_XXXXXXXX </DOCID>
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
    ):
        super().__init__()

        # =============================== Encoder (CodeBERT) ===============================
        self.encoder = CodeBERTEncoder(
            model_name=encoder_name,
            freeze=freeze_encoder,
            prefix_length=prefix_length,  # pass prefix_length to encoder
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
        special_tokens = {
            "additional_special_tokens": ["<DOCID>", "</DOCID>"]
        }
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

        if freeze_decoder:
            for p in self.decoder.parameters():
                p.requires_grad = False

        # =============================== Alignment head ===============================
        self.hidden_size = int(hidden_size)
        self.alignment_proj = nn.Linear(
            self.encoder.hidden_size,
            self.hidden_size,
        )

    # ------------------------------------------------ Config-based construction ------------------------------------------------
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
        )

    # ------------------------------------------------ Helper: tokenize structured docids ------------------------------------------------
    def _tokenize_docids(self, answer_ids):
        return self.decoder_tokenizer(
            answer_ids,
            padding=True,
            truncation=True,
            max_length=self.decoder_max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )

    # ------------------------------------------------ Training forward ------------------------------------------------
    def forward(self, samples):
        device = self.device

        # 1) Encoder inputs
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

        # 2) Encode (B, L_enc, Hc)
        enc_hidden = self.encoder(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
        )

        # 3) Align to LLaMA hidden space (B, L_enc, H)
        encoder_lm_embeds = self.alignment_proj(enc_hidden).to(
            dtype=self.decoder.dtype
        )

        B, L_enc, _ = encoder_lm_embeds.shape

        # ===== Prefix attention mask (SAFE) =====
        prefix_attention_mask = torch.ones(
            (B, L_enc),
            device=device,
            dtype=torch.long,
        )

        # 4) Tokenize structured docids
        doc = self._tokenize_docids(samples["answer_id"])
        doc = {k: v.to(device) for k, v in doc.items()}

        doc_input_ids = doc["input_ids"]
        doc_attention_mask = doc.get("attention_mask", torch.ones_like(doc_input_ids))

        # 5) Doc embeddings
        embed_layer = self.decoder.get_input_embeddings()
        doc_embeds = embed_layer(doc_input_ids).to(dtype=self.decoder.dtype)

        # 6) Concat encoder and DOCID embeddings
        inputs_embeds = torch.cat(
            [encoder_lm_embeds, doc_embeds], dim=1
        )
        attention_mask = torch.cat(
            [prefix_attention_mask, doc_attention_mask], dim=1
        )

        # 7) Labels
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

        encoder_lm_embeds = self.alignment_proj(enc_hidden).to(
            dtype=self.decoder.dtype
        )

        B, L_enc, _ = encoder_lm_embeds.shape
        prefix_attention_mask = torch.ones(
            (B, L_enc),
            device=device,
            dtype=torch.long,
        )

        if "use_cache" not in gen_kwargs:
            gen_kwargs["use_cache"] = True

        return self.decoder.generate(
            inputs_embeds=encoder_lm_embeds,
            attention_mask=prefix_attention_mask,
            **gen_kwargs,
        )
