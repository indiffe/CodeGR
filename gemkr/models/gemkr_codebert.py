# gemkr/models/gemkr_codebert_dsi.py
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
    CodeBERT -> (token-level) Alignment -> LLaMA -> Generative Retrieval (DSI)

    Alignment logic (MMGenerativeIR-style):
      - Encode (query, code) with CodeBERT to token embeddings (B, L, Hc)
      - Project every token embedding into LLaMA hidden space (B, L, Hllm)
      - Inject as prefix/context via LLaMA inputs_embeds (NOT prefix_len pooling)
      - Teacher forcing on docid tokens; encoder part labels are masked (-100)
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
    ):
        super().__init__()

        # -------------------------------
        # Encoder (CodeBERT)
        # -------------------------------
        self.encoder = CodeBERTEncoder(model_name=encoder_name, freeze=freeze_encoder)
        self.encoder_tokenizer = RobertaTokenizer.from_pretrained(encoder_name)
        self.encoder_max_length = int(encoder_max_length)

        # -------------------------------
        # Decoder (LLaMA)
        # -------------------------------
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

        self.decoder_tokenizer = LlamaTokenizer.from_pretrained(decoder_name, use_fast=False)
        if self.decoder_tokenizer.pad_token is None:
            self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token

        # training-time memory controls
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

        # -------------------------------
        # Alignment head (MM-style)
        # CodeBERT hidden -> LLaMA hidden
        # -------------------------------
        # IMPORTANT: hidden_size must match decoder hidden size (e.g., LLaMA-7B: 4096)
        self.hidden_size = int(hidden_size)
        self.alignment_proj = nn.Linear(self.encoder.hidden_size, self.hidden_size)

    # ------------------------------------------------
    # Config-based construction (REQUIRED by framework)
    # ------------------------------------------------
    @classmethod
    def from_config(cls, cfg):
        """
        The framework calls: model_cls.from_config(model_config)
        so this must exist.
        """
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

    # ------------------------------------------------
    # Helper: tokenize docids
    # ------------------------------------------------
    def _tokenize_docids(self, answer_ids):
        tok = self.decoder_tokenizer(
            answer_ids,
            padding=True,
            truncation=True,
            max_length=self.decoder_max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return tok

    # ------------------------------------------------
    # Training forward
    # ------------------------------------------------
    def forward(self, samples):
        """
        samples:
          {
            "query": List[str],
            "code": List[str],
            "answer_id": List[str]
          }
        """
        device = self.device

        # 1) Build encoder inputs
        queries = samples["query"]
        codes = samples["code"]
        encoder_texts = [f"Query: {q}\nCode:\n{c}" for q, c in zip(queries, codes)]

        enc = self.encoder_tokenizer(
            encoder_texts,
            padding=True,
            truncation=True,
            max_length=self.encoder_max_length,
            return_tensors="pt",
        ).to(device)

        # 2) Encode with CodeBERT: (B, L_enc, Hc)
        enc_hidden = self.encoder(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
        )

        # 3) Alignment (token-level): (B, L_enc, H_llama)
        encoder_lm_embeds = self.alignment_proj(enc_hidden).to(dtype=self.decoder.dtype)

        # 4) Tokenize DocID
        doc = self._tokenize_docids(samples["answer_id"])
        doc = {k: v.to(device) for k, v in doc.items()}

        doc_input_ids = doc["input_ids"]  # (B, L_doc)
        doc_attention_mask = doc.get("attention_mask", None)
        if doc_attention_mask is None:
            doc_attention_mask = torch.ones_like(doc_input_ids, device=device)

        # 5) Doc token embeddings
        embed_layer = self.decoder.get_input_embeddings()
        doc_embeds = embed_layer(doc_input_ids).to(dtype=self.decoder.dtype)  # (B, L_doc, H)

        # 6) Concat encoder aligned embeds + doc embeds
        inputs_embeds = torch.cat([encoder_lm_embeds, doc_embeds], dim=1)  # (B, L_enc+L_doc, H)

        # attention mask
        full_attention_mask = torch.cat([enc.attention_mask, doc_attention_mask], dim=1)

        # 7) Labels: encoder part = -100; doc part = doc_input_ids (pad -> -100)
        B = doc_input_ids.size(0)
        L_enc = encoder_lm_embeds.size(1)
        L_doc = doc_input_ids.size(1)

        labels = torch.full(
            (B, L_enc + L_doc),
            fill_value=-100,
            dtype=torch.long,
            device=device,
        )
        labels[:, L_enc:] = doc_input_ids
        labels[labels == self.decoder_tokenizer.pad_token_id] = -100

        # ensure cache off
        try:
            self.decoder.config.use_cache = False
        except Exception:
            pass

        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=labels,
            return_dict=True,
            use_cache=False,
        )
        return outputs

    # ------------------------------------------------
    # Generation (retrieval)
    # ------------------------------------------------
    @torch.no_grad()
    def generate(self, samples, **gen_kwargs):
        device = self.device

        queries = samples["query"]
        codes = samples["code"]
        encoder_texts = [f"Query: {q}\nCode:\n{c}" for q, c in zip(queries, codes)]

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

        encoder_lm_embeds = self.alignment_proj(enc_hidden).to(dtype=self.decoder.dtype)
        encoder_attention_mask = enc.attention_mask

        if "use_cache" not in gen_kwargs:
            gen_kwargs["use_cache"] = True

        outputs = self.decoder.generate(
            inputs_embeds=encoder_lm_embeds,
            attention_mask=encoder_attention_mask,
            **gen_kwargs,
        )
        return outputs
