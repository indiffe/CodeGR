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
    CodeBERT + LLaMA generative retrieval model (DSI-style).

    Key memory fix vs. previous version:
      - DO NOT use full encoder sequence (e.g., 512) as prefix tokens.
      - Pool encoder outputs -> project into K prefix tokens (prefix_len, e.g., 16).
      - Optionally freeze decoder; enable gradient checkpointing; disable KV cache in training.
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
        prefix_len: int = 16,
        encoder_max_length: int = 512,
        decoder_max_length: int = 64,
        decoder_dtype: str = "float16",
        enable_gradient_checkpointing: bool = True,
    ):
        super().__init__()

        # =========================
        # Encoder (CodeBERT)
        # =========================
        self.encoder = CodeBERTEncoder(
            model_name=encoder_name,
            freeze=freeze_encoder,
        )
        self.encoder_tokenizer = RobertaTokenizer.from_pretrained(encoder_name)

        self.encoder_max_length = int(encoder_max_length)
        self.decoder_max_length = int(decoder_max_length)

        # =========================
        # Prefix: pool -> K prefix tokens
        # =========================
        self.prefix_len = int(prefix_len)
        self.hidden_size = int(hidden_size)

        # lightweight nonlinearity after pooling (helps stability)
        self.enc_pooler = nn.Sequential(
            nn.Linear(self.encoder.hidden_size, self.encoder.hidden_size),
            nn.Tanh(),
        )
        # project pooled vector (B, Hc) -> (B, K*H_llama) -> reshape (B, K, H_llama)
        self.prefix_proj = nn.Linear(
            self.encoder.hidden_size,
            self.prefix_len * self.hidden_size,
            bias=False,
        )

        # =========================
        # Decoder (LLaMA)
        # =========================
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

        self.decoder_tokenizer = LlamaTokenizer.from_pretrained(decoder_name)
        if self.decoder_tokenizer.pad_token is None:
            self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token

        # training-time memory controls
        # IMPORTANT: KV cache must be OFF for training.
        try:
            self.decoder.config.use_cache = False
        except Exception:
            pass

        if enable_gradient_checkpointing:
            # HF supports this method on PreTrainedModel
            try:
                self.decoder.gradient_checkpointing_enable()
            except Exception:
                # some wrappers expose it on .model
                try:
                    self.decoder.model.gradient_checkpointing_enable()
                except Exception:
                    pass

        if freeze_decoder:
            for p in self.decoder.parameters():
                p.requires_grad = False

    # ------------------------------------------------
    # Config-based construction
    # ------------------------------------------------
    @classmethod
    def from_config(cls, cfg):
        return cls(
            encoder_name=cfg.encoder_name,
            decoder_name=cfg.decoder_name,
            hidden_size=cfg.hidden_size,
            freeze_encoder=cfg.get("freeze_encoder", True),
            freeze_decoder=cfg.get("freeze_decoder", True),
            prefix_len=cfg.get("prefix_len", 16),
            encoder_max_length=cfg.get("encoder_max_length", 512),
            decoder_max_length=cfg.get("decoder_max_length", 64),
            decoder_dtype=cfg.get("decoder_dtype", "float16"),
            enable_gradient_checkpointing=cfg.get("enable_gradient_checkpointing", True),
        )

    # ------------------------------------------------
    # DocID → decoder tokenization helper
    # ------------------------------------------------
    def _tokenize_docids(self, answer_ids):
        """
        Tokenize DocID strings into decoder input_ids and attention_mask.
        """
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
    # Pool encoder outputs -> (B, Hc)
    # ------------------------------------------------
    def _masked_mean_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (B, L, H)
        attention_mask: (B, L) with 1 for tokens, 0 for pads
        returns: (B, H)
        """
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # (B, L, 1)
        denom = mask.sum(dim=1).clamp_min(1.0)                       # (B, 1)
        pooled = (hidden_states * mask).sum(dim=1) / denom           # (B, H)
        return pooled

    # ------------------------------------------------
    # Training forward (Prefix-LM teacher forcing)
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

        # 2) Encode with CodeBERT: (B, L, Hc)
        enc_hidden = self.encoder(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
        )

        # 3) Pool -> K prefix tokens: (B, K, H_llama)
        pooled = self._masked_mean_pool(enc_hidden, enc.attention_mask)  # (B, Hc)
        pooled = self.enc_pooler(pooled)                                 # (B, Hc)
        prefix_flat = self.prefix_proj(pooled)                           # (B, K*H)
        prefix_embeds = prefix_flat.view(pooled.size(0), self.prefix_len, self.hidden_size)

        # Cast to decoder dtype (important when decoder is fp16/bf16)
        prefix_embeds = prefix_embeds.to(dtype=self.decoder.dtype)

        # 4) Tokenize DocID
        doc = self._tokenize_docids(samples["answer_id"])
        doc = {k: v.to(device) for k, v in doc.items()}

        doc_input_ids = doc["input_ids"]  # (B, L_doc)
        doc_attention_mask = doc.get("attention_mask", None)
        if doc_attention_mask is None:
            doc_attention_mask = torch.ones_like(doc_input_ids, device=device)

        # Get decoder token embeddings safely
        embed_layer = self.decoder.get_input_embeddings()
        doc_embeds = embed_layer(doc_input_ids).to(dtype=self.decoder.dtype)  # (B, L_doc, H)

        # 5) Concat prefix + doc embeds
        inputs_embeds = torch.cat([prefix_embeds, doc_embeds], dim=1)  # (B, K+L_doc, H)

        # attention mask: prefix is all-ones
        prefix_attention_mask = torch.ones(
            (inputs_embeds.size(0), self.prefix_len),
            device=device,
            dtype=doc_attention_mask.dtype,
        )
        full_attention_mask = torch.cat([prefix_attention_mask, doc_attention_mask], dim=1)  # (B, K+L_doc)

        # 6) Labels: prefix part = -100, doc part = doc_input_ids (pad -> -100)
        B = doc_input_ids.size(0)
        L_doc = doc_input_ids.size(1)
        L_pref = self.prefix_len

        labels = torch.full(
            (B, L_pref + L_doc),
            fill_value=-100,
            dtype=torch.long,
            device=device,
        )
        labels[:, L_pref:] = doc_input_ids
        labels[labels == self.decoder_tokenizer.pad_token_id] = -100

        # 7) LLaMA forward (teacher forcing)
        # Ensure training-time cache is off (some wrappers may toggle it)
        try:
            self.decoder.config.use_cache = False
        except Exception:
            pass

        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=labels,
            return_dict=True,
            use_cache=False,  # explicit
        )
        return outputs

    # ------------------------------------------------
    # DSI-style generation (retrieval)
    # ------------------------------------------------
    @torch.no_grad()
    def generate(self, samples, **gen_kwargs):
        """
        Autoregressive DocID generation conditioned on pooled prefix embeds.
        """
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

        pooled = self._masked_mean_pool(enc_hidden, enc.attention_mask)
        pooled = self.enc_pooler(pooled)
        prefix_flat = self.prefix_proj(pooled)
        prefix_embeds = prefix_flat.view(pooled.size(0), self.prefix_len, self.hidden_size)
        prefix_embeds = prefix_embeds.to(dtype=self.decoder.dtype)

        prefix_attention_mask = torch.ones(
            (prefix_embeds.size(0), self.prefix_len),
            device=device,
            dtype=enc.attention_mask.dtype,
        )

        # For generation, ensure cache can be on (it helps speed). Let user override.
        if "use_cache" not in gen_kwargs:
            gen_kwargs["use_cache"] = True

        outputs = self.decoder.generate(
            inputs_embeds=prefix_embeds,
            attention_mask=prefix_attention_mask,
            **gen_kwargs,
        )
        return outputs
