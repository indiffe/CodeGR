import torch
import torch.nn as nn

from gemkr.common.registry import registry
from gemkr.models.base_model import BaseModel
from gemkr.models.modeling_llama import LlamaForCausalLM
from gemkr.models.codebert_encoder import CodeBERTEncoder


@registry.register_model("gemkr_codebert_dsi")
class GEMKRCodeBERTDSI(BaseModel):
    """
    CodeBERT + LLaMA generative retrieval model (DSI-style).

    Pipeline:
        (NL Query, Code)
            -> CodeBERT Encoder
            -> Linear Projection (prefix)
            -> LLaMA Decoder
            -> Autoregressive DocID generation
    """

    def __init__(
        self,
        encoder_name: str,
        decoder_name: str,
        hidden_size: int,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        # =========================
        # Encoder (CodeBERT)
        # =========================
        self.encoder = CodeBERTEncoder(
            model_name=encoder_name,
            freeze=freeze_encoder,
        )

        # =========================
        # Prefix Projection
        # =========================
        self.prefix_proj = nn.Linear(
            self.encoder.hidden_size,
            hidden_size,
            bias=False,
        )

        # =========================
        # Decoder (LLaMA)
        # =========================
        self.decoder = LlamaForCausalLM.from_pretrained(
            decoder_name,
            torch_dtype=torch.float16,
        )

        self.hidden_size = hidden_size

    # ------------------------------------------------
    # Config-based construction (required by BaseTask)
    # ------------------------------------------------
    @classmethod
    def from_config(cls, cfg):
        return cls(
            encoder_name=cfg.encoder_name,
            decoder_name=cfg.decoder_name,
            hidden_size=cfg.hidden_size,
            freeze_encoder=cfg.get("freeze_encoder", True),
        )

    # ------------------------------------------------
    # Training forward (teacher forcing)
    # ------------------------------------------------
    def forward(
        self,
        encoder_input_ids,
        encoder_attention_mask,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
    ):
        """
        Args:
            encoder_input_ids: (B, L_enc)
            encoder_attention_mask: (B, L_enc)
            decoder_input_ids: (B, L_dec)
            labels: (B, L_dec)

        Returns:
            HuggingFace CausalLMOutputWithCrossAttentions
        """

        # 1. Encode query + code
        enc_hidden = self.encoder(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
        )  # (B, L_enc, Hc)

        # 2. Project to LLaMA embedding space
        prefix_embeds = self.prefix_proj(enc_hidden)  # (B, L_enc, Hl)

        # 3. LLaMA forward
        outputs = self.decoder(
            inputs_embeds=prefix_embeds,
            attention_mask=encoder_attention_mask,
            input_ids=decoder_input_ids,
            labels=labels,
            return_dict=True,
        )

        return outputs

    # ------------------------------------------------
    # DSI-style generation (retrieval)
    # ------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        encoder_input_ids,
        encoder_attention_mask,
        **gen_kwargs,
    ):
        """
        Autoregressive DocID generation.
        """

        enc_hidden = self.encoder(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
        )

        prefix_embeds = self.prefix_proj(enc_hidden)

        outputs = self.decoder.generate(
            inputs_embeds=prefix_embeds,
            attention_mask=encoder_attention_mask,
            **gen_kwargs,
        )

        return outputs
