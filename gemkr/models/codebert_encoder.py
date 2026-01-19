import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer


class CodeBERTEncoder(nn.Module):
    """
    CodeBERT encoder for (NL Query, Code) textual input.

    Responsibilities (GEMKR-compatible):
        - own tokenizer
        - encode text into contextual token embeddings
    """

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        freeze: bool = True,
        prefix_length: int = 10,
    ):
        super().__init__()

        # -------------------------
        # Tokenizer（关键补齐）
        # -------------------------
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)

        # -------------------------
        # Encoder
        # -------------------------
        self.model = RobertaModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size

        # 为每一层增加可学习的前缀
        self.prefix_length = prefix_length
        self.prefix_embedding = nn.Parameter(torch.randn(self.prefix_length, self.hidden_size))

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: LongTensor, shape (B, L)
            attention_mask: LongTensor, shape (B, L)

        Returns:
            last_hidden_state: FloatTensor, shape (B, L, H)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        # 在CodeBERT的输入中加入前缀
        prefix = self.prefix_embedding.unsqueeze(0).expand(input_ids.size(0), -1, -1)
        hidden_states = torch.cat([prefix, outputs.last_hidden_state], dim=1)

        return hidden_states
