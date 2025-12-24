import torch
import torch.nn as nn
from transformers import RobertaModel


class CodeBERTEncoder(nn.Module):
    """
    CodeBERT encoder for (NL Query, Code) textual input.

    This module ONLY handles text encoding and outputs
    token-level hidden states.
    """

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        freeze: bool = True,
    ):
        super().__init__()

        self.model = RobertaModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size

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

        return outputs.last_hidden_state
