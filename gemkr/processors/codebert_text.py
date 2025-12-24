"""
CodeBERT text processor.

This processor formats (NL query, code) into a single textual sequence
and tokenizes it using a CodeBERT-compatible tokenizer.
"""

from transformers import RobertaTokenizer
from gemkr.processors.base_processor import BaseProcessor
from gemkr.common.registry import registry


@registry.register_processor("codebert_text")
class CodeBERTTextProcessor(BaseProcessor):
    """
    Text processor for CodeBERT-based models.

    Input sample format:
        {
            "query": str,
            "code": str
        }

    Output:
        {
            "input_ids": List[int],
            "attention_mask": List[int]
        }
    """

    def __init__(
        self,
        pretrained_model_name: str = "microsoft/codebert-base",
        max_length: int = 512,
    ):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(
            pretrained_model_name,
            use_fast=True,
        )
        self.max_length = max_length

    def __call__(self, sample):
        """
        Args:
            sample (dict): one data sample containing at least
                - query (str)
                - code (str)

        Returns:
            dict with tokenized inputs for CodeBERT
        """

        query = sample.get("query", "")
        code = sample.get("code", "")

        # Stable and explicit formatting (recommended for DSI)
        text = f"Query: {query}\nCode:\n{code}"

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors=None,
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }

    @classmethod
    def from_config(cls, cfg=None):
        """
        Build processor from YAML config.
        """
        if cfg is None:
            return cls()

        return cls(
            pretrained_model_name=cfg.get(
                "pretrained_model_name", "microsoft/codebert-base"
            ),
            max_length=cfg.get("max_length", 512),
        )
