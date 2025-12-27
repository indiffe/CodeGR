import json
from pathlib import Path

from gemkr.datasets.datasets.base_dataset import BaseDataset


class CoSQADSIPretrainDataset(BaseDataset):
    """
    CoSQA Dataset for DSI-style generative retrieval pretraining.

    Each sample provides:
        {
            "query": str,
            "code": str,
            "answer_id": str
        }

    Data source:
        - jsonl file produced by process_cosqa_dsi.py
        - one sample per line
    """

    def __init__(self, ann_path):
        """
        Args:
            ann_path (str or Path):
                Path to processed CoSQA DSI jsonl file.
        """
        super().__init__()

        self.ann_path = Path(ann_path)
        assert self.ann_path.exists(), f"Annotation file not found: {self.ann_path}"

        self.samples = []
        self._load_annotations()

    def _load_annotations(self):
        """
        Load jsonl annotations into memory.
        """
        with self.ann_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                data = json.loads(line)

                # 强校验，防止 silent error
                assert "query" in data, f"Missing 'query' at line {line_idx}"
                assert "code" in data, f"Missing 'code' at line {line_idx}"
                assert "docid" in data, f"Missing 'docid' at line {line_idx}"

                self.samples.append(
                    {
                        "query": data["query"],
                        "code": data["code"],
                        "answer_id": data["docid"],  # DSI supervision target
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Return a single training sample.
        """
        return self.samples[index]

    @staticmethod
    def collater(batch):
        """
        Collate function for DSI pretraining.

        This is the ONLY correct place to define batching logic
        in GEMKR / LAVIS-style datasets.
        """
        return {
            "query": [b["query"] for b in batch],
            "code": [b["code"] for b in batch],
            "answer_id": [b["answer_id"] for b in batch],
        }
