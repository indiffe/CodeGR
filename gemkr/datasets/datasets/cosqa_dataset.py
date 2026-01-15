import json
import re
from pathlib import Path

from gemkr.datasets.datasets.base_dataset import BaseDataset


DOCID_PATTERN = re.compile(r"^cosqa_\d{8}$")


class CoSQADSIPretrainDataset(BaseDataset):
    """
    CoSQA Dataset for DSI-style generative retrieval pretraining.

    Each sample provides:
        {
            "query": str,
            "code": str,
            "answer_id": str   # structured docid
        }

    answer_id format (STRICT):
        <DOCID> cosqa_XXXXXXXX </DOCID>
    """

    def __init__(self, ann_path):
        super().__init__()

        self.ann_path = Path(ann_path)
        assert self.ann_path.exists(), f"Annotation file not found: {self.ann_path}"

        self.samples = []
        self._load_annotations()

    def _load_annotations(self):
        """
        Load jsonl annotations into memory with strict validation.
        """
        dropped = 0

        with self.ann_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                data = json.loads(line)

                # ---- required fields ----
                assert "query" in data, f"Missing 'query' at line {line_idx}"
                assert "code" in data, f"Missing 'code' at line {line_idx}"
                assert "docid" in data, f"Missing 'docid' at line {line_idx}"

                docid = data["docid"].strip()

                # ---- strict docid validation ----
                if not DOCID_PATTERN.match(docid):
                    dropped += 1
                    continue

                # ---- structured supervision target ----
                answer_id = f"<DOCID> {docid} </DOCID>"

                self.samples.append(
                    {
                        "query": data["query"],
                        "code": data["code"],
                        "answer_id": answer_id,
                    }
                )

        if dropped > 0:
            print(
                f"[CoSQADSIPretrainDataset] Dropped {dropped} samples due to invalid docid format."
            )

        assert len(self.samples) > 0, "No valid samples loaded after docid filtering."

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    @staticmethod
    def collater(batch):
        """
        Collate function for DSI pretraining.
        """
        return {
            "query": [b["query"] for b in batch],
            "code": [b["code"] for b in batch],
            "answer_id": [b["answer_id"] for b in batch],
        }
