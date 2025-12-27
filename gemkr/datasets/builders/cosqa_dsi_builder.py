from gemkr.common.registry import registry
from gemkr.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from gemkr.datasets.datasets.cosqa_dataset import CoSQADSIPretrainDataset


@registry.register_builder("cosqa_dsi")
class CoSQADSIPretrainBuilder(BaseDatasetBuilder):
    """
    DatasetBuilder for CoSQA DSI pretraining.

    Expected config format:
        datasets:
          cosqa_dsi:
            data_type: text
            build_info:
              ann_path: dataset/CoSQA/cosqa_train.jsonl
    """
    DATASET_CONFIG_DICT = {
        "default": "gemkr/configs/datasets/cosqa/defaults.yaml",
    }

    def __init__(self, config):
        super().__init__(config)

    def build_datasets(self):
        """
        Build and return datasets.

        Returns:
            dict: {"train": Dataset}
        """
        ann_path = self.config.build_info.ann_path

        dataset = CoSQADSIPretrainDataset(
            ann_path=ann_path
        )

        return {
            "train": dataset
        }
