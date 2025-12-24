from gemkr.common.registry import registry
from gemkr.tasks.base_task import BaseTask


@registry.register_task("dsi_pretrain")
class DSIPretrainTask(BaseTask):
    """
    DSI-style generative retrieval pretraining task.

    Dataset (via BaseTask.build_datasets):
        datasets = {
            "wiki": {
                "train": Dataset
            }
        }

    Each dataset sample contains:
        - query
        - code
        - answer_id
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _to_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]

    def _build_encoder_texts(self, samples):
        queries = self._to_list(samples.get("query"))
        codes = self._to_list(samples.get("code"))

        assert len(queries) == len(codes), (
            f"query/code size mismatch: {len(queries)} vs {len(codes)}"
        )

        return [
            f"Query: {q}\nCode:\n{c}"
            for q, c in zip(queries, codes)
        ]

    def train_step(self, model, samples):
        """
        BaseTask.train_epoch 会调用：
            loss = self.train_step(model, samples)
        并期望返回一个 scalar loss
        """

        # samples 已经经过 prepare_sample(cuda_enabled=...)
        samples = dict(samples)

        samples["encoder_texts"] = self._build_encoder_texts(samples)
        samples["answer_id"] = self._to_list(samples.get("answer_id"))

        out = model(samples)

        # 兼容两种 model 返回风格
        if isinstance(out, dict):
            return out["loss"]
        return out

    def valid_step(self, model, samples):
        return self.train_step(model, samples)
