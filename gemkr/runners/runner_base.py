"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from gemkr.common.dist_utils import is_main_process, main_process
from gemkr.common.registry import registry
from gemkr.datasets.data_utils import reorg_datasets_by_split
from gemkr.datasets.datasets.dataloader_utils import IterLoader, PrefetchLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader


def _fallback_collate(batch: List[Any]) -> Any:
    """
    Safe collate for string-heavy samples.
    """
    if len(batch) == 0:
        return batch

    elem = batch[0]
    if isinstance(elem, dict):
        out: Dict[str, Any] = {}
        for k in elem.keys():
            out[k] = [b[k] for b in batch]
        return out

    if isinstance(elem, (list, tuple)):
        return [_fallback_collate([b[i] for b in batch]) for i in range(len(elem))]

    return batch


@registry.register_runner("runner_base")
class RunnerBase:
    """
    Modified runner with:
      - list/tuple dataset unwrap
      - safe collate_fn
      - per-epoch checkpoint saving
    """

    def __init__(self, cfg, task, model, datasets, job_id):
        self.config = cfg
        self.job_id = job_id

        self.task = task
        self.datasets = datasets

        self._model = model
        self._wrapped_model = None
        self._device = None
        self._optimizer = None
        self._scaler = None
        self._dataloaders = None
        self._lr_sched = None

        self.start_epoch = 0
        self.setup_output_dir()

    # ------------------------------------------------------------------
    # device / model
    # ------------------------------------------------------------------
    @property
    def device(self):
        if self._device is None:
            self._device = torch.device(self.config.run_cfg.device)
        return self._device

    @property
    def use_distributed(self):
        return bool(self.config.run_cfg.distributed)

    @property
    def model(self):
        if self._model.device != self.device:
            self._model = self._model.to(self.device)
            if self.use_distributed:
                if self._wrapped_model is None:
                    self._wrapped_model = DDP(self._model, device_ids=[self.config.run_cfg.gpu])
            else:
                self._wrapped_model = self._model
        return self._wrapped_model

    # ------------------------------------------------------------------
    def unwrap_dist_model(self, model):
        return model.module if self.use_distributed else model

    def _print_trainable_params(self, max_lines: int = 200):
        model_no_ddp = self.unwrap_dist_model(self.model)
        total, trainable = 0, 0
        lines = []
        for n, p in model_no_ddp.named_parameters():
            num = p.numel()
            total += num
            if p.requires_grad:
                trainable += num
                lines.append(f"[TRAIN] {n:70s} dtype={str(p.dtype):12s} shape={tuple(p.shape)} numel={num}")
        logging.info(f"Trainable params: {trainable} / {total} ({100.0 * trainable / max(total,1):.6f}%)")
        for s in lines[:max_lines]:
            logging.info(s)
        if len(lines) > max_lines:
            logging.info(f"... ({len(lines) - max_lines} more trainable params not shown)")

    # ------------------------------------------------------------------
    # optimizer / scaler / lr scheduler
    # ------------------------------------------------------------------
    @property
    def optimizer(self):
        if self._optimizer is None:
            # ensure model is moved to device / wrapped before scanning parameters
            _ = self.model

            p_wd, p_non_wd = [], []
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                    p_non_wd.append(p)
                else:
                    p_wd.append(p)

            optim_params = [
                {"params": p_wd, "weight_decay": float(self.config.run_cfg.weight_decay)},
                {"params": p_non_wd, "weight_decay": 0},
            ]
            beta2 = self.config.run_cfg.get("beta2", 0.999)
            self._optimizer = torch.optim.AdamW(
                optim_params,
                lr=float(self.config.run_cfg.init_lr),
                betas=(0.9, beta2),
            )
        return self._optimizer

    @property
    def scaler(self):
        if self.config.run_cfg.get("amp", False):
            if self._scaler is None:
                self._scaler = torch.cuda.amp.GradScaler()
        return self._scaler

    @property
    def lr_scheduler(self):
        if self._lr_sched is None:
            lr_sched_cls = registry.get_lr_scheduler_class(self.config.run_cfg.lr_sched)

            max_epoch = int(self.config.run_cfg.max_epoch)
            min_lr = float(self.config.run_cfg.min_lr)
            init_lr = float(self.config.run_cfg.init_lr)

            warmup_start_lr = self.config.run_cfg.get("warmup_lr", -1)
            warmup_steps = self.config.run_cfg.get("warmup_steps", 0)

            iters_per_epoch = self.config.run_cfg.get("iters_per_epoch", None)
            if iters_per_epoch is None:
                try:
                    iters_per_epoch = len(self.dataloaders["train"])
                except Exception:
                    iters_per_epoch = 10000

            self._lr_sched = lr_sched_cls(
                optimizer=self.optimizer,
                max_epoch=max_epoch,
                iters_per_epoch=iters_per_epoch,
                min_lr=min_lr,
                init_lr=init_lr,
                warmup_start_lr=warmup_start_lr,
                warmup_steps=warmup_steps,
            )
        return self._lr_sched

    # ------------------------------------------------------------------
    # dataloaders
    # ------------------------------------------------------------------
    @property
    def train_splits(self):
        return self.config.run_cfg.get("train_splits", [])

    @property
    def dataloaders(self):
        if self._dataloaders is None:
            logging.info("Building dataloaders")

            datasets = reorg_datasets_by_split(self.datasets)
            self.datasets = datasets

            split_names = sorted(datasets.keys())
            datasets_list = [datasets[s] for s in split_names]
            is_trains = [s in self.train_splits for s in split_names]

            batch_sizes = [
                self.config.run_cfg.batch_size_train if s == "train"
                else self.config.run_cfg.batch_size_eval
                for s in split_names
            ]

            loaders = []
            for ds, bsz, is_train in zip(datasets_list, batch_sizes, is_trains):
                if isinstance(ds, (list, tuple)):
                    if len(ds) == 0:
                        raise ValueError("Empty dataset container (list/tuple) encountered.")
                    base_ds = ds[0]
                else:
                    base_ds = ds

                collate_fn = getattr(base_ds, "collater", None)
                if collate_fn is None:
                    collate_fn = getattr(base_ds, "collate_fn", None)
                if collate_fn is None:
                    collate_fn = _fallback_collate

                loader = DataLoader(
                    base_ds,
                    batch_size=bsz,
                    shuffle=is_train,
                    num_workers=int(self.config.run_cfg.num_workers),
                    pin_memory=True,
                    drop_last=True if is_train else False,
                    collate_fn=collate_fn,
                )

                loader = PrefetchLoader(loader)
                if is_train:
                    loader = IterLoader(loader, use_distributed=False)

                loaders.append(loader)

            self._dataloaders = dict(zip(split_names, loaders))

        return self._dataloaders

    @property
    def train_loader(self):
        return self.dataloaders["train"]

    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------
    def setup_output_dir(self):
        lib_root = Path(registry.get_path("library_root"))
        output_dir = lib_root / self.config.run_cfg.output_dir / self.job_id
        result_dir = output_dir / "result"

        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        registry.register_path("result_dir", str(result_dir))
        registry.register_path("output_dir", str(output_dir))

        self.output_dir = output_dir

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def train(self):
        self.log_config()

        try:
            self._print_trainable_params()
        except Exception as e:
            logging.warning(f"Failed to print trainable params: {e}")

        for epoch in range(self.start_epoch, int(self.config.run_cfg.max_epoch)):
            logging.info(f"Epoch {epoch} start")

            # ✅ 兼容你的 YAML：gradient_accumulation_steps
            accum = (
                self.config.run_cfg.get("accum_grad_iters", None)
                or self.config.run_cfg.get("gradient_accumulation_steps", None)
                or 1
            )

            stats = self.task.train_epoch(
                epoch=epoch,
                model=self.model,
                data_loader=self.train_loader,
                optimizer=self.optimizer,
                scaler=self.scaler,
                lr_scheduler=self.lr_scheduler,
                cuda_enabled=self.device.type == "cuda",
                log_freq=self.config.run_cfg.get("log_freq", 50),
                accum_grad_iters=int(accum),
            )

            self.log_stats(stats, "train")

            if is_main_process():
                self._save_checkpoint(epoch)

        logging.info("Training finished.")

    # ------------------------------------------------------------------
    # checkpoint
    # ------------------------------------------------------------------
    @main_process
    def _save_checkpoint(self, cur_epoch: int):
        """
        ✅ 保存全量 checkpoint（不要再删冻结参数）
        否则 eval/load 时会 Missing keys 爆炸，模型退化成常量输出。
        """
        model_no_ddp = self.unwrap_dist_model(self.model)

        save_obj = {
            "model": model_no_ddp.state_dict(),  # ✅ 全量保存
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
            "config": self.config.to_dict(),
        }

        save_path = os.path.join(self.output_dir, f"checkpoint_epoch_{cur_epoch}.pth")
        logging.info(f"Saving FULL checkpoint to {save_path} (keys={len(save_obj['model'])})")
        torch.save(save_obj, save_path)

    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------
    @main_process
    def log_stats(self, stats, split_name):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps({f"{split_name}_{k}": v for k, v in stats.items()}) + "\n")

    @main_process
    def log_config(self):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self.config.to_dict(), indent=4) + "\n")
