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
    A safe collate_fn for string-heavy samples (e.g., {"query": str, "code": str, "answer_id": str}).
    It avoids torch default_collate which fails on strings / custom objects.

    Rules:
      - If elements are dict: collate to dict of lists
      - Else: return the list as-is
    """
    if len(batch) == 0:
        return batch

    elem = batch[0]
    if isinstance(elem, dict):
        out: Dict[str, Any] = {}
        keys = elem.keys()
        for k in keys:
            out[k] = [b[k] for b in batch]
        return out

    # If dataset returns tuples/lists, keep them grouped
    if isinstance(elem, (list, tuple)):
        # transpose
        return [ _fallback_collate([b[i] for b in batch]) for i in range(len(elem)) ]

    return batch


@registry.register_runner("runner_base")
class RunnerBase:
    """
    Modified runner with:
      - Correct handling of datasets wrapped in list/tuple (common in gemkr builders)
      - Always-valid collate_fn (uses dataset.collater if present; else fallback)
      - Per-epoch checkpoint saving
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
    # optimizer / scaler / lr scheduler
    # ------------------------------------------------------------------
    @property
    def optimizer(self):
        if self._optimizer is None:
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
                # ✅ 핵심修复1：处理 list/tuple 包装的 dataset
                # 你现在的报错就是 ds 是 [CoSQADSIPretrainDataset(...)] 这种情况
                if isinstance(ds, (list, tuple)):
                    if len(ds) == 0:
                        raise ValueError("Empty dataset container (list/tuple) encountered.")
                    base_ds = ds[0]
                else:
                    base_ds = ds

                # ✅ 핵심修复2：优先用框架 collater，否则用 fallback
                collate_fn = getattr(base_ds, "collater", None)
                if collate_fn is None:
                    # 有些实现用 collate_fn 命名
                    collate_fn = getattr(base_ds, "collate_fn", None)
                if collate_fn is None:
                    collate_fn = _fallback_collate

                loader = DataLoader(
                    base_ds,  # ✅ 必须是 Dataset，本体
                    batch_size=bsz,
                    shuffle=is_train,
                    num_workers=int(self.config.run_cfg.num_workers),
                    pin_memory=True,
                    drop_last=True if is_train else False,
                    collate_fn=collate_fn,
                )

                # 保持你原本的 Prefetch/IterLoader 逻辑
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

        for epoch in range(self.start_epoch, int(self.config.run_cfg.max_epoch)):
            logging.info(f"Epoch {epoch} start")

            stats = self.task.train_epoch(
                epoch=epoch,
                model=self.model,
                data_loader=self.train_loader,
                optimizer=self.optimizer,
                scaler=self.scaler,
                lr_scheduler=self.lr_scheduler,
                cuda_enabled=self.device.type == "cuda",
                log_freq=self.config.run_cfg.get("log_freq", 50),
                accum_grad_iters=self.config.run_cfg.get("accum_grad_iters", 1),
            )

            self.log_stats(stats, "train")

            if is_main_process():
                self._save_checkpoint(epoch)

        logging.info("Training finished.")

    # ------------------------------------------------------------------
    # checkpoint
    # ------------------------------------------------------------------
    def unwrap_dist_model(self, model):
        return model.module if self.use_distributed else model

    @main_process
    def _save_checkpoint(self, cur_epoch: int):
        model_no_ddp = self.unwrap_dist_model(self.model)

        param_grad = {k: v.requires_grad for k, v in model_no_ddp.named_parameters()}
        state_dict = model_no_ddp.state_dict()

        # 删除冻结参数（保持你原版逻辑）
        for k in list(state_dict.keys()):
            if k in param_grad and not param_grad[k]:
                del state_dict[k]

        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
            "config": self.config.to_dict(),
        }

        save_path = os.path.join(self.output_dir, f"checkpoint_epoch_{cur_epoch}.pth")
        logging.info(f"Saving checkpoint to {save_path}")
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
