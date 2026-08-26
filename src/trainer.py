import os
import json
import time
import math
import torch
import torch.nn as nn


class CosineScheduler:
    def __init__(self, optimizer, warmup_steps, max_steps, min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        lr_scale = 1.0
        if self.step_count < self.warmup_steps:
            lr_scale = self.step_count / max(1, self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            lr_scale = self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * lr_scale

    def get_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


class KortexTrainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        device,
        config,
        checkpoint_dir="checkpoints",
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.checkpoint_dir = checkpoint_dir

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.start_time = time.time()
        self.checkpoints = []

        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, step, val_loss=None, is_best=False):
        checkpoint = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": self.config.to_dict() if hasattr(self.config, "to_dict") else self.config,
        }

        if val_loss is not None:
            checkpoint["val_loss"] = val_loss

        if is_best:
            path = os.path.join(self.checkpoint_dir, "best.pt")
            torch.save(checkpoint, path)
            print(f"  [BEST] Saved best checkpoint at step {step} with val_loss={val_loss:.4f}")

        path = os.path.join(self.checkpoint_dir, f"step_{step}.pt")
        torch.save(checkpoint, path)
        self.checkpoints.append((step, path, val_loss or float("inf")))
        self._cleanup_checkpoints()
        print(f"  Saved checkpoint at step {step} -> {path}")
        return path

    def _cleanup_checkpoints(self):
        keep = self.config.get("save_top_k", 3) if isinstance(self.config, dict) else getattr(self.config, "save_top_k", 3)
        if len(self.checkpoints) > keep + 1:
            self.checkpoints.sort(key=lambda x: x[2])
            while len(self.checkpoints) > keep:
                step, path, _ = self.checkpoints.pop(0)
                if os.path.exists(path):
                    os.remove(path)
                    print(f"  Removed old checkpoint: {path}")

    def load_checkpoint(self, path=None):
        if path is None:
            path = self._find_latest_checkpoint()
        if path is None:
            print("No checkpoint found, starting from scratch.")
            return 0

        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"  Resumed from step {self.global_step}, best_val_loss={self.best_val_loss:.4f}")
        return self.global_step

    def _find_latest_checkpoint(self):
        best_path = os.path.join(self.checkpoint_dir, "best.pt")
        if os.path.exists(best_path):
            return best_path

        step_files = []
        for f in os.listdir(self.checkpoint_dir):
            if f.startswith("step_") and f.endswith(".pt"):
                try:
                    step_num = int(f.split("_")[1].split(".")[0])
                    step_files.append((step_num, os.path.join(self.checkpoint_dir, f)))
                except (ValueError, IndexError):
                    continue

        if step_files:
            step_files.sort(key=lambda x: x[0], reverse=True)
            return step_files[0][1]
        return None

    def time_remaining(self):
        elapsed = time.time() - self.start_time
        max_seconds = self.config.get("max_time_hours", 8.5) * 3600 if isinstance(self.config, dict) else getattr(self.config, "max_time_hours", 8.5) * 3600
        return max(0, max_seconds - elapsed)

    def time_expired(self):
        return self.time_remaining() <= 0

    def format_time(self, seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def train_step(self, x, y):
        self.model.train()
        x = x.to(self.device)
        y = y.to(self.device)

        with torch.amp.autocast(device_type="xla", dtype=torch.bfloat16, enabled=True):
            _, loss = self.model(x, targets=y)

        loss.backward()

        if hasattr(self, "_grad_accum_count"):
            self._grad_accum_count += 1
        else:
            self._grad_accum_count = 1

        grad_accum = self.config.get("gradient_accumulation_steps", 16) if isinstance(self.config, dict) else getattr(self.config, "gradient_accumulation_steps", 16)

        if self._grad_accum_count % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get("max_grad_norm", 1.0) if isinstance(self.config, dict) else getattr(self.config, "max_grad_norm", 1.0),
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            self.global_step += 1

        return loss.item()

    def evaluate(self, eval_fn, eval_steps=50):
        self.model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for i in range(eval_steps):
                batch = eval_fn()
                if batch is None:
                    break
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                with torch.amp.autocast(device_type="xla", dtype=torch.bfloat16, enabled=True):
                    _, loss = self.model(x, targets=y)
                total_loss += loss.item()
                count += 1
        return total_loss / max(count, 1)
