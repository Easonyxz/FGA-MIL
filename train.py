import argparse
import copy
import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from data import (
    create_data_loaders,
    create_train_val_test_splits,
    load_feature_cache,
    resolve_feature_paths,
    save_split_metadata,
)
from losses import create_criterion
from model import FGAMIL
from optimizers import MuonWithAuxAdam
from utils import (
    EarlyStopping,
    calculate_metrics,
    load_state_dict,
    save_results,
    seed_everything,
)


IS_TTY = sys.stdout.isatty()


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_parameter, parameter in zip(
            self.model.parameters(), model.parameters()
        ):
            ema_parameter.lerp_(parameter, 1.0 - self.decay)
        for ema_buffer, buffer in zip(self.model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def create_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    optimizer_type = config.get("optimizer", "muon_adam").lower()
    if optimizer_type == "muon_adam":
        matrix_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.ndim >= 2
        ]
        auxiliary_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.ndim < 2
        ]
        return MuonWithAuxAdam(
            [
                {
                    "params": matrix_parameters,
                    "use_muon": True,
                    "lr": config["muon_lr"],
                    "momentum": config["muon_momentum"],
                    "weight_decay": config["weight_decay"],
                },
                {
                    "params": auxiliary_parameters,
                    "use_muon": False,
                    "lr": config["adam_lr"],
                    "betas": (0.9, 0.95),
                    "weight_decay": config["weight_decay"],
                },
            ]
        )
    if optimizer_type == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config["adam_lr"],
            weight_decay=config["weight_decay"],
        )
    if optimizer_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config["adam_lr"],
            weight_decay=config["weight_decay"],
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_type}")


def create_scheduler(optimizer, config):
    warmup_epochs = config["warmup_epochs"]
    if warmup_epochs == 0:
        return CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-6)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"] - warmup_epochs,
        eta_min=1e-6,
    )
    return SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_epochs]
    )


def create_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
    epoch,
    config,
    ema,
):
    model.train()
    totals = {"loss": 0.0, "classification": 0.0, "flow": 0.0, "aux": 0.0}
    all_labels, all_probabilities = [], []
    flow_criterion = nn.MSELoss()
    amp_enabled = config.get("use_amp", True) and device.type == "cuda"

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch + 1} [train]",
        leave=False,
        dynamic_ncols=True,
        disable=not IS_TTY,
    )
    for feature_bags, labels, _ in progress:
        feature_bags = [features.to(device, non_blocking=True) for features in feature_bags]
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type, enabled=amp_enabled, dtype=torch.float16
        ):
            logits, flow_predictions, flow_targets = model(feature_bags, labels)
            classification_loss = criterion(logits, labels)
            flow_loss = (
                flow_criterion(flow_predictions, flow_targets)
                if flow_predictions is not None
                else logits.new_zeros(())
            )
            auxiliary_losses = model.get_auxiliary_losses()
            loss = (
                classification_loss
                + config.get("flow_loss_weight", 0.15) * flow_loss
                + auxiliary_losses["total"]
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.get("gradient_clip", 1.0))
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)

        totals["loss"] += loss.item()
        totals["classification"] += classification_loss.item()
        totals["flow"] += flow_loss.item()
        totals["aux"] += auxiliary_losses["total"].item()
        all_labels.extend(labels.detach().cpu().numpy())
        all_probabilities.extend(F.softmax(logits, dim=1).detach().cpu().numpy())

    batches = len(loader)
    metrics = calculate_metrics(np.asarray(all_labels), np.asarray(all_probabilities))
    return {key: value / batches for key, value in totals.items()}, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, config, phase="val", epoch=0):
    model.eval()
    total_loss = 0.0
    all_labels, all_probabilities = [], []
    amp_enabled = config.get("use_amp", True) and device.type == "cuda"
    progress = tqdm(
        loader,
        desc=f"Epoch {epoch + 1} [{phase}]",
        leave=False,
        dynamic_ncols=True,
        disable=not IS_TTY,
    )
    for feature_bags, labels, _ in progress:
        feature_bags = [features.to(device, non_blocking=True) for features in feature_bags]
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, enabled=amp_enabled, dtype=torch.float16
        ):
            logits, _, _ = model(feature_bags)
            if criterion is not None:
                total_loss += criterion(logits, labels).item()
        all_labels.extend(labels.cpu().numpy())
        all_probabilities.extend(F.softmax(logits, dim=1).cpu().numpy())

    average_loss = total_loss / len(loader) if criterion is not None else 0.0
    metrics = calculate_metrics(np.asarray(all_labels), np.asarray(all_probabilities))
    return average_loss, metrics


def validate_config(config: dict) -> None:
    required = {
        "csv_path",
        "checkpoint_dir",
        "experiment_name",
        "label_map",
        "hidden_dim",
        "batch_size",
        "epochs",
        "seed",
        "n_outer_folds",
        "test_split_ratio",
        "val_split_ratio",
        "warmup_epochs",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Missing config fields: {missing}")
    expected_labels = list(range(len(config["label_map"])))
    if sorted(config["label_map"].values()) != expected_labels:
        raise ValueError(f"label_map values must be contiguous: {expected_labels}")
    if config["epochs"] <= config["warmup_epochs"]:
        raise ValueError("epochs must be greater than warmup_epochs")
    if config.get("num_archetypes", 8) < 2:
        raise ValueError("num_archetypes must be at least 2")


def build_model(input_dim: int, config: dict, device: torch.device) -> FGAMIL:
    parameters = inspect.signature(FGAMIL.__init__).parameters
    model_kwargs = {
        name: config[name]
        for name in parameters
        if name not in {"self", "input_dim"} and name in config
    }
    return FGAMIL(input_dim=input_dim, **model_kwargs).to(device)


def run(config: dict) -> None:
    validate_config(config)
    config["num_classes"] = len(config["label_map"])
    device = torch.device(
        config.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; pass --device cpu")
    seed_everything(config["seed"])

    output_dir = Path(config["checkpoint_dir"]) / config["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)

    metadata = pd.read_csv(config["csv_path"])
    required_columns = {"case_id", "label"}
    if not required_columns.issubset(metadata.columns):
        raise ValueError(f"Metadata CSV must contain {sorted(required_columns)}")
    feature_paths = resolve_feature_paths(metadata, config)
    if not feature_paths:
        raise FileNotFoundError("No feature bags matched metadata case_id values")
    feature_cache, use_cache = load_feature_cache(feature_paths, config)

    fold_results = []
    training_history = []
    splits = list(create_train_val_test_splits(metadata, config))
    for fold, split_indices in enumerate(splits, start=1):
        fold_start = time.time()
        print(f"\n{'=' * 20} Fold {fold}/{len(splits)} {'=' * 20}")
        save_split_metadata(metadata, split_indices, output_dir / f"fold_{fold}_split.csv")
        train_loader, val_loader, test_loader, input_dim, train_df = create_data_loaders(
            metadata,
            *split_indices,
            config,
            feature_paths,
            feature_cache,
            use_cache,
        )

        model = build_model(input_dim, config, device)
        ema_decay = config.get("ema_decay", 0.999)
        ema = ModelEMA(model, ema_decay) if ema_decay > 0 else None
        evaluation_model = ema.model if ema is not None else model
        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(optimizer, config)
        criterion = create_criterion(config, train_df, device)
        scaler = create_grad_scaler(
            config.get("use_amp", True) and device.type == "cuda"
        )
        checkpoint_path = output_dir / f"fold_{fold}_best.pt"
        early_stopping = EarlyStopping(
            checkpoint_path,
            patience=config.get("early_stopping_patience", 20),
            min_epochs=config.get("min_epochs", 15),
            warmup_epochs=config["warmup_epochs"],
        )

        for epoch in range(config["epochs"]):
            train_losses, train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler,
                epoch,
                config,
                ema,
            )
            val_loss, val_metrics = evaluate(
                evaluation_model, val_loader, criterion, device, config, "val", epoch
            )
            scheduler.step()
            current_lr = "/".join(
                f"{group['lr']:.6g}" for group in optimizer.param_groups
            )
            print(
                f"Epoch {epoch + 1:03d} | lr {current_lr} | "
                f"train loss/AUC {train_losses['loss']:.4f}/{train_metrics['auc']:.4f} | "
                f"val loss/AUC/Bacc {val_loss:.4f}/{val_metrics['auc']:.4f}/"
                f"{val_metrics['bacc']:.4f} | cls {train_losses['classification']:.4f} | "
                f"flow {train_losses['flow']:.4f} | aux {train_losses['aux']:.4f}"
            )
            training_history.append(
                {
                    "fold": fold,
                    "epoch": epoch + 1,
                    "train_loss": train_losses["loss"],
                    "train_auc": train_metrics["auc"],
                    "val_loss": val_loss,
                    "val_auc": val_metrics["auc"],
                    "val_bacc": val_metrics["bacc"],
                }
            )
            pd.DataFrame(training_history).to_csv(
                output_dir / "training_log.csv", index=False
            )
            early_stopping(
                val_metrics["auc"], epoch + 1, evaluation_model, val_loss
            )
            if early_stopping.early_stop:
                break

        if not checkpoint_path.exists():
            raise RuntimeError(f"No checkpoint was saved for fold {fold}")
        model.load_state_dict(load_state_dict(checkpoint_path, device))
        _, test_metrics = evaluate(
            model, test_loader, None, device, config, "test", early_stopping.best_epoch - 1
        )
        fold_results.append(
            {
                "fold": fold,
                **test_metrics,
                "best_val_auc": early_stopping.best_score,
                "best_epoch": early_stopping.best_epoch,
                "fold_time_min": (time.time() - fold_start) / 60,
            }
        )
        print(
            f"Fold {fold} test: AUC {test_metrics['auc']:.4f}, "
            f"Bacc {test_metrics['bacc']:.4f}, Acc {test_metrics['accuracy']:.4f}"
        )

    save_results(fold_results, output_dir)


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    for key in ("csv_path", "checkpoint_dir"):
        if key not in config:
            continue
        value = Path(config[key]).expanduser()
        if not value.is_absolute():
            config[key] = str((path.parent / value).resolve())
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Train FGA-MIL")
    parser.add_argument("--config", required=True, help="JSON experiment config")
    parser.add_argument(
        "--feature-dir",
        action="append",
        help="Directory containing <case_id>.pt bags; repeat for multiple directories",
    )
    parser.add_argument(
        "--csv-path", required=True, help="Metadata CSV containing case_id and label"
    )
    parser.add_argument("--output-dir", help="Override the output root")
    parser.add_argument("--device", help="Override the PyTorch device, e.g. cuda:0 or cpu")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    experiment_config = load_config(arguments.config)
    if arguments.feature_dir:
        experiment_config.pop("feature_dir", None)
        experiment_config["feature_dirs"] = arguments.feature_dir
    if arguments.csv_path:
        experiment_config["csv_path"] = os.path.abspath(arguments.csv_path)
    if arguments.output_dir:
        experiment_config["checkpoint_dir"] = os.path.abspath(arguments.output_dir)
    if arguments.device:
        experiment_config["device"] = arguments.device
    run(experiment_config)
