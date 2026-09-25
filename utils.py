
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_state_dict(path: str | Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


class EarlyStopping:
    def __init__(
        self,
        path: str | Path,
        patience: int = 20,
        min_epochs: int = 10,
        warmup_epochs: int = 5,
        delta: float = 0.0,
    ):
        self.path = Path(path)
        self.patience = patience
        self.min_epochs = max(min_epochs, warmup_epochs)
        self.warmup_epochs = warmup_epochs
        self.delta = delta
        self.counter = 0
        self.best_score = -np.inf
        self.best_val_loss = np.inf
        self.best_epoch = 0
        self.early_stop = False

    def __call__(
        self,
        score: float,
        epoch: int,
        model: torch.nn.Module,
        val_loss: float,
    ) -> None:
        if epoch <= self.warmup_epochs:
            return
        improved = score > self.best_score + self.delta
        improved |= score == self.best_score and val_loss < self.best_val_loss
        if improved:
            self.counter = 0
            self.best_score = score
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.path)
            print(f"Validation AUC improved to {score:.4f}; saved {self.path}")
            return

        self.counter += 1
        if self.counter >= self.patience and epoch >= self.min_epochs:
            self.early_stop = True
            print(f"Early stopping at epoch {epoch}")


def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0 or not np.isfinite(y_prob).all():
        return {
            key: 0.0
            for key in (
                "auc",
                "bacc",
                "accuracy",
                "f1",
                "sensitivity",
                "specificity",
                "npv",
            )
        }

    num_classes = y_prob.shape[1]
    y_pred = y_prob.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
    }
    if num_classes == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics.update(
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
            sensitivity=tp / (tp + fn) if tp + fn else 0.0,
            specificity=tn / (tn + fp) if tn + fp else 0.0,
            npv=tn / (tn + fn) if tn + fn else 0.0,
        )
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        except ValueError:
            metrics["auc"] = 0.0
    else:
        metrics["f1"] = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )
        per_class_auc = []
        for class_index in range(num_classes):
            binary_targets = (y_true == class_index).astype(int)
            if np.unique(binary_targets).size == 2:
                per_class_auc.append(
                    roc_auc_score(binary_targets, y_prob[:, class_index])
                )
        metrics["auc"] = float(np.mean(per_class_auc)) if per_class_auc else 0.0
        metrics.update(sensitivity=0.0, specificity=0.0, npv=0.0)
    return metrics


RESULT_COLUMNS = [
    "fold",
    "auc",
    "bacc",
    "accuracy",
    "f1",
    "sensitivity",
    "specificity",
    "npv",
    "best_val_auc",
    "best_epoch",
    "fold_time_min",
]


def save_results(results: list[dict], output_dir: str | Path) -> None:
    if not results:
        raise RuntimeError("Training finished without a valid fold result")

    output_dir = Path(output_dir)
    frame = pd.DataFrame(results)[RESULT_COLUMNS]
    frame.to_csv(output_dir / "results_per_fold.csv", index=False, float_format="%.4f")
    summary = frame.drop(columns="fold").agg(["mean", "std"]).T
    summary["mean_std"] = summary.apply(
        lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}", axis=1
    )
    summary.to_csv(output_dir / "results_summary.csv", float_format="%.4f")
    print("\nPer-fold results")
    print(frame.to_string(index=False, float_format="%.4f"))
    print("\nMean ± standard deviation")
    print(summary[["mean_std"]].to_string())
