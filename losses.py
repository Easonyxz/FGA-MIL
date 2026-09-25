
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if alpha is None:
            self.alpha = None
        else:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(
            logits, targets, reduction="none", label_smoothing=self.label_smoothing
        )
        loss = (1 - torch.exp(-cross_entropy) + 1e-7) ** self.gamma * cross_entropy
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        return loss.mean()


def compute_class_weights(
    labels, num_classes: int, beta: float = 0.999, multiplier: float = 1.0
) -> torch.Tensor:
    counts = np.array([(labels == index).sum() for index in range(num_classes)])
    effective_numbers = np.where(counts > 0, 1.0 - np.power(beta, counts), 0.0)
    total = effective_numbers.sum()
    weights = np.ones(num_classes, dtype=np.float64)
    present = effective_numbers > 0
    weights[present] = total / (num_classes * effective_numbers[present])
    return torch.tensor(weights, dtype=torch.float32) * multiplier


def create_criterion(config: dict, train_df, device: torch.device) -> nn.Module:
    label_smoothing = config.get("label_smoothing", 0.0)
    if config.get("bag_loss", "cross_entropy") != "focal_loss":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)

    labels = train_df["label"].astype(str).map(config["label_map"])
    if labels.isna().any():
        unknown = sorted(train_df.loc[labels.isna(), "label"].astype(str).unique())
        raise ValueError(f"Labels missing from label_map: {unknown}")

    alpha = compute_class_weights(
        labels,
        config["num_classes"],
        multiplier=config.get("positive_weight_multiplier", 1.0),
    )
    counts = [(labels == index).sum() for index in range(config["num_classes"])]
    print(f"Class counts: {counts}")
    print(f"Class weights: {alpha.tolist()}")
    return FocalLoss(
        gamma=config.get("focal_loss_gamma", 2.0),
        alpha=alpha,
        label_smoothing=label_smoothing,
    ).to(device)
