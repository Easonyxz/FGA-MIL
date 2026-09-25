
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


IS_TTY = sys.stdout.isatty()


def torch_load(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def as_feature_tensor(obj) -> torch.Tensor:
    if torch.is_tensor(obj):
        features = obj
    elif isinstance(obj, dict):
        features = next(
            (
                obj[key]
                for key in ("features", "feats", "x", "embeddings", "embedding")
                if key in obj and torch.is_tensor(obj[key])
            ),
            None,
        )
        if features is None:
            features = next(
                (value for value in obj.values() if torch.is_tensor(value) and value.ndim >= 2),
                None,
            )
    elif isinstance(obj, (tuple, list)):
        features = next(
            (value for value in obj if torch.is_tensor(value) and value.ndim >= 2),
            None,
        )
    else:
        raise TypeError(f"Unsupported feature object: {type(obj).__name__}")

    if features is None:
        raise ValueError("The feature file does not contain a tensor")
    if features.ndim == 3 and features.shape[0] == 1:
        features = features.squeeze(0)
    if features.ndim != 2:
        raise ValueError(f"Expected a [num_instances, feature_dim] tensor, got {features.shape}")
    return features.float()


def load_feature_file(path: str | Path) -> torch.Tensor:
    return as_feature_tensor(torch_load(path))


def get_feature_dirs(config: dict) -> list[str]:
    directories = config.get("feature_dirs", config.get("feature_dir", []))
    if isinstance(directories, (str, Path)):
        directories = [directories]
    return [str(directory) for directory in directories]


def resolve_feature_paths(df: pd.DataFrame, config: dict) -> dict[str, str]:
    directories = get_feature_dirs(config)
    if not directories:
        raise ValueError("Set feature_dir in the config or pass --feature-dir")

    paths = {}
    for case_id in df["case_id"].astype(str).unique():
        for directory in directories:
            candidate = os.path.join(directory, f"{case_id}.pt")
            if os.path.isfile(candidate):
                paths[case_id] = candidate
                break

    missing = sorted(set(df["case_id"].astype(str)) - set(paths))
    if missing:
        print(
            f"Warning: {len(missing)} feature bags are missing; "
            f"first missing IDs: {missing[:10]}"
        )
    return paths


def load_feature_cache(
    feature_paths: dict[str, str], config: dict
) -> tuple[dict[str, torch.Tensor], bool]:
    if not config.get("use_cache", False):
        return {}, False

    required = sum(os.path.getsize(path) for path in feature_paths.values())
    available = psutil.virtual_memory().available
    margin = config.get("memory_safety_margin", 0.8)
    print(f"Estimated feature cache size: {required / 1024**3:.2f} GB")
    if required > available * margin:
        print("Insufficient free memory; loading feature bags on demand.")
        return {}, False

    cache = {}
    iterator = tqdm(
        feature_paths.items(),
        desc="Caching features",
        disable=not IS_TTY,
    )
    for case_id, path in iterator:
        cache[case_id] = load_feature_file(path)
    return cache, True


class MILDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_paths: dict[str, str],
        feature_cache: dict[str, torch.Tensor],
        use_cache: bool,
        label_map: dict[str, int],
    ):
        case_ids = df["case_id"].astype(str)
        self.df = df[case_ids.isin(feature_paths)].copy().reset_index(drop=True)
        self.df["case_id"] = self.df["case_id"].astype(str)
        labels = self.df["label"].astype(str).map(label_map)
        if labels.isna().any():
            unknown = sorted(self.df.loc[labels.isna(), "label"].astype(str).unique())
            raise ValueError(f"Labels missing from label_map: {unknown}")
        self.labels = labels.to_numpy(dtype=np.int64)
        self.feature_paths = feature_paths
        self.feature_cache = feature_cache
        self.use_cache = use_cache

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        case_id = self.df.iloc[index]["case_id"]
        if self.use_cache:
            features = self.feature_cache[case_id]
        else:
            features = load_feature_file(self.feature_paths[case_id])
        return features, self.labels[index], case_id


def collate_variable_bags(batch):
    feature_bags = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    case_ids = [item[2] for item in batch]
    return feature_bags, labels, case_ids


def create_train_val_test_splits(df: pd.DataFrame, config: dict):
    group_column = config.get("group_column", "case_id")
    if group_column not in df.columns:
        raise ValueError(f"group_column '{group_column}' is absent from the metadata CSV")

    num_test_splits = round(1 / config["test_split_ratio"])
    outer_splitter = StratifiedGroupKFold(
        n_splits=num_test_splits,
        shuffle=True,
        random_state=config["seed"],
    )
    outer_splits = outer_splitter.split(df, df["label"], df[group_column])

    for fold in range(config["n_outer_folds"]):
        try:
            train_val_indices, test_indices = next(outer_splits)
        except StopIteration as error:
            raise ValueError(
                "n_outer_folds cannot exceed round(1 / test_split_ratio)"
            ) from error

        train_val_df = df.iloc[train_val_indices]
        relative_val_ratio = config["val_split_ratio"] / (
            1 - config["test_split_ratio"]
        )
        inner_splitter = StratifiedGroupKFold(
            n_splits=round(1 / relative_val_ratio),
            shuffle=True,
            random_state=config["seed"] + fold,
        )
        train_local, val_local = next(
            inner_splitter.split(
                train_val_df,
                train_val_df["label"],
                train_val_df[group_column],
            )
        )
        yield train_val_indices[train_local], train_val_indices[val_local], test_indices


def _print_distribution(dataset: MILDataset, name: str) -> None:
    distribution = dataset.df["label"].astype(str).value_counts().to_dict()
    print(f"{name:<5}: {len(dataset):>5} bags {distribution}")


def create_data_loaders(
    df: pd.DataFrame,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    config: dict,
    feature_paths: dict[str, str],
    feature_cache: dict[str, torch.Tensor],
    use_cache: bool,
):
    datasets = [
        MILDataset(
            df.iloc[indices].reset_index(drop=True),
            feature_paths,
            feature_cache,
            use_cache,
            config["label_map"],
        )
        for indices in (train_indices, val_indices, test_indices)
    ]
    train_dataset, val_dataset, test_dataset = datasets
    if any(len(dataset) == 0 for dataset in datasets):
        raise ValueError("At least one data split is empty after matching feature files")

    input_dim = train_dataset[0][0].shape[1]
    sampler = None
    shuffle = True
    if config.get("use_weighted_sampler", False):
        counts = np.bincount(
            train_dataset.labels, minlength=config["num_classes"]
        )
        class_weights = np.zeros_like(counts, dtype=np.float64)
        class_weights[counts > 0] = 1.0 / counts[counts > 0]
        sample_weights = torch.as_tensor(class_weights[train_dataset.labels])
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        shuffle = False

    common = {
        "batch_size": config["batch_size"],
        "collate_fn": collate_variable_bags,
        "num_workers": config.get("num_workers", 0),
        "pin_memory": config.get("pin_memory", False),
    }
    train_loader = DataLoader(
        train_dataset, shuffle=shuffle, sampler=sampler, **common
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)

    print(f"Input feature dimension: {input_dim}")
    for dataset, name in zip(datasets, ("Train", "Val", "Test")):
        _print_distribution(dataset, name)
    return train_loader, val_loader, test_loader, input_dim, train_dataset.df


def save_split_metadata(
    df: pd.DataFrame,
    split_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    output_path: str | Path,
) -> None:
    frames = []
    for name, indices in zip(("train", "val", "test"), split_indices):
        frame = df.iloc[indices].copy()
        frame.insert(0, "split", name)
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(output_path, index=False)
