"""Stage 11: Training Script for Temporal Deep Learning Models (1D-TCN & Bi-GRU).

Trains both models under strictly controlled experimental conditions:
- Fixed seed: 42
- Optimizer: AdamW (lr=0.001, weight_decay=1e-4)
- Batch size: 32
- Early stopping: patience=15 on validation loss
- Class weights: J=1.2075, Z=0.8533
- Checkpoint targets:
    data/experiments/dynamic_jz_v1/stage11/best_tcn.pt
    data/experiments/dynamic_jz_v1/stage11/best_bigru.pt
- Training log saved to:
    data/experiments/dynamic_jz_v1/stage11/training_history.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference.dynamic_jz.stage11_models import (
    BiGRUNet,
    TemporalConvNet,
    count_parameters,
    LABEL_TO_IDX,
)


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SequenceDataset(Dataset):
    """PyTorch Dataset for (16, 69) feature sequences."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray) -> None:
        self.sequences = torch.from_numpy(sequences).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


def load_dataset(
    manifest_path: str | Path,
    jsonl_path: str | Path,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load train, val, test splits according to the split manifest."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        feature_data = {
            item["source_path"]: item for item in (json.loads(line) for line in f)
        }

    splits: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for split_name in ["train", "validation", "test"]:
        records = manifest["splits"][split_name]
        seq_list: List[np.ndarray] = []
        lbl_list: List[int] = []

        for rec in records:
            path = rec["source_path"]
            if path not in feature_data:
                raise KeyError(f"Source path {path} not found in jsonl features file!")
            feats = feature_data[path]["features"]
            label_str = rec["label"]
            seq_list.append(np.array(feats, dtype=np.float32))
            lbl_list.append(LABEL_TO_IDX[label_str])

        sequences = np.stack(seq_list, axis=0)  # (N, 16, 69)
        labels = np.array(lbl_list, dtype=np.int64)
        splits[split_name] = (sequences, labels)
        print(
            f"Loaded {split_name:10s}: {len(labels):3d} samples "
            f"(J={np.sum(labels == 0):2d}, Z={np.sum(labels == 1):2d})"
        )

    return splits


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Execute one training epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on a data split."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            logits = model(sequences)
            loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def run_training_experiment(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    checkpoint_path: Path,
    max_epochs: int = 100,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    patience: int = 15,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Train a single model with early stopping, saving best checkpoint."""
    param_count = count_parameters(model)
    print(f"\n=======================================================")
    print(f"Training {model_name} (Parameters: {param_count:,})")
    print(f"=======================================================")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_val_acc = 0.0
    epochs_no_improve = 0

    print("| Epoch | Train Loss | Val Loss | Train Accuracy | Val Accuracy |")
    print("|-------|------------|----------|----------------|--------------|")

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate_split(
            model, val_loader, criterion, device
        )

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
        }
        history.append(row)

        print(
            f"| {epoch:5d} | {train_loss:10.4f} | {val_loss:8.4f} | "
            f"{train_acc * 100:13.2f}% | {val_acc * 100:11.2f}% |"
        )

        # Early stopping check
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            # Save best checkpoint
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"\n[Early Stopping] No improvement in validation loss for "
                    f"{patience} epochs. Stopped at epoch {epoch}. "
                    f"Best epoch: {best_epoch} (val_loss={best_val_loss:.4f}, val_acc={best_val_acc * 100:.2f}%)"
                )
                break

    # Load best checkpoint back into model
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Re-evaluate final metrics on best model
    final_train_loss, final_train_acc = evaluate_split(
        model, train_loader, criterion, device
    )
    final_val_loss, final_val_acc = evaluate_split(
        model, val_loader, criterion, device
    )

    return {
        "model_name": model_name,
        "parameter_count": param_count,
        "checkpoint_path": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_acc": round(best_val_acc, 4),
        "final_train_loss": round(final_train_loss, 4),
        "final_train_acc": round(final_train_acc, 4),
        "final_val_loss": round(final_val_loss, 4),
        "final_val_acc": round(final_val_acc, 4),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 11 Temporal DL Models")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT
        / "data/experiments/dynamic_jz_binary_baseline_v1/split_manifest.json",
        help="Path to split_manifest.json",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=PROJECT_ROOT
        / "data/experiments/dynamic_jz_v1/processed/signnteam_jz_baseline_b_v1.jsonl",
        help="Path to signnteam_jz_baseline_b_v1.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "data/experiments/dynamic_jz_v1/stage11",
        help="Directory to save checkpoints and training log",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cpu")
    print(f"Using execution device: {device}")

    # Load splits
    splits = load_dataset(args.manifest, args.jsonl)
    train_x, train_y = splits["train"]
    val_x, val_y = splits["validation"]

    # Compute class weights: N / (C * N_c)
    n_train = len(train_y)
    n_j = int(np.sum(train_y == 0))
    n_z = int(np.sum(train_y == 1))
    weight_j = n_train / (2.0 * n_j)
    weight_z = n_train / (2.0 * n_z)
    class_weights = torch.tensor([weight_j, weight_z], dtype=torch.float32, device=device)
    print(f"\nClass weights: J={weight_j:.4f}, Z={weight_z:.4f} (Weights tensor: {class_weights.tolist()})")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_ds = SequenceDataset(train_x, train_y)
    val_ds = SequenceDataset(val_x, val_y)

    # Use a fixed generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, generator=g, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    results_all: Dict[str, Any] = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "max_epochs": args.epochs,
        "class_weights": {"J": round(weight_j, 4), "Z": round(weight_z, 4)},
        "splits_summary": {
            "train": {"total": n_train, "J": n_j, "Z": n_z},
            "validation": {
                "total": len(val_y),
                "J": int(np.sum(val_y == 0)),
                "Z": int(np.sum(val_y == 1)),
            },
        },
        "models": {},
    }

    # 1. Train Model A: 1D-TCN
    set_seed(args.seed)
    tcn_model = TemporalConvNet(input_dim=69, num_classes=2, hidden_channels=48, dropout=0.2).to(device)
    tcn_checkpoint = args.output_dir / "best_tcn.pt"
    tcn_res = run_training_experiment(
        model_name="1D-TCN",
        model=tcn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        checkpoint_path=tcn_checkpoint,
        max_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=device,
    )
    results_all["models"]["1D-TCN"] = tcn_res

    # 2. Train Model B: Bi-GRU
    set_seed(args.seed)
    bigru_model = BiGRUNet(input_dim=69, hidden_size=48, num_classes=2, dropout=0.2).to(device)
    bigru_checkpoint = args.output_dir / "best_bigru.pt"
    bigru_res = run_training_experiment(
        model_name="Bi-GRU",
        model=bigru_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        checkpoint_path=bigru_checkpoint,
        max_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=device,
    )
    results_all["models"]["Bi-GRU"] = bigru_res

    # Save history json
    history_path = args.output_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=2)
    print(f"\n[Training Complete] Results & history saved to {history_path}")


if __name__ == "__main__":
    main()
