"""Public minimal evaluation script for HypnoConv-BF-v1."""

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from dataset_public import SequenceDataset, build_windows, load_arrays
from model_bf import CLASS_NAMES, NUM_CLASSES, create_model


def compute_metrics(y_true, y_pred):
    per_class_f1_raw = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "per_class_f1": {
            CLASS_NAMES[i]: float(per_class_f1_raw[i]) for i in range(min(NUM_CLASSES, len(per_class_f1_raw)))
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_y, all_pred = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        center = x.shape[1] // 2
        preds = logits[:, center, :].argmax(dim=-1).cpu().numpy()
        all_pred.append(preds)
        all_y.append(y[:, center].cpu().numpy())
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    return compute_metrics(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser(description="Public evaluation helper for HypnoConv-BF-v1")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=False, help="Optional checkpoint path if made available later")
    parser.add_argument("--seq-len", type=int, default=21)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    data = load_arrays(args.data_dir)
    indices = np.arange(len(data["y"]))
    windows = build_windows(data, indices, seq_len=args.seq_len)
    ds = SequenceDataset(data["X"], data["y"], windows)
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(in_channels=int(data["X"].shape[1]), num_classes=NUM_CLASSES).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)

    metrics = evaluate(model, loader, device)
    print(json.dumps(metrics, indent=2))
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
