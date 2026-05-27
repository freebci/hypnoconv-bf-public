"""Public minimal statistical helper for paired LOSO comparison summaries."""

import json
from pathlib import Path

import numpy as np
from scipy import stats


def load_fold_metrics(root_dir):
    root_dir = Path(root_dir)
    data = {}
    for fp in sorted(root_dir.glob("fold_*/fold_metrics.json")):
        with open(fp) as f:
            fold = json.load(f)
        subj = fold["test_subject"]
        data[subj] = {
            "kappa": fold["cohen_kappa"],
            "macro_f1": fold["macro_f1"],
            "per_class_f1": fold.get("per_class_f1", {}),
        }
    return data


def paired_wilcoxon(metric_a, metric_b):
    common = sorted(set(metric_a.keys()) & set(metric_b.keys()))
    deltas = np.array([metric_a[s] - metric_b[s] for s in common], dtype=float)
    stat, p_value = stats.wilcoxon(deltas, alternative="greater")
    return {
        "n": len(common),
        "mean_delta": float(np.mean(deltas)),
        "std_delta": float(np.std(deltas)),
        "wilcoxon_p": float(p_value),
    }
