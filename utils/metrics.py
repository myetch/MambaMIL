"""Classification metrics for the MIL experiments, computed from the slide-level probabilities.

All multi-class metrics are one-vs-rest per class and macro-averaged over the classes that occur in `labels`.
Macro sensitivity (mean recall) is by definition the same number as balanced accuracy; both are reported.
Only NumPy is needed.
"""
import numpy as np

SCALARS = ["acc", "balanced_acc", "f1_macro", "f1_weighted", "auc_roc_macro", "sensitivity_macro", "specificity_macro"]


def confusion_matrix(labels, preds, n_classes):
    """Rows are the true class, columns the predicted class."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(labels, dtype=int), np.asarray(preds, dtype=int)), 1)
    return cm


def roc_auc_binary(y_true, score):
    """AUC from the rank-sum (Mann-Whitney U) statistic, ties get the average rank. NaN if one class is missing."""
    y = np.asarray(y_true).astype(bool)
    score = np.asarray(score, dtype=float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=float)
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_score[j + 1] == sorted_score[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _ratio(num, den):
    return float(num) / float(den) if den > 0 else float("nan")


def _nanmean(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    return float(values.mean()) if len(values) else float("nan")


def compute_metrics(labels, probs, n_classes, class_names=None):
    """labels: (n,) ints, probs: (n, n_classes). Returns the scalar metrics, per-class metrics and the confusion matrix."""
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=float).reshape(len(labels), n_classes)
    preds = probs.argmax(axis=1)
    class_names = list(class_names) if class_names else [str(c) for c in range(n_classes)]

    cm = confusion_matrix(labels, preds, n_classes)
    total = cm.sum()
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    fn = support - tp
    fp = cm.sum(axis=0) - tp
    tn = total - tp - fn - fp
    present = support > 0

    sensitivity = np.array([_ratio(tp[c], tp[c] + fn[c]) for c in range(n_classes)])
    specificity = np.array([_ratio(tn[c], tn[c] + fp[c]) for c in range(n_classes)])
    precision = np.array([_ratio(tp[c], tp[c] + fp[c]) for c in range(n_classes)])
    f1 = np.array([_ratio(2 * tp[c], 2 * tp[c] + fp[c] + fn[c]) for c in range(n_classes)])
    auc = np.array([roc_auc_binary(labels == c, probs[:, c]) if present[c] else float("nan")
                    for c in range(n_classes)])

    scalars = {
        "acc": _ratio(tp.sum(), total),
        "balanced_acc": _nanmean(sensitivity[present]),
        "f1_macro": _nanmean(f1[present]),
        "f1_weighted": _ratio(np.nansum(f1[present] * support[present]), support[present].sum()),
        "auc_roc_macro": _nanmean(auc),
        "sensitivity_macro": _nanmean(sensitivity[present]),
        "specificity_macro": _nanmean(specificity[present]),
    }
    per_class = {
        class_names[c]: {"sensitivity": float(sensitivity[c]), "specificity": float(specificity[c]),
                         "precision": float(precision[c]), "f1": float(f1[c]), "auc_roc": float(auc[c]),
                         "support": int(support[c])}
        for c in range(n_classes)
    }
    return {"scalars": scalars, "per_class": per_class, "confusion_matrix": cm.tolist(),
            "labels": labels.tolist(), "preds": preds.tolist(), "class_names": class_names}


def from_results(patient_results):
    """Labels and probabilities out of the `patient_results` dict returned by core_utils.summary()."""
    labels = np.array([r["label"] for r in patient_results.values()], dtype=int)
    probs = np.stack([np.asarray(r["prob"]).reshape(-1) for r in patient_results.values()])
    return labels, probs
