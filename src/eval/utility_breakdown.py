"""Utility beyond global macro-F1.

For an anomaly detector the operating cost is the false-positive rate on normal traffic and the
benefit is attack recall; macro-F1 averages them away. Every model is therefore scored on
- the test split: macro-F1, F1 of each class, FPR and attack recall;
- every client's retained training rows (the data that client still holds after the request): the same
  metrics per client. These are in-sample, because the client partition is train-only; they show which
  client a forget step or recovery harms, which the pooled test split cannot.
Undefined values (FPR without normal rows, recall without attack rows) are NaN, never 0.
"""
import numpy as np

METRICS = ('macro_f1', 'f1_normal', 'f1_attack', 'fpr', 'recall_attack', 'normal_rows', 'attack_rows')


def _f1(tp, fp, fn):
    return 2*tp/(2*tp + fp + fn) if tp + fp + fn else float('nan')


def detector_metrics(y, p, threshold=.5):
    """y: labels (1 = attack); p: P(attack). Class F1 is NaN only if the class is absent from y and from the predictions."""
    y = np.asarray(y).astype(np.int64); pred = np.asarray(p) >= threshold
    if y.shape != pred.shape or y.ndim != 1: raise ValueError('Labels and predictions need matching 1-D shapes')
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    f1_attack, f1_normal = _f1(tp, fp, fn), _f1(tn, fn, fp)
    present = [v for v in (f1_normal, f1_attack) if not np.isnan(v)]
    return {'macro_f1': float(np.mean(present)) if present else float('nan'),
            'f1_normal': f1_normal, 'f1_attack': f1_attack,
            'fpr': fp/(fp + tn) if fp + tn else float('nan'),
            'recall_attack': tp/(tp + fn) if tp + fn else float('nan'),
            'normal_rows': tn + fp, 'attack_rows': tp + fn}


def client_metrics(probabilities, labels):
    """{f'client{j}_{metric}': value} for per-client probability and label arrays."""
    out = {}
    for j, (p, y) in enumerate(zip(probabilities, labels)):
        out.update({f'client{j}_{k}': v for k, v in detector_metrics(y, p).items()})
    return out
