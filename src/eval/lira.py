"""Offline LiRA (Carlini et al., 2022) with OUT-only shadow models.

Every shadow model is trained without any member query record or its exact (optionally near)
duplicates, so per-record OUT statistics calibrate record difficulty. Shadow training itself
is supplied by the caller.
"""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors


def logit_confidence(y, p, eps=1e-7):
    """phi = log(p_y / (1 - p_y)) of the true class, roughly Gaussian across shadow models."""
    q = np.clip(np.where(np.asarray(y) == 1, p, 1 - np.asarray(p)), eps, 1 - eps)
    return np.log(q) - np.log1p(-q)


def out_statistics(shadow_phi, global_variance=True):
    shadow_phi = np.asarray(shadow_phi, dtype=float)
    if shadow_phi.ndim != 2 or len(shadow_phi) < 2:
        raise ValueError('LiRA needs at least two shadow models')
    mu = shadow_phi.mean(0)
    sd = shadow_phi.std(0, ddof=1)
    if global_variance:
        # Fixed variance is more stable than per-record variance with few shadows.
        sd = np.full_like(sd, np.sqrt(np.mean(sd**2)))
    return mu, np.maximum(sd, 1e-6)


def near_duplicate_index(train_x, nonmember_x, quantile):
    """Radius = quantile of nonmember-to-train nearest-neighbour distance in encoded space.

    Nonmembers come from group-disjoint splits, so within this radius a train row is closer to a
    query than a typical nonmember is to any training row: a near duplicate for the attack.
    """
    index = NearestNeighbors(n_neighbors=1).fit(train_x)
    distance, _ = index.kneighbors(nonmember_x)
    return index, float(np.quantile(distance[:, 0], quantile))


def near_duplicates(index, query_x, radius):
    found = index.radius_neighbors(query_x, radius=radius, return_distance=False)
    return np.unique(np.concatenate(found)).astype(np.int64) if len(found) else np.array([], dtype=np.int64)


def lira_scores(phi, mu, sd):
    """Higher means the target model is more confident than OUT models: more member-like."""
    return (np.asarray(phi, dtype=float) - mu) / sd


def calibrate_thresholds(y, scores, quantile=.99):
    return [float(np.quantile(scores[np.asarray(y) == c], quantile, method='higher')) for c in [0, 1]]


def attack_summary(member_y, member_scores, nonmember_y, nonmember_scores, thresholds):
    truth = np.r_[np.ones(len(member_scores)), np.zeros(len(nonmember_scores))]
    score = np.r_[member_scores, nonmember_scores]
    t = np.array(thresholds)
    decision = np.r_[member_scores > t[member_y], nonmember_scores > t[nonmember_y]]
    fpr, tpr, _ = roc_curve(truth, score)
    return {'mia_auc': float(roc_auc_score(truth, score)),
            'mia_balanced_accuracy': float(balanced_accuracy_score(truth, decision)),
            'member_rate': float(decision[:len(member_scores)].mean()),
            'nonmember_false_positive_rate': float(decision[len(member_scores):].mean()),
            'mia_tpr_at_fpr_0_01': float(tpr[fpr <= .01].max()) if len(nonmember_scores) >= 1000 else None,
            'members': len(member_scores), 'nonmembers': len(nonmember_scores),
            'operating_point': 'strictly_above_original_calibration_classwise_99th_percentile_lira_score'}
