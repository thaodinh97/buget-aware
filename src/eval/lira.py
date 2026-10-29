"""
LiRA (Likelihood Ratio Attack) for PyTorch classifiers.

Carlini et al., "Membership Inference Attacks From First Principles",
IEEE S&P 2022. https://arxiv.org/abs/2112.03570

- Signal: stable logit-scaled confidence phi = z_y - logsumexp(z_{y'!=y})
  computed in float64 from raw logits (paper Sec. VI-A). Never saturates.
- Online (Algorithm 1): log N(phi; mu_in, s_in^2) - log N(phi; mu_out, s_out^2).
- Offline (Eq. 4): one-sided log Pr[Z <= phi], Z ~ N(mu_out, s_out^2).
- Means are always per example; variances are per example or global
  (pooled over all IN / OUT observations, recommended below ~64 shadows,
  paper Sec. VI-B).
- Shadow training is done by the caller; masks[s, i] must be True iff pool
  point i was in shadow s's training set.
"""

import numpy as np
import torch
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve


def lira_signal(model, x, y, batch_size=4096):
    """Logit-scaled confidence log(p_y / (1 - p_y)) for every row, float64."""
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if len(x) != len(y):
        raise ValueError(f'data/labels length mismatch: {len(x)} vs {len(y)}')
    if not len(y):
        return np.empty(0, dtype=np.float64)
    model.eval(); device = next(model.parameters()).device; values = []
    with torch.no_grad():
        for i in range(0, len(y), batch_size):
            z = model(torch.as_tensor(x[i:i+batch_size], dtype=torch.float32, device=device)).double()
            t = torch.as_tensor(y[i:i+batch_size], device=device)
            if z.ndim == 1: z = z[:, None]
            if z.shape[1] == 1:
                if torch.any((t != 0) & (t != 1)): raise ValueError('One-logit model requires 0/1 labels')
                values.append(((2*t - 1)*z[:, 0]).cpu().numpy()); continue
            if torch.any(t < 0) or torch.any(t >= z.shape[1]): raise ValueError('Label outside model classes')
            zy = z.gather(1, t[:, None]).squeeze(1)
            rest = torch.logsumexp(z.scatter(1, t[:, None], float('-inf')), dim=1)
            values.append((zy - rest).cpu().numpy())
    return np.concatenate(values)


def balanced_masks(n_shadows, units, seed):
    """Each unit (rows sharing an id) is IN for exactly n_shadows/2 shadows."""
    if n_shadows < 4 or n_shadows % 2:
        raise ValueError('n_shadows must be an even number >= 4')
    _, inverse = np.unique(np.asarray(units), return_inverse=True)
    ranks = np.random.default_rng(seed).random((inverse.max() + 1, n_shadows)).argsort(1).argsort(1)
    return (ranks < n_shadows // 2)[inverse].T.copy()


class LiRA:
    def __init__(self, mode='online', variance='global', min_models_per_group=2, std_floor=1e-6):
        if mode not in {'online', 'offline'}: raise ValueError("mode must be 'online' or 'offline'")
        if variance not in {'global', 'per_example'}: raise ValueError("variance must be 'global' or 'per_example'")
        if min_models_per_group < 2: raise ValueError('min_models_per_group must be >= 2')
        self.mode, self.variance = mode, variance
        self.min_models_per_group, self.std_floor = min_models_per_group, float(std_floor)
        self.in_mean = self.in_std = self.out_mean = self.out_std = None

    @staticmethod
    def _moments(signals, mask, variance):
        counts = mask.sum(0)
        mean = np.where(mask, signals, 0.).sum(0)/np.maximum(counts, 1)
        sq = np.where(mask, (signals - mean)**2, 0.)
        std = np.sqrt(sq.sum()/counts.sum()) if variance == 'global' else np.sqrt(sq.sum(0)/np.maximum(counts, 1))
        return mean, np.broadcast_to(std, mean.shape).astype(np.float64), counts

    def fit(self, signals, masks):
        """signals, masks: arrays [n_shadows, n_pool] from pre-trained shadows."""
        signals = np.asarray(signals, dtype=np.float64); masks = np.asarray(masks, dtype=bool)
        if signals.ndim != 2 or signals.shape != masks.shape:
            raise ValueError(f'signals {signals.shape} and masks {masks.shape} must be equal 2-D shapes')
        if not np.all(np.isfinite(signals)): raise ValueError('Shadow signals contain NaN/inf')
        self.out_mean, self.out_std, out_counts = self._moments(signals, ~masks, self.variance)
        need = out_counts < self.min_models_per_group
        if self.mode == 'online':
            self.in_mean, self.in_std, in_counts = self._moments(signals, masks, self.variance)
            need |= in_counts < self.min_models_per_group
        if need.any():
            raise ValueError(f'{need.sum()} pool points have too few IN/OUT shadows; first: {np.flatnonzero(need)[:10].tolist()}')
        self.out_std = np.maximum(self.out_std, self.std_floor)
        if self.mode == 'online': self.in_std = np.maximum(self.in_std, self.std_floor)
        return self

    def score(self, observed, indices=None):
        """Membership score (higher = member) for target signals at pool indices."""
        if self.out_mean is None: raise RuntimeError('Call fit() before score()')
        observed = np.asarray(observed, dtype=np.float64).reshape(-1)
        idx = np.arange(len(self.out_mean)) if indices is None else np.asarray(indices, dtype=np.int64).reshape(-1)
        if len(idx) != len(observed): raise ValueError('observed/indices length mismatch')
        if not np.all(np.isfinite(observed)): raise ValueError('Target signals contain NaN/inf')
        if self.mode == 'offline':
            return norm.logcdf(observed, loc=self.out_mean[idx], scale=self.out_std[idx])
        return (norm.logpdf(observed, loc=self.in_mean[idx], scale=self.in_std[idx])
                - norm.logpdf(observed, loc=self.out_mean[idx], scale=self.out_std[idx]))


def evaluate_mia(member_scores, nonmember_scores, fpr_targets=(1e-3, 1e-2, 5e-2)):
    """AUC plus the best ROC operating point with FPR <= each target."""
    m = np.asarray(member_scores, dtype=np.float64).reshape(-1)
    n = np.asarray(nonmember_scores, dtype=np.float64).reshape(-1)
    if not len(m) or not len(n): raise ValueError('Need at least one member and one nonmember score.')
    if not (np.all(np.isfinite(m)) and np.all(np.isfinite(n))): raise ValueError('Scores contain NaN or infinite values.')
    truth = np.r_[np.ones(len(m)), np.zeros(len(n))]; scores = np.r_[m, n]
    fpr, tpr, _ = roc_curve(truth, scores, drop_intermediate=False)
    result = {'auc': float(roc_auc_score(truth, scores)), 'n_members': len(m), 'n_nonmembers': len(n)}
    for target in fpr_targets:
        ok = np.flatnonzero(fpr <= target); best = ok[np.argmax(tpr[ok])]
        result[f'roc_tpr_at_fpr_{target:g}'] = float(tpr[best])
        result[f'roc_actual_fpr_{target:g}'] = float(fpr[best])
    return result
