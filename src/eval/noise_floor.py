"""Seed noise floor of exact retraining and distances to a retrain ensemble."""
import numpy as np


def mae(p, q):
    return float(np.mean(np.abs(np.asarray(p, dtype=float) - np.asarray(q, dtype=float))))


def retrain_noise(retrain_predictions):
    """Leave-one-out distance of each retrain to the others: what exact retraining itself scores."""
    n = len(retrain_predictions)
    if n < 3:
        raise ValueError('Noise floor needs at least three retrains')
    pairwise = np.array([[mae(a, b) for b in retrain_predictions] for a in retrain_predictions])
    loo = pairwise.sum(1) / (n - 1)
    return {'retrains': n, 'pairwise_mae': pairwise.tolist(), 'loo_distance': loo.tolist(),
            'mean': float(loo.mean()), 'sd': float(loo.std(ddof=1)), 'max': float(loo.max())}


def distance_to_retrains(p, retrain_predictions):
    return float(np.mean([mae(p, r) for r in retrain_predictions]))


def margin(noise, sd_factor, minimum):
    """Smallest distance difference treated as larger than retraining seed noise."""
    return max(minimum, sd_factor * noise['sd'])


def identifiable(original_distance, noise, sd_factor):
    """F influence is identifiable only if the original is farther than every retrain and k sd."""
    return bool(original_distance > noise['max'] and original_distance - noise['mean'] >= sd_factor * noise['sd'])


def disagreement_to_retrains(p, retrain_predictions, threshold=.5):
    """Mean share of rows whose 0.5-thresholded label differs from each retrain's."""
    labels = np.asarray(p) >= threshold
    return float(np.mean([np.mean(labels != (np.asarray(r) >= threshold)) for r in retrain_predictions]))
