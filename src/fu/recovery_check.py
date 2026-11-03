"""Forget/recovery outer loop whose stopping rule is checked after a short recovery."""
from src.fu.conpfu_inspired import alternating, gradient_batches, mean_ce


def forget_with_recovery_check(original, forget, retain, config, seed, stop_target, recover):
    """Each outer step: one alternating cycle (local guard at stop_target), then recover(model, step),
    a short FedAvg recovery on retained data, then the check CE(F) >= stop_target on the recovered model.

    Checking after recovery accounts for recovery re-learning or overshooting what the local stage did.
    Returns the recovered model, the first local model (before any recovery) and the logs.
    """
    if stop_target is None:
        raise ValueError('Recovery-checked stopping needs a stop target')
    if config.get('kd_mode'):
        raise ValueError('KD is not combined with recovery-checked stopping')
    current, first_local, logs, total, reached = original, None, [], 0, False
    for step in range(config['max_outer']):
        local, cycle = alternating(current, forget, retain, {**config, 'cycles': 1}, seed + 1000 * step,
                                   stop_target, anchor_model=original)
        first_local = local if first_local is None else first_local
        total += gradient_batches(cycle)
        current = recover(local, step)
        after = mean_ce(current, forget.x, forget.y)
        logs.append({'outer': step, 'phase': 'outer', 'local_gradient_batches': gradient_batches(cycle),
                     'forget_ce_local': mean_ce(local, forget.x, forget.y), 'forget_ce_after_recovery': after,
                     'stop_target': stop_target, 'batches': 0})
        if after >= stop_target:
            reached = True
            break
    logs.append({'phase': 'summary', 'stop_target': stop_target, 'stop_reached': reached,
                 'outer_steps': step + 1, 'gradient_batches': total, 'batches': 0})
    return current, first_local, logs
