"""Cost forecasts and budget decisions of the Digital Twin, as pure functions over measured cost records.

Cost model (unchanged from scripts/shard/shard_dt.py, so the experiments describe the existing twin):
  profile  = {'client': {client id (str): compute seconds per retained sample and round}, 'server': seconds per round}
  rates()  fits a profile from cost records (round_costs, client_costs) exactly as shard_dt.rates;
  forecast(profile, counts, R) = R x (sum_c rate_c x n_c + server) for the clients of the touched shard, as shard_dt.plan;
  Static   keeps the profile of the base fit;
  Adaptive updates it after every observed operation, client by client and the server share, by EWMA with weight .3 on the
           new value, as shard_dt.execute does.
Oracle knows the true cost of every option (the measured one).

Decision of one request, budget B (seconds of FedAvg round wall time, as shard_dt sums it):
  precommit  shard_full if F x m <= B, else shard_fixed if X x m <= B, else reject   (F, X: forecasts of 20 and 10 rounds)
  midcourse  start if X x m <= B, else reject; after the 10th round, with e = measured wall time of rounds 1..10, continue
             to 20 rounds iff e + e x (20 - 10)/10 x m' <= B (the rule of the component's shard_checkpoint policy)
  m  = 1 without margin; with margin, the q-quantile of the last `window` ratios actual / forecast of the operations this
       planner executed (a one-sided conformal upper bound; it can be < 1 when the twin over-forecasts). The history starts
       with the ratios of the base fit (forecast of each shard's 20 fit rounds from the fit profile against their time).
  m' = 1 without margin; with margin, the same quantile over the ratios (time of rounds 11..20) / (time of rounds 1..10).
Feedback: a rejected request is escalated and executed outside the budget with shard_full, so the twin observes every
operation (the rounds actually run). `lag` delays every observation by that many requests and makes the twin's client counts
that many requests old (stale state): it has not yet registered the last `lag` deletions when it forecasts.

Simple comparators (added 2026-11-16, after the first evaluation; their definitions were fixed before they were run). They
answer whether the twin's cost model earns its complexity. The first two take no decision at all; the other three replace
only the forecast and go through the same decision rules, margin and lag as Static / Adaptive:
  always_full   every request with shard_full, no forecast, nothing rejected
  always_fixed  every request with shard_fixed, no forecast, nothing rejected
  rows_static   'cost is proportional to rows': one number, seconds per row and round, from the base fit; never updated
  rows_ewma     the same number, refreshed after every observed operation with Adaptive's EWMA weight (.3): Adaptive's
                learning rate without its per-client structure
  last_shard    'what this shard cost last time': the mean round time of the last observed operation on the same shard
                (initially its base fit), scaled by the rows the shard holds now; no model and no smoothing
PlannerConfig.alpha (default .3, the weight of shard_dt.py) is the EWMA weight of Adaptive. Other values are a sensitivity
probe added after the comparators were seen: with alpha = 1 Adaptive keeps the last observed rate of every client and the
last server share, i.e. the model of the twin with the memory of last_shard.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

R_FULL, R_FIXED = 20, 10
BUDGET_LEVELS = {'tight': .5, 'mid': .75, 'wide': 1.5}     # x the measured fit time of the largest shard (20 rounds)


def rates(rounds, clients):
    """Per-client compute seconds per sample and server seconds per round (= scripts/shard/shard_dt.rates)."""
    totals = {}
    for row in clients:
        cid = str(row['client_id'])
        seconds, samples = totals.get(cid, (0., 0))
        totals[cid] = seconds + row['compute_seconds'], samples + row['samples']
    return {'client': {c: s/n for c, (s, n) in totals.items()},
            'server': sum(r['server_seconds'] for r in rounds)/len(rounds)}


def forecast(profile, counts, rounds):
    """Seconds of `rounds` FedAvg rounds of the clients in `counts` ({client: samples}), as scripts/shard/shard_dt.plan."""
    duration = sum(profile['client'][str(c)]*n for c, n in counts.items())
    if counts:
        duration += profile['server']
    return rounds*duration


def ewma(profile, observed, alpha=.3):
    out = {'client': dict(profile['client']), 'server': profile['server']}
    for c, value in observed['client'].items():
        out['client'][c] = (1 - alpha)*out['client'][c] + alpha*value
    out['server'] = (1 - alpha)*out['server'] + alpha*observed['server']
    return out


def cost(request, rounds):
    """Measured wall time of the first `rounds` rounds of a request's shard_full run (= the cost of that option)."""
    return float(sum(r['wall_seconds'] for r in request['rounds'][:rounds]))


def observed_records(request, rounds):
    return ([r for r in request['rounds'] if r['round'] <= rounds], [c for c in request['clients'] if c['round'] <= rounds])


PROFILE_PREDICTORS = ('static', 'adaptive')
BASELINE_PREDICTORS = ('rows_static', 'rows_ewma', 'last_shard')
FIXED_POLICIES = ('always_full', 'always_fixed')


class ProfilePredictor:
    """The twin's cost model: per-client rates and a server share; Adaptive refreshes them by EWMA."""

    def __init__(self, profile, adaptive, alpha=.3):
        self.profile, self.adaptive, self.alpha = profile, adaptive, alpha

    def forecast(self, shard, counts, rounds):
        return forecast(self.profile, counts, rounds)

    def update(self, shard, rounds, clients):
        if self.adaptive:
            self.profile = ewma(self.profile, rates(rounds, clients), self.alpha)


class RowsRate:
    """Comparator 'cost is proportional to rows': seconds per row and round, one number for every shard and client,
    from the base fit (round wall time over round samples). alpha=None keeps it; otherwise EWMA on every operation."""

    def __init__(self, fit_rounds, alpha=None):
        self.rate, self.alpha = self._rate(fit_rounds), alpha

    @staticmethod
    def _rate(rounds):
        return sum(r['wall_seconds'] for r in rounds)/sum(r['samples'] for r in rounds)

    def forecast(self, shard, counts, rounds):
        return rounds*self.rate*sum(counts.values())

    def update(self, shard, rounds, clients):
        if self.alpha is not None:
            self.rate = (1 - self.alpha)*self.rate + self.alpha*self._rate(rounds)


class LastShard:
    """Comparator 'what this shard cost last time': mean round wall time and rows of the last observed operation on the
    shard (initially its base fit); the forecast scales that time with the rows the shard holds now."""

    def __init__(self, fit_rounds):
        self.last = {}
        for j in sorted({r['shard'] for r in fit_rounds}):
            self.update(j, [r for r in fit_rounds if r['shard'] == j], None)

    def forecast(self, shard, counts, rounds):
        seconds, rows = self.last[shard]
        return rounds*seconds*sum(counts.values())/rows

    def update(self, shard, rounds, clients):
        self.last[shard] = (sum(r['wall_seconds'] for r in rounds)/len(rounds), rounds[0]['samples'])


def make_predictor(name, fit_profile, fit_rounds=None, alpha=.3):
    """alpha: EWMA weight on the newest observation, for the two predictors that learn (adaptive, rows_ewma)."""
    if name in PROFILE_PREDICTORS:
        return ProfilePredictor(fit_profile, name == 'adaptive', alpha)
    if name not in BASELINE_PREDICTORS:
        raise ValueError(f'no forecasting predictor named {name!r}')
    if fit_rounds is None:
        raise ValueError(f'{name} needs the cost records of the base fit (fit_records)')
    return LastShard(fit_rounds) if name == 'last_shard' else RowsRate(fit_rounds, alpha if name == 'rows_ewma' else None)


def initial_ratios(predictor, fit_rounds, fit_clients):
    """Per shard: measured time of its base fit / the predictor's forecast of it (in-sample, the start of the margin history)."""
    out = []
    for j in sorted({r['shard'] for r in fit_rounds}):
        rs = [r for r in fit_rounds if r['shard'] == j]
        counts = {c['client_id']: c['samples'] for c in fit_clients if c['shard'] == j and c['round'] == 1}
        out.append(sum(r['wall_seconds'] for r in rs)/predictor.forecast(j, counts, len(rs)))
    return out


def fit_ratios(fit_rounds, fit_clients, profile, fixed_rounds=R_FIXED):
    """Initial error history: per shard, measured fit time of its R_base rounds / its forecast from the fit profile; and the
    per-round ratio of the time of the rounds after `fixed_rounds` to that of the first `fixed_rounds`."""
    extra = []
    for j in sorted({r['shard'] for r in fit_rounds}):
        rs = [r for r in fit_rounds if r['shard'] == j]
        first = sum(r['wall_seconds'] for r in rs if r['round'] <= fixed_rounds)
        extra.append(extrapolation_ratio(first, sum(r['wall_seconds'] for r in rs), len(rs), fixed_rounds))
    return initial_ratios(ProfilePredictor(profile, False), fit_rounds, fit_clients), extra


class Margin:
    def __init__(self, q, initial, window=10):
        self.q, self.window, self.ratios = q, window, list(initial)

    def factor(self):
        return 1. if self.q is None else float(np.quantile(self.ratios[-self.window:], self.q))

    def observe(self, ratio):
        self.ratios.append(float(ratio))


@dataclass(frozen=True)
class PlannerConfig:
    predictor: str            # 'oracle' | PROFILE_PREDICTORS | BASELINE_PREDICTORS | FIXED_POLICIES
    margin_q: object = None   # None or a quantile in (0, 1]
    lag: int = 0
    mode: str = 'precommit'   # 'precommit' | 'midcourse'
    window: int = 10
    margin_mode: str = 'ratio' # 'ratio': errors of the forecasts as made; 'rescored': past operations re-forecast now
    alpha: float = .3          # EWMA weight of adaptive / rows_ewma (.3 = scripts/shard/shard_dt.py; 1 = keep the last value)
    full_rounds: int = R_FULL  # R_base and shard_fixed_rounds of the train config ('standard': 20 and 10)
    fixed_rounds: int = R_FIXED

    @property
    def name(self):
        if self.predictor == 'oracle' or self.predictor in FIXED_POLICIES:
            return self.predictor
        margin = 'nomargin' if self.margin_q is None else ('margin' if self.margin_mode == 'ratio' else 'margin_rescored')
        learns = self.predictor in ('adaptive', 'rows_ewma') and self.alpha != .3
        parts = [self.predictor + (f'_a{self.alpha:g}' if learns else ''), margin, self.mode]
        return '_'.join(parts) + (f'_lag{self.lag}' if self.lag else '')


def oracle_choice(request, budget, full=R_FULL, fixed=R_FIXED):
    if cost(request, full) <= budget:
        return 'full'
    return 'fixed' if cost(request, fixed) <= budget else 'reject'


def twin_counts(requests, k, lag):
    """Client counts of request k's shard as a twin `lag` requests behind sees them (it misses the last `lag` deletions)."""
    counts = {int(c): int(n) for c, n in requests[k]['counts'].items()}
    for i in range(max(0, k - lag), k):
        if requests[i]['client'] in counts:
            counts[requests[i]['client']] += requests[i]['rows']
    return counts


class Twin:
    """The planner's state between requests: the predictor (Static: fixed profile; Adaptive: EWMA; or a simple comparator),
    the two error histories, and the queue of observations not yet delivered (lag). Used by simulate() and by the online
    runner. fit_records = (round_costs, client_costs) of the base fit; the simple comparators need it."""

    def __init__(self, cfg, fit_profile, fit_history, fit_records=None):
        if cfg.mode not in ('precommit', 'midcourse') or cfg.margin_mode not in ('ratio', 'rescored'):
            raise ValueError(f'bad planner config {cfg}')
        self.cfg = cfg
        self.predictor = make_predictor(cfg.predictor, fit_profile, fit_records[0] if fit_records else None, cfg.alpha)
        # the margin history starts with the predictor's own in-sample errors on the base fit
        self.initial = (list(fit_history[0]) if cfg.predictor in PROFILE_PREDICTORS
                        else initial_ratios(self.predictor, *fit_records))
        self.margin, self.extra = Margin(cfg.margin_q, self.initial, cfg.window), Margin(cfg.margin_q, fit_history[1], cfg.window)
        self.scored = []                          # (shard, counts, rounds, seconds) of every delivered operation ('rescored')
        self.pending = deque()                    # (available before request index, observation)

    def advance(self, k):
        """Deliver every observation available before request index k (0-based)."""
        while self.pending and self.pending[0][0] <= k:
            _, obs = self.pending.popleft()
            rounds, clients = obs['records']
            shard = rounds[0].get('shard')
            self.predictor.update(shard, rounds, clients)
            self.margin.observe(obs['ratio'])
            self.scored.append((shard, {c['client_id']: c['samples'] for c in clients if c['round'] == rounds[0]['round']},
                                len(rounds), sum(r['wall_seconds'] for r in rounds)))
            if obs['extra'] is not None:
                self.extra.observe(obs['extra'])

    def forecasts(self, counts, shard=None):
        return (self.predictor.forecast(shard, counts, self.cfg.full_rounds),
                self.predictor.forecast(shard, counts, self.cfg.fixed_rounds))

    def factor(self):
        """Margin factor on the pre-execution forecast. 'rescored' (differs from 'ratio' only for a predictor that learns):
        the q-quantile of measured / forecast for the last `window` operations, each re-forecast with the current predictor,
        so that an error the predictor has since absorbed (a regime change) stops inflating the margin."""
        if self.cfg.margin_q is None or self.cfg.margin_mode == 'ratio':
            return self.margin.factor()
        ratios = self.initial + [s/self.predictor.forecast(shard, counts, rounds) for shard, counts, rounds, s in self.scored]
        return float(np.quantile(ratios[-self.cfg.window:], self.cfg.margin_q))

    def factors(self):
        return self.factor(), self.extra.factor()

    def start(self, f_full, f_fixed, budget):
        """Decision before training: 'full' / 'fixed' / 'reject' (precommit), or 'start' / 'reject' (midcourse)."""
        m = self.factor()
        if self.cfg.mode == 'midcourse':
            return 'start' if f_fixed*m <= budget else 'reject'
        return 'full' if f_full*m <= budget else ('fixed' if f_fixed*m <= budget else 'reject')

    def record(self, k, records, ratio, extra):
        """Observation of request index k: cost records of the rounds run, actual / forecast of the executed option, and the
        extrapolation ratio (rounds 11..20 against 1..10, per round) when 20 rounds ran."""
        self.pending.append((k + 1 + self.cfg.lag, {'records': records, 'ratio': ratio, 'extra': extra}))


def extrapolation_ratio(seconds_fixed, seconds_full, full=R_FULL, fixed=R_FIXED):
    """Mean round time after the checkpoint over mean round time before it (1 when every round costs the same)."""
    return (seconds_full - seconds_fixed)/seconds_fixed*fixed/(full - fixed)


def simulate(requests, budget, cfg, fit_profile, fit_history, fit_records=None):
    """Decisions of one planner over a measured trajectory. requests: rows of a requests.jsonl, in order. Returns one dict
    per request: forecasts, factor, choice, executed rounds and cost, oracle choice and the outcome flags.
    fit_records: (round_costs, client_costs) of the base fit, needed by the simple comparators only."""
    forecasting = cfg.predictor != 'oracle' and cfg.predictor not in FIXED_POLICIES
    twin = Twin(cfg, fit_profile, fit_history, fit_records) if forecasting else None
    full, fixed = cfg.full_rounds, cfg.fixed_rounds
    out = []
    for k, req in enumerate(requests):
        true_full, true_fixed = cost(req, full), cost(req, fixed)
        checkpoint = None
        if cfg.predictor == 'oracle':
            f_full, f_fixed, m, m2 = true_full, true_fixed, 1., 1.
            choice = oracle_choice(req, budget, full, fixed)
        elif cfg.predictor in FIXED_POLICIES:      # no forecast and no decision
            f_full, f_fixed, m, m2 = float('nan'), float('nan'), 1., 1.
            choice = 'full' if cfg.predictor == 'always_full' else 'fixed'
        else:
            twin.advance(k)
            f_full, f_fixed = twin.forecasts(twin_counts(requests, k, cfg.lag), req['shard'])
            m, m2 = twin.factors()
            choice = twin.start(f_full, f_fixed, budget)
            if choice == 'start':          # the component's shard_checkpoint rule, on the measured rounds 1..10
                checkpoint = true_fixed + true_fixed*(full - fixed)/fixed*m2
                choice = 'full' if checkpoint <= budget else 'fixed'
        rounds = fixed if choice == 'fixed' else full               # a rejected request is escalated with shard_full
        spent = cost(req, rounds)
        predicted = f_fixed if choice == 'fixed' else f_full
        oracle = oracle_choice(req, budget, full, fixed)
        out.append({'step': req['step'], 'shard': req['shard'], 'budget': budget, 'forecast_full': f_full,
                    'forecast_fixed': f_fixed, 'factor': m, 'factor_extrapolation': m2, 'checkpoint_forecast': checkpoint,
                    'true_full': true_full, 'true_fixed': true_fixed, 'choice': choice, 'oracle': oracle,
                    'executed_rounds': rounds, 'executed_seconds': spent, 'forecast_executed': predicted,
                    'violation': choice != 'reject' and spent > budget, 'overrun_seconds': max(0., spent - budget) if choice != 'reject' else 0.,
                    'false_reject': choice == 'reject' and oracle != 'reject',
                    'downgrade': choice == 'fixed' and oracle == 'full'})
        if twin is not None:
            twin.record(k, observed_records(req, rounds), spent/predicted if predicted > 0 else 1.,
                        extrapolation_ratio(true_fixed, true_full, full, fixed) if rounds == full else None)
    return out


def summarize(rows):
    """Counts and rates over decision rows. Denominators: violations / accepted, false rejections / requests the oracle
    accepts, downgrades / requests the oracle serves with shard_full."""
    n = len(rows)
    acc = sum(r['choice'] != 'reject' for r in rows)
    o_acc = sum(r['oracle'] != 'reject' for r in rows)
    o_full = sum(r['oracle'] == 'full' for r in rows)
    v = sum(r['violation'] for r in rows); fr = sum(r['false_reject'] for r in rows); dg = sum(r['downgrade'] for r in rows)
    scored = [r for r in rows if r['forecast_executed'] > 0]              # always_full / always_fixed forecast nothing (NaN)
    err = [abs(r['forecast_executed'] - r['executed_seconds']) for r in scored]
    rel = [r['executed_seconds']/r['forecast_executed'] - 1 for r in scored]
    ratio = lambda a, b: a/b if b else float('nan')
    return {'requests': n, 'accepted': acc, 'full': sum(r['choice'] == 'full' for r in rows),
            'fixed': sum(r['choice'] == 'fixed' for r in rows), 'rejected': n - acc, 'oracle_accepted': o_acc,
            'oracle_full': o_full, 'violations': v, 'false_rejects': fr, 'downgrades': dg,
            'violation_rate': ratio(v, acc), 'false_reject_rate': ratio(fr, o_acc), 'downgrade_rate': ratio(dg, o_full),
            'acceptance_rate': ratio(acc, n), 'full_rate': ratio(sum(r['choice'] == 'full' for r in rows), n),
            'overrun_seconds': float(sum(r['overrun_seconds'] for r in rows)),
            'forecast_mae_seconds': float(np.mean(err)) if err else float('nan'),
            'forecast_rel_error_mean': float(np.mean(rel)) if rel else float('nan')}
