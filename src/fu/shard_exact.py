"""Sharded exact federated unlearning as a callable component: the FU input of the Digital Twin.

The logic is that of the v2 shard protocol (docs/shard_v2_plan.md, scripts/shard/run_shard_stream_v2.py), moved out of the
runner script so that it can be imported, called request by request, checkpointed and metered. The v2 files are hash-locked
and unchanged; tests/test_shard_exact.py runs the unmodified v2 runner on synthetic data and requires this module to reproduce
its events, round counts, costs and metrics exactly.

Model: the clients are split into S shards; every shard runs its own FedAvg; the prediction is the sum of the shard logit
margins. A deletion request touches one client, so only that client's shard is retrained, from a fresh initialisation with the
shard's seed, on the data the shard still holds. A whole-client deletion that leaves a shard with fewer than `merge_below`
clients merges it into the active shard with the fewest clients (ties: lowest index); an empty shard is removed.

Policies (how many rounds the affected shard gets):
  shard_full       R_base rounds
  shard_fixed      shard_fixed_rounds rounds
  utility_stop     round by round until the ensemble passes ok() on validation, at most R_base; then, after every request, a
                   fallback gives one more round to the included shard with the fewest rounds until ok() or every shard has
                   R_base rounds
  utility_stop_v1  the first phase of utility_stop only
  dynamicfu        re-implementation of DynamicFU's adaptive round count (contribution probes on every present client)
  drop_backfill    the affected shard leaves the ensemble at once and is retrained from a per-request round budget
ok(): validation macro-F1 >= F1_0 - eps and validation FPR <= FPR_0 + eps, (F1_0, FPR_0) of the ensemble after fit().

Exactness. After every request no included shard has trained on a deleted row (`exact`) and every included shard is trained on
exactly the current data of its shard (`structure_ok`); verify_bit_exact() retrains every included shard from scratch and
compares the weights bit for bit. For shard_full, shard_fixed and drop_backfill the round count is fixed in advance. For
utility_stop, utility_stop_v1 and dynamicfu the round count is chosen from signals of the state before the deletion (F1_0 and
FPR_0 come from models trained on the rows deleted later; DynamicFU also probes the rows being deleted), so their exactness
is conditional on the round count. EXACTNESS_SCOPE records this per policy and every result carries it.

Determinism is the caller's job: torch.use_deterministic_algorithms(True) and a fixed thread count, as in the v2 runners.
"""
import copy
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from scipy.special import expit

from src.data.federated import ids_hash
from src.eval.membership_validation import logit_margin
from src.eval.utility_breakdown import detector_metrics
from src.fl.fedavg_core import ClientWorker, fedavg, model, state
from src.fu.shard_cost import CostMeter, state_bytes

POLICIES = ('shard_full', 'shard_fixed', 'utility_stop', 'utility_stop_v1', 'dynamicfu', 'drop_backfill')
EXACTNESS_SCOPE = {'shard_full': 'unconditional', 'shard_fixed': 'unconditional', 'drop_backfill': 'unconditional',
                   'utility_stop': 'conditional_on_round_count', 'utility_stop_v1': 'conditional_on_round_count',
                   'dynamicfu': 'conditional_on_round_count'}
DYNAMICFU_DEFAULT = {'theta': [1/3, 1/3, 1/3], 'lam': [.5, .5], 'eta': [.2, 1.], 'rho': 1., 'mu': 1., 'omega': [.5, .5],
                     'tau_ctrl': .01, 'tau_drop': .02}
FORMAT = 1


@dataclass(frozen=True)
class ShardConfig:
    """fl: {'rounds' (R_base), 'local_epochs', 'batch_size', 'lr'}; the other fields as in the v2 protocol."""
    fl: dict
    shard_fixed_rounds: int
    backfill_rounds: int
    hidden: tuple = (64, 32)
    eps: float = .005
    merge_below: int = 3
    min_clients_design: int = 4
    dynamicfu: dict = field(default_factory=lambda: copy.deepcopy(DYNAMICFU_DEFAULT))
    device: str = 'cpu'

    @classmethod
    def from_train_config(cls, cfg, **overrides):
        """From an entry of scripts/shard/train_config.CONFIGS."""
        return cls(**{'fl': dict(cfg['fl']), 'shard_fixed_rounds': cfg['shard_fixed_rounds'],
                      'backfill_rounds': cfg['backfill_rounds'], **overrides})

    def to_json(self):
        return {**asdict(self), 'hidden': list(self.hidden)}

    @classmethod
    def from_json(cls, data):
        return cls(**{**data, 'hidden': tuple(data['hidden'])})


@dataclass(frozen=True)
class ForgetRequest:
    """Delete `rows` (indices into the train split) of `client`; rows=None deletes the whole client."""
    client: int
    rows: object = None


@dataclass
class Shard:
    model: object
    idx: np.ndarray          # train rows the model was trained on
    seed: int
    rounds: int
    included: bool = True
    va: object = None        # cached validation margins


@dataclass
class UnlearningResult:
    step: int
    policy: str
    client: int
    kind: str                # 'client' | 'sample' | 'backlog_done'
    rows: int
    event: str               # 'retrain' | 'merge_<j>_into_<t>' | 'removed_empty' | 'backlog_done'
    retrained_shard: object
    retired_shards: list
    shards_included: int
    exact: bool
    structure_ok: bool
    exactness_scope: str
    details: dict            # policy outputs: phase1_rounds, fallback_rounds, condition_met, contribution, R0, stop_reason, pending
    cost: dict               # CostMeter.summary() plus storage
    round_costs: list
    client_costs: list
    config_hash: str
    forget_request_hash: str
    intermediate_checkpoint: object = None
    final_checkpoint: object = None
    structured_warnings: list = field(default_factory=list)

    @property
    def timing_breakdown(self):
        return {k: v for k, v in self.cost.items() if 'seconds' in k}

    @property
    def client_bytes_up(self):
        return _by_client(self.client_costs, 'bytes_up_packed', 'bytes_up_raw')

    @property
    def client_bytes_down(self):
        return _by_client(self.client_costs, 'bytes_down_packed', 'bytes_down_raw')

    @property
    def local_losses(self):
        return [{k: c[k] for k in ('shard', 'round', 'phase', 'client_id', 'loss')} for c in self.client_costs]

    def to_json(self):
        return _py({**asdict(self), 'timing_breakdown': self.timing_breakdown, 'client_bytes_up': self.client_bytes_up,
                    'client_bytes_down': self.client_bytes_down})


def _by_client(rows, measured, raw):
    out = {}
    for r in rows:
        key = measured if r[measured] is not None else raw
        out[r['client_id']] = out.get(r['client_id'], 0) + r[key]
    return {str(k): int(v) for k, v in sorted(out.items())}


def _py(value):
    """Plain JSON types from numpy scalars / arrays, tuples and Paths."""
    if isinstance(value, dict): return {str(k): _py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)): return [_py(v) for v in (sorted(value) if isinstance(value, set) else value)]
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return value.as_posix()
    return value


def _sha(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(f'{a.dtype.str}/{a.shape};'.encode()); h.update(a.tobytes())
    return h.hexdigest()


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---- shard structure (pure functions; equal to scripts/shard/shard_v2.assign / check_design / structure_plan) ----

def assign_shards(shards, clients, seed=7):
    perm = np.random.default_rng(seed).permutation(clients)
    return [sorted(perm[j::shards].tolist()) for j in range(shards)]


def check_design(assignment, min_clients):
    sizes = [len(a) for a in assignment]
    if min(sizes) < min_clients:
        raise ValueError(f'shard sizes {sizes} violate the minimum of {min_clients} clients per shard')
    return sizes


def structure_step(assignment, client, kind, merge_below):
    """Effect of one request on the shard structure: which shard retrains, which shards retire, and the event."""
    cur = [list(a) for a in assignment]
    before = [list(a) for a in cur]
    j = next((i for i, a in enumerate(cur) if client in a), None)
    if j is None:
        raise ValueError(f'client {client} is in no active shard')
    event, retrain, retired = 'retrain', j, []
    if kind == 'client':
        cur[j].remove(client)
        others = [i for i, a in enumerate(cur) if a and i != j]
        if not cur[j]:
            event, retrain, retired = 'removed_empty', None, [j]
        elif len(cur[j]) < merge_below and others:
            t = min(others, key=lambda i: (len(cur[i]), i))
            cur[t] = sorted(cur[t] + cur[j]); cur[j] = []
            event, retrain, retired = f'merge_{j}_into_{t}', t, [j]
    elif kind != 'sample':
        raise ValueError(f'unknown request kind {kind!r}')
    return {'event': event, 'retrain': retrain, 'retired': retired, 'assign_before': before, 'assign_after': cur}


class _Trainer:
    """Round-by-round FedAvg of one shard; after r rounds from a fresh init it equals fedavg(init, workers, fl, r, seed)."""

    def __init__(self, fed, shard, idx, seed, start=None, rounds=0):
        self.fed, self.shard, self.idx, self.seed, self.r = fed, shard, np.asarray(idx, np.int64), int(seed), rounds
        cfg = fed.config
        self.m = model(fed.dim, list(cfg.hidden), self.seed, cfg.device) if start is None else copy.deepcopy(start)
        owners = fed.owner[self.idx]
        tr = fed.train
        self.ws = []
        for c in np.unique(owners):
            rows = self.idx[owners == c]
            self.ws.append(ClientWorker(int(c), tr['X'][rows], tr['y'][rows], tr['row_id'][rows], cfg.device))

    def step(self, meter, phase):
        seed = self.seed + self.r * 10000
        if meter is None:
            self.m, _ = fedavg(self.m, self.ws, self.fed.config.fl, 1, seed)
        else:
            self.m = meter.run_round(self.m, self.ws, self.fed.config.fl, seed, self.shard, self.r + 1, phase)
        self.r += 1

    def entry(self):
        return Shard(self.m, self.idx, self.seed, self.r)

    def clone(self):
        other = copy.copy(self)
        other.m = copy.deepcopy(self.m)
        return other


class ShardedUnlearner:
    """Sharded FedAvg ensemble with exact unlearning, one request at a time.

    train: {'X', 'y', 'row_id'} of the train split; validation: {'X', 'y'} (server-side auxiliary data, read by the policies);
    parts: train row indices of every client; assignment: client ids of every shard; seed: training seed (shard j uses
    seed + 1000 j); policy: default policy of unlearn().
    """

    def __init__(self, train, validation, parts, assignment, seed, policy, config, measure_payload=True):
        if policy not in POLICIES:
            raise ValueError(f'policy must be one of {POLICIES}, got {policy!r}')
        self.train, self.validation, self.config, self.seed, self.policy = train, validation, config, int(seed), policy
        self.measure_payload = measure_payload
        self.parts = [np.asarray(p, np.int64) for p in parts]
        n = len(train['y'])
        if not (len(train['X']) == n == len(train['row_id'])) or len(validation['X']) != len(validation['y']):
            raise ValueError('train / validation arrays have inconsistent lengths')
        rows = np.concatenate(self.parts) if self.parts else np.empty(0, np.int64)
        if len(rows) != n or len(np.unique(rows)) != n or (len(rows) and (rows.min() < 0 or rows.max() >= n)):
            raise ValueError('parts must partition the train rows')
        self.owner = np.empty(n, np.int64)
        for c, p in enumerate(self.parts):
            self.owner[p] = c
        clients = sorted(c for a in assignment for c in a)
        if clients != list(range(len(self.parts))):
            raise ValueError('assignment must place every client in exactly one shard')
        check_design(assignment, config.min_clients_design)
        self.assignment_initial = [sorted(int(c) for c in a) for a in assignment]
        self.assignment = [list(a) for a in self.assignment_initial]
        self.dim = train['X'].shape[1]
        self.p_train = np.bincount(train['y'], minlength=2) / n
        self.alive = np.ones(n, bool)
        self.present = set(range(len(self.parts)))
        self.shards, self.pending = [], []
        self.reference = None
        self.step = 0
        self.history = []
        self.totals = {'rows_x_rounds': 0, 'rounds': 0, 'train_bytes_raw': 0}
        self.model_bytes = state_bytes(model(self.dim, list(config.hidden), 0, config.device).state_dict())
        self.data_fingerprint = {'train': _sha(train['X'], train['y'], train['row_id']),
                                 'validation': _sha(validation['X'], validation['y']),
                                 'partition': _sha(self.owner), 'input_dim': int(self.dim)}
        self.config_hash = hashlib.sha256(json.dumps(
            {'config': config.to_json(), 'seed': self.seed, 'assignment': self.assignment_initial,
             'data': self.data_fingerprint}, sort_keys=True).encode()).hexdigest()

    # ---- ensemble ----
    @property
    def r_base(self):
        return self.config.fl['rounds']

    def included(self):
        return [j for j, e in enumerate(self.shards) if e is not None and e.included]

    def margin(self, X):
        """Ensemble logit margin: sum of the included shards' margins (zeros when no shard is included)."""
        ms = [logit_margin(self.shards[j].model, X) for j in self.included()]
        return sum(ms) if ms else np.zeros(len(X))

    def predict_proba(self, X):
        return expit(self.margin(X))

    def _va(self, entry, meter=None):
        if entry.va is None:
            if meter is None:
                entry.va = logit_margin(entry.model, self.validation['X'])
            else:
                with meter.timed('policy_eval'):
                    entry.va = logit_margin(entry.model, self.validation['X'])
        return entry.va

    def _ens(self, shards, meter=None):
        ms = [self._va(e, meter) for e in shards if e is not None and e.included]
        return sum(ms) if ms else np.zeros(len(self.validation['y']))

    def _val_stats(self, marg):
        y = self.validation['y']
        met = detector_metrics(y, expit(marg))
        return met['macro_f1'], met['fpr'], float(np.logaddexp(0, -(2*y - 1)*marg).mean())

    def _stats(self, shards, meter=None):
        marg = self._ens(shards, meter)
        if meter is None:
            return self._val_stats(marg)
        with meter.timed('policy_eval'):
            return self._val_stats(marg)

    def _ok(self, shards, meter=None):
        f1, fpr, _ = self._stats(shards, meter)
        return bool(f1 >= self.reference[0] - self.config.eps and fpr <= self.reference[1] + self.config.eps)

    def _shard_idx(self, assignment, j):
        blocks = [self.parts[c][self.alive[self.parts[c]]] for c in assignment[j]]
        return np.sort(np.concatenate(blocks)) if blocks else np.empty(0, np.int64)

    def _trainer(self, j, idx):
        return _Trainer(self, j, idx, self.seed + 1000*j)

    # ---- training ----
    def fit(self, checkpoint_dir=None):
        """Train every shard for R_base rounds on all data. Returns {'reference', 'cost', 'round_costs', 'client_costs'}."""
        if self.shards:
            raise RuntimeError('fit() was already called')
        meter = CostMeter(self.measure_payload)
        for j in range(len(self.assignment)):
            trn = self._trainer(j, self._shard_idx(self.assignment, j))
            for _ in range(self.r_base):
                trn.step(meter, 'base')
            self.shards.append(trn.entry())
        f1, fpr, _ = self._stats(self.shards, meter)
        self.reference = (float(f1), float(fpr))
        return self._finish(meter, checkpoint_dir, {'reference': {'validation_macro_f1': f1, 'validation_fpr': fpr}},
                            accumulate=False)

    def unlearn(self, request, policy=None, checkpoint_dir=None):
        """Apply one deletion request exactly. Returns an UnlearningResult; the unlearner then serves the new ensemble."""
        if not self.shards:
            raise RuntimeError('fit() or load() first')
        pol = policy or self.policy
        if pol not in POLICIES:
            raise ValueError(f'policy must be one of {POLICIES}, got {pol!r}')
        if self.pending and pol != 'drop_backfill':
            raise RuntimeError('a drop_backfill backlog is pending: call finish_backlog() before using another policy')
        c = int(request.client)
        if c not in self.present:
            raise ValueError(f'client {c} holds no data (unknown or already deleted)')
        held = self.parts[c][self.alive[self.parts[c]]]
        if request.rows is None:
            f = held
        else:
            f = np.asarray(request.rows, np.int64).ravel()
            if not len(f) or len(np.unique(f)) != len(f):
                raise ValueError('forget rows must be a non-empty set of distinct train rows')
            if not np.isin(f, held).all():
                raise ValueError(f'forget rows must be rows client {c} currently holds')
            f = np.sort(f)
        kind = 'client' if len(f) == len(held) else 'sample'
        cfg, rb, meter = self.config, self.r_base, CostMeter(self.measure_payload)
        plan = structure_step(self.assignment, c, kind, cfg.merge_below)
        alive_before = self.alive.copy()
        self.alive[f] = False
        if kind == 'client':
            self.present.discard(c)
        A, j, sh, extra = plan['assign_after'], plan['retrain'], self.shards, {}
        if pol == 'dynamicfu' and j is not None:       # contributions on the ensemble before the request
            f1_now = self._stats(sh, meter)[0]
            contrib = self._contributions(plan['assign_before'], alive_before, self.present | {c}, f1_now, meter)
        for jr in plan['retired']:                      # removed / merged-away shards leave the ensemble
            sh[jr] = None
            self.pending = [p for p in self.pending if p.shard != jr]
        self.assignment = A
        idx = self._shard_idx(A, j) if j is not None else None
        if pol in ('shard_full', 'shard_fixed') and j is not None:
            trn = self._trainer(j, idx)
            for _ in range(rb if pol == 'shard_full' else cfg.shard_fixed_rounds):
                trn.step(meter, 'retrain')
            sh[j] = trn.entry()
        elif pol in ('utility_stop', 'utility_stop_v1'):
            if j is not None:
                trn = self._trainer(j, idx)
                sh[j] = self._run_rounds(trn, j, lambda r, tmp: r >= rb or self._ok(tmp, meter), meter)
                extra['phase1_rounds'] = trn.r
            if pol == 'utility_stop':
                fallback = 0
                while not self._ok(sh, meter):
                    cand = [(e.rounds, jj) for jj, e in enumerate(sh) if e is not None and e.included and e.rounds < rb]
                    if not cand:
                        break
                    jj = min(cand)[1]
                    trn = _Trainer(self, jj, sh[jj].idx, sh[jj].seed, start=sh[jj].model, rounds=sh[jj].rounds)
                    trn.step(meter, 'fallback')
                    sh[jj] = trn.entry()
                    fallback += 1
                extra.update(fallback_rounds=fallback, condition_met=self._ok(sh, meter))
        elif pol == 'dynamicfu' and j is not None:
            extra.update(self._dynamicfu(j, idx, contrib.get(c, 0.), f1_now, meter))
        elif pol == 'drop_backfill':
            if j is not None:
                if sh[j] is not None:
                    sh[j] = replace(sh[j], included=False)
                self.pending = [p for p in self.pending if p.shard != j] + [self._trainer(j, idx)]
            self._backfill(cfg.backfill_rounds, meter)
            extra['pending'] = len(self.pending)
        self.step += 1
        record = {'step': self.step, 'client': c, 'kind': kind, 'rows': int(len(f)), 'policy': pol, 'event': plan['event'],
                  'forget_request_hash': ids_hash(self.train['row_id'][f])}
        self.history.append(record)
        return self._result(meter, checkpoint_dir, record, plan['retrain'], plan['retired'], extra)

    def finish_backlog(self, checkpoint_dir=None):
        """drop_backfill only: train every pending shard to R_base rounds and put it back into the ensemble."""
        meter = CostMeter(self.measure_payload)
        self._backfill(None, meter)
        record = {'step': self.step, 'client': -1, 'kind': 'backlog_done', 'rows': 0, 'policy': 'drop_backfill',
                  'event': 'backlog_done', 'forget_request_hash': ''}
        return self._result(meter, checkpoint_dir, record, None, [], {'pending': 0})

    def _run_rounds(self, trn, j, rule, meter):
        """Train shard j round by round; rule(rounds, shards_with_candidate) -> True to stop. Returns the new shard entry."""
        while True:
            trn.step(meter, 'retrain')
            cand = trn.entry()
            tmp = list(self.shards); tmp[j] = cand
            if rule(trn.r, tmp):
                return cand

    def _backfill(self, budget, meter):
        rb = self.r_base
        while self.pending and (budget is None or budget > 0):
            trn = self.pending[0]
            n = rb - trn.r if budget is None else min(budget, rb - trn.r)
            for _ in range(n):
                trn.step(meter, 'backfill')
            if budget is not None:
                budget -= n
            if trn.r >= rb:
                self.shards[trn.shard] = trn.entry()
                self.pending.pop(0)

    def _contributions(self, assign_before, alive_before, present, f1_now, meter):
        """DynamicFU eqs. 9-12 for every present client, on the ensemble before the request."""
        cfg, d, tr, sh = self.config, self.config.dynamicfu, self.train, self.shards
        raw, records = {}, []
        for c in sorted(present):
            j = next(i for i, a in enumerate(assign_before) if c in a)
            idx = self.parts[c][alive_before[self.parts[c]]]
            if sh[j] is None or not len(idx):
                continue
            worker = ClientWorker(c, tr['X'][idx], tr['y'][idx], tr['row_id'][idx], cfg.device)
            update, _, log = worker.train(sh[j].model, cfg.fl, sh[j].seed + 777)
            probe = model(self.dim, list(cfg.hidden), 0, cfg.device); probe.load_state_dict(update)
            base = sh[j].model.state_dict()
            u = float(torch.cat([(a - b).flatten() for a, b in zip(probe.state_dict().values(), base.values())]).norm())
            pi = np.bincount(tr['y'][idx], minlength=2) / len(idx)
            dd = d['lam'][0]*jensenshannon(pi, self.p_train)**2 + d['lam'][1]*pi[1]
            tmp = list(sh); tmp[j] = replace(sh[j], model=probe, va=None)
            raw[c] = (u, dd, abs(self._stats(tmp, meter)[0] - f1_now))
            records.append((c, len(idx), log, state(sh[j].model), update))
        meter.record_local(-1, 'contribution', records)
        a = np.array(list(raw.values()))
        norm = (a - a.min(0)) / (a.max(0) - a.min(0) + 1e-12)
        return {c: float(np.dot(d['theta'], n)) for c, n in zip(raw, norm)}

    def _dynamicfu(self, j, idx, contribution, f1_now, meter):
        d, rb, sh = self.config.dynamicfu, self.r_base, self.shards
        rmax = d.get('Rmax') or rb
        eta = float(np.clip(d['eta'][0] + (d['eta'][1] - d['eta'][0])*contribution, *d['eta']))
        trn = self._trainer(j, idx)
        tmp = list(sh); tmp[j] = Shard(trn.m, idx, trn.seed, 0)
        prev_f1, _, prev_loss = self._stats(tmp, meter)
        st = {'eta': eta}

        def rule(r, tmp2):
            f1, _, loss = self._stats(tmp2, meter)
            gain = max(0., st.get('loss', prev_loss) - loss) / (st.get('loss', prev_loss) + 1e-12)
            stab = abs(f1 - st.get('f1', prev_f1)); ctrl = d['omega'][0]*gain + d['omega'][1]*stab
            drop = max(0., f1_now - f1)
            st['eta'] = float(np.clip(st['eta'] + d['rho']*ctrl - d['mu']*drop, *d['eta']))
            target = max(1, round(rb*st['eta'])); st.update(loss=loss, f1=f1)
            stop = (r >= target and ctrl <= d['tau_ctrl']) or drop >= d['tau_drop'] or r >= rmax
            if stop:
                st['reason'] = 'drop' if drop >= d['tau_drop'] else ('rmax' if r >= rmax else 'ctrl')
            return stop

        sh[j] = self._run_rounds(trn, j, rule, meter)
        return {'contribution': contribution, 'R0': max(1, round(rb*eta)), 'stop_reason': st.get('reason')}

    # ---- checks ----
    def verify_exact(self):
        """exact: no included shard trained on a deleted row; structure_ok: every included shard trained on exactly the
        current data of its shard."""
        inc = [(j, self.shards[j]) for j in self.included()]
        return {'exact': all(not (~self.alive[e.idx]).any() for _, e in inc),
                'structure_ok': all(bool(self.assignment[j]) and np.array_equal(e.idx, self._shard_idx(self.assignment, j))
                                    for j, e in inc)}

    def verify_bit_exact(self):
        """Retrain every included shard from a fresh init (its seed, its recorded rounds, its current data) and compare."""
        bits = {}
        for j in self.included():
            e = self.shards[j]
            fresh = _Trainer(self, j, self._shard_idx(self.assignment, j), e.seed)
            for _ in range(e.rounds):
                fresh.step(None, None)
            bits[j] = all(torch.equal(a, b) for a, b in zip(fresh.m.state_dict().values(), e.model.state_dict().values()))
        active = [j for j, a in enumerate(self.assignment) if a]
        return {'bit_exact': all(bits.values()), 'shards_checked': len(bits), 'per_shard': bits,
                'included_equals_active': self.included() == active, **self.verify_exact()}

    # ---- state for the Digital Twin ----
    def client_state(self):
        """Per present client: shard, retained samples, class counts, bytes of the data it holds, version of its shard model."""
        out = []
        for c in sorted(self.present):
            j = next(i for i, a in enumerate(self.assignment) if c in a)
            rows = self.parts[c][self.alive[self.parts[c]]]
            entry = self.shards[j] if self.shards else None
            out.append({'client_id': c, 'shard': j, 'samples_retained': int(len(rows)),
                        'class_counts': np.bincount(self.train['y'][rows], minlength=2).tolist(),
                        'data_bytes': int(self.train['X'][rows].nbytes + self.train['y'][rows].nbytes),
                        'shard_rounds': None if entry is None else entry.rounds,
                        'shard_included': None if entry is None else entry.included})
        return out

    def storage(self):
        """Models the server must keep: every stored shard model and every partially trained backlog model."""
        stored = sum(e is not None for e in self.shards) + len(self.pending)
        return {'models_stored': stored, 'models_included': len(self.included()), 'model_bytes': self.model_bytes,
                'state_bytes_in_memory': stored * self.model_bytes, 'inference_forward_passes': len(self.included())}

    def benchmark_inference(self, X, repeats=5):
        """Median seconds to score X with the ensemble (one forward pass per included shard) and with one shard."""
        inc = self.included()
        if not inc:
            raise RuntimeError('no shard is included')

        def median(fn):
            samples = []
            for _ in range(repeats):
                start = time.perf_counter(); fn(); samples.append(time.perf_counter() - start)
            return float(np.median(samples))
        ens, one = median(lambda: self.margin(X)), median(lambda: logit_margin(self.shards[inc[0]].model, X))
        return {'rows': int(len(X)), 'forward_passes': len(inc), 'ensemble_seconds': ens, 'single_shard_seconds': one,
                'ensemble_seconds_per_1k_rows': 1000*ens/len(X), 'repeats': repeats}

    def fork(self, policy=None):
        """Independent copy sharing the data and the (immutable) shard models: a what-if branch for another policy."""
        other = copy.copy(self)
        other.policy = policy or self.policy
        if other.policy not in POLICIES:
            raise ValueError(f'policy must be one of {POLICIES}, got {other.policy!r}')
        other.assignment = [list(a) for a in self.assignment]
        other.alive, other.present = self.alive.copy(), set(self.present)
        other.shards = list(self.shards)
        other.pending = [p.clone() for p in self.pending]
        for p in other.pending:
            p.fed = other
        other.history, other.totals = list(self.history), dict(self.totals)
        return other

    # ---- results and checkpoints ----
    def _finish(self, meter, checkpoint_dir, head, accumulate=True):
        """Close an operation: add its training counters to the unlearning totals (not for the base fit), save, summarise."""
        if accumulate:
            for key, value in meter.train_totals().items():
                self.totals[key] += value
        files = None
        if checkpoint_dir is not None:
            with meter.timed('checkpoint'):
                files = self.save(checkpoint_dir)
        cost = meter.summary()
        cost.update(self.storage(), checkpoint_bytes_on_disk=files, cumulative_unlearning=dict(self.totals))
        return {**head, 'cost': cost, 'round_costs': meter.rounds, 'client_costs': meter.clients,
                'final_checkpoint': None if checkpoint_dir is None else Path(checkpoint_dir).as_posix()}

    def _result(self, meter, checkpoint_dir, record, retrained, retired, extra):
        checks = self.verify_exact()
        warnings = []
        if extra.get('condition_met') is False:
            warnings.append({'code': 'UTILITY_CONDITION_UNMET', 'detail': 'ok() fails although every included shard has R_base rounds'})
        if self.pending:
            warnings.append({'code': 'BACKLOG_PENDING', 'detail': f'{len(self.pending)} shard(s) out of the ensemble until backfilled',
                             'shards': [p.shard for p in self.pending]})
        if not self.included():
            warnings.append({'code': 'EMPTY_ENSEMBLE', 'detail': 'no shard is included; predictions are constant'})
        if record['event'] != 'retrain' and record['kind'] != 'backlog_done':
            warnings.append({'code': 'STRUCTURE_CHANGED', 'detail': record['event']})
        scope = EXACTNESS_SCOPE[record['policy']]
        if scope != 'unconditional':
            warnings.append({'code': 'ROUND_COUNT_DEPENDS_ON_PRE_DELETION_STATE',
                             'detail': 'the round count was chosen from signals of models trained before the deletion'})
        done = self._finish(meter, checkpoint_dir, {})
        if done['cost']['rss_bytes_end'] is None:
            warnings.append({'code': 'MEMORY_UNAVAILABLE', 'detail': 'install psutil to record process memory on this platform'})
        return UnlearningResult(
            step=record['step'], policy=record['policy'], client=record['client'], kind=record['kind'], rows=record['rows'],
            event=record['event'], retrained_shard=retrained, retired_shards=list(retired), shards_included=len(self.included()),
            exact=checks['exact'], structure_ok=checks['structure_ok'], exactness_scope=scope, details=_py(extra),
            cost=done['cost'], round_costs=done['round_costs'], client_costs=done['client_costs'], config_hash=self.config_hash,
            forget_request_hash=record['forget_request_hash'], final_checkpoint=done['final_checkpoint'],
            intermediate_checkpoint=done['final_checkpoint'] if self.pending else None, structured_warnings=warnings)

    def save(self, directory):
        """Write the full state (models, row sets, structure, history) to a new directory. Returns the bytes on disk.
        Deleted rows are stored only as an index mask and request hashes, never as records."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=False)
        arrays = {'alive': self.alive}
        meta = {'format': FORMAT, 'policy': self.policy, 'seed': self.seed, 'config': self.config.to_json(),
                'config_hash': self.config_hash, 'data': self.data_fingerprint, 'step': self.step,
                'assignment_initial': self.assignment_initial, 'assignment': self.assignment, 'present': sorted(self.present),
                'reference': self.reference, 'totals': self.totals, 'history': self.history, 'shards': [], 'pending': []}

        def write(name, mdl, idx):
            torch.save(state(mdl), out/f'{name}.pt')
            arrays[name] = idx
            return {'file': f'{name}.pt', 'sha256': _file_sha(out/f'{name}.pt')}
        for j, e in enumerate(self.shards):
            meta['shards'].append(None if e is None else {**write(f'shard_{j}', e.model, e.idx), 'seed': e.seed,
                                                          'rounds': e.rounds, 'included': e.included, 'rows': int(len(e.idx))})
        for p in self.pending:
            meta['pending'].append({**write(f'pending_{p.shard}', p.m, p.idx), 'shard': p.shard, 'seed': p.seed, 'rounds': p.r})
        np.savez_compressed(out/'rows.npz', **arrays)
        meta['rows_sha256'] = _file_sha(out/'rows.npz')
        (out/'state.json').write_text(json.dumps(_py(meta), indent=1), encoding='utf-8')
        return {'models': int(sum((out/m['file']).stat().st_size for m in meta['shards'] + meta['pending'] if m)),
                'total': int(sum(p.stat().st_size for p in out.iterdir()))}

    @classmethod
    def load(cls, directory, train, validation, parts, measure_payload=True):
        """Restore a saved state on the same data and partition; any mismatch (data, partition, file hash) is an error."""
        src = Path(directory)
        meta = json.loads((src/'state.json').read_text(encoding='utf-8'))
        if meta.get('format') != FORMAT:
            raise ValueError(f'unsupported checkpoint format {meta.get("format")!r}')
        fed = cls(train, validation, parts, meta['assignment_initial'], meta['seed'], meta['policy'],
                  ShardConfig.from_json(meta['config']), measure_payload)
        if fed.data_fingerprint != meta['data']:
            bad = sorted(k for k in meta['data'] if fed.data_fingerprint.get(k) != meta['data'][k])
            raise ValueError(f'checkpoint was made on different data: mismatch in {bad}')
        if _file_sha(src/'rows.npz') != meta['rows_sha256']:
            raise ValueError('checkpoint file hash mismatch: rows.npz')
        with np.load(src/'rows.npz', allow_pickle=False) as archive:
            arrays = dict(archive)

        def read(item):
            if _file_sha(src/item['file']) != item['sha256']:
                raise ValueError(f'checkpoint file hash mismatch: {item["file"]}')
            mdl = model(fed.dim, list(fed.config.hidden), 0, fed.config.device)
            mdl.load_state_dict(torch.load(src/item['file'], map_location=fed.config.device, weights_only=True))
            return mdl
        fed.alive = arrays['alive'].astype(bool)
        fed.assignment = [list(a) for a in meta['assignment']]
        fed.present, fed.step, fed.history, fed.totals = set(meta['present']), meta['step'], meta['history'], meta['totals']
        fed.reference = tuple(meta['reference'])
        for j, item in enumerate(meta['shards']):
            fed.shards.append(None if item is None else Shard(read(item), arrays[f'shard_{j}'], item['seed'], item['rounds'],
                                                              item['included']))
        for item in meta['pending']:
            fed.pending.append(_Trainer(fed, item['shard'], arrays[f'pending_{item["shard"]}'], item['seed'],
                                        start=read(item), rounds=item['rounds']))
        return fed


class ShardExactMethod:
    """UnlearningMethod interface of the implementation guide (section 5.4) over saved states.

    run() loads the state in `original_checkpoint`, applies one request and saves the new state to `output_dir`.
    forget_request: train row indices, or None for the whole client; config: {'policy': name} (optional).
    """

    def __init__(self, train, validation, parts, measure_payload=True):
        self.train, self.validation, self.parts, self.measure_payload = train, validation, parts, measure_payload

    def run(self, original_checkpoint, target_client, forget_request, config, output_dir):
        if Path(output_dir).exists():
            raise FileExistsError(f'Refusing overwrite: {output_dir}')
        fed = ShardedUnlearner.load(original_checkpoint, self.train, self.validation, self.parts, self.measure_payload)
        return fed.unlearn(ForgetRequest(target_client, forget_request), (config or {}).get('policy'), output_dir)
