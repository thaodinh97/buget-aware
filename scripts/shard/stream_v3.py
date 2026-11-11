"""Request streams of the v3 shard protocol (docs/shard_v3_plan.md). Importable, no side effects.

Every request is SAMPLE-LEVEL: it deletes some rows of one client and the client stays in the federation, so the shard
structure never changes (no merge, no removal). The plan of a stream depends only on metadata: the number of training rows of
each client, the shard assignment and the seeds below. It never reads features, labels or a model.

- STREAM: fixed parameters of the stream model (K requests, size distribution, arrival process, Zipf exponent, cap, seeds);
- request_sizes / arrival_times: shared by every family, so families differ only in WHERE the requests land;
- plan_stream: one family -> list of requests {step, time, client, shard, requested, rows};
  proportional  client with probability proportional to its remaining deletable rows (neutral reference);
  spread        shard k mod S in turn (scenario 6: as many different shards as possible), client inside the shard as above;
  concentrated  always the largest shard at design time (scenario 6: one shard absorbs every request), client as above;
  poisson_zipf  client by Zipf(s) over a seeded ranking of the clients, independent of their size (scenario C);
- select_rows: the rows a request deletes, uniformly among the alive rows of its client (seeded per step);
- batch_windows: request -> processing time for a batching window W (W = 0: immediately); last window flushed at the end;
- cost_plan: rows x rounds of shard_full / batch_full_W and of the single-model reference, from metadata only.

Run as a script to print the precomputed plan of docs/shard_v3_plan.md:
python scripts/shard/stream_v3.py results/shard_v2/ensemble_check_20c/meta.json
"""
import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.shard.shard_v2 import assign  # noqa: E402  (same seeded assignment as v2)

FAMILIES = ['proportional', 'spread', 'concentrated', 'poisson_zipf']
STREAM = {'K': 20, 'cap': .5, 'size_median': 200, 'size_sigma': 1.5, 'rate': 1., 'zipf_s': 1.,
          'seed_size': 203, 'seed_time': 204, 'seed_choice': 201, 'seed_rank': 202, 'seed_rows': 205}
WINDOWS = [0, 2, 5]


def request_sizes(K=STREAM['K'], median=STREAM['size_median'], sigma=STREAM['size_sigma'], seed=STREAM['seed_size']):
    """Requested sizes in rows: ceil(LogNormal(log median, sigma)), heavy-tailed; shared by every family."""
    z = np.random.default_rng(seed).normal(size=K)
    return [int(math.ceil(math.exp(math.log(median) + sigma*v))) for v in z]


def arrival_times(K=STREAM['K'], rate=STREAM['rate'], seed=STREAM['seed_time']):
    """Poisson process: cumulative sums of Exp(rate) inter-arrival times; shared by every family."""
    return np.cumsum(np.random.default_rng(seed).exponential(1/rate, size=K)).tolist()


def zipf_ranking(clients, seed=STREAM['seed_rank']):
    """rank[c] in 1..clients from a seeded permutation, so popularity is independent of client size."""
    perm = np.random.default_rng(seed).permutation(clients)
    rank = np.empty(clients, int); rank[perm] = np.arange(1, clients + 1)
    return rank


def plan_stream(family, client_rows, assignment, cfg=STREAM):
    """Requests of one family. client_rows: training rows per client; assignment: list of shards (lists of client ids)."""
    if family not in FAMILIES: raise ValueError(f'unknown family {family!r}')
    n = np.asarray(client_rows, int); C = len(n)
    if sorted(c for a in assignment for c in a) != list(range(C)): raise ValueError('assignment must cover every client once')
    shard_of = {c: j for j, a in enumerate(assignment) for c in a}
    capacity = np.floor(cfg['cap']*n).astype(int)            # rows a client may lose in total; it always keeps >= half
    sizes = request_sizes(cfg['K'], cfg['size_median'], cfg['size_sigma'], cfg['seed_size'])
    times = arrival_times(cfg['K'], cfg['rate'], cfg['seed_time'])
    rng = np.random.default_rng([cfg['seed_choice'], FAMILIES.index(family)])
    rank = zipf_ranking(C, cfg['seed_rank'])
    sizes_of_shards = [int(n[a].sum()) for a in assignment]
    target = int(np.argmax(sizes_of_shards))                 # ties: lowest index
    out = []
    for k in range(cfg['K']):
        if family == 'spread':
            order = [(k + i) % len(assignment) for i in range(len(assignment))]
            pool = next((assignment[j] for j in order if capacity[assignment[j]].sum() > 0), [])
        elif family == 'concentrated':
            pool = assignment[target]
        else:
            pool = list(range(C))
        pool = [c for c in pool if capacity[c] > 0]
        if not pool: raise ValueError(f'{family}: no client left with deletable rows at step {k + 1}')
        if family == 'poisson_zipf': w = np.array([rank[c]**-cfg['zipf_s'] for c in pool], float)
        else: w = capacity[pool].astype(float)
        c = int(pool[rng.choice(len(pool), p=w/w.sum())])
        rows = int(min(sizes[k], capacity[c])); capacity[c] -= rows
        out.append({'step': k + 1, 'time': float(times[k]), 'client': c, 'shard': shard_of[c],
                    'requested': sizes[k], 'rows': rows, 'kind': 'sample'})
    return out


def select_rows(alive_client_rows, n, step, seed=STREAM['seed_rows']):
    """Rows deleted by request `step`: n rows uniformly among the alive rows of its client (row indices only)."""
    alive = np.sort(np.asarray(alive_client_rows, np.int64))
    if not 0 < n <= len(alive): raise ValueError('request size must be in 1..alive rows of the client')
    return np.sort(np.random.default_rng([seed, step]).choice(alive, n, replace=False))


def batch_windows(times, W):
    """Processing time of every request: W = 0 -> its arrival time; else the end of its window [mW, (m+1)W).
    Requests of the last window are processed at that window's end as well (final flush), so every request is served."""
    if W == 0: return [float(t) for t in times]
    return [float((math.floor(t/W) + 1)*W) for t in times]


def cost_plan(stream, assignment, client_rows, rounds, W=0):
    """rows x rounds of shard_full (W = 0) or batch_full_W, and of the single-model reference retrained after every request.
    Every processing time retrains each touched shard once, on its data after all requests processed so far."""
    n = np.asarray(client_rows, int).copy(); N0 = int(n.sum())
    shard_rows = lambda j: int(n[assignment[j]].sum())
    proc = batch_windows([r['time'] for r in stream], W)
    groups = {}
    for r, p in zip(stream, proc): groups.setdefault(p, []).append(r)
    shard_cost, retrains, deleted = 0, 0, 0
    for p in sorted(groups):
        for r in groups[p]: n[r['client']] -= r['rows']; deleted += r['rows']
        touched = sorted({r['shard'] for r in groups[p]})
        shard_cost += sum(shard_rows(j) for j in touched)*rounds; retrains += len(touched)
    single, left = 0, N0
    for r in stream: left -= r['rows']; single += left*rounds
    lat = [p - r['time'] for r, p in zip(stream, proc)]
    return {'W': W, 'shard_rows_x_rounds': shard_cost, 'shard_retrains': retrains, 'batches': len(groups),
            'single_rows_x_rounds': single, 'cost_frac': shard_cost/single, 'deleted_rows': deleted,
            'deleted_frac': deleted/N0, 'latency_mean': float(np.mean(lat)), 'latency_max': float(np.max(lat))}


def backfill_plan(shard_seq, S, rounds=20, budget=7):
    """Shards included in the ensemble after every request under drop_backfill (same rule as run_shard_stream_v2.py): the
    requested shard leaves the ensemble and is (re)queued from a fresh init; each request then gives `budget` rounds to the
    FIFO queue; a shard rejoins once it has `rounds` rounds. Depends only on the sequence of requested shards."""
    included, pending, out = set(range(S)), [], []
    for j in shard_seq:
        included.discard(j); pending = [p for p in pending if p[0] != j] + [[j, 0]]; b = budget
        while b and pending:
            n = min(b, rounds - pending[0][1]); pending[0][1] += n; b -= n
            if pending[0][1] >= rounds: included.add(pending.pop(0)[0])
        out.append(sorted(included))
    return out


def summary(client_rows, shards=(4, 5), rounds=20, cfg=STREAM):
    rows = []
    for S in shards:
        A = assign(S, len(client_rows))
        for fam in FAMILIES:
            st = plan_stream(fam, client_rows, A, cfg)
            touched = [r['shard'] for r in st]
            for W in WINDOWS:
                c = cost_plan(st, A, client_rows, rounds, W)
                rows.append({'S': S, 'family': fam, **c, 'distinct_shards': len(set(touched)),
                             'max_share_one_shard': max(touched.count(j) for j in set(touched))/len(st),
                             'truncated': sum(r['rows'] < r['requested'] for r in st)})
    return rows


if __name__ == '__main__':
    meta = json.loads(Path(sys.argv[1]).read_text()); n = meta['client_rows']
    print('sizes', request_sizes(), '\ntimes', [round(t, 2) for t in arrival_times()])
    for S in (4, 5):
        A = assign(S, len(n)); print(f'\nS={S} shards {A} rows {[int(sum(n[c] for c in a)) for a in A]}')
        for fam in FAMILIES:
            st = plan_stream(fam, n, A)
            print(f'  {fam:13s} clients {[r["client"] for r in st]}\n  {"":13s} shards  {[r["shard"] for r in st]}')
            inc = backfill_plan([r['shard'] for r in st], S)
            print(f'  {"":13s} drop_backfill included per step {[len(i) for i in inc]}; '
                  f'empty at steps {[k + 1 for k, i in enumerate(inc) if not i]}; '
                  f'mean included rows {np.mean([sum(sum(n[c] for c in A[j]) for j in i) for i in inc])/sum(n):.2f} of train')
    print()
    for r in summary(n):
        print(f"S={r['S']} {r['family']:13s} W={r['W']}: deleted {r['deleted_rows']:6d} ({100*r['deleted_frac']:.1f}%), "
              f"retrains {r['shard_retrains']:2d} in {r['batches']:2d} batches, cost {100*r['cost_frac']:5.1f}% of single, "
              f"latency mean {r['latency_mean']:.2f} max {r['latency_max']:.2f}, distinct shards {r['distinct_shards']}, "
              f"max share {r['max_share_one_shard']:.2f}, truncated {r['truncated']}")
