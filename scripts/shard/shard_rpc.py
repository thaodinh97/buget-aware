"""Sequential HTTP workers for the seed-50042 client-10 shard-FU parity experiment."""
import argparse
import hashlib
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import torch

from scripts.shard.shard_service import DATA_DEFAULT, load_state, write_costs
from src.data.loading import load_data
from src.fl.distributed import pack_state, unpack_state
from src.fl.fedavg_core import ClientWorker, aggregate, model, state
from src.fu.shard_exact import ForgetRequest
from src.eval.utility_breakdown import detector_metrics


def serve(cid, manifest):
    spec = next(c for c in manifest['clients'] if c['client_id'] == cid)
    path = Path('/worker/client.npz')
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
        raise ValueError('Worker partition hash mismatch')
    with np.load(path, allow_pickle=False) as data:
        worker = ClientWorker(cid, data['X'], data['y'], data['row_id'], 'cuda')
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == '/health' else 404)
            self.end_headers()

        def do_POST(self):
            if self.path != '/train':
                self.send_error(404)
                return
            try:
                job = json.loads(self.headers['X-Job'])
                if job['client'] != cid:
                    raise ValueError('Unexpected worker ID')
                length = int(self.headers['Content-Length'])
                if not 0 < length <= 1048576:
                    raise ValueError('Unexpected payload size')
                mdl = model(worker.x.shape[1], job['hidden'], 0, 'cuda')
                mdl.load_state_dict(unpack_state(self.rfile.read(length), state(mdl), job['round']))
                update, count, log = worker.train(mdl, job['fl'], job['seed'])
                body = pack_state(update, job['round'])
            except Exception as exc:
                self.send_error(400, str(exc))
                return
            self.send_response(200)
            self.send_header('X-Job', self.headers['X-Job'])
            self.send_header('X-Result', json.dumps([count, log]))
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    HTTPServer(('', 8080), Handler).serve_forever()


class Remote:
    def __init__(self, cid, rows, config, records, first):
        self.client_id, self.count = cid, len(rows)
        self.hidden, self.records, self.first = list(config.hidden), records, first
        self.round = 0
        self.rows = rows


    def train(self, mdl, fl, seed):
        self.round += 1
        before = state(mdl)
        payload = pack_state(before, self.round)
        job = dict(client=self.client_id, round=self.round, hidden=self.hidden, fl=fl, seed=int(seed))
        header = json.dumps(job)
        req = Request(f'http://worker{self.client_id}:8080/train', data=payload, headers={'X-Job': header})
        start = time.perf_counter()
        with urlopen(req, timeout=180) as response:
            if response.headers['X-Job'] != header:
                raise ValueError('Unexpected RPC response')
            count, log = json.loads(response.headers['X-Result'])
            body = response.read()
        elapsed = time.perf_counter() - start
        update = unpack_state(body, before, self.round)
        if count != self.count or log['client_id'] != self.client_id:
            raise ValueError('Wrong client or number of rows')
        if self.round == 1:
            self.first.append((self.client_id, self.rows, before, update, seed))
        self.records.append(dict(round=self.round, client=self.client_id, down=len(payload), up=len(body),
                                 rpc_seconds=elapsed))
        return update, count, log


def run(args, manifest):
    data = load_data(Path(DATA_DEFAULT))
    fed, layout = load_state(data, args.base)
    if fed.seed != manifest['seed'] or fed.step != 0 or fed.config.device != 'cuda':
        raise ValueError('Expected the original GPU ensemble of this pilot')
    records, first = [], []
    specs = {c['client_id']: c for c in manifest['clients']}
    def factory(cid, rows):
        if cid not in specs or len(rows) != specs[cid]['samples']:
            raise ValueError('Assignment differs from the exported pilot')
        return Remote(cid, rows, fed.config, records, first)
    fed.worker_factory = factory
    if args.output.exists():
        raise FileExistsError(args.output)
    result = fed.unlearn(ForgetRequest(manifest['removed_client']), args.policy, args.output)
    summary = result.to_json()
    rounds, clients = summary.pop('round_costs'), summary.pop('client_costs')
    write_costs(args.output, dict(operation='unlearn', **summary), rounds, clients, layout)
    # Independent local retrain uses the normal ClientWorker for verification.
    del fed.worker_factory
    local_updates, remote_updates, counts = [], [], []
    for cid, rows, before, remote, seed in first:
        mdl = model(fed.dim, list(fed.config.hidden), 0, 'cuda')
        mdl.load_state_dict(before)
        tr = fed.train
        worker = ClientWorker(cid, tr['X'][rows], tr['y'][rows], tr['row_id'][rows], 'cuda')
        update, count, _ = worker.train(mdl, fed.config.fl, seed)
        local_updates.append(update)
        remote_updates.append(remote)
        counts.append(count)
    expected, actual = aggregate(local_updates, counts), aggregate(remote_updates, counts)
    first_equal = all(torch.equal(actual[k], v) for k, v in expected.items())
    verified = fed.verify_bit_exact()
    metrics = detector_metrics(data['validation']['y'], fed.predict_proba(data['validation']['X']))
    report = dict(first_round_bit_exact=first_equal, policy=args.policy, clients=sorted({r['client'] for r in records}), calls=len(records),
                  raw=result.cost['train_bytes_raw'], http_body=sum(r['up']+r['down'] for r in records),
                  round_seconds=sum(r['wall_seconds'] for r in rounds), validation_macro_f1=metrics['macro_f1'],
                  validation_fpr=metrics['fpr'], verified=verified)
    (args.output/'rpc_calls.json').write_text(json.dumps(records, indent=2))
    (args.output/'rpc_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not first_equal or not verified['bit_exact'] or not verified['structure_ok'] or not verified['included_equals_active']:
        raise RuntimeError('RPC checkpoint verification failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['worker','run'])
    parser.add_argument('--client', type=int)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--policy', choices=['shard_full','shard_fixed'], default='shard_full')
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    manifest = json.loads(Path('/worker/manifest.json').read_text())
    if args.mode == 'worker':
        serve(args.client, manifest)
    else:
        run(args, manifest)
