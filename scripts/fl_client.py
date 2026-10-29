"""Train one local partition and exchange model weights with the Docker FL server."""
import argparse
import hashlib
from io import BytesIO
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fl.distributed import pack_state, unpack_state
from src.fl.fedavg_core import ClientWorker, get_device, model, state


def request(url, method='GET', data=None, headers=None, retries=60):
    error = None
    for _ in range(retries):
        try:
            with urlopen(Request(url, data=data, headers=headers or {}, method=method), timeout=300) as response:
                return response.read(), response.headers.get('Content-Type', '')
        except (URLError, TimeoutError, ConnectionError) as exc:
            error = exc
            time.sleep(2)
        except HTTPError as exc:
            raise RuntimeError(f'FL server returned HTTP {exc.code}: {exc.read().decode(errors="replace")}') from exc
    raise RuntimeError(f'Unable to reach FL server: {error}')


def run(client_id, data_path, manifest_path, url, seed, config):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected = next((client for client in manifest['clients'] if client['client_id'] == client_id), None)
    if expected is None:
        raise ValueError(f'Client {client_id} is absent from shard manifest')
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if digest != expected['sha256']:
        raise ValueError('Client shard hash does not match manifest')
    with np.load(data_path, allow_pickle=False) as archive:
        x, y = archive['X'], archive['y']
    if len(y) != expected['samples'] or x.shape[1] != manifest['input_dim']:
        raise ValueError('Client shard shape does not match manifest')
    device = get_device()
    torch.set_num_threads(config['threads'])
    print(f'Client {client_id} ready: {len(y)} samples on {device}', flush=True)
    worker = ClientWorker(client_id, x, y, np.arange(len(y)), device=device)
    global_model = model(x.shape[1], config['hidden'], seed, device=device)
    while True:
        payload, content_type = request(f'{url}/next?client_id={client_id}')
        if content_type.startswith('application/json'):
            status = json.loads(payload)
            if status.get('status') == 'complete':
                print(f'Client {client_id} completed all rounds', flush=True)
                return
            raise RuntimeError(f'Unexpected server response: {status}')
        if not content_type.startswith('application/octet-stream'):
            raise RuntimeError(f'Unexpected model response content type: {content_type}')
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            round_id = int(archive['__round__'])
        global_model.load_state_dict(unpack_state(payload, state(global_model), round_id))
        local_state, sample_count, log = worker.train(
            global_model, config['fl'], seed + round_id * 10000 + client_id * 100)
        update = pack_state(local_state, round_id)
        request(f'{url}/update', method='POST', data=update, headers={
            'Content-Type': 'application/octet-stream',
            'X-Client-ID': str(client_id),
            'X-Round-ID': str(round_id),
            'X-Sample-Count': str(sample_count),
            'X-Client-Loss': str(log['loss']),
            'X-Optimizer-Steps': str(log['steps']),
        })
        print(f'Client {client_id} submitted round {round_id}; loss={log["loss"]:.6f}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--client-id', type=int, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--client-manifest', type=Path, required=True)
    parser.add_argument('--url', default=os.environ.get('FL_SERVER_URL', 'http://server:8000'))
    parser.add_argument('--config', type=Path, default=Path('configs/pilot.json'))
    args = parser.parse_args()
    run(args.client_id, args.data, args.client_manifest, args.url, 42,
        json.loads(args.config.read_text(encoding='utf-8')))
