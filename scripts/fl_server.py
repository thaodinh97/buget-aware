"""Round-synchronous HTTP FedAvg coordinator for Docker-networked clients."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from threading import Condition, Thread
from urllib.parse import parse_qs, urlparse

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fl.distributed import aggregate_updates, pack_state, unpack_state


class Coordinator:
    def __init__(self, initial_state, client_manifest, rounds, output):
        self.initial_state = initial_state
        self.expected = {int(client['client_id']): int(client['samples'])
                         for client in client_manifest['clients']}
        if not self.expected or set(self.expected) != set(range(len(self.expected))):
            raise ValueError('Client manifest IDs must be contiguous from zero')
        self.rounds = rounds
        self.output = output
        self.current = 0
        self.updates = {}
        self.logs = []
        self.completed = set()
        self.condition = Condition()
        self.stopped = False

    def next_model(self, client_id):
        with self.condition:
            if client_id not in self.expected:
                raise ValueError('Unknown client')
            self.condition.wait_for(
                lambda: self.current >= self.rounds or client_id not in self.updates,
                timeout=300,
            )
            if self.current < self.rounds and client_id in self.updates:
                raise TimeoutError('Timed out waiting for the other clients')
            if self.current >= self.rounds:
                self.completed.add(client_id)
                if len(self.completed) == len(self.expected):
                    self._save_result()
                    Thread(target=self.server.shutdown, daemon=True).start()
                return None
            return self.current, pack_state(self.initial_state, self.current)

    def submit(self, client_id, round_id, sample_count, loss, steps, payload):
        with self.condition:
            if client_id not in self.expected:
                raise ValueError('Unknown client')
            if round_id != self.current:
                raise ValueError('Stale or future round')
            if sample_count != self.expected[client_id]:
                raise ValueError('Client sample count does not match frozen partition')
            if client_id in self.updates:
                raise ValueError('Duplicate client update')
            update = unpack_state(payload, self.initial_state, round_id)
            self.updates[client_id] = (update, sample_count)
            client_logs = getattr(self, 'client_logs', {})
            client_logs[client_id] = {'loss': loss, 'optimizer_steps': steps}
            self.client_logs = client_logs
            if len(self.updates) == len(self.expected):
                ordered = [self.updates[i] for i in sorted(self.expected)]
                self.initial_state = aggregate_updates(ordered)
                self.logs.append({
                    'round': self.current,
                    'client_samples': {str(i): self.expected[i] for i in sorted(self.expected)},
                    'client_ids': sorted(self.updates),
                    'clients': {str(i): client_logs[i] for i in sorted(client_logs)},
                    'upload_payload_bytes': sum(len(pack_state(update, round_id)) for update, _ in ordered),
                    'download_payload_bytes': len(pack_state(self.initial_state, round_id)) * len(self.expected),
                })
                self.current += 1
                self.updates.clear()
                self.client_logs = {}
                if self.current == self.rounds:
                    self._save_checkpoint()
                self.condition.notify_all()

    def _save_checkpoint(self):
        torch.save({'state_dict': self.initial_state, 'rounds': self.rounds}, self.output / 'global_model.pt')

    def _save_result(self):
        result = {'status': 'completed', 'rounds': self.rounds, 'clients': len(self.expected),
                  'sample_total': sum(self.expected.values()), 'round_metrics': self.logs}
        (self.output / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        (self.output / 'manifest.json').write_text(json.dumps({
            'clients': len(self.expected), 'rounds': self.rounds,
            'aggregation': 'sample-count weighted FedAvg over HTTP',
            'transport': 'Docker internal network; HTTP; compressed NumPy NPZ; no TLS',
            'secure_aggregation': False, 'differential_privacy': False,
            'sample_ids_transmitted': False,
        }, indent=2), encoding='utf-8')


def make_handler(coordinator):
    class Handler(BaseHTTPRequestHandler):
        def _respond(self, status, body, content_type='application/json'):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == '/health':
                return self._respond(200, b'{"status":"ready"}')
            if parsed.path != '/next':
                return self._respond(404, b'{"error":"not found"}')
            try:
                client_id = int(parse_qs(parsed.query)['client_id'][0])
                response = coordinator.next_model(client_id)
                if response is None:
                    return self._respond(200, b'{"status":"complete"}')
                round_id, payload = response
                return self._respond(200, payload, 'application/octet-stream')
            except (KeyError, ValueError) as exc:
                return self._respond(400, json.dumps({'error': str(exc)}).encode())
            except TimeoutError as exc:
                return self._respond(408, json.dumps({'error': str(exc)}).encode())

        def do_POST(self):
            if urlparse(self.path).path != '/update':
                return self._respond(404, b'{"error":"not found"}')
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if length <= 0 or length > 10 * 1024 * 1024:
                    raise ValueError('Invalid update payload length')
                client_id = int(self.headers['X-Client-ID'])
                round_id = int(self.headers['X-Round-ID'])
                sample_count = int(self.headers['X-Sample-Count'])
                loss = float(self.headers['X-Client-Loss'])
                steps = int(self.headers['X-Optimizer-Steps'])
                if not torch.isfinite(torch.tensor(loss)) or steps <= 0:
                    raise ValueError('Invalid client training metrics')
                coordinator.submit(client_id, round_id, sample_count, loss, steps, self.rfile.read(length))
                return self._respond(200, b'{"status":"accepted"}')
            except (KeyError, ValueError, OSError) as exc:
                return self._respond(400, json.dumps({'error': str(exc)}).encode())

        def log_message(self, fmt, *args):
            print(f'{self.client_address[0]} {fmt % args}', flush=True)
    return Handler


def run(checkpoint, manifest_path, output, rounds, host, port):
    if output.exists():
        raise FileExistsError(f'Refusing overwrite: {output}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    output.mkdir(parents=True)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    initial = saved['state_dict']
    coordinator = Coordinator(initial, manifest, rounds, output)
    coordinator.server = ThreadingHTTPServer((host, port), make_handler(coordinator))
    print(f'Server ready on {host}:{port}; clients={len(coordinator.expected)} rounds={rounds}', flush=True)
    coordinator.server.serve_forever()
    coordinator.server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--client-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=15)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    if args.rounds <= 0:
        parser.error('--rounds must be positive')
    run(args.checkpoint, args.client_manifest, args.output, args.rounds, args.host, args.port)
