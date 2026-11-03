"""Write a small synthetic CSV with the ToN-IoT network schema used by prepare.py.

For pipeline smoke tests only: the rows are random and never evidence about FU on ToN-IoT.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

FAMILIES = ['normal', 'backdoor', 'ddos', 'dos', 'injection', 'password', 'ransomware', 'scanning', 'xss', 'mitm']


def synthetic(rows_per_family=400, normal_rows=3000, rare_rows=120, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for code, family in enumerate(FAMILIES):
        n = normal_rows if family == 'normal' else rare_rows if family == 'mitm' else rows_per_family
        centre = rng.normal(0, 1.2, 8)
        values = np.exp(centre + rng.normal(0, 1, (n, 8)))
        frames.append(pd.DataFrame({
            'src_ip': [f'10.{code}.{i // 250}.{i % 250}' for i in range(n)],
            'src_port': rng.integers(1024, 65535, n), 'dst_ip': [f'192.168.{code}.{i % 200}' for i in range(n)],
            'dst_port': rng.choice([53, 80, 443, 8080, int(rng.integers(1, 1024))], n),
            'proto': rng.choice(['tcp', 'udp', 'icmp'], n, p=rng.dirichlet(np.ones(3))),
            'service': rng.choice(['-', 'http', 'dns', 'ssl'], n, p=rng.dirichlet(np.ones(4))),
            'conn_state': rng.choice(['S0', 'SF', 'REJ', 'OTH'], n, p=rng.dirichlet(np.ones(4))),
            'duration': values[:, 0], 'src_bytes': values[:, 1].round(), 'dst_bytes': values[:, 2].round(),
            'missed_bytes': (values[:, 3] * (rng.random(n) < .1)).round(), 'src_pkts': values[:, 4].round(),
            'src_ip_bytes': values[:, 5].round(), 'dst_pkts': values[:, 6].round(), 'dst_ip_bytes': values[:, 7].round(),
            'label': int(family != 'normal'), 'type': family}))
    return pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing overwrite: {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    synthetic(seed=args.seed).to_csv(args.output, index=False)
