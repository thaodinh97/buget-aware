"""The C1/C2 runner with the MLP hidden sizes taken from the config key 'hidden' (stage s5 of docs/dt_sensitivity_plan.md).

    python scripts/sens/run_hidden.py OUT --config configs/sens/s5_h32-16.json --plan docs/dt_sensitivity_plan.md

scripts/dt/run_orchestration.py builds ShardConfig with the default hidden (64, 32); this wrapper replaces only that one
call, so training, unlearning, the twin and the audits are the runner's own code. Everything that sizes the model
(training, payload bytes, verify_bit_exact) reads config.hidden.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.dt.run_orchestration as runner
from scripts.dt.run_predictive_confirmation import keep_awake
from src.fu.shard_exact import ShardConfig


def patch(hidden):
    hidden = tuple(int(h) for h in hidden)

    class WithHidden:
        @staticmethod
        def from_train_config(cfg, **overrides):
            return ShardConfig.from_train_config(cfg, hidden=hidden, **overrides)

    runner.ShardConfig = WithHidden


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    config = Path(argv[argv.index('--config')+1])
    patch(json.loads(config.read_text(encoding='utf-8'))['hidden'])
    runner.main(argv)


if __name__ == '__main__':
    with keep_awake():
        main()
