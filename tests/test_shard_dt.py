import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from scripts.shard.shard_dt import execute
from scripts.shard import shard_service
from src.fu.shard_exact import ForgetRequest
from test_shard_exact import synthetic, TorchCase, TRAIN_CFG


class DigitalTwinTests(TorchCase):
    def test_admission_continuation_merge_and_exactness(self):
        data = synthetic(1200)
        with tempfile.TemporaryDirectory() as temp, patch.dict(shard_service.CONFIGS, tiny=TRAIN_CFG):
            root = Path(temp)
            base = root / 'init'
            fed, _ = shard_service.init_state(data, base, 2, 42, 'shard_full', clients=6, min_clients=3,
                                              train_config='tiny', device=os.environ.get('DT_TEST_DEVICE', 'cpu'))
            c = fed.assignment[0][0]
            sample = ForgetRequest(c, fed.parts[c][:1])
            rejected = execute(data, base, sample, root/'rejected', 'static', 10000, 0)
            self.assertEqual(rejected['status'], 'NO_FEASIBLE_PLAN')
            self.assertFalse((root/'rejected'/'state.json').exists())
            limit = rejected['candidates'][0]['raw_bytes'] // 2
            accepted = execute(data, base, sample, root/'one', 'adaptive', 10000, limit, True)
            self.assertEqual(accepted['selected']['policy'], 'shard_fixed')
            self.assertEqual(accepted['actual_raw_bytes'], limit)
            report = execute(data, root/'one', ForgetRequest(c), root/'two', 'adaptive', 10000, 10**9)
            self.assertTrue(report['event'].startswith('merge_'))
            self.assertNotIn(c, report['clients'])
            self.assertEqual(report['actual_raw_bytes'], report['selected']['raw_bytes'])
            final, _ = shard_service.load_state(data, root/'two')
            self.assertTrue(final.verify_bit_exact()['bit_exact'])
            profiles = json.loads((root/'two'/'twin.json').read_text())
            self.assertNotEqual(profiles['static'], profiles['adaptive'])
