import unittest

import numpy as np
import torch

from src.fl.distributed import aggregate_updates, pack_state, unpack_state


class DistributedFLTests(unittest.TestCase):
    def test_state_round_trip_and_round_guard(self):
        state = {
            'weight': torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            'counter': torch.tensor([3], dtype=torch.int64),
        }
        payload = pack_state(state, 4)
        result = unpack_state(payload, state, 4)
        self.assertTrue(torch.equal(result['weight'], state['weight']))
        self.assertTrue(torch.equal(result['counter'], state['counter']))
        with self.assertRaisesRegex(ValueError, 'Unexpected model round'):
            unpack_state(payload, state, 3)

    def test_weighted_client_aggregation(self):
        result = aggregate_updates([
            ({'weight': torch.tensor([1.0])}, 1),
            ({'weight': torch.tensor([3.0])}, 3),
        ])
        self.assertAlmostEqual(result['weight'].item(), 2.5)
        with self.assertRaisesRegex(ValueError, 'No client updates'):
            aggregate_updates([])

    def test_rejects_nonfinite_update(self):
        state = {'weight': torch.tensor([1.0])}
        payload = pack_state({'weight': torch.tensor([np.nan])}, 0)
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            unpack_state(payload, state, 0)


if __name__ == '__main__':
    unittest.main()
