import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import src.data.prepare as base
from src.data.loading import load_data
from src.data.prepare_log1p import check_numeric_transform, fit_transformer, reencode, transform


class AppendLog1pTests(unittest.TestCase):
    def test_layout_is_base_zscore_then_log1p_then_base_one_hot(self):
        fit = pd.DataFrame({'x': [0., 9., 99., np.nan], 'proto': ['tcp', 'udp', '-', 'tcp']})
        state = fit_transformer(fit, ['x'], ['proto'])
        plain_state = base.fit_transformer(fit, ['x'], ['proto'])
        logs = np.log1p(np.array([0., 9., 99., 9.]))  # median of 0, 9, 99 is 9
        self.assertAlmostEqual(state['log_mean']['x'], logs.mean())
        self.assertAlmostEqual(state['log_scale']['x'], logs.std())
        self.assertEqual(state['mean'], plain_state['mean'])
        self.assertEqual(state['feature_names'][:2], ['x', 'log1p(x)'])
        self.assertEqual(state['feature_names'][2:], plain_state['feature_names'][1:])
        frame = pd.DataFrame({'x': [1000., np.nan], 'proto': ['new', '-']})
        got, plain = transform(frame, state), base.transform(frame, plain_state)
        self.assertEqual(got.shape[1], plain.shape[1] + 1)
        np.testing.assert_array_equal(got[:, 0], plain[:, 0])
        np.testing.assert_allclose(got[:, 1], (np.log1p([1000., 9.]) - logs.mean())/logs.std(), rtol=1e-6)
        np.testing.assert_array_equal(got[:, 2:], plain[:, 1:])

    def test_negative_values_and_unfitted_state_fail(self):
        with self.assertRaisesRegex(ValueError, 'non-negative'):
            fit_transformer(pd.DataFrame({'x': [-1., 2.], 'proto': ['tcp', 'udp']}), ['x'], ['proto'])
        plain = base.fit_transformer(pd.DataFrame({'x': [1., 2.], 'proto': ['tcp', 'udp']}), ['x'], ['proto'])
        with self.assertRaisesRegex(ValueError, 'append_log1p'):
            transform(pd.DataFrame({'x': [1.], 'proto': ['tcp']}), plain)

    def test_reencode_keeps_split_rows_and_base_columns(self):
        cfg = {**json.loads(Path('configs/data.json').read_text()), 'transform_fit': 'validation_only'}
        rows = []
        for i, group in enumerate(range(100)):
            for label in [0, 1]:  # distinct feature vectors, so the cross-split overlap gate opens
                row = {col: float(10**(group % 5)*(group + 1) + label + j) for j, col in enumerate(cfg['numeric_features'])}
                row.update(src_ip=f'10.0.0.{group}', dst_ip='10.1.0.1', src_port=group, dst_port=80, proto='tcp',
                           service='dns' if group < 3 else '-', conn_state='SF', label=label,
                           type='normal' if label == 0 else 'attack')
                rows.append(row)
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            root = Path(tmp)
            source = root/'input.csv'
            pd.DataFrame(rows).to_csv(source, index=False)
            (root/'plain.json').write_text(json.dumps(cfg))
            (root/'log.json').write_text(json.dumps({**cfg, 'numeric_transform': 'append_log1p'}))
            base.prepare(source, root/'plain.json', root/'plain')
            reencode(root/'plain', source, root/'log.json', root/'log')
            check_numeric_transform(root/'log', 'append_log1p')
            check_numeric_transform(root/'plain', 'none')
            with self.assertRaises(ValueError):
                check_numeric_transform(root/'plain', 'append_log1p')
            plain, log = load_data(root/'plain'), load_data(root/'log')  # the gate and artifact hashes pass
            k = len(cfg['numeric_features'])
            for name in plain:
                for key in ('y', 'row_id', 'group_id'):
                    np.testing.assert_array_equal(plain[name][key], log[name][key])
                np.testing.assert_array_equal(plain[name]['X'][:, :k], log[name]['X'][:, :k])
                np.testing.assert_array_equal(plain[name]['X'][:, k:], log[name]['X'][:, 2*k:])
            manifest = json.loads((root/'log/manifest.json').read_text())
            self.assertEqual(manifest['numeric_transform'], 'append_log1p')
            self.assertEqual(manifest['reencoded_from']['dir'], (root/'plain').as_posix())
            with self.assertRaisesRegex(ValueError, 'numeric_transform'):
                reencode(root/'plain', source, root/'plain.json', root/'again')
            (root/'other.json').write_text(json.dumps({**cfg, 'numeric_transform': 'append_log1p', 'seed': 7}))
            with self.assertRaisesRegex(ValueError, 'seed'):
                reencode(root/'plain', source, root/'other.json', root/'again')
            with self.assertRaises(FileExistsError):
                reencode(root/'plain', source, root/'log.json', root/'log')

    def test_reencode_refuses_cross_split_collisions(self):
        cfg = {**json.loads(Path('configs/data.json').read_text()), 'transform_fit': 'validation_only'}
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            root = Path(tmp)
            rows = []
            for group in range(100):
                for label in [0, 1]:
                    row = {col: float(group + label) for col in cfg['numeric_features']}
                    row.update(src_ip=f'10.0.0.{group}', dst_ip='10.1.0.1', src_port=group, dst_port=80, proto='tcp',
                               service='-', conn_state='SF', label=label, type='normal' if label == 0 else 'attack')
                    rows.append(row)
            source = root/'input.csv'
            pd.DataFrame(rows).to_csv(source, index=False)
            (root/'plain.json').write_text(json.dumps(cfg))
            (root/'log.json').write_text(json.dumps({**cfg, 'numeric_transform': 'append_log1p'}))
            base.prepare(source, root/'plain.json', root/'plain')  # equal vectors across splits: gate closed
            with self.assertRaisesRegex(ValueError, 'gate is closed|shared across splits'):
                reencode(root/'plain', source, root/'log.json', root/'log')
            self.assertFalse((root/'log').exists())


if __name__ == '__main__':
    unittest.main()
