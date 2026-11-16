"""V3 startup refuses changed inputs; CLI retries in a new process at most once."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from scripts.shard import shard_v3 as protocol
from scripts.shard import run_shard_stream_v3 as runner
from src.fu.shard_exact import ShardConfig


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        manifest = {'artifacts_sha256': {}}
        for i, name in enumerate(('train', 'validation', 'calibration')):
            np.savez(self.root/f'{name}.npz', X=np.zeros((6, 2), np.float32), y=np.arange(6) % 2,
                     row_id=np.arange(i*10, i*10 + 6), group_id=np.arange(i*10, i*10 + 6))
            manifest['artifacts_sha256'][f'{name}.npz'] = hashlib.sha256((self.root/f'{name}.npz').read_bytes()).hexdigest()
        protocol.write_json(self.root/'manifest.json', manifest)
        protocol.write_json(self.root/'audit.json', {'training_ready': True, 'cross_split_encoded_overlap': {'train_test': 0}})

    def test_only_allowed_splits_are_loaded_without_a_test_file(self):
        data, _ = protocol.load_protocol_data(self.root)
        self.assertEqual(set(data), {'train', 'validation', 'calibration'})
        self.assertFalse((self.root/'test.npz').exists())

    def test_changed_npz_rejected_before_loading(self):
        with (self.root/'train.npz').open('ab') as handle:
            handle.write(b'tampered')
        with self.assertRaisesRegex(ValueError, 'artifact hash mismatch'):
            protocol.load_protocol_data(self.root)

    def test_closed_preprocessing_gate_is_rejected(self):
        protocol.write_json(self.root/'audit.json', {'training_ready': False})
        with self.assertRaisesRegex(ValueError, 'gate is closed'):
            protocol.load_protocol_data(self.root)

    def test_missing_tracked_source_is_rejected(self):
        with mock.patch.object(protocol, 'ROOT', self.root), mock.patch.object(protocol, 'TRACKED_V3', ['absent.py']):
            with self.assertRaisesRegex(FileNotFoundError, 'absent.py'):
                protocol.source_hashes()

    def test_tampered_frozen_metadata_is_rejected(self):
        meta = self.root/protocol.METADATA_DEFAULT
        meta.parent.mkdir(parents=True)
        protocol.write_json(meta, {'partition_seed': 42, 'client_rows': [1]*20})
        with mock.patch.dict('os.environ', {'SHARD_DESIGN': 'v3', 'SHARD_V3_FAMILY': 'spread'}, clear=True), \
                mock.patch.object(protocol, 'ROOT', self.root), mock.patch.object(protocol, 'source_hashes', return_value={}), \
                mock.patch.object(protocol, 'norm_rel', return_value=protocol.RUNNER), \
                mock.patch.object(protocol, 'load_protocol_data', return_value=({}, {})), \
                mock.patch.object(protocol, 'data_content_sha256', return_value=protocol.DATA_SHA256):
            with self.assertRaisesRegex(ValueError, 'frozen v3 client counts'):
                protocol.start_run(self.root)

    def test_runtime_cannot_reuse_standard_record_with_smaller_model(self):
        parts = [np.arange(n) for n in protocol.EXPECTED_CLIENT_ROWS]
        record = {'design_name': 'v3', 'design': copy.deepcopy(protocol.DESIGN), 'family': 'spread',
                  'runner': protocol.RUNNER, 'train_config': protocol.CONFIGS['standard'], 'train_config_name': 'standard',
                  'stream_config': protocol.STREAM, 'data_content_sha256': protocol.DATA_SHA256,
                  'client_rows': protocol.EXPECTED_CLIENT_ROWS, 'sha256_at_start': {}, 'machine': protocol.machine_info()}
        config = ShardConfig.from_train_config(protocol.CONFIGS['standard'], hidden=(8, 4))
        with mock.patch.object(runner, 'source_hashes', return_value={}), \
                mock.patch.object(runner, 'data_content_sha256', return_value=protocol.DATA_SHA256), \
                mock.patch.object(runner, 'partition', return_value=parts):
            with self.assertRaisesRegex(ValueError, 'component_config'):
                runner.validate_run_inputs(record, 'spread', {'train': {'y': np.array([0, 1])}}, parts, config,
                    protocol.DESIGN['seeds'], protocol.DESIGN['shards'], protocol.STREAM, protocol.DESIGN['reference_steps'])


SMOKE_ENV = {'SHARD_DESIGN': 'smoke_v3', 'SHARD_V3_FAMILY': 'spread'}


class SelectionTests(unittest.TestCase):
    """A forgotten variable or a mislabelled directory stops the CLI before it creates or starts anything."""

    def refused(self, out, pattern, argv=None):
        with mock.patch.object(runner.subprocess, 'run') as call, mock.patch.object(runner, 'start_run') as start:
            with self.assertRaisesRegex(ValueError, pattern):
                runner.main([str(out)] if argv is None else argv)
        self.assertFalse(out.exists())
        call.assert_not_called()
        start.assert_not_called()

    def test_design_and_family_have_no_default(self):
        for missing in SMOKE_ENV:
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as temp, mock.patch.dict('os.environ', SMOKE_ENV):
                del protocol.os.environ[missing]
                with self.assertRaisesRegex(ValueError, missing):
                    protocol.run_selection()
                self.refused(Path(temp)/'stream_spread', missing)

    def test_unknown_design_or_family_is_rejected(self):
        for key, value in (('SHARD_DESIGN', 'v2'), ('SHARD_V3_FAMILY', 'uniform')):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp, \
                    mock.patch.dict('os.environ', {**SMOKE_ENV, key: value}):
                self.refused(Path(temp)/'stream_spread', key)

    def test_output_directory_must_be_named_after_the_family(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict('os.environ', SMOKE_ENV):
            for name in ('stream_proportional', 'spread', 'family'):
                with self.subTest(name=name):
                    self.refused(Path(temp)/name, 'stream_spread')
            out = Path(temp)/'stream_proportional'/'attempt_1'
            self.refused(out, 'stream_spread', [str(out), '--attempt', '1'])
            self.assertFalse(out.parent.exists())

    def test_child_accepts_the_attempt_directory_the_parent_passes(self):
        threads = runner.torch.get_num_threads()
        deterministic = runner.torch.are_deterministic_algorithms_enabled()
        self.addCleanup(runner.torch.set_num_threads, threads)
        self.addCleanup(runner.torch.use_deterministic_algorithms, deterministic)
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict('os.environ', SMOKE_ENV):
            commands = []

            def execute(command, **kwargs):
                commands.append(command)
                return SimpleNamespace(returncode=1)
            with mock.patch.object(runner.subprocess, 'run', side_effect=execute), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'attempt 1 failed'):
                    runner.main([str(Path(temp)/'stream_spread')])
            with mock.patch.object(runner, 'start_run', side_effect=RuntimeError('reached start_run')):
                with self.assertRaisesRegex(RuntimeError, 'reached start_run'):
                    runner.main(commands[0][2:])


class RetryTests(unittest.TestCase):
    def exercise(self, reproducible):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'stream_spread'
            def execute(command, **kwargs):
                attempt_dir = Path(command[2])
                attempt = int(command[-1])
                attempt_dir.mkdir()
                protocol.write_json(attempt_dir/'run.json', {'family': 'spread', 'attempt': attempt,
                                    'completed': True, 'reproducible': reproducible[attempt - 1]})
                for name in ('steps.csv', 'exactness.csv', 'stream.json'):
                    (attempt_dir/name).write_text(f'attempt {attempt}', encoding='utf-8')
                return SimpleNamespace(returncode=0)
            with mock.patch.dict('os.environ', SMOKE_ENV), \
                    mock.patch.object(runner.subprocess, 'run', side_effect=execute) as call, contextlib.redirect_stdout(io.StringIO()):
                runner.main([str(out)])
            result = json.loads((out/'run.json').read_text(encoding='utf-8'))
            self.assertEqual(call.call_count, len(result['attempts']))
            self.assertEqual(result['selected_attempt'], call.call_count)
            self.assertEqual((out/'steps.csv').read_text(encoding='utf-8'), f'attempt {call.call_count}')
            self.assertEqual([a['reproducible'] for a in result['attempts']], reproducible[:call.call_count])
            self.assertEqual(len(list(out.glob('attempt_*'))), call.call_count)
            for call_args in call.call_args_list:
                self.assertEqual(call_args.args[0][0], runner.sys.executable)
                self.assertEqual(call_args.args[0][-2], '--attempt')
            return result

    def test_success_runs_one_process(self):
        result = self.exercise([True])
        self.assertTrue(result['reproducible'])
        self.assertEqual(result['selected_attempt'], 1)

    def test_nonreproducible_fit_retries_entire_family_once(self):
        result = self.exercise([False, True])
        self.assertTrue(result['reproducible'])
        self.assertEqual(result['selected_attempt'], 2)

    def test_two_failed_reproducibility_checks_are_preserved(self):
        result = self.exercise([False, False])
        self.assertFalse(result['reproducible'])
        self.assertEqual(result['selected_attempt'], 2)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict('os.environ', SMOKE_ENV):
            out = Path(temp)/'stream_spread'
            out.mkdir()
            with self.assertRaises(FileExistsError):
                runner.main([str(out)])
            self.assertEqual(list(out.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
