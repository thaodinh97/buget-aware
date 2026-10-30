import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data.prepare import fit_transformer, near_duplicate_keys, overlaps, prepare, split_groups, transform


class PreprocessingTests(unittest.TestCase):
    def test_group_split_is_deterministic_and_disjoint(self):
        groups = np.repeat(np.arange(100), 3)
        fractions = dict(train=.6, validation=.15, calibration=.1, test=.15)
        a = split_groups(groups, fractions, 42)
        np.testing.assert_array_equal(a, split_groups(groups, fractions, 42))
        self.assertFalse(any(overlaps(groups, a).values()))
        self.assertEqual((a == "train").sum(), 180)

    def test_train_statistics_and_unknown_category(self):
        train = pd.DataFrame({"x": [0., 2., np.nan], "proto": ["tcp", "udp", "-"]})
        state = fit_transformer(train, ["x"], ["proto"])
        self.assertEqual(state["median"]["x"], 1.)
        self.assertEqual(state["mean"]["x"], 1.)
        before = json.dumps(state)
        test = transform(pd.DataFrame({"x": [1000., np.nan], "proto": ["new", "-"]}), state)
        self.assertEqual(test.shape, (2, 5))
        self.assertEqual(test[0, 2], 1.)  # unknown, not an all-zero one-hot vector
        self.assertEqual(test[1, 1], 1.)  # missing
        self.assertEqual(test[1, 0], 0.)
        self.assertEqual(before, json.dumps(state))

    def test_log1p_counts_fit_on_train_and_keep_legacy_state(self):
        train = pd.DataFrame({"bytes": [0., np.e - 1, np.nan], "duration": [1., 3., 5.], "proto": ["tcp"] * 3})
        state = fit_transformer(train, ["bytes", "duration"], ["proto"], ["bytes"])
        self.assertEqual(state["log1p"], ["bytes"])
        self.assertAlmostEqual(state["median"]["bytes"], .5)  # median of log1p values 0 and 1
        self.assertEqual(state["mean"]["duration"], 3.)       # untouched feature stays linear
        out = transform(pd.DataFrame({"bytes": [40., 44.], "duration": [3., 3.], "proto": ["tcp", "tcp"]}), state)
        self.assertGreater(out[1, 0] - out[0, 0], .1)        # 40 vs 44 bytes stay distinguishable
        legacy = {k: v for k, v in state.items() if k != "log1p"}
        self.assertEqual(transform(train, legacy).shape, (3, 5))
        with self.assertRaises(ValueError):
            fit_transformer(pd.DataFrame({"bytes": [-1., 2.], "proto": ["tcp"] * 2}), ["bytes"], ["proto"], ["bytes"])

    def test_invalid_split_fails(self):
        with self.assertRaises(ValueError):
            split_groups(np.arange(2), dict(train=.6, validation=.15, calibration=.1, test=.15), 42)

    def test_end_to_end_smoke_and_repeat(self):
        cfg = json.loads(Path("configs/data.json").read_text())
        rows = []
        for group in range(100):
            for label in [0, 1]:
                row = {col: group + label for col in cfg["numeric_features"]}
                row.update(src_ip=f"10.0.0.{group}", dst_ip="10.1.0.1", src_port=group,
                           dst_port=80, proto="tcp", service="-", conn_state="SF",
                           label=label, type="normal" if label == 0 else "attack")
                rows.append(row)
        rows.append(rows[0].copy())
        with tempfile.TemporaryDirectory(dir=".") as tmp:
            root = Path(tmp)
            source = root / "input.csv"
            pd.DataFrame(rows).to_csv(source, index=False)
            before = source.read_bytes()
            first = prepare(source, "configs/data.json", root / "a")
            second = prepare(source, "configs/data.json", root / "b")
            self.assertEqual(first, second)
            self.assertEqual(first["exact_duplicates_removed"], 1)
            self.assertEqual(first["retained_rows"], 200)
            self.assertEqual(source.read_bytes(), before)
            mapping = pd.read_csv(root / "a/duplicate_mapping.csv")
            self.assertEqual(mapping.iloc[-1].representative_row_index, 0)
            seen_ids = set()
            for split in cfg["split_fractions"]:
                with np.load(root / f"a/{split}.npz", allow_pickle=False) as a, np.load(root / f"b/{split}.npz", allow_pickle=False) as b:
                    for key in a.files:
                        np.testing.assert_array_equal(a[key], b[key])
                    self.assertFalse(seen_ids & set(a["row_id"]))
                    seen_ids.update(a["row_id"])
                    self.assertTrue(np.isfinite(a["X"]).all())
            self.assertEqual(len(seen_ids), 200)

    def test_near_duplicate_keys_chain_only_jitter_feature(self):
        frame = pd.DataFrame({"duration": [0.0, 0.005, 0.012, 0.5, 0.0, np.nan, np.nan],
                              "src_bytes": [10, 10, 10, 10, 11, 10, 10], "proto": ["tcp"] * 7})
        keys = near_duplicate_keys(frame, ["duration", "src_bytes"], ["proto"], "duration", .01)
        self.assertEqual(len(set(keys[:3])), 1)          # log1p gaps 0.0050 and 0.0069 chain
        self.assertEqual(len(set(keys)), 5)
        self.assertNotIn(keys[3], keys[:3])             # gap 0.39 breaks the chain
        self.assertNotIn(keys[4], keys[:4])             # another predictor differs
        self.assertNotEqual(keys[5], keys[6])           # missing jitter never chains

    def test_near_duplicate_policy_keeps_jitter_copies_in_one_split(self):
        from scripts.make_synthetic_toniot import synthetic
        cfg = json.loads(Path("configs/data_near_duplicates.json").read_text())
        raw = synthetic(rows_per_family=150, normal_rows=900, rare_rows=60)
        copies = raw.sample(200, random_state=0).copy()
        copies["duration"] += 1e-6
        copies["src_port"] = 1  # new connection, so only the near-duplicate rule can link them
        raw = pd.concat([raw, copies], ignore_index=True)
        with tempfile.TemporaryDirectory(dir=".") as tmp:
            root = Path(tmp)
            raw.to_csv(root / "raw.csv", index=False)
            audit = prepare(root / "raw.csv", "configs/data_near_duplicates.json", root / "near")
            self.assertTrue(audit["training_ready"])
            self.assertFalse(any(audit["near_duplicate_audit"]["cross_split_overlap"].values()))
            self.assertGreaterEqual(audit["near_duplicate_audit"]["rows_in_near_duplicate_sets"], 400)
            meta = pd.read_csv(root / "near/metadata.csv")
            split = dict(zip(meta.row_id.str.split(":").str[1].astype(int), meta.split))
            copied = np.arange(len(raw) - len(copies), len(raw))  # copies were appended after the originals
            self.assertTrue(all(split[a] == split[b] for a, b in zip(copies.index, copied)))
            cfg.pop("near_duplicate")
            (root / "bad.json").write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):
                prepare(root / "raw.csv", root / "bad.json", root / "bad")

    def test_forbidden_label_feature_and_overwrite_fail(self):
        cfg = json.loads(Path("configs/data.json").read_text())
        with tempfile.TemporaryDirectory(dir=".") as tmp:
            path = Path(tmp) / "config.json"
            cfg["numeric_features"].append("label")
            path.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):
                prepare("nonexistent.csv", path, Path(tmp) / "new")
            with self.assertRaises(FileExistsError):
                prepare("nonexistent.csv", path, tmp)


if __name__ == "__main__":
    unittest.main()
