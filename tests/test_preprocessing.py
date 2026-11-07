import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data.prepare import fit_transformer, overlaps, prepare, split_groups, transform


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

    @staticmethod
    def _rows(cfg, train_scale=1.):
        rows = []
        for group in range(100):
            for label in [0, 1]:
                row = {col: (group + label)*train_scale for col in cfg["numeric_features"]}
                row.update(src_ip=f"10.0.0.{group}", dst_ip="10.1.0.1", src_port=group, dst_port=80, proto="tcp",
                           service="dns" if group < 3 else "-", conn_state="SF", label=label,
                           type="normal" if label == 0 else "attack")
                rows.append(row)
        return rows

    def _run(self, root, name, cfg, rows):
        source, config = root / f"{name}.csv", root / f"{name}.json"
        pd.DataFrame(rows).to_csv(source, index=False); config.write_text(json.dumps(cfg))
        prepare(source, config, root / name)
        return root / name

    def test_validation_only_transform_uses_only_validation_rows(self):
        cfg = {**json.loads(Path("configs/data.json").read_text()), "transform_fit": "validation_only"}
        with tempfile.TemporaryDirectory(dir=".") as tmp:
            out = self._run(Path(tmp), "v", cfg, self._rows(cfg))
            meta = pd.read_csv(out / "metadata.csv"); raw = pd.read_csv(Path(tmp) / "v.csv")
            val = meta[meta.split == "validation"]
            expected = fit_transformer(raw.loc[[int(r.split(":")[1]) for r in val.row_id]], cfg["numeric_features"], cfg["categorical_features"])
            self.assertEqual(json.loads((out / "transformer.json").read_text()), json.loads(json.dumps(expected)))
            manifest = json.loads((out / "manifest.json").read_text()); audit = json.loads((out / "audit.json").read_text())
            self.assertEqual(manifest["transform_fit_rows_sha256"], hashlib.sha256("\n".join(val.row_id).encode()).hexdigest())
            self.assertEqual(audit["transform_fit_split"], "validation")
            self.assertEqual(set(audit["unknown_category_rate"]), set(cfg["split_fractions"]))

    def test_validation_only_transform_ignores_training_rows(self):
        """Changing only training rows must leave the encoding, validation and test unchanged (train_only does not)."""
        base = json.loads(Path("configs/data.json").read_text())
        for fit, invariant in [("validation_only", True), ("train_only", False)]:
            cfg = {**base, "transform_fit": fit}
            with tempfile.TemporaryDirectory(dir=".") as tmp:
                root = Path(tmp); a = self._run(root, "a", cfg, self._rows(cfg))
                split = pd.read_csv(a / "metadata.csv").split.to_numpy()
                rows = self._rows(cfg); changed = self._rows(cfg, train_scale=3.)
                rows = [c if s == "train" else r for r, c, s in zip(rows, changed, split)]
                b = self._run(root, "b", cfg, rows)
                np.testing.assert_array_equal(pd.read_csv(b / "metadata.csv").split.to_numpy(), split)
                same = (a / "transformer.json").read_text() == (b / "transformer.json").read_text()
                self.assertEqual(same, invariant, fit)
                if invariant:
                    for name in ["validation", "calibration", "test"]:
                        with np.load(a / f"{name}.npz") as x, np.load(b / f"{name}.npz") as y:
                            np.testing.assert_array_equal(x["X"], y["X"])
                    with np.load(a / "train.npz") as x, np.load(b / "train.npz") as y:
                        self.assertFalse(np.array_equal(x["X"], y["X"]))

    def test_unknown_transform_fit_fails(self):
        cfg = {**json.loads(Path("configs/data.json").read_text()), "transform_fit": "all_rows"}
        with tempfile.TemporaryDirectory(dir=".") as tmp:
            path = Path(tmp) / "config.json"; path.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):
                prepare("nonexistent.csv", path, Path(tmp) / "new")

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
