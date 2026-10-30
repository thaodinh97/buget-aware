"""Offline research preprocessing; no training or raw-data server transport."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

GROUP_COLUMNS = ["src_ip", "src_port", "dst_ip", "dst_port", "proto"]
FORBIDDEN = {"label", "type", "src_ip", "dst_ip", "src_port", "dst_port"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def signatures(frame):
    # JSON preserves column boundaries; full SHA-256 avoids ambiguous concatenation.
    return np.array([
        hashlib.sha256(json.dumps(list(row), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        for row in frame.astype(str).itertuples(index=False, name=None)
    ])


def split_groups(groups, fractions, seed):
    if list(fractions) != ["train", "validation", "calibration", "test"]:
        raise ValueError("Expected ordered train/validation/calibration/test fractions")
    if any(v <= 0 for v in fractions.values()) or not np.isclose(sum(fractions.values()), 1):
        raise ValueError("Positive split fractions must sum to one")
    unique = np.unique(groups)
    np.random.default_rng(seed).shuffle(unique)
    boundaries = (np.cumsum(list(fractions.values()))[:-1] * len(unique)).astype(int)
    assignment = {}
    for name, values in zip(fractions, np.split(unique, boundaries)):
        if len(values) == 0:
            raise ValueError(f"Empty split: {name}")
        assignment.update(dict.fromkeys(values, name))
    return np.array([assignment[g] for g in groups])


def connected_groups(*keys):
    """Union rows sharing any key; transitive closure preserves all constraints."""
    parent = np.arange(len(keys[0]))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for values in keys:
        seen = {}
        for i, key in enumerate(values):
            if key in seen:
                a, b = root(i), root(seen[key])
                parent[max(a, b)] = min(a, b)
            else:
                seen[key] = i
    return np.array([str(root(i)) for i in range(len(parent))])


def near_duplicate_keys(frame, numeric, categorical, jitter, delta):
    """Rows equal in every predictor but `jitter` whose log1p(jitter) values chain with gaps <= delta share a key.

    ToN-IoT near duplicates differ only in duration, by microseconds, so a single jitter dimension
    lets the transitive closure be computed exactly by sorting. Missing jitter values never chain.
    """
    others = signatures(frame[[c for c in numeric + categorical if c != jitter]])
    values = np.log1p(numeric_frame(frame, [jitter])[jitter].clip(lower=0).to_numpy(dtype=float))
    missing = np.isnan(values)
    order = np.lexsort((np.where(missing, 0, values), missing, others))
    s_others, s_values, s_missing = others[order], values[order], missing[order]
    start = np.ones(len(order), dtype=bool)
    start[1:] = (s_others[1:] != s_others[:-1]) | s_missing[1:] | s_missing[:-1] | (np.diff(s_values) > delta)
    keys = np.empty(len(order), dtype=object)
    keys[order] = np.char.add("near:", np.cumsum(start).astype(str))
    return keys.astype(str)


def balanced_group_split(groups, fractions, seed):
    unique, counts = np.unique(groups, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(unique))
    order = sorted(order, key=lambda i: -counts[i])
    names = list(fractions)
    remaining = np.array(list(fractions.values())) * len(groups)
    mapping = {}
    for i in order:
        chosen = int(np.argmax(remaining))
        mapping[unique[i]] = names[chosen]
        remaining[chosen] -= counts[i]
    return np.array([mapping[g] for g in groups])


def numeric_frame(frame, columns, log1p=()):
    values = frame[columns].apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    if log1p:
        # Heavy-tailed counts: z-scoring raw bytes (sd ~1e7) makes 40 vs 44 bytes indistinguishable.
        log1p = list(log1p)
        if (values[log1p] < 0).any().any():
            raise ValueError("log1p features must be non-negative")
        values[log1p] = np.log1p(values[log1p])
    return values


def categorical_values(series):
    # '-' means absent/not-applicable here, not numeric zero.
    return series.astype("string").fillna("__MISSING__").replace("-", "__MISSING__")


def fit_transformer(train, numeric, categorical, log1p=()):
    values = numeric_frame(train, numeric, log1p)
    median = values.median()
    if median.isna().any():
        raise ValueError("Training numeric feature has no finite observations")
    values = values.fillna(median)
    scale = values.std(ddof=0).replace(0, 1)
    vocabulary = {}
    for col in categorical:
        observed = train[col].dropna().astype(str)
        if observed.isin(["__MISSING__", "__UNKNOWN__"]).any():
            raise ValueError("Input collides with reserved category tokens")
        vocabulary[col] = ["__MISSING__", "__UNKNOWN__"] + sorted(
            set(categorical_values(train[col])) - {"__MISSING__"})
    return {"numeric": numeric, "categorical": categorical, "log1p": list(log1p),
            "median": median.to_dict(), "mean": values.mean().to_dict(),
            "scale": scale.to_dict(), "vocabulary": vocabulary,
            "feature_names": numeric + [f"{c}={v}" for c in categorical for v in vocabulary[c]]}


def transform(frame, state):
    values = numeric_frame(frame, state["numeric"], state.get("log1p", [])).fillna(state["median"])
    arrays = [((values - pd.Series(state["mean"])) / pd.Series(state["scale"])).to_numpy(dtype=np.float32)]
    for col in state["categorical"]:
        vocab = state["vocabulary"][col]
        lookup = {value: i for i, value in enumerate(vocab)}
        codes = categorical_values(frame[col]).map(lookup).fillna(lookup["__UNKNOWN__"]).to_numpy(dtype=int)
        encoded = np.zeros((len(frame), len(vocab)), dtype=np.float32)
        encoded[np.arange(len(frame)), codes] = 1
        arrays.append(encoded)
    output = np.concatenate(arrays, axis=1)
    if not np.isfinite(output).all():
        raise ValueError("Non-finite transformed values")
    return output


def overlaps(values, splits):
    names = ["train", "validation", "calibration", "test"]
    sets = {name: set(values[splits == name]) for name in names}
    return {f"{a}__{b}": len(sets[a] & sets[b])
            for i, a in enumerate(names) for b in names[i + 1:]}


def prepare(input_path, config_path, output_dir):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output_dir}")
    expected = {"duplicate_policy": "keep_first_exact_row_with_mapping", "conflict_policy": "retain_and_report",
                "transform_fit": "train_only"}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Unsupported policy: implementation must match manifest")
    connected = ['connected_bidirectional_features', 'connected_bidirectional_features_near_duplicates']
    if config['group_policy'] not in ['directed_five_tuple', *connected]:
        raise ValueError('Unsupported group policy')
    log1p = config.get('log1p_features', [])
    if not set(log1p) <= set(config['numeric_features']) or len(log1p) != len(set(log1p)):
        raise ValueError('log1p_features must be distinct numeric predictors')
    near = config.get('near_duplicate')
    if (config['group_policy'] == connected[1]) != (near is not None):
        raise ValueError('near_duplicate settings must be given exactly with the near-duplicate group policy')
    if near and (near['jitter_feature'] not in config['numeric_features'] or near['log1p_delta'] <= 0):
        raise ValueError('Near-duplicate jitter must be a numeric predictor with a positive tolerance')
    numeric, categorical = config["numeric_features"], config["categorical_features"]
    features = numeric + categorical
    if len(features) != len(set(features)) or set(features) & FORBIDDEN:
        raise ValueError("Duplicate or forbidden predictors")
    raw = pd.read_csv(input_path, low_memory=False)
    required = set(features + GROUP_COLUMNS + ["label", "type"])
    if required - set(raw):
        raise ValueError(f"Missing columns: {required - set(raw)}")
    if raw[GROUP_COLUMNS].isna().any().any() or set(raw.label.unique()) != {0, 1}:
        raise ValueError("Invalid grouping fields or binary labels")
    source_hash = digest(input_path)
    # Exact duplicate equality uses every original column, including labels.
    keys = pd.MultiIndex.from_frame(raw)
    codes, _ = pd.factorize(keys, sort=False)
    first = pd.Series(np.arange(len(raw))).groupby(codes).min()
    representative = first.loc[codes].to_numpy()
    keep = representative == np.arange(len(raw))
    frame = raw.loc[keep].copy()
    row_ids = np.array([f"{source_hash}:{i}" for i in frame.index])
    groups = signatures(frame[GROUP_COLUMNS])
    feature_signature = signatures(frame[features])
    reverse = frame[GROUP_COLUMNS].copy()
    reverse.columns = ["dst_ip", "dst_port", "src_ip", "src_port", "proto"]
    bidirectional = np.array([min(a, b) for a, b in zip(groups, signatures(reverse[GROUP_COLUMNS]))])
    near_keys = None
    if near:
        near_keys = near_duplicate_keys(frame, numeric, categorical, near['jitter_feature'], near['log1p_delta'])
    if config['group_policy'] in connected:
        groups = connected_groups(bidirectional, feature_signature, *([near_keys] if near else []))
        splits = balanced_group_split(groups, config['split_fractions'], config['seed'])
    else:
        splits = split_groups(groups, config["split_fractions"], config["seed"])
    refinement_rounds = 0
    for refinement_rounds in range(10):
        for name in config["split_fractions"]:
            if set(frame.loc[splits == name, "label"].unique()) != {0, 1}:
                raise ValueError(f"Single-class/empty split: {name}; no seed fallback")
        state = fit_transformer(frame.loc[splits == "train"], numeric, categorical, log1p)
        encoded = transform(frame, state)
        encoded_signatures = np.array([hashlib.sha256(row.tobytes()).hexdigest() for row in encoded])
        encoded_overlap = overlaps(encoded_signatures, splits)
        if not any(encoded_overlap.values()) or config['group_policy'] == 'directed_five_tuple':
            break
        groups = connected_groups(groups, encoded_signatures)
        splits = balanced_group_split(groups, config['split_fractions'], config['seed'])
    else:
        raise ValueError('Encoded overlap refinement did not converge; no training artifacts emitted')
    feature_signature = signatures(frame[features])
    conflicts = pd.DataFrame({"signature": feature_signature, "label": frame.label.to_numpy()}).groupby("signature").label.nunique()
    reverse = frame[GROUP_COLUMNS].copy()
    reverse.columns = ["dst_ip", "dst_port", "src_ip", "src_port", "proto"]
    reversed_groups = signatures(reverse[GROUP_COLUMNS])
    bidirectional = np.array([min(a, b) for a, b in zip(groups, reversed_groups)])
    signature_overlap = overlaps(feature_signature, splits)
    group_overlap = overlaps(groups, splits)
    reverse_overlap = overlaps(bidirectional, splits)
    if any(group_overlap.values()):
        raise AssertionError("Cross-split group leakage")
    near_audit = None
    if near:
        # Rows sharing a key under tolerance d but sitting in different splits; zero at the configured d
        # by construction, other tolerances show sensitivity.
        near_audit = {"jitter_feature": near['jitter_feature'], "log1p_delta": near['log1p_delta'],
                      "rows_in_near_duplicate_sets": int(pd.Series(near_keys).duplicated(keep=False).sum()),
                      "cross_split_overlap": overlaps(near_keys, splits),
                      "sensitivity_cross_split_overlap": {
                          str(d): overlaps(near_duplicate_keys(frame, numeric, categorical, near['jitter_feature'], d), splits)
                          for d in [1e-3, 1e-2, 5e-2]}}
    warnings = []
    if near_audit and any(near_audit["cross_split_overlap"].values()):
        warnings.append('NEAR_DUPLICATE_OVERLAP_REQUIRES_PROTOCOL_REVIEW')
    if any(encoded_overlap.values()):
        warnings.append('ENCODED_FEATURE_OVERLAP_REQUIRES_PROTOCOL_REVIEW')
    if any(signature_overlap.values()):
        warnings.append("FEATURE_SIGNATURE_OVERLAP_REQUIRES_PROTOCOL_REVIEW")
    if any(reverse_overlap.values()):
        warnings.append("REVERSE_CONNECTION_OVERLAP_REQUIRES_PROTOCOL_REVIEW")
    if (conflicts > 1).any():
        warnings.append("CONFLICTING_FEATURE_LABELS_RETAINED")
    # Only create artifacts after validation. Existing output folders always fail.
    output_dir.mkdir(parents=True)
    counts = {}
    for name in config["split_fractions"]:
        mask = splits == name
        part = frame.loc[mask]
        x = transform(part, state)
        np.savez_compressed(output_dir / f"{name}.npz", X=x, y=part.label.to_numpy(dtype=np.int64),
                            row_id=row_ids[mask], group_id=groups[mask])
        counts[name] = {"rows": len(part), "groups": len(set(groups[mask])),
                        "labels": {str(k): int(v) for k, v in part.label.value_counts().items()},
                        "types": part.type.value_counts().to_dict(), "feature_dimension": x.shape[1]}
    pd.DataFrame({"row_id": row_ids, "group_id": groups, "feature_signature": feature_signature,
                  "split": splits, "type": frame.type.to_numpy()}).to_csv(output_dir / "metadata.csv", index=False)
    pd.DataFrame({"source_row_index": np.arange(len(raw)), "representative_row_index": representative}).to_csv(
        output_dir / "duplicate_mapping.csv", index=False)
    write_json(output_dir / "transformer.json", state)
    audit = {"raw_rows": len(raw), "raw_columns": len(raw.columns), "retained_rows": len(frame),
             "exact_duplicates_removed": int((~keep).sum()), "null_counts": raw.isna().sum().to_dict(),
             "dash_counts": raw.eq("-").sum().to_dict(), "splits": counts,
             "feature_conflict_groups": int((conflicts > 1).sum()),
             "feature_duplicate_extra_rows": len(frame) - len(set(feature_signature)),
             "cross_split_group_overlap": group_overlap, "cross_split_signature_overlap": signature_overlap,
             "cross_split_bidirectional_overlap": reverse_overlap,
             "largest_group_rows": int(pd.Series(groups).value_counts().max()),
             "encoded_overlap_refinement_rounds": refinement_rounds,
             "cross_split_encoded_overlap": encoded_overlap,
             "near_duplicate_audit": near_audit,
             "warnings": warnings, "training_ready": not any('OVERLAP' in w for w in warnings)}
    write_json(output_dir / "audit.json", audit)
    manifest = {"config": config, "dataset_path": str(Path(input_path).resolve()),
                "dataset_sha256": source_hash, "source_sha256": digest(__file__),
                "config_sha256": digest(config_path), "schema": raw.dtypes.astype(str).to_dict(),
                "python": platform.python_version(), "platform": platform.platform(),
                "numpy": np.__version__, "pandas": pd.__version__, "code_revision": None,
                "transform_fit_rows_sha256": hashlib.sha256("\n".join(row_ids[splits == "train"]).encode()).hexdigest(),
                "artifacts_sha256": {p.name: digest(p) for p in sorted(output_dir.iterdir())}}
    write_json(output_dir / "manifest.json", manifest)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="dataset/train_test_network.csv")
    parser.add_argument("--config", default="configs/data.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.config, args.output), indent=2))
