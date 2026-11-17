"""Audit the new frozen experiment and evaluate its predeclared practical targets."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pandas as pd
from scripts.dt.analyze_online_confirmation import audit, markdown


def analyze(root):
    lock = json.loads((root/'protocol_lock.json').read_text())
    mismatches = [p for p, digest in lock['source_sha256'].items()
                  if hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != digest]
    if mismatches:
        raise ValueError(f'code/config changed after freeze: {mismatches}')
    frame, cfg = audit(root)
    pooled = frame.groupby(['environment', 'predictor']).agg(
        requests=('requests', 'sum'), joint_success=('joint_success_requests', 'sum'),
        deadline_misses=('deadline_misses', 'sum'), service_seconds=('service_seconds', 'sum'),
        latency_mean=('latency_mean_seconds', 'mean'), full_requests=('full_requests', 'sum'),
        validation_f1_mean=('validation_macro_f1', 'mean')).reset_index()
    pairs = []
    for block, group in frame.groupby('block'):
        dt = group[group.predictor == 'dt_queue'].iloc[0]
        for name in ('rows_legacy', 'rows_queue'):
            other = group[group.predictor == name].iloc[0]
            pairs.append(dict(block=block, environment=dt.environment, comparator=name,
                joint_delta=int(dt.joint_success_requests-other.joint_success_requests),
                deadline_misses_delta=int(dt.deadline_misses-other.deadline_misses),
                service_delta=float(dt.service_seconds-other.service_seconds),
                full_delta=int(dt.full_requests-other.full_requests),
                f1_delta=float(dt.validation_macro_f1-other.validation_macro_f1)))
    paired = pd.DataFrame(pairs)
    target = cfg['practical_targets']
    dt, old = frame[frame.predictor == 'dt_queue'], frame[frame.predictor == 'rows_legacy']
    p = paired[paired.comparator == 'rows_legacy']
    gain = 100*(dt.joint_success_requests.sum()/dt.requests.sum()-old.joint_success_requests.sum()/old.requests.sum())
    environments = int((p.groupby('environment').joint_delta.sum() > 0).sum())
    service_ratio = float(dt.service_seconds.sum()/old.service_seconds.sum())
    measured = dict(joint_gain_pp=float(gain), positive_environments=environments, service_ratio=service_ratio,
                    mean_f1_delta=float(p.f1_delta.mean()), worst_f1_delta=float(p.f1_delta.min()))
    gates = dict(joint_gain=gain >= target['joint_success_gain_pp_vs_rows_legacy'],
        environments=environments >= target['environments_with_positive_gain'],
        service_cost=service_ratio <= target['max_service_ratio_vs_rows_legacy'],
        mean_utility=measured['mean_f1_delta'] >= target['min_mean_paired_validation_f1_delta'],
        worst_utility=measured['worst_f1_delta'] >= target['min_block_validation_f1_delta'],
        exactness=bool(frame.bit_exact.all() and frame.exact_all_events.all()))
    wins = paired.groupby('comparator').agg(wins=('joint_delta', lambda x: int((x > 0).sum())),
        ties=('joint_delta', lambda x: int((x == 0).sum())), losses=('joint_delta', lambda x: int((x < 0).sum())),
        joint_delta=('joint_delta', 'sum'), service_delta=('service_delta', 'sum'),
        f1_delta_mean=('f1_delta', 'mean')).reset_index()
    frame.to_csv(root/'summary.csv', index=False)
    pooled.to_csv(root/'pooled.csv', index=False)
    paired.to_csv(root/'paired.csv', index=False)
    wins.to_csv(root/'paired_summary.csv', index=False)
    result = dict(measured=measured, targets=target, gates={k: bool(v) for k, v in gates.items()},
                  all_practical_targets_met=all(gates.values()), statistical_significance_tested=False)
    (root/'targets.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    text = ['# DT dự báo service và lập kế hoạch hàng đợi: kết quả', '',
        'Cấu hình đã chốt: [protocol](../../docs/dt_predictive_plan.md), protocol_lock.json. '
        'Số liệu được audit từ timestamp và event, cùng phép chấm của bộ trước.', '',
        '## Kết quả', '', markdown(pooled), '',
        '`rows_legacy`: đối chứng mạnh nhất của bộ trước; `rows_queue`: cùng controller mới nhưng chỉ dự báo theo dòng; '
        '`dt_queue`: controller mới và twin kết hợp bốn bộ dự báo bằng sai số đã quan sát.', '',
        '## So sánh ghép cặp DT − đối chứng', '', markdown(wins), '',
        '## Tiêu chí thực dụng chốt trước', '', '```json', json.dumps(result, indent=2), '```', '',
        'Đây là các ngưỡng thực dụng trên mẫu nhỏ, không phải kiểm định ý nghĩa thống kê hay chứng minh noninferiority. '
        'Nếu DT hơn rows_legacy nhưng không hơn rows_queue thì cải thiện thuộc chủ yếu về lập lịch/đo service; '
        'không được quy toàn bộ cho mô hình twin chi tiết.', '',
        'Phạm vi: local CPU, một worker, 4 shard, full/fixed, chỉ yêu cầu đã đến; không đo HTTP, checkpoint bền vững '
        'hay distributed serving. Không dùng test/calibration hay seed v3. Không bảo đảm hard deadline.']
    (root/'report.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    print(markdown(pooled))
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    analyze(parser.parse_args().root)
