"""Audit and aggregate the completed online confirmation; no retraining."""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.dt.realtime import summarize


def audit(root):
    lock = json.loads((root/'protocol_lock.json').read_text())
    cfg = lock['config']
    expected = len(cfg['seeds']) * len(cfg['families']) * len(cfg['environments'])
    blocks = sorted(p for p in root.iterdir() if p.is_dir() and (p/'run.json').exists())
    if len(blocks) != expected:
        raise ValueError(f'incomplete experiment: {len(blocks)}/{expected} blocks')
    rows = []
    for block in blocks:
        run = json.loads((block/'run.json').read_text())
        complete = json.loads((block/'complete.json').read_text())
        requests = {r['step']: r for r in run['requests']}
        for recorded in complete['summaries']:
            name = recorded['predictor']
            events = [json.loads(s) for s in (block/name/'events.jsonl').read_text().splitlines()]
            steps = [s for e in events for s in e['steps']]
            if sorted(steps) != sorted(requests) or len(steps) != len(set(steps)):
                raise ValueError(f'missing/duplicated requests in {block}/{name}')
            previous_ready = 0.
            for event in events:
                if event['started_seconds'] < previous_ready:
                    raise ValueError('single worker overlapped operations')
                previous_ready = event['model_ready_seconds']
                for step, arrival, wait, latency in zip(event['steps'], event['arrivals'],
                                                       event['waiting_seconds'], event['latency_seconds']):
                    if arrival != requests[step]['time'] or arrival > event['started_seconds']:
                        raise ValueError('forecast/execution consumed a future request')
                    if not math.isclose(latency, wait + event['service_seconds'], abs_tol=1e-8):
                        raise ValueError('completion latency excludes service time')
                if event['deadline_misses'] != sum(t > run['deadline_seconds'] for t in event['latency_seconds']):
                    raise ValueError('incorrect deadline count')
            got = summarize(events)
            for key, value in got.items():
                if not math.isclose(value, recorded[key], rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(f'summary mismatch: {block}/{name}/{key}')
            final = json.loads((block/name/'final.json').read_text())
            if not final['summary']['bit_exact']:
                raise ValueError('bit-exact check failed')
            rows.append(recorded)
    frame = pd.DataFrame(rows)
    if len(frame) != expected * len(cfg['predictors']):
        raise ValueError('missing arms')
    return frame, cfg


def markdown(frame):
    def cell(v):
        return f'{v:.3f}' if isinstance(v, float) else str(v)
    return '\n'.join(['| ' + ' | '.join(frame.columns) + ' |', '|' + '---|'*len(frame.columns)] +
                     ['| ' + ' | '.join(cell(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args(argv)
    frame, cfg = audit(args.root)
    frame.to_csv(args.root/'summary.csv', index=False)
    pooled = frame.groupby(['environment', 'predictor'], sort=False).agg(
        blocks=('block', 'size'), requests=('requests', 'sum'), joint_success=('joint_success_requests', 'sum'),
        deadline_misses=('deadline_misses', 'sum'), latency_mean_seconds=('latency_mean_seconds', 'mean'),
        waiting_mean_seconds=('waiting_mean_seconds', 'mean'), service_seconds=('service_seconds', 'sum'),
        round_over_budget_events=('round_over_budget_events', 'sum'), events=('events', 'sum'),
        full_requests=('full_requests', 'sum'), validation_f1_mean=('validation_macro_f1', 'mean'),
        validation_fpr_mean=('validation_fpr', 'mean'), bit_exact_all=('bit_exact', 'all')).reset_index()
    pooled.to_csv(args.root/'pooled.csv', index=False)
    pairs = []
    for block, group in frame.groupby('block'):
        dt = group[group.predictor == 'adaptive'].iloc[0]
        for name in ('last_shard', 'rows_ewma'):
            other = group[group.predictor == name].iloc[0]
            pairs.append(dict(block=block, environment=dt.environment, repeat=int(dt['repeat']), comparator=name,
                joint_success_delta=int(dt.joint_success_requests-other.joint_success_requests),
                deadline_misses_delta=int(dt.deadline_misses-other.deadline_misses),
                service_seconds_delta=float(dt.service_seconds-other.service_seconds),
                full_requests_delta=int(dt.full_requests-other.full_requests),
                validation_f1_delta=float(dt.validation_macro_f1-other.validation_macro_f1)))
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(args.root/'paired.csv', index=False)
    wins = pairs.groupby(['environment', 'comparator']).agg(
        wins=('joint_success_delta', lambda x: int((x > 0).sum())),
        ties=('joint_success_delta', lambda x: int((x == 0).sum())),
        losses=('joint_success_delta', lambda x: int((x < 0).sum())),
        joint_success_delta=('joint_success_delta', 'sum'),
        service_seconds_delta=('service_seconds_delta', 'sum')).reset_index()
    wins.to_csv(args.root/'paired_summary.csv', index=False)
    # Count the requests that would appear on time if actual service were omitted.
    misleading = 0
    for path in args.root.glob('*/*/events.jsonl'):
        for line in path.read_text().splitlines():
            e = json.loads(line)
            misleading += sum(w <= e['deadline_seconds'] < t for w, t in zip(e['waiting_seconds'], e['latency_seconds']))
    text = ['# Xác nhận DT online và deadline hoàn thành', '',
        f'Đã audit {len(frame)} chuỗi, {int(frame.requests.sum())} yêu cầu. '
        'Mỗi arm xử lý đủ yêu cầu đúng một lần; không dùng yêu cầu tương lai; '
        'latency = waiting + service; số liệu tổng hợp khớp event gốc; bit-exact đều đạt.', '',
        'Thiết kế cố định trước: [protocol](../../docs/dt_online_confirmation_plan.md), '
        '`protocol_lock.json`. Ba lần lặp, hai họ chuỗi; thống kê mô tả theo block, không kiểm định ý nghĩa thống kê.', '',
        '## Kết quả gộp', '', markdown(pooled), '',
        'Joint success: hoàn thành trong deadline và batch vừa ngân sách vòng FU. '
        'Deadline tính tới model-ready trong bộ nhớ, không gồm HTTP hay lưu checkpoint. '
        'Latency mean là trung bình các chuỗi có cùng số yêu cầu; p95 từng chuỗi có trong summary.csv.', '',
        '## Đối chiếu ghép cặp', '',
        'Delta = Adaptive − đối chứng; joint_success dương tốt hơn, service_seconds âm rẻ hơn. '
        'Win/tie/loss dựa trên số joint-success trong cùng block.', '', markdown(wins), '',
        '## Tác động của phép đo deadline', '',
        f'Có **{misleading}/{int(frame.requests.sum())} yêu cầu** mà thời gian chờ <= D nhưng '
        'completion latency > D: bỏ thời gian service sẽ đánh giá sai những yêu cầu này là đúng hạn. '
        'Đây là phân rã của lịch thực đã chạy, không phải đối chứng nhân quả với scheduler mô phỏng cũ.', '',
        '## Giới hạn', '',
        '- Một máy, stress được tiêm bằng sleep thật; điều kiện lấy mẫu đầu mỗi batch. Không đo mạng phân tán.',
        '- Cả ba dùng luật gom lô full/fixed, không kiểm chứng trực tiếp lợi ích midcourse.',
        '- Mọi yêu cầu cuối cùng được xử lý, không có loại bỏ hay phản hồi miễn phí. Forced jobs và deadline miss đều tính.',
        '- Cùng budget tuyệt đối trong block, khác giữa block theo fit. Thứ tự chạy được xoay vòng; drift máy vẫn có thể còn.',
        '- Utility chỉ trên validation; khác số vòng và lịch phục vụ. Không dùng test/calibration để chọn cấu hình.',
        '- Ba repeat không đủ để suy rộng hoặc khẳng định ưu thế thống kê; không điều chỉnh alpha theo bảng này.', '']
    (args.root/'report.md').write_text('\n'.join(text), encoding='utf-8')
    print(markdown(pooled[['environment', 'predictor', 'requests', 'joint_success', 'deadline_misses', 'service_seconds']]))
    print(f'\nAudit passed. Waiting-only would hide {misleading} completion deadline misses.')


if __name__ == '__main__':
    main()
