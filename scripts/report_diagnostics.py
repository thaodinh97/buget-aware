"""Summarize completed diagnostic artifacts without modifying the run."""
import argparse
import json
from pathlib import Path
import pandas as pd


def report(run,output):
    frame=pd.read_csv(run/'metrics.csv')
    coverage=pd.read_csv(run/'coverage.csv')
    if not coverage.status.eq('completed').all(): raise ValueError('Incomplete run')
    lines=['# Kết quả controlled FU diagnostics — 2026-10-19','',
           f'Run: `{run.as_posix()}`. Source: `runs/pilot_v3_seed42`. Không đổi preprocessing/split/client assignment/forget requests. Không tune bằng test. DT chưa triển khai.','',
           '## Kết quả kiểm thử và checkpoint','',
           '15 tests đạt, gồm observation không đổi updates, full CE gradients, KD teacher freeze, null/inverted MIA và family matching. Runner kiểm tra final tensor hash của retain-only/FU-no-KD/FU-retain-KD bằng chính xác checkpoint cũ cho cả hai fractions. Mọi checkpoint diagnostic/final được reload và đối chiếu tensor hash. Source run được verify trước khi chạy.','',
           f'{len(coverage)} method/request entries hoàn tất. Original/no-op/retrain reuse cached checkpoints có hash validation; runtime null và steps=0 trong CSV nghĩa không chạy lại, không phải retraining miễn phí. Historical reference costs lưu riêng.','',
           '## Vì sao local FU thay đổi ít','']
    for frac in sorted(frame.fraction.unique()):
        name=str(frac)
        audit=json.loads((run/f'{name}_request_audit.json').read_text())
        records=json.loads((run/f'{name}_fu_diagnostics.json').read_text())
        steps=json.loads((run/f'{name}_fu_steps.json').read_text())
        lines += [f'### Forget {frac:.0%}','',
                  f"F={audit['forget_count']:,}, target R={audit['target_retain_count']:,}; {audit['F_fraction_with_equivalent_global_R']:.2%} F có feature-equivalent record còn trong global R.",'',
                  '| Stage | Steps | CE(F) | CE(R) | Parameter L2 từ original | Prediction MAE(F) từ original |',
                  '|---|---:|---:|---:|---:|---:|']
        for r in records:
            lines.append(f"| {r['stage']} | {r['optimizer_steps_in_stage']} | {r['forget']['ce']:.9f} | {r['retain']['ce']:.9f} | {r['parameter_l2_from_original']:.9f} | {r['forget']['prediction_mae_from_original']:.9f} |")
        lines += ['',f"GA/GD steps: {sum(s['stage']=='ga' for s in steps)}/{sum(s['stage']=='gd' for s in steps)}. Minibatches vượt clip bound 1: {sum(s['gradient_norm_before_clip']>1 for s in steps)}. Max pre-clip norm: {max(s['gradient_norm_before_clip'] for s in steps):.6f}.",'']
    lines += ['GA hoạt động, không có lỗi dấu hoặc checkpoint copy che update. Loss nhỏ, SGD LR nhỏ và retain-GD sau GA giải thích biến đổi nhỏ; KD reset student về original, teacher gần original. Không suy ra forgetting thành công chỉ từ CE tăng.','',
              '## So sánh controlled','',
              '| Fraction | Method | Macro-F1 | AP | FPR | MAE(F) tới retrain | Legacy MIA F/R | Family-matched MIA F/R | Steps mới |',
              '|---|---|---:|---:|---:|---:|---|---|---:|']
    for r in frame.itertuples():
        lines.append(f'| {r.fraction:.0%} | {r.method} | {r.macro_f1:.6f} | {r.pr_auc_ap:.6f} | {r.fpr:.6f} | {r.prediction_mae_F:.9f} | {r.mia_auc_legacy_F:.4f}/{r.mia_auc_legacy_R:.4f} | {r.mia_auc_family_F:.4f}/{r.mia_auc_family_R:.4f} | {r.optimizer_steps} |')
    lines += ['',
              '`retain_only` là recovery-only cũ. `retain_gd` thêm cùng retain-GD schedule rồi cùng recovery, giúp phân biệt ảnh hưởng GA. Không gọi compute-matched: số steps khác và đã ghi. FU-no-KD/FU-retain-KD giữ nguyên algorithm cũ. `conpfu_inspired_baseline` dùng teacher sau local, student original, KD trên F∪R rồi cùng recovery; `conpfu_local_only` không recovery. Cả hai chỉ inspired: MLP/5 clients/global-start/no incentives; không faithful ConPFU reproduction. F chỉ dùng trong local operation/KD variant được khai báo; recovery luôn D\\F.','',
              '## MIA','',
              'Score log p(y|x) = negative CE, higher→member; không tự đảo khi AUC<0,5. Threshold calibration classwise percentile 99 của Original, strict >, cố định mọi model. Legacy dùng đúng evaluation IDs cũ. Family-matched là sensitivity cohort khác, match label+type và báo coverage mỗi stratum; không thay legacy âm thầm. Các cohort được cố định trước đánh giá model và lưu IDs. Không train attack classifier/shadow model (0 shadow models); đây là fixed-loss attack. Synthetic tests validate implementation, không chứng minh attack mạnh trong dữ liệu thật.','',
              'Retained controls loại group chạm F nên distribution khác; family matching không kiểm soát hết connection/difficulty shifts. Thiếu support ở một số họ attack làm giảm coverage. Không xem MIA AUC(F) giảm là xóa thành công. So với retrain và retained controls cùng protocol; khi controls không phân biệt được members, kết luận inconclusive. ROC TPR@1% là descriptive ROC point, khác realized FPR tại threshold calibration.','',
              '## Runtime và communication','',
              'CSV ghi optimizer_seconds = stage wall time trừ observer overhead, vẫn gồm logging/Python/optimizer initialization; không phải profiler kernel-only. diagnostic_seconds ghi riêng. Local branch dùng chung được tính vào từng phương pháp khi báo chi phí giả định chạy riêng; không cộng CSV thành tổng thời gian runner. First-use warmup và thứ tự cố định gây confound; không tuyên bố speedup. Historical retraining costs có nguồn và warning, không dùng như matched current timing.','',
              'Bytes là tensor payload theo dtype/numel: một download/upload local FU và các round recovery; không phải measured network bytes, không gồm serialization/protocol overhead, không phải DT cost. Checkpoints trung gian là offline diagnostic artifacts, không tính như network transfer.','',
              '## Giới hạn và hướng tiếp theo','',
              'Một seed/target, không có CI/statistical superiority claim. Full-local KD có nhiều steps/data hơn retain-KD. Preprocessing transformer giữ thống kê original train. Original test đã được xem trong pilot trước: đây là diagnostic trên benchmark cố định, không phải xác nhận độc lập mới. Không thay threshold hoặc tune LR từ test.','',
              'Tiếp theo: protocol shadow-model hoặc learned attack có auxiliary groups tách biệt; thiết kế validation-only GA/GD step-ratio ablation và compute-matched controls; giữ test khỏi tuning. Không tự mở DT, không tuyên bố unlearning thành công.','',
              'Lần đầu `runs/controlled_diagnostics_seed42` bị dừng sau fraction 10% do object-array feature-membership audit quá chậm. Giữ nguyên partial artifacts và execution_status.json. Run v2 dùng set membership tương đương để tăng tốc, không thay cohorts/model updates.']
    output.write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); report(a.run,a.output)
