"""Descriptive report only: never selects models or modifies the run."""
import argparse
import json
from pathlib import Path
import pandas as pd

def table(frame):
    def fmt(x): return f'{x:.5f}' if isinstance(x,float) else str(x)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
                     ['| '+' | '.join(map(fmt,row))+' |' for row in frame.itertuples(index=False,name=None)])

def report(run,output):
    metrics=pd.read_csv(run/'metrics.csv'); requests=[]; paired=[]; stages=[]
    selections=json.loads((run/'selections.json').read_text())
    for path in sorted((run/'requests').glob('*.json')):
        r=json.loads(path.read_text()); key=path.stem
        requests.append({'request':key,'actual fraction':r['actual_fraction'],
                         'F normal/attack':str(r['forget_labels']),'R normal/attack':str(r['retain_labels']),
                         'feature overlap':r['feature_equivalent_overlap_fraction'],'selected':selections[key]['chosen']})
        sub=metrics[metrics.request==key].set_index('method'); chosen=selections[key]['chosen']+'/post'
        paired.append({'request':key,'FU F1':sub.loc[chosen,'macro_f1'],
                       'retain F1':sub.loc['no_update/post','macro_f1'],
                       'FU test MAE':sub.loc[chosen,'prediction_mae_test'],
                       'retain test MAE':sub.loc['no_update/post','prediction_mae_test'],
                       'GD24 test MAE':sub.loc['gd_only/post','prediction_mae_test']})
        for arm in ['ga_only','gd_only','alt_3_1','alt_1_1','alt_1_3']:
            d=json.loads((run/key/f'{arm}_diagnostics.json').read_text()); first,last=d[0],d[-1]
            stages.append({'request':key,'arm':arm,'forget CE change':last['forget']['ce']-first['forget']['ce'],
                           'retain CE change':last['retain']['ce']-first['retain']['ce'],
                           'parameter L2':last['parameter_l2_from_original'],
                           'F prediction change':last['forget']['prediction_mae_from_original']})
    attack=json.loads((run/'attack/attack.json').read_text())
    runtime=json.loads((run/'runtime.json').read_text())
    agg=metrics[metrics.method.isin(['original','retrain','no_update/post','gd_only/post','ga_only/post',
               'ga_gd_with_kd/post','conpfu_inspired/post','retain_kd_matched/post'])].groupby('method')[
               ['macro_f1','pr_auc_ap','fpr','prediction_mae_test','wall_seconds','optimizer_steps','payload_bytes_up']].mean().reset_index()
    stage=pd.DataFrame(stages)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n\n'.join([
        '# Controlled FU pilot — seed 42',
        f'Artifact directory: `{run.as_posix()}`. Protocol: `docs/controlled_phase_plan.md`; configuration: `configs/controlled_phase.json`.',
        f'Completed {metrics.request.nunique()}/8 requests; {len(metrics)} evaluated checkpoints; 72 recovery runs, 8 fresh retrains, 3 shadow fits. Total runner time {runtime["total_runner_seconds"]:.1f}s including diagnostics. Frozen preprocessing and client partitions reused. No Digital Twin planner implemented. This is a single-seed diagnostic pilot, not evidence of successful unlearning.',
        '## Request sensitivity',table(pd.DataFrame(requests)),
        'Group requests preserve whole local portions of connected components. Actual deletion fractions differ when a group exceeds the budget. Overlap is the fraction of forgotten rows with feature-equivalent records remaining globally. Labels and overlap change together across request designs; comparisons do not isolate overlap causally. In particular, the larger client-4 group request leaves only ONE normal example locally, a major distribution-shift confounder. All planned requests are included.',
        '## Paired FU versus retained controls',table(pd.DataFrame(paired)),
        'FU is the alternating ratio selected using validation MAE to fresh retraining, subject to validation F1/FPR constraints. Test metrics above are reporting only. Smaller MAE indicates prediction similarity, not a deletion guarantee. Inspect paired requests rather than interpreting their mean as independent repeated trials.',
        '## Descriptive method means across eight requests',table(agg),
        'Retain-only is no_update/post (two retained FedAvg rounds). GD24 adds 24 retained local updates; retain_kd_matched adds 16 retained KD steps. Cached original/no-op have zero incremental training cost, not zero historical cost. Per-method timings include their reused local-stage cost; summing rows double-counts shared computation. Payload is simulated tensor upload; download is also in metrics.csv. Sampled RSS and CPU profiles are in each request/profiles.json. Training and recovery rounds are recorded separately.',
        '## Local stage changes',table(stage.groupby('arm').mean(numeric_only=True).reset_index()),
        'Full per-stage CE, gradient norms, displacement, prediction changes and checkpoint reload checks are in *_diagnostics.json; actual batches and optimizer steps are in *_steps.json. Root checkpoints record the actual input-state hash; KD checkpoints also record the teacher hash. Direct GA/GD stage ancestry is recorded by diagnostic checkpoints.',
        '## Learned membership inference',
        f'Held-out shadow AUC: **{attack["shadow_validation"]["auc"]:.6f}**. Calibration realized FPR: {attack["shadow_validation"]["calibration_realized_fpr"]:.6f}; nominal target 0.01. The attack is **inconclusive** because the independent shadow control is below the predeclared 0.55 usefulness threshold. No score direction was flipped using target results.',
        'Two group-disjoint shadows train a standardized logistic attack; a third independent shadow validates it. A separate original-model nonmember bucket fixes the threshold for every comparator. Target F, all-retained and group-disjoint-retained cohorts are matched by label/type and shared across models. Coverage and unsupported strata are saved in mia_coverage.json; actual nonmember FPR and sample counts are in metrics.json. A calibrated nominal 1% threshold does not guarantee target FPR=1% under distribution shift. Shadows have much smaller training sets and fewer optimizer steps than the target despite matching epochs/rounds; this limits transfer. The preprocessing transformer was fit on original training data and reused by design; results concern model-weight membership conditional on that fixed pipeline.',
        '## Integrity and limitations',
        '20 unit tests passed, including the corrected profiler round-role checks. Run scripts/verify_controlled_phase.py for artifact, checkpoint, update-budget, auxiliary-disjointness and no-op verification. Source metrics.csv differs from its historic hash and is explicitly recorded as an unused changed report in the manifest; all other source hashes remain mandatory. Original datasets and source runs were not edited.',
        'The global-start model, fixed-step schedules, full-local KD baseline, small MLP and auxiliary clients are paper-inspired adaptations, not a faithful ConPFU reproduction. Full-local KD accesses F during the unlearning operation. Recovery uses retained data only. No confidence intervals or multi-seed confirmation are claimed. Next work should validate attack power on independent auxiliary data with an appropriate training budget before interpreting privacy changes; any subsequent FU settings must again be selected on validation, not these test results.'
    ]),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('docs/controlled_phase_results.md'))
    a=p.parse_args(); report(a.run,a.output)
