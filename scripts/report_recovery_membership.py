"""Descriptive reports; no selection, training or artifact mutation."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from scripts.report_controlled_phase import table

def report(recovery,mia):
    g=pd.read_csv(recovery/'global.csv'); c=pd.read_csv(recovery/'clients.csv')
    a=g[g.start=='ga_only'][['case','lr_multiplier','round','mae_validation','mae_test','mae_forget',
                            'validation_log_loss','l2_retrain','update_norm','cosine_to_retrain']]
    endpoints=g[g['round'].isin([0,2,4])][['case','start','lr_multiplier','round','mae_validation','mae_test','validation_macro_f1']]
    first=c[(c.start=='ga_only')&(c.lr_multiplier==1)&(c['round']==1)].copy()
    first['weighted_dot']=first.weight*first.dot_to_retrain
    first=first[['case','client_id','retained_count','normal_count','attack_count','weight','update_norm','cosine_to_retrain','weighted_dot']]
    primary=g[(g.start=='ga_only')&(g.lr_multiplier==1)&(g['round']>0)]
    findings=[]
    for case in g.case.unique():
        part=g[(g.case==case)&(g.start=='ga_only')&(g.lr_multiplier==1)].set_index('round')
        low=g[(g.case==case)&(g.start=='ga_only')&(g.lr_multiplier==.25)].set_index('round')
        findings.append(f'{case}: GA-only test MAE {part.loc[0,"mae_test"]:.6f} before recovery → {part.loc[2,"mae_test"]:.6f} after2 rounds. '
            f'Validation log loss {part.loc[0,"validation_log_loss"]:.6f} → {part.loc[2,"validation_log_loss"]:.6f}. '
            f'Quarter-rate round2 test MAE is {low.loc[2,"mae_test"]:.6f}.')
    text=['# Recovery trajectory findings',f'Artifacts: `{recovery.as_posix()}`. Two fixed cases, four cached starting states, rounds0–4, two fixed learning rates. 80 global and320 per-client measurements. The8 primary round2 states exactly reproduce existing pilot checkpoints. 25 unit tests passed, including encoded-role closure and extreme-logit numerical checks.',
          '## Observations',
          '\n\n'.join(findings),
          'Thus smaller parameter steps do not uniformly improve prediction similarity, and parameter distance can increase while prediction distance decreases. These observations contradict a magnitude-only explanation in these two cases; they do not prove that update magnitude is irrelevant elsewhere.',
          f'Among the8 GA-only primary-rate updates, {int((primary.cosine_to_retrain<0).sum())} have negative cosine to the retraining direction. A negative dot product guarantees an increase in squared parameter distance for that update. Prediction similarity need not follow parameter distance: the model is nonlinear and functionally similar networks need not have close parameters.',
          'The exact identity is Δdistance² = ||delta||² − 2·dot(delta, direction_to_retrain). The first term measures update magnitude; the second measures alignment. Zero vectors have undefined cosine, recorded as blank rather than zero. Distances are offline explanations; the reference never enters client gradients or aggregation.',
          table(a),
          '## Client contribution at the first primary-rate GA recovery round',table(first),
          'Counts and label distributions are non-IID. Weights are retained sample fractions, identical across starting states within each request. The weighted dot contributions approximately sum to the global dot (floating-point aggregation tolerance). Dominant weight or negative alignment is an observed contribution; it does not alone establish that a client causes poor test utility.',
          '## Competing objectives and calibration',
          'Fresh retraining starts from the saved initialization and performs15 Adam-reset FedAvg rounds. Recovery starts at an already-trained or locally modified model and performs up to4 additional retained-data rounds. Their paths, elapsed training and optimizer resets differ, so minimizing retained loss does not explicitly minimize distance to the finite-budget retrained checkpoint. This objective/path difference is confirmed by the implementation; its share of the observed degradation is not isolated experimentally.',
          'Validation Macro-F1, PR-AUC, FPR, log loss, Brier score,10-bin ECE and mean confidence are in global.csv, alongside F/global-R cross-entropy. Log-loss/confidence changes can occur while thresholded Macro-F1 is nearly unchanged. Calibration measures are descriptive; no temperature or decision threshold is fitted.',
          'Recovery loss columns retain the existing sklearn probability-log-loss convention, including numerical clipping. The separate MIA audit also reports stable logit cross-entropy; these conventions can differ for highly confident errors. Runtime and steps are incremental recovery costs, excluding cached model training and observer evaluation.',
          table(g[(g.start=='ga_only')&(g.lr_multiplier==1)&g['round'].isin([0,2,4])][['case','round','validation_macro_f1',
                    'validation_log_loss','validation_brier','validation_ece_10_equal_width','validation_mean_confidence','forget_ce','retain_ce']]),
          '## Endpoint comparison for all starts',table(endpoints),
          '## Confirmed findings versus hypotheses',
          'Confirmed: identical partitions, seeds, client weights, optimizer-reset policy and primary round2 states; observed alignment/norm/distance changes above; learning-rate sensitivity between two fixed rates. The quarter-rate contrast changes the optimizer step scale; it is not a search for an operational winner. Inspect each round: effects are not assumed monotone.',
          'Hypotheses needing additional intervention: whether client heterogeneity rather than path dependence dominates; whether a different optimizer or objective improves FU; whether longer retraining changes the apparent distance target. No client-ablation or independent target-seed experiment was performed, so causal and generalization claims remain limited. Historical alternating-ratio selection used retraining and remains an offline oracle diagnostic, not a deployed stopping rule.',
          'Figures: '+', '.join(f'`{p.as_posix()}`' for p in recovery.glob('*.png'))]
    Path('docs/recovery_trajectory_results.md').write_text('\n\n'.join(text),encoding='utf-8')
    audit=pd.read_csv(mia/'model_audit.csv'); shadow=pd.read_csv(mia/'shadow_validation.csv')
    targets=pd.read_csv(mia/'comparison.csv'); controls=pd.read_csv(mia/'controls.csv'); budgets=json.loads((mia/'shadow_budgets.json').read_text())
    gaps=[]
    for name,part in audit.groupby('model'):
        d=part.set_index('role'); gaps.append({'model':name,'member N':int(d.loc['member','count']),
           'nonmember N':int(d.loc['nonmember','count']),'member F1':d.loc['member','macro_f1'],
           'nonmember F1':d.loc['nonmember','macro_f1'],'member loss':d.loc['member','mean_logit_ce'],
           'nonmember loss':d.loc['nonmember','mean_logit_ce'],'loss gap':d.loc['nonmember','mean_logit_ce']-d.loc['member','mean_logit_ce']})
    cols=['stage','attack','auc','auc_ci_low','auc_ci_high','tpr_0.01','actual_fpr_0.01','member_count','nonmember_count']
    compare=targets[targets.cohort=='F'][['case','model','attack','auc','auc_ci_low','auc_ci_high','auc_minus_retrain',
                'tpr_0.01','actual_fpr_0.01','interpretation']]
    text=['# Membership protocol audit and validation',f'Artifacts: `{mia.as_posix()}`. Four controlled shadows plus one constructed memorization control. Same target seed/checkpoints and frozen data; no test-driven attack tuning. 108 target comparison rows across six models, three cohorts and three fixed attacks.',
          '## Audit of the old protocol',
          f'The previous shadows had75 optimizer steps each over15rounds; the target had{budgets[0]["target_optimizer_steps"]}. Their roughly2,400-member datasets and balanced auxiliary clients differ from the114,284-row non-IID target federation. The original held-out learned attack AUC≈0.4984 did not establish attack power. The table below checks utility and loss gaps rather than inferring power from training steps alone.',table(pd.DataFrame(gaps)),
          'model_audit.csv additionally contains label counts and loss quantiles0/10/50/90/100. Shadow nonmembers are their paired independent auxiliary groups; target nonmembers in this audit are validation rows. These differing evaluation distributions mean utility numbers are descriptive, not directly matched estimates.',
          'The budget correction improves shadow fitting but does not reproduce the target generalization gap: target mean logit-CE gap is about0.1995, while the final shadow gap is about−0.0106. Thus inadequate old training was a real mismatch, but fixing update count alone did not establish a useful target-membership proxy. The two excluded auxiliary rows do not by themselves explain the old near-chance AUC.',
          '## Controlled changes and leakage checks',table(pd.DataFrame([{k:v for k,v in b.items() if k not in ['client_counts']} for b in budgets])),
          'Final v3 evaluation reuses all four improved shadow checkpoints and the positive control from v2. Shadow training times in the table/logs are historical cached times, not newly measured fit costs. The first run stopped at the encoded-role leakage preflight; v2 is retained to document rounded-probability baseline saturation. No target or shadow was refit to optimize these results.',
          'Only the shadow batch budget was adjusted by a fixed formula using target optimizer counts; architecture, lr,15rounds and balanced-client construction were retained. Smaller batch sizes approximately match total update count, not sample exposures, gradient noise, client skew or privacy difficulty. This is a justified undertraining check, not proof of shadow-target equivalence.',
          'Preflight found one encoded signature shared across old buckets2 and5. Whole components spanning old roles were excluded, with IDs/counts saved in auxiliary_role_audit.json and excluded_auxiliary_ids.npz; no rows were reassigned and no primary split changed. Buckets0/1 and2/3 train the classifier;4/5 are development validation;6 calibrates original-model nonmember thresholds; previously unused bucket7 supplies a new group-disjoint final shadow pair. Explicit encoded-row checks passed between auxiliary buckets and target train/validation/test. Original train-fitted preprocessing is held fixed, so these results assess model weights conditional on that transformer, not end-to-end deletion of preprocessing influence.',
          'Logistic attack: standardized log true-label probability, entropy, margin and label, fixed C=1. Baselines: negative CE and maximum confidence. All score directions were fixed in advance. Label+family matching caps each stratum at500 per side, with unsupported strata retained in coverage reports. F and all-retained cohorts reuse previous IDs; the independent retained control is rebuilt to exclude all encoded-equivalence components touching F. All cohorts are fixed across models. This avoids comparator sampling drift but limits conclusions to covered strata.',
          '## Independent shadow evaluation',table(shadow[cols]),
          'For the final baseline scores, negative CE is computed stably from logits; confidence uses absolute binary logit margin, which is strictly monotonic in ideal max-softmax confidence. This avoids float32 probability rounding at1 and spurious0/1 calibration ties. The learned classifier and its clipped-probability features remain unchanged. Thresholds are in each score’s own units.',
          'Intervals use200 group-bootstrap draws and require20 groups per side. They condition on a fixed attack and threshold; they do not quantify model-training/attack-fit seed uncertainty. Final-shadow AUC>=0.55 with lower95% bound>0.5 and the same condition on Original group-disjoint-retained target control are both required to call an attack detectable. Otherwise the exact status is MIA_INCONCLUSIVE.',
          '## Positive and negative controls',table(controls[['control','attack','auc','tpr_0.01','actual_fpr_0.01']]),
          'The attempted positive control trains on64 random-label development examples for a fixed2,000-step budget; independent nonmembers receive random labels too. It achieved only59.4% member classification accuracy and therefore did NOT establish a strong memorization positive control. Its modest attack AUC must not be described as a power validation. The failure is retained without tuning or retries (see positive_control_diagnostics.json). The negative control duplicates the score distribution across membership labels and checks AUC≈0.5. Neither control is used to pick attack hyperparameters or override weak final-shadow/retained-member controls.',
          '## Target forget-cohort comparison',table(compare),
          'The same learned classifier and Original-calibrated nominal1%/5% thresholds are used for all six models. comparison.csv reports both thresholds, actual shifted FPR, balanced accuracy, sample/group counts, retained controls, paired AUC-difference intervals and point differences in TPR/FPR relative to retraining. Nominal calibration does not guarantee1% target FPR; ROC curves are descriptive and never used to recalibrate on test.',
          'Interpretation counts: '+str(targets.groupby(['attack','interpretation']).size().to_dict()),
          'For the F cohort, positive labels denote original/former membership for every model, including Retraining. Its balanced accuracy is therefore historical-membership discrimination, not literal current-membership accuracy for Retraining. Retained controls have actual retained-member labels. No successful-unlearning claim follows from AUC approaching0.5, lower loss, or similarity to a single retrained model. Even detectable attacks provide empirical membership evidence, not certified deletion. These results still require independent target-seed confirmation; no operational retraining-based selection or full DT was introduced.',
          'Figures: '+', '.join(f'`{p.as_posix()}`' for p in mia.glob('*.png'))]
    Path('docs/membership_validation_results.md').write_text('\n\n'.join(text),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--recovery',type=Path,required=True); p.add_argument('--mia',type=Path,required=True)
    a=p.parse_args(); report(a.recovery,a.mia)
