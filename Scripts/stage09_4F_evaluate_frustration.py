#!/usr/bin/env python3
"""Stage 09.4F: leakage-audited nested RF add-back of energetic frustration.

Run from GroElLabels project root:
  conda activate groe_frustration
  python Scripts/stage09_4F_evaluate_frustration.py --audit-only
  python Scripts/stage09_4F_evaluate_frustration.py --n-jobs 4

Uses the project's existing Stage 09 v2 FIXED implementation for tuning,
RF construction and metric definitions. Does not modify frozen inputs.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score, balanced_accuracy_score

ID = 'current_uniprot_accession'
LABEL = 'binary_label'
STRUCTURAL = [
 'structure_ca_length','relative_contact_order','median_contact_sequence_separation',
 'long_range_contact_fraction','medium_range_contact_fraction',
 'ca_contact_density_per_residue','ca_contact_pair_fraction',
 'zero_contact_residue_fraction','radius_of_gyration_normalized_n_cuberoot',
 'axis_ratio_1_to_2','asphericity','ss_helix_fraction','ss_sheet_fraction',
 'minimum_cut_cross_contact_density','best_cut_fraction'
]
FRUSTRATION = [
 'mut_mean','mut_std','mut_high_fraction','mut_minimal_fraction',
 'mut_high_residue_fraction','mut_long_range_high_fraction',
 'conf_mean','conf_std','conf_high_fraction','conf_minimal_fraction',
 'conf_high_residue_fraction','conf_long_range_high_fraction'
]

def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def load_original(path):
 spec=importlib.util.spec_from_file_location('stage09v2_original',path)
 module=importlib.util.module_from_spec(spec)
 spec.loader.exec_module(module)
 for name in ['tune_outer_training','build_rf','metric_dict','audit_outer_split']:
  if not hasattr(module,name):raise RuntimeError(f'Original Stage 09 v2 script lacks {name}')
 return module

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
 p.add_argument('--audit-only',action='store_true')
 p.add_argument('--n-iter',type=int,default=80,help='Stage 09 v2 default is 80')
 p.add_argument('--n-jobs',type=int,default=4)
 p.add_argument('--seed',type=int,default=42)
 p.add_argument('--models',nargs='+',choices=['structural_core','frustration_only','combined'],default=['structural_core','frustration_only','combined'])
 args=p.parse_args()
 root=args.root.resolve()
 files={
  'structural':root/'data/frozen/feature_set_v3/feature_set_v3_structure_core_final.csv',
  'frustration':root/'results/stage09_4F/frustration_features.csv',
  'folds':root/'data/grouping/stage07/fold_assignments_v1.0.csv',
  'original':root/'Scripts/09_v2_final_random_forest_FIXED.py',
  'baseline':root/'results/stage09_v2/rf_fold_metrics.csv'
 }
 for key,path in files.items():
  if not path.is_file():raise FileNotFoundError(f'{key}: {path}')
 out=root/'results/stage09_4F/modeling'
 out.mkdir(parents=True,exist_ok=True)
 structural=pd.read_csv(files['structural'])
 fr=pd.read_csv(files['frustration']).rename(columns={'accession':ID})
 folds=pd.read_csv(files['folds'])
 for name,df in [('structural',structural),('frustration',fr),('folds',folds)]:
  if df[ID].duplicated().any():raise ValueError(f'Duplicate IDs in {name}')
  if len(df)!=181:raise ValueError(f'{name}: expected 181, got {len(df)}')
  if set(df[ID])!=set(structural[ID]):raise ValueError(f'{name}: accession mismatch')
 for col in STRUCTURAL:
  if col not in structural:raise ValueError(f'Missing structural predictor: {col}')
 for col in FRUSTRATION:
  if col not in fr:raise ValueError(f'Missing frustration predictor: {col}')
 fold_cols=[ID,'homology_cv_fold','homology_cluster_30']
 d=structural[[ID,LABEL]+STRUCTURAL].merge(fr[[ID]+FRUSTRATION],on=ID,validate='one_to_one').merge(folds[fold_cols],on=ID,validate='one_to_one')
 d=d.reset_index(drop=True)
 if d[STRUCTURAL+FRUSTRATION].isna().any().any():raise ValueError('Missing predictors')
 if not np.isfinite(d[STRUCTURAL+FRUSTRATION].to_numpy(dtype=float)).all():raise ValueError('Nonfinite predictors')
 if set(d[LABEL].unique())!={0,1}:raise ValueError('Unexpected labels')
 if d[LABEL].sum()!=73:raise ValueError('Expected 73 positive labels')
 if len(d.homology_cv_fold.unique())!=5:raise ValueError('Expected five frozen outer folds')
 for fold in sorted(d.homology_cv_fold.unique()):
  train=d[d.homology_cv_fold!=fold];test=d[d.homology_cv_fold==fold]
  overlap=set(train.homology_cluster_30.astype(str)) & set(test.homology_cluster_30.astype(str))
  if overlap:raise ValueError(f'Group leakage in outer fold {fold}: {overlap}')
  if train[LABEL].nunique()!=2 or test[LABEL].nunique()!=2:raise ValueError('Fold class missing')
 baseline=pd.read_csv(files['baseline'])
 base=baseline[(baseline.feature_set=='structure_core_final') & (baseline.subset=='full') & (baseline.outer_fold_column=='homology_cv_fold') & (baseline.threshold_type=='fixed_0_5')]
 if len(base)!=5:raise ValueError('Could not find five reference baseline fold metrics')
 metadata={'n':len(d),'positives':int(d[LABEL].sum()),'structural_features':STRUCTURAL,'frustration_features':FRUSTRATION,'outer_fold_column':'homology_cv_fold','group_column':'homology_cluster_30','inner_cv':'StratifiedGroupKFold(n_splits=3)','n_iter':args.n_iter,'seed':args.seed,'files':{k:{'path':str(v.relative_to(root)),'sha256':sha256(v)} for k,v in files.items()},'reference_baseline_mcc_by_fold':{str(int(row.outer_fold)):float(row.mcc) for row in base.itertuples()}}
 (out/'input_audit.json').write_text(json.dumps(metadata,indent=2)+'\n')
 print('INPUT AUDIT PASSED: 181 proteins, 73 positives, 15 + 12 predictors, 5 leakage-free frozen outer folds')
 print('Original baseline fold MCC:',metadata['reference_baseline_mcc_by_fold'])
 if args.audit_only:return
 original=load_original(files['original'])
 sets={'structural_core':STRUCTURAL,'frustration_only':FRUSTRATION,'combined':STRUCTURAL+FRUSTRATION}
 predictions=[];metrics=[];selected=[];searches=[]
 for model_name in args.models:
  print('\nMODEL',model_name,flush=True)
  features=sets[model_name]
  for fold in sorted(d.homology_cv_fold.unique()):
   t0=time.time()
   train=d[d.homology_cv_fold!=fold];test=d[d.homology_cv_fold==fold]
   xtrain=train[features];ytrain=train[LABEL].to_numpy(dtype=int)
   # Preserve original baseline's Stage 09 v2 seed calculation.
   # Same tuning random stream for all three models, enabling controlled comparison.
   tune_seed=args.seed+int(fold)*100_000+len('structure_core_final')*1000+len('full')*100
   params,threshold,search,top5,inner_folds=original.tune_outer_training(
     xtrain,ytrain,train.homology_cluster_30.astype(str).to_numpy(),
     inner_folds=3,n_iter=args.n_iter,seed=tune_seed,n_jobs=args.n_jobs)
   search=search.copy();search.insert(0,'model',model_name);search.insert(1,'outer_fold',int(fold));searches.append(search)
   model=original.build_rf(params,seed=args.seed+int(fold),n_jobs=args.n_jobs)
   model.fit(xtrain,ytrain)
   prob=model.predict_proba(test[features])[:,1]
   yt=test[LABEL].to_numpy(dtype=int)
   pred=(prob>=.5).astype(int)
   met=original.metric_dict(yt,pred,prob)
   metrics.append({'model':model_name,'outer_fold':int(fold),'n_test':len(test),'threshold':.5,**met})
   for accession,y,pr,predicted in zip(test[ID],yt,prob,pred):
    predictions.append({'model':model_name,'accession':accession,'outer_fold':int(fold),'binary_label':int(y),'probability':float(pr),'prediction':int(predicted)})
   selected.append({'model':model_name,'outer_fold':int(fold),'inner_folds':int(inner_folds),'inner_selected_threshold_report_only':float(threshold),'tune_seed':tune_seed,'params':json.dumps(params,sort_keys=True,default=str),'seconds':time.time()-t0})
   print(f' fold={fold} MCC={met["mcc"]:.3f} ROC-AUC={met["roc_auc"]:.3f} elapsed={time.time()-t0:.1f}s',flush=True)
   # Checkpoint after every fold, allowing inspection if interrupted.
   pd.DataFrame(predictions).to_csv(out/'oof_predictions.csv',index=False)
   pd.DataFrame(metrics).to_csv(out/'outer_fold_metrics.csv',index=False)
   pd.DataFrame(selected).to_csv(out/'selected_hyperparameters.csv',index=False)
   pd.concat(searches,ignore_index=True).to_csv(out/'inner_search_results.csv',index=False)
 summary=[]
 for model_name in args.models:
  q=pd.DataFrame(predictions);q=q[q.model==model_name]
  if len(q)!=181 or q.accession.nunique()!=181:raise ValueError(f'OOF incomplete: {model_name}')
  yt=q.binary_label.to_numpy();pr=q.probability.to_numpy();yp=q.prediction.to_numpy()
  summary.append({'model':model_name,'n':len(q),'mcc':matthews_corrcoef(yt,yp),'balanced_accuracy':balanced_accuracy_score(yt,yp),'roc_auc':roc_auc_score(yt,pr),'pr_auc':average_precision_score(yt,pr)})
 summary=pd.DataFrame(summary)
 summary.to_csv(out/'pooled_oof_summary.csv',index=False)
 print('\nPOOLED OOF SUMMARY\n',summary.to_string(index=False))
 if 'structural_core' in args.models:
  observed=pd.DataFrame(metrics).query('model == "structural_core"').set_index('outer_fold')
  expected=base.set_index('outer_fold')
  diffs=(observed['mcc']-expected['mcc']).abs()
  report={'maximum_absolute_fold_mcc_difference':float(diffs.max()),'exact_mcc_reproduction':bool((diffs<1e-12).all()),'differences_by_fold':{str(k):float(v) for k,v in diffs.items()}}
  (out/'baseline_reproduction_audit.json').write_text(json.dumps(report,indent=2)+'\n')
  print('\nBASELINE REPRODUCTION:',report)
  if not report['exact_mcc_reproduction']:
   print('WARNING: baseline differs from original; investigate before interpreting model differences.')

if __name__=='__main__':main()
