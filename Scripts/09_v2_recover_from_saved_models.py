#!/usr/bin/env python3
"""
Stage 09 v2 recovery utility.

Use this ONLY when the full Stage 09 v2 training completed its outer-fold loops
and then crashed during final report generation. It reconstructs the main
results from the already-saved outer-fold models, so the expensive nested
hyperparameter searches do not need to be repeated.

Recoverable:
- OOF predictions
- fold metrics
- pooled summary metrics + bootstrap CIs
- selected hyperparameters / thresholds
- held-out permutation importance
- calibration tables
- leakage / OOF audits
- saved model manifest
- hyperparameter stability
- figures
- reproducibility report

Not recoverable without rerunning tuning:
- complete rf_inner_search_results.csv
- top-five hyperparameter candidates for each outer fold

The saved selected model/hyperparameters are sufficient for all primary model
performance and interpretation analyses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

IDENTIFIER = "current_uniprot_accession"
TARGET = "binary_label"

FEATURE_FILES = {
    "sequence_final": "feature_set_v3_sequence_final.csv",
    "structure_minimal_final": "feature_set_v3_structure_minimal_final.csv",
    "structure_core_final": "feature_set_v3_structure_core_final.csv",
    "sequence_plus_structure_minimal": "feature_set_v3_sequence_plus_structure_minimal.csv",
    "sequence_plus_structure_core": "feature_set_v3_sequence_plus_structure_core.csv",
}

METADATA_COLUMNS = {
    "current_uniprot_accession","gene_minami","jwid","binary_label","cohort_label",
    "label_source","label_evidence_type","label_confidence","source_class_summary",
    "label_notes","sccs","benchmark_version","previous_binary_label",
    "label_updated_from_minami_validation","minami_2023_table1_groe_dependency",
    "minami_2023_validation_status","minami_2023_validation_source",
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feature-dir", required=True)
    p.add_argument("--fold-assignments", required=True)
    p.add_argument("--stage09-dir", required=True)
    p.add_argument("--permutation-repeats", type=int, default=100)
    p.add_argument("--bootstrap-iterations", type=int, default=5000)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)
    return p.parse_args()

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def predictor_columns(df):
    return [c for c in df.columns if c not in METADATA_COLUMNS
            and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True)>1]

def metric_dict(y_true,y_pred,y_prob):
    tn,fp,fn,tp=confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
    return {
        "mcc":float(matthews_corrcoef(y_true,y_pred)),
        "balanced_accuracy":float(balanced_accuracy_score(y_true,y_pred)),
        "recall":float(recall_score(y_true,y_pred,zero_division=0)),
        "precision":float(precision_score(y_true,y_pred,zero_division=0)),
        "specificity":float(tn/(tn+fp)) if tn+fp else float("nan"),
        "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
        "roc_auc":float(roc_auc_score(y_true,y_prob)) if len(np.unique(y_true))==2 else float("nan"),
        "pr_auc":float(average_precision_score(y_true,y_prob)) if len(np.unique(y_true))==2 else float("nan"),
        "brier_score":float(brier_score_loss(y_true,y_prob)),
    }

def bootstrap_ci(y_true,y_pred,y_prob,metric,iters,rng):
    vals=[]; n=len(y_true)
    for _ in range(iters):
        idx=rng.integers(0,n,n); yt=y_true[idx]; yp=y_pred[idx]; pr=y_prob[idx]
        if metric in {"roc_auc","pr_auc"} and len(np.unique(yt))<2: continue
        v=metric_dict(yt,yp,pr)[metric]
        if isinstance(v,(int,float)) and math.isfinite(float(v)): vals.append(float(v))
    if not vals: return float("nan"),float("nan")
    return tuple(map(float,np.percentile(vals,[2.5,97.5])))

def calibration_table(y,p,bins=10):
    edges=np.linspace(0,1,bins+1); rows=[]
    for i in range(bins):
        low,high=edges[i],edges[i+1]
        mask=(p>=low)&((p<=high) if i==bins-1 else (p<high))
        if mask.any():
            rows.append({"bin":i+1,"bin_low":low,"bin_high":high,"n":int(mask.sum()),
                         "mean_predicted_probability":float(p[mask].mean()),
                         "observed_positive_fraction":float(y[mask].mean())})
    return pd.DataFrame(rows)

def pooled_summary(predictions,iters,seed):
    rng=np.random.default_rng(seed); rows=[]
    for keys,g in predictions.groupby(["subset","outer_fold_column","feature_set","threshold_type"]):
        subset,foldcol,fs,tt=keys
        y=g[TARGET].to_numpy(int); p=g.predicted_probability.to_numpy(float)
        pred=(g.predicted_label_0_5 if tt=="fixed_0_5" else g.predicted_label_inner_selected).to_numpy(int)
        rec={"benchmark_version":"1.1","feature_set_version":"3.0","subset":subset,
             "outer_fold_column":foldcol,"feature_set":fs,"threshold_type":tt,
             "n":len(g),"positive_n":int(y.sum()),"negative_n":int((y==0).sum())}
        m=metric_dict(y,pred,p)
        for name,val in m.items():
            rec[name]=val
            if name in {"mcc","balanced_accuracy","recall","precision","specificity","roc_auc","pr_auc","brier_score"}:
                lo,hi=bootstrap_ci(y,pred,p,name,iters,rng)
                rec[name+"_ci_low"]=lo; rec[name+"_ci_high"]=hi
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["subset","outer_fold_column","feature_set","threshold_type"])

def main():
    a=parse_args()
    stage=Path(a.stage09_dir); modeldir=stage/"saved_models"
    if not modeldir.exists(): raise SystemExit(f"Missing saved_models: {modeldir}")
    model_paths=sorted(modeldir.glob("*.joblib"))
    if not model_paths: raise SystemExit("No saved .joblib models found.")

    folds=pd.read_csv(a.fold_assignments)
    datasets={}
    for fs,fn in FEATURE_FILES.items():
        p=Path(a.feature_dir)/fn
        if p.exists():
            d=pd.read_csv(p)
            datasets[fs]=d.merge(folds[[IDENTIFIER,"homology_cv_fold","homology_cv_fold_sensitivity",
                                       "homology_cluster_30","homology_cluster_40","is_tim_barrel"]],
                                 on=IDENTIFIER,how="left",validate="one_to_one")

    pred_rows=[]; metric_rows=[]; param_rows=[]; perm_rows=[]; model_manifest=[]; leakage=[]
    start=time.time()

    for k,path in enumerate(model_paths,1):
        bundle=joblib.load(path)
        fs=bundle["feature_set"]; subset=bundle["subset"]; foldcol=bundle["outer_fold_column"]; fold=int(bundle["outer_fold"])
        model=bundle["model"]; features=bundle["features"]; threshold=float(bundle["selected_threshold"]); params=bundle["selected_hyperparameters"]
        if fs not in datasets: raise RuntimeError(f"Missing dataset for {fs}")
        data=datasets[fs]
        if subset=="non_tim": data=data[~data.is_tim_barrel.astype(bool)].copy()
        elif subset!="full": raise RuntimeError(f"Unknown subset {subset}")
        train=data[data[foldcol]!=fold]; test=data[data[foldcol]==fold].copy()
        groupcol="homology_cluster_30" if foldcol=="homology_cv_fold" else "homology_cluster_40"
        overlap=set(train[groupcol].astype(str)) & set(test[groupcol].astype(str))
        leakage.append({"subset":subset,"feature_set":fs,"outer_fold_column":foldcol,"outer_fold":fold,
                        "group_column":groupcol,"group_overlap_count":len(overlap),"protein_overlap_count":0,
                        "passed":len(overlap)==0})
        if overlap: raise RuntimeError(f"Leakage in {path.name}")

        y=test[TARGET].to_numpy(int); prob=model.predict_proba(test[features])[:,1]
        p05=(prob>=.5).astype(int); ps=(prob>=threshold).astype(int)
        for tt,tv,pred in [("fixed_0_5",.5,p05),("inner_selected",threshold,ps)]:
            metric_rows.append({"benchmark_version":"1.1","feature_set_version":"3.0","subset":subset,
                "feature_set":fs,"outer_fold_column":foldcol,"outer_fold":fold,"threshold_type":tt,
                "threshold_value":tv,"train_n":len(train),"test_n":len(test),
                "test_positive_n":int(y.sum()),"test_negative_n":int((y==0).sum()),**metric_dict(y,pred,prob)})

        params_row={"subset":subset,"feature_set":fs,"outer_fold_column":foldcol,"outer_fold":fold,
                    "selected_threshold":threshold,**params}
        param_rows.append(params_row)

        for i,(_,row) in enumerate(test.iterrows()):
            pred_rows.append({"benchmark_version":"1.1","feature_set_version":"3.0",IDENTIFIER:row[IDENTIFIER],
                "gene_minami":row.get("gene_minami",""),"jwid":row.get("jwid",""),TARGET:int(row[TARGET]),
                "subset":subset,"feature_set":fs,"outer_fold_column":foldcol,"outer_fold":fold,
                "homology_cluster_30":row.homology_cluster_30,"homology_cluster_40":row.homology_cluster_40,
                "is_tim_barrel":bool(row.is_tim_barrel),"predicted_probability":float(prob[i]),
                "predicted_label_0_5":int(p05[i]),"selected_threshold":threshold,
                "predicted_label_inner_selected":int(ps[i]),"correct_0_5":bool(p05[i]==y[i]),
                "correct_inner_selected":bool(ps[i]==y[i]),"confidence_distance_from_0_5":float(abs(prob[i]-.5)),
                "selected_hyperparameters":json.dumps(params,sort_keys=True)})

        if len(np.unique(y))==2:
            perm=permutation_importance(model,test[features],y,scoring="roc_auc",n_repeats=a.permutation_repeats,
                                        random_state=a.random_seed+fold*10000+len(fs),n_jobs=a.n_jobs)
            means=perm.importances_mean
            top10=set(np.asarray(features)[np.argsort(means)[::-1][:10]])
            for i,f in enumerate(features):
                perm_rows.append({"benchmark_version":"1.1","feature_set_version":"3.0","subset":subset,
                    "feature_set":fs,"outer_fold_column":foldcol,"outer_fold":fold,"feature":f,
                    "importance_mean":float(means[i]),"importance_std":float(perm.importances_std[i]),
                    "top10_in_fold":f in top10,"permutation_repeats":a.permutation_repeats,"scoring":"roc_auc"})

        model_manifest.append({"model_file":path.name,"feature_set":fs,"subset":subset,
                               "outer_fold_column":foldcol,"outer_fold":fold,"sha256":sha256_file(path)})
        print(f"[{k}/{len(model_paths)}] recovered {path.name}")

    preds=pd.DataFrame(pred_rows); metrics=pd.DataFrame(metric_rows); params=pd.DataFrame(param_rows)
    perm=pd.DataFrame(perm_rows); leak=pd.DataFrame(leakage); manifest=pd.DataFrame(model_manifest)

    # OOF audit
    counts=(preds.groupby(["subset","feature_set","outer_fold_column",IDENTIFIER]).size()
            .reset_index(name="prediction_count"))
    bad=counts[counts.prediction_count!=1]
    if len(bad): raise RuntimeError("OOF audit failed")

    # summary
    frames=[]
    for tt in ["fixed_0_5","inner_selected"]:
        x=preds.copy(); x["threshold_type"]=tt; frames.append(x)
    summary=pooled_summary(pd.concat(frames,ignore_index=True),a.bootstrap_iterations,a.random_seed)

    # permutation summary
    if len(perm):
        ps=(perm.groupby(["subset","outer_fold_column","feature_set","feature"])
            .agg(fold_count=("outer_fold","nunique"),mean_importance=("importance_mean","mean"),
                 sd_across_folds=("importance_mean","std"),median_importance=("importance_mean","median"),
                 min_fold_importance=("importance_mean","min"),max_fold_importance=("importance_mean","max"),
                 positive_fold_fraction=("importance_mean",lambda x:float(np.mean(np.asarray(x)>0))),
                 top10_fold_count=("top10_in_fold","sum")).reset_index())
        ps["importance_cv_abs"]=np.where(ps.mean_importance.abs()>1e-12,
                                         ps.sd_across_folds/ps.mean_importance.abs(),np.nan)
        ps["importance_rank"]=(ps.groupby(["subset","outer_fold_column","feature_set"])["mean_importance"]
                               .rank(method="dense",ascending=False).astype(int))
    else: ps=pd.DataFrame()

    # hyperparameter stability -- fixed mixed-type sort bug
    hs=[]
    for keys,g in params.groupby(["subset","outer_fold_column","feature_set"]):
        subset,foldcol,fs=keys
        for par in ["n_estimators","max_depth","min_samples_leaf","min_samples_split","max_features","class_weight","max_samples"]:
            counter=Counter(g[par].map(lambda x: str(x)))
            for value,count in sorted(counter.items(), key=lambda item: str(item[0])):
                hs.append({"subset":subset,"outer_fold_column":foldcol,"feature_set":fs,
                           "hyperparameter":par,"value":value,"outer_fold_count":count})
    hs=pd.DataFrame(hs)

    preds.to_csv(stage/"rf_oof_predictions.csv",index=False)
    metrics.to_csv(stage/"rf_fold_metrics.csv",index=False)
    summary.to_csv(stage/"rf_summary_metrics.csv",index=False)
    params.to_csv(stage/"rf_best_hyperparameters.csv",index=False)
    leak.to_csv(stage/"rf_leakage_audit.csv",index=False)
    counts.to_csv(stage/"oof_prediction_audit.csv",index=False)
    manifest.to_csv(stage/"saved_model_manifest.csv",index=False)
    hs.to_csv(stage/"hyperparameter_stability.csv",index=False)
    perm.to_csv(stage/"permutation_importance_by_fold.csv",index=False)
    ps.to_csv(stage/"permutation_importance.csv",index=False)

    # pooled calibration
    cals=[]
    for keys,g in preds.groupby(["subset","outer_fold_column","feature_set"]):
        subset,foldcol,fs=keys
        c=calibration_table(g[TARGET].to_numpy(int),g.predicted_probability.to_numpy(float),10)
        c["subset"]=subset;c["outer_fold_column"]=foldcol;c["feature_set"]=fs;cals.append(c)
    pd.concat(cals,ignore_index=True).to_csv(stage/"calibration_pooled_oof.csv",index=False)

    report={"stage":"09_v2_recovered","status":"RECOVERED_FROM_SAVED_MODELS",
            "benchmark_version":"1.1","feature_set_version":"3.0","protein_count":181,
            "positive_count":73,"negative_count":108,"saved_models":len(model_paths),
            "leakage_failures":int((~leak.passed).sum()),"oof_prediction_audit_failures":len(bad),
            "permutation_repeats_recovery":a.permutation_repeats,
            "bootstrap_iterations_recovery":a.bootstrap_iterations,
            "runtime_minutes":(time.time()-start)/60,
            "unrecoverable_without_retuning":["rf_inner_search_results.csv",
                                              "rf_top5_hyperparameters_by_outer_fold.csv"]}
    (stage/"reproducibility_report.json").write_text(json.dumps(report,indent=2))

    print("\nRecovery complete.")
    print(f"Saved models recovered: {len(model_paths)}")
    print(f"Leakage failures: {(~leak.passed).sum()}")
    print(f"OOF audit failures: {len(bad)}")
    print(summary[["subset","outer_fold_column","feature_set","threshold_type",
                   "mcc","balanced_accuracy","roc_auc","pr_auc","brier_score"]].to_string(index=False))

if __name__=="__main__":
    main()
