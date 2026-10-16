#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

AA20 = "ACDEFGHIKLMNPQRSTVWY"
KD = {"I":4.5,"V":4.2,"L":3.8,"F":2.8,"C":2.5,"M":1.9,"A":1.8,"G":-0.4,"T":-0.7,"S":-0.8,"W":-0.9,"Y":-1.3,"P":-1.6,"H":-3.2,"E":-3.5,"Q":-3.5,"D":-3.5,"N":-3.5,"K":-3.9,"R":-4.5}
MASS = {"A":89.09,"R":174.20,"N":132.12,"D":133.10,"C":121.15,"E":147.13,"Q":146.15,"G":75.07,"H":155.16,"I":131.17,"L":131.17,"K":146.19,"M":149.21,"F":165.19,"P":115.13,"S":105.09,"T":119.12,"W":204.23,"Y":181.19,"V":117.15}
PKA = {"D":3.9,"E":4.1,"H":6.0,"C":8.3,"Y":10.1,"K":10.5,"R":12.5}

IDENTIFIERS = {"jwid","minami_uniprot_accession","current_uniprot_accession","gene_minami","accession","gene","b_number","pdb_path","fasta_path"}
TARGETS = {"cohort_label","binary_label"}
PRIOR = {"minami_length","minami_structure_source","minami_relative_contact_order","minami_rw_prediction","minami_rwrr_prediction"}
GROUPING = {"sccs"}
QC = {"fasta_structure_length_match","chain_count","chain_break_count","mean_consecutive_ca_distance","max_consecutive_ca_distance","ss_assignment_count","ss_note","mean_plddt","median_plddt","minimum_plddt","maximum_plddt","plddt_std","fraction_plddt_below_50","fraction_plddt_below_70","fraction_plddt_at_least_90"}
CONFIG = {"contact_cutoff_angstrom","minimum_contact_sequence_separation","long_range_sequence_separation"}

REDUNDANCY_DROP = {
    "fasta_length":"duplicate of structure_ca_length after QC",
    "absolute_contact_order":"identical to mean_contact_sequence_separation",
    "mean_contact_degree":"deterministic transform of ca_contact_density_per_residue",
    "compactness_n_over_rg_cubed":"inverse transform of normalized radius of gyration",
    "principal_variance_1":"redundant global size/shape measure",
    "principal_variance_2":"redundant global size/shape measure",
    "principal_variance_3":"redundant global size/shape measure",
    "eligible_ca_pair_count":"deterministic function of length",
    "ca_contact_count":"raw size-dependent count",
    "local_contact_count":"raw count; retain fraction",
    "medium_range_contact_count":"raw count; retain fraction",
    "long_range_contact_count":"raw count; retain fraction",
    "minimum_cut_cross_contacts":"raw count; retain normalized density",
    "best_cut_position":"raw position; retain normalized fraction",
    "max_contact_sequence_separation":"strongly constrained by length",
    "maximum_ca_span":"redundant global size metric",
    "radius_of_gyration":"raw size metric; retain normalized Rg",
    "median_contact_degree":"redundant with density summaries",
    "max_contact_degree":"unstable extreme statistic",
}

MINIMAL = ["structure_ca_length","relative_contact_order","long_range_contact_fraction","ca_contact_density_per_residue","radius_of_gyration_normalized_n_cuberoot","asphericity","ss_helix_fraction","ss_sheet_fraction","minimum_cut_cross_contact_density"]
CORE = ["structure_ca_length","relative_contact_order","mean_contact_sequence_separation","median_contact_sequence_separation","long_range_contact_fraction","medium_range_contact_fraction","local_contact_fraction","ca_contact_density_per_residue","ca_contact_pair_fraction","zero_contact_residue_fraction","radius_of_gyration_normalized_n_cuberoot","axis_ratio_1_to_2","axis_ratio_1_to_3","asphericity","ss_helix_fraction","ss_sheet_fraction","ss_coil_fraction","minimum_cut_cross_contact_density","best_cut_fraction"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--merge-audit", default=None)
    p.add_argument("--output-dir", default="data/frozen/benchmark_v1_0")
    p.add_argument("--bootstrap-iterations", type=int, default=2000)
    p.add_argument("--random-seed", type=int, default=42)
    return p.parse_args()


def read_fasta(path: Path) -> str:
    seq = "".join(line.strip() for line in path.read_text(errors="replace").splitlines() if line.strip() and not line.startswith(">" )).upper()
    if not seq:
        raise ValueError(f"empty FASTA: {path}")
    bad = sorted(set(seq)-set(AA20))
    if bad:
        raise ValueError(f"non-standard residues {bad}: {path}")
    return seq


def sliding_max(values, window):
    if len(values) < window:
        return float(np.mean(values))
    return float(max(np.mean(values[i:i+window]) for i in range(len(values)-window+1)))


def net_charge(seq, ph=7.0):
    positive = 1/(1+10**(ph-8.0))
    negative = 1/(1+10**(3.1-ph))
    for aa in "HKR": positive += seq.count(aa)/(1+10**(ph-PKA[aa]))
    for aa in "DECY": negative += seq.count(aa)/(1+10**(PKA[aa]-ph))
    return float(positive-negative)


def seq_features(seq):
    n=len(seq); counts={aa:seq.count(aa) for aa in AA20}; kd=[KD[a] for a in seq]
    probs=np.array([counts[a]/n for a in AA20]); probs=probs[probs>0]
    result={
        "seq_length":n,
        "seq_log_length":float(np.log(n)),
        "seq_estimated_molecular_weight":float(sum(MASS[a] for a in seq)-18.015*max(0,n-1)),
        "seq_mean_kd_hydrophobicity":float(np.mean(kd)),
        "seq_kd_hydrophobicity_std":float(np.std(kd)),
        "seq_max_kd_window_9":sliding_max(kd,9),
        "seq_max_kd_window_19":sliding_max(kd,19),
        "seq_hydrophobic_fraction":sum(counts[a] for a in "AVILMFWYC")/n,
        "seq_aromatic_fraction":sum(counts[a] for a in "FWY")/n,
        "seq_aliphatic_fraction":sum(counts[a] for a in "AILV")/n,
        "seq_positive_fraction":sum(counts[a] for a in "KRH")/n,
        "seq_negative_fraction":sum(counts[a] for a in "DE")/n,
        "seq_charge_density":sum(counts[a] for a in "KRHDE")/n,
        "seq_net_charge_proxy_ph7":net_charge(seq),
        "seq_glycine_fraction":counts["G"]/n,
        "seq_proline_fraction":counts["P"]/n,
        "seq_cysteine_fraction":counts["C"]/n,
        "seq_entropy":float(-np.sum(probs*np.log2(probs))),
    }
    for aa in AA20: result[f"seq_fraction_{aa}"]=counts[aa]/n
    return result


def bh(pvals):
    p=pd.to_numeric(pvals,errors="coerce"); valid=p.notna(); out=pd.Series(np.nan,index=p.index)
    vals=p[valid].to_numpy()
    if len(vals)==0: return out
    order=np.argsort(vals); ranked=vals[order]; n=len(vals)
    corr=np.minimum.accumulate((ranked*n/np.arange(1,n+1))[::-1])[::-1]
    corr=np.clip(corr,0,1); restored=np.empty(n); restored[order]=corr; out.loc[valid]=restored
    return out


def add_labels(df):
    out=df.copy(); out["label_source"]="Minami_2023_Table_S1"
    out["label_evidence_type"]=np.where(out.binary_label==1,"published_GroE_obligate_positive_benchmark","published_experimentally_defined_negative_benchmark")
    out["label_confidence"]=np.where(out.binary_label==1,"high_confidence_positive","high_confidence_negative")
    out["source_class_summary"]=np.where(out.binary_label==1,"Minami positive; derived from Fujiwara/Niwa obligate set","Minami negative comparison sample")
    out["label_notes"]=np.where(out.binary_label==1,"Positive label inherited from published Minami obligate-substrate benchmark.","Negative label inherited from Minami Table S1 negative sample set.")
    return out


def audit_sequences(df):
    audit=[]; feats=[]
    for _,r in df.iterrows():
        acc=str(r.current_uniprot_accession).strip(); path=Path(str(r.fasta_path))
        try: seq=read_fasta(path); status="valid"; err=""
        except Exception as exc: seq=""; status="failed"; err=f"{type(exc).__name__}: {exc}"
        audit.append({"current_uniprot_accession":acc,"jwid":r.jwid,"binary_label":r.binary_label,"fasta_path":str(path),"sequence_status":status,"sequence_length":len(seq) if seq else np.nan,"sequence_sha256":hashlib.sha256(seq.encode()).hexdigest() if seq else "","sequence_error":err})
        if seq:
            f=seq_features(seq); f["current_uniprot_accession"]=acc; feats.append(f)
    adf=pd.DataFrame(audit); dups=[]
    for h,g in adf[adf.sequence_status=="valid"].groupby("sequence_sha256"):
        if len(g)>1:
            labels=sorted(g.binary_label.unique().tolist())
            dups.append({"sequence_sha256":h,"protein_count":len(g),"accessions":";".join(g.current_uniprot_accession.astype(str)),"jwid_values":";".join(g.jwid.astype(str)),"labels":";".join(map(str,labels)),"conflicting_labels":len(labels)>1})
    return adf,pd.DataFrame(dups),pd.DataFrame(feats)


def bootstrap_ci(y,scores,func,iters,rng):
    observed=float(func(y,scores)); vals=[]; n=len(y)
    for _ in range(iters):
        idx=rng.integers(0,n,n); yb=y[idx]
        if len(np.unique(yb))<2: continue
        vals.append(float(func(yb,scores[idx])))
    lo,hi=np.percentile(vals,[2.5,97.5]) if vals else (np.nan,np.nan)
    return observed,float(lo),float(hi)


def univariate_auc(df,features,iters,seed):
    rng=np.random.default_rng(seed); rows=[]
    for feature in features:
        values=pd.to_numeric(df[feature],errors="coerce"); valid=values.notna(); y=df.loc[valid,"binary_label"].to_numpy(int); s=values[valid].to_numpy(float)
        if len(s)<10 or len(np.unique(y))<2: continue
        if roc_auc_score(y,s)<0.5: s=-s; direction="lower_values_predict_positive"
        else: direction="higher_values_predict_positive"
        roc,rl,rh=bootstrap_ci(y,s,roc_auc_score,iters,rng); pr,pl,ph=bootstrap_ci(y,s,average_precision_score,iters,rng)
        rows.append({"feature":feature,"n":int(valid.sum()),"direction":direction,"direction_adjusted_roc_auc":roc,"roc_auc_ci_low":rl,"roc_auc_ci_high":rh,"pr_auc":pr,"pr_auc_ci_low":pl,"pr_auc_ci_high":ph})
    return pd.DataFrame(rows).sort_values("direction_adjusted_roc_auc",ascending=False)


def af_audit(df):
    rows=[]
    for f in ["mean_plddt","median_plddt","minimum_plddt","plddt_std","fraction_plddt_below_50","fraction_plddt_below_70","fraction_plddt_at_least_90"]:
        if f not in df: continue
        p=pd.to_numeric(df.loc[df.binary_label==1,f],errors="coerce").dropna(); n=pd.to_numeric(df.loc[df.binary_label==0,f],errors="coerce").dropna(); u=stats.mannwhitneyu(p,n,alternative="two-sided")
        rows.append({"feature":f,"positive_median":float(p.median()),"negative_median":float(n.median()),"median_difference":float(p.median()-n.median()),"mann_whitney_p":float(u.pvalue)})
    out=pd.DataFrame(rows)
    if not out.empty: out["fdr_q"]=bh(out.mann_whitney_p)
    return out


def structural_features(df):
    excluded=IDENTIFIERS|TARGETS|PRIOR|GROUPING|QC|CONFIG|{"label_source","label_evidence_type","label_confidence","source_class_summary","label_notes"}
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True)>1 and not c.startswith("seq_") and c not in REDUNDANCY_DROP]


def sha256_file(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def main():
    a=parse_args(); inp=Path(a.input); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    if not inp.exists(): print(f"ERROR: missing {inp}",file=sys.stderr); return 2
    df=pd.read_csv(inp)
    if len(df)!=181: print(f"ERROR: expected 181 rows, got {len(df)}",file=sys.stderr); return 2
    df=add_labels(df)
    seq_audit,dups,seqf=audit_sequences(df); seq_audit.to_csv(out/"sequence_integrity_audit.csv",index=False); dups.to_csv(out/"duplicate_sequence_audit.csv",index=False)
    failed=int((seq_audit.sequence_status!="valid").sum()); conflicts=int(dups.conflicting_labels.sum()) if not dups.empty else 0
    if failed or conflicts: print(f"ERROR: failed_sequences={failed}, conflicting_duplicates={conflicts}",file=sys.stderr); return 1
    df=df.merge(seqf,on="current_uniprot_accession",how="left",validate="one_to_one")
    broad=structural_features(df); minimal=[f for f in MINIMAL if f in df]; core=[f for f in CORE if f in df and f not in REDUNDANCY_DROP]; seqcols=[c for c in df if c.startswith("seq_")]
    meta=["current_uniprot_accession","gene_minami","jwid","binary_label","cohort_label","label_source","label_evidence_type","label_confidence","source_class_summary","label_notes","sccs"]
    outputs={
      "benchmark_v1.0.csv":meta+seqcols+broad,
      "benchmark_v1.0_structural_minimal.csv":meta+minimal,
      "benchmark_v1.0_structural_core.csv":meta+core,
      "benchmark_v1.0_structural_broad.csv":meta+broad,
      "benchmark_v1.0_sequence_only.csv":meta+seqcols,
      "benchmark_v1.0_sequence_plus_structure_core.csv":meta+seqcols+core,
    }
    for name,cols in outputs.items(): df[cols].to_csv(out/name,index=False)
    univariate_auc(df,seqcols+broad,a.bootstrap_iterations,a.random_seed).to_csv(out/"univariate_auc_statistics.csv",index=False)
    af_audit(df).to_csv(out/"alphafold_confidence_audit.csv",index=False)
    frozen=pd.read_csv(out/"benchmark_v1.0.csv"); dictionary=[]
    for c in frozen:
        role="target" if c=="binary_label" else "grouping_only" if c=="sccs" else "metadata" if c in meta else "predictor"
        sets=[]
        if c in minimal: sets.append("structural_minimal")
        if c in core: sets.append("structural_core")
        if c in broad: sets.append("structural_broad")
        if c in seqcols: sets.append("sequence_only")
        dictionary.append({"column":c,"role":role,"dtype":str(frozen[c].dtype),"feature_sets":";".join(sets),"missing_count":int(frozen[c].isna().sum()),"unique_nonmissing":int(frozen[c].nunique(dropna=True))})
    pd.DataFrame(dictionary).to_csv(out/"benchmark_v1.0_data_dictionary.csv",index=False)
    if a.merge_audit and Path(a.merge_audit).exists():
        mad=pd.read_csv(a.merge_audit); mad[mad.feature_status.astype(str).str.lower()!="found"].to_csv(out/"benchmark_v1.0_exclusions.csv",index=False)
    else: pd.DataFrame().to_csv(out/"benchmark_v1.0_exclusions.csv",index=False)
    yaml=["version: '1.0'","target:","  column: binary_label","  positive: 1","  negative: 0","contact_definition:","  atom: CA","  cutoff_angstrom: 8.0","  minimum_sequence_separation: 3","  long_range_sequence_separation: 24","feature_sets:"]
    for name,features in [("structural_minimal",minimal),("structural_core",core),("structural_broad",broad),("sequence_only",seqcols)]:
        yaml.append(f"  {name}:"); yaml.extend(f"    - {f}" for f in features)
    yaml.append("redundancy_exclusions:"); yaml.extend(f"  {k}: {json.dumps(v)}" for k,v in REDUNDANCY_DROP.items()); (out/"feature_definitions_v1.0.yaml").write_text("\n".join(yaml)+"\n")
    plan=f'''# Analysis Plan v1.0\n\n## Question\nDo global structural topology features predict obligate GroEL/GroES dependence beyond sequence-derived properties and fold-family enrichment?\n\n## Frozen benchmark\n- 181 proteins: 64 positive and 117 negative\n- 2 documented historical-model exclusions\n- 368 neutral proteins reserved for post-finalization scoring\n\n## Primary metric\nMatthews correlation coefficient (MCC).\n\n## Feature groups\n- structural_minimal: {len(minimal)}\n- structural_core: {len(core)}\n- structural_broad: {len(broad)}\n- sequence_only: {len(seqcols)}\n- sequence_plus_structure_core\n\n## Evaluation rules\n1. No ordinary random train/test split as primary evaluation.\n2. Homologs stay in the same fold.\n3. SCCS is grouping-only and never a predictor.\n4. Minami RW/RWRR and published RCO are never predictors.\n5. Tuning occurs only inside outer training folds.\n6. Preprocessing is fitted only on training folds.\n7. MCC is the primary endpoint.\n\n## Fold-bias experiments\n1. Full benchmark with homology-grouped validation.\n2. Non-TIM-barrel subset.\n3. TIM-barrel-only subset.\n4. Fold-held-out evaluation using SCCS where feasible.\n\n## Models\nMajority, length-only, contact-order-only, regularized logistic regression, and class-weighted random forest.\n\n## Neutral cohort\nNeutral proteins are excluded from training and threshold selection and scored only after the evaluation protocol is finalized.\n'''
    (out/"analysis_plan_v1.0.md").write_text(plan)
    summary={"version":"1.0","rows":181,"positive_count":64,"negative_count":117,"failed_sequences":failed,"duplicate_sequence_groups":int(len(dups)),"conflicting_duplicate_groups":conflicts,"structural_minimal_features":len(minimal),"structural_core_features":len(core),"structural_broad_features":len(broad),"sequence_features":len(seqcols),"redundancy_exclusions":REDUNDANCY_DROP}; (out/"freeze_summary.json").write_text(json.dumps(summary,indent=2))
    checks=[]
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name!="dataset_sha256.txt": checks.append(f"{sha256_file(p)}  {p.name}")
    (out/"dataset_sha256.txt").write_text("\n".join(checks)+"\n")
    print("Stage 06B complete"); print(f"  Rows: {len(df)}"); print(f"  Sequence features: {len(seqcols)}"); print(f"  Structural minimal/core: {len(minimal)} / {len(core)}"); print(f"  Structural broad: {len(broad)}"); print(f"  Duplicate sequence groups: {len(dups)}"); print(f"  Conflicting labels: {conflicts}"); print(f"  Output: {out}")
    return 0

if __name__=="__main__": raise SystemExit(main())
