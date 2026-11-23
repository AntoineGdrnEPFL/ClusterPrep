"""Prepare fixed-test paired folds and train-only normalization. Never trains."""
from pathlib import Path
from collections import Counter
import warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold
from clusterprep.matched_runner import read, load_plan
from clusterprep.matched_cohort import counts, fit_train_normalization, write_json
from clusterprep.matched_protocol import validate_partition
from clusterprep.splits import multilabel_stratum

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent

def main():
    source=ROOT/'outputs/optional_chandra_phase6'
    original=read(source/'expanded_folds.json')['paired_folds'][0]
    manifest=read(source/'expanded_manifest.json')
    keys=sum((original[n] for n in ('train','validation','test')),[])
    info={k:manifest['samples'][k] for k in sorted(keys)}
    assert len(info)==65 and all(s['chandra_path'] and all(s['radio_paths'].get(p) for p in ('RAW','T50kpcSUB')) for s in info.values())
    test=original['test']
    pool=np.array(sorted(set(keys)-set(test)))
    strata=[multilabel_stratum(info[k]) for k in pool]
    folds=[]
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter('always')
        splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(pool,strata))
    for i,(train,val) in enumerate(splits):
        fold=dict(fold=i,train=pool[train].tolist(),validation=pool[val].tolist(),test=test,
                  candidate_policy='positive',seed=42,missing_subtype_policy='unknown_masked')
        validate_partition(info,fold)
        fold['distributions']={n:counts(info,fold[n]) for n in ('train','validation','test')}
        folds.append(fold)
        write_json(OUT/f'fold_{i}.json',fold)
    assert Counter(k for f in folds for k in f['validation'])==Counter(pool)
    write_json(OUT/'expanded_manifest.json',dict(samples=info,cache_root=manifest['cache_root'],counts=counts(info,keys)))
    write_json(OUT/'expanded_folds.json',dict(paired_folds=folds,folds=folds,additional_ids=[],training_seeds=[42],
        external_test_source='optional_chandra_phase6/expanded_folds.json paired_folds[0].test'))
    fitted={}
    for f in folds:
        print(f"Fitting train normalization fold {f['fold']} (no training)",flush=True)
        fitted[str(f['fold'])]=fit_train_normalization(info,f,manifest['cache_root'])
    write_json(OUT/'fold_normalization.json',dict(paired=fitted))
    for e in 'AB':
        for i in range(5):load_plan(e,i,42,OUT)
    write_json(OUT/'preparation_audit.json',dict(n_cohort=65,n_development=52,n_external_test=13,
        stratification_warnings=[str(w.message) for w in messages],
        strata=dict(Counter(strata)),validation_once_per_cluster=True,normalization_train_only=True,
        distributions=[f['distributions'] for f in folds]))

if __name__=='__main__':main()
