"""Build full multimodal folds, with the original external test kept fixed."""
from pathlib import Path
from collections import Counter
import json
from sklearn.model_selection import StratifiedKFold
from clusterprep.splits import multilabel_stratum
from clusterprep.matched_cohort import counts

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent

def main():
    original=json.loads((ROOT/'outputs/optional_chandra_phase6/expanded_manifest.json').read_text())
    info=original['samples']
    test=json.loads((ROOT/'clusterprep/configs/phase7b/fold_0.json').read_text())['test']
    pool=sorted(set(info)-set(test))
    assert len(info)==154 and len(test)==13
    assert all(s['radio_paths'].get(p) for s in info.values() for p in ('RAW','T50kpcSUB'))
    manifest=dict(original,schema_version=2,population='full_multimodal',external_test=test,
                  eligibility_source='phase6 validated radio/physical/annotation inventory; CHANDRA optional')
    (OUT/'cohort_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    folds=[]
    strata=[multilabel_stratum(info[k]) for k in pool]
    for i,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(pool,strata)):
        f=dict(fold=i,seed=42,candidate_policy='positive',train=[pool[j] for j in tr],
               validation=[pool[j] for j in va],test=test)
        f['distributions']={n:counts(info,f[n]) for n in ('train','validation','test')}
        folds.append(f)
        (OUT/f'fold_{i}.json').write_text(json.dumps(f,indent=2)+'\n')
    assert Counter(k for f in folds for k in f['validation'])==Counter(pool)
    print([(len(f['train']),len(f['validation']),len(f['test'])) for f in folds])
if __name__=='__main__':main()
