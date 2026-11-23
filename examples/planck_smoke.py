"""Eight one-epoch real-data smoke runs (4 train / 2 validation clusters).

Run from the workspace root; writes only outputs/planck_smoke and caches.
The reserved test split is never loaded. This is not a scientific benchmark.
"""
from dataclasses import replace
from pathlib import Path
import json
import torch
from clusterprep.matched_runner import load_plan
from clusterprep.training import ExperimentConfig,run_experiment
from clusterprep.dataset import ClusterDataset
from clusterprep.planck_visualization import render

torch.set_num_threads(2)
plan=load_plan('A',0)
info=plan['samples']
# Preserve development/test boundaries; take both positive and negative examples.
pool=plan['partition']['train']
train=[]
for label in ['RR','RH','NDE']:
    key=next(k for k in pool if label in info[k]['labels'] and k not in train)
    train.append(key)
train.append(next(k for k in pool if not info[k].get('chandra_path') and k not in train))
val=plan['partition']['validation'][:2]
print([(k,info[k]['labels'],bool(info[k].get('chandra_path'))) for k in train+val],flush=True)
results=[]
for geometry in ['historical','matched_resolution']:
 for experiment in ['A','B','C','D']:
  path=Path(f'clusterprep/configs/planck/{geometry}/{experiment}/fold_0.json')
  config=ExperimentConfig(**json.loads(path.read_text())['experiments'][0])
  config=replace(config,name=f'smoke_{geometry}_{experiment}',size=32,device='cpu',epochs=1,
      batch_size=2,evaluate_test=False,augment=False,use_scheduler=False,use_mixup=False,drop_last_train=False,
      class_weight='none',use_class_weights=False,verbose=False,
      matched_cache_dir=plan['cache_root'] if geometry=='matched_resolution' else None)
  run=run_experiment(info,config,'outputs/planck_smoke',class_names=('RR','RH'),
      splits=(train,val,plan['partition']['test']),split_metadata={'fold':0,'provenance':plan['provenance']})
  results.append(dict(geometry=geometry,experiment=experiment,run=str(run)))
  print(results[-1],flush=True)
  if experiment=='C' and geometry=='historical':
   key=next(k for k in train if info[k].get('chandra_path'))
   print(render(info[key],config.planck,'outputs/planck_smoke/alignment.png',size=32),flush=True)
Path('outputs/planck_smoke/summary.json').write_text(json.dumps(results,indent=2))
