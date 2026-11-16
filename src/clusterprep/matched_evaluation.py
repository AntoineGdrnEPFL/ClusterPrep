"""Evaluate frozen matched models on the reserved external test. Never trains."""
import argparse
from dataclasses import replace
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import ClusterDataset
from .labels import task_target
from .matched_runner import DEFAULT_SOURCE,read,device_name
from .models import build_model
from .training import ExperimentConfig,Trainer,compute_task_counts,_write_predictions
from .matched_cohort import write_json


def prepare(run_dir,source=DEFAULT_SOURCE):
    run_dir=Path(run_dir).resolve();source=Path(source).resolve()
    if read(run_dir/'status.json')['status']!='completed':raise ValueError('Training not completed')
    splits=read(run_dir/'splits.json');metadata=read(run_dir/'config.json')
    config=ExperimentConfig(**metadata['config'])
    if config.radio_preprocessing!='matched_resolution' or config.task!='rr_rh':
        raise ValueError('Expected matched RR/RH experiment')
    for name,digest in splits['split_config']['provenance'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Manifest changed since training: {name}')
    manifest=read(source/'cohort_manifest.json')
    keys=splits['reserved_external_test']
    if not keys or len(keys)!=len(set(keys)) or keys!=manifest['external_test']:
        raise ValueError('External test identity mismatch')
    used=splits['train']+splits['validation']
    if set(keys)&set(used):raise ValueError('Train/validation/test overlap')
    info=manifest['samples']
    used_ids={splits['samples'][k]['cluster_id'] for k in used}
    if {info[k]['cluster_id'] for k in keys}&used_ids:raise ValueError('Cluster ID leakage')
    normalization=read(run_dir/'normalization.json')
    if set(normalization['train_keys'])!=set(splits['train']):raise ValueError('Invalid normalization training IDs')
    if normalization['bounds']!=config.normalization_bounds:raise ValueError('Normalization mismatch')
    if any(config.normalization.get(m,{}).get('scope')!='train' or m not in config.normalization_bounds for m in ('radio','xray')):
        raise ValueError('Frozen train normalization required')
    thresholds=read(run_dir/'thresholds.json')['thresholds']
    if set(thresholds)!={'RR','RH'} or not all(np.isfinite(v) and 0<=v<=1 for v in thresholds.values()):
        raise ValueError('Invalid saved thresholds')
    if not (run_dir/'best.pt').is_file():raise ValueError('Missing best checkpoint')
    return dict(source=source,run_dir=run_dir,config=config,splits=splits,info=info,keys=keys,thresholds=thresholds)


def evaluate(prepared,output,device='auto'):
    p=prepared;output=Path(output)
    if output.exists():raise FileExistsError(output)
    device=device_name(device)
    checkpoint=torch.load(p['run_dir']/'best.pt',map_location='cpu',weights_only=True)
    if checkpoint['config']['normalization_bounds']!=p['config'].normalization_bounds:
        raise ValueError('Checkpoint normalization mismatch')
    if checkpoint.get('evaluation',{}).get('thresholds')!=p['thresholds']:
        raise ValueError('Checkpoint thresholds mismatch')
    config=replace(p['config'],device=device,use_compile=False,use_scheduler=False,
                   thresholds=p['thresholds'],augment=False,verbose=False)
    model=build_model(config,2)
    model.load_state_dict(checkpoint['model_state_dict'])
    ds=ClusterDataset(p['info'],p['keys'],['RR','RH'],mode=config.mode,task=config.task,
        candidate_policy=config.candidate_policy,modalities=config.modalities,
        missing_modality_policy=config.missing_modality_policy,include_masks=config.include_masks,
        size=config.size,augment=False,training=False,radio_source=config.radio_source,
        radio_preprocessing=config.radio_preprocessing,matched_resolution=config.matched_resolution,
        matched_records=config.matched_records,matched_cache_dir=config.matched_cache_dir,
        processed_dir=config.processed_dir,normalization=config.normalization,
        normalization_bounds=config.normalization_bounds)
    if list(ds.keys)!=p['keys']:raise ValueError('Dataset filtered external test members')
    # Reuse historical masked evaluation, including loss weights from TRAIN only.
    training_info=p['splits']['samples'];train=p['splits']['train']
    known=np.asarray([task_target(training_info[k],'rr_rh',config.candidate_policy)[1] for k in train]).sum(0)
    engine=Trainer(model,config,('RR','RH'),compute_task_counts(training_info,train,'rr_rh',config.candidate_policy),
                   n_train=known,device=device,steps_per_epoch=1)
    metrics,rows=engine.evaluate(DataLoader(ds,batch_size=config.batch_size,shuffle=False,num_workers=0))
    output.mkdir(parents=True,exist_ok=False)
    _write_predictions(output/'test_predictions.csv',rows,('RR','RH'),task='rr_rh')
    write_json(output/'test_predictions.json',rows)
    write_json(output/'test_metrics.json',metrics)
    write_json(output/'evaluation.json',dict(run_dir=str(p['run_dir']),checkpoint='best.pt',
        checkpoint_sha256=hashlib.sha256((p['run_dir']/'best.pt').read_bytes()).hexdigest(),
        cluster_ids=p['keys'],thresholds=p['thresholds'],normalization_bounds=config.normalization_bounds,
        normalization_train_keys=train,device=device,training_performed=False,thresholds_refitted=False))
    print(f"Evaluated {p['run_dir'].name}: {len(rows)} clusters -> {output}",flush=True)


def export_analysis(prepared, output):
    """Attach historical analysis artifacts to an existing frozen evaluation.

    Does not load a model or images and never modifies the training directory.
    """
    import shutil
    p = prepared
    output = Path(output)
    evaluation = read(output / 'evaluation.json')
    if Path(evaluation['run_dir']).resolve() != p['run_dir']:
        raise ValueError('Evaluation training directory mismatch')
    if evaluation['cluster_ids'] != p['keys'] or evaluation['thresholds'] != p['thresholds']:
        raise ValueError('Evaluation identity/threshold mismatch')
    if evaluation['normalization_bounds'] != p['config'].normalization_bounds:
        raise ValueError('Evaluation normalization mismatch')
    if evaluation['checkpoint_sha256'] != hashlib.sha256((p['run_dir'] / 'best.pt').read_bytes()).hexdigest():
        raise ValueError('Evaluation checkpoint mismatch')
    if not (output / 'test_predictions.csv').is_file():
        raise ValueError('Missing evaluated predictions')
    metrics = read(p['run_dir'] / 'metrics.json')
    metrics['test'] = read(output / 'test_metrics.json')
    metrics['test_evaluated'] = True
    splits = dict(p['splits'])
    splits['original_split_config'] = splits['split_config']
    # Common plan identity permits grouping folds without discarding provenance.
    plan_files = [p['source'] / f'fold_{i}.json' for i in range(5)]
    splits['split_config'] = {
        'n_folds': 5, 'fold_index': splits['original_split_config']['fold'],
        'provenance': {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                       for f in [p['source'] / 'cohort_manifest.json', *plan_files]},
    }
    splits['test'] = list(p['keys'])
    splits['test_evaluated'] = True
    splits['samples'] = {**splits['samples'], **{k: p['info'][k] for k in p['keys']}}
    for name in ('config.json', 'history.csv', 'history.jsonl', 'thresholds.json',
                 'normalization.json', 'validation_predictions.csv', 'validation_metrics.json'):
        source = p['run_dir'] / name
        if source.is_file():
            shutil.copy2(source, output / name)
    write_json(output / 'metrics.json', metrics)
    write_json(output / 'splits.json', splits)
    write_json(output / 'status.json', {'status': 'completed', 'test_evaluated': True,
                                      'analysis_source': str(p['run_dir'])})
    print(f'Analysis artifacts ready: {output}', flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run-dir',type=Path)
    group.add_argument('--runs-root',type=Path)
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--output-root',type=Path,default=Path('outputs/phase7b_external_test'))
    parser.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--prepare-analysis', action='store_true',
                        help='Attach training metadata to existing evaluations; no inference or training')
    args=parser.parse_args(argv)
    dirs=[args.run_dir] if args.run_dir else sorted(p.parent for p in args.runs_root.glob('*/*/*/best.pt'))
    if not dirs:parser.error('No model checkpoints found')
    plans=[prepare(d,args.source) for d in dirs]
    outputs=[args.output_root/p['run_dir'].name for p in plans]
    if len(set(outputs))!=len(outputs):raise ValueError('Duplicate output names')
    if not args.dry_run and not args.prepare_analysis:
        for path in outputs:
            if path.exists():raise FileExistsError(f'Existing evaluation: {path}; choose a new --output-root')
    for p,out in zip(plans,outputs):
        if args.dry_run:print(f"READY {p['run_dir'].name}: {len(p['keys'])} reserved clusters, frozen normalization/thresholds")
        elif args.prepare_analysis:export_analysis(p,out)
        else:
            evaluate(p,out,args.device)
            export_analysis(p,out)

if __name__=='__main__':main()
