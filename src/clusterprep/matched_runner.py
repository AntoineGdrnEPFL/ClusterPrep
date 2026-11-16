"""Matched-resolution orchestrator using the historical training pipeline."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .matched_cache import canonical_json
from .labels import task_target
from .training import ExperimentConfig, run_experiment

EXPERIMENTS={'A':('RAW','paired'),'B':('T50kpcSUB','paired'),
             'C':('RAW','expanded'),'D':('T50kpcSUB','expanded')}
ROOT=Path(__file__).resolve().parents[3]
DEFAULT_SOURCE=ROOT/'clusterprep/configs/phase7b_multimodal'
CONFIG_ROOT=ROOT/'clusterprep/configs/matched_historical'

def read(path): return json.loads(Path(path).read_text())


def device_name(requested):
    available=torch.backends.mps.is_available()
    if requested=='auto':return 'mps' if available else 'cpu'
    if requested=='mps' and not available:raise ValueError('MPS unavailable; use --device cpu')
    if requested not in ('cpu','mps'):raise ValueError('Device must be auto, cpu or mps')
    return requested


def load_plan(experiment,fold=0,seed=42,source=DEFAULT_SOURCE,mode='multimodal'):
    source=Path(source).resolve()
    if (source/'cohort_manifest.json').exists():
        from .modalities import mode_keys
        manifest=read(source/'cohort_manifest.json'); info=manifest['samples']
        original=read(source/f'fold_{fold}.json')
        if mode not in ('multimodal','pair'):raise ValueError('Expected multimodal or pair mode')
        partition={**original,**{n:mode_keys(info,original[n],mode) for n in ('train','validation','test')}}
        flat=sum((partition[n] for n in ('train','validation','test')),[])
        expected=mode_keys(info,list(info),mode)
        if len(flat)!=len(set(flat)) or set(flat)!=set(expected):raise ValueError('Incomplete or overlapping population')
        if original['test']!=manifest['external_test']:raise ValueError('External test changed')
        if len({info[k]['cluster_id'] for k in flat})!=len(flat):raise ValueError('Duplicate cluster IDs')
        product=EXPERIMENTS[experiment][0]
        selected={k:info[k] for k in expected}
        for k in flat:
            if not selected[k]['radio_paths'].get(product):raise ValueError('Selected radio missing')
            if not np.any(task_target(selected[k],'rr_rh')[1]):raise ValueError('No known label')
        return dict(experiment=experiment,product=product,cohort=mode,mode=mode,fold=fold,seed=seed,
            samples=selected,partition=partition,cache_root=manifest['cache_root'],source=str(source),
            normalization=dict(train_keys=partition['train'],scope='train',fit_at_runtime=True),
            provenance={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in
                        ('cohort_manifest.json',f'fold_{fold}.json')})
    source=Path(source).resolve();product,cohort=EXPERIMENTS[experiment]
    manifest=read(source/'expanded_manifest.json'); folds=read(source/'expanded_folds.json')
    info=manifest['samples']
    partitions=folds['paired_folds'] if cohort=='paired' else folds['folds']
    matches=[p for p in partitions if p['fold']==fold]
    if len(matches)!=1:raise ValueError('Unknown fold')
    partition=matches[0]
    flat=sum((partition[n] for n in ('train','validation','test')),[])
    if len(flat)!=len(set(flat)) or any(k not in info for k in flat):raise ValueError('Split overlap or missing key')
    if len({info[k]['cluster_id'] for k in flat})!=len(flat):raise ValueError('Duplicate cluster IDs')
    original=next(p for p in folds['paired_folds'] if p['fold']==fold)
    if any(partition[n]!=original[n] for n in ('validation','test')):raise ValueError('External holdouts changed')
    expected=original['train']+(folds['additional_ids'] if cohort=='expanded' else [])
    if partition['train']!=expected:raise ValueError('Unexpected training membership')
    fitted=read(source/'fold_normalization.json')[cohort][str(fold)]
    if set(fitted['train_keys'])!=set(partition['train']) or len(fitted['train_keys'])!=len(partition['train']):
        raise ValueError('Normalization not fitted exclusively on this train')
    digest=hashlib.sha256(canonical_json(sorted(partition['train'])).encode()).hexdigest()
    if digest!=fitted['train_keys_sha256']:raise ValueError('Normalization train fingerprint mismatch')
    for product_bounds in fitted['products'].values():
        for bounds in product_bounds.values():
            a=np.asarray(bounds)
            if a.shape!=(1,2) or not np.isfinite(a).all() or a[0,1]<a[0,0]:raise ValueError('Invalid normalization bounds')
    if any(fitted['options'][m].get('scope')!='train' for m in ('radio','xray')):raise ValueError('Train normalization required')
    if fitted['products']['RAW']['xray']!=fitted['products']['T50kpcSUB']['xray']:raise ValueError('Unpaired X-ray normalization')
    for k in partition['train']+partition['validation']:
        _,mask=task_target(info[k],'rr_rh')
        if not np.any(mask):raise ValueError('No known training/validation label')
    return dict(experiment=experiment,product=product,cohort=cohort,fold=fold,seed=seed,
        samples=info,partition=partition,normalization=fitted,cache_root=manifest['cache_root'],
        provenance={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in
            ('expanded_manifest.json','expanded_folds.json','fold_normalization.json')},source=str(source))



def training_configuration(plan, device, max_epochs):
    if max_epochs < 1:
        raise ValueError('max-epochs must be positive')
    fold_config=CONFIG_ROOT/plan['product']/f"fold_{plan['fold']}.json"
    template=read(fold_config if fold_config.is_file() else CONFIG_ROOT/(plan['product']+'.json'))
    if 'experiments' in template:
        template=template['experiments'][0]
    return replace(ExperimentConfig(**template), seed=plan['seed'], device=device,
                   epochs=max_epochs, name=f"{plan['product']}_matched_fold_{plan['fold']}",
                   matched_cache_dir=plan['cache_root'])



def run_config_suite(spec, config_path, *, dry_run=False, quiet=False, processed_dir=None):
    """Generic CLI bridge: explicit curated splits, historical run_experiment."""
    cohort=spec['matched_cohort']
    source=(Path(config_path).resolve().parent/cohort['source']).resolve()
    fold=cohort['fold']
    prepared=[]
    for item in spec['experiments']:
        config=ExperimentConfig(**item)
        if (config.radio_source not in ('RAW','T50kpcSUB') or
            config.radio_preprocessing!='matched_resolution' or config.evaluate_test or
            config.task!='rr_rh' or config.model!='dual_encoder_cnn' or
            config.transfer is not None or config.cross_validation is not None):
            raise ValueError('Matched cohort requires RR/RH DualEncoderCNN without transfer or external test evaluation')
        if any(config.normalization.get(m,{}).get('scope')!='train' for m in ('radio','xray')):
            raise ValueError('Matched cohort requires train-only normalization')
        experiment='A' if config.radio_source=='RAW' else 'B'
        plan=load_plan(experiment,fold,config.seed,source,mode=config.mode)
        if config.mode=='multimodal' and read(source/f'fold_{fold}.json')!=plan['partition']:
            raise ValueError('Individual fold differs from manifest')
        config=replace(config, device=device_name(config.device),
                       verbose=False if quiet else config.verbose,
                       processed_dir=str(processed_dir) if processed_dir else config.processed_dir,
                       matched_cache_dir=plan['cache_root'])
        prepared.append((plan,config))
    if not prepared:raise ValueError('Empty experiments')
    # Validate every configuration before any call that can train.
    for plan,config in prepared:
        partition=plan['partition']
        if dry_run:
            print(json.dumps(dict(product=config.radio_source,fold=fold,
                train=len(partition['train']),validation=len(partition['validation']),
                reserved_test=len(partition['test']),evaluate_test=False)))
            continue
        run_experiment(plan['samples'],config,spec['output_dir'],class_names=('RR','RH'),
            splits=tuple(partition[n] for n in ('train','validation','test')),
            split_metadata=dict(fold=fold,provenance=plan['provenance']))

def training_budget(plan, max_epochs, epoch_mode='full'):
    if epoch_mode != 'full':
        raise ValueError('Unified historical pipeline supports full epochs only')
    config=training_configuration(plan,'cpu',max_epochs)
    from .augmentation import variant_count
    n=len(plan['partition']['train'])*variant_count(config.augmentation.get('zoom'))
    if not config.augment:n=len(plan['partition']['train'])
    steps=n//config.batch_size if config.drop_last_train else (n+config.batch_size-1)//config.batch_size
    return dict(batch_size=config.batch_size,steps_per_epoch=steps,
                max_optimizer_steps=steps*max_epochs,epoch_mode='full',
                train_examples=n,drop_last=config.drop_last_train,
                pipeline='historical_matched_v1')


def run(experiment, fold, seed, device, max_epochs, output, source=DEFAULT_SOURCE, epoch_mode='full'):
    plan=load_plan(experiment,fold,seed,source)
    training_budget(plan,max_epochs,epoch_mode)
    config=training_configuration(plan,device_name(device),max_epochs)
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    request=dict(plan=plan,training=asdict(config),budget=training_budget(plan,max_epochs))
    (output/'request.json').write_text(json.dumps(request,indent=2)+'\n')
    partition=plan['partition']
    result=run_experiment(plan['samples'],config,output,
        splits=tuple(partition[n] for n in ('train','validation','test')),
        class_names=('RR','RH'),split_metadata=dict(fold=fold,provenance=plan['provenance']))
    # Marker written only after successful historical exports.
    (output/'completed.json').write_text(json.dumps(dict(run_dir=str(result),pipeline='historical_matched_v1'))+'\n')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['train','check'])
    parser.add_argument('--experiment',choices=list(EXPERIMENTS),required=True)
    parser.add_argument('--fold',type=int,default=0)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--max-epochs',type=int,default=200)
    parser.add_argument('--epoch-mode',choices=['full'],default='full')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    if args.command=='check':
        plan=load_plan(args.experiment,args.fold,args.seed,args.source)
        print(json.dumps(dict(experiment=args.experiment,device=device_name(args.device),
                              budget=training_budget(plan,args.max_epochs),external_test_used=False)))
    else:
        if args.output is None:parser.error('--output required for train')
        run(args.experiment,args.fold,args.seed,args.device,args.max_epochs,args.output,args.source)

if __name__=='__main__':main()
