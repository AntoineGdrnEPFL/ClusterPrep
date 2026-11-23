"""Phase-6 extension: optional CHANDRA, fixed external folds, bounded smoke tests."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from .matched_cohort import (annotation_index, canonical_record, cohort_annotation, counts,
    fit_train_normalization, make_dataset, verify_pairs, write_csv, write_json)
from .common_wcs_validation import read_catalog, discover_sources, physical_field
from .matched_cache import MatchedPairStore
from .training import ExperimentConfig, Trainer, compute_task_counts
from .models import DualEncoderCNN
from .labels import task_target


def extend_folds(base_info, expanded, folds):
    if not set(base_info) <= set(expanded):
        raise ValueError('Original paired cohort must remain complete')
    ids=[v['cluster_id'] for v in expanded.values()]
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate cluster identities')
    extra=sorted(set(expanded)-set(base_info))
    result=[]
    for fold in folds:
        old={n:list(fold[n]) for n in ('train','validation','test')}
        flat=sum(old.values(),[])
        if len(flat)!=len(set(flat)) or set(flat)!=set(base_info):
            raise ValueError('Original fold is not a disjoint exhaustive partition')
        partition=dict(fold=fold['fold'],candidate_policy='positive',
            train=old['train']+extra,validation=old['validation'],test=old['test'])
        assert set(partition['train']).isdisjoint(partition['validation']+partition['test'])
        partition['added_train']=extra
        partition['distributions']={n:counts(expanded,partition[n]) for n in old}
        result.append(partition)
    if Counter(k for f in result for k in f['test']) != Counter(list(base_info)):
        raise ValueError('Outer tests must cover original cohort exactly once')
    return result


def inventory(project, output):
    catalog={r['cluster_id']:canonical_record(r) for r in read_catalog(project/'outputs/common_wcs_validation/inputs/lofar_psz2_catalog.fits')}
    annotations,origins=annotation_index(project/'scratch/data/PSZ2/classified')
    store=MatchedPairStore(output/'cache')
    params=dict(pixels_per_beam=3.,min_support=.999,truncate=5.,output_unit='Jy/beam')
    info={}; rows=[]
    for i,cid in enumerate(sorted(set(catalog)|set(annotations))):
        record=catalog.get(cid,dict(cluster_id=cid)); paths=discover_sources(record,project/'scratch/data/PSZ2/fits')
        available={n:Path(paths[n]).is_file() for n in ('RAW','T50kpcSUB','CHANDRA')}
        row=dict(cluster_id=cid,annotations=';'.join(sorted(annotations.get(cid,[]))),
            **{'has_'+n:v for n,v in available.items()},status='excluded',reason='')
        try:
            annotation,_,_=cohort_annotation(annotations.get(cid,[]))
            if not available['RAW'] or not available['T50kpcSUB']:
                raise ValueError('Both radio FITS required')
            if physical_field(record) is None:
                raise ValueError('Physical metadata missing')
            sources={n:paths[n] for n in ('RAW','T50kpcSUB')}
            sources['CHANDRA']=paths['CHANDRA'] if available['CHANDRA'] else None
            pair,cache=store.get(sources,record,128,params)
            if pair.metadata['coverage']['intersection']<=0:
                raise ValueError('No common radio support')
            info[cid]=dict(cluster_id=cid,raw_path=paths['RAW'],radio_paths={n:paths[n] for n in ('RAW','T50kpcSUB')},
                chandra_path=sources['CHANDRA'],physical_record=record,annotation_paths=origins.get(cid,[]),**annotation)
            target,mask=task_target(info[cid],'rr_rh')
            row.update(status='eligible',chandra_status=pair.metadata['chandra_status'],
                RR=int(target[0]) if mask[0] else '?',RH=int(target[1]) if mask[1] else '?',
                RR_known=int(mask[0]),RH_known=int(mask[1]),cache_key=cache['key'],
                **{'coverage_'+n:v for n,v in pair.metadata['coverage'].items()},
                **{n:pair.metadata['target_resolution'][n] for n in ('fwhm_arcsec','fwhm_kpc','fwhm_over_r500')})
        except Exception as exc:
            row['reason']=f'{type(exc).__name__}: {exc}'
        rows.append(row)
        write_csv(output/'expanded_cohort.csv',rows)
        if i%20==0:print(f'{i+1} inspected; {len(info)} eligible',flush=True)
    return info,rows


def budgeted_loader(dataset, seed, batch_size=16, steps_per_epoch=3):
    """Equal optimizer budgets, random draws from each full train, never holdouts.

    Keep this loader alive across epochs so the seeded generator advances.
    Product-paired datasets must have exactly the same ordered keys.
    """
    generator=torch.Generator().manual_seed(seed)
    sampler=RandomSampler(dataset,replacement=True,num_samples=batch_size*steps_per_epoch,generator=generator)
    return DataLoader(dataset,batch_size=batch_size,sampler=sampler,num_workers=0)


def configuration(product,cohort):
    return ExperimentConfig(name=f'{cohort}_{product}',task='rr_rh',mode='pair',model='dual_encoder_cnn',
        radio_source=product,missing_modality_policy='optional',device='cpu',use_compile=False,
        augment=False,drop_last_train=False,threshold_selection='f1',transfer=None,
        model_params=dict(fusion_hidden_dim=64,use_mask=True,use_coverage=True))


def experiment_specifications(output, seeds):
    return [dict(experiment=asdict(configuration(product,cohort)),cohort=cohort,
        radio_preprocessing='matched_resolution',matched_resolution=dict(radio_mask='intersection'),
        normalization_scope='train',training_budget=dict(epochs=200,steps_per_epoch=3,
            batch_size=16,max_optimizer_steps=600,sampling='seeded random with replacement; 48 train draws/epoch',
            loader='optional_chandra_cohort.budgeted_loader',early_stopping='same val_loss/patience rule'),
        folds_file=str(output/'expanded_folds.json'),normalization_file=str(output/'fold_normalization.json'),
        training_seeds=seeds,pipeline_entrypoint='make_dataset + budgeted_loader + Trainer; 3-channel DualEncoderCNN',
        status='prepared_no_full_training_authorized')
        for cohort in ('paired','expanded') for product in ('RAW','T50kpcSUB')]


def smoke(info,partition,cache,fitted,product,cohort,output):
    from dataclasses import replace
    output.mkdir(parents=True,exist_ok=False)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    # Small deterministic diagnostic subset; all weights/bounds still use the full fold train.
    missing=[k for k in partition['train'] if info[k]['chandra_path'] is None]
    present=[k for k in partition['train'] if info[k]['chandra_path'] is not None]
    keys=(missing[:4]+present)[:16]
    config=replace(configuration(product,cohort),epochs=1,batch_size=8,use_scheduler=False,
        use_mixup=False,use_early_stopping=False,verbose=False,
        normalization=fitted['options'],normalization_bounds=fitted['products'][product])
    ds=make_dataset(info,keys,product,cache,normalization=fitted['options'],bounds=fitted['products'][product])
    val=make_dataset(info,partition['validation'][:8],product,cache,normalization=fitted['options'],bounds=fitted['products'][product])
    loader=DataLoader(ds,batch_size=8);vl=DataLoader(val,batch_size=8)
    model=DualEncoderCNN((3,128,128))
    positive=compute_task_counts(info,partition['train'],'rr_rh')
    known=np.array([task_target(info[k],'rr_rh')[1] for k in partition['train']]).sum(0).tolist()
    engine=Trainer(model,config,('RR','RH'),positive,n_train=known,steps_per_epoch=len(loader),
        input_adapter=lambda b:torch.cat([b['raw'],b['chandra'],b['chandra_mask']],1))
    gate_checks=[]
    def gate(module,args):
        features=args[0]; absent=features[:,-1]==0
        if absent.any():
            radio_dim=model.radio_encoder.output_dim
            assert torch.count_nonzero(features[absent,radio_dim:-1])==0
            gate_checks.append(int(absent.sum()))
    hook=model.classifier.register_forward_pre_hook(gate)
    history=engine.fit(loader,vl,run_dir=output)
    hook.remove()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum()>0 for g in grads)
    if missing:assert gate_checks
    checkpoint=torch.load(output/'best.pt',weights_only=False,map_location='cpu')
    checkpoint['matched_resolution']=dict(size=128,radio_mask='intersection',missing_modality_policy='optional',
        normalization=fitted,partition=partition,diagnostic_train_keys=keys)
    torch.save(checkpoint,output/'best.pt')
    restored=DualEncoderCNN((3,128,128));restored.load_state_dict(checkpoint['model_state_dict']);restored.eval();model.eval()
    batch=next(iter(vl));x=torch.cat([batch['raw'],batch['chandra'],batch['chandra_mask']],1)
    with torch.no_grad():assert torch.equal(model(x),restored(x))
    result=dict(product=product,cohort=cohort,model='DualEncoderCNN',train_subset=keys,
        missing_chandra_in_subset=len(set(keys)&set(missing)),steps=len(loader),
        gated_missing_examples=sum(gate_checks),finite_nonzero_gradients=True,checkpoint_reload_exact=True,
        known_train=known,positive_train=positive,pos_weight=engine.criterion.pos_weight.tolist(),
        train_loss=history[0]['train']['loss'],validation_loss=history[0]['validation']['loss'],test_evaluated=False)
    write_json(output/'smoke_result.json',result)
    return result


def prepare(project,output):
    output.mkdir(parents=True,exist_ok=False)
    baseline=json.loads((project/'outputs/matched_cohort_phase5/matched_splits.json').read_text())
    protocol=json.loads((project/'outputs/matched_protocol_phase6/matched_folds.json').read_text())
    info,rows=inventory(project,output)
    complete(project,output,baseline,protocol,info,rows)


def complete(project,output,baseline,protocol,info,rows):
    base=baseline['samples'];expanded=extend_folds(base,info,protocol['folds'])
    for k in base:
        for name in ('labels','rr_rh_known_mask','chandra_path'):
            if info[k][name]!=base[k][name]:raise ValueError(f'Original annotation/source changed: {k}')
    cache=output/'cache'
    write_json(output/'expanded_manifest.json',dict(samples=info,cache_root=str(cache),counts=counts(info,list(info))))
    write_json(output/'expanded_folds.json',dict(folds=expanded,training_seeds=protocol['training_seeds'],
        paired_folds=protocol['folds'],additional_ids=expanded[0]['added_train'],
        evaluation_population='Original 65 paired clusters only; additions are train-only in every fold'))
    configs=[];fitted={};smokes=[]
    for cohort,partitions in [('paired',protocol['folds']),('expanded',expanded)]:
        fitted[cohort]={}
        for partition in partitions:
            f=partition['fold'];print(f'Fitting {cohort} fold {f}',flush=True)
            fitted[cohort][str(f)]=fit_train_normalization(info,partition,cache)
            assert set(fitted[cohort][str(f)]['train_keys'])==set(partition['train'])
    configs=experiment_specifications(output,protocol['training_seeds'])
    write_json(output/'fold_normalization.json',fitted)
    write_json(output/'experimental_configs.json',configs)
    for specification in configs:
        write_json(output/(specification['experiment']['name']+'_config.json'),specification)
    norm_checks=[]
    for fold in expanded:
        f=str(fold['fold'])
        unchanged=fitted['paired'][f]['products']['RAW']['xray']==fitted['expanded'][f]['products']['RAW']['xray']
        if all(info[k]['chandra_path'] is None for k in fold['added_train']):
            assert unchanged, 'Missing CHANDRA polluted normalization'
        norm_checks.append(dict(fold=fold['fold'],xray_bounds_equal_between_cohorts=unchanged,
            train_keys_exclude_validation_test=all(set(fitted[c][f]['train_keys']).isdisjoint(fold['validation']+fold['test']) for c in ('paired','expanded'))))
    write_json(output/'normalization_checks.json',norm_checks)
    # Validate every expanded member once using disjoint fold-0 partitions.
    checked,errors=verify_pairs(info,expanded[0],cache,fitted['expanded']['0'])
    write_csv(output/'pair_validation.csv',checked);write_json(output/'pair_errors.json',errors)
    if errors:raise ValueError(f'{len(errors)} incompatible pairs')
    for cohort,partitions in [('paired',protocol['folds']),('expanded',expanded)]:
        for product in ('RAW','T50kpcSUB'):
            print(f'Smoke {cohort} {product}',flush=True)
            smokes.append(smoke(info,partitions[0],cache,fitted[cohort]['0'],product,cohort,output/f'{cohort}_{product}'))
    write_json(output/'smoke_tests.json',smokes)
    groups={'expanded':list(info),'paired':list(base),'added':expanded[0]['added_train'],
            'chandra_absent':[k for k in info if info[k]['chandra_path'] is None],
            'chandra_present':[k for k in info if info[k]['chandra_path'] is not None]}
    summary={n:counts(info,keys) for n,keys in groups.items()};write_json(output/'cohort_counts.json',summary)
    write_report(output,summary,rows,smokes,expanded)


def write_report(output,summary,rows,smokes,folds):
    lines=['# Extension phase 6 — CHANDRA optionnel','',
        f"{summary['expanded']['n']} amas exploitables : {summary['paired']['n']} appariés historiques et {summary['added']['n']} supplémentaires.",
        f"{len(rows)} identifiants recensés ; {len(rows)-summary['expanded']['n']} exclus (raisons dans l’inventaire).",
        f"Annotations candidates : {summary['expanded']['annotation_cRR']} cRR et {summary['expanded']['annotation_cRH']} cRH, comptées positives.",
        'Politique candidats positive conservée ; sous-type non annoté inconnu, NDE négatif connu pour les deux labels. Annotations sources inchangées.',
        '', '| Groupe | N | RR+ | RR− connu | RR inconnu | RH+ | RH− connu | RH inconnu |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,d in summary.items():lines.append('| '+name+' | '+' | '.join(str(d[k]) for k in ('n','RR_positive','RR_negative','RR_unknown','RH_positive','RH_negative','RH_unknown'))+' |')
    lines += ['', '## Prétraitement et cache',
        'Radio RAW/T50 harmonisée sur la même grille 128², masque intersection, contrôles WCS inchangés. CHANDRA absent : image et masque nuls, has_chandra=false et chandra_status=absent. Un fichier présent sans support est present_uncovered ; une couverture partielle reste une observation présente. Un fichier présent invalide est exclu, jamais assimilé à une absence.',
        'Le cache distingue source CHANDRA nulle et empreinte d’un fichier présent. La vérification accepte des zéros hors masque uniquement pour la modalité explicitement absente. L’identité inclut le code : les anciens caches ne sont pas réutilisés silencieusement après cette évolution.',
        '', '## Partitions et comparaison',
        f"Cinq tests externes et validations internes inchangés. Chaque train élargi comprend {len(folds[0]['train'])} amas ; validation/test restent 13/13. Les ajouts sont exclusivement train dans les cinq folds et ne constituent pas une population évaluée hors fold.",
        'Les quatre expériences emploient DualEncoderCNN existant, mêmes hyperparamètres, augmentations, seeds 42/123/2026, budgets et sélection val_loss. Aucun nouveau mécanisme de fusion ni modality dropout. Elles remplacent la comparaison FusionCNN proposée dans le protocole précédent ; les quatre bras doivent être comparés entre eux.',
        'Budget commun : 200 époques maximum de 3 mises à jour, batch 16 (600 pas maximum), arrêt anticipé val_loss identique. budgeted_loader tire 48 exemples avec remise depuis tout le train à chaque époque, avec générateur seedé persistant ; cela évite de donner plus de pas au bras élargi. Les quatre bras utilisent la même règle. Les seuils sont calibrés uniquement après sélection du checkpoint.',
        'Normalisation globale P30/P99 puis asinh ajustée sur le train de chaque bras/fold, séparément par produit. Pixels CHANDRA absents exclus des percentiles. Validation/test utilisent les bornes figées. Aucun préentraînement et aucun checkpoint historique chargé. Calibration future exclusivement sur validation interne inchangée.',
        '', '## Vérifications',
        f"Toutes les {summary['expanded']['n']} paires sont contrôlées par les vrais datasets : labels, masques, beam, WCS, CHANDRA et partition identiques.",
        'Smoke : une époque bornée à deux batches (16 exemples) pour chacun des quatre bras, fold 0 ; poids et normalisation calculés sur le train complet du fold. Augmentations/MixUp/scheduler désactivés pour ce diagnostic dans les quatre bras ; aucune calibration de seuil sur le test.',
        *[f"- {s['cohort']} / {s['product']} : {s['steps']} pas ; {s['missing_chandra_in_subset']} absents ; gradients finis, checkpoint restauré exactement." for s in smokes],
        '', '## Limites',
        'Les effectifs RR des tests ne changent pas : 1/2/3/3/3 positifs. L’élargissement augmente la supervision train, pas la puissance statistique du test. La comparaison reste conditionnée à la population disposant de CHANDRA.',
        'L’annulation des features X-ray est vérifiée à l’entrée du classifieur. Les BatchNorm historiques de l’encodeur peuvent néanmoins voir les entrées nulles en entraînement ; cette architecture n’est pas modifiée ici.',
        'Les exclusions WCS ne sont pas récupérées et aucune tolérance n’est relâchée. Les pixels d’un CHANDRA absent ne créent pas de statistique de normalisation. Un train sans aucun pixel X-ray connu reste refusé par la normalisation globale.',
        '', '## Livrables',
        '[Inventaire](expanded_cohort.csv), [effectifs](cohort_counts.json), [folds](expanded_folds.json), [configurations](experimental_configs.json), [bornes train](fold_normalization.json), [smoke tests](smoke_tests.json), [tests et fichiers modifiés](test_results.md).',
        'Aucun entraînement complet lancé. Validation scientifique attendue avant phase 7.']
    (output/'expanded_cohort_report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/optional_chandra_phase6')
    args=p.parse_args();prepare(Path.cwd(),Path(args.output).resolve())
