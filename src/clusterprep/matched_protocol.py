"""Phase 6 protocol preparation and bounded smoke runs; no full CV command."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import csv
import json
from pathlib import Path
import random
import warnings

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from .matched_cohort import (counts, fit_train_normalization, make_dataset,
                             write_json, write_csv, audit_pretraining)
from .splits import multilabel_stratum
from .labels import task_target
from .training import ExperimentConfig, Trainer, compute_task_counts
from .models import FusionCNN
from .metrics import multilabel_metrics, select_multilabel_thresholds


def validate_partition(info, partition):
    groups = [partition[n] for n in ('train', 'validation', 'test')]
    flat = sum(groups, [])
    ids = [info[k]['cluster_id'] for k in flat]
    if len(ids) != len(set(ids)) or set(flat) != set(info):
        raise ValueError('Partitions must be exhaustive, disjoint unique clusters')


def prepare_folds(info, seed=42):
    keys = np.array(sorted(info))
    strata = np.array([multilabel_stratum(info[k]) for k in keys])
    folds = []
    # Four RR-only objects cannot occur in every one of five outer folds.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        outer = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(keys, strata))
    for number, (pool, test) in enumerate(outer):
        train, validation = train_test_split(pool, test_size=.25, random_state=seed+number,
                                             stratify=strata[pool])
        fold = dict(fold=number, train=keys[train].tolist(), validation=keys[validation].tolist(),
                    test=keys[test].tolist(), candidate_policy='positive')
        validate_partition(info, fold)
        fold['distributions'] = {n: counts(info, fold[n]) for n in ('train','validation','test')}
        for n in ('train','validation','test'):
            for label in ('RR','RH'):
                if min(fold['distributions'][n][label+'_positive'], fold['distributions'][n][label+'_negative']) == 0:
                    raise ValueError(f'Fold {number} {n}: {label} lacks both known classes')
        folds.append(fold)
    assert Counter(k for f in folds for k in f['test']) == Counter(keys)
    return folds


def calibrate(validation_rows, partition):
    ids = [r['cluster_id'] for r in validation_rows]
    if len(ids) != len(set(ids)) or set(ids) != set(partition['validation']):
        raise ValueError('Threshold fitting requires exactly the independent validation IDs')
    if set(ids) & (set(partition['train']) | set(partition['test'])):
        raise ValueError('Threshold calibration leakage')
    result = select_multilabel_thresholds([r['label'] for r in validation_rows],
        [r['probabilities'] for r in validation_rows],
        target_mask=[r['label_mask'] for r in validation_rows])
    result['cluster_ids'] = ids
    return result


def export_oof(rows, partition, selection, product, seed, path):
    if selection['cluster_ids'] != list(dict.fromkeys(selection['cluster_ids'])) or set(selection['cluster_ids']) != set(partition['validation']):
        raise ValueError('Invalid validation provenance')
    ids = [r['cluster_id'] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(partition['test']) or set(ids) & set(selection['cluster_ids']):
        raise ValueError('Invalid held-out IDs')
    output = []
    for row in rows:
        r = dict(cluster_id=row['cluster_id'], fold=partition.get('fold','baseline'), seed=seed, product=product)
        for j, name in enumerate(('RR','RH')):
            r.update({f'{name}_target': row['label'][j], f'{name}_known': row['label_mask'][j],
                      f'{name}_probability': row['probabilities'][j], f'{name}_threshold': selection['thresholds'][name],
                      f'{name}_prediction': int(row['probabilities'][j] >= selection['thresholds'][name])})
        output.append(r)
    write_csv(path, output)
    return output


def paired_analysis(a, b, thresholds_a=None, thresholds_b=None, draws=1000, seed=42):
    """Cluster bootstrap, paired products; fixed validation-selected thresholds."""
    a = sorted(a, key=lambda r:r['cluster_id']); b = sorted(b, key=lambda r:r['cluster_id'])
    if not a or len({r['cluster_id'] for r in a}) != len(a) or [r['cluster_id'] for r in a] != [r['cluster_id'] for r in b]:
        raise ValueError('Unique paired IDs required')
    y = np.array([r['label'] for r in a]); m = np.array([r['label_mask'] for r in a])
    if not np.array_equal(y, [r['label'] for r in b]) or not np.array_equal(m, [r['label_mask'] for r in b]):
        raise ValueError('Paired annotations differ')
    probs = [np.array([r['probabilities'] for r in rows]) for rows in (a,b)]
    cutoffs=[t if t is not None else np.array([r['thresholds'] for r in rows])
             for t,rows in zip((thresholds_a,thresholds_b),(a,b))]
    def scores(indices):
        results=[]
        for p,t in zip(probs,cutoffs):
            if isinstance(t,dict):
                result=multilabel_metrics(y[indices],p[indices],target_mask=m[indices],thresholds=t)
            else:
                # Confusions use each row's independently selected fold thresholds;
                # AP/AUC and Brier retain the original continuous probabilities.
                continuous=multilabel_metrics(y[indices],p[indices],target_mask=m[indices])
                result=multilabel_metrics(y[indices],(p[indices]>=t[indices]).astype(float),target_mask=m[indices])
                result['thresholds']='per_row_validation_selected'
                for name in ('RR','RH'):
                    for metric in ('roc_auc','average_precision','brier'):
                        result['per_label'][name][metric]=continuous['per_label'][name][metric]
                    result['per_label'][name]['threshold']='per_row_validation_selected'
                for metric in ('roc_auc_macro','average_precision_macro','brier_macro'):
                    result[metric]=continuous[metric]
                    result[metric+'_n_labels']=continuous[metric+'_n_labels']
            for values in result['per_label'].values():
                values['precision_denominator']=values['tp']+values['fp']
                values['recall_denominator']=values['tp']+values['fn']
                values['f1_denominator']=2*values['tp']+values['fp']+values['fn']
                for metric,denominator in (('precision',values['precision_denominator']),('recall',values['recall_denominator'])):
                    if denominator:
                        proportion=values['tp']/denominator; z=1.959963984540054
                        scale=1+z*z/denominator
                        center=(proportion+z*z/(2*denominator))/scale
                        radius=z*np.sqrt(proportion*(1-proportion)/denominator+z*z/(4*denominator**2))/scale
                        values[metric+'_ci95_wilson']=[float(max(0,center-radius)),float(min(1,center+radius))]
                    else:
                        values[metric+'_ci95_wilson']=None
            results.append(result)
        return results
    full = scores(np.arange(len(a)))
    paths = [('macro_f1',)] + [('per_label',n,s) for n in ('RR','RH') for s in ('precision','recall','f1','average_precision','roc_auc')]
    def get(result,path):
        for key in path: result=result[key]
        return result
    rng=np.random.default_rng(seed); samples={str(p):[[],[],[]] for p in paths}
    for _ in range(draws):
        result=scores(rng.integers(0,len(a),len(a)))
        for p in paths:
            x,z=get(result[0],p),get(result[1],p)
            for slot,v in zip(samples[str(p)], (x,z,None if x is None or z is None else z-x)):
                if v is not None: slot.append(v)
    intervals={str(p): {name:dict(n_valid=len(v), ci95=np.percentile(v,[2.5,97.5]).tolist() if v else None)
                for name,v in zip(('RAW','T50kpcSUB','delta_T50_minus_RAW'),samples[str(p)])} for p in paths}
    return dict(n_clusters=len(a), metrics_RAW=full[0], metrics_T50kpcSUB=full[1], intervals=intervals,
                draws=draws, method='paired cluster percentile bootstrap, fixed thresholds; exploratory small-N',
                limitation='Conditional on fitted models: does not estimate training variability or shared-training dependence across CV folds; seeds are not independent clusters')


def compare_oof_records(raw, t50, draws=1000):
    """Compare one seed's exported OOF rows, retaining fold-specific cutoffs."""
    if len({r['seed'] for r in raw+t50}) != 1:
        raise ValueError('Analyze one training seed at a time; never pool seeds as clusters')
    raw=sorted(raw,key=lambda r:r['cluster_id']); t50=sorted(t50,key=lambda r:r['cluster_id'])
    if [(r['cluster_id'],r['fold']) for r in raw] != [(r['cluster_id'],r['fold']) for r in t50]:
        raise ValueError('Different OOF fold membership')
    def convert(rows):
        return [dict(cluster_id=r['cluster_id'],label=[r[n+'_target'] for n in ('RR','RH')],
            label_mask=[r[n+'_known'] for n in ('RR','RH')],probabilities=[r[n+'_probability'] for n in ('RR','RH')],
            thresholds=[r[n+'_threshold'] for n in ('RR','RH')]) for r in rows]
    return paired_analysis(convert(raw),convert(t50),draws=draws)


def geometry_audit(project, excluded):
    """Read-only local geometry diagnostic; never relax the stationarity gate."""
    from .common_wcs_validation import read_catalog, discover_sources, common_grid, ValidationConfig
    from .radio_psf import read_radio, target_beam, jacobian, kernel_covariance
    catalog={r['cluster_id']:r for r in read_catalog(project/'outputs/common_wcs_validation/inputs/lofar_psz2_catalog.fits')}
    rows=[]
    for item in excluded:
        cid=item['cluster_id']
        try:
            record=catalog[cid]; sources=discover_sources(record,project/'scratch/data/PSZ2/fits')
            planes=[read_radio(sources[p]) for p in ('RAW','T50kpcSUB')]
            output_wcs,_=common_grid(record,None,ValidationConfig(output_size=128))
            target,_=target_beam(planes,output_wcs)
            for product,plane in zip(('RAW','T50kpcSUB'),planes):
                h,w=plane.data.shape; center=jacobian(plane.wcs,(w-1)/2,(h-1)/2)
                mesh=[jacobian(plane.wcs,x,y) for y in np.linspace(-.5,h-.5,5) for x in np.linspace(-.5,w-.5,5)]
                kernels=[kernel_covariance(plane.beam,target,j) for j in mesh]
                rows.append(dict(cluster_id=cid,product=product,target_fwhm_arcsec=target.major_arcsec,
                    jacobian_relative_variation=max(float(np.linalg.norm(j@np.linalg.inv(center)-np.eye(2),2)) for j in mesh),
                    min_local_kernel_eigenvalue_pix2=min(float(np.linalg.eigvalsh(k).min()) for k in kernels),
                    max_local_kernel_eigenvalue_pix2=max(float(np.linalg.eigvalsh(k).max()) for k in kernels),
                    local_covariance_feasible=True,psf_recovery_validated=False))
        except Exception as exc:
            rows.append(dict(cluster_id=cid,error=str(exc),local_covariance_feasible=None,psf_recovery_validated=False))
    return rows


def stratified_comparisons(a,b,thresholds_a,thresholds_b,metadata,draws=200):
    """Predeclared descriptive bins; do not select cuts using performance."""
    results=[]
    definitions={'coverage_CHANDRA':[0,.1,.2,.3,.5,1.000001],
                 'fwhm_over_r500':[0,.1,.12,float('inf')]}
    for variable,edges in definitions.items():
        for lo,hi in zip(edges[:-1],edges[1:]):
            ids={r['cluster_id'] for r in a if lo <= float(metadata[r['cluster_id']][variable]) < hi}
            result=paired_analysis([r for r in a if r['cluster_id'] in ids],
                [r for r in b if r['cluster_id'] in ids],thresholds_a,thresholds_b,draws=draws) if ids else None
            results.append(dict(variable=variable,lower=lo,upper=hi if np.isfinite(hi) else None,
                                n_clusters=len(ids),comparison=result))
    return results


def smoke(info, partition, cache, fitted, product, output, seed=42):
    output.mkdir(parents=True,exist_ok=False)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True); torch.set_num_threads(1)
    config = ExperimentConfig(name='phase6_smoke_'+product, task='rr_rh', mode='pair', model='fusion',
        radio_source=product, epochs=2, batch_size=8, device='cpu', use_compile=False,
        use_scheduler=False, use_early_stopping=False, drop_last_train=False, augment=False,
        use_mixup=True, verbose=False, seed=seed,
        normalization=fitted['options'],normalization_bounds=fitted['products'][product])
    datasets={n:make_dataset(info,partition[n],product,cache,normalization=fitted['options'],
                            bounds=fitted['products'][product]) for n in ('train','validation')}
    loaders={n:DataLoader(ds,batch_size=8,shuffle=n=='train',num_workers=0) for n,ds in datasets.items()}
    positive=compute_task_counts(info,partition['train'],'rr_rh')
    known=np.array([task_target(info[k],'rr_rh')[1] for k in partition['train']]).sum(0).tolist()
    model=FusionCNN(width=16)
    trainer=Trainer(model,config,('RR','RH'),positive,n_train=known,steps_per_epoch=len(loaders['train']),
        input_adapter=lambda b:torch.cat([b['raw'],b['chandra'],b['chandra_mask']],1))
    batch=next(iter(loaders['validation']))
    shape=list(batch['raw'].shape)
    initial={k:v.clone() for k,v in model.state_dict().items()}
    history=trainer.fit(loaders['train'],loaders['validation'],run_dir=output)
    metrics, predictions=trainer.evaluate(loaders['validation'])
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients) and any(g.abs().sum()>0 for g in gradients)
    assert any(not torch.equal(v,initial[k]) for k,v in model.state_dict().items())
    checkpoint=torch.load(output/'best.pt',map_location='cpu',weights_only=False)
    checkpoint['preprocessing'].update(radio_preprocessing='matched_resolution',radio_mask='intersection',
        normalization=fitted['options'],normalization_bounds=fitted['products'][product],
        normalization_train_keys=list(partition['train']))
    checkpoint['partition']={n:partition[n] for n in ('train','validation','test')}
    torch.save(checkpoint,output/'best.pt')
    restored=FusionCNN(width=16); restored.load_state_dict(checkpoint['model_state_dict']); restored.eval(); model.eval()
    x=torch.cat([batch['raw'],batch['chandra'],batch['chandra_mask']],1)
    with torch.no_grad(): assert torch.equal(restored(x),model(x))
    selection=calibrate(predictions,partition)
    write_json(output/'validation_predictions.json',predictions)
    write_json(output/'threshold_selection.json',selection)
    write_json(output/'preprocessing.json',dict(pipeline='matched_resolution',radio_mask='intersection',
        size=128,normalization=fitted,partition=partition,source_checkpoint=None))
    result=dict(product=product,seed=seed,epochs=2,batches_per_epoch=len(loaders['train']),input_shape=shape,
        finite_nonzero_gradients=True,checkpoint_reload_exact=True,known_train=known,positive_train=positive,
        pos_weight=trainer.criterion.pos_weight.tolist(),validation_metrics=metrics,
        history_losses=[(h['train']['loss'],h['validation']['loss']) for h in history],
        test_evaluated=False,transfer=None)
    write_json(output/'smoke_result.json',result)
    return result,predictions


def prepare(project, output, run_smoke=False):
    output.mkdir(parents=True,exist_ok=False)
    source=project/'outputs/matched_cohort_phase5'
    manifest=json.loads((source/'matched_splits.json').read_text()); info=manifest['samples']
    folds=prepare_folds(info)
    protocol=dict(status='pending_scientific_validation',training_seeds=[42,123,2026],fold_seed=42,
        baseline={n:manifest[n] for n in ('train','validation','test')},folds=folds,
        rare_stratum='1? has only 4 members: absent from one outer test fold',
        threshold_selection='F1 on inner validation only; outer test never used for selection',
        source_manifest=str(source/'matched_splits.json'))
    write_json(output/'matched_folds.json',protocol)
    configs=[]
    for normalization in ('train','image'):
        for product in ('RAW','T50kpcSUB'):
            cfg=ExperimentConfig(name=f'{product}_{normalization}',task='rr_rh',mode='pair',model='fusion',
                radio_source=product,device='cpu',use_compile=False,augment=False,drop_last_train=False,
                threshold_selection='f1',transfer=None)
            configs.append(dict(experiment=asdict(cfg),pipeline='matched_resolution',size=128,radio_mask='intersection',
                normalization_scope=normalization,normalization_fit='partition.train only',
                folds_file=str(output/'matched_folds.json'),training_seeds=protocol['training_seeds'],
                runner='Trainer with matched_cohort.make_dataset; not generic run_experiment',
                status='proposal_not_authorized_for_full_training'))
    write_json(output/'experimental_configs.json',configs)
    normalizations={}; audits={}
    for name,partition in [('baseline',manifest)]+[(f'fold_{f["fold"]}',f) for f in folds]:
        normalizations[name]=fit_train_normalization(info,partition,manifest['cache_root'])
        audits[name]=audit_pretraining(project/'results',partition)
    write_json(output/'fold_normalization.json',normalizations)
    write_json(output/'pretraining_exposure.json',audits)
    inventory=list(csv.DictReader((source/'matched_cohort.csv').open()))
    excluded=[r for r in inventory if r['reason'].startswith('preprocessing: ValueError: Nonstationary WCS')]
    write_csv(output/'wcs_only_exclusions.csv',excluded)
    write_csv(output/'wcs_local_geometry.csv',geometry_audit(project,excluded))
    rows=[]
    for f in folds:
        for split,d in f['distributions'].items(): rows.append(dict(fold=f['fold'],split=split,**d))
    write_csv(output/'fold_feasibility.csv',rows)
    smoke_results=[]
    if run_smoke:
        predictions=[]
        for product in ('RAW','T50kpcSUB'):
            first,pred=smoke(info,manifest,manifest['cache_root'],normalizations['baseline'],product,output/product)
            second,repeated=smoke(info,manifest,manifest['cache_root'],normalizations['baseline'],product,output/(product+'_repeat'))
            assert first==second and pred==repeated, 'CPU reproducibility mismatch'
            first['repeat_exact']=True; smoke_results.append(first); predictions.append(pred)
        write_json(output/'smoke_tests.json',smoke_results)
        # Descriptive validation diagnostics only, not unbiased performance estimates.
        cuts=[json.loads((output/p/'threshold_selection.json').read_text())['thresholds'] for p in ('RAW','T50kpcSUB')]
        write_json(output/'smoke_validation_comparison.json',paired_analysis(*predictions,*cuts,draws=200))
        metadata={r['cluster_id']:r for r in inventory}
        write_json(output/'smoke_validation_strata.json',stratified_comparisons(*predictions,*cuts,metadata))
    write_report(output,protocol,excluded,smoke_results,audits)


def write_report(output,protocol,excluded,smokes,audits):
    lines=['# Phase 6 — protocole provisoire','',
        'Baseline 65 amas conservée, partitions 39/13/13 inchangées. Aucun seuil de couverture ajouté.',
        'Cinq folds externes communs aux deux produits ; chacun : 39 train, 13 validation interne, 13 test.',
        'Stratification conjointe labels/masques. 1? ne contient que quatre amas : impossible de représenter cette strate dans chaque test.',
        'La validation interne choisit checkpoint et seuils ; les prédictions test ne sont jamais utilisées pour cela.',
        'Les folds externes ne se recouvrent pas et couvrent une seule fois les 65 amas. Les trains des folds se recouvrent normalement.',
        '', '| Fold | RR+ train / val / test | RR− connus train / val / test |', '|---|---|---|']
    for f in protocol['folds']:
        d=f['distributions']; lines.append(f"| {f['fold']} | "+' / '.join(str(d[n]['RR_positive']) for n in ('train','validation','test'))+' | '+' / '.join(str(d[n]['RR_negative']) for n in ('train','validation','test'))+' |')
    lines += ['', '## Labels, loss et normalisation',
        'BCEWithLogits élémentaire pondérée, sommée sur les labels connus et divisée par leur nombre. MixUp transporte les masques avec les cibles et pondère les deux numérateurs/dénominateurs.',
        'Un batch entièrement inconnu ne produit aucun pas optimiseur/scheduler ; un split entièrement inconnu est rejeté explicitement. Les valeurs masquées ne comptent ni dans la loss ni dans les métriques.',
        'Poids positifs = négatifs connus / positifs connus, calculés uniquement sur le train non augmenté. Le calcul des positifs a été renforcé pour exclure explicitement toute cellule masquée.',
        'P30/P99 + asinh(10), formule historique conservée. Bornes globales réellement ajustées séparément par produit et par train de fold, figées pour validation/test. CHANDRA : bornes identiques entre produits.',
        'Comparaison secondaire par image, sans statistiques de population. Configurations proposées : FusionCNN existant largeur 16, mêmes hyperparamètres, MixUp activé, augmentation géométrique désactivée dans les deux vues. Seeds 42, 123, 2026.',
        'Les configurations sont des spécifications pour le runner apparié, pas des configurations directement exécutables par run_experiment historique.',
        '', '## Évaluation et incertitude',
        'F1, précision, rappel, AP (résumé PR-AUC), ROC-AUC et matrices sur labels connus ; n_known, n_positive, n_negative et TP/FP/FN/TN donnent les dénominateurs. AUC/AP null si une seule classe connue. Macro-F1 indique le nombre de classes évaluables.',
        'export_oof exige les identifiants test exacts et une provenance de calibration validation disjointe. Aucun fichier de prédictions hors fold réelles n’est fabriqué avant les entraînements autorisés.',
        'Précision/rappel : intervalles Wilson 95 % également fournis à partir des effectifs connus ; dénominateur nul signalé explicitement.',
        'Comparaison appariée : bootstrap par amas, mêmes tirages pour les produits, intervalles percentiles 95 %, nombre de réplications définies fourni. Faible nombre de RR : intervalles exploratoires, forte variance ; ne pas traiter les seeds comme des amas indépendants.',
        'compare_oof_records conserve les seuils de chaque ligne/fold pour les confusions et les probabilités originales pour AP/AUC ; chaque seed est analysée séparément.',
        'Prévoir les analyses descriptives par couverture CHANDRA [0,.1,.2,.3,.5,1] et FWHM/R500 ; ne pas optimiser ces coupures sur les performances test. Les seuils propres à chaque fold doivent rester attachés à chaque prédiction OOF.',
        '', '## Transfer learning',
        'Baseline initialisée aléatoirement, aucun checkpoint historique chargé. Audit complet par baseline et fold dans pretraining_exposure.json.',
        'Un futur préentraînement compatible doit employer matched_resolution et exclure tous les amas de validation interne/test du fold cible, y compris pour sa sélection de checkpoint et ses statistiques. Cela implique des sources/checkpoints spécifiques aux folds ou une source externe disjointe des 65 amas.',
        'Un manifest inconnu ne prouve jamais l’absence de contamination. Les checkpoints historiques ne sont pas validés pour ces images.',
        '', '## Exclusions WCS',
        f'{len(excluded)} amas autrement éligibles exclus uniquement par le contrôle de non-stationnarité. Liste et erreurs exactes : wcs_only_exclusions.csv.',
        'La tolérance 1 % est inchangée. Une récupération plausible exige des noyaux variant spatialement : Jacobien WCS local, covariance du beam natif transformée localement, covariance résiduelle cible positive partout, convolution locale avec support/masques, puis reprojection contrôlée.',
        'Diagnostic numérique : wcs_local_geometry.csv évalue les Jacobiennes sur 25 positions par image et les covariances résiduelles du beam cible ; leur positivité est nécessaire, mais ne certifie pas la PSF finale.',
        'Sur ces données : 17 amas ont des covariances locales réalisables pour les deux produits (34 images) ; 3 restent non évaluables car le contrôle de la grille cible échoue déjà. Une autre erreur de prétraitement peut encore apparaître après correction WCS : disponibilité ne signifie pas récupération validée.',
        'Une simple reprojection plus fine ne suffit pas à certifier la PSF. Valider des sources ponctuelles sur une grille de positions, FWHM/ellipticité, flux et bords masqués, avec lissage anti-aliasing et unités Jy/beam cohérentes. Aucun amas récupéré ni ajouté sans cette validation ; faisabilité conceptuelle seulement, pas preuve numérique de récupération.',
        '', '## Smoke tests',
        ('Deux époques CPU sur les 39 amas train, validation sur 13, pour chaque produit puis répétition exacte seed 42. Test baseline jamais évalué. Scheduler désactivé pour ce diagnostic court, compilation désactivée.' if smokes else 'Smoke tests non exécutés : relancer dans un nouveau dossier avec --smoke pour cette validation.'),
        *[f"- {s['product']} : {s['epochs']} époques × {s['batches_per_epoch']} batches ; gradients finis non nuls, rechargement checkpoint exact, répétition exacte={s['repeat_exact']}." for s in smokes],
        'Les résultats sur validation sont des diagnostics de fonctionnement, pas une estimation de performance scientifique.',
        '', '## Livrables et limites',
        '[Folds](matched_folds.json), [effectifs](fold_feasibility.csv), [configurations](experimental_configs.json), [normalisation](fold_normalization.json), [smoke tests](smoke_tests.json).',
        'Les effectifs RR rendent les seuils et AUC de chaque fold fragiles. Les bornes et partitions restent provisoires jusqu’à validation scientifique. Aucune CV complète ni phase 7 lancée.',
        'Code : [matched_protocol.py](../../clusterprep/src/clusterprep/matched_protocol.py), [training.py](../../clusterprep/src/clusterprep/training.py), [tests](../../clusterprep/tests/test_matched_protocol.py). Architectures et annotations sources inchangées. Résultats automatisés : [test_results.md](test_results.md).']
    (output/'protocol_report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='outputs/matched_protocol_phase6')
    parser.add_argument('--smoke',action='store_true'); args=parser.parse_args()
    prepare(Path.cwd(),Path(args.output).resolve(),args.smoke)
