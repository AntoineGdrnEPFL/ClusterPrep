"""Descriptive phase-5 outputs, without choosing a scientific coverage threshold."""
from collections import Counter
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def write_report(output, inventory, manifest, measurements, analysis, audits, fitted, errors):
    from .matched_cohort import write_csv, write_json, counts
    keys = list(manifest['samples'])
    data = {n: np.array([measurements[k]['coverage'][n] for k in keys])
            for n in ('RAW', 'T50kpcSUB', 'CHANDRA', 'intersection')}
    data['FWHM/R500'] = np.array([measurements[k]['target_resolution']['fwhm_over_r500'] for k in keys])
    summaries = []
    for name, values in data.items():
        summaries.append(dict(quantity=name, n=len(values), minimum=float(values.min()),
            p05=float(np.percentile(values, 5)), median=float(np.median(values)),
            p95=float(np.percentile(values, 95)), maximum=float(values.max())))
    write_csv(output/'distribution_summary.csv', summaries)
    figure = Figure(figsize=(13, 8), layout='constrained'); FigureCanvasAgg(figure)
    for i, (name, values) in enumerate(data.items()):
        ax = figure.add_subplot(2, 3, i+1)
        ax.hist(values, bins=20 if name == 'FWHM/R500' else np.linspace(0, 1, 21),
                color='#286b93', edgecolor='white')
        ax.set_title(f'{name} — n={len(values)}'); ax.set_ylabel('Amas')
        ax.set_xlabel('FWHM/R500' if name == 'FWHM/R500' else 'Fraction couverte')
        if name != 'FWHM/R500':
            ax.set_xlim(0, 1)
    ax = figure.add_subplot(2, 3, 6)
    for radio in (0., .5, .8, .9, 1.):
        rows = [r for r in analysis if r['min_radio_intersection'] == radio]
        ax.plot([r['min_chandra'] for r in rows], [r['n'] for r in rows], marker='o', label=f'Radio ≥ {radio:.0%}')
    ax.set_xlabel('Seuil de couverture CHANDRA'); ax.set_ylabel('Amas conservés')
    ax.legend(fontsize=8); ax.set_title('Scénarios, aucun seuil retenu')
    figure.suptitle('Cohorte techniquement exploitable — avant choix scientifique des seuils')
    figure.savefig(output/'coverage_resolution_distributions.png', dpi=160)
    figure.savefig(output/'coverage_resolution_distributions.pdf')
    # Record compatibility of global bounds with the unchanged denominator epsilon.
    diagnostics = {}
    for product, modalities in fitted['products'].items():
        diagnostics[product] = {}
        for name, bands in modalities.items():
            lo, hi = bands[0]
            diagnostics[product][name] = dict(lower=lo, upper=hi, span=hi-lo,
                epsilon=1e-6, epsilon_over_span=1e-6/(hi-lo) if hi > lo else None)
    write_json(output/'normalization_numeric_diagnostics.json', diagnostics)
    count = manifest['cohort_counts']
    statuses = Counter(a['status'] for a in audits)
    overlap_max = max((len(a.get('target_test_in_source_train', [])) for a in audits), default=0)
    distributions = '\n'.join(f"| {r['quantity']} | {r['minimum']:.4f} | {r['p05']:.4f} | {r['median']:.4f} | {r['p95']:.4f} | {r['maximum']:.4f} |" for r in summaries)
    thresholds = '\n'.join(f"| {r['min_chandra']:.0%} | {r['n']} | {r['RR_positive']} | {r['RH_positive']} | {r['combination_00']} | {r['combination_1?']} | {r['combination_?1']} | {r['combination_11']} |"
                           for r in analysis if r['min_radio_intersection'] == 0)
    splits = '\n'.join(f"| {n} | {manifest['distributions'][n]['n']} | {manifest['distributions'][n]['RR_positive']} | {manifest['distributions'][n]['RH_positive']} | " +
        ' / '.join(str(manifest['distributions'][n]['combination_'+c]) for c in ('00', '1?', '?1', '11')) + ' |'
        for n in ('train', 'validation', 'test'))
    reasons = Counter(r['reason'] for r in inventory if r['status'] != 'technically_eligible')
    exclusions = '\n'.join(f'- {n} : {reason}' for reason, n in reasons.most_common())
    text = f'''# Phase 5 — cohorte RR/RH appariée (provisoire)

## Décision attendue

**{len(keys)} amas techniquement exploitables sur {len(inventory)} identifiants recensés.**
Aucun seuil scientifique définitif n'est choisi. Les splits 60/20/20 ci-dessous sont un master
provisoire sur les observations exploitables, avec couverture CHANDRA et radio commune strictement
positive et sans coupure supplémentaire. Une validation scientifique de la cohorte, des seuils et
de la sémantique des annotations est nécessaire avant la suite. Aucun entraînement, phase 6 ou phase 7 exécuté.

## Annotations et disponibilité

Annotations fusionnées par identifiant depuis les dossiers RAW du catalogue classifié existant ;
la disponibilité d'un produit ne change pas les annotations. Les FITS utilisés sont les sources
natives du même identifiant, et les records physiques proviennent du catalogue de validation existant.
Les catégories U/cDE/unclassified restent dans la provenance mais ne créent pas de cible RR/RH.

Politique candidats : **{manifest['candidate_policy']}**. Par défaut, cRR et cRH comptent comme positifs,
conformément au pipeline existant. La variante `ignore` masque les candidats non confirmés.
**Une annotation absente n'est jamais un négatif certain dans cette cohorte** : 1? signifie RR positif,
RH inconnu ; ?1 signifie RR inconnu, RH positif ; 11 signifie les deux positifs ; 00 correspond à NDE.
Les combinaisons 10 et 01 ne sont pas déduites du silence d'annotation.

L'audit a constaté que `task_target` historique traitait les sous-types absents comme négatifs.
Un champ opt-in `rr_rh_known_mask` est donc enregistré dans chaque sample de cette cohorte et appliqué
par `task_target`. Les annotations sources ne sont pas réécrites. Sans ce champ, le comportement
historique est inchangé. Les cibles de stockage à zéro mais masquées ne constituent pas des négatifs.

RR positifs : **{count['RR_positive']}** ; RH positifs : **{count['RH_positive']}**.
Négatifs connus : RR={count['RR_negative']}, RH={count['RH_negative']}.
Inconnus : RR={count['RR_unknown']}, RH={count['RH_unknown']}.
Combinaisons : 00={count['combination_00']}, 1?={count['combination_1?']},
?1={count['combination_?1']}, 11={count['combination_11']}.
Les annotations candidates sont aussi comptées séparément dans [coverage_analysis.csv](coverage_analysis.csv).

[matched_cohort.csv](matched_cohort.csv) contient tous les identifiants, y compris les exclus,
leurs annotations, disponibilité, validation FITS effectivement exécutée, raisons, couvertures,
résolution et split éventuel. Pour un amas exclu en amont (fichier/annotation/record absent),
les couvertures sont vides, pas artificiellement nulles, et `fits_validated=False` signifie non validé.
Les amas retenus ont tous été prétraités avec matched_resolution, 128², 4R500, masque radio d'intersection,
3 pixels/FWHM, support minimal 0,999. Des pixels CHANDRA finis à zéro sont des observations, pas des absences.

Exclusions :
{exclusions}

## Distributions sur la cohorte techniquement exploitable

| Quantité | Min | P5 | Médiane | P95 | Max |
|---|---:|---:|---:|---:|---:|
{distributions}

[Figure](coverage_resolution_distributions.png) · [PDF](coverage_resolution_distributions.pdf) · [Statistiques](distribution_summary.csv).

Impact du seul seuil CHANDRA, sans coupure radio supplémentaire :

| CHANDRA ≥ | Amas | RR+ | RH+ | 00 | 1? | ?1 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|
{thresholds}

0 % signifie aucune coupure supplémentaire sur une cohorte déjà à support strictement positif.
[coverage_analysis.csv](coverage_analysis.csv) croise aussi les seuils radio communs 0/50/80/90/100 %
avec CHANDRA 0/10/20/30/50 %. Tous les comptes sont par amas unique ; un 11 compte dans RR+ et RH+.

## Splits provisoires communs

Seed **{manifest['seed']}**, 20 % test, 20 % validation du total, reste train. Stratification sur les
combinaisons de labels et de masques ; fallback itératif multilabel seulement si les strates rares
l'exigent. Le générateur a été relancé et les appartenances sont identiques. Les ensembles sont
exhaustifs et disjoints par identifiant. Un seul [matched_splits.json](matched_splits.json) porte
les deux vues radio, les annotations, masques de labels, sources et identités du cache apparié.
Aucun ancien split écrasé. Une future coupure de couverture pourra restreindre ces ensembles sans
déplacer un amas ; l'équilibre devra alors être réaudité et les statistiques train réajustées.

| Split | Amas | RR+ | RH+ | 00 / 1? / ?1 / 11 |
|---|---:|---:|---:|---|
{splits}

Avertissements de stratification : {manifest['stratification_warnings'] or 'aucun'}.

## Compatibilité avec les préentraînements DE/NDE

Audit en lecture seule de tous les `splits.json` trouvés sous `results/`, avec lecture du task dans
leur configuration. Statuts : **{dict(statuses)}**. Les manifests sans tâche explicite ou sans
mapping clé→cluster_id sont signalés non vérifiables, pas déclarés compatibles.

Le rapport distingue le test cible exposé au train source, la validation cible exposée au train
source, et le test cible exposé à la validation source (sélection du checkpoint). Maximum observé
pour un manifest : **{overlap_max} amas du test cible présents dans son train source**.
[pretraining_audit.json](pretraining_audit.json) contient les listes exactes et l'existence des checkpoints.
Un chevauchement signale une contamination **si ce checkpoint est réutilisé** ; aucun n'est sélectionné
ou chargé. L'absence de chevauchement de splits ne certifie pas la compatibilité de prétraitement :
les checkpoints historiques ne sont pas automatiquement adaptés à matched_resolution.

## Audit de normalisation

`ClusterDataset` appelle la normalisation seulement après chargement des images physiques et
application du masque choisi. Par défaut : P30/P99 par image, écrêtage [0,1], puis
`asinh(10*x)/asinh(10)`. Les pixels invalides n'entrent pas dans les percentiles ; les sorties
invalides valent zéro avec masque. Les bornes par image sont calculées sur chaque observation,
y compris validation/test : il n'y a pas de statistique de population estimée sur ces partitions.
Ce protocole par image doit être distingué d'une normalisation globale entraînée.

Pour l'option globale, `fit_normalization` a reçu **uniquement les clés train** du master provisoire,
sans bruit ni augmentation. Les bornes radio sont ajustées séparément pour RAW et T50kpcSUB ;
les bornes CHANDRA sont strictement égales. [normalization_audit.json](normalization_audit.json)
enregistre les clés et leur empreinte, les bornes et options. Ces paramètres figés sont passés aux
datasets de validation et de test ; aucun appel d'ajustement ne reçoit ces clés.
Aucune configuration d'entraînement n'est modifiée et cette option globale n'est pas activée implicitement.

Les images harmonisées restent en Jy/beam cible et peuvent contenir des valeurs négatives : les
percentiles suivis de l'écrêtage restent définis avant asinh. Les amplitudes n'ont pas été normalisées
avant le split. Le dénominateur historique est `P99-P30+1e-6` : ce plancher dépend des unités et peut
être non négligeable pour des images de faible dynamique. Les spans et le rapport epsilon/span sont
exportés dans [normalization_numeric_diagnostics.json](normalization_numeric_diagnostics.json),
sans changer cette formule. Aucun argument d'équivalence photométrique entre produits n'en est déduit.
Les bornes sont provisoires et devront être réestimées uniquement sur le train approuvé si la cohorte change.

## Vérification exhaustive des paires

**{len(keys)} paires contrôlées, {len(errors)} erreurs**. Contrôles : identifiant, grille WCS,
beam cible, masque radio commun, CHANDRA et masque (empreintes physiques et tenseurs normalisés),
annotations, cibles, masques de labels, split et finitude des tenseurs. Les deux vues sont de vrais
ClusterDataset, avec les mêmes partitions et les paramètres globaux figés propres au produit radio.
Résultats : [pair_validation.csv](pair_validation.csv), [pair_errors.json](pair_errors.json).
Un rapport sans erreur porte seulement sur la cohérence de cohorte/paires ; il ne remplace pas la
validation scientifique complète de phase 6 et ne démontre pas une supériorité d'un produit.

## Code et reproductibilité

Modules ajoutés : [matched_cohort.py](../../clusterprep/src/clusterprep/matched_cohort.py),
[matched_cohort_report.py](../../clusterprep/src/clusterprep/matched_cohort_report.py).
Modification opt-in des masques : [labels.py](../../clusterprep/src/clusterprep/labels.py).
Les architectures, routines d'entraînement et configurations sont inchangées.

```sh
.venv/bin/python -m clusterprep.matched_cohort --output outputs/my_new_phase5
```

Le répertoire de sortie doit être nouveau ; les données sources, anciens splits et caches historiques
ne sont jamais écrasés. Un cache apparié distinct est conservé sous le dossier de sortie.
Les tests automatisés et leurs résultats sont consignés dans [test_results.md](test_results.md).

**Arrêt à la phase 5 : cohorte et seuils en attente de validation scientifique.**
'''
    (output/'cohort_validation_report.md').write_text(text)
