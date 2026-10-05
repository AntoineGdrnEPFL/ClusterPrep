"""Lancement d'une suite décrite en JSON : python -m clusterprep suite.json."""
import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path
from .io import get_data_info_dict, CLASS_NAMES


def preprocessing_main(argv):
    from .processed import prepare_processed
    parser = argparse.ArgumentParser(description="Préparer les FITS sans entraîner de modèle")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--processed-dir", type=Path, default=Path("processed"))
    parser.add_argument("--mode", choices=("raw", "pair", "multimodal"), default="multimodal")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--class-names", nargs="+", default=list(CLASS_NAMES))
    parser.add_argument("--force", action="store_true", help="Recalculer même les FITS valides")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.size < 1:
        parser.error("--size doit être positif")
    info = get_data_info_dict(args.data_dir, args.class_names)
    keys = [k for k, item in info.items() if args.mode != "pair" or item.get("chandra_path")]
    counts = prepare_processed(info, keys, args.mode, args.size, args.processed_dir,
                               verbose=not args.quiet, force=args.force)
    print(json.dumps(counts, indent=2))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "createconfig":
        from .config_wizard import create_config
        parser = argparse.ArgumentParser(description="Créer une configuration d'entraînement interactivement")
        parser.add_argument("--output", type=Path, help="Fichier JSON à créer")
        args = parser.parse_args(argv[1:])
        create_config(args.output)
        return
    if argv and argv[0] == "preprocess":
        preprocessing_main(argv[1:])
        return
    from .training import ExperimentConfig, run_experiments, compare_experiments
    from .models import list_models
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, nargs="?")
    parser.add_argument("--list-models", action="store_true", help="Afficher les modèles disponibles")
    parser.add_argument("--quiet", action="store_true", help="Masquer les barres et résumés")
    parser.add_argument("--processed-dir", type=Path, help="Dossier commun des FITS prétraités")
    args = parser.parse_args(argv)
    if args.list_models:
        print(json.dumps(list_models(), indent=2))
        return
    if args.config is None:
        parser.error("Un fichier de configuration est requis")
    spec = json.loads(args.config.read_text())
    classes = spec.get("class_names", list(CLASS_NAMES))
    info = get_data_info_dict(spec["data_dir"], classes)
    configs = [ExperimentConfig(**item) for item in spec["experiments"]]
    if args.quiet:
        configs = [replace(c, verbose=False) for c in configs]
    if args.processed_dir is not None:
        configs = [replace(c, processed_dir=str(args.processed_dir)) for c in configs]
    if not configs:
        parser.error("La suite doit contenir au moins une expérience")
    output = Path(spec.get("output_dir", "results/experiments"))
    runs = run_experiments(info, configs, output, class_names=classes,
                           **spec.get("split", {}))
    # Le fichier de comparaison appartient à cette suite et ne remplace pas une autre.
    table = output / f"comparison-{runs[0].name}.csv"
    default_cohort = "paired" if any(c.mode == "pair" for c in configs) else "all"
    compare_experiments(runs, table, cohort=spec.get("comparison_cohort", default_cohort))
    print(json.dumps({"runs": list(map(str, runs)), "comparison": str(table)}, indent=2))


if __name__ == "__main__":
    main()
