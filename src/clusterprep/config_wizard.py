"""Assistant interactif de création des suites JSON d'entraînement."""
from dataclasses import asdict, fields, replace
import json
import math
import os
from pathlib import Path
import shlex
import tempfile
from .io import CLASS_NAMES
from .models import build_model, list_models
from .training import ExperimentConfig


def create_config(output_path=None, *, input_fn=None, print_fn=None):
    """Demande les paramètres et écrit un JSON ; renvoie son Path ou None si annulé.

    Aucun entraînement ni prétraitement n'est lancé. Les valeurs par défaut
    des expériences viennent d'ExperimentConfig. input_fn permet de tester
    l'assistant ou de l'intégrer dans une autre interface.
    """
    read = input if input_fn is None else input_fn
    print_fn = print if print_fn is None else print_fn

    def ask(label, default, parser=str, valid=lambda value: True, hint="Valeur invalide"):
        while True:
            shown = json.dumps(default, ensure_ascii=False) if isinstance(default, (dict, bool)) else default
            text = read(f"{label} [{shown if shown is not None else 'aucun'}] : ").strip()
            try:
                value = default if not text else parser(text)
                if not valid(value):
                    raise ValueError(hint)
                return value
            except (ValueError, TypeError) as error:
                print_fn(f"À corriger : {error}")

    def boolean(text):
        if text.lower() in {"o", "oui", "y", "yes", "true", "1"}:
            return True
        if text.lower() in {"n", "non", "no", "false", "0"}:
            return False
        raise ValueError("répondre oui ou non")

    def path(text):
        return str(Path(text).expanduser().resolve())

    def optional_path(text):
        return None if text.lower() in {"none", "null", "aucun"} else path(text)

    def fraction(label, default):
        return ask(label, default, float, lambda v: math.isfinite(v) and 0 < v < 1,
                   "fraction strictement entre 0 et 1")

    def seed(label, default):
        return ask(label, default, int, lambda v: 0 <= v < 2**32, "entier entre 0 et 2**32-1")

    try:
        print_fn("Création d'une configuration clusterprep. Entrée conserve la valeur proposée ; Ctrl+C annule.")
        print_fn("Les chemins sont enregistrés en absolu. Les données ne sont pas chargées.")
        data_dir = ask("Dossier des données classifiées", path("scratch/data/PSZ2/classified"), path)
        output_dir = ask("Dossier des résultats", path("results/experiments"), path)
        classes = ask("Classes, séparées par des virgules", list(CLASS_NAMES),
                      lambda text: [s.strip() for s in text.split(",")],
                      lambda v: len(v) >= 2 and all(v) and len(set(v)) == len(v),
                      "au moins deux classes distinctes et non vides")
        split_seed = seed("Graine des splits", 42)
        folds = ask("Nombre de folds (aucun = séparation classique)", None,
                    lambda text: None if text.lower() in {"aucun", "none", "null"} else int(text),
                    lambda v: v is None or v >= 2, "au moins 2 folds, ou aucun")
        split = {"split_seed": split_seed}
        if folds is None:
            test_size = fraction("Fraction test", .2)
            split["test_size"] = test_size
        else:
            test_size = 1 / folds
            split.update(n_folds=folds, fold_index=ask("Index du fold de test", 0, int,
                         lambda v: 0 <= v < folds, f"entier de 0 à {folds - 1}"))
        split["val_size"] = ask("Fraction validation (du jeu complet)", .2, float,
            lambda v: math.isfinite(v) and 0 < v < 1 - test_size,
            "fraction positive ; test + validation doit être inférieur à 1")

        experiments = []
        available = list_models()
        defaults = ExperimentConfig(name="experience_1")
        labels = {
            "size": "Taille des images", "width": "Largeur du CNN (width)",
            "batch_size": "Taille des lots", "epochs": "Nombre maximal d'époques",
            "learning_rate": "Learning rate", "weight_decay": "Weight decay",
            "augment": "Data augmentation (oui/non)", "cache": "Cache mémoire (oui/non)",
            "processed_dir": "Cache FITS (aucun pour désactiver)", "verbose": "Progression (oui/non)",
            "num_workers": "Nombre de workers", "device": "Device (cpu, mps, cuda ou cuda:0)",
            "monitor": "Early stopping (val_loss, val_macro_f1, val_balanced_accuracy)",
            "patience": "Patience", "min_delta": "Amélioration minimale",
            "class_weight": "Pondération des classes (none/balanced)",
            "use_class_weights": "Activer les poids de classes (oui/non)",
            "label_smoothing": "Label smoothing (0 pour désactiver)",
            "use_scheduler": "Activer OneCycleLR (oui/non)",
            "max_lr_factor": "Facteur du learning rate maximal",
            "warmup_fraction": "Fraction de warmup",
            "use_mixup": "Activer MixUp (oui/non)",
            "mixup_probability": "Probabilité MixUp par batch",
            "mixup_alpha": "Alpha MixUp",
            "gradient_clip_norm": "Norme maximale du gradient (0 pour désactiver)",
            "use_early_stopping": "Activer l'arrêt anticipé (oui/non)",
            "drop_last_train": "Exclure le dernier lot train incomplet (oui/non)",
        }
        while True:
            print_fn(f"\nExpérience {len(experiments) + 1} — modèles : {', '.join(available)}")
            print_fn("Modèle externe accepté : module:Constructeur. Paramètres propres au modèle en JSON.")
            name = ask("Nom de l'expérience", f"experience_{len(experiments) + 1}", str,
                       lambda v: v not in {c['name'] for c in experiments} and _valid_field(defaults, "name", v),
                       "nom unique composé de lettres, chiffres, tirets, points ou underscores")
            mode = ask("Mode (raw, pair, multimodal)", defaults.mode, str,
                       lambda v: v in {"raw", "pair", "multimodal"})
            model = ask("Modèle", defaults.model, str,
                        lambda v: (v in available and mode in available[v]) or
                        (":" in v and all(v.split(":", 1))),
                        "modèle inconnu ou incompatible avec le mode")
            config = replace(defaults, name=name, mode=mode, model=model)
            config = replace(config, seed=seed("Graine du modèle", defaults.seed))
            for item in fields(config):
                key = item.name
                if key in {"name", "mode", "model", "seed", "model_params"}:
                    continue
                default = getattr(config, key)
                if key == "processed_dir":
                    default = path(default) if default is not None else None
                    parser = optional_path
                elif isinstance(default, bool):
                    parser = boolean
                else:
                    parser = type(default)
                def valid(value, key=key):
                    if key == "device":
                        import torch
                        try:
                            return torch.device(value).type in {"cpu", "mps", "cuda"}
                        except (ValueError, RuntimeError):
                            return False
                    return _valid_field(config, key, value)
                value = ask(labels.get(key, key), default, parser, valid)
                config = replace(config, **{key: value})
            params_default = {"hidden_dim": 32, "classifier_hidden_dim": 32} if model == "image_cnn" else {}
            while True:
                params = ask("Paramètres du modèle (objet JSON)", params_default, json.loads,
                             lambda v: isinstance(v, dict), "objet JSON requis")
                try:
                    config = replace(config, model_params=params)
                    # Valide constructeur, paramètres et modes, sans FITS ni transfert GPU.
                    instance = build_model(config, len(classes))
                    import torch
                    if config.batch_size < 2 and any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                                                    for m in instance.modules()):
                        config = replace(config, batch_size=ask("BatchNorm : taille des lots (minimum 2)", 2,
                                         int, lambda v: v >= 2))
                    del instance
                    break
                except (ValueError, TypeError, ImportError, RuntimeError) as error:
                    print_fn(f"Modèle invalide : {error}")
                    model = ask("Modèle à utiliser", model)
                    config = replace(config, model=model)
            experiments.append(asdict(config))
            if not ask("Ajouter une autre expérience (oui/non)", False, boolean):
                break
        paired = any(c["mode"] == "pair" for c in experiments)
        cohorts = {"paired"} if paired else {"all", "paired", "raw_only"}
        cohort = ask("Cohorte de comparaison (" + ", ".join(sorted(cohorts)) + ")",
                     "paired" if paired else "all", str, lambda v: v in cohorts)
        spec = {"data_dir": data_dir, "output_dir": output_dir, "class_names": classes,
                "split": split, "comparison_cohort": cohort, "experiments": experiments}
        destination = Path(output_path).expanduser().resolve() if output_path is not None else Path(
            ask("Fichier JSON à créer", path("training_config.json"), path))
        print_fn(f"\n{len(experiments)} expérience(s), classes {', '.join(classes)}, comparaison {cohort}.")
        for c in experiments:
            print_fn(f"  {c['name']} : {c['model']} / {c['mode']}, {c['epochs']} époques max, device={c['device']}")
        if destination.exists() and not ask(f"{destination} existe. Remplacer (oui/non)", False, boolean):
            print_fn("Annulé : fichier existant conservé.")
            return None
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".config-", suffix=".json", dir=destination.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(spec, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
        print_fn(f"Configuration enregistrée : {destination}")
        print_fn(f"Lancer : python -m clusterprep {shlex.quote(str(destination))}")
        return destination
    except (EOFError, KeyboardInterrupt):
        print_fn("\nCréation annulée ; aucun fichier écrit.")
        return None


def _valid_field(config, key, value):
    try:
        replace(config, **{key: value})
        return True
    except (ValueError, TypeError):
        return False
