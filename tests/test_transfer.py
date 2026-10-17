"""Transfer invariants and end-to-end runs using only synthetic data."""
import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from clusterprep import (
    ExperimentConfig, TransferConfig, TransferPhase, Trainer, build_model,
    run_experiment, run_experiments, validate_transfer_modules,
    load_pretrained_modules_from_state, read_checkpoint_state,
    configure_transfer_phase, resolve_transfer_splits,
)
from clusterprep.transfer import prepare_transfer_source, validate_transfer_plan
from clusterprep.experiment_loader import audit_transfer_splits


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(**kwargs):
    defaults = dict(name="test", model="cnn", size=8, width=2, batch_size=4,
                    device="cpu", epochs=2, augment=False, verbose=False,
                    processed_dir=None, use_mixup=False, use_scheduler=False,
                    use_class_weights=False, drop_last_train=False)
    defaults.update(kwargs)
    return ExperimentConfig(**defaults)


def transfer(source="unused", **kwargs):
    defaults = dict(source_experiment=str(source), module_prefixes={"radio": "encoder"},
                    phases=[TransferPhase("head", 2, {"head": 1e-3}),
                            TransferPhase("full", 2, {"radio": 1e-5, "head": 1e-4})])
    defaults.update(kwargs)
    return TransferConfig(**defaults)


@pytest.mark.parametrize("name", ["cnn", "image_cnn", "fusion", "dual_encoder_cnn", "dual_ssn", "dual_encoder_ssn"])
@pytest.mark.parametrize("mode", ["raw", "pair", "multimodal"])
def test_architecture_partition(name, mode):
    if name == "fusion" and mode == "raw":
        pytest.skip("FusionCNN explicitly requires two modalities")
    if "ssn" in name:
        pytest.importorskip("kymatio")
    model = build_model(config(model=name, mode=mode), 2)
    groups = validate_transfer_modules(model)
    assert "head" in groups
    assert sum(sum(p.numel() for p in m.parameters()) for m in groups.values()) == sum(p.numel() for p in model.parameters())
    if mode == "raw":
        assert "xray" not in groups


def test_json_configuration_and_invalid_plans():
    c = config(task="rr_rh", transfer=asdict(transfer()))
    assert isinstance(c.transfer.phases[0], TransferPhase)
    assert asdict(ExperimentConfig(**json.loads(json.dumps(asdict(c))))) == asdict(c)
    assert config().transfer is None
    for changes in ({"phases": []}, {"phases": [TransferPhase("same", 1, {"head": .01})] * 2},
                    {"module_prefixes": {"head": "classifier"}}, {"seed": -1},
                    {"reuse_splits": False}, {"module_prefixes": {}}):
        with pytest.raises(ValueError):
            transfer(**changes)
    for changes in ({"name": "../oops"}, {"epochs": 1.5}, {"learning_rates": {"head": float("nan")}}):
        values = dict(name="ok", epochs=1, learning_rates={"head": .01})
        values.update(changes)
        with pytest.raises(ValueError):
            TransferPhase(**values)
    with pytest.raises(ValueError, match="rr_rh"):
        config(transfer=transfer())


class TinyBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 3, 1), nn.BatchNorm2d(3), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Linear(3, 2)

    def transfer_modules(self):
        return {"radio": self.encoder, "head": self.classifier}

    def forward(self, x):
        return self.classifier(self.encoder(x))


def assert_same(a, b):
    assert set(a) == set(b)
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


@pytest.mark.parametrize("wrapper", [None, "model_state_dict", "model_state", "state_dict", "model"])
def test_checkpoint_formats_buffers_and_head_preservation(tmp_path, wrapper):
    source, target = TinyBN(), TinyBN()
    source(torch.randn(4, 1, 8, 8))
    state = source.state_dict()
    path = tmp_path / "best.pt"
    torch.save(state if wrapper is None else {wrapper: state}, path)
    head = copy.deepcopy(target.classifier.state_dict())
    assert_same(read_checkpoint_state(path), state)
    report = load_pretrained_modules_from_state(target, path, {"radio": "encoder"})
    assert_same(target.encoder.state_dict(), source.encoder.state_dict())
    assert_same(target.classifier.state_dict(), head)
    assert "1.running_mean" in report["radio"]["buffers"]


def test_strict_errors_and_atomic_load(tmp_path):
    source, target = TinyBN(), TinyBN()
    path = tmp_path / "best.pt"
    original = copy.deepcopy(target.state_dict())
    for corruption in ("shape", "dtype", "missing"):
        state = copy.deepcopy(source.state_dict())
        if corruption == "shape":
            state["encoder.0.weight"] = torch.zeros(4, 1, 1, 1)
        elif corruption == "dtype":
            state["encoder.0.weight"] = state["encoder.0.weight"].double()
        else:
            state.pop("encoder.1.running_mean")
        torch.save(state, path)
        with pytest.raises(ValueError):
            load_pretrained_modules_from_state(target, path, {"radio": "encoder"})
        assert_same(original, target.state_dict())
    torch.save(source.state_dict(), path)
    for mapping, match in [({"radio": "absent"}, "available prefixes"),
                           ({"head": "classifier"}, "head"),
                           ({"radio": "classifier"}, "head")]:
        with pytest.raises(ValueError, match=match):
            load_pretrained_modules_from_state(target, path, mapping)

    source = build_model(config(model="fusion", mode="multimodal"), 2)
    target = build_model(config(model="fusion", mode="multimodal"), 2)
    state = source.state_dict()
    state["chandra.0.weight"] = state["chandra.0.weight"].double()
    torch.save(state, path)
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(ValueError, match="dtype"):
        load_pretrained_modules_from_state(target, path, {"radio": "radio", "xray": "chandra"})
    assert_same(before, target.state_dict())


def test_legacy_image_encoder_migration(tmp_path):
    c = config(model="image_cnn")
    source, target = build_model(c, 2), build_model(c, 2)
    state = {k.removeprefix("encoder."): v for k, v in source.state_dict().items() if k.startswith("encoder.")}
    state["FC_classifier.weight"] = source.classifier[-1].weight.detach()
    path = tmp_path / "best.pt"
    torch.save(state, path)
    report = load_pretrained_modules_from_state(target, path, {"radio": "encoder"})
    assert_same(target.encoder.state_dict(), source.encoder.state_dict())
    assert report["radio"]["legacy_key_mapping"]


def test_freeze_unfreeze_and_group_rates():
    torch.manual_seed(9)
    model = TinyBN()
    c = config(task="rr_rh")
    trainer = Trainer(model, c, ("RR", "RH"), [4, 4], steps_per_epoch=2, n_train=8)
    loader = DataLoader([{"raw": torch.rand(1, 8, 8), "label": torch.tensor([i % 2, (i // 2) % 2], dtype=torch.float32)} for i in range(8)], batch_size=4)
    encoder = copy.deepcopy(model.encoder.state_dict())
    head = copy.deepcopy(model.classifier.state_dict())
    trainer.optimizer, trainer.frozen_modules = configure_transfer_phase(model, TransferPhase("head", 1, {"head": .01}))
    first_optimizer = trainer.optimizer
    trainer.train_one_epoch(loader)
    assert_same(encoder, model.encoder.state_dict())
    assert any(not torch.equal(v, head[k]) for k, v in model.classifier.state_dict().items())
    trainer.optimizer, trainer.frozen_modules = configure_transfer_phase(model, TransferPhase("all", 1, {"radio": .001, "head": .01}))
    assert trainer.optimizer is not first_optimizer
    assert [g["lr"] for g in trainer.optimizer.param_groups] == [.001, .01]
    trainer.train_one_epoch(loader)
    assert not torch.equal(encoder["1.running_mean"], model.encoder.state_dict()["1.running_mean"])
    assert not torch.equal(encoder["0.weight"], model.encoder.state_dict()["0.weight"])


def test_phase_restore_early_stopping_and_validation_selection(tmp_path, monkeypatch):
    model = TinyBN()
    c = config(task="rr_rh", min_delta=0, transfer=transfer(phases=[
        TransferPhase("head", 4, {"head": .01}, patience=1),
        TransferPhase("all", 4, {"radio": .001, "head": .01}, patience=1)]))
    trainer = Trainer(model, c, ("RR", "RH"), [4, 4], steps_per_epoch=2, n_train=8)
    starts, opts = [], []
    losses = iter([.3, .4, .2, .25])
    def train(*args, **kwargs):
        starts.append(model.classifier.weight.detach().clone())
        opts.append(trainer.optimizer)
        with torch.no_grad():
            model.classifier.weight.add_(1)
        return {"loss": 1., "macro_f1": .5}, []
    monkeypatch.setattr(trainer, "train_one_epoch", train)
    monkeypatch.setattr(trainer, "evaluate", lambda *args, **kwargs: ({"loss": next(losses), "macro_f1": .5}, []))
    initial = model.classifier.weight.detach().clone()
    history = trainer.fit_transfer([1, 2], [1], run_dir=tmp_path)
    assert len(history) == 4
    assert trainer.transfer_report["selected_phase"] == "all"
    assert [p["epochs_ran"] for p in trainer.transfer_report["phases"]] == [2, 2]
    torch.testing.assert_close(starts[2], initial + 1)
    torch.testing.assert_close(model.classifier.weight, initial + 2)
    assert opts[0] is not opts[2]
    assert_same(model.state_dict(), read_checkpoint_state(tmp_path / "best.pt"))
    assert trainer.best_epoch == 3


def test_scheduler_preserves_discriminative_rates():
    from clusterprep.training_core import build_scheduler
    model = TinyBN()
    optimizer, _ = configure_transfer_phase(model, TransferPhase("all", 2, {"radio": 1e-5, "head": 1e-3}))
    scheduler = build_scheduler(optimizer, config(use_scheduler=True), 4)
    assert scheduler is not None
    assert optimizer.param_groups[1]["max_lr"] / optimizer.param_groups[0]["max_lr"] == pytest.approx(100)


@pytest.fixture
def source_run(tmp_path):
    from astropy.io import fits
    from astropy.wcs import WCS
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crpix = [4, 4]
    wcs.wcs.cdelt = [-.01, .01]
    info = {}
    rng = np.random.default_rng(3)
    for i in range(30):
        path = tmp_path / f"{i}.fits"
        fits.PrimaryHDU(rng.normal(size=(8, 8)).astype("float32"), wcs.to_header()).writeto(path)
        info[str(i)] = dict(cluster_id=str(i), raw_path=str(path), chandra_path=str(path),
                            labels={"NDE"} if i % 3 == 0 else {"DE"})
    splits = ([str(i) for i in range(12)], [str(i) for i in range(12, 18)], [str(i) for i in range(18, 24)])
    source = run_experiment({k: v for k, v in info.items() if int(k) < 24}, config(), tmp_path / "runs", splits=splits)
    target = {}
    for key, item in info.items():
        labels = ({"NDE"} if int(key) % 3 == 0 else {"RR"} if int(key) % 3 == 1 else {"RH"}) if int(key) < 24 else ({"cRR"} if int(key) % 2 else {"cRH"})
        target["new_" + key] = {**item, "labels": labels}
    return source, target


@pytest.mark.parametrize("model,mode", [("cnn", "raw"), ("fusion", "multimodal")])
def test_synthetic_pipeline_artifacts_splits_and_single_test(source_run, tmp_path, monkeypatch, model, mode):
    source, info = source_run
    first = {"head": 1e-3, **({"xray": 1e-3} if model == "fusion" else {})}
    plan = transfer(source, phases=[TransferPhase("head", 1, first), TransferPhase("full", 1, {**first, "radio": 1e-5})])
    c = config(task="rr_rh", transfer=plan, model=model, mode=mode, threshold_selection="f1")
    source_bytes = (source / "best.pt").read_bytes()
    calls = []
    evaluate = Trainer.evaluate
    def spy(self, loader, description=None):
        calls.append(description)
        return evaluate(self, loader, description)
    monkeypatch.setattr(Trainer, "evaluate", spy)
    run = run_experiments(info, [c], tmp_path / "targets")[0]
    assert calls.count("Test final") == 1
    report = json.loads((run / "transfer.json").read_text())
    assert report["split_provenance"]["summary"]["new_added_count"] == 6
    manifest = json.loads((run / "splits.json").read_text())
    original = json.loads((source / "splits.json").read_text())
    for name in ("train", "validation", "test"):
        assert {"new_" + k for k in original[name]}.issubset(manifest[name])
    for path in ("best.pt", "config.json", "metrics.json", "history.csv", "history.jsonl", "splits.json",
                 "transfer.json", "transfer_splits_report.json", "validation_predictions.csv", "test_predictions.csv",
                 "phases/head/best.pt", "phases/full/best.pt", "phases/head/history.csv", "phases/full/metrics.json"):
        assert (run / path).is_file()
    assert (source / "best.pt").read_bytes() == source_bytes
    metrics = json.loads((run / "metrics.json").read_text())
    assert np.isfinite(metrics["test"]["loss"])
    assert "diffuse_emission" in metrics["test_cohorts"]
    assert_same({k.removeprefix("encoder."): v for k, v in read_checkpoint_state(source / "best.pt").items() if k.startswith("encoder.")},
                {k.removeprefix("radio." if model == "fusion" else "encoder."): v
                 for k, v in read_checkpoint_state(run / "phases/head/best.pt").items()
                 if k.startswith("radio." if model == "fusion" else "encoder.")})


def test_leakage_and_source_compatibility(source_run):
    source, info = source_run
    splits, report = resolve_transfer_splits(source, info)
    assert report["summary"]["new_added_count"] == 6
    with pytest.raises(ValueError, match="Fuite"):
        audit_transfer_splits(source, info, (splits[2], splits[1], splits[0]))
    selected = (splits[0], splits[2], splits[1])
    assert audit_transfer_splits(source, info, selected)["target_test_in_source_validation"]
    c = config(task="rr_rh", transfer=transfer(source))
    with pytest.raises(ValueError, match="size"):
        prepare_transfer_source(build_model(replace(c, size=16), 2), replace(c, size=16), build_model)
    with pytest.raises(ValueError, match="architecture/channels"):
        prepare_transfer_source(build_model(replace(c, mode="multimodal"), 2), replace(c, mode="multimodal"), build_model)
    with pytest.raises(ValueError, match="conflict"):
        run_experiment(info, c, splits=selected)


def test_split_subset_candidates_and_duplicate_ids(source_run):
    source, info = source_run
    splits, report = resolve_transfer_splits(source, info, add_new_clusters=False)
    assert sum(map(len, splits)) == 24
    assert report["summary"]["new_added_count"] == 0
    missing = dict(info)
    missing.pop("new_1")
    with pytest.raises(ValueError, match="allow_subset"):
        resolve_transfer_splits(source, missing)
    _, subset_report = resolve_transfer_splits(source, missing, allow_subset=True)
    assert subset_report["summary"]["historical_excluded_count"] == 1
    duplicated = {**info, "alias": info["new_0"]}
    with pytest.raises(ValueError, match="dupliqué"):
        resolve_transfer_splits(source, duplicated)
    with pytest.raises(ValueError, match="candidate_policy"):
        resolve_transfer_splits(source, info, candidate_policy="ignore")


def test_wizard_builds_transfer_configuration(source_run, tmp_path):
    from clusterprep.config_wizard import create_config
    source, _ = source_run
    def answer(prompt):
        for prefix, value in [
            ("Tâche", "rr_rh"), ("task", "rr_rh"),
            ("Training initialization", "2"),
            ("Répertoire de l'expérience source", str(source)),
            ("Taille des images", "8"), ("size", "8"),
            ("Largeur", "2"), ("width", "2"),
        ]:
            if prompt.startswith(prefix):
                return value
        return ""
    destination = tmp_path / "wizard.json"
    create_config(destination, input_fn=answer, print_fn=lambda _: None)
    parsed = ExperimentConfig(**json.loads(destination.read_text())["experiments"][0])
    assert parsed.transfer.module_prefixes == {"radio": "encoder"}
    assert len(parsed.transfer.phases) == 2


def test_new_multimodal_branch_cannot_be_frozen():
    model = build_model(config(model="fusion", mode="multimodal"), 2)
    with pytest.raises(ValueError, match="newly initialized.*xray"):
        validate_transfer_plan(model, transfer())


def test_transfer_group_overlap_and_missing_parameters():
    model = TinyBN()
    model.transfer_modules = lambda: {"radio": model.encoder, "head": model.encoder}
    with pytest.raises(ValueError, match="Overlapping"):
        validate_transfer_modules(model)
    model.transfer_modules = lambda: {"radio": model.encoder}
    with pytest.raises(ValueError, match="not in"):
        validate_transfer_modules(model)
