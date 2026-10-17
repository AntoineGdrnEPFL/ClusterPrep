"""Strict multi-source initialization and its independent training lifecycle."""
import copy
from dataclasses import asdict, replace
import json

import pytest
import torch
from torch import nn

from clusterprep import (
    ModularTransferConfig, TransferError, TransferPhase, Trainer, build_model,
    check_module_compatibility, initialize_transfer, resolve_module, run_experiment,
)
from clusterprep.training_core import build_criterion
from test_transfer import config, assert_same, source_run, single_thread, TinyBN


def save_source(tmp_path, name="source", cfg=None, model=None, metadata=True):
    cfg = cfg or config()
    model = model or build_model(cfg, 2)
    path = tmp_path / f"{name}.pt"
    payload = {"model_state_dict": model.state_dict()}
    if metadata:
        payload.update(config=asdict(cfg), class_names=["DE", "NDE"] if cfg.task == "de_nde" else ["RR", "RH"])
    torch.save(payload, path)
    return path, model


def plan(path, mappings=None, **kwargs):
    return ModularTransferConfig(enabled=True, sources={"source": {"checkpoint": str(path)}},
        mappings=mappings or [{"source": "source", "from": "encoder", "to": "radio_encoder"}], **kwargs)


def test_copies_buffers_without_aliasing_or_changing_head_and_rng(tmp_path):
    source, target = TinyBN(), TinyBN()
    source(torch.randn(4, 1, 8, 8))
    path, _ = save_source(tmp_path, model=source)
    before = copy.deepcopy(target.classifier.state_dict())
    rng = torch.random.get_rng_state()
    report = initialize_transfer(target, plan(path), lambda *_: TinyBN())
    assert torch.equal(rng, torch.random.get_rng_state())
    assert report["mappings"][0]["status"] == "transferred"
    assert_same(source.encoder.state_dict(), target.encoder.state_dict())
    assert_same(before, target.classifier.state_dict())
    for key, value in target.encoder.state_dict().items():
        assert value.data_ptr() != source.encoder.state_dict()[key].data_ptr()


@pytest.mark.parametrize("source_task,target_task,source_model,source_mode,target_model,target_mode", [
    ("de_nde", "rr_rh", "cnn", "raw", "fusion", "multimodal"),
    ("rr_rh", "de_nde", "fusion", "multimodal", "cnn", "raw"),
    ("de_nde", "rr_rh", "image_cnn", "raw", "dual_encoder_cnn", "multimodal"),
    ("rr_rh", "de_nde", "dual_encoder_cnn", "multimodal", "image_cnn", "raw"),
])
def test_bidirectional_tasks_architectures_modalities(tmp_path, source_task, target_task, source_model, source_mode, target_model, target_mode):
    src_config = config(task=source_task, model=source_model, mode=source_mode)
    path, source = save_source(tmp_path, cfg=src_config)
    target_config = config(task=target_task, model=target_model, mode=target_mode, transfer=plan(path))
    target = build_model(target_config, 2)
    head = copy.deepcopy(target.classifier.state_dict())
    mapping = [{"source": "source", "from": "radio", "to": "radio"}]
    initialize_transfer(target, plan(path, mapping))
    assert_same(resolve_module(source, "radio").state_dict(), resolve_module(target, "radio").state_dict())
    assert_same(target.classifier.state_dict(), head)
    assert isinstance(build_criterion(target_config, [4, 4], n_train=8), nn.CrossEntropyLoss if target_task == "de_nde" else nn.BCEWithLogitsLoss)
    target.eval()
    assert target(torch.randn(2, 1 if target_mode == "raw" else 3, 8, 8)).shape == (2, 2)


@pytest.mark.parametrize("source,target", [
    (nn.Conv2d(1, 2, 3, stride=1), nn.Conv2d(1, 2, 3, stride=2)),
    (nn.Conv2d(1, 2, 3, padding=0), nn.Conv2d(1, 2, 3, padding=1)),
    (nn.Sequential(nn.Linear(2, 2), nn.ReLU()), nn.Sequential(nn.Linear(2, 2), nn.Sigmoid())),
    (nn.BatchNorm1d(2, eps=1e-5), nn.BatchNorm1d(2, eps=1e-3)),
    (nn.LeakyReLU(.1), nn.LeakyReLU(.2)),
])
def test_same_shape_different_forward_rejected(source, target):
    assert {k: v.shape for k, v in source.state_dict().items()} == {k: v.shape for k, v in target.state_dict().items()}
    with pytest.raises(ValueError, match="architecture"):
        check_module_compatibility(source, target)


def test_custom_modules_require_explicit_signature():
    class Custom(nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale
            self.weight = nn.Parameter(torch.ones(2))
        def forward(self, x):
            return self.scale * x * self.weight
    with pytest.raises(ValueError, match="transfer_signature"):
        check_module_compatibility(Custom(1), Custom(1))
    Custom.transfer_signature = lambda self: {"version": 1, "scale": self.scale}
    check_module_compatibility(Custom(1), Custom(1))
    with pytest.raises(ValueError, match="architecture"):
        check_module_compatibility(Custom(1), Custom(2))


def test_multiple_sources_and_single_load_per_checkpoint(tmp_path, monkeypatch):
    p1, s1 = save_source(tmp_path, "radio")
    p2, s2 = save_source(tmp_path, "multi", cfg=config(model="fusion", mode="multimodal"))
    target = build_model(config(model="fusion", mode="multimodal"), 2)
    cfg = ModularTransferConfig(enabled=True, sources={
        "a": {"checkpoint": str(p1)}, "b": {"checkpoint": str(p2)}, "alias": {"checkpoint": str(p2)}}, mappings=[
        {"source": "a", "from": "encoder", "to": "radio"},
        {"source": "b", "from": "xray", "to": "xray"},
        {"source": "alias", "from": "classifier", "to": "classifier"}])
    calls, load = [], torch.load
    def spy(path, *args, **kwargs):
        calls.append(str(path))
        return load(path, *args, **kwargs)
    monkeypatch.setattr(torch, "load", spy)
    report = initialize_transfer(target, cfg)
    assert len(calls) == 2
    assert_same(target.radio.state_dict(), s1.encoder.state_dict())
    assert_same(target.chandra.state_dict(), s2.chandra.state_dict())
    assert_same(target.classifier.state_dict(), s2.classifier.state_dict())
    assert len(report["sources"]) == 3
    assert all(e["status"] == "transferred" for e in report["mappings"])


@pytest.mark.parametrize("second", ["radio", "radio_encoder", "encoder.0"])
def test_conflicts_preflight_without_loading_or_copying(tmp_path, monkeypatch, second):
    target = build_model(config(), 2)
    before = copy.deepcopy(target.state_dict())
    def no_load(*args, **kwargs):
        pytest.fail("Collision must be detected before source loading")
    monkeypatch.setattr(torch, "load", no_load)
    cfg = plan(tmp_path / "missing.pt", [
        {"source": "source", "from": "encoder", "to": "radio"},
        {"source": "source", "from": "encoder", "to": second}])
    with pytest.raises(TransferError, match="collision") as exc:
        initialize_transfer(target, cfg)
    assert exc.value.report["mappings"][-1]["status"] == "error"
    assert_same(before, target.state_dict())


def test_shared_storage_collision(tmp_path):
    model = nn.Module()
    model.a, model.b = nn.Linear(2, 2), nn.Linear(2, 2)
    model.b.weight = nn.Parameter(model.a.weight.data)
    cfg = plan(tmp_path / "missing.pt", [
        {"source": "source", "from": "a", "to": "a"},
        {"source": "source", "from": "b", "to": "b"}])
    with pytest.raises(TransferError, match="collision"):
        initialize_transfer(model, cfg)


def test_policies_local_override_and_partial_skips(tmp_path):
    path, source = save_source(tmp_path)
    target = build_model(config(model="fusion", mode="multimodal"), 2)
    original = copy.deepcopy(target.chandra.state_dict())
    cfg = plan(path, [
        {"source": "source", "from": "missing", "to": "missing", "on_missing": "warn"},
        {"source": "source", "from": "encoder", "to": "radio"},
        {"source": "source", "from": "encoder", "to": "xray", "on_incompatible": "skip"}],
        defaults={"on_missing": "error", "on_incompatible": "error"})
    with pytest.warns(UserWarning, match="absent"):
        report = initialize_transfer(target, cfg)
    assert [e["status"] for e in report["mappings"]] == ["skipped", "transferred", "skipped"]
    assert_same(original, target.chandra.state_dict())
    assert_same(source.encoder.state_dict(), target.radio.state_dict())


def test_fatal_incompatibility_does_not_copy_earlier_mapping(tmp_path):
    path, _ = save_source(tmp_path)
    target = build_model(config(model="fusion", mode="multimodal"), 2)
    before = copy.deepcopy(target.state_dict())
    cfg = plan(path, [{"source": "source", "from": "encoder", "to": "radio"},
                      {"source": "source", "from": "encoder", "to": "xray"}], defaults={"on_incompatible": "error"})
    with pytest.raises(TransferError, match="architecture"):
        initialize_transfer(target, cfg)
    assert_same(before, target.state_dict())


@pytest.mark.parametrize("mapping,policy", [
    ({"from": "missing", "to": "radio"}, "on_missing"),
    ({"from": "encoder", "to": "missing"}, "on_missing"),
    ({"from": "classifier", "to": "radio"}, "on_incompatible"),
])
@pytest.mark.parametrize("action", ["error", "warn", "skip"])
def test_no_success_always_fails(tmp_path, mapping, policy, action):
    path, _ = save_source(tmp_path)
    cfg = plan(path, [{"source": "source", **mapping, policy: action}])
    with pytest.raises(TransferError) as exc:
        if action == "warn":
            with pytest.warns(UserWarning):
                initialize_transfer(build_model(config(), 2), cfg)
        else:
            initialize_transfer(build_model(config(), 2), cfg)
    assert exc.value.report["mappings"][0]["status"] == ("error" if action == "error" else "skipped")


@pytest.mark.parametrize("change", [
    {"enabled": 1}, {"defaults": {"compatibility": "loose"}},
    {"defaults": {"on_conflict": "overwrite"}}, {"sources": []},
    {"sources": {"source": {"checkpoint": ""}}}, {"mappings": []},
    {"mappings": [{"source": "unknown", "from": "a", "to": "b"}]},
    {"mappings": [{"source": "source", "from": "a..x", "to": "b"}]},
    {"mappings": [{"source": "source", "from": "a", "to": "b", "on_missing": "ignore"}]},
])
def test_invalid_json_configuration(change):
    raw = dict(enabled=True, sources={"source": {"checkpoint": "unused.pt"}},
               mappings=[{"source": "source", "from": "encoder", "to": "radio"}])
    raw.update(change)
    with pytest.raises(ValueError):
        config(transfer=raw)


@pytest.mark.parametrize("value", [None, {}, {"enabled": False, "sources": {"invalid": {"checkpoint": "missing.pt"}}},
                                         {"sources": {"source": {"checkpoint": "missing.pt"}}}])
def test_disabled_or_absent_keeps_standard_optimizer(value):
    cfg = config(transfer=value)
    model = build_model(cfg, 2)
    trainer = Trainer(model, cfg, ["DE", "NDE"], [4, 4], steps_per_epoch=2)
    assert trainer.optimizer is not None
    if cfg.transfer is not None:
        before = copy.deepcopy(model.state_dict())
        assert initialize_transfer(model, cfg.transfer)["enabled"] is False
        assert_same(before, model.state_dict())
    assert asdict(config(**{k: v for k, v in asdict(cfg).items()})) == asdict(cfg)


def test_old_checkpoint_requires_metadata_and_accepts_explicit_config(tmp_path):
    path, source = save_source(tmp_path, metadata=False)
    target = build_model(config(), 2)
    with pytest.raises(TransferError, match="config_path"):
        initialize_transfer(target, plan(path))
    cfgpath = tmp_path / "explicit.json"
    cfgpath.write_text(json.dumps(asdict(config())))
    cfg = plan(path)
    cfg.sources["source"]["config_path"] = str(cfgpath)
    report = initialize_transfer(target, cfg)
    assert report["sources"]["source"]["metadata_origin"] == str(cfgpath)
    assert_same(source.encoder.state_dict(), target.encoder.state_dict())
    # Historical adjacent metadata remains supported as well.
    (tmp_path / "config.json").write_text(cfgpath.read_text())
    initialize_transfer(target, plan(path))


def test_resume_is_not_transfer(tmp_path):
    target = build_model(config(), 2)
    with pytest.raises(TransferError, match="mutually exclusive"):
        initialize_transfer(target, plan(tmp_path / "unused.pt"), resume=True)


@pytest.mark.parametrize("source_name", ["image_cnn", "dual_ssn"])
def test_custom_encoder_signatures_and_scattering(tmp_path, source_name):
    if source_name == "dual_ssn":
        pytest.importorskip("kymatio")
    cfg = config(model=source_name)
    path, source = save_source(tmp_path, cfg=cfg)
    target = build_model(cfg, 2)
    initialize_transfer(target, plan(path))
    assert_same(source.encoder.state_dict(), target.encoder.state_dict())


def test_dataparallel_alias_resolution_and_prefix_checkpoint(tmp_path):
    source = nn.DataParallel(build_model(config(), 2))
    path, _ = save_source(tmp_path, model=source)
    target = nn.DataParallel(build_model(config(), 2))
    initialize_transfer(target, plan(path))
    assert_same(source.module.encoder.state_dict(), target.module.encoder.state_dict())


def test_nonpersistent_buffers_copied(tmp_path):
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scratch", torch.ones(2), persistent=False)
            self.weight = nn.Parameter(torch.randn(2))
        def transfer_signature(self):
            return {"version": 1}
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
    source, target = Model(), Model()
    path, _ = save_source(tmp_path, model=source)
    target.encoder.scratch.fill_(4)
    initialize_transfer(target, plan(path, [{"source": "source", "from": "encoder", "to": "encoder"}]), lambda *_: Model())
    assert torch.equal(target.encoder.scratch, torch.ones(2))


def test_run_phases_follow_initialization_and_artifacts(source_run, tmp_path, monkeypatch):
    source, info = source_run
    cfg = config(task="rr_rh", transfer=plan(source / "best.pt"), training_phases=[
        TransferPhase("head", 1, {"head": .001}),
        TransferPhase("full", 1, {"head": .001, "radio": .0001})])
    from clusterprep import read_checkpoint_state
    source_encoder = {k.removeprefix("encoder."): v for k, v in read_checkpoint_state(source / "best.pt").items() if k.startswith("encoder.")}
    captured = []
    original = Trainer.__init__
    def spy(self, model, *args, **kwargs):
        assert_same(model.encoder.state_dict(), source_encoder)
        captured.append(True)
        original(self, model, *args, **kwargs)
    monkeypatch.setattr(Trainer, "__init__", spy)
    run = run_experiment(info, cfg, tmp_path / "target")
    assert captured
    report = json.loads((run / "transfer.json").read_text())
    assert report["mappings"][0]["status"] == "transferred"
    assert report["sources"]["source"]["preprocessing"]["size"] == 8
    assert (run / "training_phases.json").exists()
    head_phase = read_checkpoint_state(run / "phases/head/best.pt")
    assert_same(source_encoder, {k.removeprefix("encoder."): v for k, v in head_phase.items() if k.startswith("encoder.")})
    checkpoint = torch.load(run / "best.pt", weights_only=True)
    assert checkpoint["config"]["task"] == "rr_rh"
    assert checkpoint["transfer_report"]["mappings"] == report["mappings"]
    assert json.loads((run / "metrics.json").read_text())["transfer"] == report


def test_run_failure_report_persisted(source_run, tmp_path):
    source, info = source_run
    cfg = config(task="rr_rh", transfer=plan(source / "best.pt", [
        {"source": "source", "from": "absent", "to": "radio", "on_missing": "error"}]))
    with pytest.raises(TransferError):
        run_experiment(info, cfg, tmp_path / "failed")
    run, = (tmp_path / "failed").iterdir()
    assert json.loads((run / "status.json").read_text())["status"] == "failed"
    report = json.loads((run / "transfer.json").read_text())
    assert report["mappings"][0]["status"] == "error"
    assert not (run / "best.pt").exists()


def test_independent_phases_without_transfer():
    cfg = config(training_phases=[TransferPhase("head", 1, {"head": .01})])
    trainer = Trainer(build_model(cfg, 2), cfg, ["DE", "NDE"], [4, 4], steps_per_epoch=2)
    assert trainer.optimizer is None
    loader = torch.utils.data.DataLoader([{"raw": torch.rand(1, 8, 8), "label": i % 2} for i in range(8)], batch_size=4)
    before = copy.deepcopy(trainer.model.encoder.state_dict())
    trainer.fit_transfer(loader, loader)
    assert_same(before, trainer.model.encoder.state_dict())


def test_modular_wizard_for_de_nde(tmp_path):
    from clusterprep.config_wizard import create_config
    path, _ = save_source(tmp_path, cfg=config(task="rr_rh"))
    cfg = plan(path)
    def reply(prompt):
        for prefix, answer in [
            ("Training initialization", "3"),
            ("Transfert modulaire", json.dumps(asdict(cfg))),
            ("Taille des images", "8"), ("Largeur du CNN", "2"),
        ]:
            if prompt.startswith(prefix):
                return answer
        return ""
    destination = tmp_path / "wizard.json"
    def check_message(message):
        assert not message.startswith("Transfert invalide"), message
    create_config(destination, input_fn=reply, print_fn=check_message)
    parsed = config(**json.loads(destination.read_text())["experiments"][0])
    assert parsed.task == "de_nde"
    assert parsed.transfer.enabled
    assert isinstance(parsed.transfer, ModularTransferConfig)


def test_actual_rr_rh_to_de_nde_run_without_phases(source_run, tmp_path):
    source, info = source_run
    rr_run = run_experiment(info, config(task="rr_rh", transfer=plan(source / "best.pt")), tmp_path / "rr")
    de_info = {k: {**v, "labels": {"NDE"} if "NDE" in v["labels"] else {"DE"}} for k, v in info.items()}
    de_run = run_experiment(de_info, config(transfer=plan(rr_run / "best.pt")), tmp_path / "de")
    for run, task in ((rr_run, "rr_rh"), (de_run, "de_nde")):
        checkpoint = torch.load(run / "best.pt", weights_only=True)
        assert checkpoint["config"]["task"] == task
        assert checkpoint["transfer_report"]["mappings"][0]["status"] == "transferred"
        assert not (run / "training_phases.json").exists()
        assert json.loads((run / "status.json").read_text())["status"] == "completed"


def test_disabled_run_equals_absent_transfer(source_run, tmp_path):
    source, info = source_run
    cfg = config(task="rr_rh")
    plain = run_experiment(info, cfg, tmp_path / "plain")
    disabled = run_experiment(info, replace(cfg, transfer={"enabled": False, "sources": {"bad": {"checkpoint": "/absent.pt"}}}), tmp_path / "disabled")
    assert_same(torch.load(plain / "best.pt", weights_only=True)["model_state_dict"],
                torch.load(disabled / "best.pt", weights_only=True)["model_state_dict"])
    assert not (disabled / "transfer.json").exists()


@pytest.mark.parametrize("corruption", ["missing", "shape", "dtype"])
def test_source_checkpoint_corruption_reported_without_copy(tmp_path, corruption):
    path, _ = save_source(tmp_path)
    payload = torch.load(path, weights_only=True)
    key = "encoder.0.weight"
    if corruption == "missing":
        del payload["model_state_dict"][key]
    elif corruption == "shape":
        payload["model_state_dict"][key] = torch.zeros(9)
    else:
        payload["model_state_dict"][key] = payload["model_state_dict"][key].double()
    torch.save(payload, path)
    target = build_model(config(), 2)
    original = copy.deepcopy(target.state_dict())
    with pytest.raises(TransferError, match="Invalid source checkpoint") as exc:
        initialize_transfer(target, plan(path))
    assert exc.value.report["mappings"][0]["status"] == "error"
    assert_same(original, target.state_dict())


def test_same_source_two_mappings_loaded_once(tmp_path, monkeypatch):
    path, source = save_source(tmp_path)
    target = build_model(config(), 2)
    calls, load = [], torch.load
    def spy(path, **kwargs):
        calls.append(path)
        return load(path, **kwargs)
    monkeypatch.setattr(torch, "load", spy)
    initialize_transfer(target, plan(path, [
        {"source": "source", "from": "encoder", "to": "radio"},
        {"source": "source", "from": "classifier", "to": "head"}]))
    assert len(calls) == 1
    assert_same(source.state_dict(), target.state_dict())


def test_example_json_validates():
    from pathlib import Path
    path = Path(__file__).parents[1] / "examples/transfer_modular.json"
    for raw in json.loads(path.read_text())["experiments"]:
        parsed = config(**raw)
        assert parsed.transfer.enabled
        assert parsed.training_phases


@pytest.mark.parametrize('task,names', [('de_nde', ['NDE', 'DE']), ('rr_rh', ['RR', 'RH'])])
@pytest.mark.parametrize('metadata_location', ['checkpoint', 'sidecar', 'explicit'])
def test_legacy_task_restored_from_saved_classes(tmp_path, task, names, metadata_location):
    cfg = config(task=task, model='dual_encoder_cnn')
    path, source = save_source(tmp_path, cfg=cfg)
    payload = torch.load(path, weights_only=True)
    raw = payload.pop('config')
    del raw['task']
    payload.pop('class_names')
    metadata = {'config': raw, 'class_names': names}
    transfer = plan(path, [{'source': 'source', 'from': 'radio_encoder', 'to': 'radio_encoder'}])
    if metadata_location == 'checkpoint':
        payload.update(metadata)
    else:
        config_path = tmp_path / ('config.json' if metadata_location == 'sidecar' else 'legacy.json')
        config_path.write_text(json.dumps(metadata))
        if metadata_location == 'explicit':
            transfer.sources['source']['config_path'] = str(config_path)
    torch.save(payload, path)
    target = build_model(config(model='dual_encoder_cnn', mode='multimodal'), 2)
    source_bytes = path.read_bytes()
    report = initialize_transfer(target, transfer)
    provenance = report['sources']['source']
    assert provenance['task'] == task
    assert provenance['class_names'] == names
    assert provenance['metadata_migrations'] == [{'field': 'task', 'value': task, 'derived_from': 'class_names'}]
    assert 'task' not in raw
    assert path.read_bytes() == source_bytes
    assert_same(source.radio_encoder.state_dict(), target.radio_encoder.state_dict())


@pytest.mark.parametrize('names', [None, [], ['A', 'B'], ['DE', 'DE'], ['DE', 'NDE', 'NDE']])
def test_missing_task_never_guessed_without_unambiguous_labels(tmp_path, names):
    path, _ = save_source(tmp_path)
    payload = torch.load(path, weights_only=True)
    del payload['config']['task']
    payload['class_names'] = names
    torch.save(payload, path)
    with pytest.raises(TransferError, match='explicit task'):
        initialize_transfer(build_model(config(), 2), plan(path))
