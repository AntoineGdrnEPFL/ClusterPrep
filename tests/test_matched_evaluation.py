from pathlib import Path
import pytest
from clusterprep import matched_evaluation as ev
from clusterprep.__main__ import main

ROOT=Path(__file__).resolve().parents[2]/'outputs/phase7b_multimodal_cli'

def test_dry_run_all_models_never_evaluates(monkeypatch,tmp_path):
    monkeypatch.setattr(ev,'evaluate',lambda *a,**k:pytest.fail('Test data accessed'))
    main(['evaluate','--runs-root',str(ROOT),'--output-root',str(tmp_path/'unused'),'--dry-run'])
    assert not (tmp_path/'unused').exists()

def test_leakage_is_rejected(monkeypatch):
    run=next(ROOT.glob('*/*/*/best.pt')).parent
    original=ev.read
    def read(path):
        d=original(path)
        if Path(path).name=='splits.json':d['train'].append(d['reserved_external_test'][0])
        return d
    monkeypatch.setattr(ev,'read',read)
    with pytest.raises(ValueError,match='overlap'):ev.prepare(run)

def test_frozen_evaluation_exports_with_synthetic_loader(monkeypatch,tmp_path):
    run=next(ROOT.glob('*/*/*/best.pt')).parent
    p=ev.prepare(run)
    class Dataset:
        def __init__(self,info,keys,*a,**kw):
            self.keys=keys
            assert kw['augment'] is False and kw['training'] is False
            assert kw['normalization_bounds']==p['config'].normalization_bounds
        def __len__(self):return len(self.keys)
    class Engine:
        def __init__(self,model,config,*a,**kw):
            assert config.thresholds==p['thresholds']
            assert config.use_scheduler is False
        def evaluate(self,loader):
            return {'n_samples':len(loader.dataset)},[dict(cluster_id=k,key=k,has_chandra=True,
                label=[0,0],label_mask=[1,1],probabilities=[.2,.3],prediction=[0,0],loss=.1) for k in loader.dataset.keys]
    monkeypatch.setattr(ev,'ClusterDataset',Dataset)
    monkeypatch.setattr(ev,'Trainer',Engine)
    out=tmp_path/'evaluation';ev.evaluate(p,out,'cpu')
    assert (out/'test_predictions.csv').exists()
    assert ev.read(out/'evaluation.json')['thresholds_refitted'] is False
    with pytest.raises(FileExistsError):ev.evaluate(p,out,'cpu')


def test_analysis_export_without_inference(monkeypatch, tmp_path):
    import shutil
    from clusterprep import analysis as a
    run = sorted(ROOT.glob('*/*/*/best.pt'))[0].parent
    p = ev.prepare(run)
    saved = ROOT.parent/'phase7b_external_test'/run.name
    out = tmp_path/run.name
    out.mkdir()
    for name in ('evaluation.json', 'test_metrics.json', 'test_predictions.csv'):
        shutil.copy2(saved/name, out/name)
    monkeypatch.setattr(ev, 'evaluate', lambda *a, **k: pytest.fail('Inference called'))
    monkeypatch.setattr(ev.torch, 'load', lambda *a, **k: pytest.fail('Checkpoint loaded'))
    ev.export_analysis(p, out)
    loaded = a.load_run(out)
    assert loaded.fold == 0
    assert len(loaded.predictions['test']) == 13
    assert loaded.history
    assert loaded.thresholds == p['thresholds']
    assert not set(loaded.splits['test']) & set(loaded.splits['train'] + loaded.splits['validation'])
    assert a.metrics_table([loaded], 'test', 'all').shape[0] == 1
    assert not ev.read(run/'splits.json')['test']
