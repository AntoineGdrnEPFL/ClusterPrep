import importlib.util
from pathlib import Path
from types import SimpleNamespace
import json
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('phase7b_script',ROOT/'clusterprep/run_matched_5folds.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

def test_fixed_external_test_and_shared_folds():
    plans=runner.jobs(runner.SOURCE,42)
    assert len(plans)==10
    seen=[]
    for i in range(5):
        a,b=plans[i],plans[i+5]
        assert a['partition']==b['partition']
        part=a['partition'];seen+=part['validation']
        assert len(part['test'])==13
        assert len(part['train'])+len(part['validation'])==141
        assert set(a['normalization']['train_keys'])==set(part['train'])
    assert len(seen)==len(set(seen))==141

def test_dry_run_never_launches(tmp_path,monkeypatch):
    monkeypatch.setattr(runner.subprocess,'run',lambda *a,**k:pytest.fail('Training launched'))
    assert runner.main(['--device','cpu','--output',str(tmp_path/'out')])==0
    assert not (tmp_path/'out').exists()

def test_failure_stops_at_first_fold(tmp_path,monkeypatch):
    calls=[]
    def fail(command,**kwargs):calls.append(command);return SimpleNamespace(returncode=7)
    monkeypatch.setattr(runner.subprocess,'run',fail)
    assert runner.main(['--run','--device','cpu','--output',str(tmp_path)])==7
    assert len(calls)==1
    assert calls[0][calls[0].index('--fold')+1]=='0'

def test_resume_skips_complete_and_preserves_incomplete(tmp_path,monkeypatch):
    plans=runner.jobs(runner.SOURCE,42)
    first=tmp_path/'A_fold0_seed42'/'attempt_001';first.mkdir(parents=True)
    training=runner.asdict(runner.training_configuration(plans[0],'cpu',200))
    (first/'request.json').write_text(json.dumps(dict(plan=plans[0],training=training,budget=runner.training_budget(plans[0],200,'full'))))
    result=first/'historical-run';result.mkdir()
    (first/'completed.json').write_text(json.dumps(dict(run_dir=str(result),pipeline='historical_matched_v1')))
    (result/'status.json').write_text(json.dumps(dict(status='completed',test_evaluated=False)))
    for f in runner.REQUIRED:(result/f).write_text('fixture')
    incomplete=tmp_path/'A_fold1_seed42'/'attempt_001';incomplete.mkdir(parents=True)
    calls=[]
    def fail(command,**kwargs):calls.append(command);return SimpleNamespace(returncode=9)
    monkeypatch.setattr(runner.subprocess,'run',fail)
    assert runner.main(['--run','--device','cpu','--output',str(tmp_path)])==9
    assert len(calls)==1 and calls[0][calls[0].index('--fold')+1]=='1'
    assert calls[0][-1].endswith('attempt_002')
    assert incomplete.exists()
    assert not runner.completed(first,plans[0],{**training,'epochs':201})
    assert not runner.completed(first,plans[0],training,{'pipeline':'old_budgeted'})
