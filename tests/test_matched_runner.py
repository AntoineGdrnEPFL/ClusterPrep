"""Unified orchestration tests: no optimizer steps or training epochs executed."""
from dataclasses import asdict
import json
import pytest
from clusterprep import matched_runner as runner

@pytest.mark.parametrize('experiment',['A','B'])
@pytest.mark.parametrize('fold',range(5))
def test_plan_and_historical_config(experiment,fold):
    plan=runner.load_plan(experiment,fold)
    config=runner.training_configuration(plan,'cpu',200)
    assert config.radio_preprocessing=='matched_resolution'
    assert config.modalities==['radio','xray']
    assert config.augment and config.drop_last_train and config.patience==10
    assert config.threshold_selection=='fixed' and config.thresholds=={'RR':.5,'RH':.5}
    assert config.batch_size==16 and config.scheduler=='onecycle'
    assert config.evaluate_test is False and config.transfer is None
    assert config.normalization['radio']['scope']=='train'
    original=runner.read(runner.ROOT/'clusterprep/configs/rr_rh_multimodal_5folds'/f'fold_{fold}.json')
    ref=next(e for e in original['experiments'] if e['model']=='dual_encoder_cnn')
    for k,v in ref.items():
        if k not in ('name','device'):assert getattr(config,k)==v
    assert runner.training_budget(plan,200)['steps_per_epoch'] in (168,169)

def test_orchestrator_delegates_without_input_or_model_override(tmp_path,monkeypatch):
    calls=[]
    def fake(info,config,output,**kwargs):
        calls.append((info,config,kwargs));return output/'historical-run'
    monkeypatch.setattr(runner,'run_experiment',fake)
    result=runner.run('B',0,42,'cpu',200,tmp_path/'attempt')
    info,c,kw=calls[0]
    assert kw['splits']==tuple(runner.load_plan('B')['partition'][n] for n in ('train','validation','test'))
    assert 'model_factory' not in kw and 'input_adapter' not in kw
    assert all('rr_rh_known_mask' in x for x in info.values())
    assert result.name=='historical-run'
    assert (tmp_path/'attempt/completed.json').exists()

def test_check_never_trains(monkeypatch,capsys):
    monkeypatch.setattr(runner,'run_experiment',lambda *a,**k:pytest.fail('Training requested'))
    runner.main(['check','--experiment','A','--device','cpu'])
    assert json.loads(capsys.readouterr().out)['budget']['steps_per_epoch']==168

def test_legacy_budget_rejected():
    with pytest.raises(ValueError,match='full epochs only'):
        runner.training_budget(runner.load_plan('A'),200,'budgeted')

def test_fold_specific_json_is_actually_consumed(tmp_path,monkeypatch):
    plan=runner.load_plan('A',2)
    template=runner.read(runner.CONFIG_ROOT/'RAW/fold_2.json')
    template['experiments'][0]['batch_size']=32
    folder=tmp_path/'RAW';folder.mkdir()
    (folder/'fold_2.json').write_text(json.dumps(template))
    monkeypatch.setattr(runner,'CONFIG_ROOT',tmp_path)
    config=runner.training_configuration(plan,'cpu',200)
    assert config.batch_size==32
    assert runner.training_budget(plan,200)['steps_per_epoch']==84
