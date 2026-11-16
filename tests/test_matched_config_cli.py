import json
import pytest
from clusterprep.__main__ import main
from clusterprep import matched_runner as runner

@pytest.mark.parametrize('product',['RAW','T50kpcSUB'])
@pytest.mark.parametrize('fold',range(5))
def test_json_cli_no_training(product,fold,monkeypatch):
    monkeypatch.setattr(runner,'run_experiment',lambda *a,**kw:pytest.fail('Unexpected training'))
    main([str(runner.CONFIG_ROOT/product/f'fold_{fold}.json'),'--dry-run'])

def test_json_cli_delegates_exact_configuration(monkeypatch):
    calls=[]
    monkeypatch.setattr(runner,'run_experiment',lambda *a,**kw:calls.append((a,kw)))
    path=runner.CONFIG_ROOT/'T50kpcSUB/fold_2.json'
    main([str(path),'--quiet'])
    args,kwargs=calls[0]
    assert args[1].radio_source=='T50kpcSUB' and not args[1].evaluate_test
    assert args[1].augment and args[1].patience==10
    assert kwargs['splits']==tuple(runner.load_plan('B',2)['partition'][n] for n in ('train','validation','test'))
    assert all('rr_rh_known_mask' in x for x in args[0].values())

def test_cli_refuses_external_test_before_training(monkeypatch):
    path=runner.CONFIG_ROOT/'RAW/fold_0.json';spec=runner.read(path)
    spec['experiments'][0]['evaluate_test']=True
    monkeypatch.setattr(runner,'run_experiment',lambda *a,**kw:pytest.fail('Unexpected training'))
    with pytest.raises(ValueError,match='external test'):
        runner.run_config_suite(spec,path)
