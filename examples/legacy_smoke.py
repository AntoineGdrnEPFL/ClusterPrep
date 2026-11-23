"""Real FITS smoke test; writes only to a new temporary output directory."""
import json
from pathlib import Path
import tempfile
import torch
from torch.utils.data import DataLoader
from clusterprep.io import get_data_info_dict, discover_legacy
from clusterprep.dataset import ClusterDataset
from clusterprep.models import build_model
from clusterprep.training import ExperimentConfig
from clusterprep.processed import load_processed_optical
from clusterprep.visualization import plot_sample


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_dir')
    parser.add_argument('legacy_root')
    args = parser.parse_args()
    torch.set_num_threads(1)
    info = discover_legacy(get_data_info_dict(args.data_dir), args.legacy_root)
    keys = [k for k, i in info.items() if i.get('legacy_path') and i.get('chandra_path')][:3]
    if len(keys) < 2:
        raise ValueError('At least two triply observed clusters required')
    output = Path(tempfile.mkdtemp(prefix='clusterprep-legacy-smoke-'))
    cache = output / 'processed'
    ds = ClusterDataset(info, keys, ['RR', 'RH'], task='rr_rh', size=128,
                        modalities=['radio', 'xray', 'legacy'], processed_dir=cache)
    batch = next(iter(DataLoader(ds, batch_size=len(keys))))
    report = {'keys': keys, 'shape': list(batch['image'].shape), 'models': {}}
    for name in ('image_cnn', 'multi_encoder_cnn'):
        cfg = ExperimentConfig(name=name, model=name, task='rr_rh', size=128,
                               modalities=['radio', 'xray', 'legacy'], device='cpu')
        model = build_model(cfg, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(batch['image']), batch['label'])
        loss.backward()
        optimizer.step()
        model.eval()
        expected = model(batch['image']).detach()
        torch.save(model.state_dict(), output / f'{name}.pt')
        restored = build_model(cfg, 2).eval()
        restored.load_state_dict(torch.load(output / f'{name}.pt', weights_only=True))
        torch.testing.assert_close(expected, restored(batch['image']))
        report['models'][name] = {'loss': loss.item(), 'reload_equal': True}
    report['cache_reuse'] = [load_processed_optical(info[k], 128, cache, bands=['g', 'r', 'z'],
                                                  return_status=True)[1] for k in keys]
    report['coverage'] = batch['legacy_mask'].mean((1, 2, 3)).tolist()
    plot_sample(ds[0], output / 'modalities.png')
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    print(output)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
