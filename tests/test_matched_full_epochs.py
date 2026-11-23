"""The old custom loader is replaced by historical geometric dataset expansion."""
import torch
from clusterprep.augmentation import decode_variant,transform_aligned
from test_matched_dataset import catalog,dataset


def test_all_24_variants_are_aligned_and_labels_preserved(catalog):
    info,cid,_=catalog
    info[cid]['rr_rh_known_mask']=[1,0]
    options=dict(modalities=['radio','xray'],augment=True,training=True)
    a=dataset(catalog,radio_source='RAW',**options)
    b=dataset(catalog,radio_source='T50kpcSUB',**options)
    assert len(a)==len(b)==24
    base=a[0]
    for i in range(24):
        x,y=a[i],b[i]
        angle,flip,scale=decode_variant(i)
        assert torch.equal(x['image'],transform_aligned(base['image'],angle,flip,scale))
        assert torch.equal(x['chandra'],y['chandra'])
        assert torch.equal(x['chandra_mask'],y['chandra_mask'])
        assert torch.equal(x['raw_mask'],y['raw_mask'])
        assert x['label_mask'].tolist()==[1,0]
