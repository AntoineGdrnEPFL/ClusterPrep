import json
import tempfile
import unittest
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS
from torch.utils.data import DataLoader
from clusterprep import (ClusterDataset, get_data_info_dict, load_celestial_fits,
    preprocess, normalize, block_average, split_data_keys, plot_sample, plot_comparison)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.wcs = WCS(naxis=2)
        self.wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        self.wcs.wcs.crpix = [16, 16]
        self.wcs.wcs.crval = [180, 45]
        self.wcs.wcs.cdelt = [-.01, .01]
        self.image = np.arange(1024, dtype=np.float32).reshape(32, 32)
        for modality in ("RAW", "CHANDRA"):
            for label in ("DE", "NDE"):
                (self.root / modality / label).mkdir(parents=True)
        for name, image in (("RAW/DE/a.fits", self.image),
                            ("CHANDRA/DE/aCHANDRA.fits", self.image)):
            fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(image, self.wcs.to_header())]).writeto(self.root / name)
        self.info = get_data_info_dict(self.root)

    def test_data_directory_from_json(self):
        spec = json.loads(json.dumps({"data_dir": str(self.root)}))
        self.assertEqual(get_data_info_dict(spec["data_dir"]), self.info)

    def test_extension_loading_and_discovery(self):
        image, _, wcs = load_celestial_fits(self.info["a.fits"]["raw_path"])
        np.testing.assert_equal(image, self.image)
        self.assertTrue(wcs.has_celestial)
        fits.PrimaryHDU(self.image).writeto(self.root / "RAW/NDE/a.fits")
        with self.assertRaisesRegex(ValueError, "dupliqué"):
            get_data_info_dict(self.root)

    def test_alignment_and_missing_coverage(self):
        processed = preprocess(self.info["a.fits"], "pair", size=8)
        np.testing.assert_allclose(processed["raw"], processed["chandra"], atol=1e-4)
        self.assertTrue(processed["chandra_mask"].all())
        self.wcs.wcs.crval = [0, -45]
        fits.PrimaryHDU(self.image, self.wcs.to_header()).writeto(
            self.info["a.fits"]["chandra_path"], overwrite=True)
        processed = preprocess(self.info["a.fits"], "pair", size=8)
        self.assertFalse(processed["chandra_mask"].any())
        self.assertFalse(normalize(processed["chandra"]).any())

    def test_finite_block_average_and_normalization(self):
        image = np.array([[1., np.nan, np.nan, np.inf], [3., 5., np.nan, np.nan]])
        result = block_average(image, 2)
        self.assertEqual(result[0, 0], 3.)
        self.assertTrue(np.isnan(result[0, 1]))
        for image in (image, np.zeros((2, 2)), np.full((2, 2), np.nan)):
            value = normalize(image)
            self.assertTrue(np.isfinite(value).all())
            self.assertTrue(((value >= 0) & (value <= 1)).all())
        with self.assertRaises(ValueError):
            preprocess(self.info["a.fits"], "raw", size=64)

    def test_batches_augmentations_and_figures(self):
        ds = ClusterDataset(self.info, list(self.info), ["DE", "NDE"],
                            mode="pair", size=8, augment=True, processed_dir=self.root / "processed")
        self.assertEqual(len(ds), 24)
        self.assertEqual(ds.get_item_info(23)["rotation"], 330)
        for sample in ds:
            torch.testing.assert_close(sample["raw"], sample["chandra"], atol=1e-6, rtol=1e-5)
            self.assertTrue(set(sample["chandra_mask"].unique().tolist()) <= {0., 1.})
        batch = next(iter(DataLoader(ds, batch_size=4)))
        self.assertEqual(tuple(batch["raw"].shape), (4, 1, 8, 8))
        self.assertEqual(batch["label"].dtype, torch.long)
        for name, fig in (("sample", plot_sample(ds[0], self.root / "sample.png")),
                          ("comparison", plot_comparison(self.image, self.image,
                           self.wcs, self.wcs, output_path=self.root / "comparison.png"))):
            self.assertGreater((self.root / f"{name}.png").stat().st_size, 1000)
            plt.close(fig)

    def test_splits_grouping_and_fold_validation_fraction(self):
        info = {f"{i}_{j}": {"cluster_id": str(i), "class": "DE" if i < 50 else "NDE",
                            "chandra_path": "available"}
                for i in range(100) for j in range(2)}
        splits = split_data_keys(info, n_folds=5, test_size=.4, val_size=.2)
        self.assertEqual([len(s) for s in splits], [120, 40, 40])
        self.assertEqual(splits, split_data_keys(info, n_folds=5, test_size=.4, val_size=.2))
        groups = [{info[k]["cluster_id"] for k in keys} for keys in splits]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        test_sets = [set(split_data_keys(info, n_folds=5, fold_index=i)[2]) for i in range(5)]
        self.assertEqual(len(set.union(*test_sets)), len(info))
        self.assertEqual(sum(map(len, test_sets)), len(info))


if __name__ == "__main__":
    unittest.main()
