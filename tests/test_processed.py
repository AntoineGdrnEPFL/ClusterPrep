import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from clusterprep import (ClusterDataset, load_processed, processed_path,
                         prepare_processed, preprocess, processed_paths)


class ProcessedTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cache = self.root / "processed"
        self.wcs = WCS(naxis=2)
        self.wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        self.wcs.wcs.crpix = [16, 16]
        self.wcs.wcs.crval = [180, 45]
        self.wcs.wcs.cdelt = [-.01, .01]
        image = np.arange(1024, dtype=np.float32).reshape(32, 32) - 500
        image[:4, :4] = np.nan
        self.info = {"cluster_id": "cluster", "class": "DE"}
        for name in ("raw_path", "chandra_path"):
            path = self.root / f"{name}.fits"
            fits.PrimaryHDU(image, self.wcs.to_header()).writeto(path)
            self.info[name] = str(path)

    def test_roundtrip_physical_values_mask_and_wcs(self):
        fresh = preprocess(self.info, "pair", 8)
        load_processed(self.info, "pair", 8, self.cache)
        with patch("clusterprep.processed.preprocess", side_effect=AssertionError("Recalcul interdit")):
            cached, status = load_processed(self.info, "pair", 8, self.cache, return_status=True)
            raw = load_processed(self.info, "raw", 8, self.cache)
        self.assertEqual(status, "reused")
        for name in ("raw", "chandra", "chandra_mask"):
            np.testing.assert_equal(cached[name], fresh[name])
        np.testing.assert_equal(raw["raw"], fresh["raw"])
        self.assertGreater(np.nanmax(cached["raw"]), 1)
        self.assertLess(np.nanmin(cached["raw"]), 0)
        expected = self.wcs.pixel_to_world_values(1.5, 1.5)
        np.testing.assert_allclose(cached["wcs"].pixel_to_world_values(0, 0), expected)
        with fits.open(processed_path(self.info, "pair", 8, self.cache)) as hdul:
            self.assertFalse(hdul[0].header["NORMALIZ"])
            self.assertFalse(hdul[0].header["AUGMENT"])

    def test_separate_fits_missing_mask_force_and_multiple_sizes(self):
        load_processed(self.info, "pair", 8, self.cache)
        paths = processed_paths(self.info, "pair", 8, self.cache)
        for name, path in paths.items():
            self.assertEqual(path.name, f"cluster_{name.upper()}_processed_size8_v2.fits")
            with fits.open(path) as hdul:
                self.assertEqual(len(hdul), 1)
                self.assertEqual(hdul[0].data.shape, (8, 8))
                self.assertTrue(WCS(hdul[0].header).has_celestial)
                if name == "chandra_mask":
                    self.assertEqual(hdul[0].data.dtype, np.uint8)
                    self.assertTrue(np.isin(hdul[0].data, [0, 1]).all())
        raw_before = paths["raw"].stat().st_mtime_ns
        paths["chandra_mask"].unlink()
        load_processed(self.info, "pair", 8, self.cache)
        self.assertEqual(raw_before, paths["raw"].stat().st_mtime_ns)
        self.assertTrue(paths["chandra_mask"].exists())
        with patch("clusterprep.processed.preprocess", wraps=preprocess) as compute:
            load_processed(self.info, "pair", 8, self.cache, force=True)
            compute.assert_called_once()
        load_processed(self.info, "pair", 16, self.cache)
        with patch("clusterprep.processed.preprocess", side_effect=AssertionError("Recalcul interdit")):
            load_processed(self.info, "pair", 8, self.cache)
            load_processed(self.info, "pair", 16, self.cache)

    def test_preprocessing_cli_without_experiments(self):
        from clusterprep.__main__ import main
        with patch("clusterprep.__main__.get_data_info_dict", return_value={"a": self.info}), \
                patch("clusterprep.training.run_experiments", side_effect=AssertionError("Training interdit")), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            main(["preprocess", str(self.root), "--size", "8", "--processed-dir", str(self.cache), "--quiet"])
        self.assertIn('"created": 1', output.getvalue())
        self.assertEqual(len(list(self.cache.rglob("*.fits"))), 3)

    def test_reuse_across_datasets_and_augmentations(self):
        args = ({"a": self.info}, ["a"], ["DE", "NDE"])
        first = ClusterDataset(*args, mode="pair", size=8, processed_dir=self.cache)[0]
        with patch("clusterprep.processed.preprocess", side_effect=AssertionError("Recalcul interdit")):
            ds = ClusterDataset(*args, mode="multimodal", size=8, augment=True, processed_dir=self.cache)
            for sample in ds:
                self.assertTrue(np.isfinite(sample["raw"].numpy()).all())
            np.testing.assert_equal(first["raw"].numpy(), ds[0]["raw"].numpy())

    def test_invalidation_size_sources_and_corruption(self):
        load_processed(self.info, "raw", 8, self.cache)
        first_path = processed_path(self.info, "raw", 8, self.cache)
        self.assertNotEqual(first_path, processed_path(self.info, "raw", 16, self.cache))
        stat = Path(self.info["raw_path"]).stat()
        os.utime(self.info["raw_path"], ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
        changed_path = processed_path(self.info, "raw", 8, self.cache)
        self.assertEqual(first_path, changed_path)
        _, status = load_processed(self.info, "raw", 8, self.cache, return_status=True)
        self.assertEqual(status, "rebuilt")
        changed_path.write_bytes(b"incomplete")
        _, status = load_processed(self.info, "raw", 8, self.cache, return_status=True)
        self.assertEqual(status, "rebuilt")
        _, status = load_processed(self.info, "raw", 8, self.cache, return_status=True)
        self.assertEqual(status, "reused")

    def test_plus_name_and_legacy_cache_reuse(self):
        self.info["cluster_id"] = "PSZ2G099.48+55.60"
        load_processed(self.info, "pair", 8, self.cache)
        path = processed_path(self.info, "pair", 8, self.cache)
        self.assertEqual(path.parent.name, self.info["cluster_id"])
        legacy_dir = path.parent.with_name("PSZ2G099.48%2B55.60")
        path.parent.rename(legacy_dir)
        with patch("clusterprep.processed.preprocess", side_effect=AssertionError("Recalcul interdit")):
            _, status = load_processed(self.info, "pair", 8, self.cache, return_status=True)
        self.assertEqual(status, "reused")
        self.assertTrue(path.exists())
        self.assertFalse(legacy_dir.exists())
        # Deux dossiers préexistants : récupération sans écrasement des autres fichiers.
        legacy_dir.mkdir()
        path.rename(legacy_dir / path.name)
        with patch("clusterprep.processed.preprocess", side_effect=AssertionError("Recalcul interdit")):
            _, status = load_processed(self.info, "pair", 8, self.cache, return_status=True)
        self.assertEqual(status, "reused")
        self.assertTrue(path.exists())

    def test_preparation_summary_and_warning_log(self):
        def emit_warning(*args, **kwargs):
            warnings.warn("date corrigée", FITSFixedWarning)
            return preprocess(*args, **kwargs)
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream), patch("clusterprep.processed.preprocess", side_effect=emit_warning):
            first = prepare_processed({"a": self.info}, ["a"], "pair", 8, self.cache)
        self.assertEqual(first["created"], 1)
        self.assertEqual(first["fits_warnings"], 1)
        self.assertNotIn("date corrigée", stream.getvalue())
        self.assertNotIn("\r", stream.getvalue())
        self.assertIn("Cache prêt", stream.getvalue())
        self.assertIn("date corrigée", (self.cache / "fits_warnings.log").read_text())
        with contextlib.redirect_stderr(io.StringIO()) as quiet:
            second = prepare_processed({"a": self.info}, ["a"], "pair", 8, self.cache, verbose=False)
        self.assertEqual(second["reused"], 1)
        self.assertEqual(quiet.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
