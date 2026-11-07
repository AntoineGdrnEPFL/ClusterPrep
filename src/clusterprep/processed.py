"""Cache FITS atomique, avant normalisation et augmentation."""
import json
import os
from pathlib import Path
import tempfile
import time
from urllib.parse import quote
import warnings
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from .preprocessing import preprocess
from .progress import progress, message

CACHE_VERSION = 2


def _identity(info, mode, size):
    sources = {}
    for name in (("raw_path",) if mode == "raw" else ("raw_path", "chandra_path")):
        if not info.get(name):
            raise ValueError(f"Fichier manquant : {name}")
        path = Path(info[name]).resolve()
        stat = path.stat()
        sources[name] = [str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    return {"version": CACHE_VERSION, "mode": mode, "size": size, "sources": sources}


def processed_paths(info, mode="raw", size=128, processed_dir="processed"):
    """Chemins des images séparées, distingués par taille et version."""
    if mode not in {"raw", "pair"} or not isinstance(size, int) or size < 1:
        raise ValueError("mode raw/pair et size entier positif attendus")
    cluster = quote(str(info["cluster_id"]), safe="+")
    if cluster in {"", ".", ".."}:
        raise ValueError("cluster_id invalide")
    types = {"raw": "RAW"}
    if mode == "pair":
        types.update(chandra="CHANDRA", chandra_mask="CHANDRA_MASK")
    return {key: Path(processed_dir) / cluster / f"{cluster}_{kind}_processed_size{size}_v{CACHE_VERSION}.fits"
            for key, kind in types.items()}


def processed_path(info, mode="raw", size=128, processed_dir="processed"):
    """Chemin RAW (également pour une paire). Voir processed_paths pour les trois FITS."""
    return processed_paths(info, mode, size, processed_dir)["raw"]


def _read_image(path, name, identity):
    with fits.open(path, memmap=False, checksum=True) as hdul:
        hdu = hdul[0]
        if hdu.verify_checksum() != 1 or hdu.verify_datasum() != 1:
            raise ValueError("Checksum FITS invalide")
        if hdu.header["CPVER"] != CACHE_VERSION or json.loads(hdu.header["PROVEN"]) != identity:
            raise ValueError("Provenance du cache invalide")
        if hdu.header["IMTYPE"] != name.upper():
            raise ValueError("Type d'image invalide")
        data = np.array(hdu.data, dtype=bool if name == "chandra_mask" else np.float32)
        if data.shape != (identity["size"], identity["size"]):
            raise ValueError("Forme du cache invalide")
        metadata = ({k: hdu.header[k] for k in ("BMAJ", "BMIN", "BPA", "BUNIT") if k in hdu.header}
                    if hdu.header.get("RNMETA", False) else None)
        return data, hdu.header["DSFACTOR"], WCS(hdu.header).celestial, metadata


def _write_image(path, name, result, identity):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = result["wcs"].to_header()
    if name == "raw" and "radio_metadata" in result:
        header["RNMETA"] = True
        for key, value in result["radio_metadata"].items():
            header[key] = value
    header["CPVER"] = CACHE_VERSION
    header["DSFACTOR"] = result["factor"]
    header["NORMALIZ"] = False
    header["AUGMENT"] = False
    header["IMTYPE"] = name.upper()
    header["PROVEN"] = json.dumps(identity, sort_keys=True)
    data = result[name].astype(np.uint8 if name.endswith("_mask") else np.float32)
    fd, temp = tempfile.mkstemp(prefix=".building-", suffix=".fits", dir=path.parent)
    os.close(fd)
    try:
        fits.PrimaryHDU(data, header).writeto(temp, overwrite=True, checksum=True)
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def load_processed(info, mode="raw", size=128, processed_dir="processed", *,
                   return_status=False, force=False):
    """Charge les FITS séparés, prépare les absents/invalides ; force recalcule tout.

    Les sources restent nécessaires pour valider la provenance. Valeurs physiques,
    NaN et WCS sont conservés ; normalisation et augmentation restent au Dataset.
    """
    paths = processed_paths(info, mode, size, processed_dir)
    identities = {name: _identity(info, "raw" if name == "raw" else "pair", size)
                  for name in paths}
    legacy_dir = Path(processed_dir) / quote(str(info["cluster_id"]), safe="")
    parent = paths["raw"].parent
    if legacy_dir != parent and legacy_dir.is_dir() and not parent.exists():
        try:
            legacy_dir.rename(parent)
        except FileNotFoundError:
            if not parent.is_dir():
                raise
    result, pending = {}, []
    existed = False
    for name, path in paths.items():
        read_path = path if path.exists() else legacy_dir / path.name
        existed |= read_path.exists()
        if read_path.exists() and not force:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    data, factor, wcs, metadata = _read_image(read_path, name, identities[name])
                result.update({name: data, "factor": factor, "wcs": wcs})
                if name == "raw" and metadata is not None:
                    result["radio_metadata"] = metadata
                if read_path != path:
                    _write_image(path, name, result, identities[name])
                continue
            except (OSError, ValueError, KeyError, IndexError, TypeError, Warning):
                pass
        pending.append(name)
    if not pending:
        return (result, "reused") if return_status else result
    fresh = preprocess(info, mode, size)
    for name, identity in identities.items():
        if identity != _identity(info, "raw" if name == "raw" else "pair", size):
            raise RuntimeError("Les FITS sources ont changé pendant le prétraitement")
    for name in pending:
        _write_image(paths[name], name, fresh, identities[name])
    status = "rebuilt" if existed else "created"
    return (fresh, status) if return_status else fresh


def prepare_processed(data_info, keys, mode="raw", size=128, processed_dir="processed",
                      verbose=True, warning_log=None, *, force=False):
    """Prépare une fois les FITS requis avant de créer les workers d'entraînement."""
    if mode not in {"raw", "pair", "multimodal"}:
        raise ValueError("Mode inconnu")
    started = time.perf_counter()
    counts = {"created": 0, "reused": 0, "rebuilt": 0, "fits_warnings": 0}
    log_path = Path(warning_log) if warning_log else Path(processed_dir) / "fits_warnings.log"
    keys = list(dict.fromkeys(keys))
    message(f"Cache FITS : {len(keys)} amas, {size}×{size}, dossier {processed_dir}", verbose)
    with progress(keys, "Prétraitement FITS", verbose) as bar:
        for key in bar:
            info = data_info[key]
            actual_mode = "pair" if mode != "raw" and info.get("chandra_path") else "raw"
            if mode == "pair" and actual_mode != "pair":
                raise ValueError(f"Chandra manquant : {key}")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", FITSFixedWarning)
                _, status = load_processed(info, actual_mode, size, processed_dir, return_status=True, force=force)
            for warning in caught:
                if issubclass(warning.category, FITSFixedWarning):
                    counts["fits_warnings"] += 1
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with log_path.open("a") as stream:
                        stream.write(f"{key}: {warning.message}\n")
                else:
                    warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)
            counts[status] += 1
            bar.set_postfix(créés=counts["created"], réutilisés=counts["reused"], refresh=False)
    counts["seconds"] = time.perf_counter() - started
    message(f"Cache prêt : {counts['created']} créés, {counts['reused']} réutilisés, "
            f"{counts['rebuilt']} reconstruits ({counts['seconds']:.1f} s).", verbose)
    if counts["fits_warnings"]:
        message(f"{counts['fits_warnings']} avertissements de métadonnées FITS → {log_path}", verbose)
    return counts


def load_processed_optical(info, size=128, processed_dir="processed", *, bands=None,
                           hdu=None, force=False, return_status=False):
    """Independent optical cache: no writes to historical RAW/CHANDRA products."""
    import hashlib
    from .preprocessing import preprocess_optical
    options = {"bands": bands, "hdu": hdu}
    identity = _identity(info, "raw", size)
    source = Path(info["legacy_path"]).resolve()
    stat = source.stat()
    identity.update(mode="legacy", optical_version=1, options=options,
                    optical_source=[str(source), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns])
    suffix = hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()[:12]
    base = processed_paths(info, "raw", size, processed_dir)["raw"]
    paths = {name: base.with_name(base.name.replace("RAW", name.upper()).replace(".fits", f"_{suffix}.fits"))
             for name in ("legacy", "legacy_mask")}
    existed = any(p.exists() for p in paths.values())
    if not force:
        try:
            result = {}
            for name, path in paths.items():
                with fits.open(path, checksum=True) as hdul:
                    item = hdul[0]
                    if (item.verify_checksum() != 1 or item.verify_datasum() != 1
                            or json.loads(item.header["PROVEN"]) != identity
                            or item.header["IMTYPE"] != name.upper()
                            or item.header["CPVER"] != CACHE_VERSION):
                        raise ValueError("Invalid optical cache")
                    data = np.array(item.data, dtype=bool if name.endswith("mask") else np.float32)
                    if data.ndim != 3 or data.shape[1:] != (size, size):
                        raise ValueError("Invalid optical cache dimensions")
                    result.update({name: data, "factor": item.header["DSFACTOR"],
                                   "wcs": WCS(item.header).celestial})
            if result["legacy"].shape != result["legacy_mask"].shape:
                raise ValueError("Inconsistent optical mask")
            return (result, "reused") if return_status else result
        except (OSError, ValueError, KeyError, IndexError):
            pass
    result = preprocess_optical(info, size, bands, hdu)
    if source.stat().st_mtime_ns != stat.st_mtime_ns or _identity(info, "raw", size)["sources"] != identity["sources"]:
        raise RuntimeError("Sources changed during optical preprocessing")
    for name, path in paths.items():
        _write_image(path, name, result, identity)
    status = "rebuilt" if existed else "created"
    return (result, status) if return_status else result
