"""Explicit candidate inclusion for binary suites with saved subtype splits."""
import pytest
import json

from clusterprep.__main__ import _map_data_classes
from clusterprep.labels import task_target


def test_candidates_are_positive_with_source_annotations_preserved():
    info = {
        "candidate": {"labels": {"cDE"}},
        "confirmed": {"labels": {"DE"}},
        "negative": {"labels": {"NDE"}},
    }
    result = _map_data_classes(info, {"DE": "DE", "cDE": "DE", "NDE": "NDE"})
    assert result is info
    assert [task_target(item, "de_nde")[0] for item in result.values()] == [0, 0, 1]
    assert result["candidate"]["source_labels"] == ["cDE"]


def test_mapping_rejects_candidate_and_negative_contradiction():
    with pytest.raises(ValueError, match="contradictoires"):
        _map_data_classes({"ambiguous": {"labels": {"cDE", "NDE"}}},
                          {"cDE": "DE", "NDE": "NDE"})


def test_cli_passes_mapped_candidates_and_explicit_splits(monkeypatch, tmp_path):
    from clusterprep.__main__ import main
    mapping = {"DE": "DE", "NDE": "NDE", "cDE": "DE"}
    splits = [["candidate"], ["confirmed"], ["negative"]]
    spec = dict(data_dir="unused", output_dir=str(tmp_path / "results"),
                class_names=["DE", "NDE"], data_class_mapping=mapping,
                split={"splits": splits}, experiments=[{"name": "mapped", "task": "de_nde"}])
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(spec))

    def discover(directory, classes):
        assert tuple(classes) == tuple(mapping)
        return {"candidate": {"labels": {"cDE"}}, "confirmed": {"labels": {"DE"}},
                "negative": {"labels": {"NDE"}}}

    class ReachedTraining(Exception):
        pass

    def run(info, configs, output, **kwargs):
        assert kwargs["splits"] == splits
        assert kwargs["class_names"] == ["DE", "NDE"]
        assert info["candidate"]["labels"] == {"DE"}
        assert info["candidate"]["source_labels"] == ["cDE"]
        raise ReachedTraining

    monkeypatch.setattr("clusterprep.__main__.get_data_info_dict", discover)
    monkeypatch.setattr("clusterprep.__main__.radio_coverage", lambda info: {})
    monkeypatch.setattr("clusterprep.training.run_experiments", run)
    with pytest.raises(ReachedTraining):
        main([str(path)])
