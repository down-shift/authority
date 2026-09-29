import pytest

from authority_leakage.inference import read_jsonl, write_jsonl


def test_jsonl_keeps_flushed_rows_when_generation_stops(tmp_path):
    path = tmp_path / "predictions.jsonl"
    saved = []

    def interrupted_rows():
        yield {"example_id": "done-1"}
        raise RuntimeError("simulated inference failure")

    with pytest.raises(RuntimeError, match="simulated inference failure"):
        write_jsonl(path, interrupted_rows(), on_row=lambda: saved.append(True))

    assert saved == [True]
    assert read_jsonl(path) == [{"example_id": "done-1"}]
