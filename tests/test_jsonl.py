from repro import jsonl


def test_jsonl_roundtrip_and_append(tmp_path):
    p = tmp_path / "sub" / "a.jsonl"
    assert jsonl.read_jsonl(p) == []
    jsonl.append_jsonl(p, [{"a": 1}, {"b": "é"}])
    jsonl.append_jsonl(p, [{"c": 3}])
    assert jsonl.read_jsonl(p) == [{"a": 1}, {"b": "é"}, {"c": 3}]
