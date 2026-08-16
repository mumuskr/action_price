import importlib.util
import json
from pathlib import Path


def load_write_manifest():
    script_path = Path(__file__).parents[1] / "scripts" / "download_us_indexes.py"
    spec = importlib.util.spec_from_file_location("download_us_indexes", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.write_manifest


def test_write_manifest_merges_existing_symbols(tmp_path) -> None:
    write_manifest = load_write_manifest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps([{"symbol": "SPY", "bars": {"1m": 10}}]))

    write_manifest([{"symbol": "QQQ", "bars": {"1m": 20}}], path)

    assert json.loads(path.read_text()) == [
        {"symbol": "SPY", "bars": {"1m": 10}},
        {"symbol": "QQQ", "bars": {"1m": 20}},
    ]
