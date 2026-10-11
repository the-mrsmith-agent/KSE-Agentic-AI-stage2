"""Self-checks for src/fetch.py. No network: file:// fixtures only."""

import hashlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.fetch import download, snapshot_summary


def test_download_is_idempotent(tmp_path):
    src = tmp_path / "src.csv"
    src.write_text("region,started_at\nKyiv City,2022-02-25 16:36:22+00:00\n")
    dest = tmp_path / "dest.csv"

    sha1 = download(src.as_uri(), dest)
    mtime1 = dest.stat().st_mtime_ns
    sha2 = download(src.as_uri(), dest)

    assert sha1 == sha2 == hashlib.sha256(dest.read_bytes()).hexdigest()
    assert dest.stat().st_mtime_ns == mtime1


def test_snapshot_summary(tmp_path):
    csv = tmp_path / "fixture.csv"
    csv.write_text(
        "region,started_at,finished_at,naive\n"
        "Kyiv City,2022-02-25 16:36:22+00:00,2022-02-25 17:00:00+00:00,False\n"
        "Lvivska oblast,2023-05-01 01:02:03+00:00,2023-05-01 01:30:00+00:00,True\n"
    )
    summary = snapshot_summary(csv)
    assert summary["rows"] == 2
    assert summary["started_at_min"] == "2022-02-25T16:36:22+00:00"
    assert summary["started_at_max"] == "2023-05-01T01:02:03+00:00"
