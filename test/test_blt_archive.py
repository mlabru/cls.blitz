import datetime
import gzip

import pytest

import work.blt_archive as arc

# 2026-11-07 12:00 UTC
TS_DAY = datetime.datetime(2026, 11, 7, 12, 0, tzinfo=datetime.timezone.utc).timestamp()


def _rows(*triples):
    return [{"timestamp": t, "lat": la, "lng": lo} for t, la, lo in triples]


def test_archive_path_uses_utc_day_of_the_run(tmp_path):
    assert arc.archive_path(str(tmp_path), TS_DAY).endswith("slm_raw_20261107.csv.gz")
    # 23:59 UTC ainda é o mesmo dia; meia-noite é o seguinte
    assert arc.archive_path("d", TS_DAY + 12 * 3600 - 1).endswith("20261107.csv.gz")
    assert arc.archive_path("d", TS_DAY + 12 * 3600).endswith("20261108.csv.gz")


def test_append_rows_appends_members_and_load_reads_all(tmp_path):
    archive_dir = tmp_path / "archive"                     # criado pelo append

    path = arc.append_rows(str(archive_dir), _rows((1000, -20.5, -45.25)), TS_DAY)
    arc.append_rows(str(archive_dir), _rows((1060, -20.0, -45.0), (1060, -20.0, -45.0)),
                    TS_DAY + 300)

    with gzip.open(path, "rt") as fh:
        lines = fh.read().splitlines()
    assert lines[0] == "timestamp,lat,lng,fetched_ts"
    assert len(lines) == 4                                  # um cabeçalho só

    assert arc.load([path]) == [(int(TS_DAY), 1000.0, -20.5, -45.25),
                                (int(TS_DAY + 300), 1060.0, -20.0, -45.0),
                                (int(TS_DAY + 300), 1060.0, -20.0, -45.0)]


def test_append_rows_empty_writes_nothing(tmp_path):
    assert arc.append_rows(str(tmp_path / "a"), [], TS_DAY) is None
    assert not (tmp_path / "a").exists()


def test_purge_old_keeps_last_days_and_foreign_files(tmp_path):
    for name in ("slm_raw_20261007.csv.gz", "slm_raw_20261008.csv.gz",
                 "slm_raw_20261107.csv.gz", "notes.txt", "slm_raw_x.csv.gz"):
        (tmp_path / name).write_text("")

    # 30 dias contando hoje (07/11): de 09/10 em diante
    removed = arc.purge_old(str(tmp_path), TS_DAY, 30)

    assert removed == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "notes.txt", "slm_raw_20261107.csv.gz", "slm_raw_x.csv.gz"]


def test_purge_old_missing_dir(tmp_path):
    assert arc.purge_old(str(tmp_path / "nope"), TS_DAY) == 0


def test_load_plain_pipeline_csv_has_no_fetched_ts(tmp_path):
    path = tmp_path / "slm_lightning.csv"
    path.write_text("timestamp,lat,lng\n1000,-20.0,-45.0\nbad,1,2\n\n1060,-21.0,-46.0\n")

    assert arc.load([str(path)]) == [(None, 1000.0, -20.0, -45.0), (None, 1060.0, -21.0, -46.0)]


def test_load_truncated_gzip_keeps_what_was_read(tmp_path, caplog):
    path = arc.append_rows(str(tmp_path), _rows(*[(1000 + i, -20.0, -45.0) for i in range(2000)]),
                           TS_DAY)
    data = open(path, "rb").read()
    open(path, "wb").write(data[:len(data) // 2])          # downloader no meio da escrita

    rows = arc.load([path])

    assert 0 < len(rows) < 2000
    assert "truncado" in caplog.text


def test_sources_share_the_directory_and_purge_only_their_own(tmp_path):
    old_ts = TS_DAY - 40 * 86400
    arc.append_rows(str(tmp_path), _rows((1000, -20.0, -45.0)), old_ts)
    arc.append_rows(str(tmp_path), _rows((1000, -20.0, -45.0)), old_ts, arc.CS_GLM_ARCHIVE_PREFIX)
    glm_today = arc.append_rows(str(tmp_path), _rows((1000, -20.0, -45.0)), TS_DAY,
                                arc.CS_GLM_ARCHIVE_PREFIX)

    removed = arc.purge_old(str(tmp_path), TS_DAY, 30, fs_prefix=arc.CS_GLM_ARCHIVE_PREFIX)

    assert removed == 1
    assert glm_today.endswith("glm_raw_20261107.csv.gz")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["glm_raw_20261107.csv.gz",
                                                          "slm_raw_20260928.csv.gz"]
