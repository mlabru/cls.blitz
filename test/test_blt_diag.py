import csv

import pytest

import work.blt_diag as dgn
import work.blt_struct as sst


def _cell(track_id, size, **assoc):
    return sst.TCell(id=track_id, size=size, assoc=assoc)


def test_association_rows_one_per_cell_with_rounding():
    cells = [_cell("A", 30, kind="continue", dt_min=5.0, dist_km=1.23456, score=0.1,
                   size_prev=25, nis=0.5),
             _cell("B", 12, kind="new"),
             sst.TCell(id="C", size=10)]                  # sem assoc: não entra

    rows = dgn.association_rows(1000.0, cells)

    lj = {"rate_now": 0.0, "jump_sigma": 0.0, "jump": 0}
    assert rows == [{"frame_ts": 1000.0, "track_id": "A", "size": 30, "kind": "continue",
                     "dt_min": 5.0, "dist_km": 1.235, "score": 0.1, "size_prev": 25,
                     "nis": 0.5, **lj},
                    {"frame_ts": 1000.0, "track_id": "B", "size": 12, "kind": "new", **lj}]
    assert all(set(r) <= set(dgn.CLST_ASSOC_COLUMNS) for r in rows)


def test_frame_row_counts_kinds_end_reasons_and_latency():
    cells = [_cell("A", 30, kind="continue"), _cell("B", 20, kind="two_point"),
             _cell("C", 10, kind="new"), _cell("D", 15, kind="split")]
    tracks = [sst.TStormTrack(id="A"), sst.TStormTrack(id="B"),
              sst.TStormTrack(id="X", active=False, end_reason="merged"),
              sst.TStormTrack(id="Y", active=False, end_reason="missed"),
              sst.TStormTrack(id="Z", active=False, end_reason="expired")]

    row = dgn.frame_row(1000.0, 1180.5, 100, cells, tracks)

    assert set(row) == set(dgn.CLST_FRAME_COLUMNS)
    assert row["latency_s"] == pytest.approx(180.5)
    assert (row["n_window_events"], row["n_clustered"], row["n_cells"]) == (100, 75, 4)
    assert (row["n_continued"], row["n_two_point"], row["n_new"], row["n_split"]) == (1, 1, 1, 1)
    assert (row["n_merged"], row["n_missed"], row["n_expired"], row["n_active"]) == (1, 1, 1, 2)


def test_append_writes_csv_with_header(tmp_path):
    path = tmp_path / "assoc.csv"

    dgn.append(str(path), dgn.CLST_ASSOC_COLUMNS,
               dgn.association_rows(1000.0, [_cell("A", 30, kind="new")]))
    dgn.append(str(path), dgn.CLST_ASSOC_COLUMNS, [])

    with open(path, newline="") as fh:
        data = list(csv.DictReader(fh))
    assert [(r["track_id"], r["kind"], r["nis"]) for r in data] == [("A", "new", "")]


def test_rows_carry_lightning_jump():
    cell = _cell("A", 30, kind="continue")
    cell.flash_rate_now, cell.jump_sigma, cell.lightning_jump = 12.5, 3.21234, True

    row = dgn.association_rows(1000.0, [cell])[0]
    frame = dgn.frame_row(1000.0, 1060.0, 100, [cell, _cell("B", 10, kind="new")], [])

    assert (row["rate_now"], row["jump_sigma"], row["jump"]) == (12.5, 3.212, 1)
    assert frame["n_jump"] == 1
