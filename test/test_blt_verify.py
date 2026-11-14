import csv

import pytest

import work.blt_struct as sst
import work.blt_track as trk
import work.blt_verify as vrf


def _cell(track_id, t_start, lat=0.0, lng=0.0, forecast_path=None):
    return sst.TCell(id=track_id, centroid=(lat, lng), size=10, t_start=t_start, t_end=t_start,
                     forecast_path=forecast_path or [])


def _forecast(issue_ts, lead_min, lat=0.0, lng=0.0, lat0=0.0, lng0=0.0, **std):
    return {"issue_ts": issue_ts, "lead_min": lead_min, "lat": lat, "lng": lng,
            "lat0": lat0, "lng0": lng0, **std}


def test_register_cells_stores_forecasts_with_issue_position():
    track = sst.TStormTrack(id="A")
    cell = _cell("A", 1000, lat=1.0, lng=2.0,
                 forecast_path=[{"lead_min": 10, "lat": 1.1, "lng": 2.1, "lat_std": 0.1, "lng_std": 0.1}])

    vrf.register_cells([cell], [track])

    assert track.forecasts == [_forecast(1000, 10, 1.1, 2.1, 1.0, 2.0, lat_std=0.1, lng_std=0.1)]


def test_verify_track_matches_within_tolerance_and_computes_errors():
    # previsão para 1 grau a leste, emitida em (0,0); célula observada em (0, 0.5)
    track = sst.TStormTrack(id="A", forecasts=[_forecast(0, 10, lng=1.0)])
    tol_s = vrf.CF_VERIFY_TOLERANCE_MIN * 60
    cell = _cell("A", 600 + tol_s, lng=0.5)

    rows = vrf.verify_track(track, cell)

    assert len(rows) == 1
    row = rows[0]
    assert set(row) == set(vrf.CLST_VERIFY_COLUMNS)
    assert (row["track_id"], row["issue_ts"], row["lead_min"]) == ("A", 0, 10)
    assert row["error_km"] == pytest.approx(55.6, abs=0.1)     # 0.5 grau no equador
    assert row["persist_km"] == pytest.approx(55.6, abs=0.1)
    assert track.forecasts == []


def test_verify_track_keeps_future_and_reports_expired_as_gap():
    future = _forecast(0, 30)
    expired = _forecast(0, 10)
    track = sst.TStormTrack(id="A", forecasts=[expired, future])
    # 20 min depois: +10 já passou (sem célula a tempo), +30 ainda vai valer
    cell = _cell("A", 1200)

    rows = vrf.verify_track(track, cell)

    # [new] a perdida vira linha "gap" (antes sumia sem registro)
    assert [(r["lead_min"], r["status"]) for r in rows] == [(10, "gap")]
    assert (rows[0]["verify_ts"], rows[0]["error_km"], rows[0]["z2"]) == (1200, "", "")
    assert track.forecasts == [future]


def test_verify_cells_ignores_new_tracks_and_unknown_ids():
    track = sst.TStormTrack(id="A", forecasts=[_forecast(0, 10)])

    rows = vrf.verify_cells([_cell("X", 600), _cell("A", 600)], [track])

    assert [r["track_id"] for r in rows] == ["A"]


def test_append_rows_writes_header_once(tmp_path):
    path = tmp_path / "verify.csv"
    track = sst.TStormTrack(id="A", forecasts=[_forecast(0, 10), _forecast(300, 5)])
    rows = vrf.verify_track(track, _cell("A", 600))

    vrf.append_rows(str(path), rows)
    vrf.append_rows(str(path), rows)
    vrf.append_rows(str(path), [])

    with open(path, newline="") as fh:
        data = list(csv.reader(fh))

    assert data[0] == vrf.CLST_VERIFY_COLUMNS
    assert len(data) == 1 + 2 * len(rows)


def test_append_rows_rewrites_header_after_rotation(tmp_path):
    # check_log_size (blt_run.sh) deixa o arquivo vazio depois de rotacionar
    path = tmp_path / "verify.csv"
    path.write_text("")
    track = sst.TStormTrack(id="A", forecasts=[_forecast(0, 10)])

    vrf.append_rows(str(path), vrf.verify_track(track, _cell("A", 600)))

    assert path.read_text().splitlines()[0] == ",".join(vrf.CLST_VERIFY_COLUMNS)


def test_append_rows_empty_does_not_create_file(tmp_path):
    path = tmp_path / "verify.csv"

    vrf.append_rows(str(path), [])

    assert not path.exists()


def test_mean_errors():
    rows = [{"error_km": 2.0, "persist_km": 10.0}, {"error_km": 4.0, "persist_km": 20.0}]

    assert vrf.mean_errors(rows) == (3.0, 15.0)


def test_forecasts_survive_save_and_load(tmp_path):
    path = tmp_path / "tracks.json"
    track = sst.TStormTrack(id="A", history=[_cell("A", 0)], forecasts=[_forecast(0, 10)])

    trk.save_tracks(str(path), [track], {})
    loaded, _ = trk.load_tracks(str(path))

    assert loaded[0].forecasts == [_forecast(0, 10)]


def test_load_tracks_without_forecasts_key(tmp_path):
    # [fix] estado gravado antes de 2026.nov não tem "forecasts"
    path = tmp_path / "tracks.json"
    path.write_text('{"tracks": [{"id": "A", "age_minutes": 0, "missed_frames": 0, '
                    '"active": true, "history": []}], "kalman": {}}')

    loaded, _ = trk.load_tracks(str(path))

    assert loaded[0].forecasts == []


def test_register_cells_uses_frame_issue_time():
    track = sst.TStormTrack(id="A")
    cell = sst.TCell(id="A", centroid=(0.0, 0.0), t_start=0, t_ref=500.0,
                     forecast_path=[{"lead_min": 10, "lat": 0.0, "lng": 0.1}])

    vrf.register_cells([cell], [track], 1100.0)

    assert track.forecasts[0]["issue_ts"] == 1100.0


def test_register_cells_without_issue_time_uses_cell_time():
    track = sst.TStormTrack(id="A")
    cell = sst.TCell(id="A", centroid=(0.0, 0.0), t_start=0, t_ref=500.0,
                     forecast_path=[{"lead_min": 10, "lat": 0.0, "lng": 0.1}])

    vrf.register_cells([cell], [track])

    assert track.forecasts[0]["issue_ts"] == 500.0


# ─────────────────────────────────────────────────────────────────────────────────────
# incerteza, z2 e previsões perdidas (2026.nov)

def test_ok_row_has_sigma_in_km_and_z2():
    # sigma de 0.01 grau nos dois eixos; erro de 0.02 grau em lat e 0.01 em lng
    forecast = _forecast(0, 10, lat=0.0, lng=0.0, lat_std=0.01, lng_std=0.01)
    track = sst.TStormTrack(id="A", forecasts=[forecast])

    row = vrf.verify_track(track, _cell("A", 600, lat=0.02, lng=0.01))[0]

    assert row["status"] == "ok"
    assert row["std_lat_km"] == pytest.approx(1.11)
    assert row["std_lng_km"] == pytest.approx(1.11)          # equador: cos = 1
    assert row["z2"] == pytest.approx(4.0 + 1.0)


def test_sigma_lng_in_km_shrinks_with_latitude():
    forecast = _forecast(0, 10, lat=-60.0, lat_std=0.01, lng_std=0.01)
    track = sst.TStormTrack(id="A", forecasts=[forecast])

    row = vrf.verify_track(track, _cell("A", 600, lat=-60.0))[0]

    assert row["std_lng_km"] == pytest.approx(0.555, abs=1e-3)   # 1.11 · cos 60°


def test_forecast_from_old_state_has_empty_sigma_and_z2():
    track = sst.TStormTrack(id="A", forecasts=[_forecast(0, 10)])

    row = vrf.verify_track(track, _cell("A", 600, lng=0.5))[0]

    assert (row["std_lat_km"], row["std_lng_km"], row["z2"]) == ("", "", "")
    assert row["error_km"] == pytest.approx(55.6, abs=0.1)


def test_close_tracks_reports_pending_forecasts_of_ended_tracks():
    merged = sst.TStormTrack(id="M", active=False, end_reason="merged",
                             forecasts=[_forecast(0, 10), _forecast(0, 20)])
    unknown = sst.TStormTrack(id="U", active=False, forecasts=[_forecast(0, 10)])
    alive = sst.TStormTrack(id="A", forecasts=[_forecast(0, 10)])

    rows = vrf.close_tracks([merged, unknown, alive], 900.0)

    assert [(r["track_id"], r["lead_min"], r["status"]) for r in rows] == [
        ("M", 10, "merged"), ("M", 20, "merged"), ("U", 10, "ended")]
    assert all(r["verify_ts"] == 900.0 and r["error_km"] == "" for r in rows)
    assert merged.forecasts == [] and unknown.forecasts == []
    assert len(alive.forecasts) == 1


def test_mean_errors_ignores_lost_rows():
    rows = [{"error_km": 2.0, "persist_km": 10.0, "status": "ok"},
            {"error_km": "", "persist_km": "", "status": "gap"},
            {"error_km": 4.0, "persist_km": 20.0}]

    assert vrf.mean_errors(rows) == (3.0, 15.0)


def test_append_rows_only_lost_rows(tmp_path):
    path = tmp_path / "verify.csv"
    rows = vrf.close_tracks([sst.TStormTrack(id="M", active=False, end_reason="missed",
                                             forecasts=[_forecast(0, 10)])], 900.0)

    vrf.append_rows(str(path), rows)

    with open(path, newline="") as fh:
        data = list(csv.DictReader(fh))
    assert [r["status"] for r in data] == ["missed"]


def test_append_rows_rotates_file_of_previous_layout(tmp_path):
    path = tmp_path / "verify.csv"
    old_header = ",".join(vrf.CLST_VERIFY_COLUMNS[:10])
    path.write_text(old_header + "\n")
    track = sst.TStormTrack(id="A", forecasts=[_forecast(0, 10)])

    vrf.append_rows(str(path), vrf.verify_track(track, _cell("A", 600)))

    assert (tmp_path / "verify.csv.old").read_text() == old_header + "\n"
    assert path.read_text().splitlines()[0] == ",".join(vrf.CLST_VERIFY_COLUMNS)


def test_first_ten_columns_keep_previous_order():
    # o awk do cabeçalho de blt_verify usa $4 (lead), $9 (erro) e $10 (persist.)
    assert vrf.CLST_VERIFY_COLUMNS[3] == "lead_min"
    assert vrf.CLST_VERIFY_COLUMNS[8:10] == ["error_km", "persist_km"]
    assert vrf.CLST_VERIFY_COLUMNS[13] == "status"


def test_tolerance_read_at_call_time(monkeypatch):
    track = sst.TStormTrack(id="A", forecasts=[_forecast(0, 10)])
    monkeypatch.setattr(vrf, "CF_VERIFY_TOLERANCE_MIN", 10.0)

    rows = vrf.verify_track(track, _cell("A", 600 + 9 * 60))

    assert [r["status"] for r in rows] == ["ok"]
