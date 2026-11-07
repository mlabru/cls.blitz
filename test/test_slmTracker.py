import csv
import json

import pytest

import work.slm_struct as sst
import slmTracker as trkmod


def _mk(ts, amp=1.0, polarity=1, kind="CG"):
    return sst.TLightning(ts=ts, lat=0.0, lng=0.0, amp=amp, polarity=polarity, kind=kind)


def test_cluster_window_forms_single_cluster():
    n = trkmod.CI_MIN_PTS
    events = [_mk(1000.0 + i) for i in range(n)]

    cells = trkmod.cluster_window(events)

    assert len(cells) == 1
    cell = cells[0]
    assert cell.size == n
    assert cell.t_start == 1000.0
    assert cell.t_end == pytest.approx(1000.0 + (n - 1))
    assert cell.flash_rate == pytest.approx(n)  # delta_min clamped to 1.0 (few-second span)
    assert cell.max_amp == 1.0
    assert cell.cg_ratio == 1.0


def test_cluster_window_empty_input():
    assert trkmod.cluster_window([]) == []


def test_compute_cell_stats_fills_amplitude_and_ratios():
    points = [
        _mk(0, amp=10.0, polarity=1, kind="CG"),
        _mk(1, amp=20.0, polarity=-1, kind="IC"),
        _mk(2, amp=30.0, polarity=1, kind="CG"),
        _mk(3, amp=5.0, polarity=0, kind=""),
    ]
    cell = sst.TCell()

    trkmod.compute_cell_stats(cell, points)

    assert cell.max_amp == 30.0
    assert cell.mean_amp == pytest.approx(16.25)
    assert cell.positive_ratio == pytest.approx(0.5)
    assert cell.negative_ratio == pytest.approx(0.25)
    assert cell.cg_ratio == pytest.approx(0.5)
    assert cell.ic_ratio == pytest.approx(0.25)


def test_compute_cell_stats_empty_cluster_leaves_cell_untouched():
    cell = sst.TCell()

    trkmod.compute_cell_stats(cell, [])

    assert cell.max_amp == 0.0
    assert cell.mean_amp == 0.0


def test_filter_new_events():
    events = [sst.TLightning(ts=t, lat=0.0, lng=0.0) for t in (1, 2, 3, 4)]

    result = trkmod.filter_new_events(events, 2)

    assert [e.ts for e in result] == [3, 4]


def test_load_csv_parses_rows_and_skips_malformed(tmp_path):
    csv_path = tmp_path / "in.csv"
    csv_path.write_text(
        "timestamp,lat,lng,amp,polarity,kind\n"
        "1000,1.0,2.0,3.5,1,CG\n"
        "bad,line,here\n"
        "1001,4.0,5.0\n"
    )

    events = trkmod.load_csv(str(csv_path))

    assert len(events) == 2
    assert events[0].amp == 3.5
    assert events[0].kind == "CG"
    assert events[1].amp == 0.0
    assert events[1].polarity == 0
    assert events[1].kind == ""


def test_export_frame_writes_expected_fields(tmp_path):
    cell = sst.TCell(id="c1", centroid=(1.0, 2.0), size=5, density=0.1, flash_rate=2.0,
                      speed_kmh=10.0, direction_deg=90.0, forecast_path=[], history=[],
                      t_start=100, t_end=200, max_amp=3.0, mean_amp=1.5,
                      positive_ratio=0.5, negative_ratio=0.5, cg_ratio=0.2, ic_ratio=0.8)
    out_path = tmp_path / "frame.json"

    trkmod.export_frame([cell], str(out_path))

    data = json.loads(out_path.read_text())
    assert len(data) == 1
    assert data[0]["id"] == "c1"
    assert data[0]["centroid"] == [1.0, 2.0]
    assert data[0]["ic_ratio"] == 0.8
    assert "severity_index" not in data[0]


def test_process_creates_frame_and_persists_track_across_calls(tmp_path):
    n = trkmod.CI_MIN_PTS

    lines = ["timestamp,lat,lng,amp,polarity,kind"]
    lines += [f"{1000 + i},0.0,0.0,1.0,1,CG" for i in range(n)]
    csv_path = tmp_path / "raw.csv"
    csv_path.write_text("\n".join(lines) + "\n")

    json_path = tmp_path / "frame_0000.json"
    tracks_path = tmp_path / "tracks.json"

    newest_ts = trkmod.process(str(csv_path), str(json_path), 0.0, str(tracks_path))

    assert newest_ts == pytest.approx(1000 + n - 1)
    assert json_path.exists()
    assert tracks_path.exists()

    frame = json.loads(json_path.read_text())
    assert len(frame) == 1
    assert frame[0]["size"] == n
    assert len(frame[0]["forecast_path"]) == 3
    first_id = frame[0]["id"]

    # second frame: cluster shifts slightly but stays within Hungarian association range
    lines2 = ["timestamp,lat,lng,amp,polarity,kind"]
    lines2 += [f"{2000 + i},0.01,0.01,1.0,1,CG" for i in range(n)]
    csv2_path = tmp_path / "raw2.csv"
    csv2_path.write_text("\n".join(lines2) + "\n")

    json2_path = tmp_path / "frame_0001.json"
    newest_ts2 = trkmod.process(str(csv2_path), str(json2_path), 0.0, str(tracks_path))

    assert newest_ts2 == pytest.approx(2000 + n - 1)
    frame2 = json.loads(json2_path.read_text())
    assert frame2[0]["id"] == first_id


def test_process_returns_none_without_events(tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("timestamp,lat,lng\n")

    result = trkmod.process(str(csv_path),
                            str(tmp_path / "out.json"),
                            0.0,
                            str(tmp_path / "tracks.json"))

    assert result is None


def test_load_csv_empty_file_returns_empty_list(tmp_path):
    # [fix] arquivo sem nem o cabeçalho levantava StopIteration
    csv_path = tmp_path / "vazio.csv"
    csv_path.write_text("")

    assert trkmod.load_csv(str(csv_path)) == []


# ─────────────────────────────────────────────────────────────────────────────────────
# janela fixa no CSV inteiro e expiração por tempo (2026.nov)

def _write_csv(path, rows):
    path.write_text("timestamp,lat,lng\n" + "".join(f"{t},{la},{lo}\n" for t, la, lo in rows))


def _run(tmp_path, rows, since_ts, name="frame.json"):
    csv_path = tmp_path / "raw.csv"
    _write_csv(csv_path, rows)
    json_path = tmp_path / name
    newest = trkmod.process(str(csv_path), str(json_path), since_ts, str(tmp_path / "tracks.json"))
    frame = json.loads(json_path.read_text()) if newest is not None else None
    return newest, frame


def test_process_clusters_old_and_new_events_of_the_window(tmp_path):
    # [fix] antes só os eventos novos eram agrupados: 6 + 6 não formavam célula
    rows = [(1000, 0.0, 0.0)] * 6 + [(1060 + i, 0.0, 0.0) for i in range(6)]

    newest, frame = _run(tmp_path, rows, since_ts=1000)

    assert newest == 1065
    assert len(frame) == 1
    assert frame[0]["size"] == 12


def test_process_keeps_late_strikes_of_the_last_minute(tmp_path):
    # rodada 1: 10 raios no minuto 1000 -> since_ts vira 1000
    rows = [(1000, 0.0, 0.0)] * trkmod.CI_MIN_PTS
    newest, _ = _run(tmp_path, rows, since_ts=0)
    assert newest == 1000

    # rodada 2: chegam 3 raios atrasados do MESMO minuto e 10 do minuto seguinte
    rows += [(1000, 0.001, 0.001)] * 3 + [(1060, 0.0, 0.0)] * trkmod.CI_MIN_PTS
    newest, frame = _run(tmp_path, rows, since_ts=newest, name="frame2.json")

    assert newest == 1060
    assert frame[0]["size"] == 2 * trkmod.CI_MIN_PTS + 3


def test_process_window_excludes_events_older_than_window(tmp_path):
    li_window_s = trkmod.CI_WINDOW_MINUTES * 60
    rows = [(1000, 0.0, 0.0)] * 20 + [(1000 + li_window_s + 60, 0.0, 0.0)] * trkmod.CI_MIN_PTS

    _, frame = _run(tmp_path, rows, since_ts=1000)

    assert frame[0]["size"] == trkmod.CI_MIN_PTS


def test_process_without_new_events_returns_none_even_with_old_events(tmp_path):
    rows = [(1000, 0.0, 0.0)] * 20

    newest, _ = _run(tmp_path, rows, since_ts=1000)

    assert newest is None


def test_process_expired_track_does_not_inherit_cell(tmp_path):
    li_gap_s = int(trkmod.trk.CF_MAX_TRACK_GAP_MIN * 60)
    rows = [(1000, 0.0, 0.0)] * trkmod.CI_MIN_PTS
    _, frame1 = _run(tmp_path, rows, since_ts=0)

    # mesma posição, mas depois do limite de expiração: track novo
    rows2 = [(1000 + li_gap_s + 60, 0.0, 0.0)] * trkmod.CI_MIN_PTS
    _, frame2 = _run(tmp_path, rows2, since_ts=1000, name="frame2.json")

    assert frame2[0]["id"] != frame1[0]["id"]

    # dentro do limite: continua o mesmo track
    rows3 = rows2 + [(1000 + li_gap_s + 300, 0.0, 0.0)] * trkmod.CI_MIN_PTS
    _, frame3 = _run(tmp_path, rows3, since_ts=1000 + li_gap_s + 60, name="frame3.json")

    assert frame3[0]["id"] == frame2[0]["id"]


# ─────────────────────────────────────────────────────────────────────────────────────
# verificação das previsões ponta a ponta (2026.nov)

def test_process_verifies_previous_forecasts(tmp_path):
    verify_path = tmp_path / "verify.csv"
    tracks_path = tmp_path / "tracks.json"
    n = trkmod.CI_MIN_PTS

    # uma célula parada, um frame a cada 5 min (t_start avança 300 s por frame,
    # porque cada frame só tem os raios daquele minuto)
    for i in range(4):
        ts = 1000 + 300 * i
        csv_path = tmp_path / f"raw{i}.csv"
        _write_csv(csv_path, [(ts, 0.0, 0.0)] * n)
        trkmod.process(str(csv_path), str(tmp_path / f"f{i}.json"), 0.0,
                       str(tracks_path), str(verify_path))

    with open(verify_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    # +10 emitida em 1000 e em 1300 vale em 1600 e 1900; +20 de 1000 vale em 2200 (não chegou)
    assert sorted((float(r["issue_ts"]), int(r["lead_min"])) for r in rows) == [(1000, 10), (1300, 10)]
    assert all(float(r["error_km"]) == pytest.approx(0.0, abs=1e-3) for r in rows)

    # a +20 de 1000 e as de 1300/1600/1900 que ainda valem continuam pendentes
    state = json.loads(tracks_path.read_text())
    pending = {(f["issue_ts"], f["lead_min"]) for f in state["tracks"][0]["forecasts"]}
    assert (1000, 20) in pending and (1900, 10) in pending


def test_process_without_verify_file_writes_nothing(tmp_path):
    rows = [(1000, 0.0, 0.0)] * trkmod.CI_MIN_PTS

    _run(tmp_path, rows, since_ts=0)

    assert not list(tmp_path.glob("*.csv"))[1:]   # só o raw.csv


# ─────────────────────────────────────────────────────────────────────────────────────
# frame já aplicado não é reaplicado se o shell ficou para trás (2026.nov)

def test_process_does_not_reapply_frame_when_shell_lags(tmp_path):
    tracks_path = tmp_path / "tracks.json"
    rows = [(1000, 0.0, 0.0)] * trkmod.CI_MIN_PTS

    newest, frame = _run(tmp_path, rows, since_ts=0)
    state1 = tracks_path.read_text()

    # o shell morreu antes de gravar .slm_last_ts: a rodada seguinte recebe since_ts antigo
    newest2 = trkmod.process(str(tmp_path / "raw.csv"), str(tmp_path / "frame2.json"), 0.0,
                             str(tracks_path))

    assert newest2 == newest                       # o shell alcança o estado...
    assert tracks_path.read_text() == state1       # ...sem reaplicar o frame
    assert not (tmp_path / "frame2.json").exists()

    # com raios novos, o frame seguinte entra uma vez só no histórico
    rows += [(1300, 0.0, 0.0)] * trkmod.CI_MIN_PTS
    _, frame3 = _run(tmp_path, rows, since_ts=0, name="frame3.json")

    assert frame3[0]["id"] == frame[0]["id"]
    assert len(frame3[0]["history"]) == 2


def test_process_returns_none_when_shell_is_up_to_date(tmp_path):
    rows = [(1000, 0.0, 0.0)] * trkmod.CI_MIN_PTS
    newest, _ = _run(tmp_path, rows, since_ts=0)

    assert _run(tmp_path, rows, since_ts=newest)[0] is None


def test_process_state_ahead_but_no_cells_still_catches_up(tmp_path):
    # mesmo sem células novas, o shell atrasado é avisado do frame já aplicado
    rows = [(1000, 0.0, 0.0)] * trkmod.CI_MIN_PTS
    newest, _ = _run(tmp_path, rows, since_ts=0)

    assert _run(tmp_path, [(900, 0.0, 0.0)], since_ts=500)[0] == newest


# ─────────────────────────────────────────────────────────────────────────────────────
# relógio pela média dos ts, divisão e fusão ponta a ponta (2026.nov)

def test_build_cell_t_ref_is_mean_time():
    cells = trkmod.cluster_window([sst.TLightning(ts=t, lat=0.0, lng=0.0)
                                   for t in [0] * 5 + [600] * 5])

    assert cells[0].t_start == 0
    assert cells[0].t_ref == pytest.approx(300.0)


def test_export_frame_has_parent_id_and_t_ref(tmp_path):
    out = tmp_path / "f.json"

    trkmod.export_frame([sst.TCell(id="b", parent_id="a", t_ref=12.5)], str(out))

    data = json.loads(out.read_text())[0]
    assert (data["parent_id"], data["t_ref"]) == ("a", 12.5)


def test_process_moving_cell_has_speed_on_second_frame(tmp_path):
    # [fix] com a janela fixa, os dois frames têm o mesmo primeiro raio
    # (t_start); pelo t_start o dt era 0 e a célula ficava parada por 20 min
    n = trkmod.CI_MIN_PTS
    rows = [(1000, 0.0, 0.0)] * n
    newest, _ = _run(tmp_path, rows, since_ts=0)

    rows += [(1300, 0.0, 0.05)] * n                 # 0.05 grau a leste em 5 min
    _, frame = _run(tmp_path, rows, since_ts=newest, name="frame2.json")

    # a média de posição anda metade (0.025 grau) em metade do tempo (150 s):
    # a velocidade estimada é a real, 0.05 grau em 5 min
    assert frame[0]["t_ref"] == pytest.approx(1150.0)
    assert frame[0]["speed_kmh"] == pytest.approx(0.05 * 111.0 * 12, rel=0.01)
    assert frame[0]["direction_deg"] == pytest.approx(90.0, abs=1e-6)


def _frames(tmp_path, frames):
    """um CSV por frame (sem janela compartilhada), estado e verificação em comum"""
    out = []
    for i, rows in enumerate(frames):
        csv_path = tmp_path / f"raw{i}.csv"
        _write_csv(csv_path, rows)
        json_path = tmp_path / f"f{i}.json"
        trkmod.process(str(csv_path), str(json_path), 0.0, str(tmp_path / "tracks.json"))
        out.append(json.loads(json_path.read_text()))
    return out


def test_process_split_creates_child_track(tmp_path):
    n = trkmod.CI_MIN_PTS
    lng_child = 25.0 / 111.19                       # 25 km: dois clusters (eps 12 km)

    f0, f1 = _frames(tmp_path, [[(1000, 0.0, 0.0)] * n,
                                [(1300, 0.0, 0.0)] * n + [(1300, 0.0, lng_child)] * n])

    parent_id = f0[0]["id"]
    by_parent = {c["parent_id"]: c for c in f1}
    assert by_parent[""]["id"] == parent_id
    assert by_parent[parent_id]["id"] != parent_id


def test_process_merge_drops_absorbed_track(tmp_path):
    n = trkmod.CI_MIN_PTS
    lng_b = 25.0 / 111.19

    f0, f1 = _frames(tmp_path, [[(1000, 0.0, 0.0)] * n + [(1000, 0.0, lng_b)] * n,
                                [(1300, 0.0, lng_b / 2)] * (2 * n)])

    assert len(f0) == 2 and len(f1) == 1
    assert f1[0]["id"] in {c["id"] for c in f0}
    state = json.loads((tmp_path / "tracks.json").read_text())
    assert [t["id"] for t in state["tracks"]] == [f1[0]["id"]]   # o absorvido saiu na hora
