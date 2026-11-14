import work.blt_struct as sst


def test_cell_to_dict_and_back_round_trip():
    cell = sst.TCell(id="x1", centroid=(1.5, -2.5), size=42,
                      speed_kmh=12.3, direction_deg=45.0, t_start=10, t_end=20)

    restored = sst.cell_from_dict(sst.cell_to_dict(cell))

    assert restored.id == cell.id
    assert restored.centroid == cell.centroid
    assert restored.size == cell.size
    assert restored.speed_kmh == cell.speed_kmh
    assert restored.direction_deg == cell.direction_deg
    assert restored.t_start == cell.t_start
    assert restored.t_end == cell.t_end


def test_tcell_defaults():
    cell = sst.TCell()

    assert cell.centroid == (0.0, 0.0)
    assert cell.forecast_path == []
    assert cell.history == []


def test_storm_track_last_cell_empty_history():
    track = sst.TStormTrack(id="t1")

    assert track.last_cell is None


def test_storm_track_last_cell_returns_most_recent():
    c1, c2 = sst.TCell(id="a"), sst.TCell(id="b")
    track = sst.TStormTrack(id="t1", history=[c1, c2])

    assert track.last_cell is c2


# ─────────────────────────────────────────────────────────────────────────────────────
# t_ref e cell_time (2026.nov)

def test_cell_time_prefers_t_ref():
    assert sst.cell_time(sst.TCell(t_start=100, t_ref=250.0)) == 250.0
    assert sst.cell_time(sst.TCell(t_start=100)) == 100


def test_cell_dict_round_trip_keeps_t_ref():
    cell = sst.TCell(id="a", centroid=(1.0, 2.0), size=3, t_start=100, t_end=300, t_ref=200.0)

    assert sst.cell_from_dict(sst.cell_to_dict(cell)).t_ref == 200.0


def test_cell_from_dict_old_state_without_t_ref():
    data = sst.cell_to_dict(sst.TCell(id="a", t_start=100, t_end=300))
    del data["t_ref"]

    cell = sst.cell_from_dict(data)

    assert cell.t_ref == 0.0
    assert sst.cell_time(cell) == 100
