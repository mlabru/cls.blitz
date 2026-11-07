import math

import pytest

import work.slm_kalman as klm
import work.slm_motion as mtn
import work.slm_struct as sst
import work.slm_track as trk


def test_compute_centroid_mean():
    cluster = [sst.TLightning(ts=0, lat=0.0, lng=0.0),
               sst.TLightning(ts=0, lat=2.0, lng=4.0)]

    assert trk.compute_centroid(cluster) == (1.0, 2.0)


def test_compute_density_uses_max_radius_clamped_to_one_km():
    cluster = [sst.TLightning(ts=0, lat=0.0, lng=0.0) for _ in range(5)]

    density = trk.compute_density(cluster, (0.0, 0.0))

    assert density == pytest.approx(5 / math.pi, rel=1e-6)


def test_compute_flash_rate():
    assert trk.compute_flash_rate([1, 2, 3], 2.0) == pytest.approx(1.5)
    assert trk.compute_flash_rate([1, 2, 3], 0.0) == 0.0
    assert trk.compute_flash_rate([1, 2, 3], -5.0) == 0.0


def test_match_score_identical_cells_is_zero():
    prev = sst.TCell(centroid=(0.0, 0.0), size=10)
    curr = sst.TCell(centroid=(0.0, 0.0), size=10)

    assert trk.match_score(prev, curr) == 0.0


def test_match_score_penalizes_distance_and_size():
    prev = sst.TCell(centroid=(0.0, 0.0), size=10)
    curr = sst.TCell(centroid=(0.0, 1.0), size=10)  # ~111.19 km east

    score = trk.match_score(prev, curr, ff_max_dist_km=200.0)

    assert score == pytest.approx(0.6 * (111.19 / 200.0), abs=0.01)


def test_generate_track_id_unique_and_length():
    ids = {trk.generate_track_id() for _ in range(200)}

    assert len(ids) == 200
    assert all(len(i) == 8 for i in ids)


def test_associate_cells_matches_nearest_track():
    track_a = sst.TStormTrack(id="A", history=[sst.TCell(centroid=(0.0, 0.0), size=10)])
    track_b = sst.TStormTrack(id="B", history=[sst.TCell(centroid=(10.0, 10.0), size=10)])

    cell0 = sst.TCell(centroid=(0.01, 0.0), size=10)
    cell1 = sst.TCell(centroid=(10.01, 10.0), size=10)

    result = trk.associate_cells([track_a, track_b], [cell0, cell1])

    assert result == {0: 0, 1: 1}


def test_associate_cells_no_active_tracks_or_no_cells():
    assert trk.associate_cells([], [sst.TCell()]) == {}
    assert trk.associate_cells([sst.TStormTrack(id="A", history=[sst.TCell()])], []) == {}


def test_update_tracks_creates_new_track():
    cell = sst.TCell(centroid=(0.0, 0.0), size=10, t_start=0, t_end=0)

    tracks, kalman = trk.update_tracks([], {}, [cell], {})

    assert len(tracks) == 1
    new_track = tracks[0]
    assert new_track.active is True
    assert new_track.history == [cell]
    assert cell.id == new_track.id
    assert cell.id in kalman
    assert isinstance(kalman[cell.id], klm.TKalmanTracker)


def test_update_tracks_associates_and_updates_motion():
    cell0 = sst.TCell(centroid=(0.0, 0.0), size=10, t_start=0, t_end=0)
    tracks, kalman = trk.update_tracks([], {}, [cell0], {})

    cell1 = sst.TCell(centroid=(0.0, 1.0), size=10, t_start=3600, t_end=3600)
    tracks, kalman = trk.update_tracks(tracks, kalman, [cell1], {0: 0})

    assert len(tracks[0].history) == 2
    assert cell1.id == tracks[0].id
    # velocidade da célula = a do Kalman (a mesma do cone de previsão), não a EMA
    assert (cell1.speed_kmh, cell1.direction_deg) == mtn.kalman_velocity(kalman[cell1.id], 0.0)
    assert cell1.speed_kmh > 0.0
    assert cell1.direction_deg == pytest.approx(90.0, abs=1e-6)
    assert tracks[0].missed_frames == 0


def test_update_tracks_deactivates_after_max_missed_frames():
    cell0 = sst.TCell(centroid=(0.0, 0.0), size=10, t_start=0, t_end=0)
    tracks, kalman = trk.update_tracks([], {}, [cell0], {})

    for expected_missed in (1, 2, 3):
        tracks, kalman = trk.update_tracks(tracks, kalman, [], {})
        assert tracks[0].missed_frames == expected_missed
        assert tracks[0].active == (expected_missed < trk.CI_MAX_MISSED_FRAMES)


def test_save_and_load_tracks_round_trip(tmp_path):
    cell = sst.TCell(id="abc12345", centroid=(1.0, 2.0), size=7,
                      speed_kmh=10.0, direction_deg=90.0, t_start=100, t_end=200)
    track = sst.TStormTrack(id="abc12345", age_minutes=5.0, missed_frames=1, active=True,
                             history=[cell])
    kalman = {"abc12345": klm.TKalmanTracker(1.0, 2.0)}

    path = tmp_path / "tracks.json"
    trk.save_tracks(str(path), [track], kalman)

    loaded_tracks, loaded_kalman = trk.load_tracks(str(path))

    assert len(loaded_tracks) == 1
    lt = loaded_tracks[0]
    assert lt.id == "abc12345"
    assert lt.age_minutes == 5.0
    assert lt.missed_frames == 1
    assert lt.active is True
    assert lt.history[0].centroid == (1.0, 2.0)
    assert lt.history[0].size == 7

    assert "abc12345" in loaded_kalman
    assert loaded_kalman["abc12345"]._lat.x == pytest.approx(1.0)
    assert loaded_kalman["abc12345"]._lng.x == pytest.approx(2.0)


def test_load_tracks_missing_file_returns_empty(tmp_path):
    tracks, kalman = trk.load_tracks(str(tmp_path / "missing.json"))

    assert tracks == []
    assert kalman == {}


def test_load_tracks_recreates_missing_kalman_state(tmp_path):
    # [fix] track gravado sem a entrada "kalman" quebrava depois, com KeyError
    path = tmp_path / "tracks.json"
    track = sst.TStormTrack(id="abc12345", history=[sst.TCell(id="abc12345", centroid=(1.0, 2.0))])
    trk.save_tracks(str(path), [track], {})

    loaded_tracks, loaded_kalman = trk.load_tracks(str(path))

    assert "abc12345" in loaded_kalman
    lat, lng, _, _ = loaded_kalman["abc12345"].predict_position(0.0)
    assert (lat, lng) == (1.0, 2.0)


# ─────────────────────────────────────────────────────────────────────────────────────
# expiração por tempo e associação contra a posição prevista (2026.nov)

def _track(track_id, t_end, active=True):
    return sst.TStormTrack(id=track_id, active=active,
                           history=[sst.TCell(id=track_id, centroid=(0.0, 0.0), size=10,
                                              t_start=t_end, t_end=t_end)])


def test_expire_stale_tracks_deactivates_only_old_tracks():
    li_limit_s = trk.CF_MAX_TRACK_GAP_MIN * 60
    now = 100000.0
    fresh = _track("fresh", now - li_limit_s + 1)
    edge = _track("edge", now - li_limit_s)
    stale = _track("stale", now - li_limit_s - 1)
    inactive = _track("old", 0.0, active=False)
    empty = sst.TStormTrack(id="empty", history=[])

    expired = trk.expire_stale_tracks([fresh, edge, stale, inactive, empty], now)

    assert expired == 1
    assert fresh.active and edge.active and empty.active
    assert not stale.active
    assert not inactive.active


def test_expired_track_is_not_associated_nor_saved(tmp_path):
    stale = _track("stale", 0.0)
    trk.expire_stale_tracks([stale], 10 * 3600.0)

    assert trk.associate_cells([stale], [sst.TCell(centroid=(0.0, 0.0), size=10)]) == {}

    path = tmp_path / "tracks.json"
    trk.save_tracks(str(path), [stale], {"stale": klm.TKalmanTracker(0.0, 0.0)})
    assert trk.load_tracks(str(path)) == ([], {})


def _moving_track():
    # track em (0,0) no instante 0, Kalman indo para leste a 1 grau/h (~111 km/h)
    track = sst.TStormTrack(id="A", history=[sst.TCell(id="A", centroid=(0.0, 0.0), size=10,
                                                       t_start=0, t_end=0)])
    return track, {"A": klm.TKalmanTracker(0.0, 0.0, ff_vlat=0.0, ff_vlng=1.0)}


def test_associate_cells_uses_kalman_predicted_position():
    track, kalman = _moving_track()
    # 1 h depois, a ~111 km da última posição observada (fora do raio de 100 km),
    # mas exatamente onde o Kalman prevê
    cell = sst.TCell(centroid=(0.0, 1.0), size=10, t_start=3600, t_end=3600)

    assert trk.associate_cells([track], [cell]) == {}
    assert trk.associate_cells([track], [cell], fdct_kalman_states=kalman) == {0: 0}


def test_associate_cells_prediction_resolves_crossing_candidates():
    track, kalman = _moving_track()
    # 30 min depois: a célula a oeste está mais perto da última posição,
    # a de leste está onde o Kalman prevê (lng 0.5)
    west = sst.TCell(centroid=(0.0, -0.3), size=10, t_start=1800, t_end=1800)
    east = sst.TCell(centroid=(0.0, 0.5), size=10, t_start=1800, t_end=1800)

    assert trk.associate_cells([track], [west, east]) == {0: 0}
    assert trk.associate_cells([track], [west, east], fdct_kalman_states=kalman) == {1: 0}


def test_associate_cells_track_without_kalman_falls_back_to_last_position():
    track, _ = _moving_track()
    cell = sst.TCell(centroid=(0.0, 0.01), size=10, t_start=3600, t_end=3600)

    assert trk.associate_cells([track], [cell], fdct_kalman_states={}) == {0: 0}


# ─────────────────────────────────────────────────────────────────────────────────────
# corte explícito de score (2026.nov)

def _size_track(size):
    return sst.TStormTrack(id="A", history=[sst.TCell(id="A", centroid=(0.0, 0.0), size=size)])


def _cell_at_km(km_east, size):
    return sst.TCell(centroid=(0.0, km_east / 111.19), size=size)


def test_associate_cells_rejects_far_pair_with_size_mismatch():
    # 80 km e tamanho 1:5 -> score 0.48 + 0.32 = 0.80 > CF_MAX_MATCH_SCORE (antes: aceito)
    cell = _cell_at_km(80.0, 20)

    assert trk.match_score(_size_track(100).last_cell, cell) == pytest.approx(0.8, abs=0.01)
    assert trk.associate_cells([_size_track(100)], [cell]) == {}


def test_associate_cells_accepts_far_pair_of_same_size():
    assert trk.associate_cells([_size_track(100)], [_cell_at_km(80.0, 100)]) == {0: 0}


def test_associate_cells_size_alone_never_rejects():
    # no mesmo lugar, mesmo 1:10 (célula crescendo/dissipando) continua o track
    assert trk.associate_cells([_size_track(100)], [_cell_at_km(0.0, 10)]) == {0: 0}


@pytest.mark.parametrize("ratio,limit_km", [(1.0, 100.0), (0.5, 66.7), (0.25, 50.0), (0.1, 40.0)])
def test_max_match_score_limits_documented_in_constant(ratio, limit_km):
    size = int(100 * ratio)

    assert trk.associate_cells([_size_track(100)], [_cell_at_km(limit_km - 1.0, size)]) == {0: 0}
    assert trk.associate_cells([_size_track(100)], [_cell_at_km(limit_km + 1.0, size)]) == {}



def test_associate_cells_max_score_parameter():
    cell = _cell_at_km(80.0, 20)

    assert trk.associate_cells([_size_track(100)], [cell], ff_max_score=0.9) == {0: 0}
