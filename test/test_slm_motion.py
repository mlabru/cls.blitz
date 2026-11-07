import pytest

import work.slm_kalman as klm
import work.slm_motion as mtn
import work.slm_struct as sst


@pytest.mark.parametrize("lat2,lng2,expected", [
    (1.0, 0.0, 0.0),      # due north
    (0.0, 1.0, 90.0),     # due east
    (-1.0, 0.0, 180.0),   # due south
    (0.0, -1.0, 270.0),   # due west
])
def test_bearing_deg_cardinal_directions(lat2, lng2, expected):
    assert mtn.bearing_deg(0.0, 0.0, lat2, lng2) == pytest.approx(expected, abs=1e-6)


def test_compute_direction_and_speed():
    prev = sst.TCell(centroid=(0.0, 0.0), t_start=0)
    curr = sst.TCell(centroid=(0.0, 1.0), t_start=3600)

    assert mtn.compute_direction(prev, curr) == pytest.approx(90.0, abs=1e-6)
    assert mtn.compute_speed_kmh(prev, curr) == pytest.approx(111.19, rel=1e-3)


def test_compute_speed_kmh_non_positive_elapsed_is_zero():
    prev = sst.TCell(centroid=(0.0, 0.0), t_start=100)
    curr = sst.TCell(centroid=(0.0, 1.0), t_start=100)

    assert mtn.compute_speed_kmh(prev, curr) == 0.0


def test_compute_velocity_ema_constant_motion():
    history = [
        sst.TCell(centroid=(0.0, 0.0), t_start=0),
        sst.TCell(centroid=(0.0, 1.0), t_start=3600),
        sst.TCell(centroid=(0.0, 2.0), t_start=7200),
    ]

    speed, direction = mtn.compute_velocity_ema(history)

    assert speed == pytest.approx(111.19, rel=1e-3)
    assert direction == pytest.approx(90.0, abs=0.5)


def test_compute_velocity_ema_needs_two_points():
    assert mtn.compute_velocity_ema([sst.TCell()]) == (0.0, 0.0)
    assert mtn.compute_velocity_ema([]) == (0.0, 0.0)


def test_extrapolate_path_length_and_lead_times():
    kt = klm.TKalmanTracker(0.0, 0.0)
    kt.update(0.0, 1.0)

    path = mtn.extrapolate_path(kt)

    assert [p["lead_min"] for p in path] == list(mtn.CT_FORECAST_LEADS_MIN)
    assert all(p["lat_std"] >= 0.0 and p["lng_std"] >= 0.0 for p in path)


# ─────────────────────────────────────────────────────────────────────────────────────
# kalman_velocity (2026.nov)

@pytest.mark.parametrize("vlat,vlng,expected_dir", [
    (1.0, 0.0, 0.0),      # norte
    (0.0, 1.0, 90.0),     # leste
    (-1.0, 0.0, 180.0),   # sul
    (0.0, -1.0, 270.0),   # oeste
])
def test_kalman_velocity_direction(vlat, vlng, expected_dir):
    kt = klm.TKalmanTracker(0.0, 0.0, ff_vlat=vlat, ff_vlng=vlng)

    speed, direction = mtn.kalman_velocity(kt, 0.0)

    assert speed == pytest.approx(111.0, rel=0.01)
    assert direction == pytest.approx(expected_dir, abs=1e-6)


def test_kalman_velocity_longitude_shrinks_with_latitude():
    kt = klm.TKalmanTracker(-60.0, 0.0, ff_vlng=1.0)

    speed, _ = mtn.kalman_velocity(kt, -60.0)

    assert speed == pytest.approx(111.0 * 0.5, rel=0.01)


def test_kalman_velocity_stopped_filter():
    assert mtn.kalman_velocity(klm.TKalmanTracker(0.0, 0.0), 0.0) == (0.0, 0.0)
