import pytest

import work.slm_kalman as klm


def test_kalman1d_predict_grows_uncertainty_and_moves_by_velocity():
    k = klm._TKalman1D(0.0, v0=2.0)
    std0 = k.std

    k.predict(1.0)

    assert k.std > std0
    assert k.x == pytest.approx(2.0)  # x0 + v0 * dt


def test_kalman1d_update_moves_toward_measurement_and_shrinks_uncertainty():
    k = klm._TKalman1D(0.0)
    std0 = k.std

    k.update(1.0)

    assert 0.0 < k.x < 1.0
    assert k.std < std0


def test_kalman_tracker_update_and_predict_position():
    kt = klm.TKalmanTracker(0.0, 0.0)
    kt.update(1.0, 1.0)

    lat, lng, lat_std, lng_std = kt.predict_position(0.0)

    assert 0.0 < lat < 1.0
    assert 0.0 < lng < 1.0
    assert lat_std > 0.0
    assert lng_std > 0.0


def test_kalman_tracker_predict_position_does_not_mutate_state():
    kt = klm.TKalmanTracker(0.0, 0.0)
    kt.update(1.0, 1.0)
    x_before = kt._lat.x

    kt.predict_position(5.0)

    assert kt._lat.x == pytest.approx(x_before)


def test_kalman_tracker_to_dict_from_dict_round_trip():
    kt = klm.TKalmanTracker(3.0, -4.0)
    kt.update(3.1, -3.9)

    restored = klm.TKalmanTracker.from_dict(kt.to_dict())

    assert restored._lat.x == pytest.approx(kt._lat.x)
    assert restored._lat.v == pytest.approx(kt._lat.v)
    assert restored._lng.x == pytest.approx(kt._lng.x)
    assert restored._lng.v == pytest.approx(kt._lng.v)
