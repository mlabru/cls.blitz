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


# ─────────────────────────────────────────────────────────────────────────────────────
# ruído de processo proporcional a dt (2026.nov)

def _p_after(dt_h):
    k = klm._TKalman1D(0.0)
    k.p_xx = k.p_xv = k.p_vv = 0.0
    k.predict(dt_h)
    return k.p_xx, k.p_vv


def test_kalman1d_process_noise_scales_with_dt():
    ref = klm.CF_KALMAN_Q_REF_DT_H

    assert _p_after(ref) == pytest.approx((klm.CF_KALMAN_Q_POS, klm.CF_KALMAN_Q_VEL))
    assert _p_after(4 * ref) == pytest.approx((4 * klm.CF_KALMAN_Q_POS, 4 * klm.CF_KALMAN_Q_VEL))


def test_kalman1d_zero_dt_adds_no_noise():
    assert _p_after(0.0) == (0.0, 0.0)
    assert _p_after(-1.0) == (0.0, 0.0)


def test_forecast_uncertainty_grows_with_lead():
    kt = klm.TKalmanTracker(0.0, 0.0)
    kt.update(0.0, 0.0)

    stds = [kt.predict_position(lead / 60.0)[2] for lead in (10, 20, 30)]

    assert stds[0] < stds[1] < stds[2]


def test_velocity_deg_h():
    kt = klm.TKalmanTracker(0.0, 0.0, ff_vlat=0.5, ff_vlng=-1.0)

    assert kt.velocity_deg_h == (0.5, -1.0)


# ─────────────────────────────────────────────────────────────────────────────────────
# inicialização por dois pontos e filho de divisão (2026.nov)

def test_kalman1d_from_two_points_state_and_covariance():
    dt = 5.0 / 60.0
    k = klm._TKalman1D.from_two_points(1.0, 1.1, dt)

    assert k.x == 1.1
    assert k.v == pytest.approx(0.1 / dt)
    assert (k.p_xx, k.p_xv, k.p_vv) == pytest.approx((k.r, k.r / dt, 2.0 * k.r / dt ** 2))
    assert (k.q_pos, k.q_vel) == (klm.CF_KALMAN_Q_POS, klm.CF_KALMAN_Q_VEL)


def test_kalman_tracker_from_two_points_extrapolates_the_displacement():
    kt = klm.TKalmanTracker.from_two_points((0.0, 0.0), (0.5, 1.0), 1.0)

    lat, lng, _, _ = kt.predict_position(1.0)

    assert kt.velocity_deg_h == pytest.approx((0.5, 1.0))
    assert (lat, lng) == pytest.approx((1.0, 2.0))


def test_kalman_tracker_from_two_points_round_trips():
    kt = klm.TKalmanTracker.from_two_points((0.0, 0.0), (0.5, 1.0), 1.0)

    restored = klm.TKalmanTracker.from_dict(kt.to_dict())

    assert restored.to_dict() == kt.to_dict()


def test_spawn_at_inherits_velocity_and_covariance_not_position():
    parent = klm.TKalmanTracker(0.0, 0.0, ff_vlat=0.2, ff_vlng=-0.3)
    parent.predict(0.1)
    parent.update(0.02, -0.03)
    before = parent.to_dict()

    child = parent.spawn_at(5.0, 6.0)

    assert child.velocity_deg_h == parent.velocity_deg_h
    assert child.predict_position(0.0)[:2] == (5.0, 6.0)
    assert child.predict_position(1.0)[2:] == parent.predict_position(1.0)[2:]
    assert parent.to_dict() == before                     # o pai não muda
    child.update(9.0, 9.0)
    assert parent.to_dict() == before                     # e não compartilha estado
