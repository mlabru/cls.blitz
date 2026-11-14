import pytest

import work.glm_ljump as ljp

_END = 10_000.0


def _flashes(rates, end=_END, bin_min=2.0):
    """flashes distribuídos uniformemente em cada intervalo: rates[i] por minuto."""
    n = len(rates)
    start = end - n * bin_min * 60.0
    ts = []
    for i, rate in enumerate(rates):
        count = int(round(rate * bin_min))
        b0 = start + i * bin_min * 60.0
        ts += [b0 + (k + 0.5) * bin_min * 60.0 / count for k in range(count)]
    return ts


def test_rate_series_counts_per_bin_with_end_inclusive():
    ts = [_END, _END - 1, _END - 120, _END - 121, _END - 1200, _END + 1]

    rates = ljp.rate_series(ts, _END, 20.0)

    # 10 intervalos; _END-120 fecha o penúltimo; _END-1200 (início) e _END+1 ficam fora
    assert len(rates) == 10
    assert rates[-1] == 1.0 and rates[-2] == 1.0 and sum(rates) == 2.0


def test_steady_cell_has_no_jump():
    ts = _flashes([20, 21, 19, 20, 22, 20, 21, 19, 20, 21])

    result = ljp.evaluate(ts, _END, 20.0)

    assert result.rate_now == pytest.approx(21.0)
    assert not result.jump and result.sigma_level < ljp.CF_LJ_SIGMA


def test_sudden_increase_is_a_jump():
    ts = _flashes([20, 21, 19, 20, 22, 20, 21, 19, 20, 45])

    result = ljp.evaluate(ts, _END, 20.0)

    assert result.jump and result.jump_ts == _END
    assert result.sigma_level > ljp.CF_LJ_SIGMA
    assert result.dfrdt_now == pytest.approx((45 - 20) / 2.0)


def test_jump_below_activation_rate_is_not_flagged():
    ts = _flashes([2, 2.5, 2, 2, 2.5, 2, 2.5, 2, 2, 8])

    result = ljp.evaluate(ts, _END, 20.0)

    assert result.sigma_level > ljp.CF_LJ_SIGMA and not result.jump


def test_new_cell_growing_from_zero_is_not_a_jump():
    ts = _flashes([0, 0, 0, 0, 0, 0, 0, 5, 15, 40])

    result = ljp.evaluate(ts, _END, 20.0)

    assert not result.jump and result.sigma_level == 0.0


def test_jump_between_frames_is_caught_with_since_ts():
    # o salto foi no intervalo que termina 4 min antes do frame; o frame
    # anterior foi há 5 min (cron)
    ts = _flashes([20, 21, 19, 20, 22, 20, 21, 45, 46, 45])

    only_last = ljp.evaluate(ts, _END, 20.0)
    since_previous = ljp.evaluate(ts, _END, 20.0, ff_since_ts=_END - 300.0)

    assert not only_last.jump
    assert since_previous.jump and since_previous.jump_ts == _END - 240.0


def test_constants_are_read_at_call_time(monkeypatch):
    ts = _flashes([20, 21, 19, 20, 22, 20, 21, 19, 20, 45])
    monkeypatch.setattr(ljp, "CF_LJ_MIN_RATE", 100.0)

    assert not ljp.evaluate(ts, _END, 20.0).jump


def test_rate_recent_is_mean_of_last_bins():
    ts = _flashes([0, 0, 0, 0, 0, 0, 0, 10, 20, 30])

    assert ljp.evaluate(ts, _END, 20.0).rate_recent == pytest.approx(20.0)
