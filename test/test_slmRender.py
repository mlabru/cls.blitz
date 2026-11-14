import pytest

import slmRender as srr


def test_severity_color_boundaries():
    assert srr.severity_color(0.0) == "#2b83ba"
    assert srr.severity_color(0.19) == "#2b83ba"
    assert srr.severity_color(0.2) == "#abdda4"
    assert srr.severity_color(0.4) == "#ffffbf"
    assert srr.severity_color(0.6) == "#fdae61"
    assert srr.severity_color(0.8) == "#d7191c"
    assert srr.severity_color(1.0) == "#d7191c"


def test_compute_severity_local_weighted_sum():
    cell = {"size": 100, "density": 0.5, "flash_rate": 0.6, "cg_ratio": 0.4}

    # size_norm = min(100/200, 1.0) = 0.5; (0.5+0.6+0.5+0.4) * 0.25 = 0.5
    assert srr._compute_severity_local(cell) == pytest.approx(0.5)


def test_compute_severity_local_caps_at_one():
    cell = {"size": 10_000, "density": 1.0, "flash_rate": 1.0, "cg_ratio": 1.0}

    assert srr._compute_severity_local(cell) == 1.0


def test_forecast_cone_needs_at_least_two_points():
    assert srr.forecast_cone(0.0, 0.0, [{"lat": 0.0, "lng": 0.0}], 0.0) is None


def test_forecast_cone_fallback_width_without_kalman_std():
    forecast = [{"lat": 0.0, "lng": 0.0}, {"lat": 0.0, "lng": 1.0}]

    cone = srr.forecast_cone(0.0, 0.0, forecast, ff_ref_lat=0.0)

    assert cone is not None
    assert len(cone) == 3
    assert cone[2] == pytest.approx((1.0, 0.0), abs=1e-6)
    # fallback width = 25% of the forecast vector -> off_lat = (mag_km * 0.25) / 111
    assert cone[0][1] == pytest.approx(0.25, abs=2e-3)


def test_forecast_cone_uses_kalman_uncertainty_when_available():
    forecast = [
        {"lat": 0.0, "lng": 0.0, "lat_std": 0.01, "lng_std": 0.01},
        {"lat": 0.0, "lng": 1.0, "lat_std": 0.05, "lng_std": 0.05},
    ]

    cone = srr.forecast_cone(0.0, 0.0, forecast, ff_ref_lat=0.0)

    # off_lat = ((0.05*111 + 0.05*111)/2 * 2) / 111 = 0.1, distinct from the fallback's 0.25
    assert cone[0][1] == pytest.approx(0.1, abs=1e-3)


def test_validate_cell_requires_keys_and_valid_centroid():
    assert srr._validate_cell({"centroid": [1, 2], "size": 1, "t_end": 1}, 0) is True
    assert srr._validate_cell({"size": 1, "t_end": 1}, 0) is False
    assert srr._validate_cell({"centroid": "bad", "size": 1, "t_end": 1}, 0) is False
    assert srr._validate_cell({"centroid": [1, 2, 3], "size": 1, "t_end": 1}, 0) is False


class _FakeAxes:
    def __init__(self):
        self.quivers = []

    def quiver(self, *args, **kwargs):
        self.quivers.append(args)


@pytest.mark.parametrize("speed,drawn", [
    (0.0, False),
    (srr.CF_MIN_ARROW_SPEED_KMH - 0.1, False),   # Kalman quase parado: sem seta
    (srr.CF_MIN_ARROW_SPEED_KMH, True),
    (40.0, True),
])
def test_motion_arrow_only_above_min_speed(speed, drawn):
    axes = _FakeAxes()

    srr._draw_motion_arrow(axes, None, {"centroid": [0.0, 0.0], "speed_kmh": speed,
                                        "direction_deg": 90.0})

    assert bool(axes.quivers) == drawn


def test_hud_shows_source_and_its_window(monkeypatch):
    captured = {}
    monkeypatch.setattr(srr.mpb, "draw_hud", lambda ax, text, fi_fontsize: captured.update(text=text))

    srr._draw_hud(None, [], "glm")

    assert "Source : GLM" in captured["text"] and "Window : 20 min" in captured["text"]
