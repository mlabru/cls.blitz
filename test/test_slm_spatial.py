import pytest

import work.slm_spatial as spc
import work.slm_spcgrid as sgr
import work.slm_struct as sst


def _pt(lat, lng):
    return sst.TLightning(ts=0.0, lat=lat, lng=lng)


def test_haversine_km_zero_distance():
    assert spc.haversine_km(0.0, 0.0, 0.0, 0.0) == 0.0


def test_haversine_km_one_degree_longitude_at_equator():
    # 1 degree of longitude at the equator is Earth's circumference / 360
    assert spc.haversine_km(0.0, 0.0, 0.0, 1.0) == pytest.approx(111.19, abs=0.05)


def test_haversine_km_is_symmetric():
    a = spc.haversine_km(-10.0, 20.0, 15.0, -30.0)
    b = spc.haversine_km(15.0, -30.0, -10.0, 20.0)
    assert a == pytest.approx(b)


def test_dbscan_spatial_separates_cluster_from_noise():
    cluster_pts = [_pt(0.0, 0.0) for _ in range(6)]
    noise_pts = [_pt(10.0, 10.0), _pt(-10.0, -10.0)]

    clusters = spc.dbscan_spatial(cluster_pts + noise_pts, ff_eps_km=5.0, fi_min_pts=5)

    assert len(clusters) == 1
    assert len(clusters[0]) == 6


def test_dbscan_spatial_empty_input():
    assert spc.dbscan_spatial([]) == []


def test_region_query_within_radius():
    grid = sgr.TSpatialGrid(5.0)
    pts = [_pt(0.0, 0.0), _pt(0.0, 0.001), _pt(50.0, 50.0)]
    for p in pts:
        grid.insert(p)

    neighbors = spc.region_query(grid, _pt(0.0, 0.0), 5.0)

    assert len(neighbors) == 2
