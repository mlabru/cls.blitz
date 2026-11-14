import pytest

import work.blt_spatial as spc
import work.blt_spcgrid as sgr
import work.blt_struct as sst


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
    assert spc.dbscan_spatial([], ff_eps_km=5.0, fi_min_pts=5) == []


def test_region_query_within_radius():
    grid = sgr.TSpatialGrid(5.0)
    pts = [_pt(0.0, 0.0), _pt(0.0, 0.001), _pt(50.0, 50.0)]
    for p in pts:
        grid.insert(p)

    neighbors = spc.region_query(grid, _pt(0.0, 0.0), 5.0)

    assert len(neighbors) == 2


# ─────────────────────────────────────────────────────────────────────────────────────
# raio de borda em um cluster só (2026.nov)

def _two_clusters_sharing_a_border_point():
    # eps = 5 km, min_pts = 5.  a0 e b0 são centrais (5 vizinhos cada + o ponto P),
    # a 8.9 km um do outro (clusters separados).  P fica a 4.45 km dos dois, mas
    # só tem 3 vizinhos (P, a0, b0): é borda dos DOIS clusters.
    a_core = [_pt(0.0, -0.02) for _ in range(5)] + [_pt(0.0, 0.0)]
    b_core = [_pt(0.0, 0.10) for _ in range(5)] + [_pt(0.0, 0.08)]
    border = _pt(0.0, 0.04)
    return a_core, b_core, border


def test_dbscan_border_point_goes_to_one_cluster_only():
    a_core, b_core, border = _two_clusters_sharing_a_border_point()

    clusters = spc.dbscan_spatial(a_core + b_core + [border], ff_eps_km=5.0, fi_min_pts=5)

    assert len(clusters) == 2
    all_ids = [id(p) for c in clusters for p in c]
    assert len(all_ids) == len(set(all_ids)) == 13
    # fica com o primeiro cluster que o alcança
    assert border in clusters[0] and border not in clusters[1]


def test_dbscan_border_point_order_decides_cluster():
    a_core, b_core, border = _two_clusters_sharing_a_border_point()

    clusters = spc.dbscan_spatial(b_core + a_core + [border], ff_eps_km=5.0, fi_min_pts=5)

    assert border in clusters[0] and border not in clusters[1]
    assert sorted(len(c) for c in clusters) == [6, 7]


def test_expand_cluster_without_global_set_still_works():
    pts = [_pt(0.0, 0.0) for _ in range(6)]
    grid = sgr.TSpatialGrid(5.0)
    for p in pts:
        grid.insert(p)

    cluster = spc.expand_cluster(grid, pts[0], spc.region_query(grid, pts[0], 5.0),
                                 {id(pts[0])}, 5.0, 5)

    assert len(cluster) == 6
