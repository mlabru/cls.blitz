import work.slm_spcgrid as sgr
import work.slm_struct as sst


def _pt(lat, lng):
    return sst.TLightning(ts=0.0, lat=lat, lng=lng)


def test_insert_and_neighbors_same_cell():
    grid = sgr.TSpatialGrid(10.0)
    a, b = _pt(0.0, 0.0), _pt(0.01, 0.01)
    grid.insert(a)
    grid.insert(b)

    result = grid.neighbors(0.0, 0.0)

    assert len(result) == 2
    assert a in result and b in result


def test_neighbors_excludes_far_points():
    grid = sgr.TSpatialGrid(10.0)
    near = _pt(0.0, 0.0)
    far = _pt(50.0, 50.0)
    grid.insert(near)
    grid.insert(far)

    assert grid.neighbors(0.0, 0.0) == [near]


def test_neighbors_empty_grid():
    grid = sgr.TSpatialGrid(10.0)

    assert grid.neighbors(0.0, 0.0) == []
