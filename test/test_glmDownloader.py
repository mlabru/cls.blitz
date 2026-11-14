import csv
import datetime
import io
import json

import h5py
import numpy as np
import pytest

import glmDownloader as glm


# 2026-11-14 19:01:00 UTC = dia 318
_T0 = datetime.datetime(2026, 11, 14, 19, 1, 0, tzinfo=datetime.timezone.utc).timestamp()
_KEY = "GLM-L2-LCFA/2026/318/19/OR_GLM-L2-LCFA_G19_s20263181901000_e20263181901200_c20263181901224.nc"


def _key_at(ts):
    dt = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)
    stamp = dt.strftime("%Y%j%H%M%S") + "0"
    return dt.strftime("GLM-L2-LCFA/%Y/%j/%H/") + "OR_GLM-L2-LCFA_G19_s%s_e%s_c%s.nc" % (stamp, stamp, stamp)


def _packed(f, name, raw, **attrs):
    ds = f.create_dataset(name, data=np.asarray(raw, dtype=np.int16))
    for k, v in attrs.items():
        ds.attrs[k] = v
    return ds


def _glm_bytes(lats, lngs, offsets_s, quality):
    """um LCFA mínimo, empacotado como o da NOAA (int16 sem sinal + scale/offset)."""
    buf = io.BytesIO()
    scale, offset = np.float32(0.00038148), np.float32(-5.0)
    raw_t = np.round((np.asarray(offsets_s) - offset) / scale).astype(np.uint16).view(np.int16)
    with h5py.File(buf, "w") as f:
        f.create_dataset("flash_lat", data=np.asarray(lats, dtype=np.float32))
        f.create_dataset("flash_lon", data=np.asarray(lngs, dtype=np.float32))
        _packed(f, "flash_time_offset_of_first_event", raw_t,
                scale_factor=np.array([scale]), add_offset=np.array([offset]),
                _Unsigned=np.bytes_(b"true"),
                units=np.bytes_(b"seconds since 2026-11-14 19:01:00.000"))
        _packed(f, "flash_quality_flag", np.asarray(quality, dtype=np.uint16).view(np.int16),
                _FillValue=np.array([-1], dtype=np.int16), _Unsigned=np.bytes_(b"true"))
    return buf.getvalue()


def _listing(keys, truncated=False, token=None):
    ns = "http://s3.amazonaws.com/doc/2006-03-01/"
    body = "".join("<Contents><Key>%s</Key></Contents>" % k for k in keys)
    extra = "<NextContinuationToken>%s</NextContinuationToken>" % token if token else ""
    return ('<?xml version="1.0"?><ListBucketResult xmlns="%s"><IsTruncated>%s</IsTruncated>%s%s'
            "</ListBucketResult>" % (ns, "true" if truncated else "false", extra, body)).encode()


# ─────────────────────────────────────────────────────────────────────────────────────
# nome do arquivo e listagem

def test_file_start_ts_parses_day_of_year_and_tenths():
    key = _KEY.replace("s20263181901000", "s20263181901205")

    assert glm.file_start_ts(_KEY) == _T0
    assert glm.file_start_ts(key) == pytest.approx(_T0 + 20.5)
    assert glm.file_start_ts("lixo.nc") is None


def test_hour_prefixes_cover_the_interval():
    prefixes = glm._hour_prefixes(_T0 - 3600, _T0)

    assert prefixes == ["GLM-L2-LCFA/2026/318/18/", "GLM-L2-LCFA/2026/318/19/"]


def test_parse_listing_returns_token_only_when_truncated():
    assert glm.parse_listing(_listing(["a", "b"])) == (["a", "b"], None)
    assert glm.parse_listing(_listing(["a"], truncated=True, token="x")) == (["a"], "x")


def test_list_prefix_follows_pagination(monkeypatch):
    pages = [_listing(["a"], truncated=True, token="t1"), _listing(["b"])]
    calls = []

    def fake_get(url, params=None):
        calls.append(dict(params))
        return pages[len(calls) - 1]

    monkeypatch.setattr(glm, "_http_get", fake_get)

    assert glm._list_prefix("P/") == ["a", "b"]
    assert calls[1]["continuation-token"] == "t1"


def test_list_glm_files_keeps_only_the_interval_in_order(monkeypatch):
    keys = [_key_at(_T0 - 4000), _key_at(_T0 - 20), _key_at(_T0 - 40), _key_at(_T0 + 60)]
    monkeypatch.setattr(glm, "_list_prefix", lambda prefix: [k for k in keys if k.startswith(prefix)])

    result = glm.list_glm_files(_T0 - 3600, _T0)

    assert result == [_key_at(_T0 - 40), _key_at(_T0 - 20)]


def test_select_new_files_skips_seen_and_keeps_newest_when_over_limit(monkeypatch):
    keys = [_key_at(_T0 + 20 * i) for i in range(5)]
    seen = {keys[1].rsplit("/", 1)[1]}
    monkeypatch.setattr(glm, "CI_MAX_FILES_PER_RUN", 2)

    assert glm._select_new_files(keys, seen, _T0) == keys[3:]


# ─────────────────────────────────────────────────────────────────────────────────────
# leitura do NetCDF

def test_read_glm_flashes_unpacks_time_and_filters_quality_and_domain():
    data = _glm_bytes(lats=[-23.5, -23.6, 40.0, -10.0],
                      lngs=[-46.6, -46.7, -46.6, -50.0],
                      offsets_s=[1.5, -0.5, 2.0, 10.0],
                      quality=[0, 0, 0, 3])

    rows = glm.read_glm_flashes(io.BytesIO(data))

    # 40N está fora do domínio; qualidade 3 é descartada; em ordem de instante
    assert [(r["lat"], r["lng"]) for r in rows] == [(-23.6, -46.7), (-23.5, -46.6)]
    assert rows[0]["timestamp"] == pytest.approx(_T0 - 0.5, abs=1e-3)
    assert rows[1]["timestamp"] == pytest.approx(_T0 + 1.5, abs=1e-3)


def test_unpack_marks_fill_value_as_nan():
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        ds = _packed(f, "v", [-1, 7], _FillValue=np.array([-1], dtype=np.int16),
                     scale_factor=np.array([2.0]), _Unsigned=np.bytes_(b"true"))
        values = glm.unpack(ds)

    assert np.isnan(values[0]) and values[1] == 14.0


def test_read_glm_flashes_rejects_unknown_time_units():
    buf = io.BytesIO(_glm_bytes([-23.5], [-46.6], [1.0], [0]))
    with h5py.File(buf, "r+") as f:
        f["flash_time_offset_of_first_event"].attrs["units"] = np.bytes_(b"minutes since x")

    with pytest.raises(ValueError):
        glm.read_glm_flashes(io.BytesIO(buf.getvalue()))


# ─────────────────────────────────────────────────────────────────────────────────────
# deduplicação e rodada completa

def test_merge_rows_drops_rows_already_in_csv_and_repeats():
    kept = [{"timestamp": "1000.5", "lat": "-23.5", "lng": "-46.6"}]
    new = [{"timestamp": 1000.5, "lat": -23.5, "lng": -46.6},
           {"timestamp": 1001.0, "lat": -23.5, "lng": -46.6},
           {"timestamp": 1001.0, "lat": -23.5, "lng": -46.6}]

    assert glm._merge_rows(kept, new) == [new[1]]


def _args(tmp_path, archive=True):
    return glm.arg_parse(["-s", str(tmp_path / "raw")] +
                         (["-a", str(tmp_path / "archive")] if archive else []))


def _fake_bucket(monkeypatch, files):
    """files: {chave: bytes}; o bucket lista as chaves e serve o conteúdo."""
    downloads = []

    def fake_get(url, params=None):
        if params is not None:
            prefix = params["prefix"]
            return _listing([k for k in files if k.startswith(prefix)])
        key = url.split(glm.CS_BUCKET_URL + "/", 1)[1]
        downloads.append(key)
        return files.get(key)

    monkeypatch.setattr(glm, "_http_get", fake_get)
    return downloads


def test_download_glm_writes_csv_seen_file_and_archive(tmp_path, monkeypatch):
    files = {_KEY: _glm_bytes([-23.5, -23.6], [-46.6, -46.7], [1.0, 2.0], [0, 0])}
    _fake_bucket(monkeypatch, files)

    assert glm.download_glm(_args(tmp_path), ff_now_ts=_T0 + 60)

    with open(tmp_path / "raw" / glm.CS_OUTPUT_FILE) as f:
        rows = list(csv.DictReader(f))
    assert [float(r["timestamp"]) for r in rows] == pytest.approx([_T0 + 1.0, _T0 + 2.0], abs=1e-3)
    assert json.loads((tmp_path / "raw" / glm.CS_SEEN_FILE).read_text()) == [_KEY.rsplit("/", 1)[1]]
    assert list((tmp_path / "archive").glob("glm_raw_*.csv.gz"))


def test_download_glm_second_run_does_not_download_or_duplicate(tmp_path, monkeypatch):
    files = {_KEY: _glm_bytes([-23.5], [-46.6], [1.0], [0])}
    downloads = _fake_bucket(monkeypatch, files)

    glm.download_glm(_args(tmp_path, archive=False), ff_now_ts=_T0 + 60)
    glm.download_glm(_args(tmp_path, archive=False), ff_now_ts=_T0 + 120)

    assert downloads == [_KEY]
    assert len((tmp_path / "raw" / glm.CS_OUTPUT_FILE).read_text().splitlines()) == 2


def test_download_glm_lost_seen_file_rereads_without_duplicating(tmp_path, monkeypatch):
    files = {_KEY: _glm_bytes([-23.5], [-46.6], [1.0], [0])}
    downloads = _fake_bucket(monkeypatch, files)

    glm.download_glm(_args(tmp_path, archive=False), ff_now_ts=_T0 + 60)
    (tmp_path / "raw" / glm.CS_SEEN_FILE).write_text("{corrompido")
    glm.download_glm(_args(tmp_path, archive=False), ff_now_ts=_T0 + 120)

    assert downloads == [_KEY, _KEY]
    assert len((tmp_path / "raw" / glm.CS_OUTPUT_FILE).read_text().splitlines()) == 2


def test_download_glm_unreadable_file_is_retried_next_run(tmp_path, monkeypatch):
    files = {_KEY: b"nao e hdf5"}
    _fake_bucket(monkeypatch, files)

    assert glm.download_glm(_args(tmp_path, archive=False), ff_now_ts=_T0 + 60)

    assert json.loads((tmp_path / "raw" / glm.CS_SEEN_FILE).read_text()) == []


def test_download_glm_drops_rows_older_than_retention(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / glm.CS_OUTPUT_FILE).write_text("timestamp,lat,lng\n%r,-23.5,-46.6\n" % (_T0 - 7200))
    _fake_bucket(monkeypatch, {})

    assert glm.download_glm(_args(tmp_path, archive=False), ff_now_ts=_T0)

    assert (raw / glm.CS_OUTPUT_FILE).read_text().splitlines() == ["timestamp,lat,lng"]


def test_main_returns_fail_when_bucket_is_unreachable(tmp_path, monkeypatch):
    monkeypatch.setattr(glm, "_http_get", lambda url, params=None: None)

    assert glm.main(["-s", str(tmp_path)]) == glm.CI_RC_FAIL


def test_csv_is_readable_by_the_tracker(tmp_path, monkeypatch):
    import bltTracker as trkmod

    files = {_KEY: _glm_bytes([-23.5], [-46.6], [1.0], [0])}
    _fake_bucket(monkeypatch, files)
    glm.download_glm(_args(tmp_path, archive=False), ff_now_ts=_T0 + 60)

    events = trkmod.load_csv(str(tmp_path / "raw" / glm.CS_OUTPUT_FILE))

    assert [(e.lat, e.lng) for e in events] == [(-23.5, -46.6)]
    assert events[0].ts == pytest.approx(_T0 + 1.0, abs=1e-3)
