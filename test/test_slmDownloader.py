import datetime as real_datetime
import json

import pytest

import slmDownloader as dl


class _FixedDateTime(real_datetime.datetime):
    """monkeypatch target for dl.datetime.datetime — freezes 'now' for deterministic tests."""

    @classmethod
    def now(cls, tz=None):
        fixed = real_datetime.datetime(2026, 7, 9, 0, 5, tzinfo=real_datetime.timezone.utc)
        return fixed if tz is None else fixed.astimezone(tz)


class _FakeResponse:
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text


def test_parse_hora_gmt_same_day(monkeypatch):
    monkeypatch.setattr(dl.datetime, "datetime", _FixedDateTime)

    result = dl.parse_hora_gmt("00:10")

    assert (result.day, result.hour, result.minute) == (9, 0, 10)


def test_parse_hora_gmt_midnight_rollback(monkeypatch):
    monkeypatch.setattr(dl.datetime, "datetime", _FixedDateTime)

    # "now" is 00:05 on the 9th; "23:59" is > 1h in the future, so it must
    # belong to the previous day.
    result = dl.parse_hora_gmt("23:59")

    assert (result.day, result.hour, result.minute) == (8, 23, 59)


def test_parse_hora_gmt_invalid_format_returns_none():
    assert dl.parse_hora_gmt("not-a-time") is None


def test_filter_recent_rows_drops_old_and_malformed():
    rows = [
        {"timestamp": "1000"},
        {"timestamp": "2000"},
        {"timestamp": "abc"},
        {"lat": "1"},
    ]
    cutoff = real_datetime.datetime.fromtimestamp(1500, tz=real_datetime.timezone.utc)

    result = dl.filter_recent_rows(rows, cutoff)

    assert result == [{"timestamp": "2000"}]


def test_row_key_normalizes_types():
    row_str = {"timestamp": "100", "lat": "1.5", "lng": "2.5"}
    row_mixed = {"timestamp": 100.0, "lat": "1.5", "lng": "2.5"}

    assert dl._row_key(row_str) == dl._row_key(row_mixed) == (100, "1.5", "2.5")


def test_load_existing_data_reads_rows(tmp_path):
    csv_path = tmp_path / "in.csv"
    csv_path.write_text("timestamp,lat,lng\n1,2,3\n4,5,6\n")

    rows = dl.load_existing_data(str(csv_path))

    assert len(rows) == 2
    assert rows[0]["timestamp"] == "1"


def test_load_existing_data_missing_file_returns_empty(tmp_path):
    assert dl.load_existing_data(str(tmp_path / "missing.csv")) == []


def test_get_redemet_raios_success(monkeypatch):
    payload = {"status": True, "data": {"anima": ["12:00"], "stsc": [[{"la": 1.0, "lo": 2.0}]]}}
    monkeypatch.setattr(dl.requests, "get", lambda *a, **k: _FakeResponse(200, json.dumps(payload)))

    result = dl.get_redemet_raios()

    assert result == {"anima": ["12:00"], "stsc": [[{"la": 1.0, "lo": 2.0}]]}


def test_get_redemet_raios_http_error(monkeypatch):
    monkeypatch.setattr(dl.requests, "get", lambda *a, **k: _FakeResponse(500, ""))

    assert dl.get_redemet_raios() is None


def test_get_redemet_raios_bad_json(monkeypatch):
    monkeypatch.setattr(dl.requests, "get", lambda *a, **k: _FakeResponse(200, "not json"))

    assert dl.get_redemet_raios() is None


def test_get_redemet_raios_status_false(monkeypatch):
    payload = {"status": False}
    monkeypatch.setattr(dl.requests, "get", lambda *a, **k: _FakeResponse(200, json.dumps(payload)))

    assert dl.get_redemet_raios() is None


def test_get_redemet_raios_missing_data_field(monkeypatch):
    payload = {"status": True}
    monkeypatch.setattr(dl.requests, "get", lambda *a, **k: _FakeResponse(200, json.dumps(payload)))

    assert dl.get_redemet_raios() is None


def test_download_raios_writes_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "get_redemet_raios", lambda: {
        "anima": ["00:00", "00:01"],
        "stsc": [[{"la": 1.0, "lo": 2.0}], [{"la": 3.0, "lo": 4.0}]],
    })
    monkeypatch.setattr(dl.datetime, "datetime", _FixedDateTime)

    class _Args:
        saida = str(tmp_path)
        fname = "out.csv"
        archive_dir = None

    assert dl.download_raios(_Args()) is True

    lines = (tmp_path / "out.csv").read_text().strip().splitlines()
    assert lines[0] == "timestamp,lat,lng"
    assert len(lines) == 3


def test_download_raios_no_data_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "get_redemet_raios", lambda: None)

    class _Args:
        saida = str(tmp_path)
        fname = "out.csv"
        archive_dir = None

    assert dl.download_raios(_Args()) is False


def test_extract_events_sem_anima_retorna_none():
    # [fix] "data" sem "anima"/"stsc" levantava KeyError
    assert dl._extract_events({"status": True, "data": {"stsc": []}}) is None


def test_snapshot_to_rows_pula_posicao_sem_la_lo():
    cutoff = real_datetime.datetime(2000, 1, 1, tzinfo=real_datetime.timezone.utc)

    rows = dl._snapshot_to_rows(["12:00"], [[{"la": -10.0, "lo": -50.0}, {"la": -11.0}]], cutoff)

    assert [(r["lat"], r["lng"]) for r in rows] == [(-10.0, -50.0)]


# ─────────────────────────────────────────────────────────────────────────────────────
# deduplicação por contagem (2026.nov)

def _rows(*keys):
    return [{"timestamp": t, "lat": la, "lng": lo} for t, la, lo in keys]


def test_drop_duplicates_snapshot_repeating_csv_adds_nothing():
    kept = _rows((60, "1.5", "2.5"), (60, "1.5", "2.5"), (120, "3.0", "4.0"))
    new = _rows((60, 1.5, 2.5), (60, 1.5, 2.5), (120, 3.0, 4.0))

    assert dl._drop_duplicates(new, kept) == []


def test_drop_duplicates_keeps_late_strike_with_same_key():
    # [fix] o 2º raio real no mesmo minuto e ponto chegou atrasado: antes era descartado
    kept = _rows((60, "1.5", "2.5"))
    new = _rows((60, 1.5, 2.5), (60, 1.5, 2.5))

    assert dl._drop_duplicates(new, kept) == _rows((60, 1.5, 2.5))


def test_drop_duplicates_first_download_keeps_all_repeats():
    new = _rows((60, 1.5, 2.5), (60, 1.5, 2.5))

    assert dl._drop_duplicates(new, []) == new


def test_drop_duplicates_preserves_order_of_new_rows():
    kept = _rows((60, "1.5", "2.5"))
    new = _rows((120, 9.0, 9.0), (60, 1.5, 2.5), (180, 8.0, 8.0))

    assert dl._drop_duplicates(new, kept) == _rows((120, 9.0, 9.0), (180, 8.0, 8.0))


# ─────────────────────────────────────────────────────────────────────────────────────
# arquivo diário dos raios (2026.nov)

def _snapshot(monkeypatch, positions_per_minute):
    monkeypatch.setattr(dl, "get_redemet_raios", lambda: {
        "anima": ["00:00", "00:01"][:len(positions_per_minute)],
        "stsc": positions_per_minute,
    })
    monkeypatch.setattr(dl.datetime, "datetime", _FixedDateTime)


def _args(tmp_path, archive):
    class _Args:
        saida = str(tmp_path)
        fname = "out.csv"
        archive_dir = archive
    return _Args()


def test_download_raios_archives_only_new_rows(tmp_path, monkeypatch):
    archive = tmp_path / "archive"
    monkeypatch.setattr(dl.time, "time", lambda: 1_000_000.0)
    _snapshot(monkeypatch, [[{"la": 1.0, "lo": 2.0}]])
    assert dl.download_raios(_args(tmp_path, str(archive))) is True

    # a API repete a hora inteira: só o raio do minuto novo é arquivado
    monkeypatch.setattr(dl.time, "time", lambda: 1_000_300.0)
    _snapshot(monkeypatch, [[{"la": 1.0, "lo": 2.0}], [{"la": 3.0, "lo": 4.0}]])
    assert dl.download_raios(_args(tmp_path, str(archive))) is True

    rows = dl.arc.load([str(p) for p in archive.iterdir()])
    assert [(r[0], r[2], r[3]) for r in rows] == [(1_000_000.0, 1.0, 2.0), (1_000_300.0, 3.0, 4.0)]


def test_download_raios_archive_failure_is_only_a_warning(tmp_path, monkeypatch, caplog):
    blocker = tmp_path / "blocker"
    blocker.write_text("")                                # um arquivo onde iria o diretório
    _snapshot(monkeypatch, [[{"la": 1.0, "lo": 2.0}]])

    assert dl.download_raios(_args(tmp_path, str(blocker / "archive"))) is True

    assert (tmp_path / "out.csv").exists()
    assert "falha ao arquivar" in caplog.text


def test_download_raios_without_archive_dir_writes_no_archive(tmp_path, monkeypatch):
    _snapshot(monkeypatch, [[{"la": 1.0, "lo": 2.0}]])

    assert dl.download_raios(_args(tmp_path, None)) is True

    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.csv"]


def test_arg_parse_archive_dir():
    assert dl.arg_parse([]).archive_dir is None
    assert dl.arg_parse(["-a", "/x"]).archive_dir == "/x"
