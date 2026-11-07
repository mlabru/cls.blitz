import math

import pytest

import work.slm_diag as dgn
import work.slm_report as rpt
import work.slm_verify as vrf


def _v(lead, status="ok", error="", persist="", z2=""):
    return {"lead_min": str(lead), "status": status, "error_km": str(error),
            "persist_km": str(persist), "z2": str(z2)}


def test_summarize_verification_per_lead():
    rows = [_v(10, error=2, persist=10, z2=0.5), _v(10, error=4, persist=10, z2=3.0),
            _v(10, "gap"), _v(10, "missed"), _v(20, error=5, persist=20, z2=9.0)]

    leads = rpt.summarize_verification(rows)

    assert [lead["lead_min"] for lead in leads] == [10, 20]
    ten = leads[0]
    assert (ten["n"], ten["n_ok"], ten["frac_ok"]) == (4, 2, 0.5)
    assert ten["lost"] == {"gap": 1, "missed": 1}
    assert (ten["error_km"], ten["persist_km"], ten["skill"]) == (3.0, 10.0, pytest.approx(0.7))
    assert (ten["z2_mean"], ten["frac_z2_le_1"], ten["frac_z2_le_4"]) == (1.75, 0.5, 1.0)
    assert leads[1]["frac_z2_le_4"] == 0.0


def test_summarize_verification_rows_without_z2():
    lead = rpt.summarize_verification([_v(10, error=1, persist=2)])[0]

    assert lead["n_z2"] == 0 and math.isnan(lead["z2_mean"])


def test_chi2_reference_values():
    assert rpt.CF_CHI2_P_LE_1 == pytest.approx(0.3935, abs=1e-4)
    assert rpt.CF_CHI2_P_LE_4 == pytest.approx(0.8647, abs=1e-4)
    assert rpt.CF_CHI2_P95 == pytest.approx(5.991, abs=1e-3)


def _a(track, ts, kind, dist="", score="", nis=""):
    return {"frame_ts": str(ts), "track_id": track, "kind": kind, "dist_km": str(dist),
            "score": str(score), "nis": str(nis)}


def test_summarize_association_and_tracks():
    rows = [_a("A", 0, "new"), _a("B", 0, "new"),
            _a("A", 300, "two_point", 5.0, 0.1),
            _a("A", 600, "continue", 1.0, 0.05, 1.0),
            _a("A", 2700, "continue", 3.0, 0.1, 7.0),
            _a("C", 2700, "new", 15.0, 0.7), _a("D", 2700, "split", 30.0, 0.8)]

    assoc = rpt.summarize_association(rows)

    assert assoc["kinds"] == {"new": 3, "two_point": 1, "continue": 2, "split": 1}
    assert assoc["dist_km"]["p50"] == pytest.approx(2.0)
    assert (assoc["n_nis"], assoc["nis_mean"], assoc["nis_frac_above_p95"]) == (2, 4.0, 0.5)
    assert (assoc["new_with_candidate"], assoc["new_near"]) == (2, 1)

    tracks = rpt.summarize_tracks(rows)
    assert tracks["n_tracks"] == 4
    assert tracks["duration_min"]["p50"] == 0.0             # B, C e D: um frame só
    assert tracks["frac_verifiable"] == 0.25


def test_summarize_frames():
    row = {"latency_s": "60", "n_window_events": "100", "n_clustered": "80", "n_cells": "2",
           "n_new": "1", "n_split": "0", "n_missed": "1", "n_merged": "0", "n_expired": "2"}

    frames = rpt.summarize_frames([row, dict(row, latency_s="120", n_clustered="60")])

    assert frames["n_frames"] == 2
    assert frames["frac_clustered"] == pytest.approx(0.7)
    assert frames["latency_s"]["p50"] == pytest.approx(90.0)
    assert (frames["n_new"], frames["n_missed"], frames["n_expired"]) == (2, 2, 4)


def test_build_and_format_on_empty_dir(tmp_path):
    text = rpt.format_report(rpt.build(str(tmp_path)), ["constantes: padrão"])

    assert text.startswith("constantes: padrão\n")
    assert "(sem linhas)" in text and "frames" in text


def test_build_reads_csvs_written_by_the_pipeline(tmp_path):
    row = {"verify_ts": 0, "track_id": "A", "issue_ts": 0, "lead_min": 10, "fcst_lat": 0,
           "fcst_lng": 0, "obs_lat": 0, "obs_lng": 0, "error_km": 1.0, "persist_km": 4.0,
           "std_lat_km": 1, "std_lng_km": 1, "z2": 2.0, "status": "ok"}
    vrf.append_rows(str(tmp_path / rpt.CS_VERIFY_NAME), [row])
    dgn.append(str(tmp_path / rpt.CS_ASSOC_NAME), dgn.CLST_ASSOC_COLUMNS,
               [{"frame_ts": 0, "track_id": "A", "kind": "new", "size": 10}])

    summary = rpt.build(str(tmp_path))
    text = rpt.format_report(summary)

    assert summary["verification"][0]["skill"] == pytest.approx(0.75)
    assert summary["association"]["kinds"] == {"new": 1}
    assert "  10       1   100%" in text
