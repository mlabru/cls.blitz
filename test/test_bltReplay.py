import ast
import csv
import random

import pytest

import bltReplay as rp
import work.blt_kalman as klm
import work.blt_report as rpt
import work.blt_track as trk


def _moving_cell_csv(path, minutes=120, seed=3):
    """uma célula indo para leste a ~40 km/h, alguns raios por minuto, sem fetched_ts"""
    rnd = random.Random(seed)
    lines = ["timestamp,lat,lng"]
    for m in range(minutes):
        ts = 1_000_000 + 60 * m
        lng = -45.0 + (40.0 / 104.0) * (m / 60.0)
        for _ in range(rnd.randint(3, 8)):
            lines.append(f"{ts},{-20.0 + rnd.gauss(0, 0.03)},{lng + rnd.gauss(0, 0.03)}")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# ─────────────────────────────────────────────────────────────────────────────────────
# as constantes ajustáveis são lidas na chamada

@pytest.mark.parametrize("module_name", sorted(rp.CDCT_TUNABLE_MODULES))
def test_no_function_default_freezes_a_tunable_constant(module_name):
    # [fix] um padrão "ff_x=CF_Y" é avaliado UMA vez, na definição: o --set do
    # replay trocaria a constante do módulo e a função seguiria com o valor velho
    path = rp.CDCT_TUNABLE_MODULES[module_name].__file__
    tree = ast.parse(open(path, encoding="utf-8").read())
    frozen = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for default in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
            names = {n.id for n in ast.walk(default) if isinstance(n, ast.Name)}
            frozen += [f"{node.name}: {n}" for n in names if rp.CRE_TUNABLE_NAME.match(n)]

    assert frozen == []


# ─────────────────────────────────────────────────────────────────────────────────────
# --set

def test_apply_overrides_sets_and_coerces(monkeypatch):
    for module, name in ((trk, "CF_MAX_DIST_KM"), (trk, "CI_MAX_MISSED_FRAMES"),
                         (rp.mtn, "CT_FORECAST_LEADS_MIN")):
        monkeypatch.setattr(module, name, getattr(module, name))     # desfaz no fim

    lines = rp.apply_overrides(["blt_track.CF_MAX_DIST_KM=30",
                                "blt_track.CI_MAX_MISSED_FRAMES = 5",
                                "blt_motion.CT_FORECAST_LEADS_MIN=[10, 60]"])

    assert trk.CF_MAX_DIST_KM == 30.0 and isinstance(trk.CF_MAX_DIST_KM, float)
    assert trk.CI_MAX_MISSED_FRAMES == 5
    assert rp.mtn.CT_FORECAST_LEADS_MIN == (10, 60)
    assert lines[0] == "blt_track.CF_MAX_DIST_KM = 30.0 (era 100.0)"


@pytest.mark.parametrize("text", [
    "blt_track.CF_MAX_DIST_KM",               # sem "="
    "CF_MAX_DIST_KM=3",                       # sem módulo
    "bltRender.CF_MIN_ARROW_SPEED_KMH=1",     # módulo fora da lista
    "blt_track.CF_NOPE=1",                    # constante inexistente
    "blt_track.M_LOG=1",                      # não é constante de algoritmo
    "blt_track.CF_MAX_DIST_KM=abc",           # valor inválido
    "blt_track.CI_MAX_MISSED_FRAMES=2.5",     # float numa constante int
    "blt_track.CF_MAX_DIST_KM=True",          # booleano
    "blt_motion.CT_FORECAST_LEADS_MIN=10",    # escalar numa tupla
])
def test_apply_overrides_rejects(text):
    with pytest.raises(ValueError):
        rp.apply_overrides([text])


def test_apply_overrides_all_or_nothing():
    before = trk.CF_MAX_DIST_KM

    with pytest.raises(ValueError):
        rp.apply_overrides(["blt_track.CF_MAX_DIST_KM=30", "blt_track.CF_NOPE=1"])

    assert trk.CF_MAX_DIST_KM == before


# ─────────────────────────────────────────────────────────────────────────────────────
# calendário das rodadas

def test_schedule_with_fetched_ts_uses_real_runs():
    rows = [(1600.0, 1200.0, 0.0, 0.0), (1300.0, 1200.0, 1.0, 1.0), (1300.0, 1260.0, 2.0, 2.0)]

    available, runs = rp.schedule(rows, 300.0)

    assert runs == [1300.0, 1600.0]
    assert [r[0] for r in available] == [1300.0, 1300.0, 1600.0]


def test_schedule_without_fetched_ts_steps_from_first_strike():
    rows = [(None, 1000.0, 0.0, 0.0), (None, 1700.0, 0.0, 0.0)]

    available, runs = rp.schedule(rows, 300.0)

    assert runs == [1300.0, 1600.0, 1900.0]
    assert [r[0] for r in available] == [1000.0, 1700.0]


def test_schedule_empty():
    assert rp.schedule([], 300.0) == ([], [])


# ─────────────────────────────────────────────────────────────────────────────────────
# reprocessamento

def test_replay_respects_availability_and_latency(tmp_path):
    n = 10
    rows = ([(1300.0, 1200.0, 0.0, 0.0)] * n +                  # rodada 1300
            [(1600.0, 1200.0, 0.001, 0.0)] * 5 +                # atrasados do minuto 1200
            [(1600.0, 1500.0, 0.0, 0.0)] * n)                   # rodada 1600

    runs, frames = rp.replay(rows, str(tmp_path))

    assert (runs, frames) == (2, 2)
    data = _read(tmp_path / rpt.CS_FRAMES_NAME)
    assert [int(r["n_window_events"]) for r in data] == [n, 2 * n + 5]
    assert [float(r["latency_s"]) for r in data] == [100.0, 100.0]


def test_replay_moving_cell_beats_persistence(tmp_path):
    rows = rp.arc.load([_moving_cell_csv(tmp_path / "raw.csv")])

    runs, frames = rp.replay(rows, str(tmp_path / "out"))

    assert runs == frames == 24
    leads = rpt.summarize_verification(_read(tmp_path / "out" / rpt.CS_VERIFY_NAME))
    assert [lead["lead_min"] for lead in leads] == [10, 20, 30]
    assert all(lead["n_ok"] > 0 and lead["skill"] > 0.5 for lead in leads)
    assert len({r["track_id"] for r in _read(tmp_path / "out" / rpt.CS_ASSOC_NAME)}) == 1


def test_replay_clears_previous_outputs(tmp_path):
    rows = rp.arc.load([_moving_cell_csv(tmp_path / "raw.csv", minutes=40)])

    def without_ids():
        # o ID de track é um UUID novo a cada execução
        return [{k: v for k, v in r.items() if "track_id" != k}
                for r in _read(tmp_path / "out" / rpt.CS_VERIFY_NAME)]

    rp.replay(rows, str(tmp_path / "out"))
    first = without_ids()
    rp.replay(rows, str(tmp_path / "out"))

    assert first and without_ids() == first


# ─────────────────────────────────────────────────────────────────────────────────────
# programa

def test_main_writes_report_with_overrides(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(trk, "CF_MAX_DIST_KM", trk.CF_MAX_DIST_KM)
    raw = _moving_cell_csv(tmp_path / "raw.csv", minutes=60)
    out = tmp_path / "out"

    rc = rp.main(["-o", str(out), "--set", "blt_track.CF_MAX_DIST_KM=0.001", raw])

    assert rc == rp.CI_RC_OK
    report = (out / rp.CS_REPORT_NAME).read_text()
    assert report == capsys.readouterr().out
    assert report.startswith("replay: ")
    assert "blt_track.CF_MAX_DIST_KM = 0.001 (era 100.0)" in report
    # com o raio de associação de 1 m nenhuma célula continua um track
    kinds = {r["kind"] for r in _read(out / rpt.CS_ASSOC_NAME)}
    assert "continue" not in kinds and "two_point" not in kinds


def test_main_kalman_override_changes_cone(tmp_path, monkeypatch):
    monkeypatch.setattr(klm, "CF_KALMAN_R_STD_KM", klm.CF_KALMAN_R_STD_KM)
    raw = _moving_cell_csv(tmp_path / "raw.csv", minutes=60)

    rp.main(["-o", str(tmp_path / "a"), raw])
    rp.main(["-o", str(tmp_path / "b"), "--set", "blt_kalman.CF_KALMAN_R_STD_KM=11.1", raw])

    # a última linha: filtro já convergido
    std_a = float(_read(tmp_path / "a" / rpt.CS_VERIFY_NAME)[-1]["std_lat_km"])
    std_b = float(_read(tmp_path / "b" / rpt.CS_VERIFY_NAME)[-1]["std_lat_km"])
    assert std_b > 1.5 * std_a


def test_main_report_only_reads_existing_csvs(tmp_path, capsys):
    rp.replay(rp.arc.load([_moving_cell_csv(tmp_path / "raw.csv", minutes=60)]),
              str(tmp_path))
    capsys.readouterr()

    assert rp.main(["--report-only", "-o", str(tmp_path)]) == rp.CI_RC_OK

    out = capsys.readouterr().out
    assert out.startswith("constantes: padrão")
    assert "verificação" in out
    assert not (tmp_path / rp.CS_REPORT_NAME).exists()


def test_main_errors(tmp_path):
    with pytest.raises(SystemExit) as exc:
        rp.main(["-o", str(tmp_path)])                         # sem arquivos
    assert exc.value.code == rp.CI_RC_BAD_ARGS

    with pytest.raises(SystemExit):
        rp.main(["-o", str(tmp_path), "--step-min", "0", "x.csv"])

    raw = _moving_cell_csv(tmp_path / "raw.csv", minutes=5)
    assert rp.main(["-o", str(tmp_path), "--set", "blt_track.CF_NOPE=1", raw]) == rp.CI_RC_BAD_ARGS

    empty = tmp_path / "empty.csv"
    empty.write_text("timestamp,lat,lng\n")
    assert rp.main(["-o", str(tmp_path), str(empty)]) == rp.CI_RC_NO_DATA


def test_report_only_reads_the_csvs_of_the_chosen_source(tmp_path, capsys):
    # GLM e STSC dividem o diretório: o resumo do GLM lê os glm_*
    head = ("frame_ts,run_ts,latency_s,n_window_events,n_clustered,n_cells,"
            "n_new,n_split,n_missed,n_merged,n_expired\n")
    (tmp_path / "glm_frames.csv").write_text(head + "1000,1060,60,100,90,7,7,0,0,0,0\n")
    (tmp_path / "slm_frames.csv").write_text(head + "1000,1900,900,50,20,3,3,0,0,0,0\n")

    assert rp.main(["--report-only", "--source", "glm", "-o", str(tmp_path)]) == rp.CI_RC_OK
    glm_report = capsys.readouterr().out
    assert rp.main(["--report-only", "-o", str(tmp_path)]) == rp.CI_RC_OK
    slm_report = capsys.readouterr().out

    assert glm_report != slm_report
