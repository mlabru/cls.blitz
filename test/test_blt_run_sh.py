# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

test_blt_run_sh
testes do blt_run.sh (mesmo padrão do test_nwc_interpol_sh.py do 27.nowcast):
- funções carregadas com SLM_RUN_NO_MAIN=true, sem executar o main;
- main rodado numa cópia do script em tmp_path, com os programas Python e o
  ffmpeg falsos (nunca toca no data/ real nem na REDEMET): ordem dos passos,
  argumentos, estado gravado, lock de instância e destino dos logs.

2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >──────────────────────────────────────────────────────────────────────────

# python library
import os
import pathlib
import re
import shutil
import subprocess

# pytest
import pytest

# < constants >────────────────────────────────────────────────────────────────────────

# raiz do projeto (onde está blt_run.sh)
MPTH_ROOT = pathlib.Path(__file__).resolve().parent.parent

# nome do script
CS_SCRIPT = "blt_run.sh"

# linha de log(): "[AAAA-MM-DD HH:MM:SS] <mensagem>"
CS_LOG_LINE = r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] "

# programa falso: registra a chamada (nome|cwd|argv), cria o arquivo de
# saída que o shell espera e sai com o rc pedido
CS_FAKE_APP = """\
import os, pathlib, sys
lpth = pathlib.Path(__file__).resolve().parent.parent / "calls.txt"
with lpth.open("a") as lfh:
    lfh.write("{name}|" + os.getcwd() + "|" + " ".join(sys.argv[1:]) + "\\n")
largs = sys.argv[1:]
if 0 == {rc}:
    if "{name}".endswith("Downloader"):
        ldir = largs[largs.index("-s") + 1]
        lfile = largs[largs.index("-f") + 1]
        with open(os.path.join(ldir, lfile), "a") as lfh:
            lfh.write("timestamp,lat,lng\\n")
    elif "bltMapper" == "{name}":
        pathlib.Path(largs[largs.index("-f") + 1]).write_text("png")
    elif "bltTracker" == "{name}":
        pathlib.Path(largs[-1]).write_text("[]")
        print("NEWEST_TS=1234.0")
    elif "bltRender" == "{name}":
        pathlib.Path(largs[-1]).write_text("png")
sys.exit({rc})
"""

# ffmpeg falso: cria o arquivo de saída (último argumento)
CS_FAKE_FFMPEG = """#!/usr/bin/env bash
echo "ffmpeg|${@: -1}" >> "$(dirname "$0")/../calls.txt"
touch "${@: -1}"
"""

# nomes dos programas Python chamados pelo blt_run.sh
CLST_APPS = ["slmDownloader", "bltMapper", "bltTracker", "bltRender"]

# os mesmos com a fonte GLM (SLM_SOURCE=glm)
CLST_APPS_GLM = ["glmDownloader", "bltMapper", "bltTracker", "bltRender"]

pytestmark = pytest.mark.skipif(shutil.which("bash") is None or shutil.which("flock") is None,
                                reason="bash/flock indisponíveis")

# ─────────────────────────────────────────────────────────────────────────────────────
def _env(fpth_home: pathlib.Path, fpth_proj: pathlib.Path | None = None,
         fs_source: str | None = None) -> dict:
    """
    ambiente dos subprocessos: HOME isolado (sem ~/.venv), sem PYTHONPATH,
    SLM_* ou MET_* herdados; com o ffmpeg falso no PATH e SLM_PATH no projeto
    """
    ldct_env = {ls_key: ls_value for ls_key, ls_value in os.environ.items()
                if "PYTHONPATH" != ls_key and not ls_key.startswith(("MET_", "SLM_"))}
    ldct_env["HOME"] = str(fpth_home)

    if fpth_proj is not None:
        ldct_env["SLM_PATH"] = str(fpth_proj)
        ldct_env["PATH"] = str(fpth_proj / "bin") + os.pathsep + ldct_env.get("PATH", "")

    if fs_source is not None:
        ldct_env["SLM_SOURCE"] = fs_source

    return ldct_env

# ─────────────────────────────────────────────────────────────────────────────────────
def _project(fpth_tmp: pathlib.Path, fdct_rc: dict | None = None) -> pathlib.Path:
    """
    cópia do blt_run.sh num projeto falso, com os programas e o ffmpeg substituídos

    :param fdct_rc: rc de cada programa (nome -> rc); padrão 0
    """
    ldct_rc = fdct_rc or {}
    lpth_proj = fpth_tmp / "proj"
    (lpth_proj / "apps").mkdir(parents=True)
    (lpth_proj / "bin").mkdir()

    for ls_file in (CS_SCRIPT, "blt_cfg.sh", "blt_common.sh", "cls_common.sh"):
        shutil.copy(MPTH_ROOT / ls_file, lpth_proj / ls_file)

    for ls_name in CLST_APPS + CLST_APPS_GLM[:1]:
        ls_code = CS_FAKE_APP.format(name=ls_name, rc=ldct_rc.get(ls_name, 0))
        (lpth_proj / "apps" / f"{ls_name}.py").write_text(ls_code)

    lpth_ffmpeg = lpth_proj / "bin" / "ffmpeg"
    lpth_ffmpeg.write_text(CS_FAKE_FFMPEG)
    lpth_ffmpeg.chmod(0o755)

    return lpth_proj

# ─────────────────────────────────────────────────────────────────────────────────────
def _run(fpth_proj: pathlib.Path, flst_prefix: tuple = (),
         fs_source: str | None = None) -> subprocess.CompletedProcess:
    """
    roda o main do script copiado (de outro diretório, para provar o cd)
    """
    return subprocess.run([*flst_prefix, "bash", str(fpth_proj / CS_SCRIPT)],
                          capture_output=True, text=True,
                          env=_env(fpth_proj.parent, fpth_proj, fs_source), cwd=fpth_proj.parent)

# ─────────────────────────────────────────────────────────────────────────────────────
def _source(fs_body: str, fpth_proj: pathlib.Path) -> subprocess.CompletedProcess:
    """
    carrega o script com SLM_RUN_NO_MAIN=true e roda fs_body no mesmo shell
    """
    ls_command = f'SLM_RUN_NO_MAIN=true; source "{fpth_proj / CS_SCRIPT}"; {fs_body}'

    return subprocess.run(["bash", "-c", ls_command], capture_output=True, text=True,
                          env=_env(fpth_proj.parent, fpth_proj), cwd=fpth_proj.parent)

# ─────────────────────────────────────────────────────────────────────────────────────
def _calls(fpth_proj: pathlib.Path) -> list:
    """
    [(programa, cwd, argumentos)] na ordem em que foram chamados
    """
    lpth_calls = fpth_proj / "calls.txt"

    if not lpth_calls.exists():
        return []

    return [tuple(ls_line.split("|")) for ls_line in lpth_calls.read_text().splitlines()]

# ─────────────────────────────────────────────────────────────────────────────────────
def _names(fpth_proj: pathlib.Path) -> list:
    """
    só os nomes dos programas chamados, em ordem
    """
    return [lt_call[0] for lt_call in _calls(fpth_proj)]

# < SLM_RUN_NO_MAIN >──────────────────────────────────────────────────────────────────

class TestNoMain:

    def test_carrega_funcoes_sem_rodar_o_main(self, tmp_path):
        lpth_proj = _project(tmp_path)

        lproc = _source("type -t log main generate_map track_new_frame render_frame", lpth_proj)

        assert 0 == lproc.returncode, lproc.stderr
        assert ["function"] * 5 == lproc.stdout.split()
        # o main não rodou: nada criado, nenhum programa chamado
        assert not (lpth_proj / "logs").exists()
        assert not (lpth_proj / "data").exists()
        assert [] == _calls(lpth_proj)

    def test_caminhos_relativos_a_slm_path(self, tmp_path):
        lpth_proj = _project(tmp_path)

        lproc = _source('echo "$SLM_LOG_APP|$SLM_RUN_LOCK|$SLM_RAW_CSV"', lpth_proj)

        assert 0 == lproc.returncode, lproc.stderr
        assert [str(lpth_proj / "logs" / "slm_app.log"),
                str(lpth_proj / "data" / "state" / ".slm_run.lock"),
                str(lpth_proj / "data" / "raw" / "slm_lightning.csv")] == \
               lproc.stdout.strip().split("|")

# < log >──────────────────────────────────────────────────────────────────────────────

class TestLog:

    def test_grava_no_log_e_em_stderr(self, tmp_path):
        lpth_proj = _project(tmp_path)

        lproc = _source(f'SLM_LOG_APP="{tmp_path}/a.log"; SLM_LOG_ERR="{tmp_path}/e.log"; '
                        'log "[info]: um"', lpth_proj)

        assert 0 == lproc.returncode, lproc.stderr
        assert "" == lproc.stdout
        ls_line = (tmp_path / "a.log").read_text()
        assert re.match(CS_LOG_LINE + r"\[info\]: um$", ls_line)
        assert ls_line == lproc.stderr

# < funções auxiliares >───────────────────────────────────────────────────────────────

class TestHelpers:

    def test_get_state_sem_arquivo_devolve_zero(self, tmp_path):
        lproc = _source(f'get_state "{tmp_path}/nao_existe"', _project(tmp_path))

        assert "0" == lproc.stdout.strip()

    def test_get_state_arquivo_vazio_devolve_zero(self, tmp_path):
        (tmp_path / "vazio").write_text("")

        lproc = _source(f'get_state "{tmp_path}/vazio"', _project(tmp_path))

        assert "0" == lproc.stdout.strip()

    def test_extract_newest_ts(self, tmp_path):
        lproc = _source('extract_newest_ts "$(printf "x\\nNEWEST_TS=99.5\\n")"; '
                        'echo "[$(extract_newest_ts "nada")]"', _project(tmp_path))

        assert 0 == lproc.returncode, lproc.stderr
        assert ["99.5", "[]"] == lproc.stdout.split()

    def test_remove_oldest_files_mantem_os_mais_recentes(self, tmp_path):
        lpth_dir = tmp_path / "frames"
        lpth_dir.mkdir()
        for li_index in range(5):
            (lpth_dir / f"frame_{li_index:04d}.png").write_text("")

        lproc = _source(f'GI_MAX_FRAMES=2; remove_oldest_files "{lpth_dir}" "frame_*.png"',
                        _project(tmp_path))

        assert 0 == lproc.returncode, lproc.stderr
        assert ["frame_0003.png", "frame_0004.png"] == sorted(ls.name for ls in lpth_dir.iterdir())

    def test_remove_oldest_files_ordem_numerica_apos_9999(self, tmp_path):
        # [fix] com sort alfabético frame_10000 vinha antes de frame_9999
        lpth_dir = tmp_path / "frames"
        lpth_dir.mkdir()
        for li_index in (9998, 9999, 10000, 10001):
            (lpth_dir / f"frame_{li_index:04d}.png").write_text("")

        lproc = _source(f'GI_MAX_FRAMES=2; remove_oldest_files "{lpth_dir}" "frame_*.png"',
                        _project(tmp_path))

        assert 0 == lproc.returncode, lproc.stderr
        assert ["frame_10000.png", "frame_10001.png"] == sorted(ls.name for ls in lpth_dir.iterdir())

# < main >─────────────────────────────────────────────────────────────────────────────

class TestMain:

    def test_pipeline_completo(self, tmp_path):
        lpth_proj = _project(tmp_path)

        lproc = _run(lpth_proj)

        assert 0 == lproc.returncode, lproc.stderr
        # os quatro programas, em ordem, depois o ffmpeg (paleta, GIF, MP4)
        assert CLST_APPS + ["ffmpeg"] * 3 == _names(lpth_proj)
        # cwd na raiz de execução (SLM_PATH)
        assert all(pathlib.Path(lt[1]) == lpth_proj for lt in _calls(lpth_proj) if "ffmpeg" != lt[0])

        # argumentos de cada programa (contrato com os apps/*.py)
        ldct_args = {lt[0]: lt[2] for lt in _calls(lpth_proj) if "ffmpeg" != lt[0]}
        lpth_data = lpth_proj / "data"
        assert (f"-f slm_lightning.csv -s {lpth_data}/raw "
                f"-a {lpth_data}/archive") == ldct_args["slmDownloader"]
        assert ldct_args["bltMapper"].startswith(f"-s {lpth_data}/output -f {lpth_data}/temp/slm/tmp.")
        assert (f"--since-ts 0 --source stsc --tracks-file {lpth_data}/state/slm_tracks_state.json "
                f"--verify-file {lpth_data}/state/slm_verification.csv "
                f"--assoc-file {lpth_data}/state/slm_assoc.csv "
                f"--frames-file {lpth_data}/state/slm_frames.csv "
                f"{lpth_data}/raw/slm_lightning.csv "
                f"{lpth_data}/temp/slm/json/frame_0000.json") == ldct_args["bltTracker"]
        assert (f"--source stsc {lpth_data}/temp/slm/json/frame_0000.json "
                f"{lpth_data}/temp/slm/png/frame_0000.png") == ldct_args["bltRender"]

        # estado avançado e produtos publicados
        assert "1234.0" == (lpth_data / "state" / ".slm_last_ts").read_text().strip()
        assert "1" == (lpth_data / "state" / ".slm_frame_counter").read_text().strip()
        for ls_product in ("slm_now.png", "slm_prv.png", "slm_out.png", "slm_out.gif", "slm_out.mp4"):
            assert (lpth_data / "output" / ls_product).exists(), ls_product

        ls_log = (lpth_proj / "logs" / "slm_app.log").read_text()
        assert "blt_run: iniciando" in ls_log
        assert "blt_run: execução concluída." in ls_log
        assert "[error]" not in ls_log and "[warn]" not in ls_log

    def test_fonte_glm_usa_o_downloader_e_os_arquivos_glm(self, tmp_path):
        lpth_proj = _project(tmp_path)

        lproc = _run(lpth_proj, fs_source="glm")

        assert 0 == lproc.returncode, lproc.stderr
        assert CLST_APPS_GLM + ["ffmpeg"] * 3 == _names(lpth_proj)
        ldct_args = {lt[0]: lt[2] for lt in _calls(lpth_proj) if "ffmpeg" != lt[0]}
        lpth_data = lpth_proj / "data"
        assert (f"-f glm_lightning.csv -s {lpth_data}/raw "
                f"-a {lpth_data}/archive") == ldct_args["glmDownloader"]
        assert (f"--since-ts 0 --source glm --tracks-file {lpth_data}/state/glm_tracks_state.json "
                f"--verify-file {lpth_data}/state/glm_verification.csv "
                f"--assoc-file {lpth_data}/state/glm_assoc.csv "
                f"--frames-file {lpth_data}/state/glm_frames.csv "
                f"{lpth_data}/raw/glm_lightning.csv "
                f"{lpth_data}/temp/glm/json/frame_0000.json") == ldct_args["bltTracker"]
        assert ldct_args["bltRender"].startswith(f"--source glm {lpth_data}/temp/glm/json/")
        # mesmos data/ e logs/, só arquivos glm_: nada do STSC é tocado
        for ls_product in ("glm_now.png", "glm_out.png", "glm_out.gif", "glm_out.mp4"):
            assert (lpth_data / "output" / ls_product).exists(), ls_product
        assert "1" == (lpth_data / "state" / ".glm_frame_counter").read_text().strip()
        for ls_log in ("glm_app.log", "glm_downloader.log", "glm_tracker.log"):
            assert (lpth_proj / "logs" / ls_log).exists(), ls_log
        assert not list((lpth_proj / "data").rglob("slm_*"))
        assert not list((lpth_proj / "logs").glob("slm_*"))

    def test_as_duas_fontes_convivem_no_mesmo_diretorio(self, tmp_path):
        lpth_proj = _project(tmp_path)

        _run(lpth_proj)
        lproc = _run(lpth_proj, fs_source="glm")

        assert 0 == lproc.returncode, lproc.stderr
        lpth_state = lpth_proj / "data" / "state"
        # cada fonte com seu contador de frames e seu since_ts
        assert "1" == (lpth_state / ".slm_frame_counter").read_text().strip()
        assert "1" == (lpth_state / ".glm_frame_counter").read_text().strip()
        assert (lpth_proj / "data" / "output" / "slm_out.png").exists()
        assert (lpth_proj / "data" / "output" / "glm_out.png").exists()

    def test_fonte_invalida_falha_antes_de_tudo(self, tmp_path):
        lpth_proj = _project(tmp_path)

        lproc = _run(lpth_proj, fs_source="xyz")

        assert 0 != lproc.returncode
        assert "SLM_SOURCE inválida" in lproc.stderr
        assert [] == _calls(lpth_proj)

    def test_segunda_rodada_usa_o_estado(self, tmp_path):
        lpth_proj = _project(tmp_path)

        _run(lpth_proj)
        lproc = _run(lpth_proj)

        assert 0 == lproc.returncode, lproc.stderr
        llst_tracker = [lt[2] for lt in _calls(lpth_proj) if "bltTracker" == lt[0]]
        assert llst_tracker[1].startswith("--since-ts 1234.0 ")
        assert llst_tracker[1].endswith("frame_0001.json")

    def test_stdout_dos_programas_vai_para_logs(self, tmp_path):
        lpth_proj = _project(tmp_path)

        lproc = _run(lpth_proj)

        assert 0 == lproc.returncode, lproc.stderr
        # o NEWEST_TS do tracker é lido pelo shell, não vaza para o cron
        assert "NEWEST_TS" not in lproc.stdout + lproc.stderr
        for ls_name in ("downloader", "mapper", "tracker", "render"):
            assert (lpth_proj / "logs" / f"slm_{ls_name}.log").exists(), ls_name

    def test_falha_no_download_nao_impede_o_resto(self, tmp_path):
        lpth_proj = _project(tmp_path)
        # primeira rodada cria o CSV; na segunda o download falha
        _run(lpth_proj)
        (lpth_proj / "apps" / "slmDownloader.py").write_text(
            CS_FAKE_APP.format(name="slmDownloader", rc=1))

        lproc = _run(lpth_proj)

        assert 0 == lproc.returncode, lproc.stderr
        assert "bltTracker" in _names(lpth_proj)[4:]
        assert "download falhou" in (lpth_proj / "logs" / "slm_app.log").read_text()

    def test_sem_csv_pula_mapa_tracker_e_render(self, tmp_path):
        lpth_proj = _project(tmp_path, {"slmDownloader": 1})

        lproc = _run(lpth_proj)

        assert 0 == lproc.returncode, lproc.stderr
        assert ["slmDownloader"] == _names(lpth_proj)
        assert "nenhum CSV" in (lpth_proj / "logs" / "slm_app.log").read_text()

    def test_falha_no_tracker_nao_renderiza(self, tmp_path):
        lpth_proj = _project(tmp_path, {"bltTracker": 2})

        lproc = _run(lpth_proj)

        assert 0 == lproc.returncode, lproc.stderr
        assert ["slmDownloader", "bltMapper", "bltTracker"] == _names(lpth_proj)
        assert "[error]: blt_run: tracker falhou" in (lpth_proj / "logs" / "slm_app.log").read_text()
        # o estado não avança
        assert "" == (lpth_proj / "data" / "state" / ".slm_frame_counter").read_text().strip()

    def test_lock_ocupado_sai_sem_rodar(self, tmp_path):
        lpth_proj = _project(tmp_path)
        lpth_lock = lpth_proj / "data" / "state" / ".slm_run.lock"
        lpth_lock.parent.mkdir(parents=True)

        # outro processo segura o lock durante toda a execução
        lproc = _run(lpth_proj, ("flock", str(lpth_lock)))

        assert 0 == lproc.returncode, lproc.stderr
        assert [] == _calls(lpth_proj)
        assert "execução anterior ainda está em andamento" in \
               (lpth_proj / "logs" / "slm_app.log").read_text()

    def test_ativa_o_venv_se_existir(self, tmp_path):
        lpth_proj = _project(tmp_path)

        # venv falso em $HOME (= tmp_path): marca que foi carregado
        lpth_venv = tmp_path / ".venv" / "bin"
        lpth_venv.mkdir(parents=True)
        (lpth_venv / "activate").write_text(f'touch "{tmp_path}/venv_ativado"\n')

        lproc = _run(lpth_proj)

        assert 0 == lproc.returncode, lproc.stderr
        assert (tmp_path / "venv_ativado").exists()

# < trava MET_ENV=test >───────────────────────────────────────────────────────────────

class TestGenerateMap:
    """
    [fix] o CSV só é marcado como processado depois do mapa (ou de "sem raios", rc 3)
    """

    def _twice(self, fpth_tmp: pathlib.Path, fi_mapper_rc: int) -> tuple:
        lpth_proj = _project(fpth_tmp, {"bltMapper": fi_mapper_rc})
        lpth_csv = fpth_tmp / "raw.csv"
        lpth_csv.write_text("timestamp,lat,lng\n")

        lproc = _source(f'slm_enter_run_dir; generate_map "{lpth_csv}" t; generate_map "{lpth_csv}" t',
                        lpth_proj)

        lpth_state = lpth_proj / "data" / "state" / ".slm_last_processed"
        ls_state = lpth_state.read_text().strip() if lpth_state.exists() else ""
        ls_log = (lpth_proj / "logs" / "slm_app.log").read_text()

        return lproc, lpth_proj, lpth_csv, ls_state, ls_log

    def test_sucesso_marca_e_nao_repete(self, tmp_path):
        lproc, lpth_proj, lpth_csv, ls_state, ls_log = self._twice(tmp_path, 0)

        assert 0 == lproc.returncode, lproc.stderr
        assert ["bltMapper"] == _names(lpth_proj)
        assert str(int(lpth_csv.stat().st_mtime)) == ls_state
        assert (lpth_proj / "data" / "output" / "slm_now.png").exists()

    def test_falha_nao_marca_e_tenta_de_novo(self, tmp_path):
        lproc, lpth_proj, _, ls_state, ls_log = self._twice(tmp_path, 1)

        assert 0 == lproc.returncode, lproc.stderr
        assert ["bltMapper", "bltMapper"] == _names(lpth_proj)
        assert "" == ls_state
        assert "nova tentativa na próxima rodada" in ls_log
        assert not list((lpth_proj / "data" / "temp").glob("tmp.*.png"))

    def test_sem_raios_marca_sem_erro(self, tmp_path):
        lproc, lpth_proj, lpth_csv, ls_state, ls_log = self._twice(tmp_path, 3)

        assert 0 == lproc.returncode, lproc.stderr
        assert ["bltMapper"] == _names(lpth_proj)
        assert str(int(lpth_csv.stat().st_mtime)) == ls_state
        assert "sem raios na região" in ls_log
        assert "[error]" not in ls_log
        assert not (lpth_proj / "data" / "output" / "slm_now.png").exists()

class TestTestEnvGuard:

    def test_recusa_slm_path_fora_da_raiz_de_teste(self, tmp_path):
        lpth_proj = _project(tmp_path)

        lproc = subprocess.run(["bash", str(lpth_proj / CS_SCRIPT)], capture_output=True,
                               text=True, cwd=tmp_path,
                               env={**_env(tmp_path, lpth_proj), "MET_ENV": "test",
                                    "MET_TST_ROOT": str(tmp_path / "tst")})

        assert 1 == lproc.returncode
        assert "fora de" in lproc.stderr
        assert [] == _calls(lpth_proj)

# < the end >──────────────────────────────────────────────────────────────────────────
