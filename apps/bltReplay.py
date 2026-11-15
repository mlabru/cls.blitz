# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

Simple Lightning Model
bltReplay
reprocessa raios arquivados (blt_archive) com o tracker de verdade
(bltTracker.process), rodada a rodada, opcionalmente com constantes
trocadas, e resume verificação e diagnósticos (blt_report).  É a bancada
de calibração: mesmos raios, constantes diferentes, números comparáveis.

fluxo (replay):
  1) lê os arquivos (slm_raw_AAAAMMDD.csv.gz; também aceita CSV simples);
  2) troca as constantes pedidas em --set (apply_overrides);
  3) monta o calendário de rodadas: os instantes fetched_ts do archive (as
     rodadas reais do cron), ou, sem fetched_ts, uma rodada a cada
     --step-min contada dos próprios raios;
  4) a cada rodada, o CSV de entrada tem os raios já disponíveis naquele
     instante e da última hora (como o do downloader), e o tracker roda com
     o since_ts devolvido pela rodada anterior e o instante simulado
     (latência); o estado e os CSVs ficam em --out-dir;
  5) imprime o relatório e o grava em <out-dir>/report.txt.

exemplos:

    # constantes atuais sobre dois dias
    python3 apps/bltReplay.py -o /tmp/rp/base data/archive/slm_raw_2026110[67].csv.gz

    # grade de um parâmetro
    for q in 2.0 3.5 6.0; do
        python3 apps/bltReplay.py -o /tmp/rp/qvel_$q --set blt_kalman.CF_KALMAN_Q_VEL_STD_KMH=$q \\
                data/archive/slm_raw_20261106.csv.gz
    done

    # só o relatório, sobre os CSVs da operação
    python3 apps/bltReplay.py --report-only -o $SLM_PATH/data/state

--set aceita MÓDULO.CONSTANTE=VALOR para as constantes CF_/CI_/CT_ de
bltTracker, blt_track, blt_kalman, blt_motion e blt_verify (o DBSCAN é
bltTracker.CF_EPS_KM / CI_MIN_PTS).  Os
módulos leem esses valores na chamada (test_bltReplay garante), então a
troca vale para tudo o que o tracker faz.  Atenção: os ruídos do Kalman são
gravados em cada filtro; num estado já existente valem os gravados -- o
replay sempre começa do zero, então não é afetado.

códigos de retorno: 0 ok, 1 nenhum raio nos arquivos, 2 argumentos inválidos
(inclusive --set).

2026.nov  mlabru  [fix] saídas e --report-only com o prefixo da fonte
                  (slm_*/glm_*, _source_tag): o resumo do GLM lia os CSVs do STSC
2026.nov  mlabru  --source (stsc/glm), repassado ao tracker (janela da fonte)
2026.nov  mlabru  initial version
"""
# < imports >──────────────────────────────────────────────────────────────────────────

# python library
from __future__ import annotations

import argparse
import ast
import bisect
import logging
import os
import re
import sys

# locals
import bltTracker as tracker
import work.blt_archive as arc
import work.blt_kalman as klm
import work.glm_ljump as ljp
import work.blt_motion as mtn
import work.blt_report as rpt
import work.blt_track as trk
import work.blt_verify as vrf

# < constants >────────────────────────────────────────────────────────────────────────

# módulos cujas constantes --set pode trocar (nome usado no --set: módulo)
CDCT_TUNABLE_MODULES = {
    "bltTracker":  tracker,
    "blt_track":   trk,
    "blt_kalman":  klm,
    "blt_motion":  mtn,
    "blt_verify":  vrf,
    "glm_ljump":   ljp,
}

# só constantes numéricas de algoritmo (CF_ float, CI_ int, CT_ tupla)
CRE_TUNABLE_NAME = re.compile(r"^C[FIT]_[A-Z0-9_]+$")

# retenção do CSV de entrada (s): a mesma do downloader (CI_MAX_HOURS = 1)
CI_RETENTION_S = 3600

# intervalo entre rodadas (min) quando os raios não têm fetched_ts
CF_DEFAULT_STEP_MIN = 5.0

# arquivos dentro de --out-dir (os CSVs com os nomes que o blt_report procura)
CS_WORK_SUBDIR = "work"
CS_RAW_NAME = "raw.csv"
CS_FRAME_NAME = "frame.json"
CS_TRACKS_SUFFIX = "_tracks_state.json"
CS_REPORT_NAME = "report.txt"

# fonte (--source) -> prefixo dos arquivos (o SLM_TAG do blt_cfg.sh)
CDCT_SOURCE_TAG = {"stsc": "slm", "glm": "glm"}

# códigos de retorno
CI_RC_OK = 0
CI_RC_NO_DATA = 1
CI_RC_BAD_ARGS = 2

# < logging >──────────────────────────────────────────────────────────────────────────

M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# ═════════════════════════════════════════════════════════════════════════════════════
# constantes trocadas
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def apply_overrides(flst_texts: list) -> list:
    """
    troca constantes dos módulos do tracker (setattr no módulo).

    :param flst_texts: textos "MÓDULO.CONSTANTE=VALOR" (VALOR em sintaxe Python)
    :returns: uma linha legível por troca ("blt_track.CF_MAX_DIST_KM = 30.0 (era 100.0)")
    :raises ValueError: formato, módulo, constante ou tipo inválido (nada é
                        trocado se alguma falhar)
    """
    llst_parsed = [_parse_override(ls_text) for ls_text in flst_texts]
    llst_lines = []

    for l_module, ls_module_name, ls_name, l_value in llst_parsed:
        llst_lines.append("%s.%s = %r (era %r)" % (ls_module_name, ls_name, l_value,
                                                   getattr(l_module, ls_name)))
        setattr(l_module, ls_name, l_value)

    return llst_lines

# ─────────────────────────────────────────────────────────────────────────────────────
def _parse_override(fs_text: str) -> tuple:
    """
    valida um "MÓDULO.CONSTANTE=VALOR".

    :param fs_text: texto do --set
    :returns: (módulo, nome do módulo, nome da constante, valor já convertido)
    :raises ValueError: ver apply_overrides
    """
    ls_target, ls_sep, ls_value = fs_text.partition("=")
    ls_module_name, ls_dot, ls_name = ls_target.strip().partition(".")

    if not ls_sep or not ls_dot:
        raise ValueError("esperado MÓDULO.CONSTANTE=VALOR: %r" % fs_text)

    l_module = CDCT_TUNABLE_MODULES.get(ls_module_name)

    if l_module is None:
        raise ValueError("módulo %r não é ajustável (use um de: %s)"
                         % (ls_module_name, ", ".join(sorted(CDCT_TUNABLE_MODULES))))

    if not CRE_TUNABLE_NAME.match(ls_name) or not hasattr(l_module, ls_name):
        raise ValueError("%s não tem a constante ajustável %r" % (ls_module_name, ls_name))

    try:
        l_value = ast.literal_eval(ls_value.strip())

    except (ValueError, SyntaxError) as l_error:
        raise ValueError("valor inválido em %r: %s" % (fs_text, l_error)) from l_error

    return (l_module, ls_module_name, ls_name,
            _coerce(l_value, getattr(l_module, ls_name), fs_text))

# ─────────────────────────────────────────────────────────────────────────────────────
def _coerce(f_value: object, f_current: object, fs_text: str) -> object:
    """
    converte o valor novo para o tipo da constante atual.

    :param f_value: valor lido
    :param f_current: valor atual da constante
    :param fs_text: texto do --set (para a mensagem de erro)
    :returns: valor do mesmo tipo (int -> float é aceito para CF_)
    :raises ValueError: tipo incompatível
    """
    if isinstance(f_current, bool) or isinstance(f_value, bool):
        raise ValueError("constante booleana não é ajustável: %r" % fs_text)

    if isinstance(f_current, float) and isinstance(f_value, (int, float)):
        return float(f_value)

    if isinstance(f_current, int) and isinstance(f_value, int):
        return f_value

    if isinstance(f_current, tuple) and isinstance(f_value, (tuple, list)):
        return tuple(f_value)

    raise ValueError("tipo incompatível em %r: a constante é %s"
                     % (fs_text, type(f_current).__name__))

# ═════════════════════════════════════════════════════════════════════════════════════
# reprocessamento
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def schedule(flst_rows: list, ff_step_s: float) -> tuple:
    """
    quando cada raio fica disponível e quando roda cada rodada.

    com fetched_ts (archive): o raio fica disponível na rodada que o trouxe,
    e as rodadas são esses instantes.  Sem fetched_ts (CSV simples, dados
    sintéticos): disponível no próprio ts, rodadas a cada ff_step_s a partir
    do primeiro raio.

    :param flst_rows: tuplas (fetched_ts ou None, ts, lat, lng) (blt_archive.load)
    :param ff_step_s: intervalo entre rodadas sem fetched_ts (s)
    :returns: (raios como tuplas (disponível_em, ts, lat, lng) em ordem de
               disponibilidade, instantes das rodadas em ordem)
    """
    llst_available = sorted((ft if ft is not None else ts, ts, lat, lng)
                            for ft, ts, lat, lng in flst_rows)

    if not llst_available:
        return [], []

    if all(lt_row[0] is not None for lt_row in flst_rows):
        return llst_available, sorted({lt_row[0] for lt_row in llst_available})

    lf_first = llst_available[0][0]
    lf_last = llst_available[-1][0]
    li_runs = int((lf_last - lf_first) // ff_step_s) + 1

    return llst_available, [lf_first + li_run * ff_step_s for li_run in range(1, li_runs + 1)]

# ─────────────────────────────────────────────────────────────────────────────────────
def replay(flst_rows: list, fs_out_dir: str, ff_step_s: float = CF_DEFAULT_STEP_MIN * 60.0,
           fs_source: str | None = None) -> tuple:
    """
    roda o tracker rodada a rodada (ver o fluxo no cabeçalho).

    :param flst_rows: tuplas (fetched_ts ou None, ts, lat, lng)
    :param fs_out_dir: diretório de saída (criado; saídas anteriores apagadas)
    :param ff_step_s: intervalo entre rodadas sem fetched_ts (s)
    :param fs_source: fonte dos raios, decide a janela do tracker (None = stsc)
    :returns: (rodadas com raio novo, frames gerados)
    """
    ldct_paths = _prepare_out_dir(fs_out_dir, _source_tag(fs_source))
    llst_available, llst_runs = schedule(flst_rows, ff_step_s)
    llst_avail_ts = [lt_row[0] for lt_row in llst_available]

    llst_window = []
    li_next = 0
    lf_since_ts = 0.0
    li_runs_with_data = 0
    li_frames = 0

    for lf_run_ts in llst_runs:
        li_end = bisect.bisect_right(llst_avail_ts, lf_run_ts)

        # rodada sem raio novo: o tracker não geraria frame
        if li_end == li_next:
            continue

        llst_window.extend(llst_available[li_next:li_end])
        li_next = li_end
        li_runs_with_data += 1

        lf_cutoff_ts = lf_run_ts - CI_RETENTION_S
        llst_window = [lt_row for lt_row in llst_window if lt_row[1] >= lf_cutoff_ts]

        _write_raw_csv(ldct_paths["raw"], llst_window)

        lf_newest_ts = tracker.process(ldct_paths["raw"], ldct_paths["frame"], lf_since_ts,
                                       ldct_paths["tracks"], ldct_paths["verify"],
                                       ldct_paths["assoc"], ldct_paths["frames"],
                                       ff_run_ts=lf_run_ts, fs_source=fs_source)

        if lf_newest_ts is not None:
            lf_since_ts = lf_newest_ts
            li_frames += 1

    return li_runs_with_data, li_frames

# ─────────────────────────────────────────────────────────────────────────────────────
def _source_tag(fs_source: str | None) -> str:
    """
    :param fs_source: fonte (--source; None = stsc)
    :returns: prefixo dos arquivos da fonte ("slm" ou "glm")
    """
    return CDCT_SOURCE_TAG[fs_source or tracker.CS_SOURCE_STSC]

# ─────────────────────────────────────────────────────────────────────────────────────
def _prepare_out_dir(fs_out_dir: str, fs_tag: str = rpt.CS_DEFAULT_TAG) -> dict:
    """
    cria o diretório de saída e apaga as saídas de um replay anterior da
    mesma fonte.

    :param fs_out_dir: diretório de saída
    :param fs_tag: prefixo dos arquivos da fonte
    :returns: caminhos {raw, frame, tracks, verify, assoc, frames}
    """
    ls_work_dir = os.path.join(fs_out_dir, CS_WORK_SUBDIR)
    os.makedirs(ls_work_dir, exist_ok=True)

    ldct_names = rpt.csv_names(fs_tag)
    ls_tracks_name = fs_tag + CS_TRACKS_SUFFIX

    for ls_name in (*ldct_names.values(), ls_tracks_name, CS_REPORT_NAME):
        ls_path = os.path.join(fs_out_dir, ls_name)

        if os.path.exists(ls_path):
            os.remove(ls_path)

    return {
        "raw":    os.path.join(ls_work_dir, CS_RAW_NAME),
        "frame":  os.path.join(ls_work_dir, CS_FRAME_NAME),
        "tracks": os.path.join(fs_out_dir, ls_tracks_name),
        "verify": os.path.join(fs_out_dir, ldct_names["verify"]),
        "assoc":  os.path.join(fs_out_dir, ldct_names["assoc"]),
        "frames": os.path.join(fs_out_dir, ldct_names["frames"]),
    }

# ─────────────────────────────────────────────────────────────────────────────────────
def _write_raw_csv(fs_path: str, flst_window: list) -> None:
    """
    grava o CSV de entrada do tracker (formato do downloader).

    :param fs_path: caminho do CSV
    :param flst_window: tuplas (disponível_em, ts, lat, lng)
    """
    with open(fs_path, "w", encoding="utf-8") as l_file:
        l_file.write("timestamp,lat,lng\n")
        l_file.writelines("%r,%r,%r\n" % (ts, lat, lng) for _, ts, lat, lng in flst_window)

# ═════════════════════════════════════════════════════════════════════════════════════
# programa
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def _parse_args(flst_argv: list | None) -> argparse.Namespace:
    """
    lê a linha de comando.

    :param flst_argv: argumentos (None = sys.argv[1:])
    :returns: namespace com inputs, out_dir, overrides, step_min, source,
              report_only, verbose
    """
    l_parser = argparse.ArgumentParser(
                   description="Simple Lightning Model — reprocessamento para calibração")

    l_parser.add_argument("inputs", nargs="*",
                          help="arquivos de raios (slm_raw_AAAAMMDD.csv.gz ou CSV)")
    l_parser.add_argument("-o", "--out-dir", required=True,
                          help="diretório de saída (estado, CSVs, report.txt)")
    l_parser.add_argument("--set", dest="overrides", action="append", default=[],
                          metavar="MÓDULO.CONSTANTE=VALOR",
                          help="troca uma constante (pode repetir)")
    l_parser.add_argument("--step-min", type=float, default=CF_DEFAULT_STEP_MIN,
                          help="minutos entre rodadas quando os raios não têm fetched_ts")
    l_parser.add_argument("--source", choices=tracker.CLST_SOURCES,
                          default=tracker.CS_SOURCE_STSC,
                          help="fonte dos raios do archive: stsc (padrão) ou glm")
    l_parser.add_argument("--report-only", action="store_true",
                          help="só resume os CSVs já existentes em --out-dir")
    l_parser.add_argument("-v", "--verbose", action="store_true",
                          help="mostra o log do tracker a cada rodada")

    l_args = l_parser.parse_args(flst_argv)

    if not l_args.report_only and not l_args.inputs:
        l_parser.error("informe os arquivos de raios (ou use --report-only)")

    if l_args.step_min <= 0.0:
        l_parser.error("--step-min precisa ser positivo")

    return l_args

# ─────────────────────────────────────────────────────────────────────────────────────
def main(flst_argv: list | None = None) -> int:
    """
    reprocessa (ou só resume) e imprime o relatório.

    :param flst_argv: argumentos (None = sys.argv[1:])
    :returns: CI_RC_OK, CI_RC_NO_DATA ou CI_RC_BAD_ARGS
    """
    l_args = _parse_args(flst_argv)

    try:
        llst_override_lines = apply_overrides(l_args.overrides)

    except ValueError as l_error:
        M_LOG.error("--set: %s", l_error)
        return CI_RC_BAD_ARGS

    llst_header = ["constantes: " + ("; ".join(llst_override_lines) or "padrão")]

    if not l_args.report_only:
        llst_rows = arc.load(l_args.inputs)

        if not llst_rows:
            M_LOG.error("nenhum raio em %s.", ", ".join(l_args.inputs))
            return CI_RC_NO_DATA

        llst_header.insert(0, _run_quietly(l_args, llst_rows))

    ls_report = rpt.format_report(rpt.build(l_args.out_dir, _source_tag(l_args.source)),
                                  llst_header)

    if not l_args.report_only:
        with open(os.path.join(l_args.out_dir, CS_REPORT_NAME), "w", encoding="utf-8") as l_file:
            l_file.write(ls_report)

    print(ls_report, end="")

    return CI_RC_OK

# ─────────────────────────────────────────────────────────────────────────────────────
def _run_quietly(f_args: argparse.Namespace, flst_rows: list) -> str:
    """
    roda o replay; sem --verbose, o log do tracker (uma dezena de linhas por
    rodada) é silenciado.

    :param f_args: argumentos
    :param flst_rows: raios lidos
    :returns: linha de resumo para o topo do relatório
    """
    if not f_args.verbose:
        logging.disable(logging.WARNING)

    try:
        li_runs, li_frames = replay(flst_rows, f_args.out_dir, f_args.step_min * 60.0,
                                    f_args.source)

    finally:
        logging.disable(logging.NOTSET)

    return ("replay: %d raios de %d arquivo(s), %d rodadas com raio novo, %d frames"
            % (len(flst_rows), len(f_args.inputs), li_runs, li_frames))

# ─────────────────────────────────────────────────────────────────────────────────────
# this is the bootstrap process

if "__main__" == __name__:
    logging.basicConfig(datefmt="%Y-%m-%d %H:%M:%S",
                        format="[%(asctime)s] [%(levelname)s] %(message)s",
                        level=logging.INFO)

    sys.exit(main())

# < the end >──────────────────────────────────────────────────────────────────────────
