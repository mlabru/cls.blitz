# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

Simple Lightning Model
slm_diag
registros de diagnóstico do tracker, para calibrar as constantes do modelo
(slmReplay.py --report / --report-only os resume).

  - associação (slm_assoc.csv): uma linha por célula do frame -- como ela
    entrou no tracking (kind), a distância até a posição prevista pelo
    Kalman, o match_score, os tamanhos e o NIS (inovação normalizada).
    Calibra CF_MAX_DIST_KM, CF_MAX_MATCH_SCORE e, pelo NIS, Q/R do Kalman;
  - frames (slm_frames.csv): uma linha por frame -- latência (instante da
    rodada menos o raio mais recente), raios na janela e quantos viraram
    célula, e o ciclo de vida dos tracks (novos, divisões, fusões, perdidos,
    expirados).  Calibra CF_EPS_KM/CI_MIN_PTS e a idade dos tracks.

kinds de célula (TCell.assoc, preenchido por slm_track.update_tracks):
  continue   continuou um track (predict -> update do Kalman; tem NIS)
  two_point  2a célula do track: Kalman reiniciado pelos dois pontos (sem NIS)
  new        abriu track; dist_km/score/size_prev = o track sem par mais
             próximo (vazios se não havia nenhum)
  split      abriu track filho de uma divisão (idem)

os dois CSVs crescem por append (slm_ioutil.append_csv_rows) e são
rotacionados pelo check_log_size do slm_run.sh, como o de verificação.

2026.nov  mlabru  initial version
"""
# < imports >──────────────────────────────────────────────────────────────────────────

# python library
from __future__ import annotations

import logging

# locals
import work.slm_ioutil as iou

# < constants >────────────────────────────────────────────────────────────────────────

# colunas do CSV de associação, na ordem do arquivo
CLST_ASSOC_COLUMNS = ["frame_ts", "track_id", "kind", "dt_min", "dist_km", "score",
                      "size_prev", "size", "nis"]

# colunas do CSV de frames, na ordem do arquivo
CLST_FRAME_COLUMNS = ["frame_ts", "run_ts", "latency_s", "n_window_events", "n_clustered",
                      "n_cells", "n_continued", "n_two_point", "n_new", "n_split",
                      "n_missed", "n_merged", "n_expired", "n_active"]

# motivos de fim de track contados no CSV de frames (TStormTrack.end_reason)
CT_END_REASONS = ("missed", "merged", "expired")

# casas decimais gravadas
CI_DECIMALS = 3

# < logging >──────────────────────────────────────────────────────────────────────────

M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────────────
def association_rows(ff_frame_ts: float, flst_cells: list) -> list:
    """
    uma linha por célula do frame, a partir de TCell.assoc.

    :param ff_frame_ts: instante do frame (raio mais recente, epoch UTC)
    :param flst_cells: células do frame, depois de update_tracks
    :returns: dicts com CLST_ASSOC_COLUMNS (campos ausentes ficam vazios)
    """
    llst_rows = []

    for l_cell in flst_cells:
        if not l_cell.assoc:
            continue

        ldct_row = {"frame_ts": ff_frame_ts, "track_id": l_cell.id, "size": l_cell.size}

        for ls_key, l_value in l_cell.assoc.items():
            ldct_row[ls_key] = round(l_value, CI_DECIMALS) if isinstance(l_value, float) else l_value

        llst_rows.append(ldct_row)

    return llst_rows

# ─────────────────────────────────────────────────────────────────────────────────────
def frame_row(ff_frame_ts: float, ff_run_ts: float, fi_window_events: int,
              flst_cells: list, flst_tracks: list) -> dict:
    """
    o resumo de um frame.

    :param ff_frame_ts: instante do frame (raio mais recente, epoch UTC)
    :param ff_run_ts: instante da rodada (relógio do sistema; no replay, o
                      instante simulado da rodada)
    :param fi_window_events: raios na janela de agrupamento
    :param flst_cells: células do frame, depois de update_tracks
    :param flst_tracks: tracks depois de update_tracks (os desativados neste
                        frame têm end_reason)
    :returns: dict com CLST_FRAME_COLUMNS
    """
    ldct_kinds = {"continue": 0, "two_point": 0, "new": 0, "split": 0}

    for l_cell in flst_cells:
        ls_kind = l_cell.assoc.get("kind")

        if ls_kind in ldct_kinds:
            ldct_kinds[ls_kind] += 1

    ldct_row = {
        "frame_ts":        ff_frame_ts,
        "run_ts":          round(ff_run_ts, CI_DECIMALS),
        "latency_s":       round(ff_run_ts - ff_frame_ts, CI_DECIMALS),
        "n_window_events": fi_window_events,
        "n_clustered":     sum(l_cell.size for l_cell in flst_cells),
        "n_cells":         len(flst_cells),
        "n_continued":     ldct_kinds["continue"],
        "n_two_point":     ldct_kinds["two_point"],
        "n_new":           ldct_kinds["new"],
        "n_split":         ldct_kinds["split"],
        "n_active":        sum(1 for l_track in flst_tracks if l_track.active),
    }

    for ls_reason in CT_END_REASONS:
        ldct_row["n_" + ls_reason] = sum(1 for l_track in flst_tracks
                                         if ls_reason == l_track.end_reason)

    return ldct_row

# ─────────────────────────────────────────────────────────────────────────────────────
def append(fs_path: str, flst_columns: list, flst_rows: list) -> None:
    """
    acrescenta linhas a um dos CSVs de diagnóstico.

    :param fs_path: caminho do CSV (o diretório já precisa existir)
    :param flst_columns: CLST_ASSOC_COLUMNS ou CLST_FRAME_COLUMNS
    :param flst_rows: linhas; vazia = nada é gravado
    """
    if iou.append_csv_rows(fs_path, flst_columns, flst_rows):
        M_LOG.warning("%s tinha outro layout de colunas: renomeado para .old.", fs_path)

# < the end >──────────────────────────────────────────────────────────────────────────
