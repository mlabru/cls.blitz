# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

Simple Lightning Model
slm_verify
verificação das previsões de posição: quanto a previsão de +10/+20/+30 min
errou em relação à célula realmente observada depois.

por quê: sem medir o erro, nenhuma constante do modelo (CF_EPS_KM, ruído do
Kalman, pesos e raio da associação) pode ser ajustada com evidência.

como funciona (chamado por slmTracker.process a cada frame):
  1) verify_cells: para cada célula que continuou um track, as previsões
     pendentes do track cujo instante de validade (emissão + lead) cai a até
     CF_VERIFY_TOLERANCE_MIN do instante da célula viram uma linha de
     verificação; as que já passaram sem célula são descartadas;
  2) register_cells: as previsões novas de cada célula (forecast_path) entram
     como pendentes no track (TStormTrack.forecasts, persistido com o estado);
  3) append_rows: as linhas vão para o CSV de verificação.

o relógio é o mesmo de todo o tracking: TCell.t_start (o dt do Kalman e de
elapsed_hours), então a previsão emitida na célula de t_start = T com lead L
vale para a célula com t_start ≈ T + L.

cada linha traz também o erro da PERSISTÊNCIA (supor que a célula ficou
parada na posição de emissão): a previsão só tem valor se errar menos que ela.

o CSV cresce por append (um só escritor, serializado pelo flock do
slm_run.sh) e é rotacionado pelo check_log_size do slm_run.sh; cabeçalho é
escrito quando o arquivo está vazio ou não existe.  Análise offline (erro
médio por lead, previsão x persistência), sem dependências:

    awk -F, 'NR>1 {n[$4]++; e[$4]+=$9; p[$4]+=$10}
             END {for (l in n) printf "%s min: %.1f km (persist. %.1f km, n=%d)\n",
                                      l, e[l]/n[l], p[l]/n[l], n[l]}' slm_verification.csv

2026.nov  mlabru  initial version
"""
# < imports >──────────────────────────────────────────────────────────────────────────

# python library
from __future__ import annotations

import csv
import logging
import os

# locals
import work.slm_spatial as spc
import work.slm_struct as sst

# < constants >────────────────────────────────────────────────────────────────────────

# folga (minutos) entre o instante de validade da previsão e o da célula
# observada; metade do intervalo do cron (5 min), para cada previsão casar com
# no máximo um frame
CF_VERIFY_TOLERANCE_MIN = 2.5

# colunas do CSV de verificação, na ordem do arquivo
CLST_VERIFY_COLUMNS = ["verify_ts", "track_id", "issue_ts", "lead_min",
                       "fcst_lat", "fcst_lng", "obs_lat", "obs_lng",
                       "error_km", "persist_km"]

# casas decimais gravadas (graus: ~1 m; km: 1 m)
CI_DEG_DECIMALS = 5
CI_KM_DECIMALS = 3

# < logging >──────────────────────────────────────────────────────────────────────────

M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# ═════════════════════════════════════════════════════════════════════════════════════
# verificação
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def verify_cells(flst_cells: list, flst_tracks: list,
                 ff_tolerance_min: float = CF_VERIFY_TOLERANCE_MIN) -> list:
    """
    verifica as previsões pendentes de cada track contra a célula do frame
    atual que o continuou.  Chamar DEPOIS de update_tracks e ANTES de
    register_cells (senão a previsão recém-emitida seria testada contra a
    própria célula).

    :param flst_cells: células do frame atual, já com ID de track
    :param flst_tracks: tracks atualizados (os forecasts são ALTERADOS)
    :param ff_tolerance_min: folga entre validade e instante da célula
    :returns: linhas de verificação (dicts com CLST_VERIFY_COLUMNS)
    """
    ldct_tracks_by_id = {l_track.id: l_track for l_track in flst_tracks}
    llst_rows = []

    for l_cell in flst_cells:
        l_track = ldct_tracks_by_id.get(l_cell.id)

        if l_track is None:
            continue

        llst_rows.extend(verify_track(l_track, l_cell, ff_tolerance_min))

    return llst_rows

# ─────────────────────────────────────────────────────────────────────────────────────
def verify_track(f_track: sst.TStormTrack, f_cell: sst.TCell,
                 ff_tolerance_min: float = CF_VERIFY_TOLERANCE_MIN) -> list:
    """
    compara as previsões pendentes de um track com a célula observada.

    cada previsão pendente tem um de três destinos:
      - validade a até ff_tolerance_min da célula: vira linha e sai;
      - validade ainda no futuro: continua pendente;
      - validade já passou sem célula (frame perdido): é descartada.

    :param f_track: track (f_track.forecasts é ALTERADO)
    :param f_cell: célula do frame atual que continuou o track
    :param ff_tolerance_min: folga entre validade e instante da célula
    :returns: linhas de verificação deste track
    """
    lf_tolerance_s = ff_tolerance_min * 60.0
    llst_rows = []
    llst_pending = []

    for ldct_forecast in f_track.forecasts:
        lf_valid_ts = ldct_forecast["issue_ts"] + ldct_forecast["lead_min"] * 60.0

        if abs(lf_valid_ts - f_cell.t_start) <= lf_tolerance_s:
            llst_rows.append(_verification_row(f_track.id, ldct_forecast, f_cell))

        elif lf_valid_ts > f_cell.t_start:
            llst_pending.append(ldct_forecast)

    f_track.forecasts = llst_pending

    return llst_rows

# ─────────────────────────────────────────────────────────────────────────────────────
def _verification_row(fs_track_id: str, fdct_forecast: dict, f_cell: sst.TCell) -> dict:
    """
    uma linha do CSV: a previsão, a posição observada, o erro da previsão e o
    da persistência.

    :param fs_track_id: ID do track
    :param fdct_forecast: previsão pendente (issue_ts, lead_min, lat, lng, lat0, lng0)
    :param f_cell: célula observada
    :returns: dict com CLST_VERIFY_COLUMNS
    """
    lf_obs_lat, lf_obs_lng = f_cell.centroid

    lf_error_km = spc.haversine_km(fdct_forecast["lat"], fdct_forecast["lng"],
                                   lf_obs_lat, lf_obs_lng)
    lf_persist_km = spc.haversine_km(fdct_forecast["lat0"], fdct_forecast["lng0"],
                                     lf_obs_lat, lf_obs_lng)

    return {
        "verify_ts":  f_cell.t_start,
        "track_id":   fs_track_id,
        "issue_ts":   fdct_forecast["issue_ts"],
        "lead_min":   fdct_forecast["lead_min"],
        "fcst_lat":   round(fdct_forecast["lat"], CI_DEG_DECIMALS),
        "fcst_lng":   round(fdct_forecast["lng"], CI_DEG_DECIMALS),
        "obs_lat":    round(lf_obs_lat, CI_DEG_DECIMALS),
        "obs_lng":    round(lf_obs_lng, CI_DEG_DECIMALS),
        "error_km":   round(lf_error_km, CI_KM_DECIMALS),
        "persist_km": round(lf_persist_km, CI_KM_DECIMALS),
    }

# ═════════════════════════════════════════════════════════════════════════════════════
# registro das previsões novas
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def register_cells(flst_cells: list, flst_tracks: list) -> None:
    """
    guarda no track de cada célula as previsões que ela acabou de emitir.
    Chamar depois de preenchido o forecast_path (e depois de verify_cells).

    :param flst_cells: células do frame atual, com ID de track e forecast_path
    :param flst_tracks: tracks atualizados (os forecasts são ALTERADOS)
    """
    ldct_tracks_by_id = {l_track.id: l_track for l_track in flst_tracks}

    for l_cell in flst_cells:
        l_track = ldct_tracks_by_id.get(l_cell.id)

        if l_track is None:
            continue

        l_track.forecasts.extend(_pending_forecasts(l_cell))

# ─────────────────────────────────────────────────────────────────────────────────────
def _pending_forecasts(f_cell: sst.TCell) -> list:
    """
    as previsões de uma célula no formato pendente (o que verify_track lê).

    :param f_cell: célula com forecast_path (slm_motion.extrapolate_path)
    :returns: lista de dicts issue_ts, lead_min, lat, lng, lat0, lng0
    """
    lf_lat0, lf_lng0 = f_cell.centroid

    return [{"issue_ts": f_cell.t_start,
             "lead_min": ldct_point["lead_min"],
             "lat":      ldct_point["lat"],
             "lng":      ldct_point["lng"],
             "lat0":     lf_lat0,
             "lng0":     lf_lng0}
            for ldct_point in f_cell.forecast_path]

# ═════════════════════════════════════════════════════════════════════════════════════
# saída
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def append_rows(fs_path: str, flst_rows: list) -> None:
    """
    acrescenta as linhas ao CSV de verificação e resume o frame no log.

    [doc] append, não escrita atômica: o arquivo é um registro que só cresce,
    com um só escritor (o flock do slm_run.sh serializa as rodadas), e
    nenhum outro processo do pipeline o lê.

    :param fs_path: CSV de verificação (o diretório já precisa existir)
    :param flst_rows: linhas de verify_cells; vazia = nada é gravado
    """
    if not flst_rows:
        return

    lb_needs_header = not os.path.isfile(fs_path) or 0 == os.path.getsize(fs_path)

    with open(fs_path, "a", newline="") as l_file:
        l_writer = csv.DictWriter(l_file, fieldnames=CLST_VERIFY_COLUMNS)

        if lb_needs_header:
            l_writer.writeheader()

        l_writer.writerows(flst_rows)

    lf_error_km, lf_persist_km = mean_errors(flst_rows)

    M_LOG.info("verificação: %d previsões, erro médio %.1f km (persistência %.1f km).",
               len(flst_rows), lf_error_km, lf_persist_km)

# ─────────────────────────────────────────────────────────────────────────────────────
def mean_errors(flst_rows: list) -> tuple:
    """
    erro médio da previsão e da persistência.

    :param flst_rows: linhas de verificação (não vazia)
    :returns: (erro médio da previsão em km, erro médio da persistência em km)
    """
    li_count = len(flst_rows)

    return (sum(ldct_row["error_km"] for ldct_row in flst_rows) / li_count,
            sum(ldct_row["persist_km"] for ldct_row in flst_rows) / li_count)

# < the end >──────────────────────────────────────────────────────────────────────────
