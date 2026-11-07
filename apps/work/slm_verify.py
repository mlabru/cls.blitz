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
     CF_VERIFY_TOLERANCE_MIN do instante da célula viram uma linha "ok"; as
     que já passaram sem célula viram uma linha "gap";
  2) close_tracks: as previsões pendentes dos tracks desativados neste frame
     viram linhas com o motivo do fim ("missed", "merged", "expired");
  3) register_cells: as previsões novas de cada célula (forecast_path) entram
     como pendentes no track (TStormTrack.forecasts, persistido com o estado);
  4) append_rows: as linhas vão para o CSV de verificação.

a previsão é emitida no instante do FRAME (o raio mais recente) e vale para
emissão + lead -- é assim que o slmTracker a calcula.  A célula observada é
comparada pelo instante dela (sst.cell_time, a média dos raios: o instante a
que o centroide corresponde), então a previsão emitida no frame T com lead
L é conferida com a célula de instante ≈ T + L, que em geral aparece uns
10 min depois, num frame posterior.

cada linha "ok" traz:
  - o erro da previsão e o da PERSISTÊNCIA (supor que a célula ficou parada
    na posição de emissão): a previsão só tem valor se errar menos que ela;
  - a incerteza 1-sigma que o Kalman atribuiu à previsão (std_lat_km,
    std_lng_km) e z2 = soma dos dois erros normalizados ao quadrado.  Com o
    cone bem calibrado, z2 segue uma qui-quadrado de 2 graus de liberdade:
    média 2, 39% das linhas com z2 <= 1, 86% com z2 <= 4.

[new] as linhas não "ok" (status gap/missed/merged/expired) existem para não
haver viés de sobrevivência: sem elas, uma constante que matasse os tracks
difíceis pareceria melhorar o erro médio.  Nelas obs/erro/z2 ficam vazios.

o CSV cresce por append (slm_ioutil.append_csv_rows) e é rotacionado pelo
check_log_size do slm_run.sh.  Análise offline: slmReplay.py --report-only,
ou, sem dependências (só as linhas ok):

    awk -F, 'NR>1 && $14=="ok" {n[$4]++; e[$4]+=$9; p[$4]+=$10}
             END {for (l in n) printf "%s min: %.1f km (persist. %.1f km, n=%d)\n",
                                      l, e[l]/n[l], p[l]/n[l], n[l]}' slm_verification.csv

2026.nov  mlabru  σ da previsão, z2 e status por linha; previsões perdidas
                  (gap, track desativado) também viram linha; append pelo
                  slm_ioutil.append_csv_rows (rotação se o layout mudou)
2026.nov  mlabru  [fix] previsão emitida no instante do frame (register_cells)
2026.nov  mlabru  [fix] relógio pelo sst.cell_time (era t_start)
2026.nov  mlabru  initial version
"""
# < imports >──────────────────────────────────────────────────────────────────────────

# python library
from __future__ import annotations

import logging

# locals
import work.slm_geo as geo
import work.slm_ioutil as iou
import work.slm_spatial as spc
import work.slm_struct as sst

# < constants >────────────────────────────────────────────────────────────────────────

# folga (minutos) entre o instante de validade da previsão e o da célula
# observada; metade do intervalo do cron (5 min), para cada previsão casar com
# no máximo um frame
CF_VERIFY_TOLERANCE_MIN = 2.5

# colunas do CSV de verificação, na ordem do arquivo.  As 10 primeiras são
# as da versão anterior (o awk do cabeçalho conta com $4, $9 e $10)
CLST_VERIFY_COLUMNS = ["verify_ts", "track_id", "issue_ts", "lead_min",
                       "fcst_lat", "fcst_lng", "obs_lat", "obs_lng",
                       "error_km", "persist_km",
                       "std_lat_km", "std_lng_km", "z2", "status"]

# status de uma linha: previsão conferida com a célula observada
CS_STATUS_OK = "ok"
# a validade passou sem célula do track (o track seguiu, mas pulou frames)
CS_STATUS_GAP = "gap"
# track desativado sem TStormTrack.end_reason (não deveria acontecer)
CS_STATUS_ENDED = "ended"

# casas decimais gravadas (graus: ~1 m; km: 1 m)
CI_DEG_DECIMALS = 5
CI_KM_DECIMALS = 3
CI_Z2_DECIMALS = 4

# < logging >──────────────────────────────────────────────────────────────────────────

M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# ═════════════════════════════════════════════════════════════════════════════════════
# verificação
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def verify_cells(flst_cells: list, flst_tracks: list,
                 ff_tolerance_min: float | None = None) -> list:
    """
    verifica as previsões pendentes de cada track contra a célula do frame
    atual que o continuou.  Chamar DEPOIS de update_tracks e ANTES de
    register_cells (senão a previsão recém-emitida seria testada contra a
    própria célula).

    :param flst_cells: células do frame atual, já com ID de track
    :param flst_tracks: tracks atualizados (os forecasts são ALTERADOS)
    :param ff_tolerance_min: folga entre validade e instante da célula
                             (None = CF_VERIFY_TOLERANCE_MIN)
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
                 ff_tolerance_min: float | None = None) -> list:
    """
    compara as previsões pendentes de um track com a célula observada.

    cada previsão pendente tem um de três destinos:
      - validade a até ff_tolerance_min da célula: vira linha "ok" e sai;
      - validade ainda no futuro: continua pendente;
      - validade já passou sem célula (frame perdido): vira linha "gap" e sai.

    :param f_track: track (f_track.forecasts é ALTERADO)
    :param f_cell: célula do frame atual que continuou o track
    :param ff_tolerance_min: folga entre validade e instante da célula
                             (None = CF_VERIFY_TOLERANCE_MIN)
    :returns: linhas de verificação deste track
    """
    if ff_tolerance_min is None:
        ff_tolerance_min = CF_VERIFY_TOLERANCE_MIN

    lf_tolerance_s = ff_tolerance_min * 60.0
    lf_cell_ts = sst.cell_time(f_cell)
    llst_rows = []
    llst_pending = []

    for ldct_forecast in f_track.forecasts:
        lf_valid_ts = ldct_forecast["issue_ts"] + ldct_forecast["lead_min"] * 60.0

        if abs(lf_valid_ts - lf_cell_ts) <= lf_tolerance_s:
            llst_rows.append(_verification_row(f_track.id, ldct_forecast, f_cell))

        elif lf_valid_ts > lf_cell_ts:
            llst_pending.append(ldct_forecast)

        else:
            llst_rows.append(_lost_row(f_track.id, ldct_forecast, lf_cell_ts, CS_STATUS_GAP))

    f_track.forecasts = llst_pending

    return llst_rows

# ─────────────────────────────────────────────────────────────────────────────────────
def close_tracks(flst_tracks: list, ff_frame_ts: float) -> list:
    """
    as previsões pendentes dos tracks desativados neste frame: nunca vão ser
    conferidas, então viram linhas com o motivo do fim do track
    (TStormTrack.end_reason) e saem do track.

    [new] sem isto as previsões de tracks que morriam sumiam sem registro
    (viés de sobrevivência na verificação).

    :param flst_tracks: tracks depois de update_tracks (forecasts dos
                        inativos são ALTERADOS)
    :param ff_frame_ts: instante do frame (vai para verify_ts)
    :returns: linhas de verificação (status = motivo do fim)
    """
    llst_rows = []

    for l_track in flst_tracks:
        if l_track.active or not l_track.forecasts:
            continue

        ls_status = l_track.end_reason or CS_STATUS_ENDED

        llst_rows.extend(_lost_row(l_track.id, ldct_forecast, ff_frame_ts, ls_status)
                         for ldct_forecast in l_track.forecasts)

        l_track.forecasts = []

    return llst_rows

# ─────────────────────────────────────────────────────────────────────────────────────
def _verification_row(fs_track_id: str, fdct_forecast: dict, f_cell: sst.TCell) -> dict:
    """
    uma linha "ok": a previsão, a posição observada, o erro da previsão, o
    da persistência e o erro normalizado pela incerteza (z2).

    :param fs_track_id: ID do track
    :param fdct_forecast: previsão pendente (issue_ts, lead_min, lat, lng,
                          lat0, lng0 e, desde 2026.nov, lat_std, lng_std)
    :param f_cell: célula observada
    :returns: dict com CLST_VERIFY_COLUMNS
    """
    lf_obs_lat, lf_obs_lng = f_cell.centroid

    lf_error_km = spc.haversine_km(fdct_forecast["lat"], fdct_forecast["lng"],
                                   lf_obs_lat, lf_obs_lng)
    lf_persist_km = spc.haversine_km(fdct_forecast["lat0"], fdct_forecast["lng0"],
                                     lf_obs_lat, lf_obs_lng)

    ldct_row = _forecast_fields(fs_track_id, fdct_forecast, sst.cell_time(f_cell))

    ldct_row.update({
        "obs_lat":    round(lf_obs_lat, CI_DEG_DECIMALS),
        "obs_lng":    round(lf_obs_lng, CI_DEG_DECIMALS),
        "error_km":   round(lf_error_km, CI_KM_DECIMALS),
        "persist_km": round(lf_persist_km, CI_KM_DECIMALS),
        "z2":         _normalized_error_sq(fdct_forecast, lf_obs_lat, lf_obs_lng),
        "status":     CS_STATUS_OK,
    })

    return ldct_row

# ─────────────────────────────────────────────────────────────────────────────────────
def _lost_row(fs_track_id: str, fdct_forecast: dict, ff_verify_ts: float,
              fs_status: str) -> dict:
    """
    uma linha de previsão que não pôde ser conferida (obs, erro e z2 vazios).

    :param fs_track_id: ID do track
    :param fdct_forecast: previsão pendente
    :param ff_verify_ts: instante em que a perda foi constatada
    :param fs_status: motivo (CS_STATUS_GAP ou o end_reason do track)
    :returns: dict com CLST_VERIFY_COLUMNS
    """
    ldct_row = _forecast_fields(fs_track_id, fdct_forecast, ff_verify_ts)

    ldct_row.update({"obs_lat": "", "obs_lng": "", "error_km": "", "persist_km": "",
                     "z2": "", "status": fs_status})

    return ldct_row

# ─────────────────────────────────────────────────────────────────────────────────────
def _forecast_fields(fs_track_id: str, fdct_forecast: dict, ff_verify_ts: float) -> dict:
    """
    as colunas que vêm só da previsão (comuns a toda linha).

    :param fs_track_id: ID do track
    :param fdct_forecast: previsão pendente
    :param ff_verify_ts: instante da verificação
    :returns: dict parcial (verify_ts, track_id, issue_ts, lead_min,
              fcst_lat, fcst_lng, std_lat_km, std_lng_km); std vazio se a
              previsão for de um estado anterior a 2026.nov (sem lat_std)
    """
    lf_lat_std = fdct_forecast.get("lat_std")
    lf_lng_std = fdct_forecast.get("lng_std")

    return {
        "verify_ts":  ff_verify_ts,
        "track_id":   fs_track_id,
        "issue_ts":   fdct_forecast["issue_ts"],
        "lead_min":   fdct_forecast["lead_min"],
        "fcst_lat":   round(fdct_forecast["lat"], CI_DEG_DECIMALS),
        "fcst_lng":   round(fdct_forecast["lng"], CI_DEG_DECIMALS),
        "std_lat_km": "" if lf_lat_std is None else
                      round(lf_lat_std * geo.CF_KM_PER_DEG_LAT, CI_KM_DECIMALS),
        "std_lng_km": "" if lf_lng_std is None else
                      round(lf_lng_std * geo.km_per_deg_lng(fdct_forecast["lat"]),
                            CI_KM_DECIMALS),
    }

# ─────────────────────────────────────────────────────────────────────────────────────
def _normalized_error_sq(fdct_forecast: dict, ff_obs_lat: float, ff_obs_lng: float) -> float | str:
    """
    z2: erro de cada eixo dividido pelo 1-sigma do eixo, ao quadrado, somados.

    a conta é feita em graus: o fator km/grau de cada eixo aparece no erro e
    no sigma e se cancela.

    :param fdct_forecast: previsão pendente (lat, lng, lat_std, lng_std)
    :param ff_obs_lat: latitude observada
    :param ff_obs_lng: longitude observada
    :returns: z2; "" se a previsão não tiver sigma (estado antigo) ou sigma <= 0
    """
    lf_lat_std = fdct_forecast.get("lat_std")
    lf_lng_std = fdct_forecast.get("lng_std")

    if not lf_lat_std or not lf_lng_std or lf_lat_std <= 0.0 or lf_lng_std <= 0.0:
        return ""

    lf_z_lat = (ff_obs_lat - fdct_forecast["lat"]) / lf_lat_std
    lf_z_lng = (ff_obs_lng - fdct_forecast["lng"]) / lf_lng_std

    return round(lf_z_lat * lf_z_lat + lf_z_lng * lf_z_lng, CI_Z2_DECIMALS)

# ═════════════════════════════════════════════════════════════════════════════════════
# registro das previsões novas
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def register_cells(flst_cells: list, flst_tracks: list,
                   ff_issue_ts: float | None = None) -> None:
    """
    guarda no track de cada célula as previsões que ela acabou de emitir.
    Chamar depois de preenchido o forecast_path (e depois de verify_cells).

    :param flst_cells: células do frame atual, com ID de track e forecast_path
    :param flst_tracks: tracks atualizados (os forecasts são ALTERADOS)
    :param ff_issue_ts: instante a partir do qual os leads contam (o do frame);
                        None = o instante de cada célula
    """
    ldct_tracks_by_id = {l_track.id: l_track for l_track in flst_tracks}

    for l_cell in flst_cells:
        l_track = ldct_tracks_by_id.get(l_cell.id)

        if l_track is None:
            continue

        l_track.forecasts.extend(_pending_forecasts(l_cell, ff_issue_ts))

# ─────────────────────────────────────────────────────────────────────────────────────
def _pending_forecasts(f_cell: sst.TCell, ff_issue_ts: float | None = None) -> list:
    """
    as previsões de uma célula no formato pendente (o que verify_track lê).

    :param f_cell: célula com forecast_path (slm_motion.extrapolate_path)
    :param ff_issue_ts: instante de emissão; None = o instante da célula
    :returns: lista de dicts issue_ts, lead_min, lat, lng, lat_std, lng_std,
              lat0, lng0 (std em graus; None se o ponto não tiver)
    """
    lf_lat0, lf_lng0 = f_cell.centroid
    lf_issue_ts = sst.cell_time(f_cell) if ff_issue_ts is None else ff_issue_ts

    return [{"issue_ts": lf_issue_ts,
             "lead_min": ldct_point["lead_min"],
             "lat":      ldct_point["lat"],
             "lng":      ldct_point["lng"],
             "lat_std":  ldct_point.get("lat_std"),
             "lng_std":  ldct_point.get("lng_std"),
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

    :param fs_path: CSV de verificação (o diretório já precisa existir)
    :param flst_rows: linhas de verify_cells/close_tracks; vazia = nada é gravado
    """
    if not flst_rows:
        return

    if iou.append_csv_rows(fs_path, CLST_VERIFY_COLUMNS, flst_rows):
        M_LOG.warning("%s tinha outro layout de colunas: renomeado para .old.", fs_path)

    li_lost = sum(1 for ldct_row in flst_rows if not _is_ok(ldct_row))

    if li_lost == len(flst_rows):
        M_LOG.info("verificação: %d previsões perdidas, nenhuma conferida.", li_lost)
        return

    lf_error_km, lf_persist_km = mean_errors(flst_rows)

    M_LOG.info("verificação: %d previsões, erro médio %.1f km (persistência %.1f km); "
               "%d perdidas.", len(flst_rows) - li_lost, lf_error_km, lf_persist_km, li_lost)

# ─────────────────────────────────────────────────────────────────────────────────────
def mean_errors(flst_rows: list) -> tuple:
    """
    erro médio da previsão e da persistência, só nas linhas "ok".

    :param flst_rows: linhas de verificação (pelo menos uma "ok"; linha sem
                      a chave status conta como "ok")
    :returns: (erro médio da previsão em km, erro médio da persistência em km)
    """
    llst_ok_rows = [ldct_row for ldct_row in flst_rows if _is_ok(ldct_row)]
    li_count = len(llst_ok_rows)

    return (sum(ldct_row["error_km"] for ldct_row in llst_ok_rows) / li_count,
            sum(ldct_row["persist_km"] for ldct_row in llst_ok_rows) / li_count)

# ─────────────────────────────────────────────────────────────────────────────────────
def _is_ok(fdct_row: dict) -> bool:
    """
    :param fdct_row: linha de verificação
    :returns: True se a previsão foi conferida (status "ok" ou ausente)
    """
    return CS_STATUS_OK == fdct_row.get("status", CS_STATUS_OK)

# < the end >──────────────────────────────────────────────────────────────────────────
