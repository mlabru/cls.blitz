# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

Simple Lightning Model
blt_report
resumo dos CSVs de verificação e de diagnóstico (blt_verify, blt_diag): os
números que guiam a calibração das constantes.  Usado pelo bltReplay.py
(depois de um reprocessamento, ou --report-only sobre os CSVs da operação).

o que cada bloco mede e o valor de referência:

  verificação, por lead
    erro / persist.   erro médio da previsão e da persistência (km)
    skill             1 − erro/persist.: > 0 = a previsão vale algo
    conferidas        fração das previsões emitidas que foram conferidas;
                      o resto, por motivo (gap, missed, merged, expired)
    E[z2]             média do erro normalizado pela incerteza do cone:
                      2 = cone calibrado; < 2 cone largo; > 2 cone estreito
    z2<=1, z2<=4      fração dentro de 1 e de 2 sigmas: 39% e 86% se calibrado

  associação
    dist_km           distância até a posição prevista nas associações
                      aceitas (continue): p99 com folga = CF_MAX_DIST_KM
    NIS               média 2 e 5% acima de 5,99 se Q/R do Kalman estão certos
    células novas     distância até o track sem par mais próximo: muitas
                      perto (<= CF_NEAR_KM) = fragmentação (corte apertado)

  tracks              duração (do 1o ao último frame); só tracks com
                      >= CI_VERIFIABLE_TRACK_MIN conferem a previsão de +30

  frames              latência, células por frame, fração dos raios da
                      janela que virou célula (DBSCAN) e o ciclo de vida

2026.nov  mlabru  initial version
"""
# < imports >──────────────────────────────────────────────────────────────────────────

# python library
from __future__ import annotations

import collections
import csv
import math
import os

# numPy
import numpy as np

# < constants >────────────────────────────────────────────────────────────────────────

# nomes dos CSVs num diretório de saída (os mesmos do blt_cfg.sh): o prefixo
# da fonte (slm = STSC, glm = GLM) + o sufixo; as duas dividem data/state/
CS_VERIFY_SUFFIX = "_verification.csv"
CS_ASSOC_SUFFIX = "_assoc.csv"
CS_FRAMES_SUFFIX = "_frames.csv"
CS_DEFAULT_TAG = "slm"

# os nomes do STSC (padrão)
CS_VERIFY_NAME = CS_DEFAULT_TAG + CS_VERIFY_SUFFIX
CS_ASSOC_NAME = CS_DEFAULT_TAG + CS_ASSOC_SUFFIX
CS_FRAMES_NAME = CS_DEFAULT_TAG + CS_FRAMES_SUFFIX

# qui-quadrado de 2 graus de liberdade (z2 e NIS bem calibrados)
CF_CHI2_MEAN = 2.0
CF_CHI2_P_LE_1 = 1.0 - math.exp(-0.5)     # 0.393: fração com z2 <= 1
CF_CHI2_P_LE_4 = 1.0 - math.exp(-2.0)     # 0.865: fração com z2 <= 4
CF_CHI2_P95 = -2.0 * math.log(0.05)       # 5.99: 95% abaixo

# percentis mostrados
CT_PERCENTILES = (50, 90, 99)

# célula nova a até esta distância (km) de um track sem par: candidata a
# fragmentação (o mesmo sistema virou dois tracks)
CF_NEAR_KM = 20.0

# duração mínima (min) para um track conferir a previsão de +30 min (lead +
# ~10 min do atraso do centroide)
CI_VERIFIABLE_TRACK_MIN = 40

# status "ok" do CSV de verificação (blt_verify.CS_STATUS_OK)
CS_STATUS_OK = "ok"

# ═════════════════════════════════════════════════════════════════════════════════════
# leitura
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def read_csv(fs_path: str) -> list:
    """
    :param fs_path: CSV com cabeçalho
    :returns: lista de dicts (valores str); vazia se o arquivo não existir
    """
    if not os.path.isfile(fs_path):
        return []

    with open(fs_path, encoding="utf-8", newline="") as l_file:
        return list(csv.DictReader(l_file))

# ─────────────────────────────────────────────────────────────────────────────────────
def csv_names(fs_tag: str = CS_DEFAULT_TAG) -> dict:
    """
    os nomes dos três CSVs de uma fonte.

    :param fs_tag: prefixo da fonte ("slm" ou "glm")
    :returns: {"verify": ..., "assoc": ..., "frames": ...}
    """
    return {"verify": fs_tag + CS_VERIFY_SUFFIX,
            "assoc":  fs_tag + CS_ASSOC_SUFFIX,
            "frames": fs_tag + CS_FRAMES_SUFFIX}

# ─────────────────────────────────────────────────────────────────────────────────────
def build(fs_dir: str, fs_tag: str = CS_DEFAULT_TAG) -> dict:
    """
    resume os três CSVs de uma fonte num diretório.

    [fix] com data/ comum às duas fontes, os nomes levam o prefixo da fonte:
    antes eram fixos (slm_*) e o --report-only nunca via os CSVs do GLM.

    :param fs_dir: diretório com <tag>_verification.csv, <tag>_assoc.csv e
                   <tag>_frames.csv (os que faltarem contam como vazios)
    :param fs_tag: prefixo da fonte ("slm" ou "glm")
    :returns: {"verification": [...], "association": {...}, "tracks": {...}, "frames": {...}}
    """
    ldct_names = csv_names(fs_tag)
    llst_assoc_rows = read_csv(os.path.join(fs_dir, ldct_names["assoc"]))

    return {
        "verification": summarize_verification(read_csv(os.path.join(fs_dir, ldct_names["verify"]))),
        "association":  summarize_association(llst_assoc_rows),
        "tracks":       summarize_tracks(llst_assoc_rows),
        "frames":       summarize_frames(read_csv(os.path.join(fs_dir, ldct_names["frames"]))),
    }

# ═════════════════════════════════════════════════════════════════════════════════════
# resumos
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def summarize_verification(flst_rows: list) -> list:
    """
    verificação por lead.

    :param flst_rows: linhas do CSV de verificação (valores str ou números)
    :returns: lista (por lead crescente) de dicts lead_min, n, n_ok, frac_ok,
              lost {status: n}, error_km, persist_km, skill, n_z2, z2_mean,
              frac_z2_le_1, frac_z2_le_4 (nan quando não há dado)
    """
    ldct_by_lead = collections.defaultdict(list)

    for ldct_row in flst_rows:
        ldct_by_lead[int(float(ldct_row["lead_min"]))].append(ldct_row)

    return [_summarize_lead(li_lead, ldct_by_lead[li_lead]) for li_lead in sorted(ldct_by_lead)]

# ─────────────────────────────────────────────────────────────────────────────────────
def _summarize_lead(fi_lead_min: int, flst_rows: list) -> dict:
    """
    :param fi_lead_min: lead
    :param flst_rows: linhas desse lead
    :returns: dict (ver summarize_verification)
    """
    llst_ok = [ldct_row for ldct_row in flst_rows
               if CS_STATUS_OK == ldct_row.get("status", CS_STATUS_OK)]

    lf_error_km = _mean(_floats(llst_ok, "error_km"))
    lf_persist_km = _mean(_floats(llst_ok, "persist_km"))
    llst_z2 = _floats(llst_ok, "z2")

    return {
        "lead_min":     fi_lead_min,
        "n":            len(flst_rows),
        "n_ok":         len(llst_ok),
        "frac_ok":      len(llst_ok) / len(flst_rows),
        "lost":         dict(collections.Counter(ldct_row["status"] for ldct_row in flst_rows
                                                 if CS_STATUS_OK != ldct_row.get("status",
                                                                                 CS_STATUS_OK))),
        "error_km":     lf_error_km,
        "persist_km":   lf_persist_km,
        "skill":        1.0 - lf_error_km / lf_persist_km if lf_persist_km > 0.0 else math.nan,
        "n_z2":         len(llst_z2),
        "z2_mean":      _mean(llst_z2),
        "frac_z2_le_1": _fraction(llst_z2, 1.0),
        "frac_z2_le_4": _fraction(llst_z2, 4.0),
    }

# ─────────────────────────────────────────────────────────────────────────────────────
def summarize_association(flst_rows: list) -> dict:
    """
    associação: contagem por kind, distância e score das continuações, NIS
    e a vizinhança das células novas.

    :param flst_rows: linhas do CSV de associação
    :returns: dict kinds {kind: n}, dist_km {p50,p90,p99}, score {...},
              nis_mean, nis_frac_above_p95, n_nis, new_with_candidate,
              new_near (<= CF_NEAR_KM), new_dist_km {...}
    """
    llst_continue = [ldct_row for ldct_row in flst_rows if "continue" == ldct_row["kind"]]
    llst_new = [ldct_row for ldct_row in flst_rows if ldct_row["kind"] in ("new", "split")]
    llst_nis = _floats(llst_continue, "nis")
    llst_new_dist = _floats(llst_new, "dist_km")

    return {
        "kinds":              dict(collections.Counter(ldct_row["kind"] for ldct_row in flst_rows)),
        "dist_km":            _percentiles(_floats(llst_continue, "dist_km")),
        "score":              _percentiles(_floats(llst_continue, "score")),
        "n_nis":              len(llst_nis),
        "nis_mean":           _mean(llst_nis),
        "nis_frac_above_p95": 1.0 - _fraction(llst_nis, CF_CHI2_P95) if llst_nis else math.nan,
        "new_with_candidate": len(llst_new_dist),
        "new_near":           sum(1 for lf_km in llst_new_dist if lf_km <= CF_NEAR_KM),
        "new_dist_km":        _percentiles(llst_new_dist),
    }

# ─────────────────────────────────────────────────────────────────────────────────────
def summarize_tracks(flst_assoc_rows: list) -> dict:
    """
    duração dos tracks, do primeiro ao último frame em que aparecem.

    :param flst_assoc_rows: linhas do CSV de associação
    :returns: dict n_tracks, duration_min {p50,p90,p99},
              frac_verifiable (>= CI_VERIFIABLE_TRACK_MIN)
    """
    ldct_span = {}

    for ldct_row in flst_assoc_rows:
        lf_ts = float(ldct_row["frame_ts"])
        lf_first, lf_last = ldct_span.get(ldct_row["track_id"], (lf_ts, lf_ts))
        ldct_span[ldct_row["track_id"]] = (min(lf_first, lf_ts), max(lf_last, lf_ts))

    llst_minutes = [(lf_last - lf_first) / 60.0 for lf_first, lf_last in ldct_span.values()]

    return {
        "n_tracks":        len(llst_minutes),
        "duration_min":    _percentiles(llst_minutes),
        "frac_verifiable": (sum(1 for lf_min in llst_minutes if lf_min >= CI_VERIFIABLE_TRACK_MIN)
                            / len(llst_minutes)) if llst_minutes else math.nan,
    }

# ─────────────────────────────────────────────────────────────────────────────────────
def summarize_frames(flst_rows: list) -> dict:
    """
    frames: latência, células, fração agrupada e totais do ciclo de vida.

    :param flst_rows: linhas do CSV de frames
    :returns: dict n_frames, latency_s {p50,p90,p99}, cells_mean,
              frac_clustered e totais n_new, n_split, n_missed, n_merged, n_expired
    """
    li_window = sum(int(ldct_row["n_window_events"]) for ldct_row in flst_rows)
    li_clustered = sum(int(ldct_row["n_clustered"]) for ldct_row in flst_rows)

    ldct_summary = {
        "n_frames":       len(flst_rows),
        "latency_s":      _percentiles(_floats(flst_rows, "latency_s")),
        "cells_mean":     _mean(_floats(flst_rows, "n_cells")),
        "frac_clustered": li_clustered / li_window if li_window > 0 else math.nan,
    }

    for ls_key in ("n_new", "n_split", "n_missed", "n_merged", "n_expired"):
        ldct_summary[ls_key] = sum(int(ldct_row[ls_key]) for ldct_row in flst_rows)

    return ldct_summary

# ═════════════════════════════════════════════════════════════════════════════════════
# texto
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def format_report(fdct_summary: dict, flst_header_lines: list | None = None) -> str:
    """
    o resumo como texto para o terminal (ver o cabeçalho do módulo).

    :param fdct_summary: retorno de build
    :param flst_header_lines: linhas no topo (ex.: as constantes trocadas)
    :returns: texto com quebras de linha
    """
    llst_lines = list(flst_header_lines or [])

    llst_lines += _verification_lines(fdct_summary["verification"])
    llst_lines += _association_lines(fdct_summary["association"], fdct_summary["tracks"])
    llst_lines += _frames_lines(fdct_summary["frames"])

    return "\n".join(llst_lines) + "\n"

# ─────────────────────────────────────────────────────────────────────────────────────
def _verification_lines(flst_leads: list) -> list:
    """
    :param flst_leads: summarize_verification
    :returns: linhas de texto
    """
    llst_lines = ["", "verificação (z2 calibrado: E=2.00, <=1: %.0f%%, <=4: %.0f%%)"
                  % (100 * CF_CHI2_P_LE_1, 100 * CF_CHI2_P_LE_4),
                  "  lead    n_ok  conf.  erro_km  persist  skill   E[z2]  z2<=1  z2<=4  perdidas"]

    if not flst_leads:
        return llst_lines + ["  (sem linhas)"]

    for ldct_lead in flst_leads:
        ls_lost = " ".join("%s=%d" % lt for lt in sorted(ldct_lead["lost"].items())) or "-"

        llst_lines.append("  %3d  %6d  %4.0f%%  %7.1f  %7.1f  %5.2f  %6.2f  %4.0f%%  %4.0f%%  %s"
                          % (ldct_lead["lead_min"], ldct_lead["n_ok"], 100 * ldct_lead["frac_ok"],
                             ldct_lead["error_km"], ldct_lead["persist_km"], ldct_lead["skill"],
                             ldct_lead["z2_mean"], 100 * ldct_lead["frac_z2_le_1"],
                             100 * ldct_lead["frac_z2_le_4"], ls_lost))

    return llst_lines

# ─────────────────────────────────────────────────────────────────────────────────────
def _association_lines(fdct_assoc: dict, fdct_tracks: dict) -> list:
    """
    :param fdct_assoc: summarize_association
    :param fdct_tracks: summarize_tracks
    :returns: linhas de texto
    """
    return [
        "",
        "associação",
        "  células por tipo:     %s" % (" ".join("%s=%d" % lt
                                                 for lt in sorted(fdct_assoc["kinds"].items()))
                                        or "-"),
        "  dist. até previsão:   %s km (continue)" % _format_percentiles(fdct_assoc["dist_km"]),
        "  score:                %s" % _format_percentiles(fdct_assoc["score"], "%.2f"),
        "  NIS:                  média %.2f (ref. %.2f), acima de %.2f: %.0f%% (ref. 5%%), n=%d"
        % (fdct_assoc["nis_mean"], CF_CHI2_MEAN, CF_CHI2_P95,
           100 * fdct_assoc["nis_frac_above_p95"], fdct_assoc["n_nis"]),
        "  novas c/ candidato:   %d, a até %.0f km: %d; dist. %s km"
        % (fdct_assoc["new_with_candidate"], CF_NEAR_KM, fdct_assoc["new_near"],
           _format_percentiles(fdct_assoc["new_dist_km"])),
        "",
        "tracks",
        "  n=%d, duração %s min, com >= %d min: %.0f%%"
        % (fdct_tracks["n_tracks"], _format_percentiles(fdct_tracks["duration_min"]),
           CI_VERIFIABLE_TRACK_MIN, 100 * fdct_tracks["frac_verifiable"]),
    ]

# ─────────────────────────────────────────────────────────────────────────────────────
def _frames_lines(fdct_frames: dict) -> list:
    """
    :param fdct_frames: summarize_frames
    :returns: linhas de texto
    """
    return [
        "",
        "frames",
        "  n=%d, latência %s s, células/frame %.1f, raios agrupados %.0f%%"
        % (fdct_frames["n_frames"], _format_percentiles(fdct_frames["latency_s"]),
           fdct_frames["cells_mean"], 100 * fdct_frames["frac_clustered"]),
        "  ciclo de vida: novos=%d divisões=%d perdidos=%d fusões=%d expirados=%d"
        % (fdct_frames["n_new"], fdct_frames["n_split"], fdct_frames["n_missed"],
           fdct_frames["n_merged"], fdct_frames["n_expired"]),
    ]

# ═════════════════════════════════════════════════════════════════════════════════════
# estatística
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def _floats(flst_rows: list, fs_key: str) -> list:
    """
    :param flst_rows: linhas
    :param fs_key: coluna
    :returns: os valores da coluna como float, sem os vazios
    """
    return [float(ldct_row[fs_key]) for ldct_row in flst_rows
            if ldct_row.get(fs_key) not in (None, "")]

# ─────────────────────────────────────────────────────────────────────────────────────
def _mean(flst_values: list) -> float:
    """
    :param flst_values: números
    :returns: média; nan se vazia
    """
    return sum(flst_values) / len(flst_values) if flst_values else math.nan

# ─────────────────────────────────────────────────────────────────────────────────────
def _fraction(flst_values: list, ff_limit: float) -> float:
    """
    :param flst_values: números
    :param ff_limit: limite
    :returns: fração dos valores <= ff_limit; nan se vazia
    """
    if not flst_values:
        return math.nan

    return sum(1 for lf_value in flst_values if lf_value <= ff_limit) / len(flst_values)

# ─────────────────────────────────────────────────────────────────────────────────────
def _percentiles(flst_values: list) -> dict:
    """
    :param flst_values: números
    :returns: {"p50": ..., "p90": ..., "p99": ...}; nan se vazia
    """
    if not flst_values:
        return {"p%d" % li_p: math.nan for li_p in CT_PERCENTILES}

    return {"p%d" % li_p: float(np.percentile(flst_values, li_p)) for li_p in CT_PERCENTILES}

# ─────────────────────────────────────────────────────────────────────────────────────
def _format_percentiles(fdct_percentiles: dict, fs_format: str = "%.1f") -> str:
    """
    :param fdct_percentiles: retorno de _percentiles
    :param fs_format: formato de cada número
    :returns: "p50 a / p90 b / p99 c"
    """
    return " / ".join(("%s " + fs_format) % (ls_key, lf_value)
                      for ls_key, lf_value in fdct_percentiles.items())

# < the end >──────────────────────────────────────────────────────────────────────────
