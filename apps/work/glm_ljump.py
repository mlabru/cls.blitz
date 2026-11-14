# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

Simple Lightning Model
glm_ljump
taxa de raios e salto de raios (lightning jump) de uma célula, a partir dos
instantes dos flashes do cluster.

o salto de raios é um aumento brusco da taxa total de raios de uma célula;
na literatura, precede granizo, rajadas e tornados em ~10-30 min.  Algoritmo
"2σ" (Schultz et al., 2009, J. Appl. Meteor. Climatol. 48; 2011, Wea.
Forecasting 26), adaptado:
  1) a taxa (flashes/min) em intervalos de CF_LJ_BIN_MIN (2 min), terminando
     no instante do frame (rate_series);
  2) DFRDT = variação da taxa de um intervalo para o seguinte (flashes/min²);
  3) num intervalo i, há salto se a taxa >= CF_LJ_MIN_RATE (ativação) e o
     DFRDT de i passa de CF_LJ_SIGMA desvios-padrão dos CI_LJ_HISTORY DFRDT
     anteriores (10 min);
  4) só valem intervalos com a célula ativa (taxa > 0) em todo o histórico
     usado: uma célula que acaba de nascer não "salta" do zero.
o frame avalia todos os intervalos que terminam depois do frame anterior
(since_ts): com o cron de 5 min e intervalos de 2 min, o último intervalo
sozinho deixaria passar saltos entre dois frames.

só faz sentido com uma fonte de descargas (GLM): no STSC cada evento é uma
célula de grade por imagem, e a "taxa" mede área, não raios.

[doc] os limiares de Schultz vieram de redes de solo de raios totais (LMA);
o GLM detecta menos flashes (eficiência ~70-90%, menor de dia) e os agrupa de
outro jeito, então CF_LJ_MIN_RATE e CF_LJ_SIGMA precisam de calibração (bltReplay
--set glm_ljump.CF_LJ_MIN_RATE=...).

2026.nov  mlabru  rate_recent: média dos últimos CI_LJ_RECENT_BINS intervalos (a
                  severidade do render usa esta, mais estável que a do último)
2026.nov  mlabru  initial version (Linux/Python)
"""
# < imports >──────────────────────────────────────────────────────────────────────────

# python library
from __future__ import annotations

import dataclasses
import math

# < constants >────────────────────────────────────────────────────────────────────────

# largura de cada intervalo da série de taxa (minutos)
CF_LJ_BIN_MIN = 2.0

# quantos DFRDT anteriores dão o desvio-padrão (5 x 2 min = 10 min, Schultz)
CI_LJ_HISTORY = 5

# o DFRDT precisa passar de CF_LJ_SIGMA desvios-padrão do histórico
CF_LJ_SIGMA = 2.0

# taxa mínima (flashes/min) no intervalo para valer salto (ativação de Schultz)
CF_LJ_MIN_RATE = 10.0

# intervalos da taxa recente (3 x 2 min = 6 min): base da severidade; a do
# último intervalo sozinho (2 min) oscila demais de um frame para o outro
CI_LJ_RECENT_BINS = 3

# < TLightningJump >───────────────────────────────────────────────────────────────────

@dataclasses.dataclass(slots=True)
class TLightningJump:
    """
    taxa de raios e salto de uma célula num frame.
    """
    # taxa no intervalo mais recente (flashes/min)
    rate_now: float = 0.0
    # taxa média dos últimos CI_LJ_RECENT_BINS intervalos (flashes/min)
    rate_recent: float = 0.0
    # variação da taxa no intervalo mais recente (flashes/min²)
    dfrdt_now: float = 0.0
    # maior DFRDT / desvio-padrão entre os intervalos avaliados (0 = nenhum avaliável)
    sigma_level: float = 0.0
    # houve salto em algum intervalo avaliado
    jump: bool = False
    # fim do intervalo do salto (epoch UTC; 0 = sem salto)
    jump_ts: float = 0.0

# ─────────────────────────────────────────────────────────────────────────────────────
def evaluate(flst_ts: list, ff_end_ts: float, ff_window_min: float,
             ff_since_ts: float | None = None) -> TLightningJump:
    """
    taxa e salto de raios de uma célula (ver o algoritmo no cabeçalho).

    :param flst_ts: instantes dos flashes do cluster (epoch UTC, qualquer ordem)
    :param ff_end_ts: fim da série: o instante do frame (flash mais recente)
    :param ff_window_min: comprimento da série (a janela do tracker, minutos)
    :param ff_since_ts: instante do frame anterior: só intervalos que terminam
                        depois dele são avaliados (None = só o último)
    :returns: TLightningJump
    """
    llst_rates = rate_series(flst_ts, ff_end_ts, ff_window_min)
    llst_dfrdt = _dfrdt(llst_rates)

    llst_recent = llst_rates[-CI_LJ_RECENT_BINS:]

    l_result = TLightningJump(rate_now=llst_rates[-1] if llst_rates else 0.0,
                              rate_recent=sum(llst_recent) / len(llst_recent) if llst_recent else 0.0,
                              dfrdt_now=llst_dfrdt[-1] if llst_dfrdt else 0.0)

    for li_bin in _bins_to_evaluate(len(llst_rates), ff_end_ts, ff_since_ts):
        lf_level = _sigma_level(llst_rates, llst_dfrdt, li_bin)

        if lf_level is None:
            continue

        l_result.sigma_level = max(l_result.sigma_level, lf_level)

        if lf_level > CF_LJ_SIGMA and llst_rates[li_bin] >= CF_LJ_MIN_RATE:
            l_result.jump = True
            l_result.jump_ts = _bin_end_ts(ff_end_ts, len(llst_rates), li_bin)

    return l_result

# ─────────────────────────────────────────────────────────────────────────────────────
def rate_series(flst_ts: list, ff_end_ts: float, ff_window_min: float) -> list:
    """
    taxa de flashes (por minuto) em intervalos de CF_LJ_BIN_MIN que terminam
    em ff_end_ts e cobrem ff_window_min (o intervalo mais antigo primeiro).
    Flashes fora da série são ignorados; o fim é inclusivo.

    :param flst_ts: instantes dos flashes
    :param ff_end_ts: fim da série (epoch UTC)
    :param ff_window_min: comprimento da série (minutos)
    :returns: lista de taxas, uma por intervalo
    """
    li_bins = max(1, int(round(ff_window_min / CF_LJ_BIN_MIN)))
    lf_bin_s = CF_LJ_BIN_MIN * 60.0
    lf_start_ts = ff_end_ts - li_bins * lf_bin_s

    llst_counts = [0] * li_bins

    for lf_ts in flst_ts:
        if lf_ts <= lf_start_ts or lf_ts > ff_end_ts:
            continue

        # (início, fim]: um flash exatamente no fim de um intervalo é dele
        li_bin = min(li_bins - 1, int(math.ceil((lf_ts - lf_start_ts) / lf_bin_s)) - 1)
        llst_counts[li_bin] += 1

    return [li_count / CF_LJ_BIN_MIN for li_count in llst_counts]

# ─────────────────────────────────────────────────────────────────────────────────────
def _dfrdt(flst_rates: list) -> list:
    """
    variação da taxa entre intervalos consecutivos (flashes/min²); o
    elemento i é a variação do intervalo i-1 para i (o 0 vale 0).

    :param flst_rates: taxas, do intervalo mais antigo ao mais novo
    :returns: lista do mesmo tamanho
    """
    return [0.0] + [(flst_rates[li] - flst_rates[li - 1]) / CF_LJ_BIN_MIN
                    for li in range(1, len(flst_rates))]

# ─────────────────────────────────────────────────────────────────────────────────────
def _bins_to_evaluate(fi_bins: int, ff_end_ts: float, ff_since_ts: float | None) -> list:
    """
    os intervalos que terminam depois do frame anterior (pelo menos o último).

    :param fi_bins: quantos intervalos a série tem
    :param ff_end_ts: fim da série
    :param ff_since_ts: instante do frame anterior (None = só o último)
    :returns: índices dos intervalos, em ordem
    """
    if ff_since_ts is None:
        return [fi_bins - 1]

    llst_bins = [li_bin for li_bin in range(fi_bins)
                 if _bin_end_ts(ff_end_ts, fi_bins, li_bin) > ff_since_ts]

    return llst_bins or [fi_bins - 1]

# ─────────────────────────────────────────────────────────────────────────────────────
def _bin_end_ts(ff_end_ts: float, fi_bins: int, fi_bin: int) -> float:
    """
    :returns: instante do fim do intervalo fi_bin (epoch UTC)
    """
    return ff_end_ts - (fi_bins - 1 - fi_bin) * CF_LJ_BIN_MIN * 60.0

# ─────────────────────────────────────────────────────────────────────────────────────
def _sigma_level(flst_rates: list, flst_dfrdt: list, fi_bin: int) -> float | None:
    """
    DFRDT do intervalo fi_bin em desvios-padrão dos CI_LJ_HISTORY anteriores.

    :param flst_rates: taxas
    :param flst_dfrdt: variações (_dfrdt)
    :param fi_bin: intervalo avaliado
    :returns: DFRDT / σ; None se não dá para avaliar (histórico curto, célula
              inativa em algum intervalo do histórico ou σ nulo)
    """
    li_first = fi_bin - CI_LJ_HISTORY

    # o DFRDT de li_first precisa do intervalo li_first - 1: série inteira ativa
    if li_first < 1 or min(flst_rates[li_first - 1:fi_bin]) <= 0.0:
        return None

    llst_history = flst_dfrdt[li_first:fi_bin]
    lf_mean = sum(llst_history) / len(llst_history)
    lf_sigma = math.sqrt(sum((lf - lf_mean) ** 2 for lf in llst_history) / len(llst_history))

    if lf_sigma <= 0.0:
        return None

    return flst_dfrdt[fi_bin] / lf_sigma

# < the end >──────────────────────────────────────────────────────────────────────────
