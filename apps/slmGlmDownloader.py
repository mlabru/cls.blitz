# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

Simple Lightning Model
slmGlmDownloader
baixa os flashes do GLM (Geostationary Lightning Mapper, GOES-19) do bucket
público da NOAA na AWS e os acrescenta ao CSV acumulado
(data/raw/slm_lightning.csv), que guarda só a última hora.  Alternativa ao
slmDownloader (STSC da REDEMET): mesmo CSV, mesmo contrato com o tracker e o
mapper.

o que o GLM entrega: um arquivo NetCDF-4 (HDF5) a cada 20 s, com os flashes
(raios totais: nuvem-solo e intranuvem, sem distinção) do disco inteiro do
satélite.  Cada linha do CSV é um flash: o instante do 1o evento óptico e o
centroide.  Diferente do STSC, é uma fonte contínua de descargas: o tracker
deve usar a janela em minutos (slmTracker.CI_WINDOW_IMAGES = 0).

fluxo (download_glm):
  1) lista no bucket os arquivos da última CI_MAX_HOURS (list_glm_files) e
     separa os que ainda não foram lidos (_select_new_files, com o registro
     de arquivos lidos ao lado do CSV, CS_SEEN_FILE);
  2) baixa e decodifica cada um (read_glm_flashes): só flashes com
     flash_quality_flag = 0 e dentro do domínio (CLST_DOMAIN);
  3) lê o CSV existente, mantém a janela de retenção e tira das linhas novas
     as que já estavam lá (_merge_rows);
  4) grava o CSV de forma atômica (slm_ioutil.atomic_write_csv) e só DEPOIS
     o registro de arquivos lidos;
  5) com --archive-dir, acrescenta as linhas novas ao arquivo diário
     (slm_archive, para o slmReplay).  Falha aqui só gera aviso.

formato do CSV (contrato com slmTracker.load_csv e slmMapper.le_input):
    timestamp,lat,lng      (timestamp = epoch UTC em segundos, com milissegundos)

    python3 apps/slmGlmDownloader.py -f slm_lightning.csv -s ./data/raw -a ./data/archive

códigos de retorno: 0 ok (inclusive sem arquivo novo), 1 falha (bucket
inacessível, erro de gravação do CSV).  Um arquivo que falhe no download ou
na leitura é pulado com aviso e tentado de novo na rodada seguinte.

2026.nov  mlabru  initial version (Linux/Python)
"""
# < imports >──────────────────────────────────────────────────────────────────────────

# python library
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import io
import json
import logging
import os
import re
import sys
import time
import xml.etree.ElementTree as ET

# h5py: o NetCDF-4 do GLM é um HDF5 (não precisa da libnetcdf)
import h5py

# numPy
import numpy as np

# requests
import requests

# locals
import work.slm_archive as arc
import work.slm_ioutil as ioutil
import work.slm_mapdefs as mdf

# < constants >────────────────────────────────────────────────────────────────────────

# diretório e nome padrão do CSV acumulado (os mesmos do slmDownloader)
CS_OUTPUT_DIR = "./data/raw"
CS_OUTPUT_FILE = "slm_lightning.csv"

# registro dos arquivos do GLM já lidos, no diretório do CSV.  É só uma
# economia de download: perdido ou corrompido, os arquivos da última hora são
# relidos e a deduplicação (_merge_rows) tira as repetições
CS_SEEN_FILE = ".slm_glm_seen.json"

# bucket público da NOAA (GOES-19 = GOES-Leste, que cobre o Brasil) e produto
CS_BUCKET_URL = "https://noaa-goes19.s3.amazonaws.com"
CS_PRODUCT = "GLM-L2-LCFA"

# janela de retenção do CSV (horas)
CI_MAX_HOURS = 1

# no máximo isto de arquivos por rodada (1 h = 180 arquivos de 20 s).  A 1a
# rodada, ou a volta de uma parada longa, baixa a hora inteira
CI_MAX_FILES_PER_RUN = 200

# downloads simultâneos: um arquivo tem ~350 kB e a latência domina; em
# série, a hora inteira (1a rodada) levava mais de 2 min
CI_HTTP_WORKERS = 8

# timeout das chamadas HTTP (segundos) e status de sucesso
CI_HTTP_TIMEOUT = 30
CI_HTTP_OK = 200

# domínio dos flashes guardados: o mapa fixo do slmRender
# ([lng_min, lng_max, lat_min, lat_max]: continente + ilhas oceânicas)
CLST_DOMAIN = mdf.CLST_BRAZIL_EXTENT

# flash_quality_flag "bom" (os outros valores marcam flashes degradados)
CI_GOOD_QUALITY = 0

# casas decimais gravadas: ms no instante, ~10 m na posição
CI_TS_DECIMALS = 3
CI_DEG_DECIMALS = 4

# cabeçalho do CSV (ordem das colunas no arquivo)
CLST_HEAD = ["timestamp", "lat", "lng"]

# instante de início no nome do arquivo: _sAAAADDDHHMMSSd (d = décimo de segundo)
CRE_START = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d)_")

# namespace do XML de listagem do S3
CS_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

# códigos de retorno do programa
CI_RC_OK = 0
CI_RC_FAIL = 1

# < logging >──────────────────────────────────────────────────────────────────────────

M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────────────
def arg_parse(flst_argv: list | None = None) -> argparse.Namespace:
    """
    lê os argumentos da linha de comando (os mesmos do slmDownloader).

    :param flst_argv: argumentos (None = sys.argv[1:])
    :returns: namespace com fname, saida e archive_dir (None = não arquiva)
    """
    l_parser = argparse.ArgumentParser(description="Simple Lightning Model download (GOES-19 GLM).")

    l_parser.add_argument("-f", "--fname",
                          help="Output file.",
                          dest="fname", type=str, default=CS_OUTPUT_FILE)
    l_parser.add_argument("-s", "--saida",
                          help="Output folder.",
                          dest="saida", type=str, default=CS_OUTPUT_DIR)
    l_parser.add_argument("-a", "--archive-dir",
                          help="Daily archive folder of new flashes (for slmReplay).",
                          dest="archive_dir", type=str, default=None)

    return l_parser.parse_args(flst_argv)

# ─────────────────────────────────────────────────────────────────────────────────────
def download_glm(f_args: argparse.Namespace, ff_now_ts: float | None = None) -> bool:
    """
    uma rodada: lista, baixa, decodifica e grava (ver o fluxo no cabeçalho).

    :param f_args: argumentos (fname, saida, archive_dir)
    :param ff_now_ts: instante da rodada (None = agora; os testes passam outro)
    :returns: True se o CSV ficou atualizado (ou não havia arquivo novo)
    """
    lf_now_ts = time.time() if ff_now_ts is None else ff_now_ts
    lf_cutoff_ts = lf_now_ts - CI_MAX_HOURS * 3600.0

    os.makedirs(f_args.saida, exist_ok=True)
    ls_csv_path = os.path.join(f_args.saida, f_args.fname)
    ls_seen_path = os.path.join(f_args.saida, CS_SEEN_FILE)

    # 1) arquivos novos no bucket
    llst_keys = list_glm_files(lf_cutoff_ts, lf_now_ts)

    if llst_keys is None:
        return False

    lset_seen = _load_seen(ls_seen_path)
    llst_new_keys = _select_new_files(llst_keys, lset_seen, lf_cutoff_ts)

    M_LOG.info("%d arquivos GLM na última hora, %d novos.", len(llst_keys), len(llst_new_keys))

    # 2) flashes dos arquivos novos
    llst_new_rows, llst_read_keys = _fetch_flashes(llst_new_keys, lf_cutoff_ts)

    # 3) junta com o CSV, na janela de retenção
    llst_kept_rows = _recent_rows(_load_csv(ls_csv_path), lf_cutoff_ts)
    llst_unique_new_rows = _merge_rows(llst_kept_rows, llst_new_rows)

    M_LOG.info("flashes novos: %d (após deduplicação: %d).",
               len(llst_new_rows), len(llst_unique_new_rows))

    # 4) CSV primeiro, registro depois: se o processo morrer entre os dois, os
    #    arquivos são relidos e a deduplicação tira as repetições
    if not _write_csv(ls_csv_path, llst_kept_rows + llst_unique_new_rows):
        return False

    _save_seen(ls_seen_path, lset_seen | set(llst_read_keys), lf_cutoff_ts)

    # 5) arquivo diário (não afeta o resultado)
    if f_args.archive_dir:
        _archive_new_rows(f_args.archive_dir, llst_unique_new_rows, lf_now_ts)

    return True

# ═════════════════════════════════════════════════════════════════════════════════════
# passo 1: listagem do bucket
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def list_glm_files(ff_since_ts: float, ff_now_ts: float) -> list | None:
    """
    os arquivos do produto com início entre ff_since_ts e ff_now_ts.
    o bucket é organizado por hora (PRODUTO/AAAA/DDD/HH/), então lista cada
    hora do intervalo.

    :param ff_since_ts: início do intervalo (epoch UTC)
    :param ff_now_ts: fim do intervalo (epoch UTC)
    :returns: chaves (caminhos no bucket) em ordem de início; None se o
              bucket não respondeu (já logado)
    """
    llst_keys = []

    for ls_prefix in _hour_prefixes(ff_since_ts, ff_now_ts):
        llst_hour_keys = _list_prefix(ls_prefix)

        if llst_hour_keys is None:
            return None

        llst_keys.extend(llst_hour_keys)

    return sorted((ls_key for ls_key in llst_keys
                   if ff_since_ts <= (file_start_ts(ls_key) or -1.0) <= ff_now_ts),
                  key=file_start_ts)

# ─────────────────────────────────────────────────────────────────────────────────────
def _hour_prefixes(ff_since_ts: float, ff_now_ts: float) -> list:
    """
    os prefixos horários do bucket que cobrem o intervalo.

    :param ff_since_ts: início (epoch UTC)
    :param ff_now_ts: fim (epoch UTC)
    :returns: ["GLM-L2-LCFA/AAAA/DDD/HH/", ...], do mais antigo ao mais novo
    """
    ldt_hour = datetime.datetime.fromtimestamp(ff_since_ts, datetime.timezone.utc).replace(
                   minute=0, second=0, microsecond=0)
    ldt_end = datetime.datetime.fromtimestamp(ff_now_ts, datetime.timezone.utc)

    llst_prefixes = []

    while ldt_hour <= ldt_end:
        llst_prefixes.append(ldt_hour.strftime(CS_PRODUCT + "/%Y/%j/%H/"))
        ldt_hour += datetime.timedelta(hours=1)

    return llst_prefixes

# ─────────────────────────────────────────────────────────────────────────────────────
def _list_prefix(fs_prefix: str) -> list | None:
    """
    as chaves de um prefixo (ListObjectsV2, seguindo a paginação).

    :param fs_prefix: prefixo no bucket
    :returns: lista de chaves; None em erro (já logado)
    """
    llst_keys = []
    ldct_params = {"list-type": "2", "prefix": fs_prefix}

    while True:
        lby_body = _http_get(CS_BUCKET_URL, ldct_params)

        if lby_body is None:
            return None

        try:
            llst_page_keys, ls_next_token = parse_listing(lby_body)

        except ET.ParseError as l_err:
            M_LOG.error("listagem do bucket ilegível (%s): %s", fs_prefix, str(l_err))
            return None

        llst_keys.extend(llst_page_keys)

        if not ls_next_token:
            return llst_keys

        ldct_params["continuation-token"] = ls_next_token

# ─────────────────────────────────────────────────────────────────────────────────────
def parse_listing(fby_xml: bytes) -> tuple:
    """
    decodifica uma página do ListObjectsV2.

    :param fby_xml: corpo da resposta
    :returns: (chaves, token da próxima página ou None)
    :raises ET.ParseError: XML malformado
    """
    l_root = ET.fromstring(fby_xml)

    llst_keys = [l_key.text for l_key in l_root.iter(CS_S3_NS + "Key") if l_key.text]

    ls_truncated = l_root.findtext(CS_S3_NS + "IsTruncated", default="false")
    ls_next_token = l_root.findtext(CS_S3_NS + "NextContinuationToken")

    return llst_keys, (ls_next_token if "true" == ls_truncated else None)

# ─────────────────────────────────────────────────────────────────────────────────────
def file_start_ts(fs_key: str) -> float | None:
    """
    instante de início de um arquivo, pelo nome (_sAAAADDDHHMMSSd_).

    :param fs_key: chave ou nome do arquivo
    :returns: epoch UTC; None se o nome não tiver o padrão
    """
    l_match = CRE_START.search(fs_key)

    if l_match is None:
        return None

    li_year, li_doy, li_hour, li_min, li_sec, li_tenth = (int(ls) for ls in l_match.groups())

    ldt_start = (datetime.datetime(li_year, 1, 1, tzinfo=datetime.timezone.utc) +
                 datetime.timedelta(days=li_doy - 1, hours=li_hour, minutes=li_min,
                                    seconds=li_sec + li_tenth / 10.0))

    return ldt_start.timestamp()

# ─────────────────────────────────────────────────────────────────────────────────────
def _select_new_files(flst_keys: list, fset_seen: set, ff_cutoff_ts: float) -> list:
    """
    os arquivos ainda não lidos, dentro da retenção, até CI_MAX_FILES_PER_RUN
    (os mais NOVOS, se passar do limite: o tracker só usa a janela recente).

    :param flst_keys: chaves em ordem de início
    :param fset_seen: nomes já lidos
    :param ff_cutoff_ts: início da retenção (epoch UTC)
    :returns: chaves a baixar, em ordem de início
    """
    llst_new_keys = [ls_key for ls_key in flst_keys
                     if os.path.basename(ls_key) not in fset_seen
                     and (file_start_ts(ls_key) or -1.0) >= ff_cutoff_ts]

    return llst_new_keys[-CI_MAX_FILES_PER_RUN:]

# ═════════════════════════════════════════════════════════════════════════════════════
# passo 2: download e leitura dos arquivos
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def _fetch_flashes(flst_keys: list, ff_cutoff_ts: float) -> tuple:
    """
    baixa (CI_HTTP_WORKERS em paralelo) e decodifica os arquivos.  Arquivo
    que falha é pulado (não entra no registro de lidos, então é tentado de
    novo na próxima rodada).

    :param flst_keys: chaves a baixar
    :param ff_cutoff_ts: início da retenção (flashes anteriores são descartados)
    :returns: (linhas novas, chaves lidas com sucesso -- só o nome do arquivo)
    """
    llst_rows = []
    llst_read_keys = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=CI_HTTP_WORKERS) as l_pool:
        llst_results = list(l_pool.map(_fetch_file, flst_keys))

    for ls_key, llst_file_rows in zip(flst_keys, llst_results):
        if llst_file_rows is None:
            continue

        llst_rows.extend(ldct_row for ldct_row in llst_file_rows
                         if ldct_row["timestamp"] >= ff_cutoff_ts)
        llst_read_keys.append(os.path.basename(ls_key))

    return llst_rows, llst_read_keys

# ─────────────────────────────────────────────────────────────────────────────────────
def _fetch_file(fs_key: str) -> list | None:
    """
    baixa e decodifica um arquivo (roda numa thread do _fetch_flashes).

    :param fs_key: chave no bucket
    :returns: linhas do arquivo; None em erro (já logado)
    """
    lby_content = _http_get(CS_BUCKET_URL + "/" + fs_key)

    if lby_content is None:
        return None

    try:
        return read_glm_flashes(io.BytesIO(lby_content))

    except (OSError, KeyError, ValueError) as l_err:
        M_LOG.warning("arquivo GLM ilegível '%s': %s", fs_key, str(l_err))
        return None

# ─────────────────────────────────────────────────────────────────────────────────────
def read_glm_flashes(f_source) -> list:
    """
    os flashes de um arquivo LCFA: instante do 1o evento e centroide, só os
    de qualidade boa e dentro de CLST_DOMAIN.

    :param f_source: caminho ou objeto tipo arquivo (bytes) do NetCDF-4
    :returns: lista de dicts {timestamp, lat, lng}, em ordem de instante
    :raises OSError: não é um HDF5 válido
    :raises KeyError: falta uma variável
    :raises ValueError: unidade de tempo fora do padrão
    """
    with h5py.File(f_source, "r") as l_file:
        lnp_lat = unpack(l_file["flash_lat"])
        lnp_lng = unpack(l_file["flash_lon"])
        lnp_offset_s = unpack(l_file["flash_time_offset_of_first_event"])
        lnp_quality = unpack(l_file["flash_quality_flag"])
        lf_base_ts = _time_base_ts(l_file["flash_time_offset_of_first_event"])

    lf_lng_min, lf_lng_max, lf_lat_min, lf_lat_max = CLST_DOMAIN

    lnp_keep = ((lnp_quality == CI_GOOD_QUALITY) &
                (lnp_lat >= lf_lat_min) & (lnp_lat <= lf_lat_max) &
                (lnp_lng >= lf_lng_min) & (lnp_lng <= lf_lng_max) &
                np.isfinite(lnp_offset_s))

    llst_rows = [{"timestamp": round(lf_base_ts + float(lf_offset), CI_TS_DECIMALS),
                  "lat": round(float(lf_lat), CI_DEG_DECIMALS),
                  "lng": round(float(lf_lng), CI_DEG_DECIMALS)}
                 for lf_offset, lf_lat, lf_lng in zip(lnp_offset_s[lnp_keep],
                                                     lnp_lat[lnp_keep],
                                                     lnp_lng[lnp_keep])]

    return sorted(llst_rows, key=lambda ldct_row: ldct_row["timestamp"])

# ─────────────────────────────────────────────────────────────────────────────────────
def unpack(f_variable) -> np.ndarray:
    """
    valores físicos de uma variável NetCDF empacotada, como a libnetcdf faria:
    _Unsigned = "true" reinterpreta o inteiro sem sinal; _FillValue vira NaN;
    depois valor · scale_factor + add_offset.

    [doc] lido com h5py, os atributos CF não são aplicados sozinhos: o
    flash_time_offset_of_first_event vem como int16 e só vira segundos aqui.

    :param f_variable: dataset h5py
    :returns: array float64
    """
    lnp_raw = f_variable[()]
    ldct_attrs = f_variable.attrs

    lnp_fill = None

    if "_FillValue" in ldct_attrs:
        lnp_fill = lnp_raw == np.asarray(ldct_attrs["_FillValue"]).astype(lnp_raw.dtype).item()

    if _attr_text(ldct_attrs.get("_Unsigned")) == "true" and lnp_raw.dtype.kind == "i":
        lnp_raw = lnp_raw.view(np.dtype("u%d" % lnp_raw.dtype.itemsize))

    lnp_values = lnp_raw.astype(np.float64)
    lnp_values = lnp_values * _attr_float(ldct_attrs.get("scale_factor"), 1.0)
    lnp_values = lnp_values + _attr_float(ldct_attrs.get("add_offset"), 0.0)

    if lnp_fill is not None:
        lnp_values[lnp_fill] = np.nan

    return lnp_values

# ─────────────────────────────────────────────────────────────────────────────────────
def _time_base_ts(f_variable) -> float:
    """
    o instante a que os offsets se referem, pela unidade CF
    ("seconds since AAAA-MM-DD HH:MM:SS.fff", UTC).

    :param f_variable: dataset h5py com o atributo units
    :returns: epoch UTC
    :raises ValueError: unidade fora do padrão
    """
    ls_units = _attr_text(f_variable.attrs.get("units"))
    ls_prefix = "seconds since "

    if not ls_units.startswith(ls_prefix):
        raise ValueError("unidade de tempo inesperada: %r" % ls_units)

    ldt_base = datetime.datetime.fromisoformat(ls_units[len(ls_prefix):].strip())

    return ldt_base.replace(tzinfo=datetime.timezone.utc).timestamp()

# ─────────────────────────────────────────────────────────────────────────────────────
def _attr_text(f_value) -> str:
    """
    :param f_value: atributo HDF5 (bytes, str ou None)
    :returns: texto (vazio se None)
    """
    if f_value is None:
        return ""

    if isinstance(f_value, bytes):
        return f_value.decode("utf-8", "replace")

    return str(f_value)

# ─────────────────────────────────────────────────────────────────────────────────────
def _attr_float(f_value, ff_default: float) -> float:
    """
    :param f_value: atributo HDF5 numérico (escalar ou array de 1 elemento) ou None
    :param ff_default: valor se ausente
    :returns: o número
    """
    if f_value is None:
        return ff_default

    return float(np.asarray(f_value).reshape(-1)[0])

# ═════════════════════════════════════════════════════════════════════════════════════
# passo 3: CSV existente e deduplicação
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def _load_csv(fs_path: str) -> list:
    """
    lê o CSV acumulado, se existir.

    :param fs_path: caminho do CSV
    :returns: lista de dicts (valores como str); vazia se não existe ou não
              pôde ser lido (aviso no log)
    """
    if not os.path.exists(fs_path):
        return []

    try:
        with open(fs_path, "r", encoding="utf-8", newline="") as l_file:
            return list(csv.DictReader(l_file))

    except (OSError, csv.Error) as l_err:
        M_LOG.warning("não foi possível ler '%s': %s", fs_path, str(l_err))
        return []

# ─────────────────────────────────────────────────────────────────────────────────────
def _recent_rows(flst_rows: list, ff_cutoff_ts: float) -> list:
    """
    as linhas com timestamp >= corte; malformadas são descartadas.

    :param flst_rows: linhas do CSV
    :param ff_cutoff_ts: início da retenção (epoch UTC)
    :returns: linhas na janela, na ordem original
    """
    llst_kept_rows = []

    for ldct_row in flst_rows:
        try:
            if float(ldct_row["timestamp"]) >= ff_cutoff_ts:
                llst_kept_rows.append(ldct_row)

        except (KeyError, TypeError, ValueError):
            continue

    return llst_kept_rows

# ─────────────────────────────────────────────────────────────────────────────────────
def _merge_rows(flst_kept_rows: list, flst_new_rows: list) -> list:
    """
    as linhas novas que ainda não estão no CSV (nem repetidas entre si).

    [doc] dois flashes distintos com o mesmo instante (ms) e a mesma posição
    (~10 m) não acontecem; a mesma linha repetida vem de um arquivo relido
    (registro de lidos perdido, ou o processo morreu entre o CSV e o registro).

    :param flst_kept_rows: linhas já gravadas (valores str)
    :param flst_new_rows: linhas dos arquivos novos (valores float)
    :returns: linhas novas únicas, na ordem original
    """
    lset_known = {_row_key(ldct_row) for ldct_row in flst_kept_rows}

    llst_unique_rows = []

    for ldct_row in flst_new_rows:
        lt_key = _row_key(ldct_row)

        if lt_key in lset_known:
            continue

        lset_known.add(lt_key)
        llst_unique_rows.append(ldct_row)

    return llst_unique_rows

# ─────────────────────────────────────────────────────────────────────────────────────
def _row_key(fdct_row: dict) -> tuple:
    """
    chave de deduplicação: os valores arredondados como no CSV, para a linha
    lida (str) e a nova (float) compararem igual.

    :param fdct_row: linha com timestamp, lat e lng
    :returns: (timestamp, lat, lng) arredondados
    """
    return (round(float(fdct_row["timestamp"]), CI_TS_DECIMALS),
            round(float(fdct_row["lat"]), CI_DEG_DECIMALS),
            round(float(fdct_row["lng"]), CI_DEG_DECIMALS))

# ═════════════════════════════════════════════════════════════════════════════════════
# passo 4: gravação
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def _write_csv(fs_csv_path: str, flst_rows: list) -> bool:
    """
    grava o CSV completo de forma atômica, em ordem de instante.

    :param fs_csv_path: caminho final do CSV
    :param flst_rows: todas as linhas (mantidas + novas)
    :returns: True se gravou, False em erro (já logado)
    """
    llst_sorted_rows = sorted(flst_rows, key=lambda ldct_row: float(ldct_row["timestamp"]))

    try:
        ioutil.atomic_write_csv(fs_csv_path, CLST_HEAD, llst_sorted_rows)

    except OSError as l_err:
        M_LOG.error("falha ao gravar '%s': %s", fs_csv_path, str(l_err))
        return False

    M_LOG.info("total de registros persistidos em '%s': %d.", fs_csv_path, len(llst_sorted_rows))

    return True

# ─────────────────────────────────────────────────────────────────────────────────────
def _load_seen(fs_path: str) -> set:
    """
    os nomes dos arquivos já lidos.

    :param fs_path: registro (CS_SEEN_FILE)
    :returns: conjunto de nomes; vazio se não existe ou está corrompido
    """
    try:
        with open(fs_path, "r", encoding="utf-8") as l_file:
            return {ls_name for ls_name in json.load(l_file) if isinstance(ls_name, str)}

    except FileNotFoundError:
        return set()

    except (OSError, ValueError, TypeError) as l_err:
        M_LOG.warning("registro de arquivos lidos ilegível ('%s'): %s -- relendo a última hora.",
                      fs_path, str(l_err))
        return set()

# ─────────────────────────────────────────────────────────────────────────────────────
def _save_seen(fs_path: str, fset_seen: set, ff_cutoff_ts: float) -> None:
    """
    grava (atomicamente) o registro de arquivos lidos, só os da retenção.
    Falha aqui só gera aviso: a próxima rodada relê e deduplica.

    :param fs_path: registro (CS_SEEN_FILE)
    :param fset_seen: nomes lidos
    :param ff_cutoff_ts: início da retenção (nomes mais antigos saem)
    """
    llst_names = sorted(ls_name for ls_name in fset_seen
                        if (file_start_ts(ls_name) or -1.0) >= ff_cutoff_ts)

    try:
        ioutil.atomic_write_json(fs_path, llst_names)

    except OSError as l_err:
        M_LOG.warning("falha ao gravar '%s': %s", fs_path, str(l_err))

# ─────────────────────────────────────────────────────────────────────────────────────
def _archive_new_rows(fs_archive_dir: str, flst_new_rows: list, ff_now_ts: float) -> None:
    """
    acrescenta as linhas novas ao arquivo diário e apaga os dias vencidos
    (slm_archive).  Erro vira aviso: o pipeline não depende do archive.

    :param fs_archive_dir: diretório do archive
    :param flst_new_rows: linhas novas desta rodada
    :param ff_now_ts: instante da rodada (vai para fetched_ts)
    """
    try:
        arc.append_rows(fs_archive_dir, flst_new_rows, ff_now_ts)
        li_removed = arc.purge_old(fs_archive_dir, ff_now_ts)

    except OSError as l_err:
        M_LOG.warning("falha ao arquivar em '%s': %s", fs_archive_dir, str(l_err))
        return

    M_LOG.info("arquivados %d registros novos em '%s'.", len(flst_new_rows), fs_archive_dir)

    if li_removed > 0:
        M_LOG.info("%d arquivos vencidos apagados do archive.", li_removed)

# ═════════════════════════════════════════════════════════════════════════════════════
# HTTP
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def _http_get(fs_url: str, fdct_params: dict | None = None) -> bytes | None:
    """
    GET simples (o bucket é público, sem autenticação).

    :param fs_url: URL
    :param fdct_params: parâmetros da query
    :returns: corpo da resposta; None em erro (já logado)
    """
    try:
        l_response = requests.get(fs_url, params=fdct_params, timeout=CI_HTTP_TIMEOUT)

    except requests.RequestException as l_err:
        M_LOG.error("falha de acesso a '%s': %s", fs_url, str(l_err))
        return None

    if CI_HTTP_OK != l_response.status_code:
        M_LOG.error("'%s' respondeu HTTP %d.", fs_url, l_response.status_code)
        return None

    return l_response.content

# ─────────────────────────────────────────────────────────────────────────────────────
def main(flst_argv: list | None = None) -> int:
    """
    ponto de entrada: lê os argumentos e baixa.

    :param flst_argv: argumentos (None = sys.argv[1:])
    :returns: CI_RC_OK ou CI_RC_FAIL
    """
    l_args = arg_parse(flst_argv)

    if not download_glm(l_args):
        M_LOG.error("execução encerrada com falha.")
        return CI_RC_FAIL

    return CI_RC_OK

# ─────────────────────────────────────────────────────────────────────────────────────
# this is the bootstrap process

if "__main__" == __name__:
    logging.basicConfig(datefmt="%Y-%m-%d %H:%M:%S",
                        format="[%(asctime)s] [%(levelname)s] %(message)s",
                        level=logging.INFO)

    sys.exit(main())

# < the end >──────────────────────────────────────────────────────────────────────────
