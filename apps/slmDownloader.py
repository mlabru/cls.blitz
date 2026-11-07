# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

Simple Lightning Model
slmDownloader
baixa o snapshot mais recente de raios da API STSC da REDEMET e o acrescenta
ao CSV acumulado (data/raw/slm_lightning.csv), que guarda só a última hora.

fluxo (download_raios):
  1) pede o snapshot à REDEMET (get_redemet_raios): requisição HTTP,
     decodificação do JSON e conferência dos campos;
  2) lê o CSV existente e mantém só as linhas dentro da janela de retenção
     (CI_MAX_HOURS) -- filter_recent_rows;
  3) converte o snapshot em linhas {timestamp, lat, lng}, descartando as
     horas fora da janela -- _snapshot_to_rows;
  4) tira as linhas novas que já estavam no CSV (a API devolve a última
     hora inteira a cada chamada) -- _drop_duplicates;
  5) grava o CSV completo de forma atômica (slm_ioutil.atomic_write_csv),
     porque o tracker e o mapper leem esse arquivo em outro processo.

formato do CSV (contrato com slmTracker.load_csv e slmMapper.le_input):
    timestamp,lat,lng      (timestamp = epoch UTC em segundos, inteiro)

    python3 apps/slmDownloader.py -f slm_lightning.csv -s ./data/raw

códigos de retorno (contrato com o slm_run.sh, que só testa "deu zero?"):
0 ok, 1 falha (sem API key, REDEMET indisponível/resposta inválida, erro de
gravação).  Em falha o CSV anterior fica intacto.

2026.out  mlabru  [fix] resposta sem "anima"/"stsc" e posição sem "la"/"lo"
                  são logadas e tratadas, não mais KeyError não capturado
2026.out  mlabru  refatoração para legibilidade: download_raios e
                  get_redemet_raios divididos em funções de um passo só;
                  escrita atômica via slm_ioutil.atomic_write_csv; alias de
                  slm_defs "df" virou "defs"; argumentos lidos antes da checagem
                  da API key (para --help funcionar sem .env); comentários
                  errados corrigidos; logs de debug removidos
2026.may  mlabru  corrections and refactoring
2022.apr  mlabru  initial version (Linux/Python)
"""
# < imports >──────────────────────────────────────────────────────────────────────────

# python library
from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import os
import sys

# requests
import requests

# locals
import work.slm_defs as defs
import work.slm_ioutil as ioutil

# < constants >────────────────────────────────────────────────────────────────────────

# diretório padrão do CSV acumulado
CS_OUTPUT_DIR = "./data/raw"

# nome padrão do CSV acumulado
CS_OUTPUT_FILE = "slm_lightning.csv"

# janela máxima de dados retidos (horas)
#
# [doc] precisa ficar em sincronia com o parâmetro anima= da URL da REDEMET
# (defs.CS_REDEMET_URL, em slm_defs.py -- anima=60 minutos == CI_MAX_HOURS=1h).
# ver o comentário em slm_defs.py para o efeito de dessincronizar os dois.
CI_MAX_HOURS = 1

# timeout para chamadas HTTP (segundos)
CI_HTTP_TIMEOUT = 30

# status HTTP de sucesso
CI_HTTP_OK = 200

# cabeçalho do CSV (ordem das colunas no arquivo)
CLST_HEAD = ["timestamp", "lat", "lng"]

# códigos de retorno do programa
CI_RC_OK = 0
CI_RC_FAIL = 1

# < logging >──────────────────────────────────────────────────────────────────────────

M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────────────
def arg_parse(flst_argv: list | None = None) -> argparse.Namespace:
    """
    lê os argumentos da linha de comando.

    :param flst_argv: argumentos (None = sys.argv[1:])
    :returns: namespace com fname (nome do CSV) e saida (diretório)
    """
    l_parser = argparse.ArgumentParser(description="Simple Lightning Model download.")

    l_parser.add_argument("-f", "--fname",
                          help="Output file.",
                          dest="fname", type=str, default=CS_OUTPUT_FILE)
    l_parser.add_argument("-s", "--saida",
                          help="Output folder.",
                          dest="saida", type=str, default=CS_OUTPUT_DIR)

    return l_parser.parse_args(flst_argv)

# ═════════════════════════════════════════════════════════════════════════════════════
# orquestrador
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def download_raios(f_args: argparse.Namespace) -> bool:
    """
    baixa o snapshot da REDEMET e atualiza o CSV acumulado, mantendo só os
    registros das últimas CI_MAX_HOURS horas (ver o fluxo no cabeçalho).

    :param f_args: argumentos da linha de comando (usa saida e fname)
    :returns: True em sucesso, False em falha (o CSV anterior fica intacto)
    """
    # 1) snapshot da API
    ldct_snapshot = get_redemet_raios()

    if not _snapshot_is_usable(ldct_snapshot):
        return False

    os.makedirs(f_args.saida, exist_ok=True)
    ls_csv_path = os.path.join(f_args.saida, f_args.fname)

    ldt_cutoff = _retention_cutoff()

    # 2) linhas já gravadas, dentro da janela
    llst_kept_rows = filter_recent_rows(load_existing_data(ls_csv_path), ldt_cutoff)

    M_LOG.info("registros anteriores mantidos (últimas %dh): %d.",
               CI_MAX_HOURS, len(llst_kept_rows))

    # 3) linhas novas, vindas da API
    llst_new_rows = _snapshot_to_rows(ldct_snapshot["anima"], ldct_snapshot["stsc"], ldt_cutoff)

    M_LOG.info("novos registros obtidos da API: %d.", len(llst_new_rows))

    # 4) só o que ainda não estava no CSV
    llst_unique_new_rows = _drop_duplicates(llst_new_rows, llst_kept_rows)

    M_LOG.info("novos registros após deduplicação: %d.", len(llst_unique_new_rows))

    # 5) grava tudo de uma vez
    return _write_csv(ls_csv_path, llst_kept_rows + llst_unique_new_rows)

# ═════════════════════════════════════════════════════════════════════════════════════
# passo 1: snapshot da REDEMET
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def get_redemet_raios() -> dict | None:
    """
    pede à API STSC da REDEMET os raios da última hora.

    :returns: {"anima": [horas "HH:MM"], "stsc": [[{"la", "lo"}, ...], ...]},
              listas paralelas (as posições de stsc[i] são da hora anima[i]);
              None em qualquer erro (já logado)
    """
    l_response = _request_snapshot()

    if l_response is None:
        return None

    ldct_body = _decode_snapshot(l_response.text)

    if ldct_body is None:
        return None

    return _extract_events(ldct_body)

# ─────────────────────────────────────────────────────────────────────────────────────
def _request_snapshot() -> requests.Response | None:
    """
    faz a requisição HTTP à REDEMET.

    :returns: a resposta (status 200), ou None em erro de rede ou status diferente
    """
    ls_url = defs.CS_REDEMET_URL.format(defs.CS_REDEMET_KEY)

    try:
        l_response = requests.get(ls_url, timeout=CI_HTTP_TIMEOUT)

    except requests.exceptions.RequestException as l_err:
        M_LOG.error("REDEMET STSC. Error request: %s.", str(l_err))
        return None

    if CI_HTTP_OK != l_response.status_code:
        M_LOG.error("REDEMET STSC. Data not found. Code: %s", str(l_response.status_code))
        return None

    return l_response

# ─────────────────────────────────────────────────────────────────────────────────────
def _decode_snapshot(fs_text: str) -> dict | None:
    """
    decodifica o corpo JSON da resposta.

    :param fs_text: corpo da resposta HTTP
    :returns: o JSON decodificado, ou None se não for JSON válido
    """
    try:
        return json.loads(fs_text)

    except json.decoder.JSONDecodeError as l_err:
        M_LOG.error("REDEMET STSC. Data decoding error: %s.", str(l_err))
        return None

# ─────────────────────────────────────────────────────────────────────────────────────
def _extract_events(fdct_body: dict) -> dict | None:
    """
    confere o envelope da resposta ({"status": true, "data": {...}}) e tira
    dele as duas listas de eventos.

    :param fdct_body: JSON decodificado
    :returns: {"anima": ..., "stsc": ...}, ou None se status for falso/ausente
              ou se não houver "data", "anima" ou "stsc"
    """
    if not fdct_body.get("status", None):
        M_LOG.error("REDEMET STSC. Data status error: %s", str(fdct_body))
        return None

    ldct_data = fdct_body.get("data", None)

    if not ldct_data:
        M_LOG.error("REDEMET STSC. Data have no data field: %s", str(fdct_body))
        return None

    # [fix] sem "anima" ou "stsc" o KeyError subia como traceback (rc 1 sem mensagem)
    if "anima" not in ldct_data or "stsc" not in ldct_data:
        M_LOG.error("REDEMET STSC. Data without anima/stsc: %s", str(fdct_body))
        return None

    return {"anima": ldct_data["anima"],
            "stsc":  ldct_data["stsc"]}

# ─────────────────────────────────────────────────────────────────────────────────────
def _snapshot_is_usable(fdct_snapshot: dict | None) -> bool:
    """
    confere se o snapshot veio e se as duas listas têm conteúdo.

    :param fdct_snapshot: retorno de get_redemet_raios
    :returns: True se dá para seguir; False (já logado) caso contrário
    """
    if not fdct_snapshot:
        M_LOG.error("falha ao obter dados da REDEMET.")
        return False

    if not fdct_snapshot.get("anima", []):
        M_LOG.warning("campo 'anima' vazio na resposta.")
        return False

    if not fdct_snapshot.get("stsc", []):
        M_LOG.warning("campo 'stsc' vazio na resposta.")
        return False

    return True

# ═════════════════════════════════════════════════════════════════════════════════════
# passos 2 a 4: janela de retenção, conversão e deduplicação
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def _retention_cutoff() -> datetime.datetime:
    """
    instante mais antigo que ainda fica no CSV: agora (UTC) menos CI_MAX_HOURS.

    :returns: datetime UTC de corte
    """
    ldt_now_utc = datetime.datetime.now(datetime.timezone.utc)

    return ldt_now_utc - datetime.timedelta(hours=CI_MAX_HOURS)

# ─────────────────────────────────────────────────────────────────────────────────────
def load_existing_data(fs_filepath: str) -> list[dict]:
    """
    lê o CSV acumulado, se existir.

    :param fs_filepath: caminho completo do CSV
    :returns: lista de dicts (valores como str); vazia se o arquivo não existe
              ou não pôde ser lido (aviso no log)
    """
    if not os.path.exists(fs_filepath):
        return []

    llst_rows = []

    try:
        with open(fs_filepath, "r", encoding="utf-8", newline="") as l_file:
            for ldct_row in csv.DictReader(l_file):
                llst_rows.append(ldct_row)

    except (OSError, csv.Error) as l_err:
        # devolve o que conseguiu ler até o erro
        M_LOG.warning("não foi possível ler '%s': %s", fs_filepath, str(l_err))

    return llst_rows

# ─────────────────────────────────────────────────────────────────────────────────────
def filter_recent_rows(flst_rows: list[dict],
                       fdt_cutoff: datetime.datetime) -> list[dict]:
    """
    mantém só as linhas com timestamp >= cutoff.
    linhas sem timestamp ou com timestamp malformado são descartadas.

    :param flst_rows: linhas lidas do CSV
    :param fdt_cutoff: datetime UTC de corte
    :returns: linhas dentro da janela de retenção, na ordem original
    """
    lf_cutoff_ts = fdt_cutoff.timestamp()

    llst_recent_rows = []

    for ldct_row in flst_rows:
        try:
            lf_row_ts = float(ldct_row["timestamp"])

        except (KeyError, ValueError):
            M_LOG.debug("linha descartada por timestamp inválido: %s", ldct_row)
            continue

        if lf_row_ts >= lf_cutoff_ts:
            llst_recent_rows.append(ldct_row)

    return llst_recent_rows

# ─────────────────────────────────────────────────────────────────────────────────────
def _snapshot_to_rows(flst_hours: list, flst_positions_per_hour: list,
                      fdt_cutoff: datetime.datetime) -> list[dict]:
    """
    converte o snapshot da API em linhas do CSV.

    a API manda duas listas paralelas: flst_hours[i] ("HH:MM", GMT) e
    flst_positions_per_hour[i] (lista de {"la": lat, "lo": lng} daquele minuto).
    Horas inválidas ou anteriores ao corte e posições sem "la"/"lo" são puladas.

    :param flst_hours: horas "HH:MM"
    :param flst_positions_per_hour: posições de cada hora
    :param fdt_cutoff: datetime UTC de corte
    :returns: lista de dicts {timestamp (int), lat, lng}
    """
    llst_rows = []

    for ls_hour, llst_positions in zip(flst_hours, flst_positions_per_hour):
        ldt_hour = parse_hora_gmt(ls_hour)

        if ldt_hour is None:
            continue

        if ldt_hour < fdt_cutoff:
            M_LOG.debug("entrada ignorada por ser antiga: %s", ls_hour)
            continue

        li_timestamp = int(ldt_hour.timestamp())

        for ldct_position in llst_positions:
            # [fix] posição sem "la"/"lo" é pulada (antes: KeyError e o download inteiro perdido)
            if "la" not in ldct_position or "lo" not in ldct_position:
                M_LOG.warning("posição sem la/lo ignorada: %r", ldct_position)
                continue

            llst_rows.append({
                "timestamp": li_timestamp,
                "lat":       ldct_position["la"],
                "lng":       ldct_position["lo"],
            })

    return llst_rows

# ─────────────────────────────────────────────────────────────────────────────────────
def parse_hora_gmt(fs_hora: str) -> datetime.datetime | None:
    """
    converte "HH:MM" (GMT) num datetime UTC com a data de hoje.

    virada de meia-noite: a API manda só a hora, então logo depois da
    meia-noite chegam horas como "23:59" que são de ontem.  Se o resultado
    ficar mais de 1 hora no futuro, ele é tratado como do dia anterior.

    :param fs_hora: hora no formato "HH:MM"
    :returns: datetime UTC (segundos zerados), ou None se o formato for inválido
    """
    try:
        li_hour = int(fs_hora[0:2])
        li_minute = int(fs_hora[3:5])

    except (ValueError, IndexError) as l_err:
        M_LOG.error("formato de hora inválido '%s': %s", fs_hora, str(l_err))
        return None

    ldt_now_utc = datetime.datetime.now(datetime.timezone.utc)

    ldt_candidate = ldt_now_utc.replace(hour=li_hour, minute=li_minute,
                                        second=0, microsecond=0)

    if ldt_candidate > ldt_now_utc + datetime.timedelta(hours=1):
        ldt_candidate -= datetime.timedelta(days=1)

    return ldt_candidate

# ─────────────────────────────────────────────────────────────────────────────────────
def _drop_duplicates(flst_new_rows: list[dict], flst_kept_rows: list[dict]) -> list[dict]:
    """
    tira das linhas novas as que já estão no CSV (a API devolve a hora
    inteira a cada chamada, então chamadas seguidas se repetem).

    :param flst_new_rows: linhas vindas da API
    :param flst_kept_rows: linhas já gravadas (dentro da janela)
    :returns: linhas novas que não estavam no CSV, na ordem original
    """
    lset_known_keys = {_row_key(ldct_row) for ldct_row in flst_kept_rows}

    return [ldct_row for ldct_row in flst_new_rows if _row_key(ldct_row) not in lset_known_keys]

# ─────────────────────────────────────────────────────────────────────────────────────
def _row_key(fdct_row: dict) -> tuple:
    """
    chave de deduplicação de uma linha.

    a mesma linha pode vir do CSV (tudo str, ex. "1.5") ou da API (números,
    ex. 1.5); timestamp vira int e lat/lng viram str para as duas comparar igual.

    :param fdct_row: linha com timestamp, lat e lng
    :returns: (timestamp inteiro, lat como str, lng como str)
    """
    return (int(float(fdct_row["timestamp"])),
            str(fdct_row["lat"]),
            str(fdct_row["lng"]))

# ═════════════════════════════════════════════════════════════════════════════════════
# passo 5: gravação
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def _write_csv(fs_csv_path: str, flst_rows: list[dict]) -> bool:
    """
    grava o CSV completo de forma atômica: o tracker e o mapper, em outro
    processo, sempre leem o arquivo anterior inteiro ou o novo inteiro.

    :param fs_csv_path: caminho final do CSV
    :param flst_rows: todas as linhas (mantidas + novas)
    :returns: True se gravou, False em erro de gravação (já logado)
    """
    try:
        ioutil.atomic_write_csv(fs_csv_path, CLST_HEAD, flst_rows)

    except OSError as l_err:
        M_LOG.error("falha ao gravar '%s': %s", fs_csv_path, str(l_err))
        return False

    M_LOG.info("total de registros persistidos em '%s': %d.", fs_csv_path, len(flst_rows))

    return True

# ─────────────────────────────────────────────────────────────────────────────────────
def main(flst_argv: list | None = None) -> int:
    """
    ponto de entrada: lê os argumentos, confere a API key e baixa.

    :param flst_argv: argumentos (None = sys.argv[1:])
    :returns: CI_RC_OK ou CI_RC_FAIL
    """
    l_args = arg_parse(flst_argv)

    # sem API key (.env ausente ou incompleto): falha cedo e com uma mensagem
    # clara, em vez de a URL da REDEMET levar a string "None" e a falha
    # aparecer depois como um erro de autenticação obscuro
    if not defs.CS_REDEMET_KEY:
        M_LOG.error("CS_REDEMET_KEY não definida -- verifique apps/.env.")
        return CI_RC_FAIL

    if not download_raios(l_args):
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
