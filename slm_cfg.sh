# Centro Logístico de Simulação Meteorológica
# Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo
#
# Simple Lightning Model
# slm_cfg.sh
# configuração de ambiente (onde: código, dados, logs, locks, arquivos de
# estado e produtos) e os poucos parâmetros do shell (frames retidos,
# FPS/escala do GIF, qualidade do MP4) usados pelo slm_run.sh.
#
# os parâmetros de algoritmo (DBSCAN, Kalman, EMA, horizontes de previsão)
# ficam no topo dos módulos Python em apps/ -- o shell nunca os lê.
#
# sempre "source"ado, nunca executado -- não define "set -e": herda as
# opções de shell de quem importa.
#
# 2026.nov  mlabru  SLM_ASSOC_FILE/SLM_FRAMES_FILE (diagnóstico do tracker) e
#                   SLM_ARCHIVE (arquivo diário dos raios, para o slmReplay)
# 2026.nov  mlabru  SLM_VERIFY_FILE (CSV de verificação, passado ao tracker)
# 2026.out  mlabru  SLM_TRACKS_STATE_FILE explícito (passado ao tracker)
# 2026.out  mlabru  - só cron: saem a FIFO tracker -> renderer, o sentinela
#                     de shutdown, o lock do daemon, o lock do processor e
#                     os intervalos de polling (só serviam aos workers do
#                     daemon, movidos para .bkp/), além de SLM_ASSETS e
#                     SLM_ERR_PNG, que ninguém usava
#                   - SLM_LIB_DIR (código) e SLM_VENV (padrão ~/.venv),
#                     como NWC_LIB_DIR/NWC_VENV do 27.nowcast
#                   - trava do ambiente de teste via met_check_test_env
#                     (cls_common.sh, symlink), comum aos outros projetos
#                   - os logs do próprio shell passam a ser definidos pelo
#                     script (slm_run.sh), como nas irmãs
# 2026.aug  mlabru  extraído de slm_common.sh

# depends on: bash, cls_common.sh
# shellcheck shell=bash

# < configuração >---------------------------------------------------------------------

# código (apps/*.py) -- o diretório deste arquivo
SLM_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# raiz de execução: data/ e logs/.  Pode ser sobreposta pelo ambiente (deploy
# em outra máquina, execução isolada de teste).  ${USER} pode não estar
# exportado (cron, systemd): usa id -un para não quebrar sob 'set -u'
SLM_PATH="${SLM_PATH:-/home/${USER:-$(id -un)}/Public/bltz}"

# diretórios de dados e de logs
SLM_DATA="${SLM_PATH}/data"
SLM_LOGS="${SLM_PATH}/logs"

# raw: CSV baixado; state: locks e estado entre execuções;
# temp: JSON/PNG por frame; output: produtos finais;
# archive: raios de cada dia (slm_raw_AAAAMMDD.csv.gz), base do reprocessamento
# para calibração (apps/slmReplay.py); o downloader cria o diretório e apaga
# os dias além de CI_ARCHIVE_DAYS (slm_archive.py)
SLM_ARCHIVE="${SLM_DATA}/archive"
SLM_OUT="${SLM_DATA}/output"
SLM_RAW="${SLM_DATA}/raw"
SLM_STATE="${SLM_DATA}/state"
SLM_TEMP="${SLM_DATA}/temp"

# frames intermediários do tracker (JSON) e do render (PNG)
SLM_JSON="${SLM_TEMP}/json"
SLM_PNG="${SLM_TEMP}/png"

# nome do CSV acumulado mantido pelo downloader (dentro de SLM_RAW)
SLM_RAW_FILE="slm_lightning.csv"

# lock de instância do slm_run.sh (uma rodada do cron não atropela a anterior)
SLM_RUN_LOCK="${SLM_STATE}/.slm_run.lock"

# estado entre execuções:
#   frame counter     -> índice do próximo frame_NNNN.json
#   last processed    -> mtime do CSV na última geração do mapa simples
#   last ts           -> timestamp do raio mais novo já rastreado (--since-ts)
SLM_FRAME_COUNTER_FILE="${SLM_STATE}/.slm_frame_counter"
SLM_LAST_PROC_FILE="${SLM_STATE}/.slm_last_processed"
SLM_LAST_TS_FILE="${SLM_STATE}/.slm_last_ts"

# tracks e filtros de Kalman entre frames (slmTracker.py --tracks-file).
# [fix] antes não era passado e o tracker usava o padrão relativo
# ./data/state/slm_tracks_state.json -- o mesmo arquivo, mas só porque o
# slm_run.sh faz cd para SLM_PATH.  Apagar este arquivo zera os IDs de track
SLM_TRACKS_STATE_FILE="${SLM_STATE}/slm_tracks_state.json"

# verificação das previsões (slmTracker.py --verify-file): uma linha por
# previsão conferida contra a célula observada; cresce por append e é
# rotacionado pelo check_log_size do slm_run.sh (GI_LOG_MAX_SIZE)
SLM_VERIFY_FILE="${SLM_STATE}/slm_verification.csv"

# diagnóstico do tracker (--assoc-file/--frames-file): uma linha por célula
# (associação: distância até a previsão, score, NIS) e uma por frame
# (latência, ciclo de vida dos tracks).  Append e rotação como o de cima
SLM_ASSOC_FILE="${SLM_STATE}/slm_assoc.csv"
SLM_FRAMES_FILE="${SLM_STATE}/slm_frames.csv"

# métricas (chave=valor) e o lock que serializa a escrita delas
SLM_METRICS="${SLM_STATE}/.slm_metrics.txt"
SLM_METRICS_LOCK="${SLM_STATE}/.slm_metrics.lock"

# produtos finais (dentro de SLM_OUT):
#   now/prv -> mapa simples atual e anterior (slmMapper.py)
#   out.png -> último frame rastreado (slmRender.py); out.gif/out.mp4 -> animação
SLM_NOW_PNG="slm_now.png"
SLM_PRV_PNG="slm_prv.png"
SLM_OUT_PNG="slm_out.png"
SLM_OUT_GIF="slm_out.gif"
SLM_OUT_MP4="slm_out.mp4"

# virtualenv dos programas Python (numpy, scipy, cartopy, ...); sem ele, o
# python3 do sistema (slm_setup_python, em slm_common.sh)
SLM_VENV="${SLM_VENV:-${HOME:-/home/$(id -un)}/.venv}"

# limite de tamanho dos logs (10MB), usado na rotação simples
GI_LOG_MAX_SIZE=$((10 * 1024 * 1024))

# animação: quadros por segundo, largura do GIF (pixels), frames retidos e
# qualidade do MP4 (CRF: quanto menor, melhor e maior)
GI_FPS=2
GI_GIF_SCALE=1280
GI_MAX_FRAMES=12
GI_MP4_CRF=23

# < trava do ambiente de teste >-------------------------------------------------------

# met_check_test_env (cls_common.sh, compartilhado pelos projetos met)
source "${SLM_LIB_DIR}/cls_common.sh"

# com MET_ENV=test (21.mvol/tst_env), SLM_PATH precisa estar dentro de
# MET_TST_ROOT.  Roda no source, antes de qualquer mkdir/log/lock
met_check_test_env -b "${SLM_PATH}" "${SLM_PATH}" || exit 1

# < the end >--------------------------------------------------------------------------
