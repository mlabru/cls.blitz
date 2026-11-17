# Centro Logístico de Simulação Meteorológica
# Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo
#
# Simple Lightning Model
# blt_cfg.sh
# configuração de ambiente (onde: código, dados, logs, locks, arquivos de
# estado e produtos) e os poucos parâmetros do shell (frames retidos,
# FPS/escala do GIF, qualidade do MP4) usados pelo blt_run.sh.
#
# os parâmetros de algoritmo (DBSCAN, Kalman, EMA, horizontes de previsão)
# ficam no topo dos módulos Python em apps/ -- o shell nunca os lê.
#
# sempre "source"ado, nunca executado -- não define "set -e": herda as
# opções de shell de quem importa.
#
# 2026.nov  mlabru  CLS_LIB obrigatória (passo 2): sai o symlink cls_common.sh
#                   da raiz do repo; sem cls.env/CLS_LIB, erro explícito
#
# 2026.nov  mlabru  - cls.env (CLS_ENV_FILE, padrão ~/.config/cls/cls.env)
#                     carregado antes dos padrões; cls_common.sh vem de
#                     CLS_LIB, com o symlink ao lado do script como reserva
#                   - prefixo met -> cls (cls_log, cls_check_test_env,
#                     CLS_ENV, CLS_TST_ROOT)
#
# 2026.nov  mlabru  data/ e logs/ únicos para as duas fontes: SLM_TAG (slm/glm)
#                   prefixa todo arquivo de dados, estado, produto e log; os
#                   frames temporários ficam em temp/<tag>/ (eram data_glm/ e
#                   logs_glm/ separados); logs dos programas em SLM_LOG_*
# 2026.nov  mlabru  SLM_SOURCE (stsc/glm): escolhe o downloader (SLM_DOWNLOADER)
#                   e separa data/ e logs/ da GLM (SLM_SOURCE_TAG)
# 2026.nov  mlabru  SLM_ASSOC_FILE/SLM_FRAMES_FILE (diagnóstico do tracker) e
#                   SLM_ARCHIVE (arquivo diário dos raios, para o bltReplay)
# 2026.nov  mlabru  SLM_VERIFY_FILE (CSV de verificação, passado ao tracker)
# 2026.out  mlabru  SLM_TRACKS_STATE_FILE explícito (passado ao tracker)
# 2026.out  mlabru  - só cron: saem a FIFO tracker -> renderer, o sentinela
#                     de shutdown, o lock do daemon, o lock do processor e
#                     os intervalos de polling (só serviam aos workers do
#                     daemon, movidos para .bkp/), além de SLM_ASSETS e
#                     SLM_ERR_PNG, que ninguém usava
#                   - SLM_LIB_DIR (código) e SLM_VENV (padrão ~/.venv),
#                     como NWC_LIB_DIR/NWC_VENV do 27.nowcast
#                   - trava do ambiente de teste via cls_check_test_env
#                     (cls_common.sh, symlink), comum aos outros projetos
#                   - os logs do próprio shell passam a ser definidos pelo
#                     script (blt_run.sh), como nas irmãs
# 2026.aug  mlabru  extraído de blt_common.sh

# depends on: bash, cls_common.sh
# shellcheck shell=bash

# < configuração >---------------------------------------------------------------------

# código (apps/*.py) -- o diretório deste arquivo
SLM_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ambiente da instalação (cls.env: CLS_LIB e sobreposições), carregado
# antes dos padrões abaixo; "set -a" exporta para os programas Python.
# Precedência: ambiente do processo > cls.env > padrões deste arquivo (ver
# cls.common/etc/cls.env.example)
CLS_ENV_FILE="${CLS_ENV_FILE:-${HOME:-}/.config/cls/cls.env}"
if [ -r "${CLS_ENV_FILE}" ]; then set -a; source "${CLS_ENV_FILE}"; set +a; fi

# raiz de execução: data/ e logs/.  Pode ser sobreposta pelo ambiente (deploy
# em outra máquina, execução isolada de teste).  ${USER} pode não estar
# exportado (cron, systemd): usa id -un para não quebrar sob 'set -u'
SLM_PATH="${SLM_PATH:-/home/${USER:-$(id -un)}/Public/bltz}"

# fonte dos eventos, sobreponível pelo ambiente (no crontab: SLM_SOURCE=glm):
#   stsc -> REDEMET STSC, células de grade CB por imagem (slmDownloader.py)
#   glm  -> GOES-19 GLM, flashes contínuos (glmDownloader.py)
# o tracker e o render recebem --source (janela de agrupamento, HUD).
SLM_SOURCE="${SLM_SOURCE:-stsc}"

# prefixo de todo arquivo da fonte (dados, estado, produtos, logs, lock):
# slm_ = STSC, glm_ = GLM.  As duas fontes dividem data/ e logs/ sem se
# misturar e podem rodar ao mesmo tempo (locks separados)
case "${SLM_SOURCE}" in
    stsc) SLM_DOWNLOADER="slmDownloader"
          SLM_TAG="slm" ;;
    glm)  SLM_DOWNLOADER="glmDownloader"
          SLM_TAG="glm" ;;
    *)    echo "blt_cfg.sh: SLM_SOURCE inválida: '${SLM_SOURCE}' (use stsc ou glm)" >&2
          exit 1 ;;
esac

# diretórios de dados e de logs (comuns às duas fontes)
SLM_DATA="${SLM_PATH}/data"
SLM_LOGS="${SLM_PATH}/logs"

# logs dos programas Python (stdout/stderr redirecionados pelo blt_common.sh)
SLM_LOG_DOWNLOADER="${SLM_LOGS}/${SLM_TAG}_downloader.log"
SLM_LOG_MAPPER="${SLM_LOGS}/${SLM_TAG}_mapper.log"
SLM_LOG_TRACKER="${SLM_LOGS}/${SLM_TAG}_tracker.log"
SLM_LOG_RENDER="${SLM_LOGS}/${SLM_TAG}_render.log"

# raw: CSV baixado; state: locks e estado entre execuções;
# temp: JSON/PNG por frame (um subdiretório por fonte: os frames se chamam
# frame_NNNN nas duas); output: produtos finais;
# archive: eventos de cada dia (<tag>_raw_AAAAMMDD.csv.gz), base do reprocessamento
# para calibração (apps/bltReplay.py); o downloader cria o diretório e apaga
# os dias além de CI_ARCHIVE_DAYS (blt_archive.py)
SLM_ARCHIVE="${SLM_DATA}/archive"
SLM_OUT="${SLM_DATA}/output"
SLM_RAW="${SLM_DATA}/raw"
SLM_STATE="${SLM_DATA}/state"
SLM_TEMP="${SLM_DATA}/temp/${SLM_TAG}"

# frames intermediários do tracker (JSON) e do render (PNG)
SLM_JSON="${SLM_TEMP}/json"
SLM_PNG="${SLM_TEMP}/png"

# nome do CSV acumulado mantido pelo downloader (dentro de SLM_RAW)
SLM_RAW_FILE="${SLM_TAG}_lightning.csv"

# lock de instância do blt_run.sh (uma rodada do cron não atropela a anterior)
SLM_RUN_LOCK="${SLM_STATE}/.${SLM_TAG}_run.lock"

# estado entre execuções:
#   frame counter     -> índice do próximo frame_NNNN.json
#   last processed    -> mtime do CSV na última geração do mapa simples
#   last ts           -> timestamp do raio mais novo já rastreado (--since-ts)
SLM_FRAME_COUNTER_FILE="${SLM_STATE}/.${SLM_TAG}_frame_counter"
SLM_LAST_PROC_FILE="${SLM_STATE}/.${SLM_TAG}_last_processed"
SLM_LAST_TS_FILE="${SLM_STATE}/.${SLM_TAG}_last_ts"

# tracks e filtros de Kalman entre frames (bltTracker.py --tracks-file).
# [fix] antes não era passado e o tracker usava o padrão relativo
# ./data/state/slm_tracks_state.json -- o mesmo arquivo, mas só porque o
# blt_run.sh faz cd para SLM_PATH.  Apagar este arquivo zera os IDs de track
SLM_TRACKS_STATE_FILE="${SLM_STATE}/${SLM_TAG}_tracks_state.json"

# verificação das previsões (bltTracker.py --verify-file): uma linha por
# previsão conferida contra a célula observada; cresce por append e é
# rotacionado pelo check_log_size do blt_run.sh (GI_LOG_MAX_SIZE)
SLM_VERIFY_FILE="${SLM_STATE}/${SLM_TAG}_verification.csv"

# diagnóstico do tracker (--assoc-file/--frames-file): uma linha por célula
# (associação: distância até a previsão, score, NIS) e uma por frame
# (latência, ciclo de vida dos tracks).  Append e rotação como o de cima
SLM_ASSOC_FILE="${SLM_STATE}/${SLM_TAG}_assoc.csv"
SLM_FRAMES_FILE="${SLM_STATE}/${SLM_TAG}_frames.csv"

# métricas (chave=valor) e o lock que serializa a escrita delas
SLM_METRICS="${SLM_STATE}/.${SLM_TAG}_metrics.txt"
SLM_METRICS_LOCK="${SLM_STATE}/.${SLM_TAG}_metrics.lock"

# produtos finais (dentro de SLM_OUT):
#   now/prv -> mapa simples atual e anterior (bltMapper.py)
#   out.png -> último frame rastreado (bltRender.py); out.gif/out.mp4 -> animação
SLM_NOW_PNG="${SLM_TAG}_now.png"
SLM_PRV_PNG="${SLM_TAG}_prv.png"
SLM_OUT_PNG="${SLM_TAG}_out.png"
SLM_OUT_GIF="${SLM_TAG}_out.gif"
SLM_OUT_MP4="${SLM_TAG}_out.mp4"

# virtualenv dos programas Python (numpy, scipy, cartopy, ...); sem ele, o
# python3 do sistema (slm_setup_python, em blt_common.sh)
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

# cls_check_test_env (cls_common.sh, compartilhado pelos projetos cls)
# CLS_LIB (cls.env) é obrigatória: sem ela não há cls_common.sh -- falha
# aqui, com a causa, em vez de num "command not found" mais adiante
if [ ! -r "${CLS_LIB:-}/cls_common.sh" ]; then
    echo "[error]: CLS_LIB='${CLS_LIB:-}' sem cls_common.sh; defina-a no cls.env (${CLS_ENV_FILE:-~/.config/cls/cls.env})" >&2
    exit 1
fi
source "${CLS_LIB}/cls_common.sh"

# com CLS_ENV=test (cls.test), SLM_PATH precisa estar dentro de
# CLS_TST_ROOT.  Roda no source, antes de qualquer mkdir/log/lock
cls_check_test_env -b "${SLM_PATH}" "${SLM_PATH}" || exit 1

# < the end >--------------------------------------------------------------------------
