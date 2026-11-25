#!/usr/bin/env bash
#
# Centro Logístico de Simulação Meteorológica
# Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo
#
# Simple Lightning Model
# blt_run.sh
# script cron-driven do SLM: uma passada completa do pipeline por chamada.
#
# workflow:
# - downloader (SLM_DOWNLOADER, conforme SLM_SOURCE): apps/slmDownloader.py
#   (STSC da REDEMET) ou apps/glmDownloader.py (GOES-19 GLM); atualiza o
#   CSV acumulado (1h de retenção); uma falha aqui não impede o resto (segue
#   com o CSV que já existe);
# - apps/bltMapper.py: mapa simples dos raios (slm_now.png/slm_prv.png), só
#   se o CSV mudou;
# - apps/bltTracker.py: clusteriza e rastreia os raios novos num frame JSON;
# - apps/bltRender.py: desenha o frame (slm_out.png) e o ffmpeg regenera
#   slm_out.gif/slm_out.mp4 com os últimos GI_MAX_FRAMES frames.
#
# crontab: a REDEMET atualiza o snapshot a cada ~5 min, então o cron roda a
# cada 5 min (com */10, metade dos snapshots se perdia):
#   */5 * * * * /caminho/para/blt_run.sh >> /caminho/para/logs/slm_run_cron.log 2>&1
# com o GLM (arquivos a cada 20 s; mesmos data/ e logs/, arquivos glm_*; pode
# rodar junto com o STSC):
#   */5 * * * * SLM_SOURCE=glm /caminho/para/blt_run.sh >> /caminho/para/logs/glm_run_cron.log 2>&1
#
# 2026.nov  mlabru  renomeado (era slm_run.sh): blt_ = comum às duas fontes;
#                   logs com o prefixo da fonte (SLM_TAG)
# 2026.nov  mlabru  fonte escolhida por SLM_SOURCE (blt_cfg.sh): downloader e
#                   log dele vêm de SLM_DOWNLOADER
# 2026.nov  mlabru  CSVs de diagnóstico (SLM_ASSOC_FILE, SLM_FRAMES_FILE) rotacionados
# 2026.nov  mlabru  CSV de verificação (SLM_VERIFY_FILE) rotacionado como os logs
# 2026.out  mlabru  [fix] crontab de exemplo a cada 5 min (era */10)
# 2026.out  mlabru  - refatoração no molde do nwc_interpol.sh (27.nowcast):
#                     corpo em funções de um passo (acquire_instance_lock,
#                     run_download, run_mapper, run_tracker, run_render) e
#                     main(); SLM_RUN_NO_MAIN para os testes
#                   - só cron: sai o lock compartilhado com o slm_daemon.sh e
#                     o sentinela de shutdown (o daemon e seus workers foram
#                     para .bkp/)
#                   - lock ocupado (rodada anterior ainda em andamento) sai
#                     com 0 e [info], como nas irmãs: não é erro, o próximo
#                     cron tenta de novo
#                   - venv (SLM_VENV, se existir) e PYTHONPATH por
#                     slm_setup_python; logs dos programas rotacionados
# 2026.aug  mlabru  versão inicial (cron, sem daemon)
#
# depends on: blt_cfg.sh, blt_common.sh, cls_common.sh, python3, flock,
#             ffmpeg, apps/slmDownloader.py ou apps/glmDownloader.py, apps/bltMapper.py,
#             apps/bltTracker.py, apps/bltRender.py

set -euo pipefail

# < configuração >---------------------------------------------------------------------

# diretório deste script
LS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ambiente (SLM_LIB_DIR, SLM_PATH, diretórios, estado, lock, trava
# CLS_ENV=test) e funções comuns (log, passos do pipeline)
source "${LS_SCRIPT_DIR}/blt_cfg.sh"
source "${LS_SCRIPT_DIR}/blt_common.sh"

# log do próprio shell (log()) e erros
SLM_LOG_APP="${SLM_LOGS}/${SLM_TAG}_app.log"
SLM_LOG_ERR="${SLM_LOGS}/${SLM_TAG}_error.log"

# CSV acumulado mantido pelo downloader
SLM_RAW_CSV="${SLM_RAW}/${SLM_RAW_FILE}"

# -------------------------------------------------------------------------------------
# adquire o lock de instância (fd 201, mantido aberto até o fim do script).
# Se outra rodada ainda estiver em andamento, sai com 0
#
acquire_instance_lock()
{
    exec 201>"${SLM_RUN_LOCK}"

    if ! flock -n 201; then
        log "[info]: blt_run: uma execução anterior ainda está em andamento; saindo."
        exit 0
    fi

} # acquire_instance_lock

# -------------------------------------------------------------------------------------
# baixa as ocorrências novas (SLM_DOWNLOADER).  Uma falha só é logada: o
# pipeline segue com o CSV que já existe
#
run_download()
{
    log "[info]: blt_run: baixando ocorrências (fonte ${SLM_SOURCE})..."

    if ! run_slm_downloader; then
        log "[error]: blt_run: download falhou — usando o CSV existente; veja ${SLM_LOG_DOWNLOADER}"
    fi

} # run_download

# -------------------------------------------------------------------------------------
# mapa simples dos raios, se o CSV mudou (generate_map, em blt_common.sh)
#
run_mapper()
{
    generate_map "${SLM_RAW_CSV}" "blt_run"

} # run_mapper

# -------------------------------------------------------------------------------------
# rastreia um frame novo (track_new_frame, em blt_common.sh)
#
# retorno: imprime o caminho do JSON gerado, ou nada se não houve frame novo
#
run_tracker()
{
    track_new_frame "${SLM_RAW_CSV}" "blt_run"

} # run_tracker

# -------------------------------------------------------------------------------------
# desenha o frame e atualiza a animação (render_frame, em blt_common.sh)
#
# uso: run_render <json>
#
run_render()
{
    render_frame "$1" "blt_run"

} # run_render

# -------------------------------------------------------------------------------------
# fluxo principal:
#   1. raiz de execução e diretórios, lock de instância, rotação dos logs dos
#      programas, venv e PYTHONPATH;
#   2. download -- falha aqui só avisa;
#   3. sem CSV: avisa e termina;
#   4. mapa simples (se o CSV mudou);
#   5. tracker; se houve frame novo, render + GIF/MP4.
#
main()
{
    local ls_json_file

    # diretórios e cd para SLM_PATH -- precisa ser a primeira ação: log()
    # escreve em SLM_LOGS e o lock fica em SLM_STATE
    slm_enter_run_dir

    log "-----------------------------------------------------------------"
    log "[info]: blt_run: iniciando em $(hostname)..."

    acquire_instance_lock

    # os CSVs de verificação e de diagnóstico também só crescem: mesma
    # rotação dos logs (o histórico longo para calibrar está no SLM_ARCHIVE)
    check_log_size "${SLM_LOG_DOWNLOADER}" "${SLM_LOG_MAPPER}" \
                   "${SLM_LOG_TRACKER}" "${SLM_LOG_RENDER}" \
                   "${SLM_VERIFY_FILE}" "${SLM_ASSOC_FILE}" "${SLM_FRAMES_FILE}"

    slm_setup_python

    # 2. download
    run_download

    # 3. sem CSV não há o que mapear nem rastrear
    if [ ! -f "${SLM_RAW_CSV}" ]; then
        log "[warn]: blt_run: nenhum CSV em ${SLM_RAW} — pulando mapa/tracker/render."
        log "[info]: blt_run: execução concluída."
        return 0
    fi

    # 4. mapa simples
    run_mapper

    # 5. tracker e, se houve frame novo, render
    ls_json_file=$(run_tracker)

    if [ -n "${ls_json_file}" ]; then
        run_render "${ls_json_file}"
    fi

    log "[info]: blt_run: execução concluída."

} # main

# < main >-----------------------------------------------------------------------------

# permite carregar a configuração e as funções sem executar o main (testes)
[ "${SLM_RUN_NO_MAIN:-false}" = true ] && return 0

main

# < the end >--------------------------------------------------------------------------
