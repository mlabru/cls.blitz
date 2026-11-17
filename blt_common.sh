# Centro Logístico de Simulação Meteorológica
# Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo
#
# Simple Lightning Model
# blt_common.sh
# funções usadas pelo blt_run.sh: o log() do SLM (por cima do cls_log), o
# início comum (diretórios, venv) e os passos do pipeline -- mapa simples,
# tracker, render e animação.  Só define funções: nada roda no source.
#
# cada passo do pipeline é uma função "orquestradora" curta (generate_map,
# track_new_frame, render_frame) que chama funções de um passo só; os
# programas Python são chamados sempre pelas funções run_slm_*.
#
# 2026.nov  mlabru  cls_common.sh vem de CLS_LIB (cls.env), com o symlink
#                   ao lado do script como reserva; met_log -> cls_log
#
# 2026.nov  mlabru  logs dos programas em SLM_LOG_* (prefixo da fonte)
# 2026.nov  mlabru  run_slm_downloader chama SLM_DOWNLOADER (fonte SLM_SOURCE);
#                   tracker e render recebem --source
# 2026.nov  mlabru  [fix] generate_map só marca o CSV como processado depois
#                   do mapa (ou de "sem raios", rc 3): falha é retentada
# 2026.nov  mlabru  run_slm_downloader passa -a (SLM_ARCHIVE); run_slm_tracker
#                   passa --assoc-file/--frames-file (diagnóstico)
# 2026.nov  mlabru  run_slm_tracker passa --verify-file (SLM_VERIFY_FILE)
# 2026.out  mlabru  [fix] frames ordenados com sort -V (frame_10000 depois de
#                   frame_9999); tracker recebe --tracks-file; get_state
#                   devolve 0 também para arquivo vazio
# 2026.out  mlabru  - refatoração para legibilidade, no estilo do 27.nowcast:
#                     separador "# ---", chave na linha seguinte, bloco
#                     propósito/uso/retorno em cada função; os orquestradores
#                     divididos em funções de um passo só
#                   - só cron: init_fifo e is_running (FIFO e sentinela de
#                     shutdown do daemon) saem com os workers, movidos para
#                     .bkp/
#                   - check_log_size e o corpo do log() vêm do cls_common.sh
#                     (symlink para 04.dvl/bash); timestamp do log em UTC
#                   - slm_enter_run_dir (era ensure_dirs + cd) e
#                     slm_setup_python (venv, PYTHONPATH), como no nowcast;
#                     os programas são chamados por SLM_LIB_DIR/apps, não
#                     mais por "apps/" relativo ao cwd
# 2026.aug  mlabru  configuração extraída para blt_cfg.sh
# 2026.may  mlabru  refatorado para orquestrador de scripts independentes
# 2026.mar  mlabru  versão inicial (Linux/Bash)
#
# depends on: bash, cls_common.sh, ffmpeg, flock, python3; variáveis do
#             blt_cfg.sh; SLM_LOG_APP/SLM_LOG_ERR (definidas pelo script)
#             para log()
# shellcheck shell=bash

# check_log_size e cls_log (cls_common.sh, compartilhado pelos projetos cls)
source "${CLS_LIB:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/cls_common.sh"

# < início comum >---------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# logger com timestamp UTC e rotação (cls_log): grava em SLM_LOG_APP,
# ecoando em stderr -- o nível ([info]/[warn]/[error]) vem no início de $*
#
# uso: log "[info]: mensagem"
#
log()
{
    cls_log "${SLM_LOG_APP}" "${SLM_LOG_ERR}" "$@"

} # log

# -------------------------------------------------------------------------------------
# entra na raiz de execução (SLM_PATH) e garante que os diretórios e os
# arquivos de estado existam.  Tem de ser a primeira ação do main: log()
# escreve em SLM_LOGS e o lock fica em SLM_STATE.
#
# uso: slm_enter_run_dir
#
slm_enter_run_dir()
{
    mkdir -p "${SLM_DATA}" "${SLM_LOGS}" "${SLM_OUT}" "${SLM_RAW}" \
             "${SLM_STATE}" "${SLM_TEMP}" "${SLM_JSON}" "${SLM_PNG}"

    touch "${SLM_FRAME_COUNTER_FILE}" "${SLM_LAST_TS_FILE}" "${SLM_METRICS}"

    cd "${SLM_PATH}" || return 1

} # slm_enter_run_dir

# -------------------------------------------------------------------------------------
# prepara o python3 dos programas: ativa o virtualenv SLM_VENV só se ele
# existir, e põe apps/ e apps/util no PYTHONPATH (os módulos de apoio ficam
# em apps/work/ e se importam como "import work.blt_struct as sst").
#
# uso: slm_setup_python
#
slm_setup_python()
{
    if [ -f "${SLM_VENV}/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "${SLM_VENV}/bin/activate"
    fi

    export PYTHONPATH="${SLM_LIB_DIR}/apps:${SLM_LIB_DIR}/apps/util${PYTHONPATH:+:${PYTHONPATH}}"

} # slm_setup_python

# < estado e métricas >----------------------------------------------------------------

# -------------------------------------------------------------------------------------
# lê o valor gravado num arquivo de estado
#
# uso: valor=$(get_state <arquivo>)
# retorno: imprime o conteúdo do arquivo, ou 0 se ele não existir ou estiver
#          vazio ([fix] slm_enter_run_dir cria os arquivos de estado vazios, e
#          a primeira rodada passava --since-ts "" e logava "frame ...")
#
get_state()
{
    local ls_state_file="$1"
    local ls_value=""

    if [ -f "${ls_state_file}" ]; then
        ls_value=$(cat "${ls_state_file}")
    fi

    echo "${ls_value:-0}"

} # get_state

# -------------------------------------------------------------------------------------
# grava um valor num arquivo de forma atômica (tmp + mv): quem lê nunca vê o
# arquivo pela metade
#
# uso: safe_write <arquivo> <valor>
#
safe_write()
{
    local ls_file="$1"
    local ls_value="$2"

    echo "${ls_value}" > "${ls_file}.tmp"
    mv "${ls_file}.tmp" "${ls_file}"

} # safe_write

# -------------------------------------------------------------------------------------
# grava "chave=valor" no arquivo de métricas, preservando as outras chaves.
# A escrita é serializada por flock em SLM_METRICS_LOCK e feita via tmp + mv.
#
# uso: update_metrics <chave> <valor>
#
update_metrics()
{
    local ls_key="$1"
    local ls_value="$2"

    (
        flock -x 9

        {
            # todas as linhas que NÃO são desta chave, e depois a chave nova
            grep -v "^${ls_key}=" "${SLM_METRICS}" 2>/dev/null || true
            echo "${ls_key}=${ls_value}"
        } > "${SLM_METRICS}.tmp"

        mv "${SLM_METRICS}.tmp" "${SLM_METRICS}"

    ) 9>"${SLM_METRICS_LOCK}"

} # update_metrics

# < programas Python >-----------------------------------------------------------------

# -------------------------------------------------------------------------------------
# baixa os eventos novos da fonte (SLM_DOWNLOADER: slmDownloader.py, STSC da
# REDEMET, ou glmDownloader.py, GOES-19 GLM) e atualiza o CSV acumulado e
# o arquivo diário (SLM_ARCHIVE); stdout e stderr vão para
# SLM_LOG_DOWNLOADER (logs/<tag>_downloader.log).  Os dois aceitam os mesmos argumentos
#
# uso: run_slm_downloader
# retorno: o código de saída do downloader
#
run_slm_downloader()
{
    local li_rc=0

    python3 "${SLM_LIB_DIR}/apps/${SLM_DOWNLOADER}.py" -f "${SLM_RAW_FILE}" -s "${SLM_RAW}" \
            -a "${SLM_ARCHIVE}" >> "${SLM_LOG_DOWNLOADER}" 2>&1 || li_rc=$?

    return "${li_rc}"

} # run_slm_downloader

# -------------------------------------------------------------------------------------
# gera o mapa simples de raios (bltMapper.py) num PNG; stdout e stderr vão
# para SLM_LOG_MAPPER (logs/<tag>_mapper.log)
#
# uso: run_slm_mapper <csv> <png_de_saída>
# retorno: o código de saída do bltMapper.py
#
run_slm_mapper()
{
    local ls_csv="$1"
    local ls_png="$2"
    local li_rc=0

    python3 "${SLM_LIB_DIR}/apps/bltMapper.py" -s "${SLM_OUT}" -f "${ls_png}" "${ls_csv}" \
            >> "${SLM_LOG_MAPPER}" 2>&1 || li_rc=$?

    return "${li_rc}"

} # run_slm_mapper

# -------------------------------------------------------------------------------------
# clusteriza e rastreia os raios novos do CSV num frame JSON (bltTracker.py).
# O stdout do tracker é o contrato com o shell (linha "NEWEST_TS=<epoch>"
# quando houve raios novos) e é repassado ao chamador; o stderr (o logging
# do Python) vai para SLM_LOG_TRACKER (logs/<tag>_tracker.log).  As previsões conferidas vão para
# SLM_VERIFY_FILE e os diagnósticos para SLM_ASSOC_FILE/SLM_FRAMES_FILE
#
# uso: ls_output=$(run_slm_tracker <csv> <json_de_saída> <since_ts>)
# retorno: o código de saída do bltTracker.py
#
run_slm_tracker()
{
    local ls_csv="$1"
    local ls_json="$2"
    local ls_since_ts="$3"
    local li_rc=0

    python3 "${SLM_LIB_DIR}/apps/bltTracker.py" --since-ts "${ls_since_ts}" \
            --source "${SLM_SOURCE}" \
            --tracks-file "${SLM_TRACKS_STATE_FILE}" \
            --verify-file "${SLM_VERIFY_FILE}" \
            --assoc-file "${SLM_ASSOC_FILE}" --frames-file "${SLM_FRAMES_FILE}" \
            "${ls_csv}" "${ls_json}" 2>> "${SLM_LOG_TRACKER}" || li_rc=$?

    return "${li_rc}"

} # run_slm_tracker

# -------------------------------------------------------------------------------------
# desenha um frame JSON num PNG (bltRender.py); stdout e stderr vão para
# SLM_LOG_RENDER (logs/<tag>_render.log)
#
# uso: run_slm_render <json> <png_de_saída>
# retorno: o código de saída do bltRender.py
#
run_slm_render()
{
    local ls_json="$1"
    local ls_png="$2"
    local li_rc=0

    python3 "${SLM_LIB_DIR}/apps/bltRender.py" --source "${SLM_SOURCE}" "${ls_json}" "${ls_png}" \
            >> "${SLM_LOG_RENDER}" 2>&1 || li_rc=$?

    return "${li_rc}"

} # run_slm_render

# < mapa simples >---------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# gera o mapa simples (scatter, sem tracking) a partir do CSV acumulado, se o
# CSV mudou desde a última geração (mtime do CSV vs. SLM_LAST_PROC_FILE).
#
# fluxo:
#   1. sem CSV ou sem novidade: não faz nada (não é erro);
#   2. gera o PNG num temporário e, se der certo, publica (NOW -> PRV, novo
#      -> NOW) e grava a métrica last_update;
#   3. grava o mtime do CSV como processado -- só se deu certo ou se o
#      mapper disse "sem raios na região" (rc 3, CI_RC_NO_LIGHTNING do
#      bltMapper.py, logado como [info]).  Outra falha: loga [error] e a
#      próxima rodada tenta de novo, mesmo sem o CSV mudar.
#
# [fix] o mtime era gravado ANTES de rodar o mapper: uma falha passageira
# (cartopy, disco) deixava o mapa velho até o CSV mudar.
#
# uso: generate_map <csv> [prefixo-de-log]   (prefixo padrão: "mapper")
#
generate_map()
{
    local ls_csv="$1"
    local ls_log_prefix="${2:-mapper}"
    local li_csv_mtime li_last_processed ls_temporary_png
    local li_rc=0
    local -r li_rc_no_lightning=3    # CI_RC_NO_LIGHTNING do bltMapper.py

    if [ ! -f "${ls_csv}" ]; then
        return 0
    fi

    li_csv_mtime=$(stat -c %Y "${ls_csv}")
    li_last_processed=$(get_state "${SLM_LAST_PROC_FILE}")

    if (( li_csv_mtime <= li_last_processed )); then
        log "[info]: ${ls_log_prefix}: CSV sem novidade — mapa não gerado."
        return 0
    fi

    log "[info]: ${ls_log_prefix}: gerando mapa..."

    # sufixo $$ (PID) evita colisão com outro processo gerando ao mesmo tempo
    ls_temporary_png="${SLM_TEMP}/tmp.$$.png"

    run_slm_mapper "${ls_csv}" "${ls_temporary_png}" || li_rc=$?

    if (( li_rc == li_rc_no_lightning )); then
        log "[info]: ${ls_log_prefix}: sem raios na região — mapa não gerado."
        rm -f "${ls_temporary_png}"
        safe_write "${SLM_LAST_PROC_FILE}" "${li_csv_mtime}"
        return 0
    fi

    if (( 0 != li_rc )); then
        log "[error]: ${ls_log_prefix}: mapper falhou na geração do mapa (rc ${li_rc}); nova tentativa na próxima rodada."
        rm -f "${ls_temporary_png}"
        return 0
    fi

    publish_map_png "${ls_temporary_png}"

    safe_write "${SLM_LAST_PROC_FILE}" "${li_csv_mtime}"

    update_metrics "last_update" "$(date)"

} # generate_map

# -------------------------------------------------------------------------------------
# publica um mapa novo: o atual (SLM_NOW_PNG) vira o anterior (SLM_PRV_PNG) e
# o novo vira o atual.  Na primeira vez (sem atual), o novo é usado nos dois.
#
# uso: publish_map_png <png_novo>   (o arquivo é movido)
#
publish_map_png()
{
    local ls_new_png="$1"
    local ls_now_png="${SLM_OUT}/${SLM_NOW_PNG}"
    local ls_previous_png="${SLM_OUT}/${SLM_PRV_PNG}"

    if [ -f "${ls_now_png}" ]; then
        cp "${ls_now_png}" "${ls_previous_png}"
    else
        cp "${ls_new_png}" "${ls_previous_png}"
    fi

    # mv é atômico dentro do mesmo sistema de arquivos
    mv "${ls_new_png}" "${ls_now_png}"

} # publish_map_png

# < tracker >--------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# rastreia um frame novo a partir do CSV acumulado, usando o estado gravado
# (contador de frames e o since-ts).
#
# fluxo:
#   1. lê o contador de frames e o timestamp do último raio já rastreado;
#   2. roda o tracker para frame_NNNN.json;
#   3. sem "NEWEST_TS=" no stdout (nada novo), tracker com erro ou JSON não
#      gerado: loga e não imprime nada;
#   4. grava o novo since-ts e avança o contador ANTES de devolver o JSON
#      (o estado fica consistente mesmo que o processo morra depois).
#
# uso: ls_json=$(track_new_frame <csv> [prefixo-de-log])   (padrão: "tracker")
#      [ -n "${ls_json}" ] && ...
# retorno: imprime o caminho do JSON gerado SÓ se houve frame novo
#
track_new_frame()
{
    local ls_csv="$1"
    local ls_log_prefix="${2:-tracker}"
    local li_frame_index lf_last_ts ls_json_file ls_tracker_output lf_newest_ts

    li_frame_index=$(get_state "${SLM_FRAME_COUNTER_FILE}")
    lf_last_ts=$(get_state "${SLM_LAST_TS_FILE}")

    ls_json_file=$(printf "%s/frame_%04d.json" "${SLM_JSON}" "${li_frame_index}")

    log "[info]: ${ls_log_prefix}: processando ${ls_csv##*/} → frame ${li_frame_index}..."

    if ! ls_tracker_output=$(run_slm_tracker "${ls_csv}" "${ls_json_file}" "${lf_last_ts}"); then
        log "[error]: ${ls_log_prefix}: tracker falhou para ${ls_csv##*/}."
        return 0
    fi

    lf_newest_ts=$(extract_newest_ts "${ls_tracker_output}")

    if [ -z "${lf_newest_ts}" ]; then
        log "[info]: ${ls_log_prefix}: sem novos eventos em ${ls_csv##*/}."
        return 0
    fi

    if [ ! -f "${ls_json_file}" ]; then
        log "[error]: ${ls_log_prefix}: JSON não gerado: ${ls_json_file}."
        return 0
    fi

    safe_write "${SLM_LAST_TS_FILE}" "${lf_newest_ts}"
    safe_write "${SLM_FRAME_COUNTER_FILE}" "$(( li_frame_index + 1 ))"

    echo "${ls_json_file}"

} # track_new_frame

# -------------------------------------------------------------------------------------
# extrai o valor da linha "NEWEST_TS=<epoch>" do stdout do bltTracker.py
#
# uso: lf_ts=$(extract_newest_ts "<stdout do tracker>")
# retorno: imprime o epoch, ou nada se a linha não existir
#
extract_newest_ts()
{
    local ls_tracker_output="$1"

    echo "${ls_tracker_output}" | grep "NEWEST_TS=" | cut -d= -f2 || true

} # extract_newest_ts

# < render e animação >----------------------------------------------------------------

# -------------------------------------------------------------------------------------
# desenha um frame JSON em PNG, publica como o último frame (SLM_OUT_PNG),
# poda os frames antigos e regenera o GIF e o MP4.  Uma falha do render só
# é logada.
#
# uso: render_frame <json> [prefixo-de-log]   (prefixo padrão: "renderer")
#
render_frame()
{
    local ls_json_file="$1"
    local ls_log_prefix="${2:-renderer}"
    local ls_frame_name ls_png_file

    ls_frame_name=$(basename "${ls_json_file}" .json)
    ls_png_file="${SLM_PNG}/${ls_frame_name}.png"

    log "[info]: ${ls_log_prefix}: renderizando ${ls_frame_name}..."

    if ! run_slm_render "${ls_json_file}" "${ls_png_file}"; then
        log "[error]: ${ls_log_prefix}: bltRender.py falhou para ${ls_frame_name}."
        return 0
    fi

    cp "${ls_png_file}" "${SLM_OUT}/${SLM_OUT_PNG}"

    log "[info]: ${ls_log_prefix}: PNG gerado: ${ls_frame_name}.png"

    update_animation

} # render_frame

# -------------------------------------------------------------------------------------
# mantém só os GI_MAX_FRAMES frames mais recentes e regenera o GIF e o MP4
# com eles.  A lista de frames é calculada uma vez e usada pelos dois.
# Falha do ffmpeg não interrompe o pipeline.
#
# uso: update_animation
#
update_animation()
{
    local ls_framelist

    cleanup_old_frames

    ls_framelist=$(generate_framelist)

    generate_gif "${ls_framelist}" || true
    generate_mp4 "${ls_framelist}" || true

} # update_animation

# -------------------------------------------------------------------------------------
# apaga os frames (JSON e PNG) mais antigos, deixando os GI_MAX_FRAMES mais
# recentes de cada tipo
#
# uso: cleanup_old_frames
#
cleanup_old_frames()
{
    remove_oldest_files "${SLM_JSON}" "frame_*.json"
    remove_oldest_files "${SLM_PNG}" "frame_*.png"

} # cleanup_old_frames

# -------------------------------------------------------------------------------------
# apaga os arquivos mais antigos de um diretório que casam com um padrão,
# deixando os GI_MAX_FRAMES mais recentes.  "Mais antigo" = primeiro na
# ordem numérica do nome (sort -V): frame_9999 vem antes de frame_10000.
# [fix] com sort simples (alfabético) frame_10000 vinha antes de frame_9999
# e, depois do frame 9999, os frames novos é que eram apagados.
#
# uso: remove_oldest_files <diretório> <padrão>
#
remove_oldest_files()
{
    local ls_directory="$1"
    local ls_pattern="$2"
    local llst_files li_excess li_index

    # o padrão fica sem aspas de propósito, para o shell expandir o glob
    # shellcheck disable=SC2086
    mapfile -t llst_files < <(ls -1 "${ls_directory}"/${ls_pattern} 2>/dev/null | sort -V)

    li_excess=$(( ${#llst_files[@]} - GI_MAX_FRAMES ))

    for (( li_index = 0; li_index < li_excess; li_index++ )); do
        rm -f "${llst_files[li_index]}"
    done

} # remove_oldest_files

# -------------------------------------------------------------------------------------
# escreve a lista dos últimos GI_MAX_FRAMES PNGs no formato do demuxer
# concat do ffmpeg ("file '<caminho>'" por linha), em SLM_PNG/framelist.txt.
#
# por quê: depois da poda a numeração tem buracos (ex.: só 0004–0015) e o
# padrão frame_%04d.png do ffmpeg exige sequência contígua a partir de 0.
#
# uso: ls_list=$(generate_framelist)
# retorno: imprime o caminho da lista gerada
#
generate_framelist()
{
    local ls_list="${SLM_PNG}/framelist.txt"
    local ls_png

    ls -1 "${SLM_PNG}"/frame_*.png 2>/dev/null \
        | sort -V \
        | tail -n "${GI_MAX_FRAMES}" \
        | while IFS= read -r ls_png; do
              printf "file '%s'\n" "${ls_png}"
          done > "${ls_list}"

    echo "${ls_list}"

} # generate_framelist

# -------------------------------------------------------------------------------------
# gera o GIF animado (SLM_OUT/SLM_OUT_GIF) em duas passadas do ffmpeg:
# 1) uma paleta de cores otimizada para estes frames; 2) o GIF usando a
# paleta (fica bem melhor que a paleta padrão de 256 cores)
#
# uso: generate_gif [framelist]   (sem argumento, gera a própria lista)
# retorno: 0 também quando não há frames (só avisa)
#
generate_gif()
{
    local ls_list ls_palette

    log "[info]: animação: gerando GIF..."

    ls_list="${1:-$(generate_framelist)}"

    if [ ! -s "${ls_list}" ]; then
        log "[warn]: animação: sem frames para GIF."
        return 0
    fi

    ls_palette="${SLM_PNG}/palette.png"

    ffmpeg -y \
        -r "${GI_FPS}" \
        -f concat -safe 0 -i "${ls_list}" \
        -vf "scale=${GI_GIF_SCALE}:-1:flags=lanczos,palettegen" \
        "${ls_palette}" \
        >/dev/null 2>&1

    ffmpeg -y \
        -r "${GI_FPS}" \
        -f concat -safe 0 -i "${ls_list}" \
        -i "${ls_palette}" \
        -filter_complex "setpts=3.0*PTS,scale=${GI_GIF_SCALE}:-1:flags=lanczos[x];[x][1:v]paletteuse" \
        "${SLM_OUT}/${SLM_OUT_GIF}" \
        >/dev/null 2>&1

    log "[info]: animação: GIF atualizado."

} # generate_gif

# -------------------------------------------------------------------------------------
# gera o vídeo MP4 (SLM_OUT/SLM_OUT_MP4, H.264) a partir da lista de frames
#
# uso: generate_mp4 [framelist]   (sem argumento, gera a própria lista)
# retorno: 0 também quando não há frames (só avisa)
#
generate_mp4()
{
    local ls_list

    log "[info]: animação: gerando MP4..."

    ls_list="${1:-$(generate_framelist)}"

    if [ ! -s "${ls_list}" ]; then
        log "[warn]: animação: sem frames para MP4."
        return 0
    fi

    ffmpeg -y \
        -r "${GI_FPS}" \
        -f concat -safe 0 -i "${ls_list}" \
        -c:v libx264 \
        -pix_fmt yuv420p \
        -preset medium \
        -crf "${GI_MP4_CRF}" \
        "${SLM_OUT}/${SLM_OUT_MP4}" \
        >/dev/null 2>&1

    log "[info]: animação: MP4 atualizado."

} # generate_mp4

# < the end >--------------------------------------------------------------------------
