#!/usr/bin/env bash
#
# Atualização incremental das knowledge bases da LIV IA.
#
# Re-baixa a documentação das fontes CURADAS E FIXAS (definidas em
# bootstrap_kb.sh) e re-indexa tudo no Chroma. Pensado para rodar via cron,
# tipicamente à noite. É seguro por design:
#   - só usa as fontes confiáveis já definidas (sem navegação web aberta);
#   - roda uma vez e termina (sem loop autônomo, sem auto-treino);
#   - usa um lock para nunca sobrepor execuções;
#   - registra tudo em log com timestamp.
#
# Uso:
#   ./scripts/update_kb.sh                 # atualiza todas as bases
#   ./scripts/update_kb.sh nodejs gps      # atualiza apenas as informadas
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"
VENV_PY="${PROJECT_DIR}/.venv/bin/python"
LOG_DIR="${PROJECT_DIR}/logs"
LOG_FILE="${LOG_DIR}/update_kb.log"
LOCK_FILE="${PROJECT_DIR}/.update_kb.lock"

mkdir -p "${LOG_DIR}"

ts() { date "+%Y-%m-%d %H:%M:%S"; }
log() { printf '%s [update_kb] %s\n' "$(ts)" "$*" | tee -a "${LOG_FILE}"; }

# Garante execução única: se já houver uma rodando, aborta sem erro.
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  log "Já existe uma atualização em andamento. Abortando esta execução."
  exit 0
fi

# Escolhe o interpretador: prefere o venv do projeto; cai pro python3 do sistema.
if [ -x "${VENV_PY}" ]; then
  PYTHON="${VENV_PY}"
else
  PYTHON="$(command -v python3 || true)"
fi

if [ -z "${PYTHON}" ]; then
  log "ERRO: nenhum interpretador Python encontrado (nem .venv nem python3)."
  exit 1
fi

TARGETS=("$@")

log "=== Início da atualização (alvos: ${TARGETS[*]:-todos}) ==="

# 1) Re-baixa as fontes fixas e confiáveis.
log "Baixando fontes (bootstrap_kb.sh)..."
if bash "${SCRIPTS_DIR}/bootstrap_kb.sh" "${TARGETS[@]}" >>"${LOG_FILE}" 2>&1; then
  log "Download concluído."
else
  log "AVISO: bootstrap retornou erro; seguindo para indexar o que houver."
fi

# 2) Re-indexa no Chroma (requer Ollama no ar para gerar embeddings).
log "Indexando (ingest_kb.py)..."
if "${PYTHON}" "${SCRIPTS_DIR}/ingest_kb.py" "${TARGETS[@]}" >>"${LOG_FILE}" 2>&1; then
  log "Indexação concluída com sucesso."
else
  log "ERRO: a indexação falhou. Verifique se o Ollama está rodando (docker compose up -d)."
  exit 1
fi

log "=== Fim da atualização ==="
