#!/usr/bin/env bash
#
# Agenda a atualização noturna das knowledge bases da LIV IA via cron (local).
#
# Instala (ou atualiza) uma entrada no crontab do seu usuário que roda
# scripts/update_kb.sh em um horário noturno. Idempotente: rodar de novo
# apenas atualiza a entrada, sem duplicar.
#
# Uso:
#   ./scripts/install_cron.sh                 # padrão: todo dia às 03:00
#   ./scripts/install_cron.sh "0 2 * * *"     # horário customizado (cron expr)
#   ./scripts/install_cron.sh --remove        # remove o agendamento
#
# Observações:
#   - É um cron LOCAL desta máquina (nada de TermHub, nada de web aberta).
#   - A máquina precisa estar ligada no horário para o job rodar. Se ela
#     costuma ficar desligada à noite, considere um horário em que ela esteja
#     ligada, ou use 'anacron' (não coberto aqui).
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPDATE_SCRIPT="${PROJECT_DIR}/scripts/update_kb.sh"
MARKER="# LIV-IA-UPDATE-KB"  # identifica nossa linha no crontab

DEFAULT_SCHEDULE="0 3 * * *"  # todo dia às 03:00 (noite)

remove_entry() {
  local current
  current="$(crontab -l 2>/dev/null || true)"
  if printf '%s\n' "${current}" | grep -qF "${MARKER}"; then
    printf '%s\n' "${current}" | grep -vF "${MARKER}" | crontab -
    echo "Agendamento removido do crontab."
  else
    echo "Nenhum agendamento da LIV IA encontrado no crontab."
  fi
}

if [ "${1:-}" = "--remove" ]; then
  remove_entry
  exit 0
fi

SCHEDULE="${1:-$DEFAULT_SCHEDULE}"

if [ ! -x "${UPDATE_SCRIPT}" ]; then
  chmod +x "${UPDATE_SCRIPT}" 2>/dev/null || true
fi

# Linha de cron: roda via bash, dentro do diretório do projeto.
CRON_LINE="${SCHEDULE} cd ${PROJECT_DIR} && /usr/bin/env bash scripts/update_kb.sh >/dev/null 2>&1 ${MARKER}"

# Reescreve o crontab preservando outras entradas e trocando a nossa.
current="$(crontab -l 2>/dev/null || true)"
new="$(printf '%s\n' "${current}" | grep -vF "${MARKER}" || true)"
new="$(printf '%s\n%s\n' "${new}" "${CRON_LINE}" | sed '/^$/d')"
printf '%s\n' "${new}" | crontab -

echo "Agendamento instalado:"
echo "  quando:  ${SCHEDULE}  (default = ${DEFAULT_SCHEDULE}, ou seja 03:00)"
echo "  roda:    ${UPDATE_SCRIPT}"
echo "  log:     ${PROJECT_DIR}/logs/update_kb.log"
echo
echo "Verifique com:  crontab -l"
echo "Para remover:   ./scripts/install_cron.sh --remove"
