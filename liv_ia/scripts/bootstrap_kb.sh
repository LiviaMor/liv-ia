#!/usr/bin/env bash
#
# Bootstrap das knowledge bases da LIV IA.
#
# Baixa documentação curada para três domínios e organiza em pastas separadas,
# prontas para serem indexadas (cada pasta vira uma collection no Chroma via
# scripts/ingest_kb.py).
#
#   knowledge/nodejs/        -> Node.js (docs oficiais da API, Markdown)
#   knowledge/microservices/ -> arquitetura event-driven e microsserviços
#   knowledge/gps/           -> funcionamento do GPS (spec oficial IS-GPS-200)
#
# Uso:
#   ./scripts/bootstrap_kb.sh            # baixa tudo
#   ./scripts/bootstrap_kb.sh nodejs     # baixa só uma base
#
# Observação de licença:
#   - Node.js docs: MIT (repo nodejs/node).
#   - IS-GPS-200: domínio público ("Approved for public release").
#   - Fontes de microsserviços: ver URLs; respeite a licença de cada repo.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KB_DIR="${ROOT_DIR}/knowledge"
mkdir -p "${KB_DIR}"

log() { printf '\033[36m[bootstrap]\033[0m %s\n' "$*"; }
err() { printf '\033[31m[erro]\033[0m %s\n' "$*" >&2; }

# Clona apenas uma subpasta de um repo (sparse checkout) — econômico.
sparse_clone() {
  local repo_url="$1" subdir="$2" dest="$3"
  rm -rf "$dest"
  git clone --depth 1 --filter=blob:none --sparse "$repo_url" "$dest" >/dev/null 2>&1
  (cd "$dest" && git sparse-checkout set "$subdir" >/dev/null 2>&1)
}

fetch_nodejs() {
  log "Node.js: baixando docs oficiais da API (nodejs/node:doc/api)..."
  local dest="${KB_DIR}/nodejs"
  sparse_clone "https://github.com/nodejs/node.git" "doc/api" "$dest"
  local n
  n=$(find "$dest/doc/api" -name '*.md' 2>/dev/null | wc -l | tr -d ' ')
  log "Node.js: ${n} arquivos .md em ${dest}/doc/api"
}

fetch_reactnative() {
  log "React Native: baixando docs oficiais (facebook/react-native-website:docs)..."
  local dest="${KB_DIR}/reactnative"
  sparse_clone "https://github.com/facebook/react-native-website.git" "docs" "$dest"
  local n
  n=$(find "$dest/docs" -name '*.md' 2>/dev/null | wc -l | tr -d ' ')
  log "React Native: ${n} arquivos .md em ${dest}/docs"
}

fetch_microservices() {
  log "Microsserviços: baixando padrões event-driven e microsserviços..."
  local dest="${KB_DIR}/microservices"
  mkdir -p "$dest"
  # Awesome lists e material em Markdown de referência (licenças abertas).
  # Ajuste/adicione fontes conforme sua necessidade.
  local srcs=(
    "https://raw.githubusercontent.com/mfornos/awesome-microservices/master/README.md|awesome-microservices.md"
    "https://raw.githubusercontent.com/mehdihadeli/awesome-software-architecture/main/README.md|awesome-software-architecture.md"
    "https://raw.githubusercontent.com/donnemartin/system-design-primer/master/README.md|system-design-primer.md"
  )
  local pair url name ok=0
  for pair in "${srcs[@]}"; do
    url="${pair%%|*}"; name="${pair##*|}"
    if curl -fsSL "$url" -o "${dest}/${name}" 2>/dev/null; then
      log "  ok: ${name}"
      ok=$((ok + 1))
    else
      err "  falhou (ignorado): ${url}"
    fi
  done
  log "Microsserviços: ${ok} arquivo(s) em ${dest}"
  if [ "$ok" -eq 0 ]; then
    err "Nenhuma fonte de microsserviços baixou. Adicione seus próprios .md em ${dest}"
  fi
}

fetch_gps() {
  log "GPS: baixando especificação oficial IS-GPS-200N (gps.gov, domínio público)..."
  local dest="${KB_DIR}/gps"
  mkdir -p "$dest"
  local url="https://www.gps.gov/sites/default/files/2025-07/IS-GPS-200N.pdf"
  if curl -fsSL "$url" -o "${dest}/IS-GPS-200N.pdf" 2>/dev/null; then
    log "GPS: IS-GPS-200N.pdf salvo em ${dest}"
  else
    err "GPS: falha ao baixar ${url}"
    err "     Baixe manualmente de https://www.gps.gov/technical/icwg/ e coloque em ${dest}"
  fi
}

main() {
  local target="${1:-all}"
  case "$target" in
    nodejs)        fetch_nodejs ;;
    reactnative)   fetch_reactnative ;;
    microservices) fetch_microservices ;;
    gps)           fetch_gps ;;
    all)           fetch_nodejs; fetch_reactnative; fetch_microservices; fetch_gps ;;
    *) err "Alvo inválido: ${target} (use: nodejs | reactnative | microservices | gps | all)"; exit 1 ;;
  esac
  log "Concluído. Próximo passo: ./scripts/ingest_kb.py"
}

main "$@"
