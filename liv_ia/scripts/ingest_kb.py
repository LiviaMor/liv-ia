#!/usr/bin/env python3
"""
Indexa as knowledge bases baixadas por bootstrap_kb.sh.

Cada pasta em knowledge/ é indexada em uma collection separada do Chroma:

    knowledge/nodejs/        -> collection "nodejs"
    knowledge/microservices/ -> collection "microservices"
    knowledge/gps/           -> collection "gps"

Uso:
    python scripts/ingest_kb.py              # indexa todas as bases encontradas
    python scripts/ingest_kb.py nodejs gps   # indexa apenas as bases informadas

Requer o Ollama rodando (docker compose up -d) com o modelo nomic-embed-text,
pois a geração de embeddings acontece aqui.
"""

import sys
from pathlib import Path

# Permite rodar a partir da raiz do projeto, importando ingestor.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestor import DocumentProcessor  # noqa: E402

KB_ROOT = PROJECT_ROOT / "knowledge"

# Mapa pasta -> nome da collection (knowledge base).
KNOWLEDGE_BASES = {
    "nodejs": "nodejs",
    "microservices": "microservices",
    "gps": "gps",
}


def ingest_one(processor: DocumentProcessor, folder: str, collection: str) -> bool:
    """Indexa uma base. Retorna True em sucesso, False em falha/pulada."""
    path = KB_ROOT / folder
    if not path.exists():
        print(f"[skip] {folder}: pasta não encontrada ({path}). Rode bootstrap_kb.sh antes.")
        return False
    try:
        count = processor.ingest_directory(str(path), collection_name=collection)
        print(f"[ok]   {folder}: {count} documento(s) indexado(s) na base '{collection}'.")
        return True
    except ValueError as exc:
        print(f"[vazio] {folder}: {exc}")
        return False
    except Exception as exc:  # pragma: no cover - erros de runtime do Ollama/Chroma
        print(f"[erro] {folder}: {exc}")
        return False


def main(argv) -> int:
    targets = argv or list(KNOWLEDGE_BASES.keys())
    unknown = [t for t in targets if t not in KNOWLEDGE_BASES]
    if unknown:
        print(f"Base(s) desconhecida(s): {', '.join(unknown)}")
        print(f"Disponíveis: {', '.join(KNOWLEDGE_BASES)}")
        return 1

    processor = DocumentProcessor()
    results = {folder: ingest_one(processor, folder, KNOWLEDGE_BASES[folder]) for folder in targets}

    failed = [folder for folder, ok in results.items() if not ok]
    if failed:
        print(f"\nFalha ao indexar: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
