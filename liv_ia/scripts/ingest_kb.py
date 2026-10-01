#!/usr/bin/env python3
"""
Indexa a documentação baixada por bootstrap_kb.sh na base única da LIV IA.

Todas as pastas em knowledge/ (nodejs, microservices, gps, ...) são indexadas
na MESMA collection ("livia_default"). A busca é unificada: a LIV IA procura
em todo o conhecimento, sem fragmentar por tema.

Uso:
    python scripts/ingest_kb.py                 # indexa todas as pastas
    python scripts/ingest_kb.py nodejs gps      # indexa apenas as pastas dadas

Requer o Ollama rodando (docker compose up -d) com o modelo nomic-embed-text,
pois a geração de embeddings acontece aqui.
"""

import sys
from pathlib import Path

# Permite rodar a partir da raiz do projeto, importando ingestor.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestor import DEFAULT_COLLECTION, DocumentProcessor  # noqa: E402

KB_ROOT = PROJECT_ROOT / "knowledge"


def _available_folders():
    """Lista as subpastas de knowledge/ (cada uma é uma fonte de documentos)."""
    if not KB_ROOT.exists():
        return []
    return sorted(p.name for p in KB_ROOT.iterdir() if p.is_dir())


def ingest_one(processor: DocumentProcessor, folder: str) -> bool:
    """Indexa uma pasta na base única. Retorna True em sucesso."""
    path = KB_ROOT / folder
    if not path.exists():
        print(f"[skip] {folder}: pasta não encontrada ({path}). Rode bootstrap_kb.sh antes.")
        return False
    try:
        count = processor.ingest_directory(str(path), collection_name=DEFAULT_COLLECTION)
        print(f"[ok]   {folder}: {count} documento(s) indexado(s) na base única.")
        return True
    except ValueError as exc:
        print(f"[vazio] {folder}: {exc}")
        return False
    except Exception as exc:  # pragma: no cover - erros de runtime do Ollama/Chroma
        print(f"[erro] {folder}: {exc}")
        return False


def main(argv) -> int:
    targets = argv or _available_folders()
    if not targets:
        print("Nenhuma pasta em knowledge/. Rode scripts/bootstrap_kb.sh primeiro.")
        return 1

    processor = DocumentProcessor()
    results = {folder: ingest_one(processor, folder) for folder in targets}

    failed = [folder for folder, ok in results.items() if not ok]
    if failed:
        print(f"\nFalha ao indexar: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
