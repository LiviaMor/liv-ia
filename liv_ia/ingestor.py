"""
LIV IA Document Processor - Processamento e indexação de documentos.

Suporta múltiplos formatos (PDF, Markdown, texto e código-fonte) e permite
organizar o conhecimento em "knowledge bases" separadas, cada uma mapeada para
uma collection nomeada no ChromaDB. Assim é possível, por exemplo, manter uma
base "react", outra "aws" e outra "healthtech" de forma isolada.
"""

import os
from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# Collection padrão usada quando nenhuma knowledge base é informada.
# Mantém compatibilidade com o comportamento anterior (base única).
DEFAULT_COLLECTION = "livia_default"

# Mapa de extensão -> (loader_cls, loader_kwargs).
# PDFs usam o PyPDFLoader; todo o resto (texto e código) usa o TextLoader.
_TEXT_LOADER_KWARGS = {"encoding": "utf-8", "autodetect_encoding": True}
SUPPORTED_LOADERS = {
    ".pdf": (PyPDFLoader, {}),
    ".md": (TextLoader, _TEXT_LOADER_KWARGS),
    ".markdown": (TextLoader, _TEXT_LOADER_KWARGS),
    ".txt": (TextLoader, _TEXT_LOADER_KWARGS),
    ".rst": (TextLoader, _TEXT_LOADER_KWARGS),
    ".py": (TextLoader, _TEXT_LOADER_KWARGS),
    ".js": (TextLoader, _TEXT_LOADER_KWARGS),
    ".ts": (TextLoader, _TEXT_LOADER_KWARGS),
    ".tsx": (TextLoader, _TEXT_LOADER_KWARGS),
    ".jsx": (TextLoader, _TEXT_LOADER_KWARGS),
    ".java": (TextLoader, _TEXT_LOADER_KWARGS),
    ".go": (TextLoader, _TEXT_LOADER_KWARGS),
    ".rs": (TextLoader, _TEXT_LOADER_KWARGS),
    ".rb": (TextLoader, _TEXT_LOADER_KWARGS),
    ".php": (TextLoader, _TEXT_LOADER_KWARGS),
    ".c": (TextLoader, _TEXT_LOADER_KWARGS),
    ".h": (TextLoader, _TEXT_LOADER_KWARGS),
    ".cpp": (TextLoader, _TEXT_LOADER_KWARGS),
    ".cs": (TextLoader, _TEXT_LOADER_KWARGS),
    ".json": (TextLoader, _TEXT_LOADER_KWARGS),
    ".yaml": (TextLoader, _TEXT_LOADER_KWARGS),
    ".yml": (TextLoader, _TEXT_LOADER_KWARGS),
    ".toml": (TextLoader, _TEXT_LOADER_KWARGS),
    ".ini": (TextLoader, _TEXT_LOADER_KWARGS),
    ".cfg": (TextLoader, _TEXT_LOADER_KWARGS),
    ".sql": (TextLoader, _TEXT_LOADER_KWARGS),
    ".sh": (TextLoader, _TEXT_LOADER_KWARGS),
    ".html": (TextLoader, _TEXT_LOADER_KWARGS),
    ".css": (TextLoader, _TEXT_LOADER_KWARGS),
}


class DocumentProcessor:
    def __init__(
        self,
        storage_path=".livia_storage",
        ollama_base_url="http://localhost:11434",
    ):
        self.storage_path = storage_path
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=ollama_base_url,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        # Para Markdown, quebramos primeiro por cabeçalhos (seções), preservando
        # o contexto estrutural do documento, e só depois aplicamos o split por
        # tamanho nas seções grandes. Isso melhora a precisão do retrieval.
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ],
            strip_headers=False,
        )

    @staticmethod
    def _is_markdown(doc) -> bool:
        """Diz se um documento veio de um arquivo Markdown."""
        metadata = getattr(doc, "metadata", {}) or {}
        source = str(metadata.get("source", "")).lower()
        return source.endswith(".md") or source.endswith(".markdown")

    def _split_documents(self, documents):
        """Divide documentos em chunks, com tratamento especial para Markdown.

        Markdown é quebrado por cabeçalhos (seções) e depois por tamanho;
        os demais formatos usam apenas o split recursivo por tamanho.
        """
        chunks = []
        for doc in documents:
            if self._is_markdown(doc):
                sections = self.markdown_splitter.split_text(doc.page_content)
                # Mantém os metadados originais (ex.: source) em cada seção.
                for section in sections:
                    section.metadata = {**doc.metadata, **section.metadata}
                chunks.extend(self.text_splitter.split_documents(sections))
            else:
                chunks.extend(self.text_splitter.split_documents([doc]))
        return chunks

    @staticmethod
    def supported_extensions():
        """Retorna as extensões de arquivo suportadas pela ingestão."""
        return sorted(SUPPORTED_LOADERS.keys())

    def _load_documents(self, path: str):
        """Carrega todos os arquivos suportados de uma pasta (recursivo).

        Percorre o mapa de extensões e usa um DirectoryLoader por tipo,
        acumulando os documentos de todos os formatos encontrados.
        """
        documents = []
        for extension, (loader_cls, loader_kwargs) in SUPPORTED_LOADERS.items():
            loader = DirectoryLoader(
                path,
                glob=f"**/*{extension}",
                loader_cls=loader_cls,
                loader_kwargs=loader_kwargs or None,
                show_progress=False,
                silent_errors=True,
            )
            documents.extend(loader.load())
        return documents

    def _persist(self, vectorstore):
        """Persiste o vectorstore de forma compatível entre versões.

        O método Chroma.persist() foi removido em versões novas do
        langchain-chroma (a escrita passou a ser automática). Chamamos de
        forma defensiva para funcionar nas duas situações.
        """
        persist = getattr(vectorstore, "persist", None)
        if callable(persist):
            try:
                persist()
            except Exception:
                # Versões novas persistem automaticamente; ignorar é seguro.
                pass

    def ingest_directory(self, path: str, collection_name: str = DEFAULT_COLLECTION) -> int:
        """Indexa todos os arquivos suportados de uma pasta.

        Args:
            path: pasta com os documentos (varredura recursiva).
            collection_name: nome da knowledge base (collection no Chroma).

        Returns:
            Quantidade de documentos carregados (antes do split em chunks).
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pasta não encontrada: {path}")

        documents = self._load_documents(path)

        if not documents:
            raise ValueError(
                f"Nenhum documento suportado encontrado em: {path} "
                f"(formatos: {', '.join(self.supported_extensions())})"
            )

        chunks = self._split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.storage_path,
            collection_name=collection_name,
        )
        self._persist(vectorstore)

        return len(documents)

    def ingest_pdfs(self, pdf_path: str, collection_name: str = DEFAULT_COLLECTION) -> int:
        """Compatibilidade retroativa: indexa uma pasta de documentos.

        Mantido para não quebrar o main.py e integrações existentes. Hoje
        delega para ingest_directory, que também processa PDFs.
        """
        return self.ingest_directory(pdf_path, collection_name=collection_name)
