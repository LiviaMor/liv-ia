"""
Testes unitários para o módulo ingestor.py
"""

from unittest.mock import Mock, patch

import pytest

from ingestor import DEFAULT_COLLECTION, SUPPORTED_LOADERS, DocumentProcessor


def _make_doc(content, source):
    """Cria um documento mockado com conteúdo e metadados."""
    doc = Mock()
    doc.page_content = content
    doc.metadata = {"source": source}
    return doc


class TestDocumentProcessor:
    """Testes para a classe DocumentProcessor"""

    @patch("ingestor.OllamaEmbeddings")
    def test_init(self, mock_embeddings):
        """Testa inicialização do processor"""
        processor = DocumentProcessor()

        assert processor.storage_path == ".livia_storage"
        mock_embeddings.assert_called_once()

    @patch("ingestor.OllamaEmbeddings")
    def test_init_custom_params(self, mock_embeddings):
        """Testa inicialização com parâmetros customizados"""
        processor = DocumentProcessor(
            storage_path="custom_storage",
            ollama_base_url="http://custom:11434",
        )

        assert processor.storage_path == "custom_storage"
        mock_embeddings.assert_called_once_with(
            model="nomic-embed-text",
            base_url="http://custom:11434",
        )

    @patch("ingestor.OllamaEmbeddings")
    def test_supported_extensions_includes_multiple_formats(self, mock_embeddings):
        """A ingestão deve suportar PDF, Markdown, texto e código."""
        extensions = DocumentProcessor.supported_extensions()

        for ext in (".pdf", ".md", ".txt", ".py", ".js", ".json", ".yaml"):
            assert ext in extensions

    @patch("ingestor.OllamaEmbeddings")
    @patch("ingestor.os.path.exists", return_value=False)
    def test_ingest_directory_path_not_found(self, mock_exists, mock_embeddings):
        """Testa erro quando pasta não existe"""
        processor = DocumentProcessor()

        with pytest.raises(FileNotFoundError):
            processor.ingest_directory("nonexistent_path")

    @patch("ingestor.OllamaEmbeddings")
    @patch("ingestor.os.path.exists", return_value=True)
    @patch("ingestor.DirectoryLoader")
    def test_ingest_directory_no_documents(self, mock_loader, mock_exists, mock_embeddings):
        """Testa erro quando não há documentos suportados"""
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = []
        mock_loader.return_value = mock_loader_instance

        processor = DocumentProcessor()

        with pytest.raises(ValueError):
            processor.ingest_directory("empty_path")

    @patch("ingestor.OllamaEmbeddings")
    @patch("ingestor.os.path.exists", return_value=True)
    @patch("ingestor.DirectoryLoader")
    @patch("ingestor.Chroma")
    def test_ingest_directory_success(self, mock_chroma, mock_loader, mock_exists, mock_embeddings):
        """Testa ingestão bem-sucedida com documentos de formatos variados"""
        # Primeira extensão devolve dois docs; as demais, nada.
        mock_loader_instance = Mock()
        mock_loader_instance.load.side_effect = [
            [_make_doc("Conteúdo 1", "doc1.pdf"), _make_doc("Conteúdo 2", "notes.md")]
        ] + [[] for _ in range(len(SUPPORTED_LOADERS) - 1)]
        mock_loader.return_value = mock_loader_instance

        mock_vectorstore = Mock()
        mock_chroma.from_documents.return_value = mock_vectorstore

        processor = DocumentProcessor()
        count = processor.ingest_directory("test_path")

        assert count == 2
        mock_chroma.from_documents.assert_called_once()
        # Usa a collection padrão quando nenhuma é informada.
        _, kwargs = mock_chroma.from_documents.call_args
        assert kwargs["collection_name"] == DEFAULT_COLLECTION
        mock_vectorstore.persist.assert_called_once()

    @patch("ingestor.OllamaEmbeddings")
    @patch("ingestor.os.path.exists", return_value=True)
    @patch("ingestor.DirectoryLoader")
    @patch("ingestor.Chroma")
    def test_ingest_directory_named_knowledge_base(
        self, mock_chroma, mock_loader, mock_exists, mock_embeddings
    ):
        """Cada knowledge base deve virar uma collection nomeada no Chroma."""
        mock_loader_instance = Mock()
        mock_loader_instance.load.side_effect = [[_make_doc("React docs", "react.md")]] + [
            [] for _ in range(len(SUPPORTED_LOADERS) - 1)
        ]
        mock_loader.return_value = mock_loader_instance

        mock_chroma.from_documents.return_value = Mock()

        processor = DocumentProcessor()
        count = processor.ingest_directory("test_path", collection_name="react")

        assert count == 1
        _, kwargs = mock_chroma.from_documents.call_args
        assert kwargs["collection_name"] == "react"

    @patch("ingestor.OllamaEmbeddings")
    @patch("ingestor.os.path.exists", return_value=True)
    @patch("ingestor.DirectoryLoader")
    @patch("ingestor.Chroma")
    def test_persist_compatible_with_new_chroma(
        self, mock_chroma, mock_loader, mock_exists, mock_embeddings
    ):
        """Se Chroma.persist() lançar/ não existir, não deve quebrar a ingestão."""
        mock_loader_instance = Mock()
        mock_loader_instance.load.side_effect = [[_make_doc("x", "a.txt")]] + [
            [] for _ in range(len(SUPPORTED_LOADERS) - 1)
        ]
        mock_loader.return_value = mock_loader_instance

        # Vectorstore sem persist() (comportamento das versões novas).
        mock_vectorstore = Mock(spec=[])
        mock_chroma.from_documents.return_value = mock_vectorstore

        processor = DocumentProcessor()
        # Não deve levantar exceção mesmo sem persist().
        assert processor.ingest_directory("p") == 1

    @patch("ingestor.OllamaEmbeddings")
    @patch("ingestor.os.path.exists", return_value=True)
    @patch("ingestor.DirectoryLoader")
    @patch("ingestor.Chroma")
    def test_ingest_pdfs_backward_compatible(
        self, mock_chroma, mock_loader, mock_exists, mock_embeddings
    ):
        """ingest_pdfs continua funcionando (delegando para ingest_directory)."""
        mock_loader_instance = Mock()
        mock_loader_instance.load.side_effect = [[_make_doc("pdf content", "doc.pdf")]] + [
            [] for _ in range(len(SUPPORTED_LOADERS) - 1)
        ]
        mock_loader.return_value = mock_loader_instance

        mock_vectorstore = Mock()
        mock_chroma.from_documents.return_value = mock_vectorstore

        processor = DocumentProcessor()
        count = processor.ingest_pdfs("test_path")

        assert count == 1
        mock_vectorstore.persist.assert_called_once()


class TestTextSplitter:
    """Testes para o text splitter"""

    @patch("ingestor.OllamaEmbeddings")
    def test_text_splitter_config(self, mock_embeddings):
        """Testa configuração do text splitter"""
        processor = DocumentProcessor()

        assert processor.text_splitter._chunk_size == 1000
        assert processor.text_splitter._chunk_overlap == 200


class TestMarkdownChunking:
    """Testes para o chunking por cabeçalho em Markdown"""

    @patch("ingestor.OllamaEmbeddings")
    def test_is_markdown_detection(self, mock_embeddings):
        processor = DocumentProcessor()
        md = Mock()
        md.metadata = {"source": "guia.md"}
        py = Mock()
        py.metadata = {"source": "app.py"}

        assert processor._is_markdown(md) is True
        assert processor._is_markdown(py) is False

    @patch("ingestor.OllamaEmbeddings")
    def test_markdown_split_by_headers(self, mock_embeddings):
        """Markdown é quebrado por seções (cabeçalhos), preservando a fonte."""
        from langchain_core.documents import Document

        processor = DocumentProcessor()
        content = (
            "# Título\n\nIntro.\n\n" "## Seção A\n\nConteúdo A.\n\n" "## Seção B\n\nConteúdo B.\n"
        )
        doc = Document(page_content=content, metadata={"source": "guia.md"})

        chunks = processor._split_documents([doc])

        # Deve gerar mais de um chunk (uma por seção, no mínimo).
        assert len(chunks) >= 2
        # Todos preservam a fonte original.
        assert all(c.metadata.get("source") == "guia.md" for c in chunks)

    @patch("ingestor.OllamaEmbeddings")
    def test_non_markdown_uses_recursive_splitter(self, mock_embeddings):
        """Arquivos não-Markdown usam o split por tamanho normalmente."""
        from langchain_core.documents import Document

        processor = DocumentProcessor()
        doc = Document(page_content="print('hello')\n" * 50, metadata={"source": "app.py"})

        chunks = processor._split_documents([doc])

        assert len(chunks) >= 1
        assert all(c.metadata.get("source") == "app.py" for c in chunks)
