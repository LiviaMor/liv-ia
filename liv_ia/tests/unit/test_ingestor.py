"""
Testes unitários para o módulo ingestor.py
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from ingestor import DocumentProcessor


class TestDocumentProcessor:
    """Testes para a classe DocumentProcessor"""
    
    @patch('ingestor.OllamaEmbeddings')
    def test_init(self, mock_embeddings):
        """Testa inicialização do processor"""
        processor = DocumentProcessor()
        
        assert processor.storage_path == ".livia_storage"
        mock_embeddings.assert_called_once()
    
    @patch('ingestor.OllamaEmbeddings')
    def test_init_custom_params(self, mock_embeddings):
        """Testa inicialização com parâmetros customizados"""
        processor = DocumentProcessor(
            storage_path="custom_storage",
            ollama_base_url="http://custom:11434"
        )
        
        assert processor.storage_path == "custom_storage"
        mock_embeddings.assert_called_once_with(
            model="nomic-embed-text",
            base_url="http://custom:11434"
        )
    
    @patch('ingestor.OllamaEmbeddings')
    @patch('ingestor.os.path.exists', return_value=False)
    def test_ingest_pdfs_path_not_found(self, mock_exists, mock_embeddings):
        """Testa erro quando pasta não existe"""
        processor = DocumentProcessor()
        
        with pytest.raises(FileNotFoundError):
            processor.ingest_pdfs("nonexistent_path")
    
    @patch('ingestor.OllamaEmbeddings')
    @patch('ingestor.os.path.exists', return_value=True)
    @patch('ingestor.DirectoryLoader')
    def test_ingest_pdfs_no_documents(
        self, mock_loader, mock_exists, mock_embeddings
    ):
        """Testa erro quando não há PDFs"""
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = []
        mock_loader.return_value = mock_loader_instance
        
        processor = DocumentProcessor()
        
        with pytest.raises(ValueError):
            processor.ingest_pdfs("empty_path")
    
    @patch('ingestor.OllamaEmbeddings')
    @patch('ingestor.os.path.exists', return_value=True)
    @patch('ingestor.DirectoryLoader')
    @patch('ingestor.Chroma')
    def test_ingest_pdfs_success(
        self, mock_chroma, mock_loader, mock_exists, mock_embeddings
    ):
        """Testa ingestão bem-sucedida"""
        # Mock documents com metadados corretos
        mock_doc1 = Mock()
        mock_doc1.page_content = "Conteúdo do documento 1"
        mock_doc1.metadata = {"source": "doc1.pdf", "page": 1}
        
        mock_doc2 = Mock()
        mock_doc2.page_content = "Conteúdo do documento 2"
        mock_doc2.metadata = {"source": "doc2.pdf", "page": 1}
        
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [mock_doc1, mock_doc2]
        mock_loader.return_value = mock_loader_instance
        
        # Mock vectorstore
        mock_vectorstore = Mock()
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        processor = DocumentProcessor()
        count = processor.ingest_pdfs("test_path")
        
        assert count == 2
        mock_vectorstore.persist.assert_called_once()


class TestTextSplitter:
    """Testes para o text splitter"""
    
    @patch('ingestor.OllamaEmbeddings')
    def test_text_splitter_config(self, mock_embeddings):
        """Testa configuração do text splitter"""
        processor = DocumentProcessor()
        
        assert processor.text_splitter._chunk_size == 1000
        assert processor.text_splitter._chunk_overlap == 200
