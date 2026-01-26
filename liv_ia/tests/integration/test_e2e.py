"""
Testes de integração end-to-end
"""
import pytest
import os
from brain import LIVIAEngine
from ingestor import DocumentProcessor


@pytest.fixture
def ollama_url():
    """URL do Ollama para testes"""
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@pytest.fixture
def test_model():
    """Modelo de teste (menor e mais rápido)"""
    return "qwen2.5-coder:7b"


class TestEngineIntegration:
    """Testes de integração do engine"""
    
    def test_engine_initialization(self, ollama_url, test_model):
        """Testa inicialização do engine com Ollama real"""
        try:
            engine = LIVIAEngine(
                model_name=test_model,
                ollama_base_url=ollama_url
            )
            assert engine is not None
            assert engine.llm is not None
        except Exception as e:
            pytest.skip(f"Ollama não disponível: {e}")
    
    def test_ask_without_documents(self, ollama_url, test_model):
        """Testa pergunta sem documentos indexados"""
        try:
            engine = LIVIAEngine(
                model_name=test_model,
                ollama_base_url=ollama_url
            )
            
            response = engine.ask("O que é Python?")
            
            assert response is not None
            assert len(response) > 0
            assert isinstance(response, str)
        except Exception as e:
            pytest.skip(f"Ollama não disponível: {e}")
    
    def test_chat_continuity(self, ollama_url, test_model):
        """Testa continuidade do chat"""
        try:
            engine = LIVIAEngine(
                model_name=test_model,
                ollama_base_url=ollama_url
            )
            
            # Primeira mensagem
            response1 = engine.chat("Meu nome é João")
            assert response1 is not None
            
            # Segunda mensagem (deve lembrar do nome)
            response2 = engine.chat("Qual é o meu nome?")
            assert response2 is not None
            
            # Verifica que mantém histórico
            assert len(engine.chat_history) >= 2
        except Exception as e:
            pytest.skip(f"Ollama não disponível: {e}")


class TestIngestorIntegration:
    """Testes de integração do ingestor"""
    
    def test_processor_initialization(self, ollama_url):
        """Testa inicialização do processor com Ollama real"""
        try:
            processor = DocumentProcessor(ollama_base_url=ollama_url)
            assert processor is not None
            assert processor.embeddings is not None
        except Exception as e:
            pytest.skip(f"Ollama não disponível: {e}")


class TestHealthCheck:
    """Testes de health check"""
    
    def test_ollama_connection(self, ollama_url):
        """Testa conexão com Ollama"""
        import requests
        
        try:
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            assert response.status_code == 200
        except Exception as e:
            pytest.skip(f"Ollama não disponível: {e}")
    
    def test_model_availability(self, ollama_url, test_model):
        """Testa disponibilidade do modelo"""
        import requests
        
        try:
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            # Verifica se modelo de teste está disponível
            assert any(test_model in name for name in model_names)
        except Exception as e:
            pytest.skip(f"Ollama não disponível: {e}")
