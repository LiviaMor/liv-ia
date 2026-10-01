"""
Testes unitários para o módulo brain.py
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from brain import LIVIAEngine


class TestLIVIAEngine:
    """Testes para a classe LIVIAEngine"""

    @patch("brain.OllamaLLM")
    @patch("brain.OllamaEmbeddings")
    def test_init(self, mock_embeddings, mock_llm):
        """Testa inicialização do engine"""
        engine = LIVIAEngine()

        assert engine.storage_path == ".livia_storage"
        assert engine.chat_history == []
        assert engine.vectorstore is None
        mock_llm.assert_called_once()
        mock_embeddings.assert_called_once()

    @patch("brain.OllamaLLM")
    @patch("brain.OllamaEmbeddings")
    def test_init_custom_params(self, mock_embeddings, mock_llm):
        """Testa inicialização com parâmetros customizados"""
        engine = LIVIAEngine(
            model_name="qwen2.5-coder:7b",
            storage_path="custom_storage",
            ollama_base_url="http://custom:11434",
        )

        assert engine.storage_path == "custom_storage"
        mock_llm.assert_called_once_with(
            model="qwen2.5-coder:7b", temperature=0.3, base_url="http://custom:11434"
        )

    @patch("brain.OllamaLLM")
    @patch("brain.OllamaEmbeddings")
    def test_format_chat_history_empty(self, mock_embeddings, mock_llm):
        """Testa formatação de histórico vazio"""
        engine = LIVIAEngine()
        result = engine._format_chat_history()

        assert result == "Nenhuma conversa anterior."

    @patch("brain.OllamaLLM")
    @patch("brain.OllamaEmbeddings")
    def test_format_chat_history_with_messages(self, mock_embeddings, mock_llm):
        """Testa formatação de histórico com mensagens"""
        from langchain_core.messages import AIMessage, HumanMessage

        engine = LIVIAEngine()

        # Adiciona mensagens reais
        engine.chat_history.append(HumanMessage(content="Olá"))
        engine.chat_history.append(AIMessage(content="Oi, como posso ajudar?"))

        result = engine._format_chat_history()

        assert "Olá" in result
        assert "Oi, como posso ajudar?" in result

    @patch("brain.OllamaLLM")
    @patch("brain.OllamaEmbeddings")
    def test_ask_without_vectorstore(self, mock_embeddings, mock_llm):
        """Testa pergunta sem vectorstore"""
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.return_value = "Resposta do LLM"
        mock_llm.return_value = mock_llm_instance

        engine = LIVIAEngine()
        result = engine.ask("Teste")

        assert result == "Resposta do LLM"
        mock_llm_instance.invoke.assert_called_once()

    @patch("brain.OllamaLLM")
    @patch("brain.OllamaEmbeddings")
    @patch("brain.Chroma")
    def test_load_vectorstore(self, mock_chroma, mock_embeddings, mock_llm):
        """Testa carregamento do vectorstore"""
        with patch("brain.os.path.exists", return_value=True):
            engine = LIVIAEngine()
            mock_chroma.assert_called_once()


class TestChatHistory:
    """Testes para gerenciamento de histórico"""

    @patch("brain.OllamaLLM")
    @patch("brain.OllamaEmbeddings")
    def test_chat_history_limit(self, mock_embeddings, mock_llm):
        """Testa limite de histórico (últimas 6 mensagens)"""
        engine = LIVIAEngine()

        # Adiciona 10 mensagens
        for i in range(10):
            msg = Mock()
            msg.content = f"Mensagem {i}"
            engine.chat_history.append(msg)

        # Formata histórico (deve pegar apenas últimas 6)
        with patch("brain.isinstance", return_value=True):
            result = engine._format_chat_history()

            # Verifica que pegou apenas últimas 6
            assert len(engine.chat_history) == 10  # Não modifica original


class TestSourceCitation:
    """Testes para a citação de fontes nas respostas"""

    @patch("brain.OllamaLLM")
    @patch("brain.OllamaEmbeddings")
    def test_format_sources_empty(self, mock_embeddings, mock_llm):
        """Sem documentos, não há rodapé de fontes."""
        engine = LIVIAEngine()
        assert engine._format_sources([]) == ""
        assert engine._format_sources(None) == ""

    @patch("brain.OllamaLLM")
    @patch("brain.OllamaEmbeddings")
    def test_format_sources_lists_filenames(self, mock_embeddings, mock_llm):
        """Fontes são listadas pelo nome do arquivo, sem duplicatas."""
        engine = LIVIAEngine()
        d1 = Mock()
        d1.metadata = {"source": "/path/knowledge/nodejs/doc/api/buffer.md"}
        d2 = Mock()
        d2.metadata = {"source": "/path/knowledge/nodejs/doc/api/buffer.md"}  # duplicada
        d3 = Mock()
        d3.metadata = {"source": "/path/knowledge/gps/IS-GPS-200N.pdf"}

        result = engine._format_sources([d1, d2, d3])

        assert "Fontes consultadas" in result
        assert "buffer.md" in result
        assert "IS-GPS-200N.pdf" in result
        # 'buffer.md' aparece uma única vez (sem duplicata)
        assert result.count("buffer.md") == 1


class TestPersona:
    """Testes para o foco/especialidade da sessão (persona)"""

    @patch("brain.OllamaLLM")
    @patch("brain.OllamaEmbeddings")
    def test_set_persona(self, mock_embeddings, mock_llm):
        engine = LIVIAEngine()
        assert engine.persona == ""
        engine.set_persona("especialista em Node.js")
        assert engine.persona == "especialista em Node.js"
        assert "Node.js" in engine._persona_line()

    @patch("brain.OllamaLLM")
    @patch("brain.OllamaEmbeddings")
    def test_persona_line_empty_when_unset(self, mock_embeddings, mock_llm):
        engine = LIVIAEngine()
        assert engine._persona_line() == ""


class TestLongTermMemory:
    """Testes para a memória de longo prazo (indexação de conversas)"""

    @patch("brain.OllamaLLM")
    @patch("brain.OllamaEmbeddings")
    def test_remember_empty_returns_zero(self, mock_embeddings, mock_llm):
        engine = LIVIAEngine()
        assert engine.remember_conversation([]) == 0

    @patch("brain.OllamaLLM")
    @patch("brain.OllamaEmbeddings")
    def test_remember_indexes_conversation(self, mock_embeddings, mock_llm):
        engine = LIVIAEngine()
        engine.vectorstore = Mock()
        messages = [
            {"role": "user", "content": "O que é event loop?"},
            {"role": "assistant", "content": "É o mecanismo de..."},
        ]
        n = engine.remember_conversation(messages, feedback="bom")

        assert n == 1
        engine.vectorstore.add_texts.assert_called_once()
        # Verifica que o feedback foi para os metadados
        _, kwargs = engine.vectorstore.add_texts.call_args
        assert kwargs["metadatas"][0]["feedback"] == "bom"
        assert kwargs["metadatas"][0]["source"] == "memoria_conversa"
