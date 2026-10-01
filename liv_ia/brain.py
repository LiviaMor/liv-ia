"""
LIV IA Brain - Motor de IA com LangChain e Ollama
"""

import os

from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# Fluxo de decisão compartilhado pelos prompts. Guia o raciocínio da IA passo a
# passo (chain-of-thought) antes de responder — melhora muito a qualidade em
# modelos locais menores, sem precisar de fine-tuning.
DECISION_FLOW = """Siga este fluxo de decisão antes de responder:
1. Identifique o domínio da pergunta (ex.: Node.js, arquitetura/microsserviços,
   GPS, HealthTech/LGPD ou outro).
2. Verifique o contexto dos documentos abaixo. Se houver trechos relevantes,
   baseie a resposta neles e CITE as fontes pelo nome do arquivo.
3. Se os documentos não cobrirem a pergunta, diga isso claramente e então use
   seu conhecimento geral, deixando explícito que é recomendação por boas
   práticas (não extraída dos documentos).
4. Quando houver trade-offs (ex.: escolher um padrão de arquitetura), apresente
   as opções e justifique a recomendação.
5. Responda de forma clara, técnica e objetiva."""

SYSTEM_PROMPT = """Você é a LIV IA, uma Arquiteta de Soluções Sênior.
Você é analítica, especialista em HealthTech, padrões de projeto, Node.js,
arquitetura event-driven/microsserviços e automação.

{decision_flow}

Contexto dos documentos:
{context}

Pergunta: {question}

Resposta:"""

CHAT_PROMPT = """Você é a LIV IA, uma Arquiteta de Soluções Sênior.
Você é analítica, especialista em HealthTech, padrões de projeto, Node.js,
arquitetura event-driven/microsserviços e automação.

{decision_flow}

Histórico da conversa:
{chat_history}

Contexto dos documentos:
{context}

Pergunta atual: {question}

Mantenha a continuidade da conversa.

Resposta:"""


# Palavras-chave por knowledge base, usadas pelo roteador simples abaixo.
# Permite direcionar a pergunta para a collection mais provável, reduzindo
# ruído de retrieval (ex.: pergunta de GPS busca na base "gps").
KNOWLEDGE_BASE_KEYWORDS = {
    "nodejs": ["node", "nodejs", "npm", "express", "event loop", "stream", "buffer"],
    "microservices": [
        "microsserviço",
        "microservico",
        "microservice",
        "event-driven",
        "event driven",
        "cqrs",
        "event sourcing",
        "saga",
        "mensageria",
        "kafka",
        "rabbitmq",
        "arquitetura",
    ],
    "gps": ["gps", "gnss", "satélite", "satelite", "nmea", "geolocaliz", "navegação", "navegacao"],
}


def route_knowledge_base(question: str, default: str = "livia_default") -> str:
    """Escolhe a knowledge base mais provável para a pergunta.

    Heurística simples por palavras-chave: conta ocorrências por base e
    retorna a de maior pontuação. Se nada casar, usa a base padrão.
    """
    text = question.lower()
    best, best_score = default, 0
    for base, keywords in KNOWLEDGE_BASE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best, best_score = base, score
    return best


class LIVIAEngine:
    def __init__(
        self,
        model_name="deepseek-coder-v2",
        storage_path=".livia_storage",
        ollama_base_url="http://localhost:11434",
        collection_name="livia_default",
    ):
        self.storage_path = storage_path
        self.collection_name = collection_name
        self.llm = OllamaLLM(model=model_name, temperature=0.3, base_url=ollama_base_url)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_base_url)
        self.vectorstore = None
        self.chat_history = []  # Substituindo ConversationBufferMemory
        self._load_vectorstore()

    def _load_vectorstore(self):
        """Carrega o banco de vetores se existir"""
        if os.path.exists(self.storage_path):
            self.vectorstore = Chroma(
                persist_directory=self.storage_path,
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )

    @staticmethod
    def _format_sources(source_documents) -> str:
        """Monta um rodapé de citação a partir dos documentos usados.

        Agrupa pelos nomes de arquivo (metadata 'source') para que a resposta
        indique de onde o conhecimento veio, aumentando a verificabilidade.
        """
        if not source_documents:
            return ""

        sources = []
        for doc in source_documents:
            metadata = getattr(doc, "metadata", {}) or {}
            source = metadata.get("source")
            if not source:
                continue
            name = os.path.basename(str(source))
            if name not in sources:
                sources.append(name)

        if not sources:
            return ""

        linhas = "\n".join(f"- {s}" for s in sources)
        return f"\n\n---\nFontes consultadas:\n{linhas}"

    def ask(self, question: str) -> str:
        """Responde uma pergunta única consultando a base"""
        if not self.vectorstore:
            # Se não houver base, usa apenas o LLM
            return self.llm.invoke(
                "Você é a LIV IA, uma Arquiteta de Soluções Sênior especialista em HealthTech. "
                f"Responda a seguinte pergunta: {question}"
            )

        prompt = PromptTemplate(
            template=SYSTEM_PROMPT,
            input_variables=["context", "question"],
            partial_variables={"decision_flow": DECISION_FLOW},
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )

        result = qa_chain({"query": question})
        answer = result["result"]
        return answer + self._format_sources(result.get("source_documents"))

    def chat(self, message: str) -> str:
        """Mantém uma conversa com contexto"""
        # Adiciona mensagem do usuário ao histórico
        self.chat_history.append(HumanMessage(content=message))

        if not self.vectorstore:
            # Se não houver base, usa apenas o LLM com histórico
            history_text = self._format_chat_history()
            prompt = (
                "Você é a LIV IA, uma Arquiteta de Soluções Sênior especialista em HealthTech. "
                f"\n\nHistórico da conversa:\n{history_text}\n\n"
                f"Continue a conversa respondendo: {message}"
            )
            response = self.llm.invoke(prompt)
            self.chat_history.append(AIMessage(content=response))
            return response

        # Com vectorstore, usa RAG
        history_text = self._format_chat_history()
        docs = self.vectorstore.similarity_search(message, k=4)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = CHAT_PROMPT.format(
            decision_flow=DECISION_FLOW,
            chat_history=history_text,
            context=context,
            question=message,
        )

        response = self.llm.invoke(prompt)
        self.chat_history.append(AIMessage(content=response))
        return response + self._format_sources(docs)

    def _format_chat_history(self) -> str:
        """Formata o histórico de chat para o prompt"""
        if not self.chat_history:
            return "Nenhuma conversa anterior."

        formatted = []
        for msg in self.chat_history[-6:]:  # Últimas 6 mensagens (3 trocas)
            if isinstance(msg, HumanMessage):
                formatted.append(f"Usuário: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted.append(f"LIV IA: {msg.content}")

        return "\n".join(formatted)
