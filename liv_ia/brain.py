"""
LIV IA Brain - Motor de IA com LangChain e Ollama
"""

import os
import re

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


# Padrões que indicam pedido de criação de slides Marp no chat.
_MARP_INTENT = re.compile(
    r"\b(marp|slides?|apresenta[çc][aã]o|apresenta[çc][õo]es)\b",
    re.IGNORECASE,
)
_MARP_TOPIC = re.compile(
    r"(?:marp|slides?|apresenta[çc][aã]o(?:es)?)\s+(?:sobre|de|do|da|para|a respeito de)\s+(.+)",
    re.IGNORECASE,
)


def detect_slides_request(message: str):
    """Detecta pedido de slides Marp e extrai o tópico.

    Returns:
        O tópico (str) se a mensagem for um pedido de slides; senão None.
    """
    if not _MARP_INTENT.search(message):
        return None
    m = _MARP_TOPIC.search(message)
    if m:
        return m.group(1).strip(" .?!")
    # Pediu slides mas sem "sobre X" explícito: usa a mensagem toda como tema.
    return message.strip(" .?!")


class LIVIAEngine:
    def __init__(
        self,
        model_name="qwen2.5-coder:1.5b",
        storage_path=".livia_storage",
        ollama_base_url="http://localhost:11434",
        collection_name="livia_default",
    ):
        self.storage_path = storage_path
        self.collection_name = collection_name
        self.persona = ""  # Foco/especialidade definido na 1ª mensagem do chat
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

    def set_persona(self, persona: str) -> None:
        """Define o foco/especialidade do assistente para a sessão de chat.

        A persona vem da 1ª mensagem do chat (ex.: 'especialista em Node.js')
        e é injetada nos prompts para orientar o tom e o foco das respostas.
        Não troca a base de conhecimento — a busca continua unificada.
        """
        self.persona = (persona or "").strip()

    def _persona_line(self) -> str:
        """Linha de persona para injeção nos prompts (vazia se não definida)."""
        if not self.persona:
            return ""
        return f"\nFoco desta conversa: {self.persona}.\n"

    def remember_conversation(self, messages, feedback: str = "") -> int:
        """Indexa uma conversa na base RAG (memória de longo prazo).

        Transforma o diálogo em um documento e o adiciona à collection, para
        que sessões futuras possam recuperá-lo por busca semântica. É memória
        baseada em recuperação (não re-treina o modelo), portanto segura.

        Args:
            messages: lista de dicts {'role', 'content'}.
            feedback: avaliação opcional ('bom'/'ruim') guardada como metadado.

        Returns:
            1 se indexou, 0 se não havia o que indexar.
        """
        if not messages:
            return 0

        texto = "\n".join(f"{m.get('role', '?')}: {m.get('content', '')}" for m in messages)
        if not texto.strip():
            return 0

        from datetime import datetime

        metadata = {
            "source": "memoria_conversa",
            "timestamp": datetime.now().isoformat(),
            "feedback": feedback or "sem_avaliacao",
            "persona": self.persona or "",
        }

        if self.vectorstore is None:
            # Cria a base caso ainda não exista, para começar a memória.
            self.vectorstore = Chroma(
                persist_directory=self.storage_path,
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )

        self.vectorstore.add_texts([texto], metadatas=[metadata])
        return 1

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
                f"{self._persona_line()}"
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
            decision_flow=DECISION_FLOW + self._persona_line(),
            chat_history=history_text,
            context=context,
            question=message,
        )

        response = self.llm.invoke(prompt)
        self.chat_history.append(AIMessage(content=response))
        return response + self._format_sources(docs)

    def generate_slides(self, topic: str, out_dir: str = "slides"):
        """Gera uma apresentação Marp sobre um tópico e salva localmente.

        Usa o LLM (com RAG, se houver base) para produzir o conteúdo dos
        slides já no formato Marp (blocos separados por '---'), depois salva
        o arquivo .md via o módulo slides.

        Returns:
            Caminho (Path) do arquivo .md salvo.
        """
        import slides as slides_mod

        instrucao = (
            "Crie o CONTEÚDO de uma apresentação de slides sobre o tema abaixo. "
            "Responda APENAS com o corpo dos slides em Markdown, usando '---' em "
            "uma linha isolada para separar cada slide. Cada slide deve ter um "
            "título com '## ' e tópicos concisos em bullets. Não inclua "
            "frontmatter nem ```; apenas o conteúdo dos slides.\n\n"
            f"Tema: {topic}"
        )

        if self.vectorstore:
            docs = self.vectorstore.similarity_search(topic, k=4)
            contexto = "\n\n".join(d.page_content for d in docs)
            instrucao = f"Use este contexto como referência:\n{contexto}\n\n" + instrucao

        corpo = self.llm.invoke(instrucao)
        return slides_mod.save_slides(title=topic, content=corpo, out_dir=out_dir)

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
