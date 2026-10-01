"""
LIV IA Brain - Motor de IA com LangChain e Ollama
"""
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import os

SYSTEM_PROMPT = """Você é a LIV IA, uma Arquiteta de Soluções Sênior. 
Você é analítica, especialista em HealthTech, padrões de projeto e automação.

IMPORTANTE: Você tem conhecimento geral sobre programação, arquitetura de software e boas práticas.
Se houver documentos fornecidos no contexto, use-os como referência principal.
Se não houver documentos relevantes, use seu conhecimento base para ajudar.

Contexto dos documentos:
{context}

Pergunta: {question}

Responda de forma clara, técnica e fundamentada. Se usar documentos, cite-os.
Se usar conhecimento geral, deixe claro que é uma recomendação baseada em boas práticas."""

CHAT_PROMPT = """Você é a LIV IA, uma Arquiteta de Soluções Sênior. 
Você é analítica, especialista em HealthTech, padrões de projeto e automação.

IMPORTANTE: Você tem conhecimento geral sobre programação, arquitetura de software e boas práticas.
Se houver documentos fornecidos no contexto, use-os como referência principal.
Se não houver documentos relevantes, use seu conhecimento base para ajudar.

Histórico da conversa:
{chat_history}

Contexto dos documentos:
{context}

Pergunta atual: {question}

Responda de forma clara, técnica e fundamentada. Mantenha a continuidade da conversa.
Se usar documentos, cite-os. Se usar conhecimento geral, deixe claro."""

class LIVIAEngine:
    def __init__(self, model_name="deepseek-coder-v2", storage_path=".livia_storage", ollama_base_url="http://localhost:11434", collection_name="livia_default"):
        self.storage_path = storage_path
        self.collection_name = collection_name
        self.llm = OllamaLLM(
            model=model_name, 
            temperature=0.3,
            base_url=ollama_base_url
        )
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=ollama_base_url
        )
        self.vectorstore = None
        self.chat_history = []  # Substituindo ConversationBufferMemory
        self._load_vectorstore()
    
    def _load_vectorstore(self):
        """Carrega o banco de vetores se existir"""
        if os.path.exists(self.storage_path):
            self.vectorstore = Chroma(
                persist_directory=self.storage_path,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
    
    def ask(self, question: str) -> str:
        """Responde uma pergunta única consultando a base"""
        if not self.vectorstore:
            # Se não houver base, usa apenas o LLM
            return self.llm.invoke(
                f"Você é a LIV IA, uma Arquiteta de Soluções Sênior especialista em HealthTech. "
                f"Responda a seguinte pergunta: {question}"
            )
        
        prompt = PromptTemplate(
            template=SYSTEM_PROMPT,
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        result = qa_chain({"query": question})
        return result["result"]
    
    def chat(self, message: str) -> str:
        """Mantém uma conversa com contexto"""
        # Adiciona mensagem do usuário ao histórico
        self.chat_history.append(HumanMessage(content=message))
        
        if not self.vectorstore:
            # Se não houver base, usa apenas o LLM com histórico
            history_text = self._format_chat_history()
            prompt = (
                f"Você é a LIV IA, uma Arquiteta de Soluções Sênior especialista em HealthTech. "
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
            chat_history=history_text,
            context=context,
            question=message
        )
        
        response = self.llm.invoke(prompt)
        self.chat_history.append(AIMessage(content=response))
        return response
    
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
