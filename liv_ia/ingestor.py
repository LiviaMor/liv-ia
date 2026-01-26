"""
LIV IA Document Processor - Processamento e indexação de PDFs
"""
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os
from pathlib import Path

class DocumentProcessor:
    def __init__(self, storage_path=".livia_storage", ollama_base_url="http://localhost:11434"):
        self.storage_path = storage_path
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=ollama_base_url
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    def ingest_pdfs(self, pdf_path: str) -> int:
        """Processa todos os PDFs de uma pasta e salva no banco de vetores"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Pasta não encontrada: {pdf_path}")
        
        # Carrega todos os PDFs da pasta
        loader = DirectoryLoader(
            pdf_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        
        documents = loader.load()
        
        if not documents:
            raise ValueError(f"Nenhum PDF encontrado em: {pdf_path}")
        
        # Divide os documentos em chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Cria ou atualiza o banco de vetores
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.storage_path
        )
        
        vectorstore.persist()
        
        return len(documents)
