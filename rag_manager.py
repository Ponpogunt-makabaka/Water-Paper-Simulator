# rag_manager.py
"""
Manages the Retrieval-Augmented Generation (RAG) pipeline,
including vector store creation and querying using a local Ollama embedding model.
"""
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 确保导入了 config 模块
import config

class RAGManager:
    def __init__(self, papers_data: list):
        self.papers_data = papers_data
        
        # VVVVVV  核心修改：从 config 文件读取嵌入模型名称 VVVVVV
        print(f"[RAG] Initializing Ollama embedding model: {config.OLLAMA_EMBEDDING_MODEL}")
        self.embedding_model = OllamaEmbeddings(
            model=config.OLLAMA_EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL
        )
        # ^^^^^^  核心修改：从 config 文件读取嵌入模型名称 ^^^^^^

        self.vector_store_path = os.path.join(config.OUTPUT_DIR, "vector_store")
    
    def create_vector_store(self):
        """Creates and saves a FAISS vector store from the paper's full text."""
        print("[RAG] Creating vector store using local embeddings...")
        documents = []
        for paper in self.papers_data:
            if paper.get('full_text') and isinstance(paper.get('full_text'), str):
                documents.append(paper['full_text'])
        
        if not documents:
            print("[RAG] No documents to create vector store from.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_text("\n\n---\n\n".join(documents))
        
        vector_store = FAISS.from_texts(texts=splits, embedding=self.embedding_model)
        vector_store.save_local(self.vector_store_path)
        print(f"[RAG] Vector store saved to: {self.vector_store_path}")
        return self.vector_store_path

    @staticmethod
    def query_vector_store(store_path: str, query: str, k: int = 5) -> str:
        """Queries the vector store and returns relevant context."""
        if not os.path.exists(store_path):
            return "Vector store not found."
            
        print(f"[RAG] Querying store with: '{query[:50]}...'")
        
        # VVVVVV  核心修改：同样，在这里也从 config 文件读取嵌入模型名称 VVVVVV
        embedding_model = OllamaEmbeddings(
            model=config.OLLAMA_EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL
        )
        # ^^^^^^  核心修改：同样，在这里也从 config 文件读取嵌入模型名称 ^^^^^^
        
        vector_store = FAISS.load_local(store_path, embedding_model, allow_dangerous_deserialization=True)
        
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)
        
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        return context