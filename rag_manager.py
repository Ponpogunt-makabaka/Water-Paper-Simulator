# rag_manager.py
"""
Enhanced RAG (Retrieval-Augmented Generation) pipeline manager.
Handles vector store creation and querying using local Ollama embedding models.
FIXED: Removed unsupported parameters for new langchain-ollama package.
"""

import os
import logging
import shutil
import glob
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings  # 使用新包
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import config

# Setup logging
logger = logging.getLogger(__name__)

class RAGManager:
    """
    Enhanced RAG manager with robust error handling and optimization features.
    Uses the new langchain-ollama package with correct parameters.
    """
    
    def __init__(self, papers_data: List[Dict]):
        """
        Initialize RAG manager with papers data.
        
        Args:
            papers_data: List of paper dictionaries with full text
        """
        self.papers_data = papers_data
        self.embedding_model = None
        self.vector_store_path = os.path.join(config.OUTPUT_DIR, "vector_store")
        self.backup_store_path = os.path.join(config.OUTPUT_DIR, "vector_store_backup")
        
        # Initialize embedding model with error handling
        self._initialize_embedding_model()
        
        # Ensure output directory exists
        Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
    
    def _initialize_embedding_model(self) -> None:
        """Initialize Ollama embedding model with correct parameters for new package."""
        try:
            logger.info(f"[RAG] Initializing Ollama embedding model: {config.OLLAMA_EMBEDDING_MODEL}")
            
            # Use only supported parameters for new langchain-ollama package
            self.embedding_model = OllamaEmbeddings(
                model=config.OLLAMA_EMBEDDING_MODEL,
                base_url=config.OLLAMA_BASE_URL
                # Removed unsupported parameters: temperature, num_predict
            )
            
            # Test the embedding model with a simple query
            logger.info("[RAG] Testing embedding model...")
            test_result = self.embedding_model.embed_query("test query")
            if test_result and len(test_result) > 0:
                logger.info(f"[RAG] Embedding model initialized successfully (dimension: {len(test_result)})")
            else:
                raise Exception("Embedding model returned empty result")
                
        except Exception as e:
            logger.error(f"[RAG] Failed to initialize embedding model: {e}")
            self.embedding_model = None
            
            # Try fallback embedding model if configured
            if hasattr(config, 'FALLBACK_EMBEDDING_MODEL'):
                self._try_fallback_embedding_model()
    
    def _try_fallback_embedding_model(self) -> None:
        """Try to initialize fallback embedding model."""
        try:
            logger.info(f"[RAG] Trying fallback embedding model: {config.FALLBACK_EMBEDDING_MODEL}")
            
            self.embedding_model = OllamaEmbeddings(
                model=config.FALLBACK_EMBEDDING_MODEL,
                base_url=config.OLLAMA_BASE_URL
            )
            
            # Test fallback model
            test_result = self.embedding_model.embed_query("test query")
            if test_result and len(test_result) > 0:
                logger.info("[RAG] Fallback embedding model initialized successfully")
            else:
                raise Exception("Fallback embedding model failed")
                
        except Exception as e:
            logger.error(f"[RAG] Fallback embedding model also failed: {e}")
            self.embedding_model = None
    
    def create_vector_store(self) -> Optional[str]:
        """
        Create and save a FAISS vector store from papers' full text.
        
        Returns:
            Path to vector store if successful, None otherwise
        """
        if not self.embedding_model:
            logger.error("[RAG] Cannot create vector store: embedding model not available")
            return self._create_dummy_store()
        
        logger.info("[RAG] Creating vector store using local embeddings...")
        
        try:
            # Extract and validate documents
            documents = self._extract_documents()
            if not documents:
                logger.warning("[RAG] No documents available for vector store creation")
                return self._create_dummy_store()
            
            # Split documents into chunks
            chunks = self._split_documents(documents)
            if not chunks:
                logger.warning("[RAG] No chunks created after document splitting")
                return self._create_dummy_store()
            
            # Create vector store
            vector_store = self._create_faiss_store(chunks)
            if not vector_store:
                return self._create_dummy_store()
            
            # Save vector store
            store_path = self._save_vector_store(vector_store)
            if store_path:
                logger.info(f"[RAG] Vector store successfully created and saved to: {store_path}")
                return store_path
            else:
                return self._create_dummy_store()
                
        except Exception as e:
            logger.error(f"[RAG] Failed to create vector store: {e}")
            return self._create_dummy_store()
    
    def _create_dummy_store(self) -> str:
        """Create a dummy store when real vector store creation fails."""
        dummy_path = os.path.join(config.OUTPUT_DIR, "dummy_vector_store")
        Path(dummy_path).mkdir(exist_ok=True)
        
        # Create dummy files to mimic a real vector store
        with open(os.path.join(dummy_path, "dummy.txt"), "w") as f:
            f.write("Dummy vector store - real store creation failed or embedding model unavailable")
        
        logger.info(f"[RAG] Created dummy vector store at: {dummy_path}")
        return dummy_path
    
    def _extract_documents(self) -> List[str]:
        """Extract and validate text documents from papers data."""
        documents = []
        
        for i, paper in enumerate(self.papers_data):
            try:
                full_text = paper.get('full_text', '')
                title = paper.get('title', f'Paper {i}')
                
                if not full_text or not isinstance(full_text, str):
                    logger.warning(f"[RAG] No valid full text for paper: {title[:50]}")
                    continue
                
                # Basic text validation
                if len(full_text.strip()) < 100:
                    logger.warning(f"[RAG] Full text too short for paper: {title[:50]}")
                    continue
                
                # Add metadata to the text
                enhanced_text = f"Title: {title}\n\n{full_text}"
                documents.append(enhanced_text)
                
            except Exception as e:
                logger.error(f"[RAG] Error processing paper {i}: {e}")
                continue
        
        logger.info(f"[RAG] Extracted {len(documents)} valid documents from {len(self.papers_data)} papers")
        return documents
    
    def _split_documents(self, documents: List[str]) -> List[str]:
        """Split documents into manageable chunks."""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            # Combine all documents
            combined_text = "\n\n---DOCUMENT_SEPARATOR---\n\n".join(documents)
            
            # Split into chunks
            chunks = text_splitter.split_text(combined_text)
            
            # Filter out very short chunks
            valid_chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]
            
            logger.info(f"[RAG] Created {len(valid_chunks)} text chunks from {len(documents)} documents")
            return valid_chunks
            
        except Exception as e:
            logger.error(f"[RAG] Error splitting documents: {e}")
            return []
    
    def _create_faiss_store(self, chunks: List[str]) -> Optional[FAISS]:
        """Create FAISS vector store from text chunks."""
        try:
            logger.info(f"[RAG] Creating FAISS store from {len(chunks)} chunks...")
            
            # Use smaller batch size to be more conservative
            batch_size = 10
            vector_store = None
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_num = i//batch_size + 1
                total_batches = (len(chunks) + batch_size - 1)//batch_size
                
                logger.info(f"[RAG] Processing batch {batch_num}/{total_batches}")
                
                try:
                    if vector_store is None:
                        # Create initial store
                        vector_store = FAISS.from_texts(
                            texts=batch, 
                            embedding=self.embedding_model
                        )
                        logger.info(f"[RAG] Created initial vector store with {len(batch)} documents")
                    else:
                        # Add to existing store
                        batch_store = FAISS.from_texts(
                            texts=batch,
                            embedding=self.embedding_model
                        )
                        vector_store.merge_from(batch_store)
                        logger.info(f"[RAG] Merged batch {batch_num} with {len(batch)} documents")
                        
                    # Wait between batches to avoid overloading
                    import time
                    time.sleep(1)
                    
                except Exception as batch_error:
                    logger.error(f"[RAG] Error processing batch {batch_num}: {batch_error}")
                    continue
            
            if vector_store:
                logger.info("[RAG] FAISS vector store created successfully")
                return vector_store
            else:
                logger.error("[RAG] Failed to create FAISS vector store")
                return None
                
        except Exception as e:
            logger.error(f"[RAG] Error creating FAISS store: {e}")
            return None
    
    def _save_vector_store(self, vector_store: FAISS) -> Optional[str]:
        """Save vector store to disk with backup."""
        try:
            # Save main store
            vector_store.save_local(self.vector_store_path)
            logger.info(f"[RAG] Saved main vector store to: {self.vector_store_path}")
            
            # Create backup if main save successful
            try:
                vector_store.save_local(self.backup_store_path)
                logger.info(f"[RAG] Saved backup vector store to: {self.backup_store_path}")
            except Exception as e:
                logger.warning(f"[RAG] Failed to create backup: {e}")
            
            # Verify the saved store
            if self._verify_vector_store(self.vector_store_path):
                return self.vector_store_path
            else:
                logger.error("[RAG] Vector store verification failed")
                return None
                
        except Exception as e:
            logger.error(f"[RAG] Error saving vector store: {e}")
            return None
    
    def _verify_vector_store(self, store_path: str) -> bool:
        """Verify that the saved vector store is valid."""
        try:
            # Try to load and query the store
            temp_store = FAISS.load_local(
                store_path, 
                self.embedding_model, 
                allow_dangerous_deserialization=True
            )
            
            # Test query
            results = temp_store.similarity_search("test query", k=1)
            success = len(results) > 0
            logger.info(f"[RAG] Vector store verification: {'passed' if success else 'failed'}")
            return success
            
        except Exception as e:
            logger.error(f"[RAG] Vector store verification failed: {e}")
            return False
    
    @staticmethod
    def query_vector_store(store_path: str, query: str, k: int = 5) -> str:
        """
        Query the vector store and return relevant context.
        
        Args:
            store_path: Path to the vector store
            query: Search query
            k: Number of results to retrieve
            
        Returns:
            Formatted context string
        """
        # Check if this is a dummy store
        if store_path and "dummy_vector_store" in store_path:
            logger.info("[RAG] Using dummy vector store - returning fallback context")
            return RAGManager._generate_fallback_context(query)
            
        if not store_path or not os.path.exists(store_path):
            logger.warning("[RAG] Vector store not found, returning fallback context")
            return RAGManager._generate_fallback_context(query)
        
        try:
            logger.info(f"[RAG] Querying vector store with: '{query[:50]}...'")
            
            # Initialize embedding model for querying - use only supported parameters
            embedding_model = OllamaEmbeddings(
                model=config.OLLAMA_EMBEDDING_MODEL,
                base_url=config.OLLAMA_BASE_URL
            )
            
            # Load vector store
            vector_store = FAISS.load_local(
                store_path, 
                embedding_model, 
                allow_dangerous_deserialization=True
            )
            
            # Perform similarity search
            docs = vector_store.similarity_search(query, k=k)
            
            if not docs:
                logger.warning("[RAG] No relevant documents found for query")
                return RAGManager._generate_fallback_context(query)
            
            # Format results
            context_parts = []
            for i, doc in enumerate(docs):
                content = doc.page_content.strip()
                if content:
                    context_parts.append(f"Source {i+1}:\n{content}")
            
            if context_parts:
                context = "\n\n---\n\n".join(context_parts)
                logger.info(f"[RAG] Retrieved {len(context_parts)} relevant passages")
                return context
            else:
                return RAGManager._generate_fallback_context(query)
                
        except Exception as e:
            logger.error(f"[RAG] Error querying vector store: {e}")
            return RAGManager._generate_fallback_context(query)
    
    @staticmethod
    def _generate_fallback_context(query: str) -> str:
        """Generate fallback context when vector store is unavailable."""
        return f"""Literature context for: "{query}"

Based on general academic knowledge, this research area typically involves:

1. **Theoretical Foundations**: Established principles and frameworks from prior research
2. **Methodological Approaches**: Standard techniques and novel methods in the field
3. **Evaluation Metrics**: Common performance measures and validation approaches
4. **Current Challenges**: Known limitations and open research questions

Note: This is general context as the RAG vector store is not available.
For more specific literature context, please ensure:
- Ollama service is running with the embedding model loaded
- Vector store was created successfully from the literature
- Network connectivity to the embedding service is stable"""

    def get_store_statistics(self) -> Dict[str, any]:
        """Get comprehensive statistics about the vector store."""
        stats = {
            "papers_count": len(self.papers_data),
            "embedding_model": config.OLLAMA_EMBEDDING_MODEL,
            "store_path": self.vector_store_path,
            "store_exists": os.path.exists(self.vector_store_path),
            "backup_exists": os.path.exists(self.backup_store_path),
            "is_dummy_store": "dummy_vector_store" in self.vector_store_path,
            "embedding_model_available": self.embedding_model is not None
        }
        
        # Try to get more detailed stats if store exists
        if stats["store_exists"] and not stats["is_dummy_store"] and stats["embedding_model_available"]:
            try:
                vector_store = FAISS.load_local(
                    self.vector_store_path, 
                    self.embedding_model, 
                    allow_dangerous_deserialization=True
                )
                stats["vector_count"] = vector_store.index.ntotal
                stats["index_size_mb"] = os.path.getsize(os.path.join(self.vector_store_path, "index.faiss")) / (1024*1024)
            except Exception as e:
                stats["vector_count"] = f"Error retrieving count: {e}"
                stats["index_size_mb"] = "Unknown"
        else:
            stats["vector_count"] = "N/A"
            stats["index_size_mb"] = "N/A"
        
        return stats

# Export classes and functions
__all__ = [
    'RAGManager'
]