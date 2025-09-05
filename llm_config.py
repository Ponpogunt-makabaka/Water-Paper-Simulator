# llm_config.py
"""
LLM configuration and initialization module.
Handles model provider selection and setup.
"""

import os
import config
from typing import Optional
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel


def setup_langsmith():
    """Configure LangSmith tracing if enabled."""
    if config.LANGSMITH_TRACING.lower() == "true":
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = config.LANGSMITH_ENDPOINT
        os.environ["LANGCHAIN_API_KEY"] = config.LANGSMITH_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = config.LANGSMITH_PROJECT
        print("✅ LangSmith tracing enabled")


# Initialize LangSmith immediately when module loads
setup_langsmith()


class LLMManager:
    """Manages LLM instances with different temperature settings."""
    
    def __init__(self):
        self.provider = config.MODEL_PROVIDER.lower()
        self._validate_config()
        
    def _validate_config(self):
        """Validate configuration settings."""
        if self.provider not in ["ollama", "openai"]:
            raise ValueError(f"Invalid MODEL_PROVIDER: {self.provider}")
            
        if self.provider == "openai" and not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in config.py")
    
    def create_llm(self, temperature: float = 0.3) -> BaseChatModel:
        """
        Create an LLM instance with specified temperature.
        
        Args:
            temperature: Model temperature (0.0 to 1.0)
            
        Returns:
            Configured LLM instance
        """
        if self.provider == "ollama":
            return self._create_ollama_llm(temperature)
        else:
            return self._create_openai_llm(temperature)
    
    def _create_ollama_llm(self, temperature: float) -> ChatOllama:
        """Create Ollama LLM instance."""
        print(f"Initializing Ollama: {config.OLLAMA_MODEL} (temp={temperature})")
        return ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=temperature,
            num_predict=10000,  # Limit response length
            top_k=50,  # Limit vocabulary for efficiency
            top_p=0.9
        )
    
    def _create_openai_llm(self, temperature: float) -> ChatOpenAI:
        """Create OpenAI-compatible LLM instance."""
        print(f"Initializing OpenAI: {config.OPENAI_MODEL_NAME} (temp={temperature})")
        return ChatOpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_API_BASE,
            model=config.OPENAI_MODEL_NAME,
            temperature=temperature,
            max_tokens=1000,  # Limit response length
            model_kwargs={
                "top_p": 0.9,
                "frequency_penalty": 0.3,  # Reduce repetition
                "presence_penalty": 0.3
            }
        )
    
    def get_research_llm(self) -> BaseChatModel:
        """Get LLM configured for research tasks."""
        return self.create_llm(config.TEMPERATURE_RESEARCH)
    
    def get_analysis_llm(self) -> BaseChatModel:
        """Get LLM configured for analysis tasks."""
        return self.create_llm(config.TEMPERATURE_ANALYSIS)
    
    def get_writing_llm(self) -> BaseChatModel:
        """Get LLM configured for writing tasks."""
        return self.create_llm(config.TEMPERATURE_WRITING)
    
    def get_review_llm(self) -> BaseChatModel:
        """Get LLM configured for review tasks."""
        return self.create_llm(config.TEMPERATURE_REVIEW)


# Global LLM manager instance
llm_manager = LLMManager()

# Default LLM instance for backward compatibility
llm = llm_manager.create_llm()