# base_agent.py
"""
Optimized base class for all research agents.
Implements token-efficient processing and structured decision making.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.language_models import BaseChatModel

import config
from llm_config import llm_manager


class AgentError(Exception):
    """Custom exception for agent failures."""
    pass


class BaseAgent(ABC):
    """
    Abstract base class for research workflow agents.
    Optimized for token efficiency and structured outputs.
    """
    
    def __init__(self, name: str, llm_type: str = "default"):
        """
        Initialize base agent.
        
        Args:
            name: Agent identifier
            llm_type: Type of LLM to use (research/analysis/writing/review)
        """
        self.name = name
        self.llm = self._get_llm(llm_type)
        self.output_dir = Path(config.OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)
        
    def _get_llm(self, llm_type: str) -> BaseChatModel:
        """Get appropriate LLM based on agent type."""
        llm_map = {
            "research": llm_manager.get_research_llm,
            "analysis": llm_manager.get_analysis_llm,
            "writing": llm_manager.get_writing_llm,
            "review": llm_manager.get_review_llm,
            "default": llm_manager.create_llm
        }
        return llm_map.get(llm_type, llm_manager.create_llm)()
    
    @abstractmethod
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent logic. Must be implemented by subclasses."""
        pass
    
    def create_chain(self, template: str, parser=None):
        """Create processing chain with template and parser."""
        prompt = ChatPromptTemplate.from_template(template)
        if parser is None:
            parser = StrOutputParser()
        return prompt | self.llm | parser
    
    def make_decision(self, prompt: str, **kwargs) -> str:
        """
        Make a structured decision using multiple choice format.
        
        Args:
            prompt: Decision prompt template
            **kwargs: Variables for prompt
            
        Returns:
            Decision result
        """
        chain = self.create_chain(prompt)
        result = chain.invoke(kwargs)
        return self._parse_decision(result)
    
    def _parse_decision(self, response: str) -> str:
        """Parse structured decision from response."""
        # Extract decision markers like [1], [YES], [PASS], etc.
        import re
        
        # Try to find bracketed decisions
        match = re.search(r'\[([A-Z0-9]+)\]', response.upper())
        if match:
            return match.group(1)
        
        # Try to find choice numbers
        match = re.search(r'(?:Choice|Decision|Select)[:\s]*(\d+)', response, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Default to full response if no structured format found
        return response.strip()
    
    def check_yes_no(self, prompt: str, **kwargs) -> bool:
        """
        Get yes/no decision from LLM.
        
        Args:
            prompt: Yes/no question template
            **kwargs: Variables for prompt
            
        Returns:
            True for YES, False for NO
        """
        result = self.make_decision(prompt, **kwargs)
        return "YES" in result.upper()
    
    def extract_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Extract JSON from LLM response.
        
        Args:
            prompt: Prompt requesting JSON output
            **kwargs: Variables for prompt
            
        Returns:
            Parsed JSON dictionary
        """
        try:
            chain = self.create_chain(prompt, JsonOutputParser())
            return chain.invoke(kwargs)
        except Exception as e:
            self.log(f"JSON parsing failed: {e}", "WARNING")
            return {}
    
    def truncate_text(self, text: str, max_chars: int) -> str:
        """
        Truncate text to maximum characters while preserving words.
        
        Args:
            text: Text to truncate
            max_chars: Maximum character count
            
        Returns:
            Truncated text
        """
        if len(text) <= max_chars:
            return text
        
        # Find last space before limit
        truncated = text[:max_chars]
        last_space = truncated.rfind(' ')
        
        if last_space > 0:
            return truncated[:last_space] + "..."
        return truncated + "..."
    
    def save_file(self, content: str, filename: str, append: bool = False) -> Path:
        """Save content to file."""
        filepath = self.output_dir / filename
        mode = "a" if append else "w"
        
        with open(filepath, mode, encoding="utf-8") as f:
            f.write(content)
        
        self.log(f"Saved: {filepath}")
        return filepath
    
    def read_file(self, filename: str) -> str:
        """Read content from file."""
        filepath = self.output_dir / filename
        
        if not filepath.exists():
            raise AgentError(f"File not found: {filepath}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    
    def log(self, message: str, level: str = "INFO") -> None:
        """Log message with agent context."""
        print(f"[{self.name}] {level}: {message}")
    
    def validate_state(self, state: Dict[str, Any], required: List[str]) -> None:
        """Validate required state keys exist."""
        missing = [k for k in required if k not in state or not state[k]]
        if missing:
            raise AgentError(f"Missing required state: {missing}")
    
    def get_revision_count(self, state: Dict[str, Any]) -> int:
        """Get current revision count."""
        return state.get("revision_count", 0)
    
    def is_revision(self, state: Dict[str, Any]) -> bool:
        """Check if this is a revision round."""
        fb = state.get("feedback_breadth", "").strip()
        fd = state.get("feedback_depth", "").strip()
        
        return bool(fb and fb.upper() != "PASS") or bool(fd and fd.upper() != "PASS")