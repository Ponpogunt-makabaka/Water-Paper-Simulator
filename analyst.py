# analyst.py
"""
Optimized analyst agent for research evaluation and planning.
Uses structured decision-making for efficiency.
Enhanced with flexible topic extraction for better robustness.
"""

from typing import Dict, Any
import re
import config
import prompts
from base_agent import BaseAgent, AgentError


class AnalystAgent(BaseAgent):
    """
    Agent responsible for analyzing research plans and creating outlines.
    Optimized for structured evaluation and minimal iterations.
    Enhanced with robust text parsing methods.
    """
    
    def __init__(self):
        super().__init__("Analyst", "analysis")
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute analysis phase.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with final plan and references
        """
        try:
            self.validate_state(state, ["innovation_plans"])
            
            # Read references
            references = self._read_references()
            
            # Evaluate plans and select best
            final_topic = self._evaluate_plans(
                state["innovation_plans"],
                references
            )
            
            # Check foundation
            is_solid = self._check_foundation(final_topic, references)
            
            if not is_solid:
                self.log("Foundation needs strengthening", "WARNING")
                # In production, would trigger additional search
            
            # Generate outline
            final_plan = self._generate_outline(final_topic)
            
            return {
                "final_topic": final_topic,
                "final_plan": final_plan,
                "references": references
            }
            
        except Exception as e:
            self.log(f"Analysis failed: {e}", "ERROR")
            raise AgentError(f"Analyst failed: {e}")
    
    def _read_references(self) -> str:
        """Read references from file."""
        try:
            return self.read_file(config.REFERENCE_FILE)
        except:
            self.log("No references found", "WARNING")
            return "No references available"
    
    def _evaluate_plans(self, plans: str, references: str) -> str:
        """
        Evaluate innovation plans and select best with improved topic extraction.
        Enhanced to handle multiple possible topic formats.
        """
        result = self.make_decision(
            prompts.ANALYST_EVALUATE_PROMPT,
            plans=self.truncate_text(plans, 300),
            references=self.truncate_text(references, 500)
        )
        
        # Enhanced topic extraction with flexible regex patterns
        # Match "Topic=", "Topic:", "Final Topic =", etc., ignoring case
        match = re.search(r"Topic\s*[:=]\s*(.*)", result, re.IGNORECASE)
        if match:
            topic = match.group(1).strip().split("\n")[0]  # Take first line
            self.log(f"Selected topic: {topic}")
            return topic
        
        # Try alternative patterns for topic selection
        # Look for "Choice=[#] Topic=[topic]" format
        choice_match = re.search(r"Choice\s*=\s*\[\d+\]\s*Topic\s*=\s*(.*)", result, re.IGNORECASE)
        if choice_match:
            topic = choice_match.group(1).strip().split("\n")[0]
            self.log(f"Selected topic from choice format: {topic}")
            return topic
        
        # Look for "Selected:" or "Best:" patterns
        selected_match = re.search(r"(?:Selected|Best)\s*[:=]\s*(.*)", result, re.IGNORECASE)
        if selected_match:
            topic = selected_match.group(1).strip().split("\n")[0]
            self.log(f"Selected topic from selection format: {topic}")
            return topic
        
        # Fallback to first non-empty line from plans
        self.log("Using fallback topic selection", "WARNING")
        lines = plans.split("\n")
        for line in lines:
            if line.strip() and len(line.strip()) > 10:  # Ensure meaningful content
                return line.strip()[:100]
        
        # Ultimate fallback
        return plans.split("\n")[0][:100] if plans else "Default research topic"
    
    def _check_foundation(self, topic: str, references: str) -> bool:
        """Check if theoretical foundation is solid."""
        result = self.make_decision(
            prompts.ANALYST_FOUNDATION_CHECK,
            topic=self.truncate_text(topic, 100),
            references=self.truncate_text(references, 800)
        )
        
        is_solid = "SOLID" in result.upper()
        self.log(f"Foundation check: {'solid' if is_solid else 'needs work'}")
        
        return is_solid
    
    def _generate_outline(self, topic: str) -> str:
        """Generate paper outline."""
        sections = [
            "1. Abstract - Summary of research",
            "2. Introduction - Problem and contribution",
            "3. Literature Review - Prior work analysis",
            "4. Method - Technical approach",
            "5. Evaluation - Experimental design",
            "6. Conclusion - Summary and future work"
        ]
        
        outline = f"# Outline for: {topic}\n\n"
        outline += "\n".join(sections)
        
        self.log("Generated outline")
        return outline


def analyst_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Function wrapper for compatibility."""
    agent = AnalystAgent()
    return agent.execute(state)


# Export both class and function
__all__ = ['AnalystAgent', 'analyst_agent']