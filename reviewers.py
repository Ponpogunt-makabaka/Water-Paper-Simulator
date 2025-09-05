# reviewers.py
"""
Optimized review agents for paper evaluation.
Generates detailed review reports with specific feedback.
Enhanced with robust parsing and JSON output for better reliability.
"""

from typing import Dict, Any, List, Tuple
from datetime import datetime
from pathlib import Path
import re
import config
import prompts
from base_agent import BaseAgent, AgentError
from langchain_core.output_parsers import JsonOutputParser


class ReviewReport:
    """Manages review report generation and saving."""
    
    def __init__(self, reviewer_type: str, version: int):
        self.reviewer_type = reviewer_type
        self.version = version
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.issues = []
        self.suggestions = []
        self.strengths = []
        self.score = 0.0
        
    def add_issue(self, category: str, issue: str):
        """Add an issue to the report."""
        self.issues.append({"category": category, "issue": issue})
        
    def add_suggestion(self, suggestion: str):
        """Add a suggestion to the report."""
        self.suggestions.append(suggestion)
        
    def add_strength(self, strength: str):
        """Add a strength to the report."""
        self.strengths.append(strength)
        
    def set_score(self, score: float):
        """Set the overall score."""
        self.score = max(0.0, min(1.0, score))
        
    def generate_report(self) -> str:
        """Generate formatted report text."""
        report = f"""# Review Report - {self.reviewer_type}
Date: {self.timestamp}
Draft Version: {self.version}
Overall Score: {self.score:.2f}/1.00
Pass Threshold: {config.REVIEW_PASS_THRESHOLD:.2f}
Status: {"PASS" if self.score >= config.REVIEW_PASS_THRESHOLD else "NEEDS REVISION"}

## Summary
This review evaluates the paper from a {self.reviewer_type.lower()} perspective.

## Strengths
"""
        if self.strengths:
            for strength in self.strengths:
                report += f"- {strength}\n"
        else:
            report += "- No significant strengths identified\n"
            
        report += "\n## Issues Identified\n"
        if self.issues:
            for issue in self.issues:
                report += f"\n### {issue['category']}\n"
                report += f"{issue['issue']}\n"
        else:
            report += "No critical issues identified.\n"
            
        report += "\n## Suggestions for Improvement\n"
        if self.suggestions:
            for i, suggestion in enumerate(self.suggestions, 1):
                report += f"{i}. {suggestion}\n"
        else:
            report += "No specific suggestions at this time.\n"
            
        report += f"\n## Recommendation\n"
        if self.score >= config.REVIEW_PASS_THRESHOLD:
            report += "The paper is ready for acceptance with minor optional improvements."
        else:
            report += "The paper requires revision to address the identified issues."
            
        return report
    
    def save_report(self, output_dir: Path) -> Path:
        """Save report to file."""
        filename = f"{config.REVIEW_REPORT_PREFIX}{self.reviewer_type}_{self.timestamp}.txt"
        filepath = output_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.generate_report())
            
        return filepath


class BreadthReviewAgent(BaseAgent):
    """
    Agent for macro-level paper review.
    Checks structure, coverage, and coherence.
    Enhanced with JSON-based output parsing for better reliability.
    """
    
    def __init__(self):
        super().__init__("BreadthReviewer", "review")
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute breadth review with report generation.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with breadth feedback
        """
        try:
            self.validate_state(state, ["current_draft"])
            
            draft = state["current_draft"]
            version = state.get("revision_count", 1)
            
            # Create report
            report = ReviewReport("Breadth", version)
            
            # Perform detailed review
            feedback, score, details = self._detailed_review(draft)
            
            # Populate report
            report.set_score(score)
            for issue in details.get("issues", []):
                report.add_issue(issue["category"], issue["description"])
            for suggestion in details.get("suggestions", []):
                report.add_suggestion(suggestion)
            for strength in details.get("strengths", []):
                report.add_strength(strength)
            
            # Save report if enabled
            if config.GENERATE_REVIEW_REPORTS:
                report_path = report.save_report(self.output_dir)
                self.log(f"Review report saved: {report_path}")
            
            self.log(f"Breadth review: Score={score:.2f}, Status={feedback[:50]}...")
            
            return {
                "feedback_breadth": feedback,
                "score_breadth": score
            }
            
        except Exception as e:
            self.log(f"Breadth review failed: {e}", "ERROR")
            raise AgentError(f"Breadth reviewer failed: {e}")
    
    def _detailed_review(self, draft: str) -> Tuple[str, float, Dict]:
        """
        Perform detailed breadth review with scoring using JSON output.
        
        Returns:
            Tuple of (feedback_text, score, details_dict)
        """
        # Truncate draft for efficiency
        draft_summary = self.truncate_text(draft, 2000)
        
        # Enhanced structured review prompt with JSON output requirement
        review_prompt = f"""Review this paper's macro-level aspects and provide your response in JSON format.

Paper draft:
{draft_summary}

Evaluate these criteria (each 0-1 score):
1. Core Argument Clarity: Is the main thesis clear and compelling?
2. Literature Coverage: Are key references and prior work adequately covered?
3. Logical Structure: Does the paper flow logically from introduction to conclusion?
4. Contribution Clarity: Is the paper's contribution clearly stated?
5. Writing Quality: Is the writing clear and professional?

You MUST provide your response in JSON format with the following keys: "scores", "issues", "suggestions", "strengths".
The "scores" key must be a list of 5 floating-point numbers between 0.0 and 1.0.
The "issues" must be a list of objects, each with "category" and "description".

Example JSON output:
{{
    "scores": [0.8, 0.7, 0.9, 0.8, 0.7],
    "issues": [
        {{"category": "Argument", "description": "The main thesis is not clear enough."}}
    ],
    "suggestions": ["Sharpen the focus of the introduction."],
    "strengths": ["The methodology is sound and well-explained."]
}}"""

        try:
            # Use JSON output parser for more reliable parsing
            chain = self.create_chain(review_prompt, JsonOutputParser())
            review_data = chain.invoke({})
            
            scores = review_data.get("scores", [config.REVIEW_STRICTNESS] * 5)
            details = {
                "issues": review_data.get("issues", []),
                "suggestions": review_data.get("suggestions", []),
                "strengths": review_data.get("strengths", [])
            }
            
        except Exception as e:
            self.log(f"Failed to parse review JSON: {e}", "WARNING")
            # Fallback to text parsing if JSON fails
            chain = self.create_chain(review_prompt)
            result = chain.invoke({})
            
            scores = self._extract_scores_fallback(result)
            details = {
                "issues": self._extract_issues_fallback(result),
                "suggestions": self._extract_suggestions_fallback(result),
                "strengths": self._extract_strengths_fallback(result)
            }
        
        # Calculate average score
        avg_score = sum(scores) / len(scores) if scores else 0.5
        
        # Adjust score based on strictness
        adjusted_score = avg_score * (1.0 + (1.0 - config.REVIEW_STRICTNESS))
        adjusted_score = min(1.0, adjusted_score)
        
        # Generate feedback text
        if adjusted_score >= config.REVIEW_PASS_THRESHOLD:
            feedback = "PASS"
        else:
            issues_text = "; ".join([i.get("description", "") for i in details["issues"][:3]])
            feedback = f"ISSUES: {issues_text}"
            
        return feedback, adjusted_score, details
    
    def _extract_scores_fallback(self, text: str) -> List[float]:
        """Fallback method to extract numerical scores from review text."""
        # Look for SCORES: [...] pattern
        match = re.search(r'SCORES:\s*\[([\d.,\s]+)\]', text)
        if match:
            scores_str = match.group(1)
            try:
                scores = [float(s.strip()) for s in scores_str.split(',')]
                return scores[:5]
            except:
                pass
        
        # Look for any decimal numbers between 0 and 1
        numbers = re.findall(r'0\.\d+|1\.0+', text)
        if numbers:
            return [float(n) for n in numbers[:5]]
        
        # Default scores based on strictness
        return [config.REVIEW_STRICTNESS] * 5
    
    def _extract_issues_fallback(self, text: str) -> List[Dict]:
        """Fallback method to extract issues from review text using flexible regex."""
        issues = []
        categories = ["Argument", "Literature", "Structure", "Contribution", "Writing"]
        
        # Use flexible regex to match ISSUES section
        match = re.search(r"ISSUES:(.*?)(?:SUGGESTIONS:|STRENGTHS:|$)", text, re.DOTALL | re.IGNORECASE)
        if match:
            issues_text = match.group(1).strip()
            issue_lines = issues_text.strip().split('\n')
            
            for i, line in enumerate(issue_lines[:5]):
                if line.strip():
                    issues.append({
                        "category": categories[min(i, len(categories)-1)],
                        "description": line.strip()[:200]
                    })
        
        return issues
    
    def _extract_suggestions_fallback(self, text: str) -> List[str]:
        """Fallback method to extract suggestions from review text using flexible regex."""
        suggestions = []
        
        match = re.search(r"SUGGESTIONS:(.*?)(?:STRENGTHS:|$)", text, re.DOTALL | re.IGNORECASE)
        if match:
            sugg_text = match.group(1).strip()
            sugg_lines = sugg_text.strip().split('\n')
            
            for line in sugg_lines[:5]:
                if line.strip():
                    suggestions.append(line.strip()[:200])
        
        return suggestions
    
    def _extract_strengths_fallback(self, text: str) -> List[str]:
        """Fallback method to extract strengths from review text using flexible regex."""
        strengths = []
        
        match = re.search(r"STRENGTHS:(.*?)$", text, re.DOTALL | re.IGNORECASE)
        if match:
            str_text = match.group(1).strip()
            str_lines = str_text.strip().split('\n')
            
            for line in str_lines[:3]:
                if line.strip():
                    strengths.append(line.strip()[:200])
        
        return strengths


class DepthReviewAgent(BaseAgent):
    """
    Agent for micro-level technical review.
    Checks rigor, methodology, and technical details.
    Enhanced with flexible section extraction and JSON output parsing.
    """
    
    def __init__(self):
        super().__init__("DepthReviewer", "review")
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute depth review with report generation.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with depth feedback
        """
        try:
            self.validate_state(state, ["current_draft"])
            
            draft = state["current_draft"]
            version = state.get("revision_count", 1)
            
            # Create report
            report = ReviewReport("Depth", version)
            
            # Perform detailed review
            feedback, score, details = self._detailed_review(draft)
            
            # Populate report
            report.set_score(score)
            for issue in details.get("issues", []):
                report.add_issue(issue["category"], issue["description"])
            for suggestion in details.get("suggestions", []):
                report.add_suggestion(suggestion)
            for strength in details.get("strengths", []):
                report.add_strength(strength)
            
            # Save report if enabled
            if config.GENERATE_REVIEW_REPORTS:
                report_path = report.save_report(self.output_dir)
                self.log(f"Review report saved: {report_path}")
            
            self.log(f"Depth review: Score={score:.2f}, Status={feedback[:50]}...")
            
            return {
                "feedback_depth": feedback,
                "score_depth": score
            }
            
        except Exception as e:
            self.log(f"Depth review failed: {e}", "ERROR")
            raise AgentError(f"Depth reviewer failed: {e}")
    
    def _detailed_review(self, draft: str) -> Tuple[str, float, Dict]:
        """
        Perform detailed depth review focusing on technical aspects with JSON output.
        
        Returns:
            Tuple of (feedback_text, score, details_dict)
        """
        # Extract technical sections with improved method
        method_eval = self._extract_technical_sections(draft)
        
        review_prompt = f"""Review this paper's technical depth and rigor and provide your response in JSON format.

Technical sections:
{self.truncate_text(method_eval, 2000)}

Evaluate these criteria (each 0-1 score):
1. Methodological Rigor: Are methods clearly described and justified?
2. Mathematical Correctness: Are formulas and derivations correct?
3. Experimental Design: Is the evaluation methodology sound?
4. Technical Innovation: Does the work present novel technical contributions?
5. Implementation Details: Are sufficient details provided for reproduction?

You MUST provide your response in JSON format with the following keys: "scores", "issues", "suggestions", "strengths".
The "scores" key must be a list of 5 floating-point numbers between 0.0 and 1.0.
The "issues" must be a list of objects, each with "category" and "description".

Example JSON output:
{{
    "scores": [0.8, 0.7, 0.9, 0.8, 0.7],
    "issues": [
        {{"category": "Methodology", "description": "The experimental setup lacks proper controls."}}
    ],
    "suggestions": ["Add more detailed explanation of the algorithm."],
    "strengths": ["The mathematical formulation is rigorous."]
}}"""

        try:
            # Use JSON output parser for more reliable parsing
            chain = self.create_chain(review_prompt, JsonOutputParser())
            review_data = chain.invoke({})
            
            scores = review_data.get("scores", [config.REVIEW_STRICTNESS] * 5)
            details = {
                "issues": review_data.get("issues", []),
                "suggestions": review_data.get("suggestions", []),
                "strengths": review_data.get("strengths", [])
            }
            
        except Exception as e:
            self.log(f"Failed to parse review JSON: {e}", "WARNING")
            # Fallback to text parsing if JSON fails
            chain = self.create_chain(review_prompt)
            result = chain.invoke({})
            
            scores = self._extract_scores_fallback(result)
            details = {
                "issues": self._extract_issues_fallback(result),
                "suggestions": self._extract_suggestions_fallback(result),
                "strengths": self._extract_strengths_fallback(result)
            }
        
        # Calculate average score
        avg_score = sum(scores) / len(scores) if scores else 0.5
        
        # Adjust score based on strictness
        adjusted_score = avg_score * (1.0 + (1.0 - config.REVIEW_STRICTNESS))
        adjusted_score = min(1.0, adjusted_score)
        
        # Generate feedback text
        if adjusted_score >= config.REVIEW_PASS_THRESHOLD:
            feedback = "PASS"
        else:
            issues_text = "; ".join([i.get("description", "") for i in details["issues"][:3]])
            feedback = f"ISSUES: {issues_text}"
            
        return feedback, adjusted_score, details
    
    def _extract_technical_sections(self, draft: str) -> str:
        """Extract technical sections with improved flexible regex matching."""
        sections = []
        
        # Use flexible regex to match Method or Methodology sections (case-insensitive)
        method_match = re.search(r"^##+.*Method.*?\n(.*?)(?=\n##+)", draft, re.DOTALL | re.IGNORECASE | re.MULTILINE)
        if method_match:
            sections.append(method_match.group(1)[:1000])
        
        # Use flexible regex to match Evaluation or Experiment sections
        eval_match = re.search(r"^##+.*Evaluation.*?\n(.*?)(?=\n##+)", draft, re.DOTALL | re.IGNORECASE | re.MULTILINE)
        if eval_match:
            sections.append(eval_match.group(1)[:1000])
        
        # If no specific sections found, return a portion of the draft
        return "\n---\n".join(sections) if sections else draft[:2000]
    
    # Use same fallback extraction methods as BreadthReviewAgent
    def _extract_scores_fallback(self, text: str) -> List[float]:
        """Fallback method to extract numerical scores from review text."""
        match = re.search(r'SCORES:\s*\[([\d.,\s]+)\]', text)
        if match:
            scores_str = match.group(1)
            try:
                scores = [float(s.strip()) for s in scores_str.split(',')]
                return scores[:5]
            except:
                pass
        
        numbers = re.findall(r'0\.\d+|1\.0+', text)
        if numbers:
            return [float(n) for n in numbers[:5]]
        
        return [config.REVIEW_STRICTNESS] * 5
    
    def _extract_issues_fallback(self, text: str) -> List[Dict]:
        """Fallback method to extract issues from review text."""
        issues = []
        categories = ["Methodology", "Mathematics", "Experiments", "Innovation", "Implementation"]
        
        match = re.search(r"ISSUES:(.*?)(?:SUGGESTIONS:|STRENGTHS:|$)", text, re.DOTALL | re.IGNORECASE)
        if match:
            issues_text = match.group(1).strip()
            issue_lines = issues_text.strip().split('\n')
            
            for i, line in enumerate(issue_lines[:5]):
                if line.strip():
                    issues.append({
                        "category": categories[min(i, len(categories)-1)],
                        "description": line.strip()[:200]
                    })
        
        return issues
    
    def _extract_suggestions_fallback(self, text: str) -> List[str]:
        """Fallback method to extract suggestions from review text."""
        suggestions = []
        
        match = re.search(r"SUGGESTIONS:(.*?)(?:STRENGTHS:|$)", text, re.DOTALL | re.IGNORECASE)
        if match:
            sugg_text = match.group(1).strip()
            sugg_lines = sugg_text.strip().split('\n')
            
            for line in sugg_lines[:5]:
                if line.strip():
                    suggestions.append(line.strip()[:200])
        
        return suggestions
    
    def _extract_strengths_fallback(self, text: str) -> List[str]:
        """Fallback method to extract strengths from review text."""
        strengths = []
        
        match = re.search(r"STRENGTHS:(.*?)$", text, re.DOTALL | re.IGNORECASE)
        if match:
            str_text = match.group(1).strip()
            str_lines = str_text.strip().split('\n')
            
            for line in str_lines[:3]:
                if line.strip():
                    strengths.append(line.strip()[:200])
        
        return strengths


def breadth_supervisor_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Function wrapper for breadth review."""
    agent = BreadthReviewAgent()
    return agent.execute(state)


def depth_supervisor_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Function wrapper for depth review."""
    agent = DepthReviewAgent()
    return agent.execute(state)


# Export all
__all__ = [
    'BreadthReviewAgent', 
    'DepthReviewAgent',
    'breadth_supervisor_agent',
    'depth_supervisor_agent',
    'ReviewReport'
]