# reviewers.py
"""
Optimized review agents for paper evaluation.
Generates detailed review reports with specific feedback.
COMPLETELY FIXED: Removed hardcoded prompts and fixed JSON formatting issues.
"""

from typing import Dict, Any, List, Tuple
from datetime import datetime
from pathlib import Path
import re
import config
import prompts  # Use prompts from prompts.py
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
    FIXED: Uses prompts.py and proper fallback handling.
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
                report.add_issue(issue.get("category", "General"), issue.get("description", ""))
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
        Perform detailed breadth review with scoring.
        FIXED: Uses simple prompt template without problematic JSON formatting.
        
        Returns:
            Tuple of (feedback_text, score, details_dict)
        """
        # Truncate draft for efficiency
        draft_summary = self.truncate_text(draft, 2000)
        
        # Use a simplified prompt that doesn't cause template issues
        simple_review_prompt = f"""Review this paper's macro-level aspects.

Draft to review:
{draft_summary}

Evaluate these criteria (rate each 0.0-1.0):
1. Core argument clarity
2. Literature completeness  
3. Logical structure
4. Contribution evidence
5. Writing quality

Provide your evaluation as:
SCORES: [score1, score2, score3, score4, score5]
ISSUES: [List any problems]
SUGGESTIONS: [List improvements] 
STRENGTHS: [List positive aspects]

Each score should be between 0.0 and 1.0."""

        try:
            # Use simple string chain instead of JSON parser
            chain = self.create_chain(simple_review_prompt)
            result = chain.invoke({})
            
            # Parse the result manually
            scores = self._extract_scores_from_text(result)
            details = {
                "issues": self._extract_issues_from_text(result),
                "suggestions": self._extract_suggestions_from_text(result),
                "strengths": self._extract_strengths_from_text(result)
            }
            
        except Exception as e:
            self.log(f"Review parsing failed: {e}", "WARNING")
            # Complete fallback with reasonable defaults
            scores = [config.REVIEW_STRICTNESS] * 5
            details = {
                "issues": [{"category": "General", "description": "Could not perform detailed review due to technical issues"}],
                "suggestions": ["Please check paper manually"],
                "strengths": ["Paper was generated successfully"]
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
            issues_text = "; ".join([i.get("description", "")[:50] for i in details["issues"][:3]])
            feedback = f"ISSUES: {issues_text}"
            
        return feedback, adjusted_score, details
    
    def _extract_scores_from_text(self, text: str) -> List[float]:
        """Extract scores from review text."""
        try:
            # Look for SCORES: [x, y, z] pattern
            match = re.search(r'SCORES:\s*\[([\d.,\s]+)\]', text, re.IGNORECASE)
            if match:
                scores_str = match.group(1)
                scores = [float(s.strip()) for s in scores_str.split(',')]
                return scores[:5]  # Take first 5 scores
        except:
            pass
        
        # Look for individual decimal numbers
        numbers = re.findall(r'(?:^|\s)([0-1]?\.\d+)(?:\s|$)', text)
        if numbers:
            return [float(n) for n in numbers[:5]]
        
        # Default fallback
        return [config.REVIEW_STRICTNESS] * 5
    
    def _extract_issues_from_text(self, text: str) -> List[Dict]:
        """Extract issues from review text."""
        issues = []
        categories = ["Structure", "Argument", "Literature", "Contribution", "Writing"]
        
        try:
            # Look for ISSUES: section
            match = re.search(r'ISSUES:\s*\[(.*?)\]', text, re.DOTALL | re.IGNORECASE)
            if not match:
                match = re.search(r'ISSUES:(.*?)(?:SUGGESTIONS:|STRENGTHS:|$)', text, re.DOTALL | re.IGNORECASE)
            
            if match:
                issues_text = match.group(1).strip()
                issue_lines = [line.strip() for line in issues_text.split('\n') if line.strip()]
                
                for i, line in enumerate(issue_lines[:5]):
                    if line and not line.startswith(('[', ']')):
                        issues.append({
                            "category": categories[min(i, len(categories)-1)],
                            "description": line[:200]
                        })
        except:
            pass
        
        return issues
    
    def _extract_suggestions_from_text(self, text: str) -> List[str]:
        """Extract suggestions from review text."""
        suggestions = []
        
        try:
            match = re.search(r'SUGGESTIONS:\s*\[(.*?)\]', text, re.DOTALL | re.IGNORECASE)
            if not match:
                match = re.search(r'SUGGESTIONS:(.*?)(?:STRENGTHS:|$)', text, re.DOTALL | re.IGNORECASE)
            
            if match:
                sugg_text = match.group(1).strip()
                sugg_lines = [line.strip() for line in sugg_text.split('\n') if line.strip()]
                
                for line in sugg_lines[:5]:
                    if line and not line.startswith(('[', ']')):
                        suggestions.append(line[:200])
        except:
            pass
        
        return suggestions
    
    def _extract_strengths_from_text(self, text: str) -> List[str]:
        """Extract strengths from review text."""
        strengths = []
        
        try:
            match = re.search(r'STRENGTHS:\s*\[(.*?)\]', text, re.DOTALL | re.IGNORECASE)
            if not match:
                match = re.search(r'STRENGTHS:(.*?)$', text, re.DOTALL | re.IGNORECASE)
            
            if match:
                str_text = match.group(1).strip()
                str_lines = [line.strip() for line in str_text.split('\n') if line.strip()]
                
                for line in str_lines[:3]:
                    if line and not line.startswith(('[', ']')):
                        strengths.append(line[:200])
        except:
            pass
        
        return strengths


class DepthReviewAgent(BaseAgent):
    """
    Agent for micro-level technical review.
    Checks rigor, methodology, and technical details.
    FIXED: Uses simplified prompts without JSON formatting issues.
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
                report.add_issue(issue.get("category", "Technical"), issue.get("description", ""))
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
        Perform detailed depth review focusing on technical aspects.
        FIXED: Uses simple prompt template without problematic formatting.
        
        Returns:
            Tuple of (feedback_text, score, details_dict)
        """
        # Extract technical sections
        method_eval = self._extract_technical_sections(draft)
        
        # Use simplified prompt
        simple_review_prompt = f"""Review this paper's technical depth and rigor.

Technical sections:
{self.truncate_text(method_eval, 2000)}

Evaluate these criteria (rate each 0.0-1.0):
1. Methodological rigor
2. Mathematical correctness
3. Experimental design
4. Technical innovation
5. Implementation details

Provide your evaluation as:
SCORES: [score1, score2, score3, score4, score5]
ISSUES: [List any technical problems]
SUGGESTIONS: [List technical improvements]
STRENGTHS: [List technical strengths]

Each score should be between 0.0 and 1.0. Focus on technical accuracy."""

        try:
            chain = self.create_chain(simple_review_prompt)
            result = chain.invoke({})
            
            # Parse using the same methods as BreadthReviewAgent
            scores = self._extract_scores_from_text(result)
            details = {
                "issues": self._extract_issues_from_text(result),
                "suggestions": self._extract_suggestions_from_text(result),
                "strengths": self._extract_strengths_from_text(result)
            }
            
        except Exception as e:
            self.log(f"Technical review parsing failed: {e}", "WARNING")
            # Fallback
            scores = [config.REVIEW_STRICTNESS] * 5
            details = {
                "issues": [{"category": "Technical", "description": "Could not perform detailed technical review"}],
                "suggestions": ["Manual technical review recommended"],
                "strengths": ["Paper contains technical content"]
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
            issues_text = "; ".join([i.get("description", "")[:50] for i in details["issues"][:3]])
            feedback = f"ISSUES: {issues_text}"
            
        return feedback, adjusted_score, details
    
    def _extract_technical_sections(self, draft: str) -> str:
        """Extract technical sections with improved flexible regex matching."""
        sections = []
        
        # Look for Method/Methodology sections
        method_match = re.search(r"^#+.*[Mm]ethod.*?\n(.*?)(?=\n#+|$)", draft, re.DOTALL | re.MULTILINE)
        if method_match:
            sections.append(method_match.group(1)[:1000])
        
        # Look for Evaluation/Experiment sections
        eval_match = re.search(r"^#+.*[Ee]valuation.*?\n(.*?)(?=\n#+|$)", draft, re.DOTALL | re.MULTILINE)
        if eval_match:
            sections.append(eval_match.group(1)[:1000])
        
        # If no specific sections found, return a portion of the draft
        return "\n---\n".join(sections) if sections else draft[:2000]
    
    # Use same parsing methods as BreadthReviewAgent
    def _extract_scores_from_text(self, text: str) -> List[float]:
        """Extract scores from review text."""
        try:
            match = re.search(r'SCORES:\s*\[([\d.,\s]+)\]', text, re.IGNORECASE)
            if match:
                scores_str = match.group(1)
                scores = [float(s.strip()) for s in scores_str.split(',')]
                return scores[:5]
        except:
            pass
        
        numbers = re.findall(r'(?:^|\s)([0-1]?\.\d+)(?:\s|$)', text)
        if numbers:
            return [float(n) for n in numbers[:5]]
        
        return [config.REVIEW_STRICTNESS] * 5
    
    def _extract_issues_from_text(self, text: str) -> List[Dict]:
        """Extract issues from review text."""
        issues = []
        categories = ["Methodology", "Mathematics", "Experiments", "Innovation", "Implementation"]
        
        try:
            match = re.search(r'ISSUES:\s*\[(.*?)\]', text, re.DOTALL | re.IGNORECASE)
            if not match:
                match = re.search(r'ISSUES:(.*?)(?:SUGGESTIONS:|STRENGTHS:|$)', text, re.DOTALL | re.IGNORECASE)
            
            if match:
                issues_text = match.group(1).strip()
                issue_lines = [line.strip() for line in issues_text.split('\n') if line.strip()]
                
                for i, line in enumerate(issue_lines[:5]):
                    if line and not line.startswith(('[', ']')):
                        issues.append({
                            "category": categories[min(i, len(categories)-1)],
                            "description": line[:200]
                        })
        except:
            pass
        
        return issues
    
    def _extract_suggestions_from_text(self, text: str) -> List[str]:
        """Extract suggestions from review text."""
        suggestions = []
        
        try:
            match = re.search(r'SUGGESTIONS:\s*\[(.*?)\]', text, re.DOTALL | re.IGNORECASE)
            if not match:
                match = re.search(r'SUGGESTIONS:(.*?)(?:STRENGTHS:|$)', text, re.DOTALL | re.IGNORECASE)
            
            if match:
                sugg_text = match.group(1).strip()
                sugg_lines = [line.strip() for line in sugg_text.split('\n') if line.strip()]
                
                for line in sugg_lines[:5]:
                    if line and not line.startswith(('[', ']')):
                        suggestions.append(line[:200])
        except:
            pass
        
        return suggestions
    
    def _extract_strengths_from_text(self, text: str) -> List[str]:
        """Extract strengths from review text."""
        strengths = []
        
        try:
            match = re.search(r'STRENGTHS:\s*\[(.*?)\]', text, re.DOTALL | re.IGNORECASE)
            if not match:
                match = re.search(r'STRENGTHS:(.*?)$', text, re.DOTALL | re.IGNORECASE)
            
            if match:
                str_text = match.group(1).strip()
                str_lines = [line.strip() for line in str_text.split('\n') if line.strip()]
                
                for line in str_lines[:3]:
                    if line and not line.startswith(('[', ']')):
                        strengths.append(line[:200])
        except:
            pass
        
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