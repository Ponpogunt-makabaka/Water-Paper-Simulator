# consistency_manager.py
"""
Content Consistency Manager for research paper generation.
Implements outline consistency checking, concept dependency tracking,
logical flow enhancement, and terminology consistency.
"""

import re
import json
from typing import Dict, Any, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class Concept:
    """Represents a concept with its dependencies and definitions."""
    name: str
    definition: str = ""
    aliases: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    introduced_in_section: str = ""
    usage_count: int = 0
    
    def __post_init__(self):
        # Normalize concept name
        self.name = self.name.lower().strip()
        self.aliases = {alias.lower().strip() for alias in self.aliases}

@dataclass
class Section:
    """Represents a paper section with its outline and content."""
    title: str
    outline: str
    content: str = ""
    concepts_introduced: Set[str] = field(default_factory=set)
    concepts_used: Set[str] = field(default_factory=set)
    previous_section: Optional[str] = None
    next_section: Optional[str] = None

class ConsistencyManager:
    """
    Manages content consistency across the research paper.
    Tracks concepts, terminology, and logical flow.
    """
    
    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.sections: Dict[str, Section] = {}
        self.terminology_map: Dict[str, str] = {}  # Maps variations to canonical terms
        self.section_order: List[str] = []
        self.global_outline: str = ""
        
        # Initialize with common academic concepts
        self._initialize_base_concepts()
    
    def _initialize_base_concepts(self):
        """Initialize with fundamental academic concepts."""
        base_concepts = {
            "machine learning": Concept(
                "machine learning",
                "A subset of artificial intelligence that enables computers to learn without explicit programming",
                {"ml", "statistical learning"}
            ),
            "neural network": Concept(
                "neural network", 
                "A computing system inspired by biological neural networks",
                {"neural net", "artificial neural network", "ann"},
                {"machine learning"}
            ),
            "deep learning": Concept(
                "deep learning",
                "Machine learning using deep neural networks with multiple layers",
                {"deep neural networks", "dnn"},
                {"neural network", "machine learning"}
            ),
            "transformer": Concept(
                "transformer",
                "A neural network architecture based on self-attention mechanisms",
                {"transformer model", "transformer architecture"},
                {"neural network", "attention mechanism"}
            ),
            "attention mechanism": Concept(
                "attention mechanism",
                "A mechanism that allows models to focus on relevant parts of input",
                {"attention", "self-attention"},
                {"neural network"}
            )
        }
        
        for concept_name, concept in base_concepts.items():
            self.concepts[concept_name] = concept
    
    def set_global_outline(self, outline: str) -> None:
        """Set the global paper outline for consistency checking."""
        self.global_outline = outline
        self._extract_section_structure(outline)
    
    def _extract_section_structure(self, outline: str) -> None:
        """Extract section structure from outline."""
        # Parse outline to identify sections
        lines = outline.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Look for section headers (numbered or bulleted)
            section_match = re.match(r'^[\d\.\-\*\#]+\s*([^-]+)', line)
            if section_match:
                section_title = section_match.group(1).strip()
                section_title = re.sub(r'\s*-.*$', '', section_title)  # Remove descriptions
                
                if section_title:
                    self.section_order.append(section_title)
                    self.sections[section_title] = Section(
                        title=section_title,
                        outline=line
                    )
                    current_section = section_title
            elif current_section and line:
                # Add detail lines to current section outline
                self.sections[current_section].outline += f"\n{line}"
        
        # Set previous/next relationships
        for i, section_title in enumerate(self.section_order):
            if i > 0:
                self.sections[section_title].previous_section = self.section_order[i-1]
            if i < len(self.section_order) - 1:
                self.sections[section_title].next_section = self.section_order[i+1]
        
        logger.info(f"Extracted {len(self.section_order)} sections from outline")
    
    def add_concept(self, name: str, definition: str = "", aliases: List[str] = None, 
                   dependencies: List[str] = None, section: str = "") -> None:
        """Add a new concept to track."""
        concept = Concept(
            name=name,
            definition=definition,
            aliases=set(aliases or []),
            dependencies=set(dependencies or []),
            introduced_in_section=section
        )
        
        self.concepts[concept.name] = concept
        
        # Update terminology map
        self.terminology_map[concept.name] = concept.name
        for alias in concept.aliases:
            self.terminology_map[alias] = concept.name
        
        logger.debug(f"Added concept: {name} in section {section}")
    
    def validate_section_consistency(self, section_title: str, content: str, 
                                   section_plan: str) -> Tuple[bool, List[str]]:
        """
        Validate that section content is consistent with outline and plan.
        
        Returns:
            Tuple of (is_consistent, list_of_issues)
        """
        issues = []
        
        if section_title not in self.sections:
            issues.append(f"Section '{section_title}' not found in global outline")
            return False, issues
        
        section = self.sections[section_title]
        
        # Check outline alignment
        outline_keywords = self._extract_keywords(section.outline)
        plan_keywords = self._extract_keywords(section_plan)
        content_keywords = self._extract_keywords(content)
        
        # Verify plan aligns with outline
        outline_plan_overlap = len(outline_keywords & plan_keywords) / max(len(outline_keywords), 1)
        if outline_plan_overlap < 0.3:  # Less than 30% overlap
            issues.append(f"Section plan doesn't align well with outline (overlap: {outline_plan_overlap:.1%})")
        
        # Verify content covers planned topics
        plan_content_overlap = len(plan_keywords & content_keywords) / max(len(plan_keywords), 1)
        if plan_content_overlap < 0.5:  # Less than 50% coverage
            issues.append(f"Content doesn't fully cover planned topics (coverage: {plan_content_overlap:.1%})")
        
        # Check concept usage consistency
        concept_issues = self._check_concept_consistency(section_title, content)
        issues.extend(concept_issues)
        
        return len(issues) == 0, issues
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text."""
        # Remove common stop words and extract significant terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were'}
        
        # Extract words, filter stop words, keep significant terms
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = {word for word in words if word not in stop_words and len(word) > 3}
        
        return keywords
    
    def _check_concept_consistency(self, section_title: str, content: str) -> List[str]:
        """Check concept usage consistency in a section."""
        issues = []
        
        # Find concepts mentioned in content
        mentioned_concepts = set()
        for concept_name, concept in self.concepts.items():
            # Check main name and aliases
            all_names = {concept_name} | concept.aliases
            for name in all_names:
                if re.search(r'\b' + re.escape(name) + r'\b', content, re.IGNORECASE):
                    mentioned_concepts.add(concept_name)
                    concept.usage_count += 1
        
        # Check if dependencies are satisfied
        for concept_name in mentioned_concepts:
            concept = self.concepts[concept_name]
            for dependency in concept.dependencies:
                if dependency not in mentioned_concepts:
                    # Check if dependency was introduced in earlier sections
                    dep_introduced = False
                    section_idx = self.section_order.index(section_title) if section_title in self.section_order else -1
                    
                    for i in range(section_idx):
                        earlier_section = self.section_order[i]
                        if dependency in self.sections[earlier_section].concepts_introduced:
                            dep_introduced = True
                            break
                    
                    if not dep_introduced:
                        issues.append(f"Concept '{concept_name}' used without introducing dependency '{dependency}'")
        
        # Update section's concept tracking
        if section_title in self.sections:
            self.sections[section_title].concepts_used.update(mentioned_concepts)
        
        return issues
    
    def check_concept_dependencies(self, content: str) -> List[str]:
        """Check if all concept dependencies are satisfied."""
        issues = []
        
        # Find all concepts used in content
        used_concepts = set()
        for concept_name, concept in self.concepts.items():
            all_names = {concept_name} | concept.aliases
            for name in all_names:
                if re.search(r'\b' + re.escape(name) + r'\b', content, re.IGNORECASE):
                    used_concepts.add(concept_name)
        
        # Check dependencies
        for concept_name in used_concepts:
            concept = self.concepts[concept_name]
            for dependency in concept.dependencies:
                if dependency not in used_concepts:
                    # Check if it was defined earlier in the paper
                    if not self._is_concept_previously_defined(dependency):
                        issues.append(f"Concept '{concept_name}' requires '{dependency}' to be defined first")
        
        return issues
    
    def _is_concept_previously_defined(self, concept_name: str) -> bool:
        """Check if a concept was defined in previous sections."""
        for section_title in self.section_order:
            if concept_name in self.sections[section_title].concepts_introduced:
                return True
        return False
    
    def enhance_logical_flow(self, section_title: str, content: str) -> str:
        """Enhance logical flow by adding transition sentences."""
        if section_title not in self.sections:
            return content
        
        section = self.sections[section_title]
        enhanced_content = content
        
        # Add transition from previous section
        if section.previous_section:
            prev_section = section.previous_section
            transition_intro = self._generate_transition_sentence(prev_section, section_title, "intro")
            if transition_intro:
                enhanced_content = transition_intro + "\n\n" + enhanced_content
        
        # Add transition to next section
        if section.next_section:
            next_section = section.next_section
            transition_outro = self._generate_transition_sentence(section_title, next_section, "outro")
            if transition_outro:
                enhanced_content = enhanced_content + "\n\n" + transition_outro
        
        # Enhance paragraph connections within the section
        enhanced_content = self._enhance_paragraph_connections(enhanced_content)
        
        return enhanced_content
    
    def _generate_transition_sentence(self, from_section: str, to_section: str, 
                                    transition_type: str) -> str:
        """Generate transition sentences between sections."""
        transition_templates = {
            ("Introduction", "Literature Review", "outro"): 
                "Having established the research context and objectives, we now turn to examining the existing body of work in this area.",
            
            ("Literature Review", "Method", "outro"): 
                "Building upon the insights from previous research, we now present our proposed methodology.",
            
            ("Method", "Evaluation", "outro"): 
                "With our approach clearly defined, we proceed to evaluate its effectiveness through comprehensive experiments.",
            
            ("Evaluation", "Conclusion", "outro"): 
                "Based on the experimental results and analysis presented above, we can now draw conclusions about our work.",
            
            ("Abstract", "Introduction", "outro"): 
                "This paper explores these concepts in detail, beginning with the broader context and motivation.",
            
            # Intro transitions
            ("Literature Review", "Method", "intro"): 
                "Given the gaps and opportunities identified in the literature, our methodology addresses these challenges through the following approach.",
            
            ("Method", "Evaluation", "intro"): 
                "To validate the effectiveness of our proposed approach, we designed a comprehensive evaluation framework.",
            
            ("Evaluation", "Conclusion", "intro"): 
                "The experimental results provide valuable insights into the performance and limitations of our approach."
        }
        
        # Look for specific transition
        key = (from_section, to_section, transition_type)
        if key in transition_templates:
            return transition_templates[key]
        
        # Generate generic transition
        if transition_type == "outro":
            return f"In the following section, we will examine {to_section.lower()} in greater detail."
        else:  # intro
            return f"Building on the {from_section.lower()} presented previously, this section focuses on {to_section.lower()}."
    
    def _enhance_paragraph_connections(self, content: str) -> str:
        """Add connecting phrases between paragraphs."""
        paragraphs = content.split('\n\n')
        
        if len(paragraphs) <= 1:
            return content
        
        connection_phrases = [
            "Furthermore,", "Additionally,", "Moreover,", "In contrast,", 
            "However,", "Nevertheless,", "Consequently,", "As a result,",
            "Building on this,", "Similarly,", "On the other hand,", 
            "This approach", "Subsequently,", "Therefore,"
        ]
        
        enhanced_paragraphs = [paragraphs[0]]  # Keep first paragraph as-is
        
        for i in range(1, len(paragraphs)):
            para = paragraphs[i].strip()
            if not para:
                continue
            
            # Check if paragraph already starts with a connection phrase
            has_connector = any(para.startswith(phrase) for phrase in connection_phrases)
            
            if not has_connector and len(para) > 50:  # Only for substantial paragraphs
                # Choose appropriate connector based on content
                if any(word in para.lower() for word in ['however', 'but', 'although', 'despite']):
                    connector = "However, "
                elif any(word in para.lower() for word in ['result', 'therefore', 'thus', 'hence']):
                    connector = "Consequently, "
                elif any(word in para.lower() for word in ['also', 'addition', 'furthermore']):
                    connector = "Additionally, "
                else:
                    connector = "Furthermore, "
                
                # Don't add connector if sentence already has good flow
                if not re.match(r'^(This|These|The|Our|We|It)\s', para):
                    para = connector + para[0].lower() + para[1:]
            
            enhanced_paragraphs.append(para)
        
        return '\n\n'.join(enhanced_paragraphs)
    
    def normalize_terminology(self, content: str) -> str:
        """Ensure consistent terminology throughout the text."""
        normalized_content = content
        
        # Apply terminology mappings
        for variant, canonical in self.terminology_map.items():
            if variant != canonical:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(variant) + r'\b'
                normalized_content = re.sub(pattern, canonical, normalized_content, flags=re.IGNORECASE)
        
        # Ensure consistent acronym usage
        normalized_content = self._normalize_acronyms(normalized_content)
        
        return normalized_content
    
    def _normalize_acronyms(self, content: str) -> str:
        """Normalize acronym usage (spell out first time, then use acronym)."""
        acronym_patterns = {
            r'\b(artificial intelligence|AI)\b': 'artificial intelligence (AI)',
            r'\b(machine learning|ML)\b': 'machine learning (ML)', 
            r'\b(natural language processing|NLP)\b': 'natural language processing (NLP)',
            r'\b(deep neural network|DNN)\b': 'deep neural network (DNN)',
            r'\b(convolutional neural network|CNN)\b': 'convolutional neural network (CNN)',
            r'\b(recurrent neural network|RNN)\b': 'recurrent neural network (RNN)'
        }
        
        # Track which acronyms have been introduced
        introduced = set()
        
        for pattern, full_form in acronym_patterns.items():
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            
            if matches:
                # Replace first occurrence with full form
                first_match = matches[0]
                acronym = first_match.group().upper()
                
                if acronym not in introduced:
                    content = content[:first_match.start()] + full_form + content[first_match.end():]
                    introduced.add(acronym)
                    
                    # Update subsequent matches to use acronym only
                    short_form = full_form.split('(')[1].rstrip(')')  # Extract acronym
                    content = re.sub(pattern, short_form, content[first_match.end():], flags=re.IGNORECASE)
        
        return content
    
    def get_consistency_report(self) -> Dict[str, Any]:
        """Generate a comprehensive consistency report."""
        report = {
            "total_concepts": len(self.concepts),
            "total_sections": len(self.sections),
            "concept_usage": {},
            "terminology_mappings": len(self.terminology_map),
            "section_flow": [],
            "issues": []
        }
        
        # Concept usage statistics
        for name, concept in self.concepts.items():
            report["concept_usage"][name] = {
                "usage_count": concept.usage_count,
                "dependencies": list(concept.dependencies),
                "introduced_in": concept.introduced_in_section
            }
        
        # Section flow analysis
        for section_title in self.section_order:
            section = self.sections[section_title]
            report["section_flow"].append({
                "title": section_title,
                "concepts_introduced": len(section.concepts_introduced),
                "concepts_used": len(section.concepts_used),
                "has_content": bool(section.content)
            })
        
        return report
    
    def update_section_content(self, section_title: str, content: str) -> None:
        """Update section content and track concepts."""
        if section_title in self.sections:
            self.sections[section_title].content = content
            
            # Extract concepts introduced in this section
            introduced_concepts = self._extract_introduced_concepts(content)
            self.sections[section_title].concepts_introduced.update(introduced_concepts)
            
            # Mark concepts as introduced in this section
            for concept_name in introduced_concepts:
                if concept_name in self.concepts:
                    self.concepts[concept_name].introduced_in_section = section_title

    def _extract_introduced_concepts(self, content: str) -> Set[str]:
        """Extract concepts that are being defined/introduced in the content."""
        introduced = set()
        
        # Look for definition patterns
        definition_patterns = [
            r'(\w+(?:\s+\w+){0,2})\s+is\s+defined\s+as',
            r'(\w+(?:\s+\w+){0,2})\s+refers\s+to',
            r'we\s+define\s+(\w+(?:\s+\w+){0,2})\s+as',
            r'(\w+(?:\s+\w+){0,2})\s+can\s+be\s+understood\s+as',
            r'the\s+concept\s+of\s+(\w+(?:\s+\w+){0,2})'
        ]
        
        for pattern in definition_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                concept = match.group(1).lower().strip()
                if len(concept) > 2:  # Avoid very short matches
                    introduced.add(concept)
        
        return introduced

# Factory function
def create_consistency_manager() -> ConsistencyManager:
    """Create and return a ConsistencyManager instance."""
    return ConsistencyManager()

# Export
__all__ = ['ConsistencyManager', 'Concept', 'Section', 'create_consistency_manager']