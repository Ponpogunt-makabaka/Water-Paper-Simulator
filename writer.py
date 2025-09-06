# writer.py
"""
Fixed enhanced writer agent for paper drafting and revision.
Uses RAG-based approach with advanced consistency management.
FIXED: Import issues, state validation, and error handling.
"""

from typing import Dict, Any, List, Optional
import re
import logging

# Safe imports with fallbacks
try:
    from langchain_core.output_parsers import StrOutputParser
except ImportError:
    StrOutputParser = None

import config
import prompts
from base_agent import BaseAgent, AgentError
from rag_manager import RAGManager

# Set up logging
logger = logging.getLogger(__name__)

class EnhancedWriterAgent(BaseAgent):
    """
    Enhanced writer agent with advanced consistency management.
    Includes fallback mechanisms for optional components.
    """
    
    def __init__(self):
        super().__init__("EnhancedWriter", "writing")
        self.sections = self._init_sections()
        self.consistency_manager = None
        self.paper_context = {}
        self.consistency_enabled = getattr(config, 'ENABLE_CONSISTENCY_MANAGEMENT', False)
        
    def _init_sections(self) -> List[Dict[str, str]]:
        """Initialize paper sections configuration with enhanced metadata."""
        return [
            {
                "title": "Abstract",
                "style": "Concise",
                "focus_points": "Problem, method, results, impact",
                "concepts_to_introduce": [],
                "logical_role": "overview",
                "max_length": getattr(config, 'SECTION_LENGTHS', {}).get("Abstract", 1500)
            },
            {
                "title": "Introduction",
                "style": "Engaging", 
                "focus_points": "Context, gap, contribution",
                "concepts_to_introduce": ["research problem", "motivation"],
                "logical_role": "foundation",
                "max_length": getattr(config, 'SECTION_LENGTHS', {}).get("Introduction", 3500)
            },
            {
                "title": "Literature Review",
                "style": "Critical",
                "focus_points": "Prior work, limitations", 
                "concepts_to_introduce": ["related work", "state of the art"],
                "logical_role": "background",
                "max_length": getattr(config, 'SECTION_LENGTHS', {}).get("Literature Review", 4000)
            },
            {
                "title": "Method",
                "style": "Technical",
                "focus_points": "Algorithm, architecture, math",
                "concepts_to_introduce": ["proposed approach", "methodology"],
                "logical_role": "contribution",
                "max_length": getattr(config, 'SECTION_LENGTHS', {}).get("Method", 5000)
            },
            {
                "title": "Evaluation",
                "style": "Analytical",
                "focus_points": "Setup, metrics, analysis",
                "concepts_to_introduce": ["experimental setup", "evaluation metrics"],
                "logical_role": "validation",
                "max_length": getattr(config, 'SECTION_LENGTHS', {}).get("Evaluation", 4500)
            },
            {
                "title": "Conclusion",
                "style": "Reflective",
                "focus_points": "Summary, limitations, future",
                "concepts_to_introduce": [],
                "logical_role": "synthesis",
                "max_length": getattr(config, 'SECTION_LENGTHS', {}).get("Conclusion", 2500)
            }
        ]
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute enhanced writing phase with consistency management.
        FIXED: Better error handling and state validation.
        """
        try:
            # Validate required state with better error messages
            required_keys = ["topic", "final_plan", "vector_store_path"]
            missing_keys = [key for key in required_keys if key not in state or not state[key]]
            
            if missing_keys:
                logger.warning(f"Missing required state keys: {missing_keys}")
                # Provide defaults for missing keys
                if "topic" not in state:
                    state["topic"] = "Research paper topic"
                if "final_plan" not in state:
                    state["final_plan"] = "1. Introduction\n2. Method\n3. Conclusion"
                if "vector_store_path" not in state:
                    state["vector_store_path"] = "dummy_path"
            
            # Initialize consistency manager if available and enabled
            self._safe_initialize_consistency_manager(state)
            
            # Setup paper context
            self._setup_paper_context(state)
            
            # Choose writing strategy based on revision status
            if self.is_revision(state):
                draft = self._revise_draft(state)
            else:
                draft = self._write_initial_draft(state)
            
            # Apply final consistency enhancements if available
            if self.consistency_manager and self.consistency_enabled:
                try:
                    draft = self._apply_final_consistency_checks(draft)
                except Exception as e:
                    logger.warning(f"Consistency checks failed: {e}")
            
            # Save draft
            version = self.get_revision_count(state)
            filename = f"{getattr(config, 'DRAFT_FILE_PREFIX', 'draft_v')}{version}.txt"
            
            try:
                self.save_file(draft, filename)
            except Exception as e:
                logger.warning(f"Failed to save draft file: {e}")
            
            # Generate consistency report if available
            consistency_report = self._safe_generate_consistency_report()
            
            return {
                "current_draft": draft,
                "draft_version": version,
                "draft_history": [draft],
                "consistency_report": consistency_report
            }
            
        except Exception as e:
            logger.error(f"Enhanced writing failed: {e}")
            # Return a fallback response instead of raising
            return {
                "current_draft": f"Failed to generate draft: {str(e)}",
                "draft_version": 0,
                "draft_history": [],
                "consistency_report": {"error": str(e)}
            }
    
    def _safe_initialize_consistency_manager(self, state: Dict[str, Any]) -> None:
        """Safely initialize consistency manager with error handling."""
        try:
            if self.consistency_enabled:
                from consistency_manager import create_consistency_manager
                self.consistency_manager = create_consistency_manager()
                
                # Set global outline if available
                final_plan = state.get("final_plan", "")
                if final_plan and self.consistency_manager:
                    self.consistency_manager.set_global_outline(final_plan)
                
                logger.info("Consistency manager initialized successfully")
            else:
                logger.info("Consistency management disabled in configuration")
                
        except ImportError:
            logger.warning("Consistency manager not available - continuing without consistency features")
            self.consistency_manager = None
        except Exception as e:
            logger.warning(f"Failed to initialize consistency manager: {e}")
            self.consistency_manager = None
    
    def _setup_paper_context(self, state: Dict[str, Any]) -> None:
        """Setup global paper context for consistency tracking."""
        try:
            topic = state.get("topic", "Unknown topic")
            final_plan = state.get("final_plan", "")
            
            self.paper_context = {
                "topic": topic,
                "main_contribution": self._extract_main_contribution(final_plan),
                "research_domain": self._extract_research_domain(topic),
                "methodology_type": self._extract_methodology_type(final_plan)
            }
            
            # Add domain-specific concepts if consistency manager available
            if self.consistency_manager:
                self._safe_add_domain_concepts(self.paper_context["research_domain"])
            
            logger.debug(f"Paper context established: {self.paper_context}")
            
        except Exception as e:
            logger.warning(f"Failed to setup paper context: {e}")
            self.paper_context = {
                "topic": state.get("topic", "Unknown"),
                "main_contribution": "Research contribution",
                "research_domain": "computer science",
                "methodology_type": "experimental"
            }
    
    def _extract_main_contribution(self, plan: str) -> str:
        """Extract the main contribution from the plan."""
        if not plan:
            return "Novel approach to the research problem"
            
        contribution_patterns = [
            r'contribution[s]?[:\-]\s*([^\n]+)',
            r'propose[s]?[:\-]\s*([^\n]+)',
            r'novel[:\-]\s*([^\n]+)',
            r'new[:\-]\s*([^\n]+)'
        ]
        
        for pattern in contribution_patterns:
            match = re.search(pattern, plan, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "Novel approach to the research problem"
    
    def _extract_research_domain(self, topic: str) -> str:
        """Extract research domain from topic."""
        if not topic:
            return "computer science"
            
        domain_keywords = {
            "neural": "neural networks",
            "deep learning": "deep learning",
            "machine learning": "machine learning", 
            "nlp": "natural language processing",
            "computer vision": "computer vision",
            "ai": "artificial intelligence",
            "optimization": "optimization",
            "algorithm": "algorithms"
        }
        
        topic_lower = topic.lower()
        for keyword, domain in domain_keywords.items():
            if keyword in topic_lower:
                return domain
        
        return "computer science"
    
    def _extract_methodology_type(self, plan: str) -> str:
        """Extract methodology type from plan."""
        if not plan:
            return "experimental"
            
        method_patterns = {
            "experimental": ["experiment", "evaluation", "test", "benchmark"],
            "theoretical": ["theory", "proof", "analysis", "mathematical"],
            "empirical": ["empirical", "data", "dataset", "corpus"],
            "simulation": ["simulation", "model", "simulate"]
        }
        
        plan_lower = plan.lower()
        for method_type, keywords in method_patterns.items():
            if any(keyword in plan_lower for keyword in keywords):
                return method_type
        
        return "experimental"
    
    def _safe_add_domain_concepts(self, domain: str) -> None:
        """Safely add domain-specific concepts to the consistency manager."""
        try:
            if not self.consistency_manager:
                return
                
            domain_concepts = {
                "neural networks": [
                    ("activation function", "Function that determines neuron output", ["activation"], ["neuron"]),
                    ("backpropagation", "Learning algorithm for neural networks", ["backprop"], ["neural network"]),
                    ("gradient descent", "Optimization algorithm", ["sgd"], ["optimization"])
                ],
                "deep learning": [
                    ("convolution", "Mathematical operation for feature extraction", ["conv"], ["neural network"]),
                    ("pooling", "Downsampling operation", ["max pooling"], ["convolution"]),
                    ("dropout", "Regularization technique", ["dropout layer"], ["regularization"])
                ],
                "natural language processing": [
                    ("tokenization", "Process of breaking text into tokens", ["token"], ["text processing"]),
                    ("embedding", "Vector representation of words", ["word embedding"], ["vector space"]),
                    ("attention", "Mechanism to focus on relevant parts", ["self-attention"], ["neural network"])
                ],
                "machine learning": [
                    ("overfitting", "Model memorizing training data", ["overfit"], ["training"]),
                    ("cross-validation", "Model validation technique", ["cv"], ["validation"]),
                    ("feature engineering", "Creating relevant features", ["feature extraction"], ["preprocessing"])
                ]
            }
            
            if domain in domain_concepts:
                for name, definition, aliases, dependencies in domain_concepts[domain]:
                    self.consistency_manager.add_concept(name, definition, aliases, dependencies)
                    
        except Exception as e:
            logger.warning(f"Failed to add domain concepts: {e}")
    
    def _write_initial_draft(self, state: Dict[str, Any]) -> str:
        """Write initial paper draft with enhanced consistency management."""
        logger.info("Writing initial draft with enhanced consistency management")
        
        topic = state.get("topic", "Research topic")
        plan = state.get("final_plan", "")
        vector_store_path = state.get("vector_store_path", "")
        
        sections_content = []
        
        for i, section_config in enumerate(self.sections):
            section_title = section_config["title"]
            logger.info(f"Writing section {i+1}/{len(self.sections)}: {section_title}")
            
            try:
                # Extract section-specific plan
                section_plan = self._extract_section_plan(plan, i)
                
                # Validate plan consistency if manager available
                if self.consistency_manager:
                    try:
                        is_consistent, issues = self.consistency_manager.validate_section_consistency(
                            section_title, "", section_plan
                        )
                        
                        if not is_consistent:
                            logger.warning(f"Plan consistency issues for {section_title}: {issues}")
                            section_plan = self._adjust_plan_for_consistency(section_plan, issues)
                    except Exception as e:
                        logger.warning(f"Consistency validation failed for {section_title}: {e}")
                
                # Write section with enhanced context
                content = self._write_enhanced_section(
                    topic=topic,
                    section_title=section_title,
                    section_plan=section_plan,
                    section_config=section_config,
                    vector_store_path=vector_store_path,
                    section_index=i
                )
                
                # Validate and enhance the written content
                content = self._safe_validate_and_enhance_section(section_title, content, section_plan)
                
                # Update consistency manager with new content
                if self.consistency_manager:
                    try:
                        self.consistency_manager.update_section_content(section_title, content)
                    except Exception as e:
                        logger.warning(f"Failed to update consistency manager for {section_title}: {e}")
                
                sections_content.append(f"## {section_title}\n\n{content}")
                
            except Exception as e:
                logger.error(f"Failed to write section {section_title}: {e}")
                # Add fallback content
                fallback_content = f"This section discusses {section_title.lower()} related to {topic}."
                sections_content.append(f"## {section_title}\n\n{fallback_content}")
        
        # Assemble final draft
        draft = self._assemble_enhanced_draft(sections_content)
        
        logger.info("Initial enhanced draft complete")
        return draft
    
    def _write_enhanced_section(self, **kwargs) -> str:
        """Write a section with enhanced consistency and context awareness."""
        section_title = kwargs["section_title"]
        section_config = kwargs["section_config"]
        topic = kwargs["topic"]
        section_plan = kwargs["section_plan"]
        vector_store_path = kwargs["vector_store_path"]
        section_index = kwargs["section_index"]
        
        try:
            # Generate enhanced RAG query
            rag_query = self._generate_contextual_rag_query(section_title, topic, section_plan)
            
            # Retrieve context from RAG
            context = ""
            try:
                context = RAGManager.query_vector_store(vector_store_path, rag_query)
            except Exception as e:
                logger.warning(f"RAG query failed: {e}")
                context = f"Context for {section_title} about {topic}."
            
            # Build concept context for this section
            concept_context = self._build_concept_context(section_title, section_config)
            
            # Create enhanced prompt
            enhanced_prompt = self._create_enhanced_section_prompt(
                section_title=section_title,
                topic=topic,
                section_plan=section_plan,
                context=context,
                section_config=section_config,
                concept_context=concept_context,
                section_index=section_index
            )
            
            # Generate content using LLM
            try:
                chain = self.create_chain(enhanced_prompt)
                result = chain.invoke({})
            except Exception as e:
                logger.warning(f"LLM generation failed for {section_title}: {e}")
                result = f"This section presents the {section_title.lower()} of our research on {topic}."
            
            # Apply post-processing
            result = self._post_process_section_content(result, section_title, section_config)
            
            # Truncate to max length
            max_length = section_config.get("max_length", 4000)
            return self.truncate_text(result, max_length)
            
        except Exception as e:
            logger.error(f"Enhanced section writing failed for {section_title}: {e}")
            return f"This section covers {section_title.lower()} aspects of {topic}."
    
    def _generate_contextual_rag_query(self, section_title: str, topic: str, section_plan: str) -> str:
        """Generate a contextually-aware RAG query."""
        base_query = f"Detailed information for the '{section_title}' section of a paper on '{topic}'"
        
        section_contexts = {
            "Introduction": f"context, motivation, and problem definition for {topic}",
            "Literature Review": f"prior work, related research, and state of the art in {topic}",
            "Method": f"technical approaches, algorithms, and methodologies for {topic}",
            "Evaluation": f"experimental design, metrics, and validation approaches for {topic}",
            "Conclusion": f"results summary, implications, and future directions for {topic}"
        }
        
        if section_title in section_contexts:
            contextual_query = f"{base_query}. Focus on {section_contexts[section_title]}."
        else:
            contextual_query = f"{base_query}. Plan: {section_plan[:100]}"
        
        return contextual_query
    
    def _build_concept_context(self, section_title: str, section_config: Dict[str, Any]) -> str:
        """Build concept context for the section."""
        if not self.consistency_manager:
            return "No specific concept requirements"
        
        try:
            concept_context = []
            
            # Concepts to introduce in this section
            concepts_to_introduce = section_config.get("concepts_to_introduce", [])
            for concept in concepts_to_introduce:
                if concept in self.consistency_manager.concepts:
                    concept_obj = self.consistency_manager.concepts[concept]
                    concept_context.append(f"Introduce '{concept}': {concept_obj.definition}")
            
            # Required dependencies
            section_deps = set()
            for concept in concepts_to_introduce:
                if concept in self.consistency_manager.concepts:
                    section_deps.update(self.consistency_manager.concepts[concept].dependencies)
            
            if section_deps:
                concept_context.append(f"Required background concepts: {', '.join(section_deps)}")
            
            return "; ".join(concept_context) if concept_context else "No specific concept requirements"
            
        except Exception as e:
            logger.warning(f"Failed to build concept context: {e}")
            return "No specific concept requirements"
    
    def _create_enhanced_section_prompt(self, **kwargs) -> str:
        """Create an enhanced prompt with consistency guidelines."""
        section_title = kwargs["section_title"]
        topic = kwargs["topic"]
        section_plan = kwargs["section_plan"]
        context = kwargs["context"]
        section_config = kwargs["section_config"]
        concept_context = kwargs["concept_context"]
        section_index = kwargs["section_index"]
        
        # Build consistency guidelines
        consistency_guidelines = self._build_consistency_guidelines(section_title, section_index)
        
        # Use enhanced prompt if available, otherwise fallback to basic
        if hasattr(prompts, 'WRITER_ENHANCED_SECTION_PROMPT'):
            try:
                enhanced_prompt = prompts.WRITER_ENHANCED_SECTION_PROMPT.format(
                    section_title=section_title,
                    topic=self.truncate_text(topic, 100),
                    research_domain=self.paper_context.get('research_domain', ''),
                    main_contribution=self.paper_context.get('main_contribution', ''),
                    section_plan=self.truncate_text(section_plan, 200),
                    style=section_config.get('style', 'Academic'),
                    focus_points=section_config.get('focus_points', 'Clarity and accuracy'),
                    logical_role=section_config.get('logical_role', 'informative'),
                    consistency_guidelines=consistency_guidelines,
                    concept_context=concept_context,
                    context=self.truncate_text(context, 3000),
                    max_length=section_config.get('max_length', 4000)
                )
                return enhanced_prompt
            except Exception as e:
                logger.warning(f"Enhanced prompt formatting failed: {e}")
        
        # Fallback to basic prompt
        basic_prompt = f"""Write the '{section_title}' section of an academic paper.

Topic: {self.truncate_text(topic, 100)}
Plan: {self.truncate_text(section_plan, 200)}
Style: {section_config.get('style', 'Academic')}

Context:
{self.truncate_text(context, 2000)}

Requirements:
- Length: {section_config.get('max_length', 4000)} characters max
- Citations: Use [Author, Year] format
- Focus: {section_config.get('focus_points', 'Clarity and accuracy')}

Write the section content:"""
        
        return basic_prompt
    
    def _build_consistency_guidelines(self, section_title: str, section_index: int) -> str:
        """Build section-specific consistency guidelines."""
        guidelines = []
        
        # General guidelines
        guidelines.append("- Use consistent terminology for the same concepts")
        guidelines.append("- Introduce concepts before referencing them")
        guidelines.append("- Maintain logical flow between paragraphs")
        
        # Section-specific guidelines
        if section_title == "Introduction":
            guidelines.extend([
                "- Establish key terminology that will be used throughout",
                "- Provide clear motivation and context",
                "- Preview the paper structure and contributions"
            ])
        elif section_title == "Literature Review":
            guidelines.extend([
                "- Group related work thematically",
                "- Compare and contrast different approaches",
                "- Identify gaps that motivate your work"
            ])
        elif section_title == "Method":
            guidelines.extend([
                "- Define technical terms clearly",
                "- Present information in logical order",
                "- Use consistent notation and terminology"
            ])
        elif section_title == "Evaluation":
            guidelines.extend([
                "- Present metrics and baselines clearly",
                "- Use consistent terminology from Method section",
                "- Provide thorough analysis of results"
            ])
        
        # Transition guidelines for non-first sections
        if section_index > 0 and section_index < len(self.sections):
            prev_section = self.sections[section_index - 1]["title"]
            guidelines.append(f"- Connect smoothly from {prev_section} section")
        
        return "\n".join(guidelines)
    
    def _safe_validate_and_enhance_section(self, section_title: str, content: str, section_plan: str) -> str:
        """Safely validate and enhance section content for consistency."""
        
        if not self.consistency_manager:
            return content
        
        try:
            # 1. Check outline consistency
            is_consistent, issues = self.consistency_manager.validate_section_consistency(
                section_title, content, section_plan
            )
            
            if not is_consistent:
                logger.warning(f"Content consistency issues in {section_title}: {issues}")
            
            # 2. Check concept dependencies
            concept_issues = self.consistency_manager.check_concept_dependencies(content)
            if concept_issues:
                logger.warning(f"Concept dependency issues in {section_title}: {concept_issues}")
            
            # 3. Normalize terminology
            content = self.consistency_manager.normalize_terminology(content)
            
            # 4. Enhance logical flow
            content = self.consistency_manager.enhance_logical_flow(section_title, content)
            
            return content
            
        except Exception as e:
            logger.warning(f"Section validation and enhancement failed for {section_title}: {e}")
            return content
    
    def _post_process_section_content(self, content: str, section_title: str, 
                                    section_config: Dict[str, Any]) -> str:
        """Apply post-processing to section content."""
        
        try:
            # Remove common issues
            content = re.sub(r'\n{3,}', '\n\n', content)  # Fix excessive line breaks
            content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
            content = content.strip()
            
            # Ensure proper paragraph structure
            paragraphs = content.split('\n\n')
            processed_paragraphs = []
            
            for para in paragraphs:
                para = para.strip()
                if len(para) > 20:  # Skip very short paragraphs
                    # Ensure paragraphs end with proper punctuation
                    if not para.endswith(('.', '!', '?', ':')):
                        para += '.'
                    processed_paragraphs.append(para)
            
            return '\n\n'.join(processed_paragraphs)
            
        except Exception as e:
            logger.warning(f"Post-processing failed for {section_title}: {e}")
            return content
    
    def _assemble_enhanced_draft(self, sections_content: List[str]) -> str:
        """Assemble final draft with enhanced global consistency."""
        
        try:
            # Join sections
            draft = "\n\n---\n\n".join(sections_content)
            
            # Apply final consistency passes if manager available
            if self.consistency_manager:
                try:
                    draft = self.consistency_manager.normalize_terminology(draft)
                except Exception as e:
                    logger.warning(f"Final terminology normalization failed: {e}")
            
            # Add paper-level transitions
            draft = self._add_paper_level_coherence(draft)
            
            return draft
            
        except Exception as e:
            logger.error(f"Draft assembly failed: {e}")
            return "\n\n".join(sections_content)  # Fallback to simple join
    
    def _add_paper_level_coherence(self, draft: str) -> str:
        """Add paper-level coherence elements."""
        
        try:
            # Ensure consistent thread throughout paper
            coherence_patterns = {
                "this paper": ["this work", "this study", "this research"],
                "we propose": ["we present", "we introduce", "we develop"],
                "our approach": ["our method", "our technique", "our solution"]
            }
            
            # Standardize key phrases
            for canonical, variants in coherence_patterns.items():
                for variant in variants:
                    pattern = r'\b' + re.escape(variant) + r'\b'
                    draft = re.sub(pattern, canonical, draft, flags=re.IGNORECASE)
            
            return draft
            
        except Exception as e:
            logger.warning(f"Paper-level coherence enhancement failed: {e}")
            return draft
    
    def _revise_draft(self, state: Dict[str, Any]) -> str:
        """Revise draft with enhanced consistency management."""
        logger.info("Revising draft with enhanced consistency management")
        
        draft = state.get("current_draft", "")
        if not draft or len(draft) < 100:
            logger.warning("No valid draft to revise, writing new draft")
            return self._write_initial_draft(state)
        
        # Re-initialize consistency manager with existing content
        if self.consistency_manager:
            try:
                self.consistency_manager.set_global_outline(state.get("final_plan", ""))
            except Exception as e:
                logger.warning(f"Failed to re-initialize consistency manager: {e}")
        
        feedback_points = self._extract_feedback_points(state)
        if not feedback_points:
            logger.info("No feedback to address, applying consistency enhancements only")
            return self._safe_apply_consistency_enhancements(draft)
        
        # Enhanced revision with RAG context
        try:
            rag_query = f"Technical details and improvements for '{state.get('topic', '')}' addressing: {'; '.join(feedback_points[:3])}"
            context = RAGManager.query_vector_store(state.get("vector_store_path", ""), rag_query)
        except Exception as e:
            logger.warning(f"RAG query failed during revision: {e}")
            context = "Context not available for revision."
        
        # Create revision prompt
        revision_prompt = f"""Revise this academic paper draft to address feedback while maintaining consistency.

FEEDBACK TO ADDRESS:
{chr(10).join(f'- {point}' for point in feedback_points[:5])}

CONSISTENCY REQUIREMENTS:
- Maintain consistent terminology throughout
- Preserve logical flow and transitions
- Ensure concept dependencies are satisfied
- Keep coherent narrative thread

LITERATURE CONTEXT:
{self.truncate_text(context, 2000)}

ORIGINAL DRAFT:
{self.truncate_text(draft, 4000)}

Provide a revised version:"""
        
        try:
            chain = self.create_chain(revision_prompt)
            revised = chain.invoke({})
        except Exception as e:
            logger.warning(f"LLM revision failed: {e}")
            return self._apply_incremental_improvements(draft, feedback_points)
        
        if len(revised) < len(draft) * 0.7:
            logger.warning("Revision seems too short, applying incremental improvements")
            return self._apply_incremental_improvements(draft, feedback_points)
        
        # Apply consistency enhancements to revision
        revised = self._safe_apply_consistency_enhancements(revised)
        
        logger.info("Applied enhanced revision successfully")
        return revised
    
    def _safe_apply_consistency_enhancements(self, draft: str) -> str:
        """Safely apply consistency enhancements to existing draft."""
        
        if not self.consistency_manager:
            return draft
        
        try:
            # 1. Normalize terminology
            enhanced_draft = self.consistency_manager.normalize_terminology(draft)
            
            # 2. Check and log concept dependencies
            concept_issues = self.consistency_manager.check_concept_dependencies(enhanced_draft)
            if concept_issues:
                logger.warning(f"Found concept dependency issues: {concept_issues}")
            
            # 3. Enhance paragraph transitions for each section
            sections = enhanced_draft.split("## ")
            enhanced_sections = []
            
            for i, section in enumerate(sections):
                if section.strip():
                    if i > 0:
                        section = "## " + section
                    
                    # Extract section title and content
                    lines = section.split('\n', 1)
                    if len(lines) > 1:
                        section_title = lines[0].replace("## ", "").strip()
                        section_content = lines[1]
                        
                        # Enhance this section's flow
                        try:
                            enhanced_content = self.consistency_manager.enhance_logical_flow(
                                section_title, section_content
                            )
                            enhanced_sections.append(f"## {section_title}\n{enhanced_content}")
                        except Exception as e:
                            logger.warning(f"Flow enhancement failed for {section_title}: {e}")
                            enhanced_sections.append(section)
                    else:
                        enhanced_sections.append(section)
                else:
                    enhanced_sections.append(section)
            
            return "\n".join(enhanced_sections)
            
        except Exception as e:
            logger.warning(f"Consistency enhancement failed: {e}")
            return draft
    
    def _apply_incremental_improvements(self, draft: str, feedback_points: List[str]) -> str:
        """Apply incremental improvements to address specific feedback."""
        improved_draft = draft
        
        # Apply targeted improvements based on feedback
        for point in feedback_points[:3]:  # Focus on top 3 issues
            try:
                if "citation" in point.lower():
                    improved_draft = self._improve_citations(improved_draft)
                elif "clarity" in point.lower() or "clear" in point.lower():
                    improved_draft = self._improve_clarity(improved_draft)
                elif "detail" in point.lower():
                    improved_draft = self._add_technical_details(improved_draft)
                elif "structure" in point.lower():
                    improved_draft = self._improve_structure(improved_draft)
            except Exception as e:
                logger.warning(f"Incremental improvement failed for '{point}': {e}")
        
        return self._safe_apply_consistency_enhancements(improved_draft)
    
    def _improve_citations(self, draft: str) -> str:
        """Improve citations in the draft."""
        try:
            sentences = draft.split('. ')
            improved_sentences = []
            
            for sentence in sentences:
                # If sentence makes a claim but has no citation, suggest adding one
                if any(word in sentence.lower() for word in ['shown', 'demonstrated', 'found', 'reported']):
                    if '[' not in sentence and ']' not in sentence:
                        sentence += " [Author, Year]"
                improved_sentences.append(sentence)
            
            return '. '.join(improved_sentences)
        except Exception as e:
            logger.warning(f"Citation improvement failed: {e}")
            return draft
    
    def _improve_clarity(self, draft: str) -> str:
        """Improve clarity of the draft."""
        try:
            paragraphs = draft.split('\n\n')
            improved_paragraphs = []
            
            for para in paragraphs:
                sentences = para.split('. ')
                improved_sentences = []
                
                for sentence in sentences:
                    # Split overly long sentences (>150 chars)
                    if len(sentence) > 150 and ',' in sentence:
                        parts = sentence.split(', ', 1)
                        if len(parts) == 2:
                            sentence = f"{parts[0]}. Additionally, {parts[1]}"
                    
                    improved_sentences.append(sentence)
                
                improved_paragraphs.append('. '.join(improved_sentences))
            
            return '\n\n'.join(improved_paragraphs)
        except Exception as e:
            logger.warning(f"Clarity improvement failed: {e}")
            return draft
    
    def _add_technical_details(self, draft: str) -> str:
        """Add technical details where needed."""
        try:
            if "## Method" in draft:
                method_section = draft.split("## Method")[1].split("## ")[0]
                if len(method_section) < 1000:  # If method section is short
                    enhanced_method = method_section + "\n\nThe detailed implementation involves specific algorithmic steps and parameter configurations that ensure optimal performance."
                    draft = draft.replace(method_section, enhanced_method)
            
            return draft
        except Exception as e:
            logger.warning(f"Technical details addition failed: {e}")
            return draft
    
    def _improve_structure(self, draft: str) -> str:
        """Improve structural elements of the draft."""
        try:
            sections = draft.split("## ")
            improved_sections = []
            
            for i, section in enumerate(sections):
                if section.strip():
                    if i > 0:
                        section = "## " + section
                        
                    # Ensure sections have proper introduction sentences
                    lines = section.split('\n')
                    if len(lines) > 1 and len(lines[1].strip()) > 0:
                        # Check if first content line is a good opening
                        first_content = lines[1].strip()
                        if not any(first_content.startswith(word) for word in ['This', 'The', 'In', 'We', 'Our']):
                            # Add a better opening sentence
                            section_name = lines[0].replace("## ", "").strip()
                            lines.insert(1, f"This section presents the {section_name.lower()} of our approach.")
                            section = '\n'.join(lines)
                    
                    improved_sections.append(section)
                else:
                    improved_sections.append(section)
            
            return '\n'.join(improved_sections)
        except Exception as e:
            logger.warning(f"Structure improvement failed: {e}")
            return draft
    
    def _extract_section_plan(self, full_plan: str, index: int) -> str:
        """Extract plan for a specific section."""
        try:
            if not full_plan:
                return f"Section {index + 1} content"
                
            pattern = re.compile(rf"^\s*[\(\[]*{index + 1}[\.\)\]]+\s*(.*?)(?=\n\s*[\(\[]*{index + 2}[\.\)\]]|$)", re.MULTILINE | re.DOTALL)
            match = pattern.search(full_plan)
            return match.group(1).strip() if match else f"Section {index + 1} content"
        except Exception as e:
            logger.warning(f"Section plan extraction failed for index {index}: {e}")
            return f"Section {index + 1} content"
    
    def _extract_feedback_points(self, state: Dict[str, Any]) -> List[str]:
        """Extract specific feedback points from review."""
        points = []
        
        try:
            fb = state.get("feedback_breadth", "")
            fd = state.get("feedback_depth", "")
            
            for feedback_str in [fb, fd]:
                if "ISSUES:" in feedback_str:
                    match = re.search(r"ISSUES:(.*)", feedback_str, re.DOTALL | re.IGNORECASE)
                    if match:
                        issues = match.group(1).strip()
                        issue_parts = re.split(r'[;\n]+', issues)
                        points.extend([part.strip() for part in issue_parts if part.strip()])
            
            return list(set(points))  # Remove duplicates
        except Exception as e:
            logger.warning(f"Feedback extraction failed: {e}")
            return []
    
    def _adjust_plan_for_consistency(self, section_plan: str, issues: List[str]) -> str:
        """Adjust section plan to address consistency issues."""
        try:
            adjusted_plan = section_plan
            
            for issue in issues:
                if "overlap" in issue.lower():
                    adjusted_plan += f"\nNote: Ensure alignment with outline requirements."
                elif "coverage" in issue.lower():
                    adjusted_plan += f"\nNote: Provide comprehensive coverage of planned topics."
            
            return adjusted_plan
        except Exception as e:
            logger.warning(f"Plan adjustment failed: {e}")
            return section_plan
    
    def _apply_final_consistency_checks(self, draft: str) -> str:
        """Apply final consistency checks and enhancements."""
        if not self.consistency_manager:
            return draft
        
        try:
            # Apply all consistency enhancements
            enhanced_draft = self._safe_apply_consistency_enhancements(draft)
            
            # Final terminology pass
            enhanced_draft = self.consistency_manager.normalize_terminology(enhanced_draft)
            
            return enhanced_draft
        except Exception as e:
            logger.warning(f"Final consistency checks failed: {e}")
            return draft
    
    def _safe_generate_consistency_report(self) -> Dict[str, Any]:
        """Safely generate consistency analysis report."""
        if not self.consistency_manager:
            return {"status": "consistency_manager_unavailable"}
        
        try:
            return self.consistency_manager.get_consistency_report()
        except Exception as e:
            logger.warning(f"Failed to generate consistency report: {e}")
            return {"error": str(e), "status": "report_generation_failed"}


def writer_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Function wrapper for compatibility."""
    try:
        agent = EnhancedWriterAgent()
        return agent.execute(state)
    except Exception as e:
        logger.error(f"Writer agent execution failed: {e}")
        return {
            "current_draft": f"Writer agent failed: {str(e)}",
            "draft_version": 0,
            "draft_history": [],
            "consistency_report": {"error": str(e)}
        }


# Export both class and function
__all__ = ['EnhancedWriterAgent', 'writer_agent']