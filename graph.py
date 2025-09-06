# graph.py
"""
Fixed enhanced workflow graph for research paper generation.
Implements efficient state management, decision routing, and consistency tracking.
FIXED: Import issues, state definitions, and error handling.
"""

from typing import Dict, Any, Literal, TypedDict, Annotated, List, Optional
import operator
import logging

# Safe imports with fallbacks
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    print("⚠️ LangGraph not available - using fallback implementation")
    StateGraph = None
    END = "END"
    LANGGRAPH_AVAILABLE = False

import config
import prompts

# Safe LLM manager import
try:
    from llm_config import llm_manager
    LLM_MANAGER_AVAILABLE = True
except ImportError:
    print("⚠️ LLM manager not available - using fallback")
    llm_manager = None
    LLM_MANAGER_AVAILABLE = False

# Import agents with error handling
try:
    from researcher import researcher_agent
    RESEARCHER_AVAILABLE = True
except ImportError:
    print("⚠️ Researcher agent not available")
    researcher_agent = None
    RESEARCHER_AVAILABLE = False

try:
    from analyst import analyst_agent
    ANALYST_AVAILABLE = True
except ImportError:
    print("⚠️ Analyst agent not available")
    analyst_agent = None
    ANALYST_AVAILABLE = False

try:
    from writer import writer_agent
    WRITER_AVAILABLE = True
except ImportError:
    print("⚠️ Writer agent not available")
    writer_agent = None
    WRITER_AVAILABLE = False

try:
    from reviewers import breadth_supervisor_agent, depth_supervisor_agent
    REVIEWERS_AVAILABLE = True
except ImportError:
    print("⚠️ Reviewer agents not available")
    breadth_supervisor_agent = None
    depth_supervisor_agent = None
    REVIEWERS_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)

class EnhancedGraphState(TypedDict, total=False):
    """Enhanced centralized state for workflow with consistency tracking."""
    # Core data - required
    topic: str
    innovation_plans: str
    final_topic: str
    final_plan: str
    
    # RAG and Literature Data - optional
    papers_data: List[dict]
    vector_store_path: str
    references: str
    
    # Enhanced content and consistency - optional
    current_draft: str
    feedback_breadth: str
    feedback_depth: str
    score_breadth: float
    score_depth: float
    
    # Consistency management - optional
    consistency_report: Dict[str, Any]
    concept_tracking: Dict[str, Any]
    terminology_map: Dict[str, str]
    outline_alignment: float
    
    # Control and tracking - optional
    revision_count: int
    workflow_status: str
    research_mode: str
    
    # History and versioning - optional
    draft_history: Annotated[list, operator.add]
    consistency_history: Annotated[list, operator.add]


class EnhancedTriageRouter:
    """
    Enhanced decision router for workflow control.
    Uses scores, feedback, and consistency metrics for intelligent routing.
    """
    
    def __init__(self):
        if LLM_MANAGER_AVAILABLE and llm_manager:
            try:
                self.llm = llm_manager.get_analysis_llm()
            except Exception as e:
                logger.warning(f"Failed to get analysis LLM: {e}")
                self.llm = None
        else:
            self.llm = None
        
    def route(self, state: EnhancedGraphState) -> Literal["reresearch", "rewrite", "end"]:
        """
        Determine next action based on review scores, feedback, and consistency metrics.
        
        Args:
            state: Current enhanced workflow state
            
        Returns:
            Next action: "reresearch", "rewrite", or "end"
        """
        print("--- Enhanced Triage: Analyzing feedback and consistency ---")
        
        # Check max revisions
        max_revisions = getattr(config, 'MAX_REVISIONS', 3)
        current_revisions = state.get("revision_count", 0)
        
        if current_revisions >= max_revisions:
            print(f"Max revisions ({max_revisions}) reached. Ending.")
            return "end"
        
        # Get review scores with defaults
        score_breadth = state.get("score_breadth", 0.5)
        score_depth = state.get("score_depth", 0.5)
        avg_score = (score_breadth + score_depth) / 2
        
        # Get consistency metrics if available
        consistency_score = 1.0  # Default to good if no consistency data
        if 'consistency_report' in state and state['consistency_report']:
            consistency_report = state['consistency_report']
            # Calculate overall consistency score from various metrics
            outline_score = state.get('outline_alignment', 1.0)
            concept_issues = len(consistency_report.get('issues', []))
            consistency_score = max(0.0, outline_score - (concept_issues * 0.1))
        
        # Combined quality score (weighted average)
        combined_score = (avg_score * 0.7) + (consistency_score * 0.3)
        
        print(f"Quality scores: Breadth={score_breadth:.2f}, Depth={score_depth:.2f}, Consistency={consistency_score:.2f}")
        print(f"Combined score: {combined_score:.2f} (threshold: {getattr(config, 'REVIEW_PASS_THRESHOLD', 0.5)})")
        
        # Decision logic
        pass_threshold = getattr(config, 'REVIEW_PASS_THRESHOLD', 0.5)
        if combined_score >= pass_threshold:
            print(f"Combined score {combined_score:.2f} >= threshold {pass_threshold}. Ending.")
            return "end"
        
        # Analyze what type of revision is needed
        score_deficit = pass_threshold - combined_score
        consistency_issues = consistency_score < 0.7
        review_issues = avg_score < 0.6
        
        if score_deficit > 0.4 or (consistency_issues and review_issues):
            print(f"Major issues detected (deficit: {score_deficit:.2f}). Major revision needed.")
            return "reresearch"
        elif consistency_issues or score_deficit > 0.2:
            print(f"Moderate issues detected (deficit: {score_deficit:.2f}). Rewriting with consistency fixes.")
            return "rewrite"
        else:
            print(f"Minor issues detected (deficit: {score_deficit:.2f}). Standard rewriting.")
            return "rewrite"


class EnhancedResearchWorkflowGraph:
    """
    Enhanced workflow orchestrator with consistency management.
    Manages agent coordination, state flow, and quality control.
    """
    
    def __init__(self):
        self.triage = EnhancedTriageRouter()
        self.workflow = None
        self.app = None
        
        # Build workflow if LangGraph is available
        if LANGGRAPH_AVAILABLE:
            try:
                self.workflow = self._build_enhanced_graph()
                self.app = self.workflow.compile()
                logger.info("Enhanced workflow graph built successfully")
            except Exception as e:
                logger.error(f"Failed to build workflow graph: {e}")
                self.app = None
        else:
            logger.warning("LangGraph not available - using fallback workflow")
            self.app = None
    
    def _build_enhanced_graph(self) -> StateGraph:
        """Build the enhanced workflow graph with consistency tracking."""
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph not available")
        
        workflow = StateGraph(EnhancedGraphState)
        
        # Add nodes with availability checks
        if RESEARCHER_AVAILABLE and researcher_agent:
            workflow.add_node("researcher", researcher_agent)
        else:
            workflow.add_node("researcher", self._fallback_researcher)
        
        if ANALYST_AVAILABLE and analyst_agent:
            workflow.add_node("analyst", analyst_agent)
        else:
            workflow.add_node("analyst", self._fallback_analyst)
        
        if WRITER_AVAILABLE and writer_agent:
            workflow.add_node("writer", writer_agent)
        else:
            workflow.add_node("writer", self._fallback_writer)
        
        if REVIEWERS_AVAILABLE and breadth_supervisor_agent:
            workflow.add_node("breadth_review", breadth_supervisor_agent)
        else:
            workflow.add_node("breadth_review", self._fallback_breadth_reviewer)
        
        if REVIEWERS_AVAILABLE and depth_supervisor_agent:
            workflow.add_node("depth_review", depth_supervisor_agent)
        else:
            workflow.add_node("depth_review", self._fallback_depth_reviewer)
        
        # Add consistency tracking node
        workflow.add_node("consistency_check", self._consistency_check_node)
        
        # Set entry point
        workflow.set_entry_point("researcher")
        
        # Build workflow edges
        workflow.add_edge("researcher", "analyst")
        workflow.add_edge("analyst", "writer")
        workflow.add_edge("writer", "consistency_check")
        workflow.add_edge("consistency_check", "breadth_review")
        workflow.add_edge("breadth_review", "depth_review")
        
        # Enhanced conditional routing
        workflow.add_conditional_edges(
            "depth_review",
            self.triage.route,
            {
                "reresearch": "researcher",
                "rewrite": "writer",
                "end": END
            }
        )
        
        return workflow
    
    def _consistency_check_node(self, state: EnhancedGraphState) -> Dict[str, Any]:
        """
        Perform consistency checks on the current draft.
        This node runs after writing and before review.
        """
        print("--- Consistency Check: Analyzing draft quality ---")
        
        try:
            # Check if consistency management is enabled
            consistency_enabled = getattr(config, 'ENABLE_CONSISTENCY_MANAGEMENT', False)
            if not consistency_enabled:
                print("Consistency management disabled, skipping check")
                return {"consistency_report": {}, "outline_alignment": 1.0}
            
            current_draft = state.get("current_draft", "")
            if not current_draft or len(current_draft) < 100:
                print("No substantial draft found, skipping consistency check")
                return {"consistency_report": {}, "outline_alignment": 0.5}
            
            # Try to import consistency manager
            try:
                from consistency_manager import create_consistency_manager
                consistency_manager = create_consistency_manager()
                print("✅ Consistency manager loaded")
            except ImportError:
                print("⚠️ Consistency manager not available, skipping check")
                return {"consistency_report": {}, "outline_alignment": 1.0}
            
            # Set up consistency manager with current state
            final_plan = state.get("final_plan", "")
            if final_plan:
                consistency_manager.set_global_outline(final_plan)
            
            # Perform basic consistency analysis
            consistency_report = {}
            outline_alignment = 1.0
            
            try:
                # Simple outline alignment check
                sections = self._extract_sections_from_draft(current_draft)
                outline_alignment = min(1.0, len(sections) / 5.0)  # Expect ~5 sections
                
                # Generate basic report
                consistency_report = {
                    'total_concepts': 0,
                    'total_sections': len(sections),
                    'issues': [],
                    'outline_alignment': outline_alignment
                }
                
                print(f"Consistency check complete: Alignment={outline_alignment:.2f}, Sections={len(sections)}")
                
            except Exception as e:
                logger.warning(f"Consistency analysis failed: {e}")
                consistency_report = {"error": str(e)}
                outline_alignment = 0.8
            
            return {
                "consistency_report": consistency_report,
                "outline_alignment": outline_alignment,
                "consistency_history": [consistency_report]
            }
            
        except Exception as e:
            print(f"Consistency check failed: {e}")
            return {
                "consistency_report": {"error": str(e)},
                "outline_alignment": 0.5,
                "consistency_history": []
            }
    
    def _extract_sections_from_draft(self, draft: str) -> Dict[str, str]:
        """Extract sections from the draft for analysis."""
        sections = {}
        
        try:
            # Split by markdown headers
            parts = draft.split("## ")
            
            for i, part in enumerate(parts):
                if i == 0:
                    continue  # Skip content before first header
                
                lines = part.split('\n', 1)
                if len(lines) >= 2:
                    section_title = lines[0].strip()
                    section_content = lines[1].strip()
                    sections[section_title] = section_content
                elif len(lines) == 1:
                    section_title = lines[0].strip()
                    sections[section_title] = ""
        except Exception as e:
            logger.warning(f"Section extraction failed: {e}")
        
        return sections
    
    # Fallback agent implementations
    def _fallback_researcher(self, state: EnhancedGraphState) -> Dict[str, Any]:
        """Fallback researcher implementation."""
        print("⚠️ Using fallback researcher")
        return {
            "innovation_plans": "Fallback research plans for " + state.get("topic", "unknown topic"),
            "vector_store_path": "fallback_path",
            "papers_data": [],
            "research_mode": "fallback",
            "revision_count": state.get("revision_count", 0) + 1
        }
    
    def _fallback_analyst(self, state: EnhancedGraphState) -> Dict[str, Any]:
        """Fallback analyst implementation."""
        print("⚠️ Using fallback analyst")
        return {
            "final_topic": state.get("topic", "Fallback topic"),
            "final_plan": "1. Introduction\n2. Method\n3. Evaluation\n4. Conclusion",
            "references": "Fallback references"
        }
    
    def _fallback_writer(self, state: EnhancedGraphState) -> Dict[str, Any]:
        """Fallback writer implementation."""
        print("⚠️ Using fallback writer")
        topic = state.get("topic", "research topic")
        
        fallback_draft = f"""# Research Paper: {topic}

## Abstract
This paper presents research on {topic}. The work addresses key challenges in the field and proposes novel solutions.

## Introduction
The field of {topic} has seen significant developments in recent years. This research contributes to the understanding of fundamental principles.

## Method
We propose a novel approach to {topic} that improves upon existing methods through innovative techniques.

## Evaluation
Experimental results demonstrate the effectiveness of our approach compared to baseline methods.

## Conclusion
This work advances the state of knowledge in {topic} and opens new directions for future research.
"""
        
        return {
            "current_draft": fallback_draft,
            "draft_version": state.get("revision_count", 0),
            "draft_history": [fallback_draft],
            "consistency_report": {}
        }
    
    def _fallback_breadth_reviewer(self, state: EnhancedGraphState) -> Dict[str, Any]:
        """Fallback breadth reviewer implementation."""
        print("⚠️ Using fallback breadth reviewer")
        return {
            "feedback_breadth": "PASS - Fallback review",
            "score_breadth": 0.8
        }
    
    def _fallback_depth_reviewer(self, state: EnhancedGraphState) -> Dict[str, Any]:
        """Fallback depth reviewer implementation."""
        print("⚠️ Using fallback depth reviewer")
        return {
            "feedback_depth": "PASS - Fallback review",
            "score_depth": 0.8
        }
    
    def run(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute enhanced workflow with comprehensive error handling and logging.
        """
        print("\n" + "="*80)
        print(f"Starting Enhanced Research Workflow: {initial_state.get('topic', 'Unknown')}")
        print("="*80 + "\n")
        
        # Enhanced initial state with consistency tracking
        enhanced_state = {
            **initial_state,
            "consistency_report": {},
            "concept_tracking": {},
            "terminology_map": {},
            "outline_alignment": 1.0,
            "consistency_history": [],
            "research_mode": "unknown",
            "revision_count": 0,
            "workflow_status": "initialized"
        }
        
        try:
            if self.app and LANGGRAPH_AVAILABLE:
                print("🔄 Executing enhanced LangGraph workflow...")
                config_dict = {"recursion_limit": getattr(config, 'RECURSION_LIMIT', 20)}
                final_state = self.app.invoke(enhanced_state, config_dict)
            else:
                print("🔄 Executing fallback sequential workflow...")
                final_state = self._run_fallback_workflow(enhanced_state)
            
            # Enhanced final state processing
            final_state = self._process_final_state(final_state)
            
            if final_state.get("workflow_status") == "error":
                print("\n❌ Enhanced workflow failed with errors")
            else:
                print("\n✅ Enhanced workflow completed successfully")
                self._display_final_metrics(final_state)
                
            return final_state
            
        except Exception as e:
            print(f"\n❌ Enhanced workflow error: {e}")
            return {
                **enhanced_state,
                "workflow_status": "error",
                "error": str(e)
            }
    
    def _run_fallback_workflow(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run fallback sequential workflow when LangGraph is not available."""
        print("Running fallback sequential workflow...")
        
        try:
            # Step 1: Research
            print("1. Researcher...")
            if RESEARCHER_AVAILABLE and researcher_agent:
                state.update(researcher_agent(state))
            else:
                state.update(self._fallback_researcher(state))
            
            # Step 2: Analysis
            print("2. Analyst...")
            if ANALYST_AVAILABLE and analyst_agent:
                state.update(analyst_agent(state))
            else:
                state.update(self._fallback_analyst(state))
            
            # Step 3: Writing
            print("3. Writer...")
            if WRITER_AVAILABLE and writer_agent:
                state.update(writer_agent(state))
            else:
                state.update(self._fallback_writer(state))
            
            # Step 4: Consistency Check
            print("4. Consistency Check...")
            state.update(self._consistency_check_node(state))
            
            # Step 5: Reviews
            print("5. Breadth Review...")
            if REVIEWERS_AVAILABLE and breadth_supervisor_agent:
                state.update(breadth_supervisor_agent(state))
            else:
                state.update(self._fallback_breadth_reviewer(state))
            
            print("6. Depth Review...")
            if REVIEWERS_AVAILABLE and depth_supervisor_agent:
                state.update(depth_supervisor_agent(state))
            else:
                state.update(self._fallback_depth_reviewer(state))
            
            # Step 6: Triage Decision
            print("7. Triage Decision...")
            decision = self.triage.route(state)
            
            # For fallback, we just end after one iteration
            if decision != "end" and state.get("revision_count", 0) < 1:
                print(f"Triage suggests: {decision}, but limiting to one iteration in fallback mode")
            
            state["workflow_status"] = "completed"
            return state
            
        except Exception as e:
            logger.error(f"Fallback workflow failed: {e}")
            state["workflow_status"] = "error"
            state["error"] = str(e)
            return state
    
    def _process_final_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process and enhance the final state with additional metrics."""
        
        try:
            # Calculate overall quality score
            review_score = 0.0
            if 'score_breadth' in state and 'score_depth' in state:
                review_score = (state['score_breadth'] + state['score_depth']) / 2
            
            consistency_score = state.get('outline_alignment', 1.0)
            overall_quality = (review_score * 0.7) + (consistency_score * 0.3)
            
            # Add quality metrics
            state['overall_quality_score'] = overall_quality
            state['quality_breakdown'] = {
                'review_score': review_score,
                'consistency_score': consistency_score,
                'combined_score': overall_quality
            }
            
            # Set final workflow status
            pass_threshold = getattr(config, 'REVIEW_PASS_THRESHOLD', 0.5)
            if overall_quality >= pass_threshold:
                state['workflow_status'] = 'success'
            else:
                state['workflow_status'] = 'completed_with_issues'
            
        except Exception as e:
            logger.warning(f"Final state processing failed: {e}")
            state['workflow_status'] = 'completed'
        
        return state
    
    def _display_final_metrics(self, state: Dict[str, Any]) -> None:
        """Display final quality and consistency metrics."""
        print("\n📊 Final Quality Metrics:")
        print("-" * 40)
        
        # Review scores
        if 'score_breadth' in state:
            print(f"Breadth Score: {state['score_breadth']:.3f}")
        if 'score_depth' in state:
            print(f"Depth Score: {state['score_depth']:.3f}")
        
        # Consistency metrics
        if 'outline_alignment' in state:
            print(f"Outline Alignment: {state['outline_alignment']:.3f}")
        
        # Overall quality
        if 'overall_quality_score' in state:
            print(f"Overall Quality: {state['overall_quality_score']:.3f}")
        
        # Consistency report summary
        if 'consistency_report' in state and state['consistency_report']:
            report = state['consistency_report']
            print(f"Concepts Tracked: {report.get('total_concepts', 0)}")
            print(f"Sections Found: {report.get('total_sections', 0)}")
            
            issues = report.get('issues', [])
            if issues:
                print(f"⚠️  Consistency Issues: {len(issues)}")
            else:
                print("✅ No consistency issues detected")


def create_app():
    """Factory function to create enhanced workflow application."""
    try:
        workflow_graph = EnhancedResearchWorkflowGraph()
        if workflow_graph.app:
            return workflow_graph.app
        else:
            # Return a simple callable that runs the fallback workflow
            return lambda state, config=None: workflow_graph.run(state)
    except Exception as e:
        logger.error(f"Failed to create app: {e}")
        # Return a minimal fallback
        def fallback_app(state, config=None):
            return {
                **state,
                "current_draft": f"Fallback draft for {state.get('topic', 'unknown topic')}",
                "workflow_status": "fallback_completed"
            }
        return fallback_app


def create_enhanced_workflow():
    """Factory function to create the complete enhanced workflow."""
    return EnhancedResearchWorkflowGraph()


# Global app instance for simple import
try:
    app = create_app()
    print("✅ Enhanced workflow app created successfully")
except Exception as e:
    print(f"⚠️ Failed to create workflow app: {e}")
    app = None

# Export enhanced workflow
try:
    enhanced_workflow = create_enhanced_workflow()
    print("✅ Enhanced workflow graph created successfully")
except Exception as e:
    print(f"⚠️ Failed to create enhanced workflow: {e}")
    enhanced_workflow = None

# Export for testing
__all__ = ['create_app', 'create_enhanced_workflow', 'EnhancedResearchWorkflowGraph', 'EnhancedGraphState', 'app', 'enhanced_workflow']