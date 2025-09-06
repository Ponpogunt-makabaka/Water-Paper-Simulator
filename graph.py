# graph.py
"""
Optimized workflow graph for research paper generation.
Implements efficient state management and decision routing.
"""

from typing import Dict, Any, Literal, TypedDict, Annotated, List
import operator
from langgraph.graph import StateGraph, END

import config
import prompts
from llm_config import llm_manager

# Import agents
from researcher import researcher_agent
from analyst import analyst_agent
from writer import writer_agent
from reviewers import breadth_supervisor_agent, depth_supervisor_agent


class GraphState(TypedDict):
    """Centralized state for workflow."""
    # Core data
    topic: str
    innovation_plans: str
    final_topic: str
    final_plan: str
    
    # RAG and Literature Data
    papers_data: List[dict] # 新增: 存储论文的详细元数据和解析后的文本
    vector_store_path: str # 新增: RAG向量数据库的路径
    references: str # 保留，用于存储给人类阅读的摘要
    
    # Draft and Feedback
    current_draft: str
    feedback_breadth: str
    feedback_depth: str
    score_breadth: float
    score_depth: float
    
    # Control
    revision_count: int
    workflow_status: str
    
    # History
    draft_history: Annotated[list, operator.add]


class TriageRouter:
    """
    Decision router for workflow control.
    Uses scores and feedback for intelligent routing.
    """
    
    def __init__(self):
        self.llm = llm_manager.get_analysis_llm()
        
    def route(self, state: GraphState) -> Literal["reresearch", "rewrite", "end"]:
        """
        Determine next action based on review scores and feedback.
        """
        print("--- Triage: Analyzing feedback ---")
        
        if state.get("revision_count", 0) >= config.MAX_REVISIONS:
            print(f"Max revisions ({config.MAX_REVISIONS}) reached. Ending.")
            return "end"
        
        score_breadth = state.get("score_breadth", 0.5)
        score_depth = state.get("score_depth", 0.5)
        avg_score = (score_breadth + score_depth) / 2
        
        print(f"Review scores: Breadth={score_breadth:.2f}, Depth={score_depth:.2f}, Avg={avg_score:.2f}")
        
        if avg_score >= config.REVIEW_PASS_THRESHOLD:
            print(f"Average score {avg_score:.2f} >= threshold {config.REVIEW_PASS_THRESHOLD}. Ending.")
            return "end"
        
        score_deficit = config.REVIEW_PASS_THRESHOLD - avg_score
        
        if score_deficit > 0.3:
            print(f"Large score deficit ({score_deficit:.2f}). Major revision needed.")
            return "reresearch"
        else:
            print(f"Moderate to small score deficit ({score_deficit:.2f}). Rewriting.")
            return "rewrite"


class ResearchWorkflowGraph:
    """
    Main workflow orchestrator.
    Manages agent coordination and state flow.
    """
    
    def __init__(self):
        self.triage = TriageRouter()
        self.workflow = self._build_graph()
        self.app = self.workflow.compile()
        
    def _build_graph(self) -> StateGraph:
        """Build the workflow graph."""
        workflow = StateGraph(GraphState)
        
        workflow.add_node("researcher", researcher_agent)
        workflow.add_node("analyst", analyst_agent)
        workflow.add_node("writer", writer_agent)
        workflow.add_node("breadth_review", breadth_supervisor_agent)
        workflow.add_node("depth_review", depth_supervisor_agent)
        
        workflow.set_entry_point("researcher")
        workflow.add_edge("researcher", "analyst")
        workflow.add_edge("analyst", "writer")
        workflow.add_edge("writer", "breadth_review")
        workflow.add_edge("breadth_review", "depth_review")
        
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
    
    def run(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute workflow with initial state.
        """
        print("\n" + "="*60)
        print(f"Starting Research: {initial_state['topic']}")
        print("="*60 + "\n")
        
        config_dict = {"recursion_limit": config.RECURSION_LIMIT}
        
        try:
            final_state = self.app.invoke(initial_state, config_dict)
            
            if final_state.get("workflow_status") == "error":
                print("\n❌ Workflow failed with errors")
            else:
                print("\n✅ Workflow completed successfully")
                
            return final_state
            
        except Exception as e:
            print(f"\n❌ Workflow error: {e}")
            return {"workflow_status": "error", "error": str(e)}

def create_app():
    """Factory function to create workflow."""
    return ResearchWorkflowGraph().app

# Global app instance for simple import
app = create_app()