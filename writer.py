# writer.py
"""
Optimized writer agent for paper drafting and revision.
Uses a RAG-based approach for high-quality, context-aware content generation.
"""

from typing import Dict, Any, List
import re
from langchain_core.output_parsers import StrOutputParser
import config
import prompts
from base_agent import BaseAgent, AgentError
from rag_manager import RAGManager # 新增RAG管理器的导入

class WriterAgent(BaseAgent):
    """
    Agent responsible for writing and revising papers using a RAG pipeline
    to ensure content is grounded in the sourced literature.
    """
    
    def __init__(self):
        super().__init__("Writer", "writing")
        self.sections = self._init_sections()
        
    def _init_sections(self) -> List[Dict[str, str]]:
        """Initialize paper sections configuration."""
        return [
            {"title": "Abstract", "style": "Concise", "focus_points": "Problem, method, results, impact"},
            {"title": "Introduction", "style": "Engaging", "focus_points": "Context, gap, contribution"},
            {"title": "Literature Review", "style": "Critical", "focus_points": "Prior work, limitations"},
            {"title": "Method", "style": "Technical", "focus_points": "Algorithm, architecture, math"},
            {"title": "Evaluation", "style": "Analytical", "focus_points": "Setup, metrics, analysis"},
            {"title": "Conclusion", "style": "Reflective", "focus_points": "Summary, limitations, future"}
        ]
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute writing phase, either drafting or revising.
        """
        try:
            self.validate_state(state, ["topic", "final_plan", "vector_store_path"])
            
            if self.is_revision(state):
                draft = self._revise_draft(state)
            else:
                draft = self._write_initial_draft(state)
            
            version = self.get_revision_count(state)
            filename = f"{config.DRAFT_FILE_PREFIX}{version}.txt"
            self.save_file(draft, filename)
            
            return {
                "current_draft": draft,
                "draft_version": version,
                "draft_history": draft # 将当前稿件添加到历史记录
            }
            
        except Exception as e:
            self.log(f"Writing failed: {e}", "ERROR")
            raise AgentError(f"Writer failed: {e}")
    
    def _write_initial_draft(self, state: Dict[str, Any]) -> str:
        """Write initial paper draft using the RAG-enabled writer."""
        self.log("Writing initial draft with RAG")
        
        topic = state["topic"]
        plan = state["final_plan"]
        vector_store_path = state["vector_store_path"]
        
        sections_content = []
        for i, section_config in enumerate(self.sections):
            self.log(f"Writing section {i+1}/{len(self.sections)}: {section_config['title']}")
            
            content = self._write_section(
                topic=topic,
                section_title=section_config["title"],
                section_plan=self._extract_section_plan(plan, i),
                vector_store_path=vector_store_path
            )
            sections_content.append(f"## {section_config['title']}\n\n{content}")
        
        draft = "\n\n---\n\n".join(sections_content)
        self.log("Initial draft complete")
        return draft
    
    def _revise_draft(self, state: Dict[str, Any]) -> str:
        """Revise draft based on feedback using RAG for context."""
        self.log("Revising draft with RAG context")
        
        draft = state.get("current_draft", "")
        if not draft or len(draft) < 100:
            self.log("No valid draft to revise, writing new draft", "WARNING")
            return self._write_initial_draft(state)
        
        feedback_points = self._extract_feedback_points(state)
        if not feedback_points:
            self.log("No feedback to address, returning original draft")
            return draft
            
        # 增强修订提示，加入RAG
        rag_query = f"Technical details or clarifications for '{state['topic']}' based on feedback: {' '.join(feedback_points)}"
        context = RAGManager.query_vector_store(state["vector_store_path"], rag_query)

        revision_prompt = f"""Revise this academic paper draft to address these specific issues, using the provided context for accuracy.

Issues to fix:
{chr(10).join(f'- {point}' for point in feedback_points[:3])}

Context from literature for this revision:
--- CONTEXT ---
{self.truncate_text(context, 2000)}
--- END CONTEXT ---

Draft to revise:
{self.truncate_text(draft, 3000)}

Output the complete revised draft:"""
        
        chain = self.create_chain(revision_prompt)
        revised = chain.invoke({})
        
        if len(revised) < len(draft) * 0.7:
            self.log("Revision seems too short, returning original draft", "WARNING")
            return draft
        
        self.log("Applied revision successfully")
        return revised
    
    def _write_section(self, **kwargs) -> str:
        """Write a single paper section using RAG."""
        section_title = kwargs["section_title"]
        topic = kwargs["topic"]
        section_plan = kwargs.get("section_plan", "")
        vector_store_path = kwargs["vector_store_path"]
        
        # 1. 为RAG制定查询
        rag_query = f"Detailed information, methods, or findings for the '{section_title}' section of a paper on '{topic}'. The plan is: {section_plan}"
        
        # 2. 从向量数据库检索上下文
        context = RAGManager.query_vector_store(vector_store_path, rag_query)
        
        # 3. 创建RAG增强的提示
        rag_prompt = prompts.WRITER_SECTION_PROMPT.format(
            section_title=section_title,
            topic=self.truncate_text(topic, 100),
            section_plan=self.truncate_text(section_plan, 200),
            context=self.truncate_text(context, 3500), # 将上下文注入提示
            max_length=config.SECTION_LENGTHS.get(section_title, config.MAX_SECTION_LENGTH),
            # 以下为旧版参数，可按需保留或修改prompt模板
            style=kwargs.get("style", "Academic"),
            focus_points=kwargs.get("focus_points", "Clarity and accuracy")
        )

        # 4. 使用LLM生成章节
        chain = self.create_chain(rag_prompt)
        result = chain.invoke({})
        
        return self.truncate_text(result, config.SECTION_LENGTHS.get(section_title, config.MAX_SECTION_LENGTH))
    
    def _extract_section_plan(self, full_plan: str, index: int) -> str:
        """Extract plan for a specific section."""
        try:
            pattern = re.compile(rf"^\s*[\(\[]*{index + 1}[\.\)\]]+\s*(.*?)(?=\n\s*[\(\[]*{index + 2}[\.\)\]]|$)", re.MULTILINE | re.DOTALL)
            match = pattern.search(full_plan)
            return match.group(1).strip() if match else ""
        except Exception:
            return ""
    
    def _extract_feedback_points(self, state: Dict[str, Any]) -> List[str]:
        """Extract specific feedback points from review."""
        points = []
        fb = state.get("feedback_breadth", "")
        fd = state.get("feedback_depth", "")
        
        for feedback_str in [fb, fd]:
            if "ISSUES:" in feedback_str:
                match = re.search(r"ISSUES:(.*)", feedback_str, re.DOTALL | re.IGNORECASE)
                if match:
                    issues = match.group(1).strip()
                    issue_parts = re.split(r'[;\n]+', issues)
                    points.extend([part.strip() for part in issue_parts if part.strip()])
        return list(set(points)) # 返回去重后的问题点

def writer_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Function wrapper for compatibility."""
    agent = WriterAgent()
    return agent.execute(state)

# Export both class and function
__all__ = ['WriterAgent', 'writer_agent']