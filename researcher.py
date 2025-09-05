# researcher.py
"""
Optimized researcher agent for literature search and ideation.
Uses structured prompts, enhanced search tools, and full-text analysis to build a RAG store.
"""
import time
from typing import Dict, Any, List
import config
import prompts
from base_agent import BaseAgent, AgentError
from tools import enhanced_literature_search # 替换了旧的导入
from rag_manager import RAGManager # 新增RAG管理器的导入

class ResearcherAgent(BaseAgent):
    """
    Agent responsible for research ideation, full-text literature search,
    and building the RAG vector store.
    """
    
    def __init__(self):
        super().__init__("Researcher", "research")
        self.search_tool = enhanced_literature_search # 使用新的增强搜索工具
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the enhanced research phase.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with research results including the RAG vector store path.
        """
        try:
            self.validate_state(state, ["topic"])
            
            # 修订逻辑可以后续根据RAG进行增强，这里我们聚焦于初次研究
            if self.is_revision(state):
                return self._handle_revision(state)
            else:
                return self._handle_initial_research(state)
                
        except Exception as e:
            self.log(f"Research failed: {e}", "ERROR")
            raise AgentError(f"Researcher failed: {e}")
    
    def _handle_initial_research(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initial research round with full-text analysis and RAG indexing."""
        self.log("Starting enhanced initial research with full-text analysis")
        
        topic = state["topic"]
        
        # 步骤 1: 生成创新角度
        plans = self._generate_innovations(topic)
        
        # 步骤 2: 提取关键词
        keywords = self._extract_keywords(plans)
        
        # 步骤 3: 搜索文献 (返回包含全文的字典列表)
        papers_data = self._search_literature(keywords)
        
        if not papers_data:
            self.log("No papers found or processed. Cannot proceed.", "ERROR")
            raise AgentError("Literature search yielded no usable papers.")

        # 步骤 4: 创建 RAG 索引 (向量数据库)
        rag_manager = RAGManager(papers_data)
        vector_store_path = rag_manager.create_vector_store()
        
        # 步骤 5: 保存供人类阅读的参考文献摘要
        self._save_references_summary(papers_data, topic)
        
        return {
            "innovation_plans": plans,
            "vector_store_path": vector_store_path, # 传递RAG索引的路径
            "papers_data": papers_data, # 传递论文的元数据
            "research_complete": True,
            "revision_count": 1 # 初始化修订次数
        }
    
    def _handle_revision(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle revision research based on feedback."""
        revision = self.get_revision_count(state) + 1
        self.log(f"Revision research round {revision}")
        
        # (此部分可以进一步增强，例如，使用反馈来查询现有的RAG索引以获得更精确的信息)
        query = self._create_revision_query(state)
        # 也可以选择进行一次补充搜索
        # new_papers_data = self._search_literature([query])
        
        self.log(f"Generated revision query: {query}. For now, proceeding with existing literature.")
        
        return {
            "revision_count": revision,
            "innovation_plans": state.get("innovation_plans", "")
        }
    
    def _generate_innovations(self, topic: str) -> str:
        """Generate three innovation angles."""
        result = self.make_decision(
            prompts.RESEARCH_INNOVATION_PROMPT,
            topic=self.truncate_text(topic, 100)
        )
        self.log(f"Generated innovations: {result[:100]}...")
        return result
    
    def _extract_keywords(self, plans: str) -> List[str]:
        """Extract search keywords from plans."""
        fallback_keywords = [
            "new type transformer",
            "AI integration hardware",
            "embbeding system"
        ]
        
        try:
            json_result = self.extract_json(
                prompts.RESEARCH_KEYWORDS_PROMPT,
                plans=self.truncate_text(plans, 200)
            )
            keywords = json_result.get("queries", [])
            valid_keywords = [k for k in keywords if len(k) > 5]
            
            if valid_keywords:
                self.log(f"Extracted {len(valid_keywords)} valid keywords")
                return valid_keywords[:config.MAX_SEARCH_RESULTS]
            else:
                self.log("Keywords too generic, using fallback", "WARNING")
                return fallback_keywords
                
        except Exception as e:
            self.log(f"Keyword extraction failed: {e}. Using fallback.", "WARNING")
            return fallback_keywords
    
    def _search_literature(self, queries: List[str]) -> List[dict]:
        """Search for literature using the enhanced tool."""
        all_papers = []
        # 使用集合来存储paperId，避免重复处理
        processed_paper_ids = set()

        for query in queries[:config.MAX_SEARCH_RESULTS]:
            try:
                self.log(f"Searching: {query[:50]}...")
                # self.search_tool 指向 enhanced_literature_search
                results = self.search_tool.invoke({"query": query, "max_results": 2})
                if results:
                    for paper in results:
                        if paper.get('paperId') not in processed_paper_ids:
                            all_papers.append(paper)
                            processed_paper_ids.add(paper.get('paperId'))
            except Exception as e:
                self.log(f"Search failed for query '{query}': {e}", "WARNING")
        
        self.log(f"Found and processed {len(all_papers)} unique papers.")
        self.log("Waiting for a moment to respect API rate limits...")
        time.sleep(20)
        return all_papers

    def _save_references_summary(self, papers_data: List[dict], topic: str) -> None:
        """Save a human-readable summary of references."""
        header = f"# References for: {topic}\n\n"
        summaries = []
        for paper in papers_data:
            authors = ", ".join(paper.get('authors', ['Unknown']))
            summary = (
                f"Title: {paper.get('title', 'N/A')}\n"
                f"Authors: {authors}\n"
                f"Year: {paper.get('year', 'N/A')}\n"
                f"Abstract: {str(paper.get('abstract', 'N/A'))[:300]}...\n"
            )
            summaries.append(summary)
        
        content = header + "\n---\n".join(summaries)
        self.save_file(content, config.REFERENCE_FILE)
        self.log(f"Saved {len(papers_data)} reference summaries.")

    def _create_revision_query(self, state: Dict[str, Any]) -> str:
        """Create search query from feedback."""
        topic = state.get("topic", "")
        fb = state.get("feedback_breadth", "")
        fd = state.get("feedback_depth", "")
        
        issues = []
        if "ISSUES:" in fb:
            issues.append(fb.split("ISSUES:")[1][:50])
        if "ISSUES:" in fd:
            issues.append(fd.split("ISSUES:")[1][:50])
        
        query = f"{topic[:50]} {' '.join(issues)}"[:100]
        return query

def researcher_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Function wrapper for compatibility."""
    agent = ResearcherAgent()
    return agent.execute(state)

# Export both class and function
__all__ = ['ResearcherAgent', 'researcher_agent']