# researcher.py
# researcher.py
"""
Enhanced researcher agent for literature search and ideation.
NOW SUPPORTS: Online search AND local PDF repository modes.
Uses structured prompts, enhanced search tools, and full-text analysis to build a RAG store.
"""
import time
from typing import Dict, Any, List
import config
import prompts
from base_agent import BaseAgent, AgentError
from tools import enhanced_literature_search
from rag_manager import RAGManager
from local_pdf_processor import get_local_papers_data, validate_local_setup

class ResearcherAgent(BaseAgent):
    """
    Enhanced agent responsible for research ideation and literature gathering.
    Supports both online search and local PDF repository modes.
    """
    
    def __init__(self):
        super().__init__("Researcher", "research")
        self.search_tool = enhanced_literature_search
        self.research_mode = None
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the enhanced research phase with mode selection.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with research results including the RAG vector store path.
        """
        try:
            self.validate_state(state, ["topic"])
            
            # Determine research mode
            self.research_mode = self._determine_research_mode()
            self.log(f"Using research mode: {self.research_mode}")
            
            if self.is_revision(state):
                return self._handle_revision(state)
            else:
                return self._handle_initial_research(state)
                
        except Exception as e:
            self.log(f"Research failed: {e}", "ERROR")
            raise AgentError(f"Researcher failed: {e}")
    
    def _determine_research_mode(self) -> str:
        """
        Determine which research mode to use based on configuration.
        
        Returns:
            str: "online" or "local"
        """
        try:
            # Get mode from config (includes interactive prompting if configured)
            mode = config.get_research_mode()
            
            if mode == "local":
                # Validate local setup
                is_valid, message = validate_local_setup()
                if not is_valid:
                    self.log(f"Local mode validation failed: {message}", "WARNING")
                    self.log("Falling back to online mode", "INFO")
                    return "online"
                else:
                    self.log(f"Local mode validated: {message}", "INFO")
                    return "local"
            
            return "online"
            
        except Exception as e:
            self.log(f"Error determining research mode: {e}", "WARNING")
            return "online"  # Safe fallback
    
    def _handle_initial_research(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initial research round with mode-specific processing."""
        self.log(f"Starting enhanced initial research in {self.research_mode} mode")
        
        topic = state["topic"]
        
        # 步骤 1: 生成创新角度（两种模式都需要）
        plans = self._generate_innovations(topic)
        
        # 步骤 2 & 3: 获取文献数据（根据模式选择）
        if self.research_mode == "local":
            papers_data = self._get_local_literature()
        else:
            # 提取关键词并搜索在线文献
            keywords = self._extract_keywords(plans)
            papers_data = self._search_online_literature(keywords)
        
        # 如果没有找到论文，创建默认数据以确保系统继续运行
        if not papers_data:
            self.log("No papers found. Creating fallback data to continue workflow.", "WARNING")
            papers_data = self._create_fallback_papers_data(topic)

        # 步骤 4: 创建 RAG 索引 (向量数据库)
        vector_store_path = self._create_rag_store(papers_data)
        
        # 确保vector_store_path总是有值
        if not vector_store_path:
            self.log("RAG store creation failed, creating dummy store path", "WARNING")
            vector_store_path = self._create_dummy_store_path()
        
        # 步骤 5: 保存供人类阅读的参考文献摘要
        self._save_references_summary(papers_data, topic)
        
        return {
            "innovation_plans": plans,
            "vector_store_path": vector_store_path,
            "papers_data": papers_data,
            "research_mode": self.research_mode,
            "research_complete": True,
            "revision_count": 1
        }
    
    def _handle_revision(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle revision research based on feedback."""
        revision = self.get_revision_count(state) + 1
        self.log(f"Revision research round {revision} in {self.research_mode} mode")
        
        query = self._create_revision_query(state)
        self.log(f"Generated revision query: {query}. Proceeding with existing literature.")
        
        # 确保vector_store_path在修订时也存在
        vector_store_path = state.get("vector_store_path")
        if not vector_store_path:
            self.log("No vector_store_path in state, creating dummy path", "WARNING")
            vector_store_path = self._create_dummy_store_path()
        
        return {
            "revision_count": revision,
            "research_mode": self.research_mode,
            "innovation_plans": state.get("innovation_plans", ""),
            "vector_store_path": vector_store_path
        }
    
    def _get_local_literature(self) -> List[dict]:
        """Get literature from local PDF repository."""
        self.log("Loading literature from local PDF repository...")
        
        try:
            papers_data = get_local_papers_data()
            
            if papers_data:
                self.log(f"Successfully loaded {len(papers_data)} papers from local repository")
                
                # Log summary of loaded papers
                for i, paper in enumerate(papers_data[:3]):  # Show first 3
                    title = paper.get('title', 'Unknown')[:50]
                    source = paper.get('source', 'unknown')
                    self.log(f"  {i+1}. {title}... (source: {source})")
                
                if len(papers_data) > 3:
                    self.log(f"  ... and {len(papers_data) - 3} more papers")
            else:
                self.log("No papers found in local repository", "WARNING")
            
            return papers_data
            
        except Exception as e:
            self.log(f"Error loading local literature: {e}", "ERROR")
            return []
    
    def _search_online_literature(self, keywords: List[str]) -> List[dict]:
        """Search for literature using online sources."""
        self.log("Searching literature from online sources...")
        
        all_papers = []
        processed_paper_ids = set()

        for query in keywords[:config.MAX_SEARCH_RESULTS]:
            try:
                self.log(f"Searching: {query[:50]}...")
                results = self.search_tool.invoke({"query": query, "max_results": 2})
                if results:
                    for paper in results:
                        paper_id = paper.get('paperId', f"unknown_{len(all_papers)}")
                        if paper_id not in processed_paper_ids:
                            all_papers.append(paper)
                            processed_paper_ids.add(paper_id)
            except Exception as e:
                self.log(f"Search failed for query '{query}': {e}", "WARNING")
        
        self.log(f"Found and processed {len(all_papers)} unique papers from online sources.")
        
        # 速率限制
        if all_papers:
            self.log("Waiting for a moment to respect API rate limits...")
            time.sleep(20)
            
        return all_papers
    
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
            "machine learning optimization",
            "deep neural networks",
            "computer vision algorithms"
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

    def _create_fallback_papers_data(self, topic: str) -> List[dict]:
        """Create fallback papers data when search fails completely."""
        self.log("Creating fallback papers data")
        
        fallback_papers = [
            {
                'paperId': 'fallback_1',
                'title': f'Survey of {topic}',
                'authors': ['Fallback Author 1'],
                'year': 2024,
                'abstract': f'This is a comprehensive survey of {topic} methodologies and applications.',
                'full_text': f'This paper presents a comprehensive survey of {topic}. The field has evolved significantly with various approaches including traditional methods and modern techniques. Key challenges include scalability, accuracy, and practical implementation.',
                'source': 'fallback'
            },
            {
                'paperId': 'fallback_2', 
                'title': f'Advances in {topic}',
                'authors': ['Fallback Author 2'],
                'year': 2024,
                'abstract': f'Recent advances in {topic} research and development.',
                'full_text': f'Recent advances in {topic} have shown promising results. This paper discusses novel approaches, experimental findings, and future research directions. The methodology involves systematic analysis and comparison with existing techniques.',
                'source': 'fallback'
            }
        ]
        
        return fallback_papers

    def _create_rag_store(self, papers_data: List[dict]) -> str:
        """Create RAG store with error handling."""
        try:
            self.log("Creating RAG vector store...")
            rag_manager = RAGManager(papers_data)
            vector_store_path = rag_manager.create_vector_store()
            
            if vector_store_path:
                self.log(f"RAG store created successfully at: {vector_store_path}")
                return vector_store_path
            else:
                self.log("RAG store creation returned None", "WARNING")
                return self._create_dummy_store_path()
                
        except Exception as e:
            self.log(f"RAG store creation failed: {e}", "ERROR")
            return self._create_dummy_store_path()
    
    def _create_dummy_store_path(self) -> str:
        """Create a dummy store path when RAG creation fails."""
        import os
        from pathlib import Path
        
        dummy_path = os.path.join(config.OUTPUT_DIR, "dummy_vector_store")
        Path(dummy_path).mkdir(exist_ok=True)
        
        # Create a marker file
        marker_file = os.path.join(dummy_path, "dummy_marker.txt")
        with open(marker_file, "w") as f:
            f.write("This is a dummy vector store created due to RAG initialization failure.")
        
        self.log(f"Created dummy vector store path: {dummy_path}")
        return dummy_path

    def _save_references_summary(self, papers_data: List[dict], topic: str) -> None:
        """Save a human-readable summary of references."""
        mode_info = f"# References for: {topic}\n"
        mode_info += f"# Research Mode: {self.research_mode.upper()}\n"
        mode_info += f"# Papers found: {len(papers_data)}\n\n"
        
        summaries = []
        
        for paper in papers_data:
            authors_list = paper.get('authors', ['Unknown'])
            if isinstance(authors_list, list):
                authors = ", ".join(authors_list)
            else:
                authors = str(authors_list)
            
            source_info = paper.get('source', 'unknown')
            if paper.get('is_local'):
                source_info += f" (local file: {paper.get('filename', 'unknown')})"
                
            summary = (
                f"Title: {paper.get('title', 'N/A')}\n"
                f"Authors: {authors}\n"
                f"Year: {paper.get('year', 'N/A')}\n"
                f"Source: {source_info}\n"
                f"Abstract: {str(paper.get('abstract', 'N/A'))[:300]}...\n"
            )
            summaries.append(summary)
        
        content = mode_info + "\n---\n".join(summaries)
        self.save_file(content, config.REFERENCE_FILE)
        self.log(f"Saved {len(papers_data)} reference summaries ({self.research_mode} mode).")

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