# tools.py
"""
Optimized and enhanced search tools for literature retrieval.
This version uses the ArXiv API as a stable temporary backend.
Integrates PDF download and full-text analysis via GROBID.
"""
import config
import requests
import os
import time
import arxiv # ArXiv库
from grobid_client.grobid_client import GrobidClient
from langchain_core.tools import tool
from typing import List, Dict

def download_pdf(url: str, filepath: str, is_arxiv: bool = False) -> bool:
    """Downloads a PDF from a URL, returning True on success."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # ArXiv PDF链接通常不需要特殊的User-Agent
        response = requests.get(url, timeout=20, headers=headers if not is_arxiv else None)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(response.content)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")
        return False

@tool
def enhanced_literature_search(query: str, max_results: int = 3) -> List[Dict]:
    """
    Performs an enhanced literature search using the ArXiv API,
    downloads PDFs, and parses them into structured text using GROBID.
    """
    print(f"[Search] Starting ArXiv search for: {query[:50]}...")
    pdf_dir = os.path.join(config.OUTPUT_DIR, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    
    papers_data = []
    
    try:
        # 1. 初始化 ArXiv 搜索
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        results = list(arxiv.Client().results(search))

        if not results:
            print(f"[Search] No results found on ArXiv for query: {query}")
            return []

        # 2. 初始化 GROBID 客户端
        server_url = f"{config.GROBID_HOST}:{config.GROBID_PORT}"
        print(f"[GROBID] Initializing client for server at {server_url}")
        client = GrobidClient(grobid_server=server_url, check_server=True)

        for paper in results:
            paper_id = paper.get_short_id()
            title = paper.title
            
            # ArXiv 论文默认都是开放获取的
            pdf_url = paper.pdf_url
            
            # 3. 下载 PDF
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '_')).rstrip()
            pdf_path = os.path.join(pdf_dir, f"{paper_id}_{safe_title[:30]}.pdf")
            
            if download_pdf(pdf_url, pdf_path, is_arxiv=True):
                print(f"[Search] Downloaded PDF for paper: {title[:30]}...")
                
                # 4. 使用 GROBID 解析 PDF
                try:
                    _, status_code, response_text = client.process_pdf(
                        "processFulltextDocument", pdf_path,
                        generateIDs=False, consolidate_header=True,
                        consolidate_citations=False, include_raw_citations=False,
                        include_raw_affiliations=False, tei_coordinates=False,
                        segment_sentences=False
                    )
                    
                    if status_code == 200:
                        papers_data.append({
                            'paperId': paper_id,
                            'title': title,
                            'authors': [author.name for author in paper.authors],
                            'year': paper.published.year,
                            'abstract': paper.summary,
                            'full_text': response_text
                        })
                        print(f"[GROBID] Successfully parsed PDF for: {title[:30]}...")
                    else:
                         print(f"[GROBID] Parsing failed for {paper_id}.pdf with status {status_code}")
                except Exception as e:
                    print(f"[GROBID] An exception occurred during GROBID parsing: {e}")

    except Exception as e:
        print(f"ArXiv search process failed: {e}")
        return papers_data

    return papers_data

# 导出工具
search_tool = enhanced_literature_search
tools = [enhanced_literature_search]