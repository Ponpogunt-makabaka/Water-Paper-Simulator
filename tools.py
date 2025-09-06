# tools.py
"""
Enhanced and optimized search tools for literature retrieval.
This version uses the ArXiv API with comprehensive error handling,
PDF download capabilities, and full-text analysis via GROBID.
Includes fallback mechanisms for robust operation.
"""

import config
import requests
import os
import time
import arxiv
from grobid_client.grobid_client import GrobidClient
from langchain_core.tools import tool
from typing import List, Dict, Optional
import logging

# Setup logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_pdf(url: str, filepath: str, is_arxiv: bool = False, max_retries: int = 3) -> bool:
    """
    Downloads a PDF from a URL with retry logic and better error handling.
    
    Args:
        url: URL of the PDF to download
        filepath: Local path to save the PDF
        is_arxiv: Whether this is an ArXiv PDF (affects headers)
        max_retries: Maximum number of retry attempts
        
    Returns:
        True if download successful, False otherwise
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for attempt in range(max_retries):
        try:
            # ArXiv PDF links usually don't need special User-Agent
            response = requests.get(
                url, 
                timeout=30, 
                headers=headers if not is_arxiv else None,
                stream=True  # Use streaming for large files
            )
            response.raise_for_status()
            
            # Check if response is actually a PDF
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and is_arxiv:
                logger.warning(f"Response may not be PDF (content-type: {content_type})")
            
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Verify file was created and has content
            if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:  # At least 1KB
                logger.info(f"Successfully downloaded PDF: {os.path.basename(filepath)}")
                return True
            else:
                logger.warning(f"Downloaded file is too small or empty: {filepath}")
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Download attempt {attempt + 1} failed for {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            
        except Exception as e:
            logger.error(f"Unexpected error downloading {url}: {e}")
            break
    
    return False

def initialize_grobid_client() -> Optional[GrobidClient]:
    """
    Initialize GROBID client with error handling.
    
    Returns:
        GrobidClient instance if successful, None otherwise
    """
    try:
        server_url = f"{config.GROBID_HOST}:{config.GROBID_PORT}"
        logger.info(f"Initializing GROBID client for server at {server_url}")
        
        client = GrobidClient(grobid_server=server_url, check_server=True)
        
        # Test connection
        response = requests.get(f"{server_url}/api/isalive", timeout=10)
        if response.status_code == 200:
            logger.info("GROBID server is responsive")
            return client
        else:
            logger.warning(f"GROBID server returned status {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to initialize GROBID client: {e}")
        return None

def parse_pdf_with_grobid(client: GrobidClient, pdf_path: str) -> Optional[str]:
    """
    Parse PDF using GROBID with comprehensive error handling.
    
    Args:
        client: GROBID client instance
        pdf_path: Path to PDF file
        
    Returns:
        Parsed text if successful, None otherwise
    """
    try:
        logger.info(f"Parsing PDF with GROBID: {os.path.basename(pdf_path)}")
        
        _, status_code, response_text = client.process_pdf(
            "processFulltextDocument", 
            pdf_path,
            generateIDs=False, 
            consolidate_header=True,
            consolidate_citations=False, 
            include_raw_citations=False,
            include_raw_affiliations=False, 
            tei_coordinates=False,
            segment_sentences=False
        )
        
        if status_code == 200:
            if response_text and len(response_text.strip()) > 100:  # Ensure meaningful content
                logger.info(f"Successfully parsed PDF: {os.path.basename(pdf_path)}")
                return response_text
            else:
                logger.warning(f"GROBID returned empty or too short content for {pdf_path}")
        else:
            logger.warning(f"GROBID parsing failed with status {status_code} for {pdf_path}")
            
    except Exception as e:
        logger.error(f"Exception during GROBID parsing of {pdf_path}: {e}")
    
    return None

def create_paper_entry(paper, full_text: Optional[str] = None) -> Dict:
    """
    Create a standardized paper entry dictionary.
    
    Args:
        paper: ArXiv paper object
        full_text: Parsed full text (optional)
        
    Returns:
        Dictionary with paper information
    """
    return {
        'paperId': paper.get_short_id(),
        'title': paper.title,
        'authors': [author.name for author in paper.authors],
        'year': paper.published.year,
        'abstract': paper.summary,
        'full_text': full_text if full_text else paper.summary,
        'pdf_url': paper.pdf_url,
        'arxiv_url': paper.entry_id
    }

@tool
def enhanced_literature_search(query: str, max_results: int = 3) -> List[Dict]:
    """
    Performs an enhanced literature search using the ArXiv API,
    downloads PDFs, and parses them into structured text using GROBID.
    Includes comprehensive error handling and fallback mechanisms.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of paper dictionaries with metadata and full text
    """
    logger.info(f"Starting enhanced ArXiv search for: '{query[:50]}...'")
    
    # Create output directories
    pdf_dir = os.path.join(config.OUTPUT_DIR, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    
    papers_data = []
    
    try:
        # 1. Initialize ArXiv search with improved parameters
        search = arxiv.Search(
            query=query,
            max_results=min(max_results * 2, 10),  # Search for more to account for failures
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        # Add timeout and retry mechanism
        client = arxiv.Client(delay_seconds=3, num_retries=3)
        results = list(client.results(search))

        if not results:
            logger.warning(f"No results found on ArXiv for query: {query}")
            # Try fallback search with broader terms
            return _fallback_search(query, max_results)

        logger.info(f"Found {len(results)} ArXiv results")

        # 2. Initialize GROBID client
        grobid_client = initialize_grobid_client()
        if not grobid_client:
            logger.warning("GROBID not available, will use abstracts only")

        # 3. Process papers
        processed_count = 0
        for i, paper in enumerate(results):
            if processed_count >= max_results:
                break
                
            try:
                logger.info(f"Processing paper {i+1}/{len(results)}: {paper.title[:50]}...")
                
                paper_id = paper.get_short_id()
                title = paper.title
                
                # Create safe filename
                safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '_')).rstrip()
                safe_title = safe_title[:30]  # Limit length
                pdf_filename = f"{paper_id}_{safe_title}.pdf"
                pdf_path = os.path.join(pdf_dir, pdf_filename)
                
                full_text = None
                
                # Try to download and parse PDF if GROBID is available
                if grobid_client:
                    if download_pdf(paper.pdf_url, pdf_path, is_arxiv=True):
                        full_text = parse_pdf_with_grobid(grobid_client, pdf_path)
                        
                        # Clean up PDF file after processing to save space
                        try:
                            os.remove(pdf_path)
                        except:
                            pass
                
                # Create paper entry
                paper_entry = create_paper_entry(paper, full_text)
                papers_data.append(paper_entry)
                processed_count += 1
                
                logger.info(f"Successfully processed: {title[:30]}...")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing paper {paper.get_short_id()}: {e}")
                # Still try to add basic info
                try:
                    paper_entry = create_paper_entry(paper)
                    papers_data.append(paper_entry)
                    processed_count += 1
                except:
                    continue

        # 4. Final validation and logging
        if papers_data:
            full_text_count = sum(1 for p in papers_data if len(p.get('full_text', '')) > len(p.get('abstract', '')))
            logger.info(f"Successfully processed {len(papers_data)} papers ({full_text_count} with full text)")
        else:
            logger.warning("No papers successfully processed")
            return _fallback_search(query, max_results)

    except Exception as e:
        logger.error(f"ArXiv search process failed: {e}")
        return _fallback_search(query, max_results)

    # Rate limiting between searches
    logger.info("Waiting to respect API rate limits...")
    time.sleep(5)
    
    return papers_data

def _fallback_search(query: str, max_results: int) -> List[Dict]:
    """
    Fallback search using predefined terms when main search fails.
    
    Args:
        query: Original query that failed
        max_results: Number of results needed
        
    Returns:
        List of fallback paper entries
    """
    logger.info("Using fallback search mechanism...")
    
    fallback_papers = []
    
    # Check if fallback terms are configured
    if hasattr(config, 'FALLBACK_SEARCH_TERMS'):
        fallback_terms = config.FALLBACK_SEARCH_TERMS
    else:
        # Default fallback terms based on common research areas
        fallback_terms = [
            "machine learning optimization",
            "deep neural networks", 
            "computer vision algorithms",
            "natural language processing",
            "artificial intelligence methods"
        ]
    
    # Create synthetic entries based on query and fallback terms
    for i, term in enumerate(fallback_terms[:max_results]):
        fallback_papers.append({
            'paperId': f'fallback_{i}_{hash(query) % 10000}',
            'title': f'Research on {term} - Fallback Entry',
            'authors': ['Fallback Author'],
            'year': 2024,
            'abstract': f'This is a fallback entry for research on {term}. Original query: {query}. This entry is generated when the primary search mechanism fails.',
            'full_text': f'Fallback research content related to {term}. This content is generated as a placeholder when external search services are unavailable. The original search was for: {query}.',
            'pdf_url': '',
            'arxiv_url': '',
            'is_fallback': True
        })
    
    logger.info(f"Generated {len(fallback_papers)} fallback entries")
    return fallback_papers

def _create_abstract_only_papers(arxiv_results) -> List[Dict]:
    """
    Create paper entries with only abstract when GROBID is unavailable.
    
    Args:
        arxiv_results: List of ArXiv paper results
        
    Returns:
        List of paper dictionaries with abstracts as full text
    """
    papers = []
    for paper in arxiv_results:
        paper_entry = create_paper_entry(paper)
        papers.append(paper_entry)
    
    logger.info(f"Created {len(papers)} abstract-only entries")
    return papers

# Additional utility functions for search enhancement

def validate_search_query(query: str) -> str:
    """
    Validate and clean search query.
    
    Args:
        query: Raw search query
        
    Returns:
        Cleaned and validated query
    """
    if not query or len(query.strip()) < 3:
        return "machine learning"
    
    # Remove special characters that might cause issues
    cleaned = ''.join(c for c in query if c.isalnum() or c in (' ', '-', '_'))
    
    # Limit length
    if len(cleaned) > 100:
        cleaned = cleaned[:100]
    
    return cleaned.strip()

def extract_technical_terms(text: str) -> List[str]:
    """
    Extract potential technical terms from text for search enhancement.
    
    Args:
        text: Input text
        
    Returns:
        List of extracted technical terms
    """
    # Simple technical term extraction
    words = text.lower().split()
    technical_indicators = ['algorithm', 'method', 'technique', 'approach', 'model', 'framework']
    
    terms = []
    for i, word in enumerate(words):
        if word in technical_indicators and i < len(words) - 1:
            # Get the next word as it might be a technical term
            next_word = words[i + 1]
            if len(next_word) > 3:
                terms.append(next_word)
    
    return list(set(terms))[:5]  # Return unique terms, max 5

# Export tools and functions
search_tool = enhanced_literature_search
tools = [enhanced_literature_search]

# Export utility functions for use in other modules
__all__ = [
    'enhanced_literature_search',
    'download_pdf', 
    'initialize_grobid_client',
    'parse_pdf_with_grobid',
    'validate_search_query',
    'extract_technical_terms',
    'search_tool',
    'tools'
]