# local_pdf_processor.py
"""
Local PDF processing module for the research paper generation system.
Handles PDF discovery, metadata extraction, and full-text parsing using GROBID.
"""

import os
import json
import hashlib
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime

import config
from tools import initialize_grobid_client, parse_pdf_with_grobid

logger = logging.getLogger(__name__)

class LocalPDFProcessor:
    """
    Processes local PDF files to extract metadata and full text.
    Provides caching and intelligent file management.
    """
    
    def __init__(self):
        self.pdf_dir = Path(config.LOCAL_PDF_DIR)
        self.cache_file = Path(config.OUTPUT_DIR) / config.LOCAL_PDF_METADATA_CACHE
        self.grobid_client = None
        self.metadata_cache = {}
        
        # Ensure directories exist
        self.pdf_dir.mkdir(exist_ok=True)
        Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
        
        # Load cache if it exists
        self._load_cache()
        
        # Initialize GROBID client
        self._initialize_grobid()
    
    def _load_cache(self) -> None:
        """Load metadata cache from file."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.metadata_cache = json.load(f)
                logger.info(f"Loaded metadata cache with {len(self.metadata_cache)} entries")
            else:
                self.metadata_cache = {}
                logger.info("No existing metadata cache found")
        except Exception as e:
            logger.error(f"Error loading metadata cache: {e}")
            self.metadata_cache = {}
    
    def _save_cache(self) -> None:
        """Save metadata cache to file."""
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0',
                'papers': self.metadata_cache
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved metadata cache with {len(self.metadata_cache)} entries")
            
        except Exception as e:
            logger.error(f"Error saving metadata cache: {e}")
    
    def _initialize_grobid(self) -> None:
        """Initialize GROBID client for PDF parsing."""
        try:
            self.grobid_client = initialize_grobid_client()
            if self.grobid_client:
                logger.info("GROBID client initialized successfully")
            else:
                logger.warning("GROBID client initialization failed - will use filename-based metadata")
        except Exception as e:
            logger.error(f"Error initializing GROBID: {e}")
            self.grobid_client = None
    
    def discover_pdf_files(self) -> List[Path]:
        """
        Discover all valid PDF files in the local directory.
        
        Returns:
            List of valid PDF file paths
        """
        if not self.pdf_dir.exists():
            logger.warning(f"PDF directory does not exist: {self.pdf_dir}")
            return []
        
        pdf_files = []
        
        for file_path in self.pdf_dir.glob("*.pdf"):
            if self._is_valid_pdf(file_path):
                pdf_files.append(file_path)
            else:
                logger.warning(f"Skipping invalid PDF: {file_path}")
        
        # Sort by modification time (newest first)
        pdf_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Limit to max papers
        if len(pdf_files) > config.MAX_LOCAL_PAPERS:
            logger.info(f"Found {len(pdf_files)} PDFs, limiting to {config.MAX_LOCAL_PAPERS} most recent")
            pdf_files = pdf_files[:config.MAX_LOCAL_PAPERS]
        
        logger.info(f"Discovered {len(pdf_files)} valid PDF files")
        return pdf_files
    
    def _is_valid_pdf(self, file_path: Path) -> bool:
        """
        Check if a PDF file is valid for processing.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check file exists and is readable
            if not file_path.exists() or not file_path.is_file():
                return False
            
            # Check file size
            size = file_path.stat().st_size
            if size < config.LOCAL_PDF_MIN_SIZE or size > config.LOCAL_PDF_MAX_SIZE:
                logger.warning(f"PDF size {size} bytes outside valid range")
                return False
            
            # Try to open file to check it's not corrupted
            with open(file_path, 'rb') as f:
                header = f.read(4)
                if header != b'%PDF':
                    logger.warning(f"File doesn't appear to be a valid PDF: {file_path}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating PDF {file_path}: {e}")
            return False
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file for caching."""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return str(file_path.stat().st_mtime)  # Fallback to mtime
    
    def extract_metadata_from_filename(self, file_path: Path) -> Dict:
        """
        Extract basic metadata from filename when GROBID is unavailable.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with extracted metadata
        """
        filename = file_path.stem  # Remove .pdf extension
        
        # Common patterns in academic PDF filenames
        metadata = {
            'title': filename.replace('_', ' ').replace('-', ' ').title(),
            'authors': ['Unknown Author'],
            'year': 2024,  # Default year
            'abstract': f'Local paper: {filename}',
            'filename': file_path.name,
            'source': 'local_file'
        }
        
        # Try to extract year from filename
        import re
        year_match = re.search(r'(20\d{2})', filename)
        if year_match:
            metadata['year'] = int(year_match.group(1))
        
        # Try to extract author from common patterns
        # Pattern: "author_year_title" or "author-year-title"
        parts = re.split(r'[_-]', filename.lower())
        if len(parts) >= 2:
            # First part might be author
            potential_author = parts[0].replace('.', ' ').title()
            if len(potential_author) > 2 and potential_author.isalpha():
                metadata['authors'] = [potential_author]
        
        return metadata
    
    def extract_metadata_with_grobid(self, file_path: Path) -> Optional[Dict]:
        """
        Extract metadata using GROBID parsing.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with extracted metadata or None if failed
        """
        if not self.grobid_client:
            return None
        
        try:
            # Parse PDF with GROBID
            parsed_text = parse_pdf_with_grobid(self.grobid_client, str(file_path))
            
            if not parsed_text:
                logger.warning(f"GROBID returned no text for {file_path}")
                return None
            
            # Extract metadata from parsed text
            metadata = self._parse_grobid_output(parsed_text)
            metadata['filename'] = file_path.name
            metadata['source'] = 'grobid_parsed'
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata with GROBID for {file_path}: {e}")
            return None
    
    def _parse_grobid_output(self, grobid_text: str) -> Dict:
        """
        Parse GROBID XML/text output to extract structured metadata.
        
        Args:
            grobid_text: Raw output from GROBID
            
        Returns:
            Dictionary with extracted metadata
        """
        import re
        from xml.etree import ElementTree as ET
        
        metadata = {
            'title': 'Unknown Title',
            'authors': ['Unknown Author'],
            'year': 2024,
            'abstract': '',
            'full_text': grobid_text[:5000] if grobid_text else ''  # First 5000 chars
        }
        
        try:
            # Try to parse as XML first
            if grobid_text.strip().startswith('<?xml') or grobid_text.strip().startswith('<'):
                root = ET.fromstring(grobid_text)
                
                # Extract title
                title_elem = root.find('.//title[@level="a"]')
                if title_elem is not None and title_elem.text:
                    metadata['title'] = title_elem.text.strip()
                
                # Extract authors
                authors = []
                author_elems = root.findall('.//author')
                for author in author_elems:
                    forename = author.find('.//forename')
                    surname = author.find('.//surname')
                    if forename is not None and surname is not None:
                        full_name = f"{forename.text} {surname.text}".strip()
                        if full_name:
                            authors.append(full_name)
                
                if authors:
                    metadata['authors'] = authors
                
                # Extract abstract
                abstract_elem = root.find('.//abstract')
                if abstract_elem is not None:
                    abstract_text = ''.join(abstract_elem.itertext()).strip()
                    if abstract_text:
                        metadata['abstract'] = abstract_text[:1000]  # Limit length
                
                # Extract year from publication date
                date_elem = root.find('.//date[@type="published"]')
                if date_elem is not None:
                    when = date_elem.get('when')
                    if when:
                        year_match = re.search(r'(\d{4})', when)
                        if year_match:
                            metadata['year'] = int(year_match.group(1))
            
            else:
                # Fallback: extract from plain text
                lines = grobid_text.split('\n')
                
                # Try to find title (usually in first few lines)
                for i, line in enumerate(lines[:10]):
                    line = line.strip()
                    if len(line) > 10 and not line.isupper() and not line.isdigit():
                        metadata['title'] = line[:200]  # Limit title length
                        break
                
                # Try to extract year
                year_match = re.search(r'(20\d{2})', grobid_text)
                if year_match:
                    metadata['year'] = int(year_match.group(1))
                
                # Use first paragraph as abstract
                paragraphs = [p.strip() for p in grobid_text.split('\n\n') if len(p.strip()) > 50]
                if paragraphs:
                    metadata['abstract'] = paragraphs[0][:1000]
        
        except Exception as e:
            logger.error(f"Error parsing GROBID output: {e}")
        
        return metadata
    
    def process_pdf_file(self, file_path: Path) -> Dict:
        """
        Process a single PDF file to extract all metadata and content.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with paper data
        """
        # Calculate file hash for caching
        file_hash = self._calculate_file_hash(file_path)
        cache_key = f"{file_path.name}_{file_hash}"
        
        # Check cache first
        if cache_key in self.metadata_cache:
            logger.info(f"Using cached metadata for {file_path.name}")
            cached_data = self.metadata_cache[cache_key]
            cached_data['paperId'] = cache_key  # Ensure paperId is set
            return cached_data
        
        logger.info(f"Processing PDF: {file_path.name}")
        
        # Try GROBID first, fallback to filename-based extraction
        metadata = self.extract_metadata_with_grobid(file_path)
        
        if not metadata:
            logger.info(f"GROBID failed, using filename-based extraction for {file_path.name}")
            metadata = self.extract_metadata_from_filename(file_path)
        
        # Ensure full_text is available (use abstract as fallback)
        if 'full_text' not in metadata or not metadata['full_text']:
            metadata['full_text'] = metadata.get('abstract', f'Content from {file_path.name}')
        
        # Add additional metadata
        metadata.update({
            'paperId': cache_key,
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size,
            'processed_date': datetime.now().isoformat(),
            'is_local': True
        })
        
        # Cache the result
        self.metadata_cache[cache_key] = metadata
        
        return metadata
    
    def process_all_pdfs(self) -> List[Dict]:
        """
        Process all PDF files in the local directory.
        
        Returns:
            List of paper dictionaries
        """
        logger.info("Starting local PDF processing...")
        
        pdf_files = self.discover_pdf_files()
        
        if not pdf_files:
            logger.warning("No valid PDF files found for processing")
            return []
        
        papers_data = []
        
        for i, pdf_file in enumerate(pdf_files, 1):
            try:
                logger.info(f"Processing file {i}/{len(pdf_files)}: {pdf_file.name}")
                
                paper_data = self.process_pdf_file(pdf_file)
                papers_data.append(paper_data)
                
                # Progress indicator
                if i % 3 == 0:
                    logger.info(f"Processed {i}/{len(pdf_files)} files...")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
                continue
        
        # Save cache after processing
        self._save_cache()
        
        logger.info(f"Successfully processed {len(papers_data)} PDF files")
        return papers_data
    
    def get_processing_summary(self) -> Dict:
        """Get summary of processing status."""
        pdf_files = self.discover_pdf_files()
        
        return {
            'total_pdfs_found': len(pdf_files),
            'cached_papers': len(self.metadata_cache),
            'grobid_available': self.grobid_client is not None,
            'pdf_directory': str(self.pdf_dir),
            'cache_file': str(self.cache_file),
            'max_papers': config.MAX_LOCAL_PAPERS
        }
    
    def clear_cache(self) -> None:
        """Clear the metadata cache."""
        self.metadata_cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("Cleared metadata cache")

# Factory function
def create_local_pdf_processor() -> LocalPDFProcessor:
    """Create and return a LocalPDFProcessor instance."""
    return LocalPDFProcessor()

# Utility functions for integration with existing system
def get_local_papers_data() -> List[Dict]:
    """
    Get processed papers data from local PDF files.
    This function can be called by the researcher agent.
    
    Returns:
        List of paper dictionaries ready for RAG processing
    """
    try:
        processor = create_local_pdf_processor()
        return processor.process_all_pdfs()
    except Exception as e:
        logger.error(f"Error getting local papers data: {e}")
        return []

def validate_local_setup() -> Tuple[bool, str]:
    """
    Validate that local PDF setup is working correctly.
    
    Returns:
        Tuple of (success, message)
    """
    try:
        processor = create_local_pdf_processor()
        summary = processor.get_processing_summary()
        
        if summary['total_pdfs_found'] == 0:
            return False, f"No PDF files found in {summary['pdf_directory']}"
        
        if not summary['grobid_available']:
            return True, f"Found {summary['total_pdfs_found']} PDFs. GROBID unavailable - will use filename-based extraction."
        
        return True, f"Found {summary['total_pdfs_found']} PDFs. GROBID available for full-text extraction."
        
    except Exception as e:
        return False, f"Error validating local setup: {e}"

# Export main classes and functions
__all__ = [
    'LocalPDFProcessor',
    'create_local_pdf_processor', 
    'get_local_papers_data',
    'validate_local_setup'
]