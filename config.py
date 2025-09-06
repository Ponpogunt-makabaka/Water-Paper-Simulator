# config.py
"""
Centralized configuration for the research paper generation system.
All hyperparameters and settings are documented here.
ENHANCED: Added local PDF repository support.
"""

# ==============================================================================
# CORE HYPERPARAMETERS - Adjust these to control system behavior
# ==============================================================================

# --- Research Topic ---
TOPIC = "PARA-Cell (Parameter-Adaptive Recurrent Attention-SSM Cell),a novel recurrent unit that realizes micro-level deep fusion. Within each state update, attention and the SSM no longer act as external components; instead, they function as co-processors that jointly analyze a shared historical memory buffer. Their outputs dynamically generate the core update parameters of the recurrent unit itself."
# Sample topic : The plug-and-play modules used in deep learning are optimized through trition operators so that the entire stitched model can be placed in a GPU-kernal
# The main research topic that will be investigated

# --- Research Mode Selection ---
RESEARCH_MODE = "interactive"  # Options: "online", "local", "interactive"
# "online": Use online search (ArXiv, etc.)
# "local": Use local PDF repository only
# "interactive": Ask user to choose at runtime

# --- Local PDF Repository Settings ---
LOCAL_PDF_DIR = "local_papers"  # Directory containing local PDF files
# This directory should be created in the project root
# Place your PDF papers here when using local mode

PROCESS_LOCAL_PDFS_ON_STARTUP = True  # Whether to process all PDFs at startup
# If False, PDFs will be processed on-demand

LOCAL_PDF_METADATA_CACHE = "local_papers_metadata.json"  # Cache file for PDF metadata
# Stores processed PDF information to avoid reprocessing

# --- Model Selection ---
MODEL_PROVIDER = "ollama"  # Options: "ollama" or "openai"
# Choose between local Ollama models or OpenAI API

# --- Workflow Control ---
MAX_REVISIONS = 3  # Maximum number of revision cycles (1-5 recommended)
# Higher values allow more iterations but may cause loops

RECURSION_LIMIT = 20  # LangGraph recursion limit (15-30 recommended)
# Prevents infinite loops in the workflow graph

MIN_IMPROVEMENT_THRESHOLD = 0.3  # Minimum improvement score (0.0-1.0)
# Used to detect if revisions are actually improving the paper

# --- Search Parameters ---
MAX_SEARCH_RESULTS = 3  # Results per search query (1-5 recommended)
# More results = better coverage but higher token usage

SEARCH_TIMEOUT = 30  # Timeout for search operations in seconds
# Prevents hanging on slow network connections

ARXIV_MAX_RESULTS = 3  # Maximum papers from ArXiv per query
# Limits the number of papers retrieved to manage tokens

# --- Local PDF Processing ---
MAX_LOCAL_PAPERS = 10  # Maximum number of local papers to process
# Prevents overwhelming the system with too many local papers

LOCAL_PDF_MIN_SIZE = 50000  # Minimum PDF size in bytes (50KB)
# Skip PDFs that are too small to be meaningful papers

LOCAL_PDF_MAX_SIZE = 50000000  # Maximum PDF size in bytes (50MB)
# Skip PDFs that are too large to process efficiently

# --- Content Length Limits ---
MAX_PLAN_LENGTH = 2000  # Characters for research plans
# Controls the detail level of initial research proposals

MAX_SECTION_LENGTH = 4000  # Default characters per paper section
# General limit for paper sections (can be overridden per section)

MAX_FEEDBACK_LENGTH = 500  # Characters for reviewer feedback
# Limits the verbosity of review comments

# --- Section-Specific Length Limits ---
SECTION_LENGTHS = {
    "Abstract": 1500,      # ~250 words - concise summary
    "Introduction": 3500,  # ~600 words - context and motivation
    "Literature Review": 4000,  # ~700 words - prior work analysis
    "Method": 5000,       # ~900 words - technical details
    "Evaluation": 4500,   # ~800 words - experimental design
    "Conclusion": 2500    # ~400 words - summary and future work
}

# --- Temperature Settings ---
# Controls randomness/creativity of different agents (0.0-1.0)
TEMPERATURE_RESEARCH = 0.5  # Higher for creative ideation
TEMPERATURE_ANALYSIS = 0.3  # Lower for focused evaluation
TEMPERATURE_WRITING = 0.5   # Balanced for content generation
TEMPERATURE_REVIEW = 0.2    # Critical for objective assessment

# --- Review Strictness ---
REVIEW_STRICTNESS = 0.5  # How strict reviewers are (0.0-1.0) Hint:Just like my reviewer
# Lower = more lenient, Higher = more demanding

REVIEW_PASS_THRESHOLD = 0.5  # Score needed to pass review (0.0-1.0) Hint:lower easyer pass
# Papers scoring above this threshold pass without revision

# --- Token Optimization ---
MAX_PROMPT_LENGTH = 3000  # Maximum characters in prompts to LLM
# Prevents token overflow in model inputs

TRUNCATE_REFERENCES = True  # Whether to truncate long references
# Saves tokens by limiting reference text length

# ==============================================================================
# MODEL CONFIGURATIONS
# ==============================================================================

# --- Ollama Settings (for local models) ---
OLLAMA_MODEL = "gpt-oss:latest"  # Model name in Ollama
# Smaller models are faster but less capable
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:latest" # Name of the embedding model hosted in Ollama

OLLAMA_BASE_URL = "http://127.0.0.1:11434"  # Ollama server URL
# Default local server address

# --- OpenAI-Compatible API Settings ---
OPENAI_API_KEY = "sk-your-api-key-here"  # Your API key
# Required if using OpenAI or compatible services

OPENAI_API_BASE = "https://api.openai.com/v1"  # API endpoint
# Can be changed for OpenAI-compatible services

OPENAI_MODEL_NAME = "gpt-4o-mini"  # Model to use
# Options: gpt-4, gpt-4-turbo, gpt-3.5-turbo, etc.

# ==============================================================================
# OUTPUT SETTINGS
# ==============================================================================

OUTPUT_DIR = "output"  # Directory for generated files
# All outputs (papers, logs, reports) saved here

REFERENCE_FILE = "references.txt"  # Filename for references
DRAFT_FILE_PREFIX = "draft_v"  # Prefix for draft versions
REVIEW_REPORT_PREFIX = "review_report_"  # Prefix for review reports

# --- Output Formats ---
SAVE_MARKDOWN = True  # Save papers in Markdown format
SAVE_PLAIN_TEXT = True  # Save papers in plain text
GENERATE_REVIEW_REPORTS = True  # Generate detailed review reports

# ==============================================================================
# LOGGING AND MONITORING
# ==============================================================================

# --- Logging Level ---
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
# Controls verbosity of console and file logs

# --- LangSmith Monitoring (Optional) ---
LANGSMITH_TRACING = "false"  # Set to "true" to enable tracing
# Enables detailed execution tracking in LangSmith

LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
# LangSmith API endpoint

LANGSMITH_API_KEY = "Sample_key"
# Your LangSmith API key (get from smith.langchain.com)

LANGSMITH_PROJECT = "water_paper"
# Project name in LangSmith dashboard

# ==============================================================================
# ADVANCED SETTINGS - Modify with caution
# ==============================================================================

# --- Retry Logic ---
MAX_RETRIES = 3  # Maximum retries for failed operations
RETRY_DELAY = 2  # Seconds between retries

# --- Memory Management ---
CLEAR_MEMORY_BETWEEN_RUNS = False  # Clear state between runs
CACHE_SEARCH_RESULTS = True  # Cache search results to avoid duplicates

# --- Parallel Processing ---
ENABLE_PARALLEL_SEARCH = False  # Run multiple searches in parallel
MAX_PARALLEL_SEARCHES = 2  # Maximum concurrent searches

# --- Debug Options ---
DEBUG_MODE = False  # Enable detailed debug output
SAVE_INTERMEDIATE_STATES = True  # Save state after each agent
VERBOSE_PROMPTS = False  # Show full prompts sent to LLM

# ==============================================================================
# EXPERIMENTAL FEATURES - May be unstable
# ==============================================================================

# --- Auto-tuning ---
AUTO_ADJUST_TEMPERATURES = False  # Automatically adjust temperatures based on performance
LEARN_FROM_FEEDBACK = False  # Use previous feedback to improve future runs

# --- Advanced Optimization ---
USE_BEAM_SEARCH = False  # Use beam search for better results
BEAM_WIDTH = 3  # Number of beams to maintain

# --- Multi-agent Collaboration ---
ENABLE_AGENT_DEBATE = False  # Allow agents to debate decisions
MAX_DEBATE_ROUNDS = 2  # Maximum rounds of debate

# ==============================================================================
# EXTERNAL TOOL CONFIGURATIONS
# ==============================================================================
GROBID_HOST = "http://localhost"
GROBID_PORT = "8070"

# ==============================================================================
# HELPER FUNCTIONS FOR CONFIGURATION
# ==============================================================================

def get_research_mode():
    """
    Get the research mode, with interactive fallback.
    
    Returns:
        str: "online", "local", or user choice if interactive
    """
    if RESEARCH_MODE == "interactive":
        return prompt_user_for_research_mode()
    return RESEARCH_MODE

def prompt_user_for_research_mode():
    """
    Prompt user to choose research mode interactively.
    
    Returns:
        str: User's choice of research mode
    """
    print("\n" + "="*60)
    print("RESEARCH MODE SELECTION")
    print("="*60)
    print("\nChoose your research source:")
    print("1. Online Search (ArXiv, academic databases)")
    print("   - Pros: Latest papers, broad coverage")
    print("   - Cons: May be slower, requires internet")
    print("\n2. Local PDF Repository")
    print("   - Pros: Faster, use your curated papers")
    print("   - Cons: Limited to your collection")
    
    # Check if local papers exist
    import os
    local_path = LOCAL_PDF_DIR
    pdf_count = 0
    if os.path.exists(local_path):
        pdf_files = [f for f in os.listdir(local_path) if f.lower().endswith('.pdf')]
        pdf_count = len(pdf_files)
    
    if pdf_count > 0:
        print(f"\n📁 Found {pdf_count} PDF files in '{local_path}'")
    else:
        print(f"\n⚠️  No PDF files found in '{local_path}'")
        print(f"   Create the folder and add PDF papers to use local mode")
    
    while True:
        try:
            choice = input("\nEnter your choice (1 or 2): ").strip()
            
            if choice == "1":
                print("✅ Selected: Online Search")
                return "online"
            elif choice == "2":
                if pdf_count == 0:
                    print("❌ Cannot use local mode: No PDF files found")
                    continue
                print("✅ Selected: Local PDF Repository")
                return "local"
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
                
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user. Defaulting to online search.")
            return "online"
        except Exception as e:
            print(f"❌ Error: {e}. Defaulting to online search.")
            return "online"

def validate_local_pdf_setup():
    """
    Validate that local PDF setup is correct.
    
    Returns:
        tuple: (bool, str) - (is_valid, message)
    """
    import os
    
    if not os.path.exists(LOCAL_PDF_DIR):
        return False, f"Local PDF directory '{LOCAL_PDF_DIR}' does not exist"
    
    pdf_files = [f for f in os.listdir(LOCAL_PDF_DIR) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        return False, f"No PDF files found in '{LOCAL_PDF_DIR}'"
    
    # Check file sizes
    valid_pdfs = 0
    for pdf_file in pdf_files:
        pdf_path = os.path.join(LOCAL_PDF_DIR, pdf_file)
        try:
            size = os.path.getsize(pdf_path)
            if LOCAL_PDF_MIN_SIZE <= size <= LOCAL_PDF_MAX_SIZE:
                valid_pdfs += 1
        except:
            continue
    
    if valid_pdfs == 0:
        return False, f"No valid PDF files found (size between {LOCAL_PDF_MIN_SIZE} and {LOCAL_PDF_MAX_SIZE} bytes)"
    
    return True, f"Found {valid_pdfs} valid PDF files in '{LOCAL_PDF_DIR}'"

def create_local_pdf_directory():
    """
    Create local PDF directory with instructions.
    """
    import os
    
    if not os.path.exists(LOCAL_PDF_DIR):
        os.makedirs(LOCAL_PDF_DIR)
        
        # Create a README file with instructions
        readme_content = """# Local PDF Repository

This directory is for storing PDF research papers that you want to use as source material.

## Instructions:

1. Place your PDF research papers in this directory
2. Supported formats: .pdf files only
3. File size limits: 50KB to 50MB per file
4. Maximum papers processed: 10 files

## File Naming:

For best results, name your PDF files descriptively:
- Good: "smith2023_neural_networks.pdf"
- Good: "recent_advances_in_AI.pdf"
- Avoid: "paper1.pdf", "download.pdf"

## Quality Tips:

- Use recent, high-quality research papers
- Ensure PDFs are text-searchable (not just scanned images)
- Papers should be relevant to your research topic
- Academic papers work better than general articles

## Processing:

The system will automatically:
- Extract text from PDFs using GROBID
- Create searchable vector embeddings
- Use content for research paper generation

Place your PDF files here and restart the system to use local mode.
"""
