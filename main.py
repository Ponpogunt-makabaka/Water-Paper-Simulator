# main.py
"""
Complete main entry point for enhanced research paper generation system.
Supports interactive mode selection between online search and local PDF repository.
Implements enhanced consistency management and comprehensive error handling.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import config
from graph import create_app
from local_pdf_processor import validate_local_setup


def setup_langsmith():
    """Configure LangSmith tracing if enabled."""
    if config.LANGSMITH_TRACING.lower() == "true":
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = config.LANGSMITH_ENDPOINT
        os.environ["LANGCHAIN_API_KEY"] = config.LANGSMITH_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = config.LANGSMITH_PROJECT
        print("✅ LangSmith tracing enabled")
        print(f"   Project: {config.LANGSMITH_PROJECT}")
    else:
        print("ℹ️ LangSmith tracing disabled")


def validate_system_setup():
    """
    Validate system setup and display status.
    
    Returns:
        tuple: (is_ready, status_messages)
    """
    print("\n" + "="*80)
    print("ENHANCED SYSTEM SETUP VALIDATION")
    print("="*80)
    
    status_messages = []
    all_ready = True
    
    # 1. Check output directory
    try:
        Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
        status_messages.append("✅ Output directory ready")
    except Exception as e:
        status_messages.append(f"❌ Output directory error: {e}")
        all_ready = False
    
    # 2. Check model configuration
    try:
        if config.MODEL_PROVIDER == "ollama":
            status_messages.append(f"🤖 Ollama model: {config.OLLAMA_MODEL}")
            status_messages.append(f"🔤 Embedding model: {config.OLLAMA_EMBEDDING_MODEL}")
        else:
            status_messages.append(f"🤖 OpenAI model: {config.OPENAI_MODEL_NAME}")
        
        status_messages.append(f"✅ Model configuration: {config.MODEL_PROVIDER}")
    except Exception as e:
        status_messages.append(f"❌ Model configuration error: {e}")
        all_ready = False
    
    # 3. Check consistency management
    try:
        if hasattr(config, 'ENABLE_CONSISTENCY_MANAGEMENT') and config.ENABLE_CONSISTENCY_MANAGEMENT:
            status_messages.append("🔍 Consistency management: Enabled")
        else:
            status_messages.append("⚠️  Consistency management: Disabled")
    except Exception as e:
        status_messages.append(f"⚠️  Consistency management check failed: {e}")
    
    # 4. Check local PDF setup
    try:
        local_valid, local_msg = validate_local_setup()
        if local_valid:
            status_messages.append(f"📁 Local PDFs: {local_msg}")
        else:
            status_messages.append(f"⚠️  Local PDFs: {local_msg}")
    except Exception as e:
        status_messages.append(f"⚠️  Local PDF check failed: {e}")
    
    # 5. Check GROBID availability (optional)
    try:
        import requests
        grobid_url = f"{config.GROBID_HOST}:{config.GROBID_PORT}/api/isalive"
        response = requests.get(grobid_url, timeout=5)
        if response.status_code == 200:
            status_messages.append("🔧 GROBID service: Available")
        else:
            status_messages.append("⚠️  GROBID service: Not responding")
    except:
        status_messages.append("⚠️  GROBID service: Not available (will use fallback)")
    
    # Display all status messages
    for msg in status_messages:
        print(f"  {msg}")
    
    return all_ready, status_messages


class OutputManager:
    """Enhanced output manager with research mode and consistency tracking."""
    
    def __init__(self):
        self.output_dir = Path(config.OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)
        self.setup_logging()
        
    def setup_logging(self) -> None:
        """Configure comprehensive logging with file and console output."""
        log_file = self.output_dir / "workflow.log"
        
        # Clear existing handlers to avoid conflicts
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Configure file handler
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Configure console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        # Configure root logger
        logging.basicConfig(
            level=logging.DEBUG,
            handlers=[file_handler, console_handler],
            force=True
        )
        
        # Set specific logger levels
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("faiss").setLevel(logging.WARNING)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Enhanced logging initialized. Log file: {log_file}")
        
        # Test log file writing
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"=== Enhanced Workflow started at {datetime.now()} ===\n")
            self.logger.info("Log file test successful")
        except Exception as e:
            self.logger.error(f"Log file test failed: {e}")
    
    def save_paper(self, content: str, research_mode: str = "unknown") -> Path:
        """Save final paper with enhanced metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"paper_{research_mode}_{timestamp}.txt"
        filepath = self.output_dir / filename
        
        # Add enhanced metadata header
        consistency_status = "Enabled" if getattr(config, 'ENABLE_CONSISTENCY_MANAGEMENT', False) else "Disabled"
        
        header = f"""# Enhanced Research Paper Generated by AI System
# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# Topic: {config.TOPIC}
# Research Mode: {research_mode.upper()}
# Model Provider: {config.MODEL_PROVIDER}
# Consistency Management: {consistency_status}
# System: Water-Paper-Simulator Enhanced v2.0

{'='*80}

"""
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(header + content)
        
        self.logger.info(f"Enhanced paper saved to: {filepath}")
        return filepath
    
    def save_state(self, state: Dict[str, Any]) -> None:
        """Save final state with enhanced information."""
        filepath = self.output_dir / "final_state.txt"
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("ENHANCED WORKFLOW STATE SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Topic: {config.TOPIC}\n")
            f.write(f"Research Mode: {state.get('research_mode', 'unknown').upper()}\n")
            f.write(f"Consistency Management: {'Enabled' if getattr(config, 'ENABLE_CONSISTENCY_MANAGEMENT', False) else 'Disabled'}\n\n")
            
            # Key metrics
            f.write("KEY METRICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Revision Count: {state.get('revision_count', 'N/A')}\n")
            f.write(f"Workflow Status: {state.get('workflow_status', 'N/A')}\n")
            f.write(f"Draft Version: {state.get('draft_version', 'N/A')}\n")
            f.write(f"Papers Processed: {len(state.get('papers_data', []))}\n")
            
            # Review scores if available
            if 'score_breadth' in state:
                f.write(f"Breadth Score: {state['score_breadth']:.3f}\n")
            if 'score_depth' in state:
                f.write(f"Depth Score: {state['score_depth']:.3f}\n")
            
            # Consistency metrics if available
            if 'consistency_report' in state:
                report = state['consistency_report']
                f.write(f"Concepts Tracked: {report.get('total_concepts', 'N/A')}\n")
                f.write(f"Terminology Mappings: {report.get('terminology_mappings', 'N/A')}\n")
            
            # Research mode specific info
            if state.get('research_mode') == 'local':
                f.write(f"Local Papers Used: Yes\n")
                f.write(f"Vector Store Path: {state.get('vector_store_path', 'N/A')}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("DETAILED STATE INFORMATION:\n")
            f.write("="*80 + "\n\n")
            
            for key, value in state.items():
                f.write(f"{key}:\n")
                f.write("-" * len(key) + "\n")
                
                if isinstance(value, str):
                    if len(value) > 200:
                        f.write(f"{value[:200]}...\n")
                    else:
                        f.write(f"{value}\n")
                elif isinstance(value, (list, dict)):
                    f.write(f"{type(value).__name__} with {len(value)} items\n")
                else:
                    f.write(f"{value} ({type(value).__name__})\n")
                
                f.write("\n")
        
        self.logger.info(f"Enhanced final state saved to: {filepath}")
    
    def save_workflow_summary(self, success: bool, research_mode: str = "unknown", error_msg: str = None):
        """Save enhanced workflow summary."""
        summary_file = self.output_dir / "workflow_summary.txt"
        
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("ENHANCED WORKFLOW EXECUTION SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Topic: {config.TOPIC}\n")
            f.write(f"Research Mode: {research_mode.upper()}\n")
            f.write(f"Status: {'SUCCESS' if success else 'FAILED'}\n")
            
            if error_msg:
                f.write(f"Error: {error_msg}\n")
            
            f.write(f"\nConfiguration:\n")
            f.write(f"- Model Provider: {config.MODEL_PROVIDER}\n")
            f.write(f"- Max Revisions: {config.MAX_REVISIONS}\n")
            f.write(f"- Embedding Model: {config.OLLAMA_EMBEDDING_MODEL}\n")
            f.write(f"- Research Mode Setting: {getattr(config, 'RESEARCH_MODE', 'default')}\n")
            f.write(f"- Consistency Management: {'Enabled' if getattr(config, 'ENABLE_CONSISTENCY_MANAGEMENT', False) else 'Disabled'}\n")
            
            # Check file existence
            files_created = []
            for file in self.output_dir.glob("*"):
                files_created.append(file.name)
            
            f.write(f"\nFiles Created: {len(files_created)}\n")
            for file in sorted(files_created):
                f.write(f"- {file}\n")


class EnhancedResearchRunner:
    """Enhanced runner with consistency management and research mode support."""
    
    def __init__(self):
        self.output = OutputManager()
        self.app = create_app()
        
    def run(self) -> bool:
        """
        Execute enhanced research workflow with comprehensive error handling.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get research mode (may prompt user if interactive)
            research_mode = config.get_research_mode()
            
            # Initialize state
            initial_state = {
                "topic": config.TOPIC,
                "revision_count": 0,
                "draft_history": [],
                "workflow_status": "initialized"
            }
            
            self.output.logger.info(f"Starting enhanced research workflow for topic: {config.TOPIC}")
            self.output.logger.info(f"Configuration: Provider={config.MODEL_PROVIDER}, Mode={research_mode}, MaxRevisions={config.MAX_REVISIONS}")
            self.output.logger.info(f"Consistency Management: {'Enabled' if getattr(config, 'ENABLE_CONSISTENCY_MANAGEMENT', False) else 'Disabled'}")
            
            # Execute workflow
            self.output.logger.info("Invoking enhanced workflow application...")
            final_state = self.app.invoke(initial_state)
            
            # Check status
            workflow_status = final_state.get("workflow_status", "unknown")
            if workflow_status == "error":
                error_msg = final_state.get("error", "Unknown error occurred")
                self.output.logger.error(f"Workflow failed: {error_msg}")
                self.output.save_workflow_summary(False, research_mode, error_msg)
                return False
            
            # Extract final paper
            final_paper = self._extract_paper(final_state)
            actual_research_mode = final_state.get("research_mode", research_mode)
            
            if final_paper:
                # Save paper
                filepath = self.output.save_paper(final_paper, actual_research_mode)
                
                # Display results
                self._display_results(filepath, final_paper, final_state, actual_research_mode)
                
                # Save state
                self.output.save_state(final_state)
                
                # Save summary
                self.output.save_workflow_summary(True, actual_research_mode)
                
                self.output.logger.info("Enhanced research workflow completed successfully")
                return True
            else:
                error_msg = "No paper content generated"
                self.output.logger.error(error_msg)
                self.output.save_workflow_summary(False, research_mode, error_msg)
                return False
                
        except Exception as e:
            error_msg = f"Critical workflow error: {e}"
            research_mode = config.get_research_mode() if hasattr(config, 'get_research_mode') else "unknown"
            self.output.logger.error(error_msg, exc_info=True)
            print(f"\n❌ Fatal error: {e}")
            self.output.save_workflow_summary(False, research_mode, str(e))
            return False
    
    def _extract_paper(self, state: Dict[str, Any]) -> Optional[str]:
        """Extract final paper from state with improved logic."""
        self.output.logger.debug("Extracting paper from final state...")
        
        # Try current draft first
        if "current_draft" in state and state["current_draft"]:
            content = state["current_draft"]
            self.output.logger.info(f"Extracted current draft ({len(content)} characters)")
            return content
        
        # Try draft history
        if "draft_history" in state and state["draft_history"]:
            last_draft = state["draft_history"][-1]
            
            if isinstance(last_draft, str):
                self.output.logger.info(f"Extracted from draft history ({len(last_draft)} characters)")
                return last_draft
            elif hasattr(last_draft, "content"):
                content = last_draft.content
                self.output.logger.info(f"Extracted from draft object ({len(content)} characters)")
                return content
        
        self.output.logger.warning("No paper content found in state")
        return None
    
    def _display_results(self, filepath: Path, content: str, state: Dict[str, Any], research_mode: str) -> None:
        """Display completion message with comprehensive information."""
        print("\n" + "="*80)
        print("✅ ENHANCED RESEARCH PAPER GENERATION COMPLETED")
        print("="*80)
        print(f"\nTopic: {config.TOPIC}")
        print(f"Research Mode: {research_mode.upper()}")
        print(f"Consistency Management: {'Enabled' if getattr(config, 'ENABLE_CONSISTENCY_MANAGEMENT', False) else 'Disabled'}")
        print(f"Paper saved to: {filepath}")
        print(f"Paper length: {len(content):,} characters")
        
        # Show research mode specific info
        papers_data = state.get('papers_data', [])
        if papers_data:
            papers_count = len(papers_data)
            local_papers = sum(1 for p in papers_data if p.get('is_local', False))
            
            print(f"Papers processed: {papers_count}")
            if research_mode == "local":
                print(f"Local PDF files used: {local_papers}")
            elif local_papers > 0:
                print(f"Mix of online ({papers_count - local_papers}) and local ({local_papers}) papers")
        
        # Show consistency metrics if available
        if 'consistency_report' in state:
            report = state['consistency_report']
            print(f"Concepts tracked: {report.get('total_concepts', 0)}")
            print(f"Terminology mappings: {report.get('terminology_mappings', 0)}")
        
        # Show revision info if available
        revision_count = state.get("revision_count", 0)
        if revision_count > 0:
            print(f"Revisions performed: {revision_count}")
        
        # Show review scores if available
        if 'score_breadth' in state:
            print(f"Breadth Score: {state['score_breadth']:.3f}")
        if 'score_depth' in state:
            print(f"Depth Score: {state['score_depth']:.3f}")
        
        print(f"\nPaper preview:")
        print("-" * 60)
        
        # Show first 300 characters
        preview = content[:300]
        if len(content) > 300:
            preview += "..."
        print(preview)
        
        print("-" * 60)
        print(f"\nAll outputs saved in: {self.output.output_dir}")


def display_welcome_banner():
    """Display enhanced welcome banner with new features."""
    print("\n" + "="*80)
    print("AI-POWERED RESEARCH PAPER GENERATION SYSTEM - ENHANCED v2.0")
    print("="*80)
    print("\n🚀 NEW FEATURES:")
    print("   • Advanced consistency management with concept dependency tracking")
    print("   • Intelligent outline alignment and logical flow enhancement") 
    print("   • Terminology normalization and standardization")
    print("   • Local PDF repository with GROBID integration")
    print("   • Enhanced RAG with contextual literature integration")
    print("   • Comprehensive quality analysis and reporting")


def main():
    """Enhanced main entry point with comprehensive setup and validation."""
    display_welcome_banner()
    
    # Validate system setup
    system_ready, status_messages = validate_system_setup()
    
    if not system_ready:
        print(f"\n❌ System validation failed. Please check the issues above.")
        print(f"📋 Check the logs in '{config.OUTPUT_DIR}' for details.")
        sys.exit(1)
    
    # Setup LangSmith if configured
    setup_langsmith()
    
    print(f"\n📊 Configuration Summary:")
    print(f"   • Provider: {config.MODEL_PROVIDER}")
    print(f"   • Embedding Model: {config.OLLAMA_EMBEDDING_MODEL}")
    print(f"   • Max Revisions: {config.MAX_REVISIONS}")
    print(f"   • Output Directory: {config.OUTPUT_DIR}")
    print(f"   • Research Mode: {getattr(config, 'RESEARCH_MODE', 'default')}")
    print(f"   • Consistency Management: {'Enabled' if getattr(config, 'ENABLE_CONSISTENCY_MANAGEMENT', False) else 'Disabled'}")
    print(f"   • Topic: {config.TOPIC[:60]}{'...' if len(config.TOPIC) > 60 else ''}")
    
    # Show local PDF info if available
    try:
        local_valid, local_msg = validate_local_setup()
        if local_valid:
            print(f"   • Local Papers: {local_msg}")
    except:
        pass
    
    print("\n🔄 Starting enhanced workflow execution...\n")
    
    try:
        runner = EnhancedResearchRunner()
        success = runner.run()
        
        if success:
            print(f"\n✅ SUCCESS! Check '{config.OUTPUT_DIR}' folder for all results.")
            print("📁 Generated files:")
            output_path = Path(config.OUTPUT_DIR)
            for file in sorted(output_path.glob("*")):
                size = file.stat().st_size if file.is_file() else 0
                size_str = f"({size:,} bytes)" if size > 0 else ""
                print(f"   • {file.name} {size_str}")
            
            # Show specific enhanced features used
            print(f"\n💡 Enhanced Features:")
            print(f"   • Paper includes comprehensive metadata and consistency analysis")
            print(f"   • Check 'consistency_report_*.md' for detailed quality metrics")
            print(f"   • Review 'references.txt' for source literature summary")
            print(f"   • Examine 'workflow.log' for detailed execution trace")
            
            if hasattr(config, 'LOCAL_PDF_DIR') and config.LOCAL_PDF_DIR:
                print(f"   • Add more PDFs to '{config.LOCAL_PDF_DIR}' for future runs")
            
            sys.exit(0)
        else:
            print(f"\n❌ FAILED! Check '{config.OUTPUT_DIR}/workflow.log' for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print(f"📋 Check '{config.OUTPUT_DIR}/workflow.log' for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()