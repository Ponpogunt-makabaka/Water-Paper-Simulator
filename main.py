# main.py
"""
Optimized main entry point for research paper generation system.
Implements efficient execution with minimal token usage.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import config
from graph import create_app


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


class OutputManager:
    """Manages output files and logging."""
    
    def __init__(self):
        self.output_dir = Path(config.OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)
        self.setup_logging()
        
    def setup_logging(self) -> None:
        """Configure minimal logging."""
        log_file = self.output_dir / "workflow.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def save_paper(self, content: str) -> Path:
        """Save final paper."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"paper_{timestamp}.txt"
        filepath = self.output_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        
        return filepath
    
    def save_state(self, state: Dict[str, Any]) -> None:
        """Save final state for debugging."""
        filepath = self.output_dir / "final_state.txt"
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("Final State Summary\n")
            f.write("="*50 + "\n\n")
            
            for key, value in state.items():
                f.write(f"{key}: ")
                
                if isinstance(value, str):
                    # Truncate long strings
                    if len(value) > 100:
                        f.write(f"{value[:100]}...\n")
                    else:
                        f.write(f"{value}\n")
                else:
                    f.write(f"{type(value).__name__}\n")
                
                f.write("-"*30 + "\n")


class ResearchRunner:
    """Main runner for research workflow."""
    
    def __init__(self):
        self.output = OutputManager()
        self.app = create_app()
        
    def run(self) -> bool:
        """
        Execute research workflow.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Initialize state
            initial_state = {
                "topic": config.TOPIC,
                "revision_count": 0,
                "draft_history": [],
                "workflow_status": "initialized"
            }
            
            self.output.logger.info(f"Starting research: {config.TOPIC}")
            
            # Execute workflow
            final_state = self.app.invoke(initial_state)
            
            # Check status
            if final_state.get("workflow_status") == "error":
                self.output.logger.error("Workflow failed")
                return False
            
            # Extract final paper
            final_paper = self._extract_paper(final_state)
            
            if final_paper:
                # Save paper
                filepath = self.output.save_paper(final_paper)
                
                # Display results
                self._display_results(filepath, final_paper)
                
                # Save state
                self.output.save_state(final_state)
                
                self.output.logger.info("Research completed successfully")
                return True
            else:
                self.output.logger.error("No paper generated")
                return False
                
        except Exception as e:
            self.output.logger.error(f"Critical error: {e}")
            print(f"\n❌ Fatal error: {e}")
            return False
    
    def _extract_paper(self, state: Dict[str, Any]) -> Optional[str]:
        """Extract final paper from state."""
        # Try current draft
        if "current_draft" in state:
            return state["current_draft"]
        
        # Try draft history
        if "draft_history" in state and state["draft_history"]:
            last_draft = state["draft_history"][-1]
            
            if isinstance(last_draft, str):
                return last_draft
            elif hasattr(last_draft, "content"):
                return last_draft.content
        
        return None
    
    def _display_results(self, filepath: Path, content: str) -> None:
        """Display completion message."""
        print("\n" + "="*60)
        print("✅ RESEARCH PAPER GENERATED")
        print("="*60)
        print(f"\nTopic: {config.TOPIC}")
        print(f"Saved to: {filepath}")
        print(f"\nPaper preview ({len(content)} chars total):")
        print("-"*40)
        
        # Show first 500 chars
        preview = content[:500]
        if len(content) > 500:
            preview += "..."
        print(preview)
        
        print("-"*40)


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("OPTIMIZED RESEARCH PAPER GENERATION SYSTEM")
    print("="*60)
    
    # Setup LangSmith if configured
    setup_langsmith()
    
    print(f"\nConfiguration:")
    print(f"- Provider: {config.MODEL_PROVIDER}")
    print(f"- Max revisions: {config.MAX_REVISIONS}")
    print(f"- Section length: ~{config.MAX_SECTION_LENGTH} chars")
    print(f"- Topic: {config.TOPIC[:50]}...")
    print("\nStarting workflow...\n")
    
    try:
        runner = ResearchRunner()
        success = runner.run()
        
        if success:
            print("\n✅ Success! Check output folder for results.")
            sys.exit(0)
        else:
            print("\n❌ Failed. Check logs for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()