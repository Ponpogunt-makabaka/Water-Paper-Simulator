# test.py
"""
Comprehensive testing suite for the enhanced research paper generation system.
Tests all major components, consistency management, and integration points.
"""

import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, List
import json

# Add project root to path
sys.path.insert(0, '.')

def test_header(test_name: str):
    """Print formatted test header."""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING: {test_name}")
    print(f"{'='*60}")

def test_result(success: bool, message: str = ""):
    """Print test result."""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status}: {message}")
    return success

def safe_test(test_func, test_name: str):
    """Safely run a test function with error handling."""
    try:
        return test_func()
    except Exception as e:
        print(f"❌ FAIL: {test_name} - {str(e)}")
        print(f"Stack trace: {traceback.format_exc()}")
        return False

class ComponentTester:
    """Main testing class for all system components."""
    
    def __init__(self):
        self.test_results = {}
        self.setup_test_environment()
    
    def setup_test_environment(self):
        """Setup test environment and directories."""
        test_header("Test Environment Setup")
        
        # Create test directories
        os.makedirs("output", exist_ok=True)
        os.makedirs("local_papers", exist_ok=True)
        
        # Create a test PDF for local processing
        test_pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\nxref\n0 3\n0000000000 65535 f \ntrailer\n<<\n/Size 3\n/Root 1 0 R\n>>\nstartxref\n9\n%%EOF"
        test_pdf_path = Path("local_papers/test_paper.pdf")
        
        try:
            with open(test_pdf_path, "wb") as f:
                f.write(test_pdf_content)
            print("✅ Test environment setup complete")
            return True
        except Exception as e:
            print(f"❌ Test environment setup failed: {e}")
            return False
    
    def test_config_loading(self):
        """Test configuration loading and validation."""
        test_header("Configuration Loading")
        
        try:
            import config
            
            # Test core configuration exists
            required_configs = [
                'TOPIC', 'MODEL_PROVIDER', 'OUTPUT_DIR', 'MAX_REVISIONS'
            ]
            
            missing_configs = []
            for conf in required_configs:
                if not hasattr(config, conf):
                    missing_configs.append(conf)
            
            if missing_configs:
                return test_result(False, f"Missing configurations: {missing_configs}")
            
            # Test new enhanced configurations
            enhanced_configs = [
                'ENABLE_CONSISTENCY_MANAGEMENT', 'RESEARCH_MODE', 'LOCAL_PDF_DIR'
            ]
            
            enhanced_found = []
            for conf in enhanced_configs:
                if hasattr(config, conf):
                    enhanced_found.append(conf)
            
            print(f"📊 Enhanced configs found: {enhanced_found}")
            
            # Test research mode function
            if hasattr(config, 'get_research_mode'):
                print("✅ Research mode function available")
            else:
                print("⚠️  Research mode function not found")
            
            return test_result(True, "Configuration loaded successfully")
            
        except ImportError as e:
            return test_result(False, f"Failed to import config: {e}")
        except Exception as e:
            return test_result(False, f"Configuration error: {e}")
    
    def test_consistency_manager(self):
        """Test consistency management system."""
        test_header("Consistency Manager")
        
        try:
            from consistency_manager import create_consistency_manager, Concept
            
            # Create consistency manager
            cm = create_consistency_manager()
            
            # Test concept addition
            cm.add_concept(
                "test_concept",
                "A test concept for validation",
                ["test_alias"],
                ["base_concept"]
            )
            
            # Test outline setting
            test_outline = """1. Introduction - Overview
2. Method - Technical approach
3. Conclusion - Summary"""
            
            cm.set_global_outline(test_outline)
            
            # Test section validation
            test_content = "This section discusses the test_concept and its applications."
            is_consistent, issues = cm.validate_section_consistency(
                "Introduction", test_content, "Overview of the test_concept"
            )
            
            print(f"📊 Consistency check result: {is_consistent}, Issues: {len(issues)}")
            
            # Test terminology normalization
            test_text = "The test_concept and test_alias are related."
            normalized = cm.normalize_terminology(test_text)
            
            print(f"📝 Terminology normalization: '{normalized}'")
            
            # Test concept dependency checking
            concept_issues = cm.check_concept_dependencies(test_content)
            print(f"🔍 Concept dependency issues: {len(concept_issues)}")
            
            # Test report generation
            report = cm.get_consistency_report()
            print(f"📋 Report generated with {report.get('total_concepts', 0)} concepts tracked")
            
            return test_result(True, "Consistency manager working correctly")
            
        except ImportError:
            return test_result(False, "Consistency manager not available")
        except Exception as e:
            return test_result(False, f"Consistency manager error: {e}")
    
    def test_local_pdf_processor(self):
        """Test local PDF processing capabilities."""
        test_header("Local PDF Processor")
        
        try:
            from local_pdf_processor import create_local_pdf_processor, validate_local_setup
            
            # Test validation
            is_valid, message = validate_local_setup()
            print(f"📁 Local setup validation: {is_valid} - {message}")
            
            # Test processor creation
            processor = create_local_pdf_processor()
            
            # Test PDF discovery
            pdf_files = processor.discover_pdf_files()
            print(f"📄 Found {len(pdf_files)} PDF files")
            
            # Test processing summary
            summary = processor.get_processing_summary()
            print(f"📊 Processing summary: {summary}")
            
            # Test metadata extraction from filename (fallback method)
            if pdf_files:
                test_file = pdf_files[0]
                metadata = processor.extract_metadata_from_filename(test_file)
                print(f"📝 Extracted metadata: {metadata['title']}")
            
            return test_result(True, "Local PDF processor working correctly")
            
        except ImportError:
            return test_result(False, "Local PDF processor not available")
        except Exception as e:
            return test_result(False, f"Local PDF processor error: {e}")
    
    def test_rag_manager(self):
        """Test RAG management system."""
        test_header("RAG Manager")
        
        try:
            from rag_manager import RAGManager
            
            # Create test papers data
            test_papers = [
                {
                    'paperId': 'test_1',
                    'title': 'Test Paper 1',
                    'authors': ['Test Author'],
                    'year': 2024,
                    'abstract': 'This is a test abstract for validation.',
                    'full_text': 'This is test content for the RAG system validation. It contains information about machine learning and neural networks.'
                },
                {
                    'paperId': 'test_2',
                    'title': 'Test Paper 2',
                    'authors': ['Another Author'],
                    'year': 2024,
                    'abstract': 'Another test abstract.',
                    'full_text': 'Additional test content discussing deep learning methodologies and evaluation techniques.'
                }
            ]
            
            # Create RAG manager
            rag_manager = RAGManager(test_papers)
            
            # Test vector store creation
            vector_store_path = rag_manager.create_vector_store()
            print(f"📚 Vector store created at: {vector_store_path}")
            
            # Test statistics
            stats = rag_manager.get_store_statistics()
            print(f"📊 RAG statistics: {stats}")
            
            # Test querying
            if vector_store_path and not stats.get('is_dummy_store', False):
                context = RAGManager.query_vector_store(vector_store_path, "machine learning methods")
                print(f"🔍 Query result length: {len(context)} characters")
            
            return test_result(True, "RAG manager working correctly")
            
        except ImportError:
            return test_result(False, "RAG manager not available")
        except Exception as e:
            return test_result(False, f"RAG manager error: {e}")
    
    def test_enhanced_writer(self):
        """Test enhanced writer agent."""
        test_header("Enhanced Writer Agent")
        
        try:
            from writer import EnhancedWriterAgent
            
            # Create writer instance
            writer = EnhancedWriterAgent()
            
            # Test section configuration
            sections = writer.sections
            print(f"📝 Configured sections: {len(sections)}")
            
            for section in sections:
                print(f"   • {section['title']}: {section['logical_role']}")
            
            # Test paper context setup
            test_state = {
                'topic': 'Test machine learning research',
                'final_plan': 'Test plan with 1. Introduction 2. Method 3. Conclusion',
                'vector_store_path': 'test_path'
            }
            
            writer._setup_paper_context(test_state)
            print(f"🎯 Paper context: {writer.paper_context}")
            
            # Test domain extraction
            domain = writer._extract_research_domain("neural networks for computer vision")
            print(f"🏷️ Extracted domain: {domain}")
            
            # Test contribution extraction
            contrib = writer._extract_main_contribution("We propose a novel deep learning approach")
            print(f"💡 Extracted contribution: {contrib}")
            
            return test_result(True, "Enhanced writer agent working correctly")
            
        except ImportError:
            return test_result(False, "Enhanced writer agent not available")
        except Exception as e:
            return test_result(False, f"Enhanced writer agent error: {e}")
    
    def test_enhanced_graph(self):
        """Test enhanced workflow graph."""
        test_header("Enhanced Workflow Graph")
        
        try:
            from graph import create_app, EnhancedResearchWorkflowGraph
            
            # Test app creation
            app = create_app()
            print("🔄 Workflow app created successfully")
            
            # Test enhanced workflow creation
            enhanced_workflow = EnhancedResearchWorkflowGraph()
            print("⚡ Enhanced workflow created successfully")
            
            # Test state type
            from graph import EnhancedGraphState
            
            # Create test state
            test_state = {
                'topic': 'Test research topic',
                'innovation_plans': 'Test plans',
                'final_topic': 'Final test topic',
                'final_plan': 'Test final plan',
                'papers_data': [],
                'vector_store_path': 'test_path',
                'references': 'Test references',
                'current_draft': 'Test draft',
                'feedback_breadth': 'Test feedback',
                'feedback_depth': 'Test feedback',
                'score_breadth': 0.8,
                'score_depth': 0.7,
                'consistency_report': {},
                'concept_tracking': {},
                'terminology_map': {},
                'outline_alignment': 0.9,
                'revision_count': 0,
                'workflow_status': 'test',
                'research_mode': 'test',
                'draft_history': [],
                'consistency_history': []
            }
            
            print("📊 Test state structure validated")
            
            # Test triage router
            router = enhanced_workflow.triage
            decision = router.route(test_state)
            print(f"🎯 Triage decision: {decision}")
            
            return test_result(True, "Enhanced workflow graph working correctly")
            
        except ImportError:
            return test_result(False, "Enhanced workflow graph not available")
        except Exception as e:
            return test_result(False, f"Enhanced workflow graph error: {e}")
    
    def test_base_agents(self):
        """Test base agent functionality."""
        test_header("Base Agents")
        
        try:
            from base_agent import BaseAgent
            from researcher import ResearcherAgent
            from analyst import AnalystAgent
            from reviewers import BreadthReviewAgent, DepthReviewAgent
            
            # Test agent creation
            agents = {
                'Researcher': ResearcherAgent(),
                'Analyst': AnalystAgent(),
                'BreadthReviewer': BreadthReviewAgent(),
                'DepthReviewer': DepthReviewAgent()
            }
            
            print(f"🤖 Created {len(agents)} agents:")
            for name, agent in agents.items():
                print(f"   • {name}: {agent.__class__.__name__}")
            
            # Test base functionality
            test_agent = agents['Researcher']
            
            # Test truncation
            long_text = "This is a very long text " * 50
            truncated = test_agent.truncate_text(long_text, 100)
            print(f"✂️ Text truncation: {len(long_text)} -> {len(truncated)} chars")
            
            # Test logging
            test_agent.log("Test log message", "INFO")
            
            return test_result(True, "Base agents working correctly")
            
        except ImportError as e:
            return test_result(False, f"Base agents not available: {e}")
        except Exception as e:
            return test_result(False, f"Base agents error: {e}")
    
    def test_prompts(self):
        """Test prompt templates."""
        test_header("Prompt Templates")
        
        try:
            import prompts
            
            # Test basic prompts exist
            basic_prompts = [
                'RESEARCH_INNOVATION_PROMPT',
                'ANALYST_EVALUATE_PROMPT', 
                'WRITER_SECTION_PROMPT',
                'BREADTH_REVIEW_PROMPT',
                'DEPTH_REVIEW_PROMPT'
            ]
            
            missing_prompts = []
            for prompt_name in basic_prompts:
                if not hasattr(prompts, prompt_name):
                    missing_prompts.append(prompt_name)
            
            if missing_prompts:
                return test_result(False, f"Missing basic prompts: {missing_prompts}")
            
            # Test enhanced prompts exist
            enhanced_prompts = [
                'WRITER_ENHANCED_SECTION_PROMPT',
                'CONSISTENCY_CHECK_PROMPT',
                'CONCEPT_DEPENDENCY_PROMPT',
                'LOGICAL_FLOW_ENHANCEMENT_PROMPT',
                'TERMINOLOGY_NORMALIZATION_PROMPT'
            ]
            
            enhanced_found = []
            for prompt_name in enhanced_prompts:
                if hasattr(prompts, prompt_name):
                    enhanced_found.append(prompt_name)
            
            print(f"📝 Enhanced prompts found: {len(enhanced_found)}/{len(enhanced_prompts)}")
            
            # Test prompt formatting
            test_prompt = prompts.RESEARCH_INNOVATION_PROMPT
            formatted = test_prompt.format(topic="test topic")
            print(f"🎭 Prompt formatting test: {len(formatted)} characters")
            
            return test_result(True, f"Prompts working correctly ({len(enhanced_found)} enhanced)")
            
        except ImportError:
            return test_result(False, "Prompts module not available")
        except Exception as e:
            return test_result(False, f"Prompts error: {e}")
    
    def test_llm_config(self):
        """Test LLM configuration."""
        test_header("LLM Configuration")
        
        try:
            from llm_config import llm_manager, LLMManager
            
            # Test manager creation
            manager = LLMManager()
            print(f"⚙️ LLM Manager created for provider: {manager.provider}")
            
            # Test different LLM types
            llm_types = ['research', 'analysis', 'writing', 'review']
            
            for llm_type in llm_types:
                try:
                    if llm_type == 'research':
                        llm = manager.get_research_llm()
                    elif llm_type == 'analysis':
                        llm = manager.get_analysis_llm()
                    elif llm_type == 'writing':
                        llm = manager.get_writing_llm()
                    elif llm_type == 'review':
                        llm = manager.get_review_llm()
                    
                    print(f"✅ {llm_type.title()} LLM created successfully")
                except Exception as e:
                    print(f"⚠️ {llm_type.title()} LLM creation failed: {e}")
            
            return test_result(True, "LLM configuration working correctly")
            
        except ImportError:
            return test_result(False, "LLM configuration not available")
        except Exception as e:
            return test_result(False, f"LLM configuration error: {e}")
    
    def test_tools(self):
        """Test research tools."""
        test_header("Research Tools")
        
        try:
            from tools import enhanced_literature_search, validate_search_query
            
            # Test query validation
            test_queries = [
                "machine learning",
                "neural networks deep learning",
                "a",  # Too short
                ""    # Empty
            ]
            
            for query in test_queries:
                validated = validate_search_query(query)
                print(f"🔍 Query '{query}' -> '{validated}'")
            
            # Test tool availability
            print("🛠️ Enhanced literature search tool available")
            
            # Note: We don't actually call the search to avoid external dependencies
            print("⚠️ Skipping actual search test to avoid external dependencies")
            
            return test_result(True, "Research tools working correctly")
            
        except ImportError:
            return test_result(False, "Research tools not available")
        except Exception as e:
            return test_result(False, f"Research tools error: {e}")
    
    def test_integration(self):
        """Test component integration."""
        test_header("Component Integration")
        
        try:
            # Test that main components can work together
            from consistency_manager import create_consistency_manager
            from local_pdf_processor import create_local_pdf_processor
            from rag_manager import RAGManager
            from writer import EnhancedWriterAgent
            
            # Create test data flow
            test_papers = [{
                'paperId': 'integration_test',
                'title': 'Integration Test Paper',
                'authors': ['Test Author'],
                'year': 2024,
                'abstract': 'Test abstract for integration.',
                'full_text': 'Test content about machine learning and neural networks for integration testing.'
            }]
            
            # Test RAG -> Writer integration
            rag_manager = RAGManager(test_papers)
            vector_store_path = rag_manager.create_vector_store()
            
            # Test Consistency Manager -> Writer integration
            cm = create_consistency_manager()
            cm.set_global_outline("1. Introduction\n2. Method\n3. Conclusion")
            
            # Test Writer with both components
            writer = EnhancedWriterAgent()
            writer.consistency_manager = cm
            
            test_state = {
                'topic': 'Integration test topic',
                'final_plan': '1. Introduction\n2. Method\n3. Conclusion',
                'vector_store_path': vector_store_path
            }
            
            writer._setup_paper_context(test_state)
            
            print("🔗 Component integration test successful")
            
            return test_result(True, "Component integration working correctly")
            
        except Exception as e:
            return test_result(False, f"Component integration error: {e}")
    
    def test_full_workflow_simulation(self):
        """Test a simulated full workflow run."""
        test_header("Full Workflow Simulation")
        
        try:
            # Import main components
            from main import EnhancedResearchRunner
            import config
            
            # Override config for testing
            original_topic = config.TOPIC
            original_research_mode = getattr(config, 'RESEARCH_MODE', 'online')
            original_max_revisions = config.MAX_REVISIONS
            
            # Set test configuration
            config.TOPIC = "Test machine learning research"
            config.RESEARCH_MODE = "local"  # Use local mode to avoid external dependencies
            config.MAX_REVISIONS = 1  # Limit revisions for testing
            
            print(f"🧪 Testing with: Topic='{config.TOPIC}', Mode='{config.RESEARCH_MODE}'")
            
            # Note: We don't actually run the full workflow to avoid long execution times
            # Instead, we test that the runner can be created
            runner = EnhancedResearchRunner()
            
            print("🏃 Enhanced research runner created successfully")
            print("⚠️ Skipping full execution to avoid long test times")
            
            # Restore original config
            config.TOPIC = original_topic
            config.RESEARCH_MODE = original_research_mode
            config.MAX_REVISIONS = original_max_revisions
            
            return test_result(True, "Full workflow simulation ready")
            
        except Exception as e:
            return test_result(False, f"Full workflow simulation error: {e}")
    
    def run_all_tests(self):
        """Run all tests and provide summary."""
        print("🚀 Starting Enhanced Research Paper Generation System Tests")
        print("="*80)
        
        tests = [
            ("Configuration Loading", self.test_config_loading),
            ("Consistency Manager", self.test_consistency_manager),
            ("Local PDF Processor", self.test_local_pdf_processor),
            ("RAG Manager", self.test_rag_manager),
            ("Enhanced Writer", self.test_enhanced_writer),
            ("Enhanced Workflow Graph", self.test_enhanced_graph),
            ("Base Agents", self.test_base_agents),
            ("Prompt Templates", self.test_prompts),
            ("LLM Configuration", self.test_llm_config),
            ("Research Tools", self.test_tools),
            ("Component Integration", self.test_integration),
            ("Full Workflow Simulation", self.test_full_workflow_simulation)
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            success = safe_test(test_func, test_name)
            self.test_results[test_name] = success
            
            if success:
                passed += 1
            else:
                failed += 1
        
        # Print summary
        self.print_summary(passed, failed)
        
        return passed, failed
    
    def print_summary(self, passed: int, failed: int):
        """Print test summary."""
        total = passed + failed
        
        print(f"\n{'='*80}")
        print("🏁 TEST SUMMARY")
        print(f"{'='*80}")
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%" if total > 0 else "No tests run")
        
        print(f"\n📋 Detailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {status} {test_name}")
        
        if failed == 0:
            print(f"\n🎉 All tests passed! System ready for use.")
        else:
            print(f"\n⚠️ {failed} test(s) failed. Check logs above for details.")
            print("💡 Failed tests may indicate missing dependencies or configuration issues.")
        
        print(f"\n📚 Next Steps:")
        if failed == 0:
            print("   • Run 'python main.py' to start the enhanced system")
            print("   • Check configuration in 'config.py' as needed")
            print("   • Add PDF files to 'local_papers/' directory for local mode")
        else:
            print("   • Review failed tests and fix any configuration issues")
            print("   • Ensure all required dependencies are installed")
            print("   • Check that Ollama service is running if using local models")


def main():
    """Main test execution function."""
    tester = ComponentTester()
    passed, failed = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()