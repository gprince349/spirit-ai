"""
Main Entry Point for RAG-based Chatbot

This module provides:
- Command-line interface for the chatbot
- Web API server startup
- Interactive chat sessions
- Document ingestion utilities
- System configuration and testing
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional
import asyncio
from datetime import datetime

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.llm_handler import RAGPipeline, LLMConfig
    from src.retriever import RetrievalConfig
    from src.api import run_api
    from ingest_documents import main as ingest_main
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChatbotCLI:
    """Command-line interface for the RAG chatbot"""
    
    def __init__(self):
        """Initialize the CLI"""
        self.rag_pipeline: Optional[RAGPipeline] = None
        self.session_history = []
    
    def initialize_pipeline(self) -> bool:
        """
        Initialize the RAG pipeline
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load configuration from environment or use defaults
            retrieval_config = RetrievalConfig(
                vector_store_type=os.getenv("VECTOR_STORE_TYPE", "faiss"),
                index_path=os.getenv("INDEX_PATH", "data/vector_index"),
                default_k=int(os.getenv("DEFAULT_K", "5"))
            )
            
            # Get LLM provider from environment
            llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
            
            # Configure max tokens based on provider
            if llm_provider == "gemini":
                max_tokens = int(os.getenv("GEMINI_MAX_TOKENS", "2000"))
            else:
                max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
            
            llm_config = LLMConfig(
                provider=llm_provider,
                # OpenAI Configuration
                api_key=os.getenv("OPENAI_API_KEY"),
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                # Gemini Configuration
                gemini_api_key=os.getenv("GEMINI_API_KEY"),
                gemini_model=os.getenv("GEMINI_MODEL", "gemini-pro"),
                # Common Configuration
                temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
                max_tokens=max_tokens,
                include_sources=True,
                citation_style="numbered"
            )
            
            # Check for required API key based on provider
            if llm_provider == "openai" and not llm_config.api_key:
                print("⚠️  Warning: No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
                print("   You can still test document ingestion and retrieval without LLM responses.")
                return False
            elif llm_provider == "gemini" and not llm_config.gemini_api_key:
                print("⚠️  Warning: No Gemini API key found. Set GEMINI_API_KEY environment variable.")
                print("   You can still test document ingestion and retrieval without LLM responses.")
                return False
            
            # Initialize pipeline
            print(f"🚀 Initializing RAG pipeline with {llm_provider.upper()} LLM...")
            self.rag_pipeline = RAGPipeline(retrieval_config, llm_config)
            print("✅ RAG pipeline initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize RAG pipeline: {e}")
            return False
    
    def interactive_chat(self):
        """Run interactive chat session"""
        if not self.rag_pipeline:
            print("❌ RAG pipeline not initialized. Cannot start chat session.")
            return
        
        print("\n🤖 RAG Chatbot Interactive Session")
        print("=" * 50)
        print("Ask questions about the ingested documents.")
        print("Type 'quit', 'exit', or 'bye' to end the session.")
        print("Type 'help' for available commands.")
        print("=" * 50)
        
        while True:
            try:
                # Get user input
                user_input = input("\n📝 You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye! Thanks for using the RAG chatbot.")
                    break
                
                # Check for help command
                if user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                # Check for special commands
                if user_input.lower() == 'history':
                    self.show_history()
                    continue
                
                if user_input.lower() == 'clear':
                    self.session_history.clear()
                    print("📝 Session history cleared.")
                    continue
                
                if user_input.lower() == 'stats':
                    self.show_stats()
                    continue
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Process query
                print("🤔 Thinking...")
                start_time = datetime.now()
                
                response = self.rag_pipeline.query(user_input, k=3)
                
                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds()
                
                # Display response
                print(f"\n🤖 Assistant: {response.response}")
                
                # Show sources if available
                if response.sources:
                    print(f"\n📚 Sources ({len(response.sources)} documents):")
                    for i, source in enumerate(response.sources, 1):
                        print(f"   {i}. {source.source_file} (Page {source.page_number})")
                
                print(f"\n⏱️  Response time: {response_time:.2f}s")
                
                # Add to session history
                self.session_history.append({
                    'timestamp': end_time,
                    'query': user_input,
                    'response': response.response,
                    'sources_count': len(response.sources),
                    'response_time': response_time
                })
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error processing query: {e}")
                print("Please try again with a different question.")
    
    def show_help(self):
        """Show help information"""
        print("""
🆘 Available Commands:
- help: Show this help message
- history: Show session history
- clear: Clear session history
- stats: Show pipeline statistics
- quit/exit/bye: End the chat session

💡 Tips:
- Ask specific questions about the documents
- Try different types of queries: questions, summaries, comparisons
- Use natural language - the AI understands context
        """)
    
    def show_history(self):
        """Show session history"""
        if not self.session_history:
            print("📝 No queries in session history.")
            return
        
        print(f"\n📚 Session History ({len(self.session_history)} queries):")
        print("-" * 60)
        
        for i, entry in enumerate(self.session_history, 1):
            timestamp = entry['timestamp'].strftime('%H:%M:%S')
            query_preview = entry['query'][:50] + "..." if len(entry['query']) > 50 else entry['query']
            print(f"{i}. [{timestamp}] {query_preview}")
            print(f"   Sources: {entry['sources_count']}, Time: {entry['response_time']:.2f}s")
    
    def show_stats(self):
        """Show pipeline statistics"""
        if not self.rag_pipeline:
            print("❌ Pipeline not initialized.")
            return
        
        print("\n📊 Pipeline Statistics:")
        print("-" * 30)
        print(f"Session queries: {len(self.session_history)}")
        
        if self.session_history:
            avg_response_time = sum(h['response_time'] for h in self.session_history) / len(self.session_history)
            total_sources = sum(h['sources_count'] for h in self.session_history)
            print(f"Average response time: {avg_response_time:.2f}s")
            print(f"Total sources retrieved: {total_sources}")
        
        # Vector store stats
        try:
            if hasattr(self.rag_pipeline.retriever.vector_store, 'get_stats'):
                stats = self.rag_pipeline.retriever.vector_store.get_stats()
                print(f"Vector store: {stats}")
        except Exception as e:
            print(f"Vector store stats unavailable: {e}")


def setup_environment():
    """Setup environment and check dependencies"""
    # Create necessary directories
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    (data_dir / "documents").mkdir(exist_ok=True)
    (data_dir / "uploads").mkdir(exist_ok=True)
    (data_dir / "vector_index").mkdir(exist_ok=True)
    
    # Load environment variables from .env file if it exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Environment variables loaded from .env file")
    except ImportError:
        print("ℹ️  python-dotenv not installed. Create .env file manually if needed.")
    except Exception as e:
        print(f"⚠️  Could not load .env file: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="RAG-based Chatbot - Document Q&A with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_app.py chat                    # Start interactive chat
  python main_app.py api                     # Start web API server
  python main_app.py ingest                  # Ingest documents
  python main_app.py ingest --rebuild        # Rebuild index from scratch
  python main_app.py test                    # Test the pipeline
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Start interactive chat session')
    
    # API command
    api_parser = subparsers.add_parser('api', help='Start web API server')
    api_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    api_parser.add_argument('--port', type=int, default=8000, help='Port to bind to (default: 8000)')
    api_parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest documents into vector database')
    ingest_parser.add_argument('--source', default='data/documents', 
                              help='Source directory for documents (default: data/documents)')
    ingest_parser.add_argument('--output', default='data/vector_index',
                              help='Output directory for vector index (default: data/vector_index)')
    ingest_parser.add_argument('--store-type', choices=['faiss', 'chromadb'], default='faiss',
                              help='Vector store type (default: faiss)')
    ingest_parser.add_argument('--rebuild', action='store_true',
                              help='Rebuild index from scratch')
    ingest_parser.add_argument('--chunk-size', type=int, default=1000,
                              help='Chunk size for document processing (default: 1000)')
    ingest_parser.add_argument('--chunk-overlap', type=int, default=200,
                              help='Chunk overlap for document processing (default: 200)')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test the RAG pipeline')
    test_parser.add_argument('--query', help='Test with specific query')
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Handle commands
    if args.command == 'chat':
        print("🤖 Starting RAG Chatbot CLI...")
        cli = ChatbotCLI()
        if cli.initialize_pipeline():
            cli.interactive_chat()
        else:
            print("❌ Cannot start chat without proper initialization.")
            print("💡 Try running 'python main.py ingest' first to set up the document index.")
    
    elif args.command == 'api':
        print(f"🌐 Starting RAG Chatbot API server on {args.host}:{args.port}...")
        try:
            run_api(host=args.host, port=args.port, reload=args.reload)
        except KeyboardInterrupt:
            print("\n👋 API server stopped.")
    
    elif args.command == 'ingest':
        print("📚 Starting document ingestion...")
        
        # Set environment variables for ingest script
        os.environ['SOURCE_DIR'] = args.source
        os.environ['OUTPUT_DIR'] = args.output
        os.environ['VECTOR_STORE_TYPE'] = args.store_type
        os.environ['CHUNK_SIZE'] = str(args.chunk_size)
        os.environ['CHUNK_OVERLAP'] = str(args.chunk_overlap)
        
        # Build arguments for ingest script USING its native parameter names
        # ingest_documents.py expects: --documents-dir, --index-path, --vector-store, --chunk-size, --overlap, --batch-size, --force-rebuild
        # Previous version passed different flag names causing "unrecognized arguments" error.
        ingest_args = [
            '--documents-dir', args.source,
            '--index-path', args.output,
            '--vector-store', args.store_type,
            '--chunk-size', str(args.chunk_size),
            '--overlap', str(args.chunk_overlap)
        ]

        if args.rebuild:
            ingest_args.append('--force-rebuild')
        
        # Run ingestion
        try:
            import sys
            original_argv = sys.argv
            sys.argv = ['ingest_documents.py'] + ingest_args
            ingest_main()
            sys.argv = original_argv
            print("✅ Document ingestion completed!")
        except Exception as e:
            print(f"❌ Ingestion failed: {e}")
    
    elif args.command == 'test':
        print("🧪 Testing RAG pipeline...")
        cli = ChatbotCLI()
        
        if cli.initialize_pipeline():
            if args.query:
                # Test with specific query
                print(f"Testing query: '{args.query}'")
                response = cli.rag_pipeline.query(args.query, k=3)
                print(f"Response: {response.response}")
                print(f"Sources: {len(response.sources)}")
                print(f"Response time: {response.response_time:.2f}s")
            else:
                # Run default tests
                test_queries = [
                    "What is artificial intelligence?",
                    "How is AI used in healthcare?",
                    "What are the main challenges in AI development?"
                ]
                
                for query in test_queries:
                    print(f"\n🔍 Testing: '{query}'")
                    try:
                        response = cli.rag_pipeline.query(query, k=2)
                        print(f"✅ Response generated ({response.response_time:.2f}s)")
                        print(f"   Sources: {len(response.sources)}")
                        print(f"   Response length: {len(response.response)} chars")
                    except Exception as e:
                        print(f"❌ Test failed: {e}")
                
                print("\n🧪 Testing completed!")
        else:
            print("❌ Cannot run tests without proper initialization.")
    
    else:
        # No command specified, show help
        print("🤖 RAG-based Chatbot")
        print("=" * 30)
        print("Available commands:")
        print("  python main.py chat    # Interactive chat session")
        print("  python main.py api     # Start web API server")
        print("  python main.py ingest  # Ingest documents")
        print("  python main.py test    # Test the pipeline")
        print("\nUse --help with any command for more options.")
        print("\n💡 Quick start:")
        print("  1. python main.py ingest    # Process documents")
        print("  2. Set OPENAI_API_KEY environment variable")
        print("  3. python main.py chat      # Start chatting!")


if __name__ == "__main__":
    main()
