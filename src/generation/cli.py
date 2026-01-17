#!/usr/bin/env python3
"""
RAG CLI Interface

Command-line interface for querying the RAG system.
Supports single queries, interactive mode, and batch processing.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Optional
from datetime import datetime

# Ensure proper imports - add parent directories to path
_current_dir = Path(__file__).parent
sys.path.insert(0, str(_current_dir))
sys.path.insert(0, str(_current_dir.parent / "ingestion"))
sys.path.insert(0, str(_current_dir.parent / "retrieval"))

# Use absolute imports for standalone execution
from config import GenerationConfig, RAGConfig, ResponseStyle
from llm_client import LLMClient, LLMResponse, LLMError
from prompts import RAGPromptBuilder, ContextChunk
from rag_chain import RAGChain, create_rag_chain, RAGResponse


def print_header():
    """Print CLI header."""
    print("\n" + "="*70)
    print("  📚 RAG Question Answering System")
    print("="*70)


def print_response(response: RAGResponse, show_sources: bool = True, verbose: bool = False):
    """
    Print a formatted response.
    
    Args:
        response: RAG response object
        show_sources: Whether to show sources
        verbose: Whether to show detailed metadata
    """
    if not response.success:
        print(f"\n❌ Error: {response.error}")
        return
    
    print("\n" + "-"*70)
    print("📝 Answer:")
    print("-"*70)
    print(response.answer)
    
    if show_sources and response.sources:
        print("\n" + "-"*70)
        print("📖 Sources:")
        print("-"*70)
        for i, source in enumerate(response.sources, 1):
            source_line = f"  [{i}] {source.file_name}"
            if source.page:
                source_line += f", Page {source.page}"
            if source.heading:
                source_line += f" — {source.heading}"
            print(source_line)
            if source.sub_heading:
                print(f"      └─ {source.sub_heading}")
    
    if verbose:
        print("\n" + "-"*70)
        print("📊 Metrics:")
        print("-"*70)
        print(f"  • Context chunks: {response.context_chunks}")
        print(f"  • Retrieval time: {response.retrieval_time_ms:.0f}ms")
        print(f"  • Generation time: {response.generation_time_ms:.0f}ms")
        print(f"  • Total time: {response.total_time_ms:.0f}ms")
        print(f"  • Model: {response.model}")
        if response.tokens_used:
            print(f"  • Tokens used: {response.tokens_used.get('total_tokens', 'N/A')}")
    
    print("-"*70 + "\n")


def interactive_mode(rag_chain: RAGChain, show_sources: bool = True, verbose: bool = False):
    """
    Run interactive question-answering session.
    
    Args:
        rag_chain: Initialized RAG chain
        show_sources: Whether to show sources
        verbose: Whether to show detailed metadata
    """
    print_header()
    print("\n💡 Interactive Mode")
    print("   Type your questions and press Enter.")
    print("   Commands: /clear (clear history), /stats, /help, /quit\n")
    
    while True:
        try:
            query = input("❓ You: ").strip()
            
            if not query:
                continue
            
            # Handle commands
            if query.startswith("/"):
                command = query.lower()
                
                if command in ("/quit", "/exit", "/q"):
                    print("\n👋 Goodbye!\n")
                    break
                    
                elif command == "/clear":
                    rag_chain.clear_history()
                    print("✓ Conversation history cleared.\n")
                    continue
                    
                elif command == "/stats":
                    stats = rag_chain.get_stats()
                    print(f"\n📊 System Stats:")
                    print(f"   Collection: {stats['retriever'].get('collection_name', 'N/A')}")
                    print(f"   Documents: {stats['retriever'].get('document_count', 'N/A')}")
                    print(f"   Model: {stats['generation'].get('model_name', 'N/A')}")
                    print(f"   History length: {stats['conversation_history_length']}\n")
                    continue
                    
                elif command == "/help":
                    print("\n📖 Commands:")
                    print("   /clear  - Clear conversation history")
                    print("   /stats  - Show system statistics")
                    print("   /help   - Show this help message")
                    print("   /quit   - Exit interactive mode\n")
                    continue
                    
                else:
                    print(f"❓ Unknown command: {query}. Type /help for available commands.\n")
                    continue
            
            # Process query
            print("\n⏳ Thinking...")
            response = rag_chain.query(query, include_history=True)
            print_response(response, show_sources=show_sources, verbose=verbose)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!\n")
            break
        except EOFError:
            print("\n\n👋 Goodbye!\n")
            break


def single_query_mode(
    rag_chain: RAGChain, 
    query: str, 
    show_sources: bool = True, 
    verbose: bool = False,
    output_json: Optional[str] = None,
):
    """
    Process a single query.
    
    Args:
        rag_chain: Initialized RAG chain
        query: Question to answer
        show_sources: Whether to show sources
        verbose: Whether to show detailed metadata
        output_json: Path to save JSON output
    """
    print_header()
    print(f"\n❓ Query: {query}\n")
    print("⏳ Processing...")
    
    response = rag_chain.query(query)
    
    if output_json:
        # Save JSON output
        output_path = Path(output_json)
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "response": response.to_dict(),
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"✓ JSON output saved to: {output_path}")
    
    print_response(response, show_sources=show_sources, verbose=verbose)


def batch_mode(
    rag_chain: RAGChain,
    queries_file: str,
    output_file: Optional[str] = None,
):
    """
    Process a batch of queries from a file.
    
    Args:
        rag_chain: Initialized RAG chain
        queries_file: Path to file with queries (one per line)
        output_file: Path to save JSON results
    """
    print_header()
    
    queries_path = Path(queries_file)
    if not queries_path.exists():
        print(f"❌ Error: Queries file not found: {queries_path}")
        return
    
    # Read queries
    with open(queries_path, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"\n📋 Processing {len(queries)} queries from: {queries_path}\n")
    
    results = []
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {query[:50]}...")
        response = rag_chain.query(query)
        results.append(response.to_dict())
        
        status = "✓" if response.success else "✗"
        print(f"   {status} Completed in {response.total_time_ms:.0f}ms")
    
    # Save results
    if output_file:
        output_path = Path(output_file)
    else:
        output_path = queries_path.parent / f"{queries_path.stem}_results.json"
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "queries_file": str(queries_path),
        "total_queries": len(queries),
        "successful": sum(1 for r in results if r["success"]),
        "results": results,
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_path}")
    print(f"   Successful: {output_data['successful']}/{output_data['total_queries']}\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Question Answering System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single query
  python cli.py "How do I set up the camera?"
  
  # Interactive mode
  python cli.py --interactive
  
  # Batch processing
  python cli.py --batch queries.txt --output results.json
  
  # Custom style
  python cli.py "What are the LED indicators?" --style technical --verbose
        """
    )
    
    # Query argument
    parser.add_argument(
        "query",
        nargs="?",
        help="Question to answer (omit for interactive mode)",
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    mode_group.add_argument(
        "-b", "--batch",
        metavar="FILE",
        help="Process queries from file (one per line)",
    )
    
    # Output options
    parser.add_argument(
        "-o", "--output",
        metavar="FILE",
        help="Save JSON output to file",
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Don't show source citations",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed metrics",
    )
    
    # Generation options
    parser.add_argument(
        "--style",
        choices=["concise", "detailed", "technical", "conversational"],
        default="detailed",
        help="Response style (default: detailed)",
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve (default: 10)",
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=5,
        help="Number of chunks after reranking (default: 5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="LLM temperature (default: 0.3)",
    )
    
    args = parser.parse_args()
    
    # Determine mode
    if not args.query and not args.interactive and not args.batch:
        # Default to interactive mode if no query provided
        args.interactive = True
    
    try:
        # Create RAG chain with configuration
        gen_config = GenerationConfig(
            response_style=ResponseStyle(args.style),
            temperature=args.temperature,
            include_sources=not args.no_sources,
        )
        rag_config = RAGConfig(
            retrieval_k=args.retrieval_k,
            rerank_top_n=args.rerank_top_n,
            generation=gen_config,
        )
        rag_chain = RAGChain(rag_config)
        
        # Execute appropriate mode
        if args.batch:
            batch_mode(rag_chain, args.batch, args.output)
        elif args.interactive:
            interactive_mode(
                rag_chain, 
                show_sources=not args.no_sources,
                verbose=args.verbose,
            )
        else:
            single_query_mode(
                rag_chain,
                args.query,
                show_sources=not args.no_sources,
                verbose=args.verbose,
                output_json=args.output,
            )
            
    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        print("   Make sure GOOGLE_API_KEY is set in your environment.\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
