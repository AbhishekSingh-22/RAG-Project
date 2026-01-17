"""
Retrieval Quality Evaluation Script

Evaluates the RAG retrieval pipeline with test queries and computes quality metrics.
"""
import sys
import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import statistics

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ingestion')))

from langchain_core.documents import Document
from retrieval import Retriever, RetrievalConfig


@dataclass
class TestQuery:
    """A test query with expected relevance information."""
    query: str
    description: str
    expected_topics: List[str] = field(default_factory=list)  # Topics that should appear
    expected_keywords: List[str] = field(default_factory=list)  # Keywords that should be in results
    difficulty: str = "medium"  # easy, medium, hard


@dataclass
class QueryResult:
    """Result of evaluating a single query."""
    query: str
    num_results: int
    avg_score: float
    min_score: float
    max_score: float
    score_std: float
    keyword_hits: int
    keyword_total: int
    keyword_recall: float
    topic_hits: int
    topic_total: int
    topic_recall: float
    top_headings: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    timestamp: str
    collection_stats: dict
    num_queries: int
    avg_results_per_query: float
    avg_score: float
    avg_keyword_recall: float
    avg_topic_recall: float
    avg_execution_time_ms: float
    query_results: List[dict] = field(default_factory=list)


# Test queries covering different aspects of a product manual
TEST_QUERIES = [
    TestQuery(
        query="How do I set up the camera for the first time?",
        description="Initial setup process",
        expected_topics=["setup", "installation", "getting started"],
        expected_keywords=["setup", "install", "connect", "power", "app"],
        difficulty="easy"
    ),
    TestQuery(
        query="What do the LED lights mean?",
        description="LED indicator meanings",
        expected_topics=["LED", "indicator", "status"],
        expected_keywords=["LED", "light", "indicator", "status", "blinking", "color"],
        difficulty="easy"
    ),
    TestQuery(
        query="How to connect to WiFi network?",
        description="WiFi connection process",
        expected_topics=["WiFi", "network", "connection"],
        expected_keywords=["WiFi", "network", "connect", "SSID", "password", "wireless"],
        difficulty="easy"
    ),
    TestQuery(
        query="motion detection settings",
        description="Motion detection configuration",
        expected_topics=["motion", "detection", "settings"],
        expected_keywords=["motion", "detect", "sensitivity", "zone", "alert"],
        difficulty="medium"
    ),
    TestQuery(
        query="recording video to SD card",
        description="Local storage and recording",
        expected_topics=["recording", "storage", "SD card"],
        expected_keywords=["record", "video", "SD", "card", "storage", "save"],
        difficulty="medium"
    ),
    TestQuery(
        query="troubleshooting camera offline",
        description="Troubleshooting connectivity issues",
        expected_topics=["troubleshooting", "offline", "connection"],
        expected_keywords=["offline", "troubleshoot", "problem", "fix", "connect", "reset"],
        difficulty="medium"
    ),
    TestQuery(
        query="night vision mode",
        description="Night vision functionality",
        expected_topics=["night vision", "infrared", "low light"],
        expected_keywords=["night", "vision", "infrared", "IR", "dark", "light"],
        difficulty="medium"
    ),
    TestQuery(
        query="share access with family members",
        description="Multi-user access and sharing",
        expected_topics=["sharing", "access", "users"],
        expected_keywords=["share", "family", "user", "access", "invite", "permission"],
        difficulty="medium"
    ),
    TestQuery(
        query="factory reset device",
        description="Factory reset procedure",
        expected_topics=["reset", "factory", "restore"],
        expected_keywords=["reset", "factory", "restore", "default", "button"],
        difficulty="easy"
    ),
    TestQuery(
        query="voice commands OK Google Alexa",
        description="Voice assistant integration",
        expected_topics=["voice", "assistant", "smart home"],
        expected_keywords=["voice", "Google", "Alexa", "command", "smart"],
        difficulty="hard"
    ),
]


def calculate_keyword_recall(
    results: List[Tuple[Document, float]], 
    expected_keywords: List[str]
) -> Tuple[int, int, float]:
    """
    Calculate what fraction of expected keywords appear in retrieved results.
    
    Returns: (hits, total, recall_score)
    """
    if not expected_keywords:
        return 0, 0, 1.0
    
    # Combine all result content
    all_content = " ".join([
        doc.page_content.lower() + " " + 
        doc.metadata.get('Heading', '').lower() + " " +
        doc.metadata.get('SubHeading', '').lower()
        for doc, _ in results
    ])
    
    hits = sum(1 for kw in expected_keywords if kw.lower() in all_content)
    recall = hits / len(expected_keywords)
    
    return hits, len(expected_keywords), recall


def calculate_topic_recall(
    results: List[Tuple[Document, float]], 
    expected_topics: List[str]
) -> Tuple[int, int, float]:
    """
    Calculate what fraction of expected topics appear in retrieved results.
    
    Returns: (hits, total, recall_score)
    """
    if not expected_topics:
        return 0, 0, 1.0
    
    # Combine all headings and content
    all_content = " ".join([
        doc.page_content.lower() + " " + 
        doc.metadata.get('Heading', '').lower() + " " +
        doc.metadata.get('SubHeading', '').lower()
        for doc, _ in results
    ])
    
    hits = sum(1 for topic in expected_topics if topic.lower() in all_content)
    recall = hits / len(expected_topics)
    
    return hits, len(expected_topics), recall


def get_unique_headings(results: List[Tuple[Document, float]]) -> List[str]:
    """Extract unique headings from results."""
    headings = set()
    for doc, _ in results:
        heading = doc.metadata.get('Heading', '')
        if heading:
            headings.add(heading)
    return list(headings)[:5]  # Top 5 unique headings


def evaluate_query(
    retriever: Retriever, 
    test_query: TestQuery, 
    k: int = 5
) -> QueryResult:
    """Evaluate a single test query."""
    import time
    
    start_time = time.time()
    results = retriever.search(test_query.query, k=k)
    execution_time = (time.time() - start_time) * 1000  # ms
    
    if not results:
        return QueryResult(
            query=test_query.query,
            num_results=0,
            avg_score=0.0,
            min_score=0.0,
            max_score=0.0,
            score_std=0.0,
            keyword_hits=0,
            keyword_total=len(test_query.expected_keywords),
            keyword_recall=0.0,
            topic_hits=0,
            topic_total=len(test_query.expected_topics),
            topic_recall=0.0,
            top_headings=[],
            execution_time_ms=execution_time
        )
    
    scores = [score for _, score in results]
    kw_hits, kw_total, kw_recall = calculate_keyword_recall(results, test_query.expected_keywords)
    topic_hits, topic_total, topic_recall = calculate_topic_recall(results, test_query.expected_topics)
    
    return QueryResult(
        query=test_query.query,
        num_results=len(results),
        avg_score=statistics.mean(scores),
        min_score=min(scores),
        max_score=max(scores),
        score_std=statistics.stdev(scores) if len(scores) > 1 else 0.0,
        keyword_hits=kw_hits,
        keyword_total=kw_total,
        keyword_recall=kw_recall,
        topic_hits=topic_hits,
        topic_total=topic_total,
        topic_recall=topic_recall,
        top_headings=get_unique_headings(results),
        execution_time_ms=execution_time
    )


def run_evaluation(
    retriever: Retriever, 
    test_queries: List[TestQuery], 
    k: int = 5,
    verbose: bool = True
) -> EvaluationReport:
    """Run evaluation on all test queries."""
    
    if verbose:
        print(f"\n{'='*70}")
        print("RAG RETRIEVAL EVALUATION")
        print(f"{'='*70}")
    
    # Get collection stats
    stats = retriever.get_stats()
    if verbose:
        print(f"\nCollection: {stats['collection_name']}")
        print(f"Documents: {stats['document_count']}")
        print(f"Test Queries: {len(test_queries)}")
        print(f"Results per query: {k}")
        print(f"\n{'='*70}\n")
    
    # Evaluate each query
    query_results = []
    for i, tq in enumerate(test_queries, 1):
        if verbose:
            print(f"[{i}/{len(test_queries)}] Evaluating: \"{tq.query[:50]}...\"")
        
        result = evaluate_query(retriever, tq, k=k)
        query_results.append(result)
        
        if verbose:
            status = "✓" if result.keyword_recall >= 0.5 else "⚠"
            print(f"  {status} Score: {result.avg_score:.3f} | "
                  f"Keyword Recall: {result.keyword_recall:.1%} | "
                  f"Topic Recall: {result.topic_recall:.1%} | "
                  f"Time: {result.execution_time_ms:.1f}ms")
    
    # Aggregate metrics
    avg_results = statistics.mean([r.num_results for r in query_results])
    avg_score = statistics.mean([r.avg_score for r in query_results if r.num_results > 0])
    avg_kw_recall = statistics.mean([r.keyword_recall for r in query_results])
    avg_topic_recall = statistics.mean([r.topic_recall for r in query_results])
    avg_time = statistics.mean([r.execution_time_ms for r in query_results])
    
    report = EvaluationReport(
        timestamp=datetime.now().isoformat(),
        collection_stats=stats,
        num_queries=len(test_queries),
        avg_results_per_query=avg_results,
        avg_score=avg_score,
        avg_keyword_recall=avg_kw_recall,
        avg_topic_recall=avg_topic_recall,
        avg_execution_time_ms=avg_time,
        query_results=[asdict(r) for r in query_results]
    )
    
    return report


def print_summary(report: EvaluationReport) -> None:
    """Print evaluation summary."""
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Queries Evaluated: {report.num_queries}")
    print(f"Avg Results per Query: {report.avg_results_per_query:.1f}")
    print(f"Avg Similarity Score: {report.avg_score:.4f}")
    print(f"Avg Keyword Recall: {report.avg_keyword_recall:.1%}")
    print(f"Avg Topic Recall: {report.avg_topic_recall:.1%}")
    print(f"Avg Execution Time: {report.avg_execution_time_ms:.1f}ms")
    
    # Categorize results
    high_quality = sum(1 for r in report.query_results if r['keyword_recall'] >= 0.6)
    medium_quality = sum(1 for r in report.query_results if 0.3 <= r['keyword_recall'] < 0.6)
    low_quality = sum(1 for r in report.query_results if r['keyword_recall'] < 0.3)
    
    print(f"\nQuality Distribution:")
    print(f"  ✓ High (≥60% recall): {high_quality}/{report.num_queries}")
    print(f"  ⚠ Medium (30-60%): {medium_quality}/{report.num_queries}")
    print(f"  ✗ Low (<30%): {low_quality}/{report.num_queries}")
    
    # Show worst performing queries
    sorted_results = sorted(report.query_results, key=lambda x: x['keyword_recall'])
    print(f"\nLowest Performing Queries:")
    for r in sorted_results[:3]:
        print(f"  - \"{r['query'][:40]}...\" (recall: {r['keyword_recall']:.1%})")
    
    # Overall assessment
    print(f"\n{'='*70}")
    if report.avg_keyword_recall >= 0.6:
        print("✓ OVERALL: Good retrieval quality")
    elif report.avg_keyword_recall >= 0.4:
        print("⚠ OVERALL: Moderate retrieval quality - consider improvements")
    else:
        print("✗ OVERALL: Poor retrieval quality - needs attention")
    print(f"{'='*70}\n")


def save_report(report: EvaluationReport, output_path: Path) -> None:
    """Save evaluation report to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)
    print(f"Report saved to: {output_path}")


def main():
    """Main evaluation entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    parser.add_argument("-k", type=int, default=5, help="Number of results per query")
    parser.add_argument("-o", "--output", type=str, help="Output JSON file path")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--query", type=str, help="Run single custom query instead of test set")
    args = parser.parse_args()
    
    try:
        retriever = Retriever()
        
        if args.query:
            # Single query mode
            print(f"\nEvaluating single query: \"{args.query}\"")
            test_query = TestQuery(
                query=args.query,
                description="Custom query",
                expected_topics=[],
                expected_keywords=[]
            )
            result = evaluate_query(retriever, test_query, k=args.k)
            
            print(f"\nResults: {result.num_results}")
            print(f"Avg Score: {result.avg_score:.4f}")
            print(f"Score Range: {result.min_score:.4f} - {result.max_score:.4f}")
            print(f"Execution Time: {result.execution_time_ms:.1f}ms")
            print(f"Top Headings: {', '.join(result.top_headings)}")
            
            # Also show actual results
            results = retriever.search(args.query, k=args.k)
            print(f"\n--- Retrieved Chunks ---")
            for i, (doc, score) in enumerate(results, 1):
                print(f"\n[{i}] Score: {score:.4f}")
                print(f"    Heading: {doc.metadata.get('Heading', 'N/A')}")
                print(f"    Content: {doc.page_content[:200]}...")
            return
        
        # Full evaluation
        report = run_evaluation(
            retriever, 
            TEST_QUERIES, 
            k=args.k, 
            verbose=not args.quiet
        )
        
        print_summary(report)
        
        # Save report if output path specified
        if args.output:
            save_report(report, Path(args.output))
        else:
            # Default output path
            project_root = Path(__file__).parent.parent.parent
            output_path = project_root / "data" / "evaluation_report.json"
            save_report(report, output_path)
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
