"""
Configuration runner for PDF extraction pipeline with different clustering methods.
Allows easy testing of different image block grouping techniques.
"""

from pathlib import Path
from pdf_extractor import extract_pdf


CLUSTERING_METHODS = {
    "dbscan": {
        "name": "DBSCAN (Density-Based)",
        "description": "Automatically determines clusters, handles variable-density groupings, separates noise",
        "best_for": "Most PDFs with natural clustering patterns"
    },
    "hierarchical": {
        "name": "Hierarchical (Agglomerative)",
        "description": "Builds dendrogram, flexible for different layouts, supports multiple linkage methods",
        "best_for": "Documents with variable group sizes and shapes"
    },
    "proximity": {
        "name": "Proximity (Fixed-Distance)",
        "description": "Simple fixed-distance grouping, fast, intuitive",
        "best_for": "Quick extraction, simple layouts"
    },
    "connected_components": {
        "name": "Connected Components (Graph-Based)",
        "description": "Treats blocks as graph nodes, finds connected components via DFS, handles transitive relationships",
        "best_for": "Complex spatial relationships, chained proximity groupings"
    },
    "iou": {
        "name": "Overlap/IoU-Based",
        "description": "Groups blocks by intersection over union, uses union-find, effective for fragmented images",
        "best_for": "Fragmented blocks, overlapping images, variable-size groupings"
    }
}


def run_extraction(method: str = "hierarchical"):
    """
    Run PDF extraction with specified clustering method.
    
    Args:
        method: One of the methods in CLUSTERING_METHODS
    """
    if method not in CLUSTERING_METHODS:
        print(f"Error: Unknown clustering method '{method}'")
        print(f"Available methods: {', '.join(CLUSTERING_METHODS.keys())}")
        return
    
    method_info = CLUSTERING_METHODS[method]
    
    project_root = Path(__file__).parent.parent.parent
    pdf_path = project_root / "data" / "pdfs" / "User Manual_NR-BR307_BR347.pdf"
    output_dir = project_root / f"temp_extraction_{method}"
    
    # Check if PDF exists
    if not pdf_path.exists():
        print(f"Error: PDF file not found at {pdf_path}")
        return
    
    # Print method information
    print("\n" + "="*70)
    print(f"PDF EXTRACTION - {method_info['name'].upper()}")
    print("="*70)
    print(f"Method: {method_info['name']}")
    print(f"Description: {method_info['description']}")
    print(f"Best for: {method_info['best_for']}")
    print(f"Output: {output_dir.name}/")
    print("="*70 + "\n")
    
    # Run extraction
    try:
        summary = extract_pdf(pdf_path, output_dir, clustering_method=method)
        print("\n✓ Extraction completed successfully!")
        print(f"Results saved to: {output_dir}/")
        print(f"  - Text: {output_dir}/text/")
        print(f"  - Images: {output_dir}/images/")
        print(f"  - Summary: {output_dir}/extraction_summary.json")
    except Exception as e:
        print(f"\n✗ Extraction failed with error: {str(e)}")
        raise


def list_methods():
    """Print all available clustering methods with descriptions."""
    print("\n" + "="*70)
    print("AVAILABLE CLUSTERING METHODS")
    print("="*70)
    
    for i, (method_key, method_info) in enumerate(CLUSTERING_METHODS.items(), 1):
        print(f"\n{i}. {method_info['name'].upper()} ({method_key})")
        print(f"   Description: {method_info['description']}")
        print(f"   Best for: {method_info['best_for']}")
    
    print("\n" + "="*70 + "\n")


def main():
    """Main entry point - run with default or specified method."""
    import sys
    
    if len(sys.argv) > 1:
        method = sys.argv[1].lower()
        if method == "list":
            list_methods()
            return
        run_extraction(method)
    else:
        # Default to hierarchical if no method specified
        list_methods()
        print("Usage Examples:")
        print("  python run_extraction_config.py hierarchical")
        print("  python run_extraction_config.py dbscan")
        print("  python run_extraction_config.py connected_components")
        print("  python run_extraction_config.py iou")
        print("  python run_extraction_config.py proximity")
        print("  python run_extraction_config.py list")
        print()
        run_extraction("hierarchical")


if __name__ == "__main__":
    main()
