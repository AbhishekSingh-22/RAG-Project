"""
Runner script for PDF extraction pipeline.
"""

from pathlib import Path
from pdf_extractor import extract_pdf


def main():
    """Main function to run the PDF extraction."""
    
    project_root = Path(__file__).parent.parent.parent
    
    # PDF file path
    pdf_path = project_root / "data" / "pdfs" / "Panasonic-Smart-AC-User-Manual.pdf"
    
    # Output directory
    output_dir = project_root / "temp_extraction"
    
    # Check if PDF exists
    if not pdf_path.exists():
        print(f"Error: PDF file not found at {pdf_path}")
        print(f"Please ensure the file exists.")
        return
    
    # Run extraction
    try:
        summary = extract_pdf(pdf_path, output_dir)
        print("\n✓ Extraction completed successfully!")
    except Exception as e:
        print(f"\n✗ Extraction failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()

