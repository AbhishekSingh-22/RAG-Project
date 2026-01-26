"""
Extraction Algorithm Experiment Runner
Compares different image clustering algorithms (DBSCAN, IoU, Hierarchical, Connected Components).
Output is compatible with evaluator.py to calculate average helpfulness per algorithm.
"""

import os
import random
import argparse
import fitz  # PyMuPDF
import logging
from pathlib import Path
from datetime import datetime
import sys
import json
import numpy as np

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "src" / "ingestion"))

from ingestion.pdf_extractor import PDFExtractor
from strategies import FunctionalStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_extraction_experiment(pdf_path: Path, output_dir: Path, sample_size: int = 3):
    """
    Run the extraction algorithm experiment.
    """
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return

    # Setup 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = output_dir / f"extract_exp_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Extractor
    # We use a temp dir but we will mostly use its internal methods
    temp_extract_dir = exp_dir / "temp_work"
    extractor = PDFExtractor(temp_extract_dir)
    
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    # Select pages to test
    # We want pages that actually have images.
    candidate_pages = []
    
    logger.info("Scanning for pages with images...")
    # Scan a subset to find candidates
    scan_limit = min(50, total_pages)
    for p_num in range(scan_limit):
        page = doc[p_num]
        images = [b for b in page.get_text("dict")["blocks"] if b["type"] == 1]
        if images:
            candidate_pages.append(p_num)
            
    if not candidate_pages:
        logger.error("No images found in scanned pages.")
        return

    # Select random samples
    if len(candidate_pages) > sample_size:
        selected_pages = sorted(random.sample(candidate_pages, sample_size))
    else:
        selected_pages = candidate_pages
        
    logger.info(f"Selected Pages for Test: {selected_pages}")
    
    algorithms = ["dbscan", "hierarchical", "connected_components", "iou"]
    strategy = FunctionalStrategy()
    
    # We will collect results in a flat list for the evaluator
    # Each item will represent ONE extracted image from ONE algorithm
    eval_summary = []
    
    for page_num in selected_pages:
        page = doc[page_num]
        image_blocks = [b for b in page.get_text("dict")["blocks"] if b["type"] == 1]
        
        logger.info(f"--- Processing Page {page_num} ({len(image_blocks)} raw blocks) ---")
        
        for algo in algorithms:
            logger.info(f"Running Algo: {algo}")
            
            try:
                # 1. Run Clustering
                groups = extractor._group_blocks_adaptive(image_blocks, method=algo)
                logger.info(f"  > Found {len(groups)} groups")
                
                # 2. Process each group
                for grp_idx, group in enumerate(groups, 1):
                    # basic bbox of group
                    bboxes = [g["bbox"] for g in group]
                    tight_rect = extractor._merge_bboxes(bboxes)
                    
                    # Filter tiny noise
                    if tight_rect.width < 50 or tight_rect.height < 50:
                        continue
                        
                    # 3. Apply Context Expansion (Winning technique from Phase 2)
                    # We use the internal method _expand_bbox_for_context or simulate it
                    # _expand_bbox_for_context(self, page, image_bbox, ...)
                    final_rect = extractor._expand_bbox_for_context(
                        page, 
                        [tight_rect.x0, tight_rect.y0, tight_rect.x1, tight_rect.y1]
                    )
                    
                    # 4. Extract Image
                    pix = page.get_pixmap(clip=final_rect, matrix=fitz.Matrix(2, 2))
                    img_data = pix.tobytes("png")
                    
                    image_id = f"p{page_num}_{algo}_img{grp_idx}"
                    image_filename = f"{image_id}.png"
                    image_path = exp_dir / image_filename
                    
                    # Save for verification
                    with open(image_path, "wb") as f:
                        f.write(img_data)
                        
                    # 5. Generate Description
                    try:
                        result = strategy.generate(
                            str(image_path),
                            img_data,
                            "image/png"
                        )
                        
                        txt_path = exp_dir / f"{image_id}.txt"
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(result["output"])
                            
                        # Add to summary
                        # We structure it so evaluator sees "algo" as the strategy name
                        # This allows valid comparison of averages
                        eval_summary.append({
                            "image_name": image_id,
                            "strategies": {
                                algo: {
                                    "status": "success",
                                    "output_file": str(txt_path),
                                    "output_preview": result["output"][:50]
                                }
                            }
                        })
                        
                    except Exception as e:
                        logger.error(f"Generation failed for {image_id}: {e}")
                        eval_summary.append({
                            "image_name": image_id,
                            "strategies": {
                                algo: {
                                    "status": "failed",
                                    "error": str(e)
                                }
                            }
                        })
                        
            except Exception as e:
                logger.error(f"Algorithm {algo} crashed on page {page_num}: {e}")
                
    doc.close()
    
    # Save combined summary
    with open(exp_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)
        
    logger.info(f"Experiment completed. Results in {exp_dir}")
    return exp_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", help="Path to PDF")
    parser.add_argument("--output", default="experiments/results", help="Output directory")
    parser.add_argument("--samples", type=int, default=3, help="Number of pages to sample")
    
    args = parser.parse_args()
    
    run_extraction_experiment(Path(args.pdf_path), Path(args.output), args.samples)
