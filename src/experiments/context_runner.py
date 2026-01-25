"""
Context Window Experiment Runner
Generates 3 variations of crops (Tight, Context, Full) for random images and runs the Functional strategy.
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

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "src" / "ingestion"))

from ingestion.pdf_extractor import PDFExtractor
from strategies import FunctionalStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def encode_image_bytes(img_bytes: bytes) -> bytes:
    return img_bytes

def run_context_experiment(pdf_path: Path, output_dir: Path, sample_size: int = 3):
    """
    Run the context window experiment.
    """
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return

    # Setup 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = output_dir / f"context_exp_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Extractor (to use its grouping logic)
    # We will use a temp dir for its normal output but we mainly want the object logic
    temp_extract_dir = exp_dir / "temp_extract"
    extractor = PDFExtractor(temp_extract_dir, clustering_method="dbscan")
    
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    # 1. Collect all candidate images first (Scan first 20 pages or so to save time, or random pages?)
    # To get a true random sample, we should probably scan the whole pdf or a large chunk.
    # Given the pdf size, let's scan all pages but only keep metadata first.
    
    logger.info(f"Scanning PDF for diagrams...")
    candidate_images = [] # List of (page_num, bbox, image_group_data)
    
    # scan pages (limit to first 20 for speed if needed, or all)
    pages_to_scan = range(min(len(doc), 50)) 
    
    for page_num in pages_to_scan:
        page = doc[page_num]
        
        # Get image blocks
        image_blocks = [b for b in page.get_text("dict")["blocks"] if b["type"] == 1]
        if not image_blocks:
            continue
            
        # Use DBSCAN to group them
        groups = extractor._group_blocks_dbscan(image_blocks)
        
        for group in groups:
            # Calculate tight bbox for the group
            bboxes = [g["bbox"] for g in group]
            tight_rect = extractor._merge_bboxes(bboxes)
            
            # Filter huge or tiny images
            if tight_rect.width < 50 or tight_rect.height < 50:
                 continue
            
            candidate_images.append({
                "page_num": page_num,
                "bbox": tight_rect,
                "group": group
            })
            
    logger.info(f"Found {len(candidate_images)} candidate diagrams.")
    
    # 2. Randomly Select Samples
    if len(candidate_images) > sample_size:
        selected_samples = random.sample(candidate_images, sample_size)
    else:
        selected_samples = candidate_images
        
    logger.info(f"Selected {len(selected_samples)} samples for testing.")
    
    # 3. Generate Variations & Run Strategy
    strategy = FunctionalStrategy()
    results_summary = []
    
    for idx, sample in enumerate(selected_samples, 1):
        page_num = sample["page_num"]
        tight_rect = sample["bbox"]
        page = doc[page_num]
        
        base_name = f"page_{page_num}_img_{idx}"
        logger.info(f"Processing Sample {idx}: Page {page_num}")
        
        # Variation A: Tight Crop
        # -----------------------
        pix_tight = page.get_pixmap(clip=tight_rect, matrix=fitz.Matrix(2, 2))
        tight_bytes = pix_tight.tobytes("png")
        
        # Variation B: Expanded Context (Standard)
        # ----------------------------------------
        # Logic from pdf_extractor._expand_bbox_for_context
        # We re-implement simplified version here or call it if possible.
        # Since it's a private method, we'll replicate the core logic for control.
        # Expand 100px up/down and full width
        page_rect = page.rect
        context_rect = fitz.Rect(
            25, # margin left
            max(0, tight_rect.y0 - 100),
            page_rect.width - 25, # margin right
            min(page_rect.height, tight_rect.y1 + 100)
        )
        pix_context = page.get_pixmap(clip=context_rect, matrix=fitz.Matrix(2, 2))
        context_bytes = pix_context.tobytes("png")
        
        # Variation C: Full Page
        # ----------------------
        pix_full = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        full_bytes = pix_full.tobytes("png")
        
        # Run Strategy on all 3
        variations = {
            "tight": tight_bytes,
            "context": context_bytes,
            "full_page": full_bytes
        }
        
        sample_results = {
            "image_id": base_name,
            "page": page_num,
            "variations": {}
        }
        
        for var_name, img_data in variations.items():
            # Save image for verification
            img_path = exp_dir / f"{base_name}_{var_name}.png"
            with open(img_path, "wb") as f:
                f.write(img_data)
                
            logger.info(f"  > Running strategy on {var_name} view...")
            
            try:
                result = strategy.generate(
                    str(img_path), 
                    img_data, 
                    "image/png"
                )
                
                # Save text output
                txt_path = exp_dir / f"{base_name}_{var_name}.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(result["output"])
                    
                sample_results["variations"][var_name] = {
                    "image_path": str(img_path),
                    "text_path": str(txt_path),
                    "output": result["output"],
                    "status": "success"
                }
            except Exception as e:
                logger.error(f"Failed on {var_name}: {e}")
                sample_results["variations"][var_name] = {"status": "failed", "error": str(e)}
                
        results_summary.append(sample_results)
        
    doc.close()
    
    # Save master summary for evaluator
    # We need to adapt the format to what evaluator.py expects OR update evaluator.
    # Evaluator expects: list of items with "image_name", "strategies": { "strat_name": ... }
    # Here "strategies" effectively map to "variations".
    
    eval_ready_summary = []
    for item in results_summary:
        # We treat each variation as a "strategy" output for the evaluator
        strategies_map = {}
        for var_name, var_data in item["variations"].items():
            strategies_map[var_name] = {
                "status": var_data["status"],
                "output_file": var_data.get("text_path"),
                "output_preview": var_data.get("output", "")[:50]
            }
            
        eval_ready_summary.append({
            "image_name": item["image_id"], # This will be the grouping key
            "strategies": strategies_map
        })
        
    with open(exp_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(eval_ready_summary, f, indent=2)
        
    logger.info(f"Experiment completed. Results in {exp_dir}")
    return exp_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", help="Path to PDF")
    parser.add_argument("--output", default="experiments/results", help="Output directory")
    parser.add_argument("--samples", type=int, default=3, help="Number of random samples")
    
    args = parser.parse_args()
    
    run_context_experiment(Path(args.pdf_path), Path(args.output), args.samples)
