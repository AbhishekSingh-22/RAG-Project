"""
Experiment Runner
Executes defined extraction strategies on a set of images.
"""

import os
import argparse
import base64
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List

from strategies import VisualStrategy, FunctionalStrategy, JsonStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def encode_image(image_path: str) -> bytes:
    with open(image_path, "rb") as image_file:
        return image_file.read()

def get_mime_type(image_path: str) -> str:
    ext = Path(image_path).suffix.lower()
    if ext == ".png": return "image/png"
    if ext in [".jpg", ".jpeg"]: return "image/jpeg"
    if ext == ".webp": return "image/webp"
    return "image/png"

def run_experiments(input_dir: Path, output_dir: Path, limit: int = None):
    """
    Run all strategies on images in input_dir.
    """
    strategies = [
        VisualStrategy(),
        FunctionalStrategy(),
        JsonStrategy()
    ]
    
    # Setup output structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Get images
    formats = {".png", ".jpg", ".jpeg", ".webp"}
    images = sorted([
        f for f in input_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in formats
    ])
    
    if limit:
        images = images[:limit]
        
    logger.info(f"Found {len(images)} images. Running {len(strategies)} strategies on each.")
    
    results_summary = []
    
    for img_idx, image_path in enumerate(images, 1):
        logger.info(f"Processing [{img_idx}/{len(images)}]: {image_path.name}")
        
        try:
            image_data = encode_image(str(image_path))
            mime_type = get_mime_type(str(image_path))
            
            img_results = {
                "image_name": image_path.name,
                "strategies": {}
            }
            
            for strategy in strategies:
                try:
                    result = strategy.generate(str(image_path), image_data, mime_type)
                    
                    # Save individual result file
                    strategy_dir = run_dir / strategy.name
                    strategy_dir.mkdir(exist_ok=True)
                    
                    out_file = strategy_dir / f"{image_path.stem}.txt"
                    with open(out_file, "w", encoding="utf-8") as f:
                        f.write(result["output"])
                        
                    img_results["strategies"][strategy.name] = {
                        "status": "success",
                        "output_file": str(out_file),
                        "output_preview": result["output"][:100] + "..."
                    }
                except Exception as e:
                    logger.error(f"Strategy {strategy.name} failed on {image_path.name}: {e}")
                    img_results["strategies"][strategy.name] = {"status": "failed", "error": str(e)}
            
            results_summary.append(img_results)
            
        except Exception as e:
            logger.error(f"Failed to process image {image_path.name}: {e}")
            
    # Save master summary
    with open(run_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
        
    logger.info(f"Experiment completed. Results in {run_dir}")
    return run_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="temp_extraction/images", help="Input directory of images")
    parser.add_argument("--output", default="experiments/results", help="Base output directory")
    parser.add_argument("--limit", type=int, default=5, help="Limit number of images")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent.parent
    input_dir = project_root / args.input
    output_dir = project_root / args.output
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
    else:
        run_experiments(input_dir, output_dir, args.limit)
