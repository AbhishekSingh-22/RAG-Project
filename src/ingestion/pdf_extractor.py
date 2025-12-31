"""
PDF Ingestion Pipeline
Extracts text and images from PDF files and saves them to a temporary folder for review.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import pymupdf as fitz  # PyMuPDF
from tqdm import tqdm


class PDFExtractor:
    """Extracts text and images from PDF files."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize the PDF extractor.
        
        Args:
            output_dir: Directory to save extracted content
        """
        self.output_dir = output_dir
        self.text_dir = output_dir / "text"
        self.images_dir = output_dir / "images"
        self.summary = {
            "total_pages": 0,
            "pages_with_text": 0,
            "pages_without_text": 0,
            "total_images": 0,
            "images_extracted": 0,
            "images_failed": 0,
            "text_chars_extracted": 0,
            "skipped_pages": [],
            "errors": []
        }
        
        # Create output directories
        self.text_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_text(self, page: fitz.Page, page_num: int) -> Tuple[str, bool]:
        """
        Extract text from a PDF page.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)
            
        Returns:
            Tuple of (extracted_text, success_flag)
        """
        try:
            text = page.get_text()
            return text, True
        except Exception as e:
            error_msg = f"Error extracting text from page {page_num + 1}: {str(e)}"
            self.summary["errors"].append(error_msg)
            return "", False
    
    def _group_nearby_blocks(self, blocks: List[Dict], proximity_threshold: float = 20.0) -> List[List[Dict]]:
        """
        Group image blocks that are close together (likely parts of the same image).
        
        Args:
            blocks: List of image block dictionaries with 'bbox' key
            proximity_threshold: Maximum distance between blocks to consider them part of same image (in points)
            
        Returns:
            List of grouped blocks, where each group is a list of blocks
        """
        if not blocks:
            return []
        
        groups = []
        used_blocks = set()
        
        for i, block in enumerate(blocks):
            if i in used_blocks:
                continue
            
            bbox1 = block.get("bbox", [])
            if not bbox1 or len(bbox1) < 4:
                continue
            
            # Start a new group with this block
            group = [block]
            used_blocks.add(i)
            
            # Find all blocks that are close to any block in this group
            changed = True
            while changed:
                changed = False
                for j, other_block in enumerate(blocks):
                    if j in used_blocks or j == i:
                        continue
                    
                    bbox2 = other_block.get("bbox", [])
                    if not bbox2 or len(bbox2) < 4:
                        continue
                    
                    # Check if this block is close to any block in the current group
                    for group_block in group:
                        group_bbox = group_block.get("bbox", [])
                        if self._blocks_are_nearby(group_bbox, bbox2, proximity_threshold):
                            group.append(other_block)
                            used_blocks.add(j)
                            changed = True
                            break
            
            groups.append(group)
        
        return groups
    
    def _blocks_are_nearby(self, bbox1: List[float], bbox2: List[float], threshold: float) -> bool:
        """
        Check if two bounding boxes are nearby each other.
        
        Args:
            bbox1: First bounding box [x0, y0, x1, y1]
            bbox2: Second bounding box [x0, y0, x1, y1]
            threshold: Maximum distance threshold
            
        Returns:
            True if blocks are nearby
        """
        if len(bbox1) < 4 or len(bbox2) < 4:
            return False
        
        # Calculate centers
        center1_x = (bbox1[0] + bbox1[2]) / 2
        center1_y = (bbox1[1] + bbox1[3]) / 2
        center2_x = (bbox2[0] + bbox2[2]) / 2
        center2_y = (bbox2[1] + bbox2[3]) / 2
        
        # Check if bounding boxes overlap or are close
        # Check horizontal proximity
        horizontal_gap = max(0, max(bbox1[0], bbox2[0]) - min(bbox1[2], bbox2[2]))
        vertical_gap = max(0, max(bbox1[1], bbox2[1]) - min(bbox1[3], bbox2[3]))
        
        # If they overlap or gap is small, they're nearby
        if horizontal_gap <= threshold and vertical_gap <= threshold:
            return True
        
        # Also check center distance
        distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
        return distance <= threshold * 3  # More lenient for center distance
    
    def _merge_bboxes(self, bboxes: List[List[float]]) -> fitz.Rect:
        """
        Merge multiple bounding boxes into a single rectangle.
        
        Args:
            bboxes: List of bounding boxes [x0, y0, x1, y1]
            
        Returns:
            Merged rectangle
        """
        if not bboxes:
            return fitz.Rect(0, 0, 0, 0)
        
        min_x = min(bbox[0] for bbox in bboxes if len(bbox) >= 4)
        min_y = min(bbox[1] for bbox in bboxes if len(bbox) >= 4)
        max_x = max(bbox[2] for bbox in bboxes if len(bbox) >= 4)
        max_y = max(bbox[3] for bbox in bboxes if len(bbox) >= 4)
        
        return fitz.Rect(min_x, min_y, max_x, max_y)
    
    def extract_images(self, page: fitz.Page, page_num: int, pdf_name: str) -> List[Dict]:
        """
        Extract images from a PDF page using multiple methods.
        Groups nearby image blocks to extract complete images.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)
            pdf_name: Name of the PDF file (for naming)
            
        Returns:
            List of dictionaries with image metadata
        """
        images_info = []
        image_counter = 0
        extracted_xrefs = set()  # Track extracted images to avoid duplicates
        
        try:
            # Method 1: Extract images using get_images() - standard method
            image_list = page.get_images(full=True)
            
            for img in image_list:
                try:
                    xref = img[0]
                    if xref in extracted_xrefs:
                        continue  # Skip if already extracted
                    
                    base_image = self.doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Save image
                    image_counter += 1
                    image_filename = f"{pdf_name}_page_{page_num + 1:04d}_img_{image_counter:03d}.{image_ext}"
                    image_path = self.images_dir / image_filename
                    
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    images_info.append({
                        "filename": image_filename,
                        "path": str(image_path),
                        "size_bytes": len(image_bytes),
                        "format": image_ext,
                        "page": page_num + 1,
                        "index": image_counter,
                        "method": "get_images"
                    })
                    
                    extracted_xrefs.add(xref)
                    self.summary["images_extracted"] += 1
                    
                except Exception as e:
                    error_msg = f"Error extracting image from page {page_num + 1}: {str(e)}"
                    self.summary["errors"].append(error_msg)
                    self.summary["images_failed"] += 1
            
            # Method 2: Extract images from text dictionary blocks (type=1)
            # Group nearby blocks together to extract complete images
            try:
                text_dict = page.get_text("dict")
                image_blocks = [b for b in text_dict.get("blocks", []) if b.get("type") == 1]
                
                if image_blocks:
                    # Group nearby blocks together
                    block_groups = self._group_nearby_blocks(image_blocks, proximity_threshold=30.0)
                    
                    for group_idx, block_group in enumerate(block_groups):
                        try:
                            # Try to extract using xref first (if all blocks share same xref)
                            xrefs = [b.get("xref") for b in block_group if b.get("xref")]
                            unique_xrefs = list(set(xrefs))
                            
                            # If all blocks have the same xref, extract once
                            if len(unique_xrefs) == 1 and unique_xrefs[0] and unique_xrefs[0] not in extracted_xrefs:
                                try:
                                    xref = unique_xrefs[0]
                                    base_image = self.doc.extract_image(xref)
                                    image_bytes = base_image["image"]
                                    image_ext = base_image["ext"]
                                    
                                    image_counter += 1
                                    image_filename = f"{pdf_name}_page_{page_num + 1:04d}_img_{image_counter:03d}.{image_ext}"
                                    image_path = self.images_dir / image_filename
                                    
                                    with open(image_path, "wb") as img_file:
                                        img_file.write(image_bytes)
                                    
                                    images_info.append({
                                        "filename": image_filename,
                                        "path": str(image_path),
                                        "size_bytes": len(image_bytes),
                                        "format": image_ext,
                                        "page": page_num + 1,
                                        "index": image_counter,
                                        "method": "text_dict_block_xref_grouped",
                                        "blocks_merged": len(block_group)
                                    })
                                    
                                    extracted_xrefs.add(xref)
                                    self.summary["images_extracted"] += 1
                                    continue  # Successfully extracted, skip rendering
                                except:
                                    pass  # Fall back to rendering
                            
                            # Merge bounding boxes and render as single image
                            bboxes = [b.get("bbox", []) for b in block_group if b.get("bbox")]
                            if not bboxes:
                                continue
                            
                            merged_rect = self._merge_bboxes(bboxes)
                            
                            # Render the merged area as a single image
                            pix = page.get_pixmap(clip=merged_rect, matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                            
                            if pix and pix.n > 0:
                                image_counter += 1
                                image_ext = "png"  # Rendered images saved as PNG
                                image_filename = f"{pdf_name}_page_{page_num + 1:04d}_img_{image_counter:03d}.{image_ext}"
                                image_path = self.images_dir / image_filename
                                
                                pix.save(image_path)
                                image_bytes = pix.tobytes()
                                
                                images_info.append({
                                    "filename": image_filename,
                                    "path": str(image_path),
                                    "size_bytes": len(image_bytes),
                                    "format": image_ext,
                                    "page": page_num + 1,
                                    "index": image_counter,
                                    "method": "text_dict_block_rendered_grouped",
                                    "blocks_merged": len(block_group),
                                    "bbox": list(merged_rect)
                                })
                                
                                self.summary["images_extracted"] += 1
                                pix = None  # Free memory
                                
                        except Exception as e:
                            error_msg = f"Error extracting image group {group_idx + 1} from page {page_num + 1}: {str(e)}"
                            self.summary["errors"].append(error_msg)
                            self.summary["images_failed"] += 1
                        
            except Exception as e:
                # If text dict method fails, continue
                pass
            
            # Method 3: Extract vector graphics/drawings as images
            # Some PDFs use vector graphics instead of raster images (like diagrams, illustrations)
            # Only extract if no raster images were found to avoid duplicates
            if len(images_info) == 0:
                try:
                    drawings = page.get_drawings()
                    
                    if drawings:
                        # Filter out very small drawings (likely decorative elements like lines, borders)
                        # Keep drawings that are substantial enough to be meaningful images
                        significant_drawings = []
                        for d in drawings:
                            rect = d.get("rect")
                            if rect and rect.width > 20 and rect.height > 20:
                                significant_drawings.append(d)
                        
                        if significant_drawings:
                            # Group nearby drawings together (likely parts of same diagram)
                            drawing_groups = self._group_drawings(significant_drawings, proximity_threshold=50.0)
                            
                            for group_idx, drawing_group in enumerate(drawing_groups):
                                try:
                                    # Get bounding box for the drawing group
                                    drawing_rect = self._get_drawings_bbox(drawing_group)
                                    
                                    # Skip if too small (likely not a meaningful image)
                                    # Increased threshold to filter out decorative elements
                                    if drawing_rect.width < 30 or drawing_rect.height < 30:
                                        continue
                                    
                                    # Render the drawing area as an image
                                    pix = page.get_pixmap(clip=drawing_rect, matrix=fitz.Matrix(2, 2))
                                    
                                    if pix and pix.n > 0:
                                        image_counter += 1
                                        image_ext = "png"
                                        image_filename = f"{pdf_name}_page_{page_num + 1:04d}_img_{image_counter:03d}.{image_ext}"
                                        image_path = self.images_dir / image_filename
                                        
                                        pix.save(image_path)
                                        image_bytes = pix.tobytes()
                                        
                                        images_info.append({
                                            "filename": image_filename,
                                            "path": str(image_path),
                                            "size_bytes": len(image_bytes),
                                            "format": image_ext,
                                            "page": page_num + 1,
                                            "index": image_counter,
                                            "method": "vector_graphics",
                                            "drawings_merged": len(drawing_group),
                                            "bbox": list(drawing_rect)
                                        })
                                        
                                        self.summary["images_extracted"] += 1
                                        pix = None  # Free memory
                                        
                                except Exception as e:
                                    error_msg = f"Error extracting drawing group {group_idx + 1} from page {page_num + 1}: {str(e)}"
                                    self.summary["errors"].append(error_msg)
                                    self.summary["images_failed"] += 1
                                    
                except Exception as e:
                    # If drawings method fails, continue
                    pass
                    
        except Exception as e:
            error_msg = f"Error processing images on page {page_num + 1}: {str(e)}"
            self.summary["errors"].append(error_msg)
        
        return images_info
    
    def _group_drawings(self, drawings: List[Dict], proximity_threshold: float = 50.0) -> List[List[Dict]]:
        """
        Group drawings that are close together (likely parts of the same diagram).
        
        Args:
            drawings: List of drawing dictionaries with 'rect' key
            proximity_threshold: Maximum distance between drawings to consider them part of same group
            
        Returns:
            List of grouped drawings, where each group is a list of drawings
        """
        if not drawings:
            return []
        
        groups = []
        used_drawings = set()
        
        for i, drawing in enumerate(drawings):
            if i in used_drawings:
                continue
            
            rect1 = drawing.get("rect")
            if not rect1:
                continue
            
            # Start a new group with this drawing
            group = [drawing]
            used_drawings.add(i)
            
            # Find all drawings that are close to any drawing in this group
            changed = True
            while changed:
                changed = False
                for j, other_drawing in enumerate(drawings):
                    if j in used_drawings or j == i:
                        continue
                    
                    rect2 = other_drawing.get("rect")
                    if not rect2:
                        continue
                    
                    # Check if this drawing is close to any drawing in the current group
                    for group_drawing in group:
                        group_rect = group_drawing.get("rect")
                        if self._drawings_are_nearby(group_rect, rect2, proximity_threshold):
                            group.append(other_drawing)
                            used_drawings.add(j)
                            changed = True
                            break
            
            groups.append(group)
        
        return groups
    
    def _drawings_are_nearby(self, rect1: fitz.Rect, rect2: fitz.Rect, threshold: float) -> bool:
        """
        Check if two drawing rectangles are nearby each other.
        
        Args:
            rect1: First rectangle
            rect2: Second rectangle
            threshold: Maximum distance threshold
            
        Returns:
            True if drawings are nearby
        """
        # Check if rectangles overlap or are close
        horizontal_gap = max(0, max(rect1.x0, rect2.x0) - min(rect1.x1, rect2.x1))
        vertical_gap = max(0, max(rect1.y0, rect2.y0) - min(rect1.y1, rect2.y1))
        
        # If they overlap or gap is small, they're nearby
        return horizontal_gap <= threshold and vertical_gap <= threshold
    
    def _get_drawings_bbox(self, drawings: List[Dict]) -> fitz.Rect:
        """
        Get bounding box that encompasses all drawings in a group.
        
        Args:
            drawings: List of drawing dictionaries with 'rect' key
            
        Returns:
            Merged rectangle
        """
        if not drawings:
            return fitz.Rect(0, 0, 0, 0)
        
        rects = [d.get("rect") for d in drawings if d.get("rect")]
        if not rects:
            return fitz.Rect(0, 0, 0, 0)
        
        min_x = min(r.x0 for r in rects)
        min_y = min(r.y0 for r in rects)
        max_x = max(r.x1 for r in rects)
        max_y = max(r.y1 for r in rects)
        
        return fitz.Rect(min_x, min_y, max_x, max_y)
    
    def process_pdf(self, pdf_path: Path) -> Dict:
        """
        Process a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with extraction results
        """
        pdf_name = pdf_path.stem
        results = {
            "pdf_name": pdf_name,
            "pdf_path": str(pdf_path),
            "pages": []
        }
        
        try:
            self.doc = fitz.open(pdf_path)
            total_pages = len(self.doc)
            self.summary["total_pages"] = total_pages
            
            # Process each page with progress bar
            for page_num in tqdm(range(total_pages), desc=f"Processing {pdf_name}", unit="page"):
                page = self.doc[page_num]
                page_info = {
                    "page_number": page_num + 1,
                    "text": "",
                    "text_extracted": False,
                    "images": [],
                    "has_text": False,
                    "has_images": False
                }
                
                # Extract text
                text, text_success = self.extract_text(page, page_num)
                if text_success:
                    page_info["text"] = text
                    page_info["text_extracted"] = True
                    page_info["has_text"] = len(text.strip()) > 0
                    
                    if page_info["has_text"]:
                        self.summary["pages_with_text"] += 1
                        self.summary["text_chars_extracted"] += len(text)
                    else:
                        self.summary["pages_without_text"] += 1
                        self.summary["skipped_pages"].append({
                            "page": page_num + 1,
                            "reason": "No text content"
                        })
                else:
                    self.summary["pages_without_text"] += 1
                    self.summary["skipped_pages"].append({
                        "page": page_num + 1,
                        "reason": "Text extraction failed"
                    })
                
                # Extract images
                images_info = self.extract_images(page, page_num, pdf_name)
                page_info["images"] = images_info
                page_info["has_images"] = len(images_info) > 0
                self.summary["total_images"] += len(images_info)
                
                # Save text to file
                if page_info["has_text"]:
                    text_filename = f"{pdf_name}_page_{page_num + 1:04d}.txt"
                    text_path = self.text_dir / text_filename
                    with open(text_path, "w", encoding="utf-8") as text_file:
                        text_file.write(text)
                    page_info["text_file"] = text_filename
                
                results["pages"].append(page_info)
            
            self.doc.close()
            
        except Exception as e:
            error_msg = f"Error processing PDF {pdf_name}: {str(e)}"
            self.summary["errors"].append(error_msg)
            results["error"] = error_msg
        
        return results
    
    def generate_summary(self) -> Dict:
        """
        Generate a summary of the extraction process.
        
        Returns:
            Dictionary with summary information
        """
        summary = {
            "extraction_timestamp": datetime.now().isoformat(),
            "statistics": {
                "total_pages": self.summary["total_pages"],
                "pages_with_text": self.summary["pages_with_text"],
                "pages_without_text": self.summary["pages_without_text"],
                "total_images_found": self.summary["total_images"],
                "images_extracted": self.summary["images_extracted"],
                "images_failed": self.summary["images_failed"],
                "total_text_characters": self.summary["text_chars_extracted"],
                "average_text_per_page": (
                    self.summary["text_chars_extracted"] / self.summary["pages_with_text"]
                    if self.summary["pages_with_text"] > 0 else 0
                )
            },
            "skipped_pages": self.summary["skipped_pages"],
            "errors": self.summary["errors"],
            "output_directories": {
                "text": str(self.text_dir),
                "images": str(self.images_dir)
            }
        }
        
        return summary
    
    def save_summary(self, summary: Dict, results: Dict):
        """
        Save summary and detailed results to JSON files.
        
        Args:
            summary: Summary dictionary
            results: Detailed results dictionary
        """
        summary_path = self.output_dir / "extraction_summary.json"
        results_path = self.output_dir / "extraction_results.json"
        
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


def extract_pdf(pdf_path: Path, output_dir: Path = None) -> Dict:
    """
    Main function to extract text and images from a PDF.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Output directory (defaults to temp folder in project root)
        
    Returns:
        Summary dictionary
    """
    # Set default output directory
    if output_dir is None:
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "temp_extraction"
    
    # Create extractor
    extractor = PDFExtractor(output_dir)
    
    # Process PDF
    print(f"\n{'='*60}")
    print(f"Starting PDF Extraction")
    print(f"{'='*60}")
    print(f"PDF: {pdf_path.name}")
    print(f"Output Directory: {output_dir}")
    print(f"{'='*60}\n")
    
    results = extractor.process_pdf(pdf_path)
    
    # Generate and save summary
    summary = extractor.generate_summary()
    extractor.save_summary(summary, results)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total Pages Processed: {summary['statistics']['total_pages']}")
    print(f"Pages with Text: {summary['statistics']['pages_with_text']}")
    print(f"Pages without Text: {summary['statistics']['pages_without_text']}")
    print(f"Total Images Found: {summary['statistics']['total_images_found']}")
    print(f"Images Extracted: {summary['statistics']['images_extracted']}")
    print(f"Images Failed: {summary['statistics']['images_failed']}")
    print(f"Total Text Characters: {summary['statistics']['total_text_characters']:,}")
    print(f"Average Text per Page: {summary['statistics']['average_text_per_page']:.0f} chars")
    
    if summary['skipped_pages']:
        print(f"\nSkipped Pages: {len(summary['skipped_pages'])}")
        for skipped in summary['skipped_pages'][:10]:  # Show first 10
            print(f"  - Page {skipped['page']}: {skipped['reason']}")
        if len(summary['skipped_pages']) > 10:
            print(f"  ... and {len(summary['skipped_pages']) - 10} more")
    
    if summary['errors']:
        print(f"\nErrors Encountered: {len(summary['errors'])}")
        for error in summary['errors'][:10]:  # Show first 10
            print(f"  - {error}")
        if len(summary['errors']) > 10:
            print(f"  ... and {len(summary['errors']) - 10} more")
    
    print(f"\nOutput Directories:")
    print(f"  Text: {summary['output_directories']['text']}")
    print(f"  Images: {summary['output_directories']['images']}")
    print(f"\nDetailed results saved to:")
    print(f"  Summary: {output_dir / 'extraction_summary.json'}")
    print(f"  Results: {output_dir / 'extraction_results.json'}")
    print(f"{'='*60}\n")
    
    return summary


if __name__ == "__main__":
    # Example usage
    project_root = Path(__file__).parent.parent.parent
    pdf_path = project_root / "data" / "pdfs" / "HomeHawkApp_Users_Guide_CC1803YK9100_ENG.pdf"
    
    if not pdf_path.exists():
        print(f"Error: PDF file not found at {pdf_path}")
    else:
        extract_pdf(pdf_path)

