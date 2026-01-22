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
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster


class PDFExtractor:
    """Extracts text and images from PDF files."""
    
    def __init__(self, output_dir: Path, clustering_method: str = "dbscan"):
        """
        Initialize the PDF extractor.
        
        Args:
            output_dir: Directory to save extracted content
            clustering_method: Image block clustering method ("dbscan", "hierarchical", or "proximity")
        """
        self.output_dir = output_dir
        self.text_dir = output_dir / "text"
        self.images_dir = output_dir / "images"
        self.clustering_method = clustering_method
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
    
    def _group_blocks_dbscan(self, blocks: List[Dict], eps: float = 30.0, min_samples: int = 1) -> List[List[Dict]]:
        """
        Group image blocks using DBSCAN clustering algorithm.
        
        DBSCAN automatically determines the number of clusters and handles variable-density
        groupings better than fixed proximity thresholds. It's especially effective for:
        - Detecting natural clusters in block positions
        - Handling variable-sized image groups
        - Separating noise (isolated blocks) from meaningful clusters
        
        Args:
            blocks: List of image block dictionaries with 'bbox' key
            eps: Maximum distance between blocks to be in same cluster (in points)
            min_samples: Minimum blocks required to form a cluster
            
        Returns:
            List of grouped blocks, where each group is a list of blocks
        """
        if not blocks:
            return []
        
        if len(blocks) <= 1:
            return [[b] for b in blocks]
        
        try:
            # Extract bbox centers for clustering
            centers = []
            for block in blocks:
                bbox = block.get("bbox", [])
                if bbox and len(bbox) >= 4:
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    centers.append([cx, cy])
                else:
                    # Invalid bbox, skip
                    continue
            
            if len(centers) <= 1:
                return [[b] for b in blocks if b.get("bbox")]
            
            # Apply DBSCAN clustering
            centers_array = np.array(centers)
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers_array)
            labels = clustering.labels_
            
            # Group blocks by cluster label
            # Note: label -1 means noise point (single isolated blocks)
            cluster_dict = {}
            valid_blocks = [b for b in blocks if b.get("bbox") and len(b.get("bbox")) >= 4]
            
            for block_idx, label in enumerate(labels):
                if label not in cluster_dict:
                    cluster_dict[label] = []
                cluster_dict[label].append(valid_blocks[block_idx])
            
            # Convert to list of groups (excluding noise as individual items)
            groups = []
            for label in sorted(cluster_dict.keys()):
                groups.append(cluster_dict[label])
            
            return groups
            
        except Exception as e:
            # Fallback to original method if DBSCAN fails
            return self._group_nearby_blocks(blocks, proximity_threshold=eps)
    
    def _group_blocks_hierarchical(self, blocks: List[Dict], distance_threshold: float = 100.0, 
                                      linkage_method: str = "average") -> List[List[Dict]]:
        """
        Group image blocks using Hierarchical (Agglomerative) Clustering.
        
        Hierarchical clustering builds a dendrogram of block relationships and cuts it at a
        specified distance threshold. It's effective for:
        - Variable-sized groups with flexible shapes
        - Documents where a distance threshold is easy to interpret
        - Getting stable, hierarchical relationships between blocks
        
        Args:
            blocks: List of image block dictionaries with 'bbox' key
            distance_threshold: Maximum distance for cutting the dendrogram (in points)
            linkage_method: Linkage criterion ("single", "complete", "average", "ward")
                - "single": Minimum distance between clusters (chaining effect)
                - "complete": Maximum distance between clusters (tight clusters)
                - "average": Average distance between clusters (balanced)
                - "ward": Minimizes within-cluster variance (compact clusters)
            
        Returns:
            List of grouped blocks, where each group is a list of blocks
        """
        if not blocks:
            return []
        
        if len(blocks) <= 1:
            return [[b] for b in blocks]
        
        try:
            # Extract bbox centers for clustering
            centers = []
            for block in blocks:
                bbox = block.get("bbox", [])
                if bbox and len(bbox) >= 4:
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    centers.append([cx, cy])
                else:
                    continue
            
            if len(centers) <= 1:
                return [[b] for b in blocks if b.get("bbox")]
            
            # Perform hierarchical clustering
            centers_array = np.array(centers)
            
            # Compute linkage matrix
            # For "ward" method, only Euclidean distance is supported
            # For other methods, we use Euclidean distance as well
            Z = linkage(centers_array, method=linkage_method, metric='euclidean')
            
            # Cut the dendrogram at the specified distance threshold
            labels = fcluster(Z, distance_threshold, criterion='distance') - 1  # Subtract 1 to start from 0
            
            # Group blocks by cluster label
            cluster_dict = {}
            valid_blocks = [b for b in blocks if b.get("bbox") and len(b.get("bbox")) >= 4]
            
            for block_idx, label in enumerate(labels):
                if label not in cluster_dict:
                    cluster_dict[label] = []
                cluster_dict[label].append(valid_blocks[block_idx])
            
            # Convert to list of groups
            groups = []
            for label in sorted(cluster_dict.keys()):
                groups.append(cluster_dict[label])
            
            return groups
            
        except Exception as e:
            # Fallback to proximity method if hierarchical clustering fails
            return self._group_nearby_blocks(blocks, proximity_threshold=distance_threshold)
    
    def _group_blocks_connected_components(self, blocks: List[Dict], distance_threshold: float = 50.0) -> List[List[Dict]]:
        """
        Group image blocks using Connected Components (Graph-Based) approach.
        
        Treats blocks as nodes in a graph and connects those within a distance threshold.
        Finds connected components using DFS. Effective for:
        - Detecting clusters of blocks that are transitively connected
        - Handling complex spatial relationships
        - Building natural groupings based on proximity chains
        
        Args:
            blocks: List of image block dictionaries with 'bbox' key
            distance_threshold: Maximum distance between blocks to consider them connected (in points)
            
        Returns:
            List of grouped blocks, where each group is a list of blocks
        """
        if not blocks:
            return []
        
        if len(blocks) <= 1:
            return [[b] for b in blocks]
        
        try:
            # Filter valid blocks
            valid_blocks = [b for b in blocks if b.get("bbox") and len(b.get("bbox")) >= 4]
            n = len(valid_blocks)
            
            if n <= 1:
                return [[b] for b in valid_blocks]
            
            # Build adjacency list (graph) - connect nearby blocks
            adjacency = [[] for _ in range(n)]
            
            for i in range(n):
                bbox_i = valid_blocks[i].get("bbox")
                for j in range(i + 1, n):
                    bbox_j = valid_blocks[j].get("bbox")
                    
                    # Check if blocks are nearby
                    if self._blocks_are_nearby(bbox_i, bbox_j, distance_threshold):
                        adjacency[i].append(j)
                        adjacency[j].append(i)
            
            # Find connected components using DFS
            visited = [False] * n
            components = []
            
            def dfs(node, component):
                """Depth-first search to find connected component"""
                visited[node] = True
                component.append(node)
                for neighbor in adjacency[node]:
                    if not visited[neighbor]:
                        dfs(neighbor, component)
            
            for i in range(n):
                if not visited[i]:
                    component = []
                    dfs(i, component)
                    groups = [valid_blocks[idx] for idx in component]
                    components.append(groups)
            
            return components
            
        except Exception as e:
            # Fallback to proximity method if connected components fails
            return self._group_nearby_blocks(blocks, proximity_threshold=distance_threshold)
    
    def _group_blocks_iou(self, blocks: List[Dict], iou_threshold: float = 0.1) -> List[List[Dict]]:
        """
        Group image blocks using Overlap/IoU (Intersection over Union) approach.
        
        Calculates IoU between bounding boxes and groups those with IoU >= threshold.
        Uses union-find to efficiently group overlapping regions. Effective for:
        - Merging overlapping or nearly-overlapping blocks
        - Detecting fragmented parts of the same image
        - Handling variable-size blocks with overlap patterns
        
        Args:
            blocks: List of image block dictionaries with 'bbox' key
            iou_threshold: Minimum IoU (0.0-1.0) to consider blocks as part of same group
                          0.0 = any overlap counts, 1.0 = complete overlap only
                          Default: 0.1 = 10% overlap threshold
            
        Returns:
            List of grouped blocks, where each group is a list of blocks
        """
        if not blocks:
            return []
        
        if len(blocks) <= 1:
            return [[b] for b in blocks]
        
        try:
            valid_blocks = [b for b in blocks if b.get("bbox") and len(b.get("bbox")) >= 4]
            n = len(valid_blocks)
            
            if n <= 1:
                return [[b] for b in valid_blocks]
            
            # Union-Find data structure for efficient grouping
            parent = list(range(n))
            
            def find(x):
                """Find root parent with path compression"""
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]
            
            def union(x, y):
                """Union two components"""
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py
            
            # Calculate IoU and union overlapping blocks
            for i in range(n):
                bbox_i = valid_blocks[i].get("bbox")
                rect_i = fitz.Rect(bbox_i)
                area_i = rect_i.width * rect_i.height
                
                for j in range(i + 1, n):
                    bbox_j = valid_blocks[j].get("bbox")
                    rect_j = fitz.Rect(bbox_j)
                    area_j = rect_j.width * rect_j.height
                    
                    # Calculate intersection
                    intersection = rect_i & rect_j
                    if intersection.is_empty:
                        inter_area = 0
                    else:
                        inter_area = intersection.width * intersection.height
                    
                    # Calculate union area
                    union_area = area_i + area_j - inter_area
                    
                    # Calculate IoU and union if threshold exceeded
                    if union_area > 0:
                        iou = inter_area / union_area
                        if iou >= iou_threshold:
                            union(i, j)
            
            # Group blocks by their root parent
            groups_dict = {}
            for i in range(n):
                root = find(i)
                if root not in groups_dict:
                    groups_dict[root] = []
                groups_dict[root].append(valid_blocks[i])
            
            # Convert to list of groups
            groups = list(groups_dict.values())
            return groups
            
        except Exception as e:
            # Fallback to proximity method if IoU grouping fails
            return self._group_nearby_blocks(blocks, proximity_threshold=20.0)
    
    def _group_blocks_adaptive(self, blocks: List[Dict], method: str = "dbscan") -> List[List[Dict]]:
        """
        Group blocks using specified method. Allows switching between algorithms.
        
        Args:
            blocks: List of image block dictionaries with 'bbox' key
            method: Clustering method to use - one of:
                - "dbscan": DBSCAN density-based clustering (default, good for noise handling)
                - "hierarchical": Agglomerative hierarchical clustering (good for variable group sizes)
                - "proximity": Simple fixed-distance proximity grouping (fast, simple)
                - "connected_components": Graph-based connected components (good for transitive relationships)
                - "iou": Overlap/IoU-based grouping (good for fragmented/overlapping blocks)
            
        Returns:
            List of grouped blocks
        """
        if method == "dbscan":
            return self._group_blocks_dbscan(blocks, eps=30.0, min_samples=1)
        elif method == "hierarchical":
            return self._group_blocks_hierarchical(blocks, distance_threshold=100.0, linkage_method="average")
        elif method == "proximity":
            return self._group_nearby_blocks(blocks, proximity_threshold=20.0)
        elif method == "connected_components":
            return self._group_blocks_connected_components(blocks, distance_threshold=50.0)
        elif method == "iou":
            return self._group_blocks_iou(blocks, iou_threshold=0.1)
        else:
            # Default to DBSCAN
            return self._group_blocks_dbscan(blocks, eps=30.0, min_samples=1)
    
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
    
    def _get_page_drawings_bbox(self, page: fitz.Page) -> fitz.Rect:
        """
        Get the bounding box that encompasses all vector drawings on the page.
        This helps identify annotation areas like pointer lines, callouts, etc.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Bounding box of all drawings, or empty rect if no drawings
        """
        try:
            drawings = page.get_drawings()
            if not drawings:
                return fitz.Rect()
            
            all_rects = [d.get("rect") for d in drawings if d.get("rect")]
            if not all_rects:
                return fitz.Rect()
            
            min_x = min(r.x0 for r in all_rects)
            min_y = min(r.y0 for r in all_rects)
            max_x = max(r.x1 for r in all_rects)
            max_y = max(r.y1 for r in all_rects)
            
            return fitz.Rect(min_x, min_y, max_x, max_y)
        except:
            return fitz.Rect()
    
    def _has_overlaid_annotations(self, page: fitz.Page, image_bbox: List[float]) -> bool:
        """
        Check if there are vector drawings (annotations) near/overlapping an image.
        
        Args:
            page: PyMuPDF page object
            image_bbox: Bounding box of the image [x0, y0, x1, y1]
            
        Returns:
            True if annotations exist near the image
        """
        try:
            drawings = page.get_drawings()
            if not drawings:
                return False
            
            img_rect = fitz.Rect(image_bbox)
            
            for d in drawings:
                d_rect = d.get("rect")
                if d_rect:
                    # Check if drawing is near the image (within 150 points)
                    # This catches pointer lines extending from the image
                    expanded_img = img_rect + fitz.Rect(-20, -20, 150, 20)  # Expand right for callouts
                    if expanded_img.intersects(d_rect):
                        return True
            
            return False
        except:
            return False
    
    def _is_callout_label(self, text: str) -> bool:
        """
        Check if text is likely a callout label (A, B, C, 1, 2, 3, ①, ②, ③, etc.)
        Also detects callout labels at the START of description text (e.g., "AReturns you...")
        
        Args:
            text: Text content to check
            
        Returns:
            True if text appears to be a callout label or starts with one
        """
        text = text.strip()
        if not text:
            return False
        
        # Single letter (A-Z, a-z)
        if len(text) == 1 and text.isalpha():
            return True
        
        # Single digit or small number (1-99)
        if text.isdigit() and len(text) <= 2:
            return True
        
        # Circled numbers ①②③④⑤⑥⑦⑧⑨⑩ etc.
        circled_chars = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳❶❷❸❹❺❻❼❽❾❿"
        if len(text) == 1 and text in circled_chars:
            return True
        
        # Text starting with circled number
        if text and text[0] in circled_chars:
            return True
        
        # Letter with period (A., B., 1., 2.)
        if len(text) == 2 and text[1] == '.' and (text[0].isalpha() or text[0].isdigit()):
            return True
        
        # Parenthesized letters/numbers like (A), (1), [A], [1]
        if len(text) <= 3 and (text.startswith('(') or text.startswith('[')):
            return True
        
        # Text that STARTS with a single uppercase letter followed by description
        # Pattern: "A" + description (like "AReturns you to the home screen")
        # This is common in technical documentation callouts
        if len(text) >= 2:
            first_char = text[0]
            second_char = text[1]
            # Single uppercase letter followed by uppercase letter (start of word)
            # e.g., "AReturns", "BDisplays", "CShows"
            if first_char.isupper() and second_char.isupper():
                return True
            # Single uppercase letter followed by lowercase (description continues)
            # e.g., "Ashows", but less common
        
        return False
    
    def _get_combined_annotation_bbox(self, page: fitz.Page, image_bbox: List[float]) -> fitz.Rect:
        """
        Get combined bounding box of image and all related annotations.
        Includes both vector drawings (pointer lines) AND text callout labels.
        
        Uses a two-pass approach:
        1. First find all connected drawings to expand the base area
        2. Then search for text callouts in the expanded area
        
        Args:
            page: PyMuPDF page object
            image_bbox: Bounding box of the image [x0, y0, x1, y1]
            
        Returns:
            Combined bounding box including image and annotations
        """
        try:
            img_rect = fitz.Rect(image_bbox)
            connected_rects = [img_rect]
            
            # PASS 1: Find all connected drawings (pointer lines, etc.)
            # Use generous expansion to find drawings that extend from the image
            search_area = img_rect + fitz.Rect(-50, -50, 200, 150)
            
            drawings = page.get_drawings()
            if drawings:
                for d in drawings:
                    d_rect = d.get("rect")
                    if d_rect and search_area.intersects(d_rect):
                        connected_rects.append(d_rect)
            
            # Calculate intermediate combined rect after adding drawings
            min_x = min(r.x0 for r in connected_rects)
            min_y = min(r.y0 for r in connected_rects)
            max_x = max(r.x1 for r in connected_rects)
            max_y = max(r.y1 for r in connected_rects)
            drawings_combined_rect = fitz.Rect(min_x, min_y, max_x, max_y)
            
            # PASS 2: Search for text callouts using the EXPANDED area from drawings
            # This ensures we find callouts at the end of pointer lines that extend
            # far from the original image
            expanded_search_area = drawings_combined_rect + fitz.Rect(-50, -50, 200, 150)
            
            try:
                text_dict = page.get_text("dict")
                for block in text_dict.get("blocks", []):
                    if block.get("type") == 0:  # Text block
                        block_bbox = block.get("bbox", [])
                        if len(block_bbox) >= 4:
                            block_rect = fitz.Rect(block_bbox)
                            
                            # Check if this text block is near the expanded area
                            if expanded_search_area.intersects(block_rect):
                                # Get the text content
                                text = ""
                                for line in block.get("lines", []):
                                    for span in line.get("spans", []):
                                        text += span.get("text", "")
                                
                                # If it's a callout label, include it
                                if self._is_callout_label(text):
                                    connected_rects.append(block_rect)
            except:
                pass
            
            # Combine all rectangles (image + drawings + text callouts)
            min_x = min(r.x0 for r in connected_rects)
            min_y = min(r.y0 for r in connected_rects)
            max_x = max(r.x1 for r in connected_rects)
            max_y = max(r.y1 for r in connected_rects)
            
            return fitz.Rect(min_x, min_y, max_x, max_y)
        except:
            return fitz.Rect(image_bbox)
    
    def _expand_bbox_for_context(self, page: fitz.Page, image_bbox: List[float], 
                                  margin_x: float = 25.0, 
                                  context_above: float = 100.0,
                                  context_below: float = 100.0) -> fitz.Rect:
        """
        Expand the image bounding box to include contextual content for caption generation.
        
        This method:
        1. Extends to full page width (with margins)
        2. Expands upward to include section headers and intro text
        3. Expands downward to include descriptions and captions
        
        Args:
            page: PyMuPDF page object
            image_bbox: Current image bounding box [x0, y0, x1, y1]
            margin_x: Horizontal margin from page edges (default 25 points)
            context_above: Max points to expand above for context (default 80)
            context_below: Max points to expand below for context (default 50)
            
        Returns:
            Expanded bounding box with contextual content
        """
        try:
            page_rect = page.rect
            img_rect = fitz.Rect(image_bbox)
            
            # Step 1: Extend to full page width (with margins)
            new_x0 = margin_x
            new_x1 = page_rect.width - margin_x
            
            # Step 2: Find contextual text above the image
            # Look for text blocks that could be section headers or intro text
            new_y0 = img_rect.y0
            new_y1 = img_rect.y1  # Initialize here to avoid unbound variable
            
            try:
                text_dict = page.get_text("dict")
                blocks = text_dict.get("blocks", [])
                
                # Find text blocks above the image
                blocks_above = []
                for block in blocks:
                    if block.get("type") == 0:  # Text block
                        block_bbox = block.get("bbox", [])
                        if len(block_bbox) >= 4:
                            block_y1 = block_bbox[3]  # Bottom of text block
                            block_y0 = block_bbox[1]  # Top of text block
                            
                            # Block is above the image and within context range
                            if block_y1 <= img_rect.y0 and block_y0 >= (img_rect.y0 - context_above):
                                # Get text to check if it's meaningful context
                                text = ""
                                for line in block.get("lines", []):
                                    for span in line.get("spans", []):
                                        text += span.get("text", "")
                                
                                text = text.strip()
                                # Include if it's substantial text (not just a callout label)
                                if len(text) > 5 and not self._is_callout_label(text):
                                    blocks_above.append((block_y0, block_y1, text))
                
                # Expand to include contextual blocks above
                if blocks_above:
                    # Sort by y position (closest to image first)
                    blocks_above.sort(key=lambda x: -x[0])  # Sort by y0 descending
                    
                    # Include blocks that are likely related (headers, intro)
                    for block_y0, block_y1, text in blocks_above:
                        # Check if this could be a section header or intro
                        # Section headers are usually short, intro text can be longer
                        if len(text) > 10:  # Substantial text
                            new_y0 = min(new_y0, block_y0)
                
                # Step 3: Find contextual text below the image
                # Look for descriptions, captions, or legend text
                new_y1 = img_rect.y1
                
                blocks_below = []
                for block in blocks:
                    if block.get("type") == 0:  # Text block
                        block_bbox = block.get("bbox", [])
                        if len(block_bbox) >= 4:
                            block_y0 = block_bbox[1]  # Top of text block
                            block_y1 = block_bbox[3]  # Bottom of text block
                            
                            # Block is below the image and within context range
                            if block_y0 >= img_rect.y1 and block_y0 <= (img_rect.y1 + context_below):
                                # Get text
                                text = ""
                                for line in block.get("lines", []):
                                    for span in line.get("spans", []):
                                        text += span.get("text", "")
                                
                                text = text.strip()
                                # Include if it's substantial text
                                if len(text) > 5 and not self._is_callout_label(text):
                                    blocks_below.append((block_y0, block_y1, text))
                
                # Expand to include contextual blocks below
                if blocks_below:
                    for block_y0, block_y1, text in blocks_below:
                        if len(text) > 10:  # Substantial text
                            new_y1 = max(new_y1, block_y1)
                            
            except Exception:
                pass
            
            # Step 4: Apply minimum expansion for padding
            # Add small padding if no context was found
            min_padding = 10.0
            if new_y0 == img_rect.y0:
                new_y0 = max(margin_x, img_rect.y0 - min_padding)
            if new_y1 == img_rect.y1:
                new_y1 = min(page_rect.height - margin_x, img_rect.y1 + min_padding)
            
            # Ensure we don't exceed page bounds
            new_y0 = max(margin_x, new_y0)
            new_y1 = min(page_rect.height - margin_x, new_y1)
            
            return fitz.Rect(new_x0, new_y0, new_x1, new_y1)
            
        except Exception:
            # Fallback: just extend to full width with small padding
            page_rect = page.rect
            return fitz.Rect(
                margin_x, 
                max(margin_x, image_bbox[1] - 10),
                page_rect.width - margin_x,
                min(page_rect.height - margin_x, image_bbox[3] + 10)
            )
    
    def _get_image_positions_on_page(self, page: fitz.Page) -> List[Dict]:
        """
        Get positions of all images on the page from text_dict.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            List of dicts with 'bbox', 'xref', and size info
        """
        positions = []
        try:
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                if block.get("type") == 1:  # Image block
                    bbox = block.get("bbox", [])
                    if len(bbox) >= 4:
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                        positions.append({
                            "bbox": bbox,
                            "xref": block.get("xref"),
                            "width": width,
                            "height": height,
                            "area": width * height
                        })
        except Exception:
            pass
        return positions
    
    def _merge_adjacent_strips(self, image_positions: List[Dict], 
                                width_tolerance: float = 5.0,
                                vertical_gap_tolerance: float = 5.0,
                                min_strip_height: float = 30.0) -> List[Dict]:
        """
        Merge adjacent image strips that are likely parts of the same sliced image.
        
        Some PDFs export screenshots as multiple thin horizontal strips. This method
        detects such strips (images with similar widths stacked vertically) and merges
        them into single logical images.
        
        Args:
            image_positions: List of image position dicts with 'bbox', 'width', 'height'
            width_tolerance: Maximum width difference to consider images as part of same strip group
            vertical_gap_tolerance: Maximum vertical gap between strips to consider them adjacent
            min_strip_height: Images taller than this are not considered strips
            
        Returns:
            List of merged image positions (strips combined into single entries)
        """
        if not image_positions:
            return []
        
        # Separate potential strips from regular images
        potential_strips = []
        regular_images = []
        
        for pos in image_positions:
            height = pos.get("height", 0)
            # Strips are thin horizontal images
            if height < min_strip_height and height > 0:
                potential_strips.append(pos)
            else:
                regular_images.append(pos)
        
        if len(potential_strips) < 2:
            # Not enough strips to merge, return original
            return image_positions
        
        # Sort strips by x position (left), then by y position (top to bottom)
        potential_strips.sort(key=lambda p: (p["bbox"][0], p["bbox"][1]))
        
        # Group strips by similar x position and width (same column)
        strip_groups = []
        used_strips = set()
        
        for i, strip in enumerate(potential_strips):
            if i in used_strips:
                continue
            
            bbox = strip["bbox"]
            strip_x0 = bbox[0]
            strip_width = strip["width"]
            
            # Start a new group with this strip
            group = [strip]
            used_strips.add(i)
            
            # Find all other strips with similar x position and width
            for j, other_strip in enumerate(potential_strips):
                if j in used_strips:
                    continue
                
                other_bbox = other_strip["bbox"]
                other_x0 = other_bbox[0]
                other_width = other_strip["width"]
                
                # Check if same column (similar x0 and width)
                if (abs(other_x0 - strip_x0) <= width_tolerance and 
                    abs(other_width - strip_width) <= width_tolerance):
                    group.append(other_strip)
                    used_strips.add(j)
            
            if len(group) >= 2:
                strip_groups.append(group)
            else:
                # Single strip, treat as regular image
                regular_images.append(strip)
        
        # Merge each group into a single image position
        merged_images = list(regular_images)
        
        for group in strip_groups:
            # Sort group by y position
            group.sort(key=lambda p: p["bbox"][1])
            
            # Check if strips are truly adjacent (small vertical gaps)
            is_contiguous = True
            for k in range(len(group) - 1):
                current_bottom = group[k]["bbox"][3]
                next_top = group[k + 1]["bbox"][1]
                gap = next_top - current_bottom
                
                if gap > vertical_gap_tolerance:
                    is_contiguous = False
                    break
            
            if is_contiguous and len(group) >= 3:
                # Merge all strips in this group
                all_bboxes = [p["bbox"] for p in group]
                merged_x0 = min(b[0] for b in all_bboxes)
                merged_y0 = min(b[1] for b in all_bboxes)
                merged_x1 = max(b[2] for b in all_bboxes)
                merged_y1 = max(b[3] for b in all_bboxes)
                
                merged_width = merged_x1 - merged_x0
                merged_height = merged_y1 - merged_y0
                
                # Collect all xrefs from the strips
                xrefs = [p.get("xref") for p in group if p.get("xref")]
                
                merged_images.append({
                    "bbox": [merged_x0, merged_y0, merged_x1, merged_y1],
                    "xref": xrefs[0] if xrefs else None,
                    "xrefs": xrefs,  # Keep all xrefs for reference
                    "width": merged_width,
                    "height": merged_height,
                    "area": merged_width * merged_height,
                    "merged_strips": len(group)
                })
            else:
                # Not contiguous enough, add as individual images
                merged_images.extend(group)
        
        return merged_images
    
    def _regions_overlap(self, region1: List[float], region2: List[float], threshold: float = 0.5) -> bool:
        """
        Check if two regions overlap significantly.
        
        Args:
            region1: First bounding box [x0, y0, x1, y1]
            region2: Second bounding box [x0, y0, x1, y1]
            threshold: Minimum overlap ratio to consider as overlapping
            
        Returns:
            True if regions overlap significantly
        """
        try:
            rect1 = fitz.Rect(region1)
            rect2 = fitz.Rect(region2)
            intersection = rect1 & rect2
            
            if intersection.is_empty:
                return False
            
            # Calculate overlap ratio relative to smaller region
            inter_area = intersection.width * intersection.height
            smaller_area = min(rect1.width * rect1.height, rect2.width * rect2.height)
            
            if smaller_area == 0:
                return False
            
            return (inter_area / smaller_area) >= threshold
        except Exception:
            return False
    
    def _region_contained_in(self, region: List[float], container: List[float], threshold: float = 0.8) -> bool:
        """
        Check if a region is substantially contained within another region.
        
        This is used to detect when a new extraction's render area is already
        covered by an existing extraction, even if their core image areas don't overlap.
        
        Args:
            region: Bounding box to check [x0, y0, x1, y1]
            container: Container bounding box [x0, y0, x1, y1]
            threshold: Minimum containment ratio (0.8 = 80% of region must be in container)
            
        Returns:
            True if region is substantially contained in container
        """
        try:
            rect = fitz.Rect(region)
            container_rect = fitz.Rect(container)
            intersection = rect & container_rect
            
            if intersection.is_empty:
                return False
            
            # Calculate what percentage of 'region' is inside 'container'
            inter_area = intersection.width * intersection.height
            region_area = rect.width * rect.height
            
            if region_area == 0:
                return False
            
            containment_ratio = inter_area / region_area
            return containment_ratio >= threshold
        except Exception:
            return False
    
    def extract_images(self, page: fitz.Page, page_num: int, pdf_name: str) -> List[Dict]:
        """
        Extract images from a PDF page using multiple methods.
        Groups nearby image blocks to extract complete images.
        INCLUDES overlaid annotations (numbered callouts, pointer lines) with screenshots.
        
        Small icons are extracted WITH surrounding context (text, descriptions) rather than
        as standalone images.
        
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
        extracted_regions = []   # Track rendered regions to avoid duplicates
        
        # Size thresholds - filter out very tiny decorative elements
        MIN_SIZE_FILTER = 20      # Minimum size to consider at all (pixels)
        MIN_AREA_FILTER = 400     # Minimum area to consider (pixels)
        
        try:
            # Get all image positions on the page with their sizes
            image_positions = self._get_image_positions_on_page(page)
            
            # FIRST: Merge adjacent strips before filtering
            # Some PDFs export screenshots as thin horizontal strips that need to be combined
            image_positions = self._merge_adjacent_strips(
                image_positions,
                width_tolerance=5.0,
                vertical_gap_tolerance=5.0,
                min_strip_height=30.0
            )
            
            # Filter out very tiny decorative elements
            valid_images = []
            for pos in image_positions:
                if pos["width"] < MIN_SIZE_FILTER or pos["height"] < MIN_SIZE_FILTER:
                    continue
                if pos["area"] < MIN_AREA_FILTER:
                    continue
                valid_images.append(pos)
            
            # --- DBSCAN performance/cluster stats ---
            dbscan_stats = {}
            try:
                # Group nearby images together to extract as a unit
                if valid_images:
                    image_bboxes = [{"bbox": p["bbox"], "xref": p.get("xref")} for p in valid_images]
                    import time
                    t0 = time.time()
                    # Use configured clustering method for more intelligent grouping
                    image_groups = self._group_blocks_adaptive(image_bboxes, method=self.clustering_method)
                    t1 = time.time()
                    # Cluster statistics
                    num_groups = len(image_groups)
                    group_sizes = [len(g) for g in image_groups] if image_groups else []
                    num_blocks = len(image_bboxes)
                    num_noise = sum(1 for g in image_groups if len(g) == 1)
                    dbscan_stats = {
                        "num_blocks": num_blocks,
                        "num_groups": num_groups,
                        "avg_group_size": float(np.mean(group_sizes)) if group_sizes else 0.0,
                        "num_noise_groups": num_noise,
                        "group_sizes": group_sizes,
                        "time_ms": (t1 - t0) * 1000.0
                    }
                    # Store stats for later summary
                    if not hasattr(self, "dbscan_stats_per_page"):
                        self.dbscan_stats_per_page = {}
                    self.dbscan_stats_per_page[page_num + 1] = dbscan_stats
            except Exception:
                pass
            
            # Method 2: Extract ALL images with annotations AND context
            # Every image (regardless of size) gets:
            # 1. Annotation expansion (callouts, pointer lines)
            # 2. Context expansion (headers, descriptions)
            try:
                # Group nearby images together to extract as a unit
                if valid_images:
                    image_bboxes = [{"bbox": p["bbox"], "xref": p.get("xref")} for p in valid_images]
                    # Use configured clustering method for more intelligent grouping
                    image_groups = self._group_blocks_adaptive(image_bboxes, method=self.clustering_method)
                    
                    # Sort groups by Y position (topmost first) so that images higher on the page
                    # are processed first. This tends to result in better context expansion
                    # coverage and helps eliminate redundant extractions when context areas overlap.
                    def get_group_top_y(group):
                        bboxes = [b.get("bbox", []) for b in group if b.get("bbox")]
                        if not bboxes:
                            return float('inf')
                        return min(bbox[1] for bbox in bboxes)  # Min y0 = topmost
                    
                    image_groups.sort(key=get_group_top_y)
                    
                    for group_idx, group in enumerate(image_groups):
                        try:
                            # Merge all bboxes in this group
                            bboxes = [b.get("bbox", []) for b in group if b.get("bbox")]
                            if not bboxes:
                                continue
                            
                            merged_rect = self._merge_bboxes(bboxes)
                            
                            # Step 1: Expand to include annotations (callouts, pointer lines)
                            # This captures ❶❷❸ labels and connecting lines
                            has_annotations = self._has_overlaid_annotations(page, list(merged_rect))
                            
                            if has_annotations:
                                annotation_rect = self._get_combined_annotation_bbox(page, list(merged_rect))
                            else:
                                annotation_rect = merged_rect
                            
                            # Step 2: Expand to include contextual content (headers, descriptions)
                            render_rect = self._expand_bbox_for_context(
                                page, list(annotation_rect),
                                margin_x=25.0,
                                context_above=100.0,
                                context_below=80.0
                            )
                            
                            # Check for overlap with already extracted regions
                            # Use the ORIGINAL image bbox (merged_rect) for overlap detection,
                            # not the context-expanded bbox, to avoid false duplicates when
                            # context expansion causes unrelated image groups to overlap
                            is_duplicate = False
                            original_bbox = list(merged_rect)
                            for existing_core, existing_render in extracted_regions:
                                if self._regions_overlap(original_bbox, existing_core, 0.7):
                                    is_duplicate = True
                                    break
                            
                            if is_duplicate:
                                continue
                            
                            # Also check if this render area is already substantially covered
                            # by an existing extraction (handles case where different image groups
                            # expand to overlapping context areas)
                            render_bbox = list(render_rect)
                            for existing_core, existing_render in extracted_regions:
                                if self._region_contained_in(render_bbox, existing_render, 0.8):
                                    is_duplicate = True
                                    break
                            
                            if is_duplicate:
                                continue
                            
                            # Render the complete area
                            pix = page.get_pixmap(clip=render_rect, matrix=fitz.Matrix(4, 4))
                            
                            if pix and pix.n > 0:
                                image_counter += 1
                                image_filename = f"{pdf_name}_page_{page_num + 1:04d}_img_{image_counter:03d}.png"
                                image_path = self.images_dir / image_filename
                                
                                pix.save(image_path)
                                image_bytes = pix.tobytes()
                                
                                # Track xrefs
                                for item in group:
                                    xref = item.get("xref")
                                    if xref:
                                        extracted_xrefs.add(xref)
                                
                                image_data = {
                                    "filename": image_filename,
                                    "path": str(image_path),
                                    "size_bytes": len(image_bytes),
                                    "format": "png",
                                    "page": page_num + 1,
                                    "index": image_counter,
                                    "method": "image_with_context",
                                    "images_merged": len(group),
                                    "bbox": list(render_rect),
                                    "has_annotations": has_annotations
                                }
                                
                                images_info.append(image_data)
                                # Store both core bbox and render bbox for duplicate detection
                                extracted_regions.append((original_bbox, list(render_rect)))
                                self.summary["images_extracted"] += 1
                                pix = None
                                
                        except Exception:
                            pass
                        
            except Exception:
                pass
            
            # Method 3: Extract vector graphics/drawings as images
            # Handle two cases:
            # 1. Large vector graphics: Diagrams, illustrations - extract FIRST with context
            # 2. Icon clusters: Small grouped vector elements - extract remaining with context
            # 
            # IMPORTANT: Process large graphics FIRST to avoid icon clusters fragmenting the diagram
            try:
                drawings = page.get_drawings()
                if drawings:
                    page_width = page.rect.width
                    
                    # FIRST: Check for larger meaningful graphics (diagrams, UI mockups, etc.)
                    # These should be processed before icon clusters to avoid fragmentation
                    meaningful_drawings = self._filter_decorative_drawings(drawings, page_width)
                    
                    if meaningful_drawings and len(meaningful_drawings) >= 3:
                        all_rects = [d.get("rect") for d in meaningful_drawings if d.get("rect")]
                        
                        if all_rects:
                            # Get the bounding box of meaningful drawings
                            min_x = min(r.x0 for r in all_rects)
                            min_y = min(r.y0 for r in all_rects)
                            max_x = max(r.x1 for r in all_rects)
                            max_y = max(r.y1 for r in all_rects)
                            
                            full_drawing_rect = fitz.Rect(min_x, min_y, max_x, max_y)
                            
                            # Only proceed if the drawing area is substantial
                            if full_drawing_rect.width >= 50 and full_drawing_rect.height >= 50:
                                # Expand to include annotations
                                expanded_rect = self._get_combined_annotation_bbox(page, list(full_drawing_rect))
                                
                                # Check if already covered using core bbox
                                is_duplicate = False
                                original_drawing_bbox = list(full_drawing_rect)
                                for existing_core, existing_render in extracted_regions:
                                    if self._regions_overlap(original_drawing_bbox, existing_core, 0.7):
                                        is_duplicate = True
                                        break
                                
                                if not is_duplicate:
                                    try:
                                        # Expand with context
                                        context_rect = self._expand_bbox_for_context(page, list(expanded_rect))
                                        
                                        pix = page.get_pixmap(clip=context_rect, matrix=fitz.Matrix(4, 4))
                                        
                                        if pix and pix.n > 0:
                                            image_counter += 1
                                            image_filename = f"{pdf_name}_page_{page_num + 1:04d}_img_{image_counter:03d}.png"
                                            image_path = self.images_dir / image_filename
                                            
                                            pix.save(image_path)
                                            image_bytes = pix.tobytes()
                                            
                                            image_data = {
                                                "filename": image_filename,
                                                "path": str(image_path),
                                                "size_bytes": len(image_bytes),
                                                "format": "png",
                                                "page": page_num + 1,
                                                "index": image_counter,
                                                "method": "vector_graphics_with_context",
                                                "drawings_count": len(meaningful_drawings),
                                                "bbox": list(context_rect)
                                            }
                                            
                                            images_info.append(image_data)
                                            extracted_regions.append((original_drawing_bbox, list(context_rect)))
                                            self.summary["images_extracted"] += 1
                                            pix = None
                                            
                                    except Exception:
                                        pass
                    
                    # SECOND: Check for icon clusters (small grouped drawings)
                    # These are extracted only if not already covered by larger graphics
                    icon_clusters = self._find_icon_clusters(drawings)
                    
                    for cluster_rect in icon_clusters:
                        try:
                            # Skip if area already covered by larger extraction
                            is_duplicate = False
                            cluster_center_x = (cluster_rect.x0 + cluster_rect.x1) / 2
                            cluster_center_y = (cluster_rect.y0 + cluster_rect.y1) / 2
                            original_cluster_bbox = list(cluster_rect)
                            
                            for existing_core, existing_render in extracted_regions:
                                ex_rect = fitz.Rect(existing_render)
                                if ex_rect.contains(fitz.Point(cluster_center_x, cluster_center_y)):
                                    is_duplicate = True
                                    break
                            
                            if is_duplicate:
                                continue
                            
                            # Expand to include context around the icon
                            context_rect = self._expand_bbox_for_context(
                                page, list(cluster_rect),
                                margin_x=25.0,
                                context_above=100.0,
                                context_below=100.0
                            )
                            
                            # Check overlap with existing extractions using core bbox
                            for existing_core, existing_render in extracted_regions:
                                if self._regions_overlap(original_cluster_bbox, existing_core, 0.7):
                                    is_duplicate = True
                                    break
                            
                            if is_duplicate:
                                continue
                            
                            # Render the icon with context
                            pix = page.get_pixmap(clip=context_rect, matrix=fitz.Matrix(4, 4))
                            
                            if pix and pix.n > 0:
                                image_counter += 1
                                image_filename = f"{pdf_name}_page_{page_num + 1:04d}_img_{image_counter:03d}.png"
                                image_path = self.images_dir / image_filename
                                
                                pix.save(image_path)
                                image_bytes = pix.tobytes()
                                
                                image_data = {
                                    "filename": image_filename,
                                    "path": str(image_path),
                                    "size_bytes": len(image_bytes),
                                    "format": "png",
                                    "page": page_num + 1,
                                    "index": image_counter,
                                    "method": "icon_cluster_with_context",
                                    "bbox": list(context_rect)
                                }
                                
                                images_info.append(image_data)
                                extracted_regions.append((original_cluster_bbox, list(context_rect)))
                                self.summary["images_extracted"] += 1
                                pix = None
                                
                        except Exception:
                            pass
                                    
            except Exception:
                # If drawings method fails, continue
                pass
                    
        except Exception as e:
            error_msg = f"Error processing images on page {page_num + 1}: {str(e)}"
            self.summary["errors"].append(error_msg)
        
        return images_info
    
    def _group_drawings(self, drawings: List[Dict], proximity_threshold: float = 100.0) -> List[List[Dict]]:
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
    
    def _is_decorative_drawing(self, drawing: Dict, page_width: float) -> bool:
        """
        Check if a drawing is likely a decorative layout element (not meaningful graphics).
        
        Decorative elements include:
        - Horizontal lines/dividers spanning most of the page
        - Header bar backgrounds (wide rectangles)
        - Simple bullet points/circles
        - Thin vertical/horizontal rules
        
        Args:
            drawing: Drawing dictionary from page.get_drawings()
            page_width: Width of the page in points
            
        Returns:
            True if the drawing is likely decorative
        """
        rect = drawing.get("rect")
        if not rect:
            return True
        
        width = rect.width
        height = rect.height
        
        # Very thin elements are likely lines/rules
        if width < 3 or height < 3:
            return True
        
        # Horizontal lines/bars spanning significant page width (>60%) are likely decorative
        if width > page_width * 0.6 and height < 30:
            return True
        
        # Very thin horizontal rectangles (header bars, dividers)
        aspect_ratio = width / height if height > 0 else float('inf')
        if aspect_ratio > 15 and width > page_width * 0.5:
            return True
        
        # Small circles/dots (bullet points) - tiny elements
        if width < 15 and height < 15:
            return True
        
        return False
    
    def _filter_decorative_drawings(self, drawings: List[Dict], page_width: float) -> List[Dict]:
        """
        Filter out decorative drawings, keeping only meaningful graphics.
        
        Args:
            drawings: List of drawing dictionaries
            page_width: Width of the page in points
            
        Returns:
            Filtered list of drawings that are likely meaningful graphics
        """
        meaningful = []
        for d in drawings:
            if not self._is_decorative_drawing(d, page_width):
                meaningful.append(d)
        return meaningful
    
    def _find_icon_clusters(self, drawings: List[Dict], cluster_distance: float = 100.0) -> List[fitz.Rect]:
        """
        Find clusters of small drawings that together form icons.
        
        Many icons (like dustbin, WiFi, etc.) are composed of multiple tiny 
        vector elements. This method identifies such clusters.
        
        Args:
            drawings: List of drawing dictionaries
            cluster_distance: Maximum distance between elements to be in same cluster
            
        Returns:
            List of bounding rects for identified icon clusters
        """
        # Get all small drawings (those that would normally be filtered as decorative)
        # Use 25x25 threshold to capture icons like muted speakers, WiFi symbols, etc.
        small_drawings = []
        for d in drawings:
            rect = d.get("rect")
            if rect and rect.width < 25 and rect.height < 25 and rect.width >= 1 and rect.height >= 1:
                small_drawings.append(rect)
        
        if len(small_drawings) < 2:  # Reduced from 3 to 2 for simpler icons
            return []
        
        # Find clusters using simple proximity grouping
        clusters = []
        used = set()
        
        for i, rect1 in enumerate(small_drawings):
            if i in used:
                continue
            
            # Start a new cluster
            cluster = [rect1]
            used.add(i)
            
            # Find all nearby small drawings
            for j, rect2 in enumerate(small_drawings):
                if j in used:
                    continue
                
                # Check if rect2 is close to any rect in the cluster
                for cluster_rect in cluster:
                    # Calculate distance between centers
                    c1_x = (cluster_rect.x0 + cluster_rect.x1) / 2
                    c1_y = (cluster_rect.y0 + cluster_rect.y1) / 2
                    c2_x = (rect2.x0 + rect2.x1) / 2
                    c2_y = (rect2.y0 + rect2.y1) / 2
                    dist = ((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2) ** 0.5
                    
                    if dist <= cluster_distance:
                        cluster.append(rect2)
                        used.add(j)
                        break
            
            # If cluster has enough elements, it's likely an icon
            if len(cluster) >= 2:  # Reduced from 5 to capture simpler icons (muted speaker, etc.)
                # Calculate cluster bounding box
                min_x = min(r.x0 for r in cluster)
                min_y = min(r.y0 for r in cluster)
                max_x = max(r.x1 for r in cluster)
                max_y = max(r.y1 for r in cluster)
                cluster_rect = fitz.Rect(min_x, min_y, max_x, max_y)
                
                # Only consider if the cluster is reasonably sized (icon-like, not scattered)
                if cluster_rect.width < 100 and cluster_rect.height < 100:
                    clusters.append(cluster_rect)
        
        return clusters
    
    def _has_meaningful_graphics(self, page: fitz.Page) -> Tuple[bool, List[Dict]]:
        """
        Check if a page has meaningful vector graphics (not just decorative elements).
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Tuple of (has_meaningful_graphics, filtered_drawings)
        """
        try:
            drawings = page.get_drawings()
            if not drawings:
                return False, []
            
            page_width = page.rect.width
            
            # First, check for icon clusters (many small drawings clustered together)
            icon_clusters = self._find_icon_clusters(drawings)
            if icon_clusters:
                # Found icon clusters - return the drawings that form them
                # For simplicity, return all drawings since we'll render the cluster area
                return True, drawings
            
            # Filter out decorative elements
            meaningful_drawings = self._filter_decorative_drawings(drawings, page_width)
            
            if not meaningful_drawings:
                return False, []
            
            # Need a minimum number of meaningful shapes to consider it a graphic
            # A single rectangle is likely decorative; complex diagrams have many shapes
            if len(meaningful_drawings) < 3:
                return False, []
            
            # Check the complexity - meaningful graphics typically have variety
            # Get bounding box of meaningful drawings
            rects = [d.get("rect") for d in meaningful_drawings if d.get("rect")]
            if not rects:
                return False, []
            
            # Calculate the total area covered by meaningful drawings
            total_area = sum(r.width * r.height for r in rects)
            
            # Meaningful graphics should have substantial area (at least 5000 sq points)
            if total_area < 5000:
                return False, []
            
            return True, meaningful_drawings
            
        except Exception:
            return False, []
    
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
            # Add DBSCAN cluster/performance stats if available
            "dbscan_stats": getattr(self, "dbscan_stats_per_page", {})
            ,
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


def extract_pdf(pdf_path: Path, output_dir: Path = None, clustering_method: str = "dbscan") -> Dict:
    """
    Main function to extract text and images from a PDF.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Output directory (defaults to temp folder in project root)
        clustering_method: Image block clustering method ("dbscan", "hierarchical", or "proximity")
        
    Returns:
        Summary dictionary
    """
    # Set default output directory
    if output_dir is None:
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "temp_extraction"
    
    # Create extractor with specified clustering method
    extractor = PDFExtractor(output_dir, clustering_method=clustering_method)
    
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

