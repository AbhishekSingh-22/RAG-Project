# Technique 1: Dict-Based Context Extraction

**Status:** ✅ Currently Implemented  
**File:** `src/ingestion/pdf_extractor.py`  
**Method:** `PDFExtractor.extract_images()`

---

## Overview

This technique uses PyMuPDF's `page.get_text("dict")` to extract images with their **exact positions** on the page, then expands the extraction area to include surrounding context (headers, captions, annotations).

---

## Key Characteristics

| Attribute | Value |
|-----------|-------|
| **Library** | PyMuPDF (pymupdf/fitz) |
| **Primary API** | `page.get_text("dict")` |
| **Output Format** | PNG at 4x resolution (288 DPI) |
| **Context Aware** | ✅ Yes |
| **Handles Annotations** | ✅ Yes (callouts, pointer lines) |
| **Handles Sliced Images** | ✅ Yes (strip merging) |
| **Duplicate Detection** | ✅ Yes (overlap + containment) |

---

## Pipeline Steps

### Step 1: Get Raw Image Positions
```python
image_positions = self._get_image_positions_on_page(page)
```
- Uses `page.get_text("dict")` to find all image blocks
- Extracts bbox (position), dimensions, and xref for each image

### Step 2: Merge Adjacent Strips
```python
image_positions = self._merge_adjacent_strips(image_positions, ...)
```
- **Problem solved:** Some PDFs export screenshots as thin horizontal strips
- **Detection:** Images with height < 30px, same width, vertically stacked
- **Action:** Merges strips into single logical images

### Step 3: Filter Tiny Elements
```python
MIN_SIZE_FILTER = 20   # pixels
MIN_AREA_FILTER = 400  # pixels²
```
- Removes decorative elements (bullets, separators, tiny icons)

### Step 4: Group Nearby Images
```python
image_groups = self._group_nearby_blocks(image_bboxes, proximity_threshold=100.0)
```
- Groups images within 100 points of each other
- Treats grouped images as a single extraction unit

### Step 5: Sort Groups (Topmost First)
```python
image_groups.sort(key=get_group_top_y)
```
- Processes images from top of page to bottom
- Ensures larger context expansions happen first

### Step 6: Merge Bboxes in Group
```python
merged_rect = self._merge_bboxes(bboxes)
```
- Creates single bounding box encompassing all images in group

### Step 7: Expand for Annotations
```python
has_annotations = self._has_overlaid_annotations(page, merged_rect)
annotation_rect = self._get_combined_annotation_bbox(page, merged_rect)
```
- **Detects:** Callout labels (❶❷❸, A/B/C, 1/2/3)
- **Detects:** Pointer lines (vector drawings)
- **Expands:** Bbox to include all related annotations

### Step 8: Expand for Context
```python
render_rect = self._expand_bbox_for_context(
    page, annotation_rect,
    margin_x=25.0,
    context_above=100.0,
    context_below=80.0
)
```
- Extends to full page width (with margins)
- Searches for text blocks above (headers, section titles)
- Searches for text blocks below (captions, descriptions)

### Step 9: Duplicate Detection
```python
# Check 1: Core bbox overlap (70% threshold)
self._regions_overlap(original_bbox, existing_core, 0.7)

# Check 2: Render area containment (80% threshold)
self._region_contained_in(render_bbox, existing_render, 0.8)
```
- Prevents extracting same content multiple times
- Uses original image bbox for overlap (not expanded area)
- Checks if new render area is already covered

### Step 10: Render and Save
```python
pix = page.get_pixmap(clip=render_rect, matrix=fitz.Matrix(4, 4))
pix.save(image_path)
```
- Renders at 4x resolution (288 DPI)
- Saves as PNG with metadata

---

## Coordinate System

```
Origin (0, 0) = TOP-LEFT corner of page

    (0,0) ────────────────────────► X (increases right)
       │
       │    bbox = [x0, y0, x1, y1]
       │         = [left, top, right, bottom]
       │
       │    Units: Points (1 point = 1/72 inch)
       │
       ▼
       Y (increases down)
```

---

## Output Structure

Each extracted image produces:
```json
{
    "filename": "Manual_page_0004_img_001.png",
    "path": "/path/to/images/Manual_page_0004_img_001.png",
    "size_bytes": 123456,
    "format": "png",
    "page": 4,
    "index": 1,
    "method": "image_with_context",
    "images_merged": 3,
    "bbox": [25.0, 69.3, 570.2, 817.0],
    "has_annotations": true
}
```

---

## Strengths

| Strength | Description |
|----------|-------------|
| **Context Preservation** | Includes headers, captions, and descriptions with images |
| **Annotation Handling** | Captures callout labels and pointer lines |
| **Strip Merging** | Reconstructs images that were sliced into horizontal strips |
| **Smart Grouping** | Treats nearby images as a single unit |
| **Duplicate Prevention** | Two-level checking prevents redundant extractions |
| **High Quality** | 4x resolution produces crisp, readable images |

---

## Weaknesses

| Weakness | Description |
|----------|-------------|
| **Larger Files** | Context inclusion increases file size |
| **Over-expansion** | May include unrelated text in some cases |
| **Complexity** | More complex code than simple extraction |
| **Processing Time** | Slower than raw extraction |

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_SIZE_FILTER` | 20 | Minimum dimension to consider (pixels) |
| `MIN_AREA_FILTER` | 400 | Minimum area to consider (pixels²) |
| `proximity_threshold` | 100.0 | Max distance for grouping (points) |
| `margin_x` | 25.0 | Horizontal margin from edges (points) |
| `context_above` | 100.0 | Max expansion above image (points) |
| `context_below` | 80.0 | Max expansion below image (points) |
| `render_scale` | 4 | Resolution multiplier (4x = 288 DPI) |
| `overlap_threshold` | 0.7 | Min overlap for duplicate detection |
| `containment_threshold` | 0.8 | Min containment for duplicate detection |

---

## Test Results

### Panasonic-Smart-AC-User-Manual.pdf (12 pages)

| Metric | Value |
|--------|-------|
| Total images extracted | 10 |
| Pages with images | 10 |
| Processing time | ~5 seconds |
| Avg image size | ~500 KB |

### Issues Resolved

1. **Page 9 - Sliced Screenshot**
   - Problem: Screenshot exported as 33 thin strips (13px each)
   - Solution: Strip merging reconstructs complete image

2. **Page 4 - Redundant Extraction**
   - Problem: Two overlapping extractions from different image groups
   - Solution: Containment check + Y-sorting eliminates redundancy

---

## Code Location

```
src/ingestion/pdf_extractor.py

Key Methods:
├── extract_images()                    # Main extraction method
├── _get_image_positions_on_page()      # Step 1
├── _merge_adjacent_strips()            # Step 2
├── _group_nearby_blocks()              # Step 4
├── _merge_bboxes()                     # Step 6
├── _has_overlaid_annotations()         # Step 7
├── _get_combined_annotation_bbox()     # Step 7
├── _expand_bbox_for_context()          # Step 8
├── _regions_overlap()                  # Step 9
└── _region_contained_in()              # Step 9
```

---

## When to Use This Technique

✅ **Best for:**
- Technical manuals with screenshots
- Documents with annotations (callouts, pointer lines)
- PDFs with sliced/fragmented images
- Content where context is important for understanding

❌ **Not ideal for:**
- Simple documents with standalone images
- When file size is critical
- Scanned PDFs (use full-page rendering instead)
- When original image quality is paramount
