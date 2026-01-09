"""
Utility to map text chunk headings/subheadings to image description chunks by page.
"""
import json
from pathlib import Path
from typing import Dict, Tuple

def build_heading_mapping(text_chunks_json: Path) -> Dict[Tuple[str, int], Tuple[str, str]]:
    """
    Build a mapping from (fileName, page_no) to (heading, subheading) from text chunk metadata.
    Args:
        text_chunks_json: Path to the JSON file containing text chunk metadata (e.g., chunks_metadata.json)
    Returns:
        mapping: dict with key (fileName, page_no) and value (heading, subheading)
    """
    mapping = {}
    with open(text_chunks_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for chunk in data.get('chunks', []):
            meta = chunk.get('metadata', {})
            file_name = meta.get('fileName')
            page_no = meta.get('Page No')
            heading = meta.get('Heading', '')
            sub_heading = meta.get('SubHeading', '')
            if file_name and page_no is not None:
                mapping[(file_name, int(page_no))] = (heading, sub_heading)
    return mapping

def parse_image_file_context(image_file: Path) -> Tuple[str, int]:
    """
    Parse image file name to extract (fileName, page_no).
    Example: HomeHawkApp_Users_Guide_CC1803YK9100_ENG_page_0007_img_001_description.md
    Returns:
        (fileName, page_no)
    """
    stem = image_file.stem
    # Remove trailing _img_..._description if present
    base = stem.split('_page_')[0]
    page_part = stem.split('_page_')[1] if '_page_' in stem else ''
    page_no = 0
    if page_part:
        try:
            page_no = int(page_part.split('_')[0])
        except Exception:
            pass
    return base, page_no

def update_image_chunk_metadata_with_context(
    image_chunk_dir: Path,
    mapping: Dict[Tuple[str, int], Tuple[str, str]]
):
    """
    Update all image chunk JSON files in a directory to inject heading/subheading from mapping.
    Args:
        image_chunk_dir: Path to directory with image chunk JSON files
        mapping: dict from (fileName, page_no) to (heading, subheading)
    """
    for chunk_file in image_chunk_dir.glob('*.json'):
        with open(chunk_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        meta = data.get('metadata', {})
        file_name, page_no = parse_image_file_context(chunk_file)
        heading, sub_heading = mapping.get((file_name, page_no), ('', ''))
        meta['Heading'] = heading
        meta['SubHeading'] = sub_heading
        data['metadata'] = meta
        with open(chunk_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # Example usage
    project_root = Path(__file__).parent.parent.parent
    text_chunks_json = project_root / 'temp_extraction' / 'chunks' / 'chunks_metadata.json'
    image_chunk_dir = project_root / 'temp_extraction' / 'image_chunks_review'
    mapping = build_heading_mapping(text_chunks_json)
    update_image_chunk_metadata_with_context(image_chunk_dir, mapping)
    print(f"Updated image chunk metadata with heading/subheading context from text chunks.")
