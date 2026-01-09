"""
Production-grade image description generator using Google Gemini LLM.
Generates structured descriptions of images for RAG pipeline.
"""

import os
import json
import base64
from pathlib import Path
from typing import Dict, List, Optional
import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

SYSTEM_PROMPT = """You are an expert technical documentation analyst specializing in visual content description for Retrieval-Augmented Generation (RAG) systems. 

Your task is to analyze images from product manuals and user guides and create structured, comprehensive descriptions that capture all essential information for later retrieval and contextual understanding.

CRITICAL REQUIREMENTS:
1. Maintain consistency with provided template format
2. Be precise and factual - avoid speculation
3. Identify UI elements, buttons, text, layout, and visual hierarchy
4. Note color schemes, design patterns, and user interaction cues
5. Preserve all visible text exactly as shown
6. Categorize the image type (screenshot, diagram, icon, illustration, etc.)
7. For app interfaces: identify screen purpose, user flow context, and actionable elements
8. For diagrams/illustrations: note components, relationships, and spatial arrangements

RESPONSE TEMPLATE (MUST USE THIS EXACT STRUCTURE, USING MARKDOWN SUBHEADINGS):

---
## Summary
[2-3 sentences describing the main subject and purpose]

## Image Type
[screenshot/diagram/icon/illustration/flowchart/etc.]

## Scene Overview
• [Key visual element 1]
• [Key visual element 2]
• [Visual hierarchy and layout description]
• [Color palette and design characteristics]
• [Visual emphasis and focal points]

## Technical Details
• [UI components: buttons, input fields, menus, etc.]
• [Text content - quote important labels, instructions, or data]
• [Visual styling: fonts, sizes, spacing]
• [Measurements or scale information if visible]
• [Specific technical elements: icons, indicators, status badges]

## Spatial Relationships
• [Top-to-bottom organization]
• [Left-to-right flow]
• [Element positioning and alignment]
• [Grouping and visual connections]
• [Hierarchy of information]

## Functional Context
• [Purpose of the screen/diagram/element]
• [User actions this enables]
• [Information this conveys]
• [Connection to larger workflow/system]

## Analysis & Metadata
• [Key insights for RAG retrieval]
• [Important patterns or warnings]
• [User experience implications]
• [Technical implications]
• [Accessibility considerations if relevant]

---

QUALITY CHECKLIST:
- Descriptions are objective and detailed
- All visible text is accurately captured
- Visual elements are categorized and organized logically
- Information would be useful for context retrieval
- No assumptions; only observations
- Language is clear and professional
- Descriptions are comprehensive enough to reconstruct mental image
"""

def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")

def get_image_media_type(image_path: str) -> str:
    """Determine media type from file extension."""
    ext = Path(image_path).suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    return media_types.get(ext, "image/png")

def generate_image_description(image_path: str, image_filename: str) -> Optional[Dict]:
    """
    Generate structured description for a single image using Gemini.
    
    Args:
        image_path: Full path to the image file
        image_filename: Just the filename for metadata
        
    Returns:
        Dictionary containing description or None if failed
    """
    try:
        # Encode image
        image_data = encode_image_to_base64(image_path)
        media_type = get_image_media_type(image_path)
        
        # Prepare message with image
        message = {
            "role": "user",
            "parts": [
                {
                    "text": SYSTEM_PROMPT + "\n\nPlease analyze this image and provide a comprehensive description following the template above."
                },
                {
                    "inline_data": {
                        "mime_type": media_type,
                        "data": image_data,
                    }
                }
            ]
        }
        
        # Call Gemini API
        model = genai.GenerativeModel("gemini-2.5-flash-lite-preview-09-2025")
        response = model.generate_content(message)
        
        # Extract description
        description = response.text
        
        return {
            "filename": image_filename,
            "image_path": image_path,
            "description": description,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "filename": image_filename,
            "image_path": image_path,
            "error": str(e),
            "status": "failed"
        }

def process_images_directory(
    images_dir: str, 
    output_dir: str, 
    max_images: Optional[int] = None,
    start_index: int = 0
) -> Dict:
    """
    Process all images in a directory and save descriptions.
    
    Args:
        images_dir: Directory containing images
        output_dir: Directory to save descriptions
        max_images: Limit number of images to process (None = all)
        start_index: Start processing from this index
        
    Returns:
        Summary dictionary with processing results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = sorted([
        f for f in os.listdir(images_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))
    ])
    
    # Apply limits
    if max_images:
        image_files = image_files[start_index:start_index + max_images]
    else:
        image_files = image_files[start_index:]
    
    results = {
        "total_processed": 0,
        "successful": 0,
        "failed": 0,
        "descriptions": []
    }

    for idx, image_filename in enumerate(image_files, 1):
        image_path = os.path.join(images_dir, image_filename)
        print(f"Processing [{idx}/{len(image_files)}]: {image_filename}...")

        result = generate_image_description(image_path, image_filename)
        results["descriptions"].append(result)
        results["total_processed"] += 1

        if result["status"] == "success":
            results["successful"] += 1

            # Save description as markdown file
            output_file = os.path.join(
                output_dir, 
                f"{Path(image_filename).stem}_description.md"
            )
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"# Description for: {image_filename}\n\n")
                f.write(f"**Source Image**: {image_filename}\n")
                f.write(f"**Extraction Date**: {str(Path(image_path).stat().st_mtime)}\n\n")
                f.write("---\n\n")
                f.write(result["description"])

            print(f"  ✓ Markdown description saved")
        else:
            results["failed"] += 1
            print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
    
    # Save summary
    summary_file = os.path.join(output_dir, "descriptions_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results

def create_rag_index_file(descriptions_dir: str, index_file: str) -> None:
    """
    Create an index file for RAG system from all descriptions.
    
    Args:
        descriptions_dir: Directory containing description files
        index_file: Output index file path
    """
    index = {
        "version": "1.0",
        "created": str(Path(descriptions_dir).stat().st_mtime),
        "images": []
    }
    
    for desc_file in sorted(os.listdir(descriptions_dir)):
        if desc_file.endswith("_description.md"):
            file_path = os.path.join(descriptions_dir, desc_file)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Extract filename from first line
                lines = content.split("\n")
                image_name = lines[0].replace("# Description for: ", "") if lines else "unknown"
                index["images"].append({
                    "filename": image_name,
                    "description_file": desc_file,
                    "path": file_path
                })
    
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

if __name__ == "__main__":
    images_directory = r"d:\Panasonic\main_project\temp_extraction\images"
    output_directory = r"d:\Panasonic\main_project\temp_extraction\image_descriptions"
    
    print("="*80)
    print("IMAGE DESCRIPTION GENERATOR FOR RAG SYSTEM")
    print("="*80)
    print(f"\nProcessing images from: {images_directory}")
    print(f"Saving markdown descriptions to: {output_directory}\n")

    # Process first 5 images for review
    results = process_images_directory(
        images_directory, 
        output_directory,
        max_images=5,
        start_index=0
    )

    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    print(f"Total Processed: {results['total_processed']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"\nMarkdown descriptions saved to: {output_directory}/*_description.md")
