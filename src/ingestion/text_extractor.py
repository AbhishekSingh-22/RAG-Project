"""
Enhanced Text Extraction Module
Handles document structure, headings, callouts, bullet points, and formatting.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import pymupdf as fitz


@dataclass
class TextSpan:
    """Represents a text span with styling information."""
    text: str
    font_size: float
    font_name: str
    is_bold: bool
    

@dataclass 
class TextLine:
    """Represents a line of text with styling."""
    spans: List[TextSpan] = field(default_factory=list)
    
    @property
    def text(self) -> str:
        return ''.join(s.text for s in self.spans)
    
    @property
    def avg_font_size(self) -> float:
        if not self.spans:
            return 0
        return sum(s.font_size for s in self.spans) / len(self.spans)
    
    @property
    def is_bold(self) -> bool:
        return any(s.is_bold for s in self.spans)


@dataclass
class TextBlock:
    """Represents a text block with position, content, and styling."""
    lines: List[TextLine] = field(default_factory=list)
    x0: float = 0
    y0: float = 0
    x1: float = 0
    y1: float = 0
    block_type: str = "text"
    heading_level: int = 0  # 0 = not a heading, 1-3 = heading levels
    
    @property
    def text(self) -> str:
        return '\n'.join(line.text.strip() for line in self.lines if line.text.strip())
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        return self.y1 - self.y0
    
    @property
    def avg_font_size(self) -> float:
        sizes = [line.avg_font_size for line in self.lines if line.spans]
        return sum(sizes) / len(sizes) if sizes else 0
    
    @property
    def is_bold(self) -> bool:
        return any(line.is_bold for line in self.lines)


class TextExtractor:
    """
    Enhanced text extractor that handles:
    - Heading hierarchy (H1, H2, H3) based on font size
    - Callout labels merged with descriptions
    - Bullet points properly formatted
    - Numbered steps as markdown lists
    - Voice commands as code
    - Note/Important blocks as blockquotes
    - Menu paths bolded
    """
    
    # Font size thresholds for headings
    H1_FONT_SIZE = 14.0  # >= 14pt is H1
    H2_FONT_SIZE = 12.0  # >= 12pt and < 14pt is H2
    H3_FONT_SIZE = 8.0   # 8pt bold (short lines) is H3
    
    # Callout patterns
    CALLOUT_LABELS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    CIRCLED_NUMBERS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳❶❷❸❹❺❻❼❽❾❿"
    
    # Bullet markers
    BULLET_MARKERS = {"–", "-", "•", "R", "●", "○", "■", "□", "►", "▶", "◆", "◇"}
    
    # Page structure thresholds
    HEADER_ZONE_HEIGHT = 30
    FOOTER_ZONE_HEIGHT = 40
    
    # Patterns
    STEP_PATTERN = re.compile(r'^(\d+):?\s*(.*)$')
    VOICE_COMMAND_PATTERN = re.compile(r'(OK Google,\s*[^.]+\.?|Alexa,\s*[^.]+\.?)', re.IGNORECASE)
    MENU_PATTERN = re.compile(r'\[([^\]]+)\]')
    
    def __init__(self):
        self.page_width = 0
        self.page_height = 0
    
    def extract_text_blocks(self, page: fitz.Page) -> List[TextBlock]:
        """Extract all text blocks with styling information."""
        self.page_width = page.rect.width
        self.page_height = page.rect.height
        
        blocks = []
        text_dict = page.get_text("dict")
        
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            
            bbox = block.get("bbox", [0, 0, 0, 0])
            if len(bbox) < 4:
                continue
            
            text_block = TextBlock(
                x0=bbox[0],
                y0=bbox[1],
                x1=bbox[2],
                y1=bbox[3]
            )
            
            for line in block.get("lines", []):
                text_line = TextLine()
                
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if not text:
                        continue
                    
                    font_size = span.get("size", 0)
                    font_name = span.get("font", "")
                    flags = span.get("flags", 0)
                    
                    # Check for bold
                    is_bold = "Bold" in font_name or "bold" in font_name or (flags & 16)
                    
                    text_span = TextSpan(
                        text=text,
                        font_size=font_size,
                        font_name=font_name,
                        is_bold=is_bold
                    )
                    text_line.spans.append(text_span)
                
                if text_line.spans:
                    text_block.lines.append(text_line)
            
            if text_block.lines:
                text_block.block_type = self._classify_block(text_block)
                text_block.heading_level = self._detect_heading_level(text_block)
                blocks.append(text_block)
        
        return blocks
    
    def _classify_block(self, block: TextBlock) -> str:
        """Classify a text block by its type."""
        text = block.text.strip()
        
        if self._is_page_number(block):
            return "page_number"
        
        if block.y0 < self.HEADER_ZONE_HEIGHT:
            return "header"
        
        if block.y1 > self.page_height - self.FOOTER_ZONE_HEIGHT:
            if text.isdigit():
                return "page_number"
            return "footer"
        
        if self._is_standalone_callout_block(block):
            return "standalone_callout"
        
        if self._is_footnote(text):
            return "footnote"
        
        return "text"
    
    def _detect_heading_level(self, block: TextBlock) -> int:
        """Detect if block is a heading and what level."""
        text = block.text.strip()
        avg_size = block.avg_font_size
        is_bold = block.is_bold
        
        # Skip very short text or text that looks like labels
        if len(text) < 3:
            return 0
        
        # Skip if contains bullet patterns
        if text.startswith('•') or text.startswith('-') or text.startswith('–'):
            return 0
        
        # H1: Large font (>= 14pt)
        if avg_size >= self.H1_FONT_SIZE:
            return 1
        
        # H2: Medium font (>= 12pt and < 14pt)
        if avg_size >= self.H2_FONT_SIZE:
            return 2
        
        # H3: Bold 8pt text that is relatively short (sub-section headers)
        if is_bold and avg_size >= 7.5 and avg_size < self.H2_FONT_SIZE:
            # Should be short (typical heading length)
            if len(text) < 80 and '\n' not in text:
                # Not a step number
                if not text[0].isdigit():
                    return 3
        
        return 0
    
    def _is_page_number(self, block: TextBlock) -> bool:
        """Check if block is a page number."""
        text = block.text.strip()
        if not text.isdigit():
            return False
        if block.width > 30 or block.height > 20:
            return False
        if block.y1 > self.page_height - self.FOOTER_ZONE_HEIGHT:
            return True
        return False
    
    def _is_standalone_callout_block(self, block: TextBlock) -> bool:
        """Check if block contains ONLY callout labels without descriptions."""
        text = block.text.strip()
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if not lines:
            return False
        
        for line in lines:
            if len(line) == 1 and line in self.CALLOUT_LABELS:
                continue
            if len(line) == 1 and line in self.CIRCLED_NUMBERS:
                continue
            if len(line) == 1 and line.isdigit():
                continue
            return False
        
        return True
    
    def _is_footnote(self, text: str) -> bool:
        """Check if text is a footnote."""
        text = text.strip()
        if re.match(r'^\*\d+(\s|\n)', text):
            return True
        if re.match(r'^\*\d+$', text):
            return True
        return False
    
    def _is_callout_label(self, text: str) -> bool:
        """Check if text is a callout label."""
        text = text.strip()
        if len(text) == 1:
            if text in self.CALLOUT_LABELS or text in self.CIRCLED_NUMBERS or text.isdigit():
                return True
        if len(text) == 2 and text[1] == '.':
            if text[0] in self.CALLOUT_LABELS or text[0].isdigit():
                return True
        return False
    
    def _is_bullet_marker(self, text: str) -> bool:
        """Check if text is a bullet marker."""
        return text.strip() in self.BULLET_MARKERS
    
    def _starts_with_bullet(self, line: str) -> bool:
        """Check if line starts with a bullet marker."""
        for marker in self.BULLET_MARKERS:
            if line.startswith(marker + ' '):
                return True
        return False
    
    def _strip_bullet_prefix(self, line: str) -> str:
        """Remove bullet marker prefix from line."""
        for marker in self.BULLET_MARKERS:
            if line.startswith(marker + ' '):
                return line[len(marker):].strip()
        return line
    
    def _is_step_number(self, text: str) -> bool:
        """Check if text is a step number like '1', '2', '1:', '2:'."""
        text = text.strip()
        if text.isdigit() and int(text) <= 20:
            return True
        if re.match(r'^\d+:$', text):
            return True
        return False
    
    def process_text(self, text: str, is_bold: bool = False) -> str:
        """
        Process text content with all formatting enhancements.
        """
        lines = text.split('\n')
        processed_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
            
            # Check for step numbers (standalone "1", "2" or "1:", "2:")
            if self._is_step_number(line):
                step_num = line.rstrip(':')
                # Look for step content on next lines
                content_parts = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if not next_line:
                        j += 1
                        continue
                    if self._is_step_number(next_line):
                        break
                    if self._is_callout_label(next_line):
                        break
                    content_parts.append(next_line)
                    j += 1
                    # Only take first logical block
                    if len(content_parts) >= 2:
                        break
                
                if content_parts:
                    content = ' '.join(content_parts)
                    content = self._format_menu_paths(content)
                    processed_lines.append(f"{step_num}. {content}")
                    i = j
                else:
                    i += 1
                continue
            
            # Check for callout label
            if self._is_callout_label(line):
                description_lines = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if self._is_callout_label(next_line):
                        break
                    if self._is_step_number(next_line):
                        break
                    if next_line:
                        description_lines.append(next_line)
                    j += 1
                
                if description_lines:
                    description = ' '.join(description_lines)
                    description = self._format_menu_paths(description)
                    processed_lines.append(f"**{line}:** {description}")
                    i = j
                else:
                    processed_lines.append(line)
                    i += 1
                continue
            
            # Check for standalone bullet marker
            if self._is_bullet_marker(line):
                content_lines = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if self._is_bullet_marker(next_line):
                        break
                    if self._starts_with_bullet(next_line):
                        break
                    if self._is_step_number(next_line):
                        break
                    if next_line:
                        content_lines.append(next_line)
                    j += 1
                
                if content_lines:
                    content = ' '.join(content_lines)
                    content = self._format_line(content)
                    processed_lines.append(f"- {content}")
                    i = j
                else:
                    i += 1
                continue
            
            # Check for inline bullet
            if self._starts_with_bullet(line):
                content = self._strip_bullet_prefix(line)
                content = self._format_line(content)
                processed_lines.append(f"- {content}")
                i += 1
                continue
            
            # Regular line - apply formatting
            formatted = self._format_line(line)
            processed_lines.append(formatted)
            i += 1
        
        return '\n'.join(processed_lines)
    
    def _format_line(self, line: str) -> str:
        """Apply inline formatting to a line."""
        # Format voice commands
        line = self._format_voice_commands(line)
        # Format menu paths
        line = self._format_menu_paths(line)
        return line
    
    def _format_voice_commands(self, text: str) -> str:
        """Format voice commands as inline code."""
        def replace_command(match):
            cmd = match.group(1).strip()
            return f'`{cmd}`'
        return self.VOICE_COMMAND_PATTERN.sub(replace_command, text)
    
    def _format_menu_paths(self, text: str) -> str:
        """Format menu paths with bold brackets."""
        # Replace [MenuName] with **[MenuName]**
        # But avoid double-bolding
        def replace_menu(match):
            menu = match.group(1)
            return f'**[{menu}]**'
        return self.MENU_PATTERN.sub(replace_menu, text)
    
    def _format_heading(self, text: str, level: int) -> str:
        """Format text as a markdown heading."""
        prefix = '#' * level
        return f"{prefix} {text}"
    
    def process_note_block(self, text: str) -> str:
        """Process Note: or Important: blocks as blockquotes."""
        lines = text.split('\n')
        processed_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('R '):
                processed_lines.append(f"> - {stripped[2:]}")
            elif stripped.startswith('Note:'):
                processed_lines.append(f"> **Note:**")
            elif stripped.startswith('Important:'):
                processed_lines.append(f"> ⚠️ **Important:**")
            elif stripped:
                processed_lines.append(f"> {stripped}")
        
        return '\n'.join(processed_lines)
    
    def process_footnote_block(self, text: str) -> str:
        """Process footnote blocks."""
        lines = text.strip().split('\n')
        processed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            match = re.match(r'^\*(\d+)\s*(.*)', line)
            if match:
                num = match.group(1)
                content_parts = []
                if match.group(2):
                    content_parts.append(match.group(2))
                
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if re.match(r'^\*\d+', next_line):
                        break
                    if next_line:
                        content_parts.append(next_line)
                    j += 1
                
                content = ' '.join(content_parts)
                processed_lines.append(f"[^{num}]: {content}")
                i = j
            else:
                processed_lines.append(line)
                i += 1
        
        return '\n'.join(processed_lines)
    
    def extract_page_text(self, page: fitz.Page) -> str:
        """Extract and process text from a page with full formatting."""
        blocks = self.extract_text_blocks(page)
        
        content_blocks = []
        footnote_blocks = []
        
        for block in blocks:
            if block.block_type in ("header", "footer", "page_number", "standalone_callout"):
                continue
            elif block.block_type == "footnote":
                footnote_blocks.append(block)
            else:
                content_blocks.append(block)
        
        content_blocks.sort(key=lambda b: (b.y0, b.x0))
        footnote_blocks.sort(key=lambda b: (b.y0, b.x0))
        
        processed_parts = []
        prev_heading_level = 0
        
        for block in content_blocks:
            text = block.text
            
            # Handle headings
            if block.heading_level > 0:
                heading_text = text.strip()
                formatted = self._format_heading(heading_text, block.heading_level)
                
                # Add separator before major sections (H1)
                if block.heading_level == 1 and processed_parts:
                    processed_parts.append("\n---\n")
                
                processed_parts.append(formatted)
                prev_heading_level = block.heading_level
                continue
            
            # Handle Note/Important blocks
            if text.strip().startswith('Note:') or text.strip().startswith('Important:'):
                processed_parts.append(self.process_note_block(text))
                continue
            
            # Regular content
            processed = self.process_text(text, block.is_bold)
            if processed.strip():
                processed_parts.append(processed)
        
        # Add footnotes
        if footnote_blocks:
            processed_parts.append("\n---\n**Footnotes:**")
            for block in footnote_blocks:
                processed_parts.append(self.process_footnote_block(block.text))
        
        return '\n\n'.join(processed_parts)


def extract_page(pdf_path: Path, page_num: int) -> str:
    """Extract text from a specific page."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    extractor = TextExtractor()
    text = extractor.extract_page_text(page)
    doc.close()
    return text


def extract_pdf_to_markdown(pdf_path: Path, output_dir: Path) -> Dict:
    """Extract all pages from a PDF and save as markdown files."""
    from tqdm import tqdm
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_name = pdf_path.stem
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    extractor = TextExtractor()
    
    stats = {
        "pdf_name": pdf_name,
        "total_pages": total_pages,
        "pages_extracted": 0,
        "pages_empty": 0,
        "total_chars": 0,
        "output_dir": str(output_dir)
    }
    
    print(f"\nExtracting text from: {pdf_name}")
    print(f"Output directory: {output_dir}")
    print(f"Total pages: {total_pages}\n")
    
    for page_num in tqdm(range(total_pages), desc="Extracting pages"):
        page = doc[page_num]
        text = extractor.extract_page_text(page)
        
        if text.strip():
            md_filename = f"{pdf_name}_page_{page_num + 1:04d}.md"
            md_path = output_dir / md_filename
            
            # Page number as comment for reference
            md_content = f"<!-- Page {page_num + 1} -->\n\n{text}"
            
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            
            stats["pages_extracted"] += 1
            stats["total_chars"] += len(text)
        else:
            stats["pages_empty"] += 1
    
    doc.close()
    
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Pages extracted: {stats['pages_extracted']}")
    print(f"Pages empty: {stats['pages_empty']}")
    print(f"Total characters: {stats['total_chars']:,}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    return stats


def compare_extraction(pdf_path: Path, page_num: int):
    """Compare original vs improved extraction for a page."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    original_text = page.get_text()
    
    extractor = TextExtractor()
    improved_text = extractor.extract_page_text(page)
    
    doc.close()
    
    print(f"\n{'='*80}")
    print(f"Page {page_num + 1} - ORIGINAL")
    print(f"{'='*80}")
    print(original_text)
    
    print(f"\n{'='*80}")
    print(f"Page {page_num + 1} - ENHANCED MARKDOWN")
    print(f"{'='*80}")
    print(improved_text)


if __name__ == "__main__":
    import sys
    
    project_root = Path(__file__).parent.parent.parent
    pdf_path = project_root / "data" / "pdfs" / "HomeHawkApp_Users_Guide_CC1803YK9100_ENG.pdf"
    output_dir = project_root / "temp_extraction" / "markdown"
    
    if not pdf_path.exists():
        print(f"PDF not found at {pdf_path}")
        sys.exit(1)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        test_pages = [5, 20, 39, 40]  # Pages 6, 21, 40, 41
        for page_num in test_pages:
            compare_extraction(pdf_path, page_num)
            print("\n")
    else:
        extract_pdf_to_markdown(pdf_path, output_dir)
