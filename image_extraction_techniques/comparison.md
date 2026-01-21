# Comparison of different image extraction techniques

## 1. dictionary based image extraction via PyMuPDF library

#### Overview

This technique uses PyMuPDF's `page.get_text("dict")` to extract images with their **exact positions** on the page, then expands the extraction area to include surrounding context (headers, captions, annotations).

---

#### Key Characteristics

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