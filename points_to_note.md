# Requirements installed
- unstructured Library for pdf loading
- poppler of OCR, used by pdf2image

# findings
- Pdf is not only text. More of a graphical visual documentation, which means basic pdf loading fails there, for that I have to switch either to unstructuredPDFLoader or PyPDFium2Loader 
    - Reference -> [Medium blog](https://medium.com/@sangitapokhrel911/different-methods-to-read-pdf-files-in-langchain-5d547206bcef)

# Markdown files