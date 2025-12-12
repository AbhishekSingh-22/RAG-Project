from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredPDFLoader
from pathlib import Path

ROOT_DIR = Path().resolve() # Root directory of the project
DATA_DIR = ROOT_DIR.joinpath("data") # Data directory
PDF_DIR = DATA_DIR.joinpath("pdfs") # PDFs directory

loader = UnstructuredPDFLoader("D:\Panasonic\main_project\data\pdfs\CS_NU18VKYW.pdf", mode="paged")   # uses partition_pdf internally
docs = loader.load()

# docs is a list of LangChain Documents. To inspect unstructured elements use unstructured.partition_pdf (next example)
print(len(docs))
print(docs[1].page_content)
# print(docs[3])
