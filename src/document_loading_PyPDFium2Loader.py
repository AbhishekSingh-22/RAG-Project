from langchain_community.document_loaders import PyPDFium2Loader

loader = PyPDFium2Loader("D:\Panasonic\main_project\data\pdfs\CS_NU18VKYW.pdf")
documents = loader.load()  # one Document per page by default

print(f"Pages loaded: {len(documents)}")
print("\nFirst page preview:")
print(documents[1].page_content)
