from pypdf import PdfReader

def load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text= " "

    for page_number, page in enumerate(reader.pages):
        extracted = page.extract_text()
        if extracted:
            text += f"\n--- Page {page_number + 1} ---\n"
            text += extracted
    return text

