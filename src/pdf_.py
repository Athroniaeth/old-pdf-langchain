import io

import fitz
from PIL import Image
from langchain_core.documents import Document


def parse_documents(list_documents: list[Document]) -> Image:
    for page in list_documents:
        pdf_path = list_documents[0].metadata['file_path']
        pdf_document = fitz.open(pdf_path)
        pdf_content = page.page_content

        pdf_document = highlight_text(pdf_document, pdf_content)
        page = pdf_document.load_page(0)

        pix = page.get_pixmap()
        bytes = pix.tobytes("ppm")
        img = Image.open(io.BytesIO(bytes))
        return img

def highlight_text(document: fitz.Document, text: str) -> fitz.Document:
    for page in document:  # Parcourir chaque page du document
        text_instances = page.search_for(text)

        # Surligner chaque occurrence trouvée
        for inst in text_instances:
            highlight = page.add_highlight_annot(inst)  # Ajouter une annotation de surlignage
            highlight.set_colors(stroke=(1, 1, 0))  # Définir la couleur du surlignage en jaune (RGB)
            highlight.update()  # Mettre à jour l'annotation pour appliquer les modifications

    return document


def pdf_to_image(pdf_path: str, page_number: int = 0):
    """ Convertit le PDF charger dans Gradio en une image. """
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_number)

        pix = page.get_pixmap()
        bytes = pix.tobytes("ppm")
        img = Image.open(io.BytesIO(bytes))
        return img


def update_page(pdf_path, current_page: int = 0):
    """ Change la page courant du PDF charger dans Gradio. """
    if pdf_path is None:
        return None

    with fitz.open(pdf_path) as doc:
        max_pages = doc.page_count

    next_page = (current_page + 1) % max_pages
    return pdf_to_image(pdf_path, next_page)
