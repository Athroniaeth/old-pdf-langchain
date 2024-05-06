import io

import fitz
import gradio as gr
from PIL import Image


def pdf_to_image(pdf_path: str, page_number: int = 0):
    """ Convertit le PDF charger dans Gradio en une image. """
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_number)

        pix = page.get_pixmap()
        img = pix.tobytes("ppm")
        return Image.open(io.BytesIO(img))


def update_page(pdf_path, current_page: int = 0):
    """ Change la page courant du PDF charger dans Gradio. """
    if pdf_path is None:
        return None, 0

    with fitz.open(pdf_path) as doc:
        max_pages = doc.page_count

    next_page = (current_page + 1) % max_pages
    return pdf_to_image(pdf_path, next_page), next_page


class PDFViewerInterface(gr.Blocks):
    image: gr.Image
    file_input: gr.File
    page_number: gr.Number

    button: gr.Button

    def __init__(self):
        super().__init__()

        with gr.Row():
            self.image = gr.Image()
            self.file_input = gr.File()
            self.page_number = gr.Number(label="Page Number", value=0)

        self.button = gr.Button("Next Page")

        inputs = [self.file_input, self.page_number]
        outputs = [self.image, self.page_number]

        self.file_input.change(update_page, inputs=inputs, outputs=outputs)
        self.button.click(update_page, inputs=inputs, outputs=outputs)
        self.button.click(update_page, inputs=inputs, outputs=outputs)
