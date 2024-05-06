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

    can_insert_file: bool = True

    def __init__(self):
        super().__init__()

        with self:
            self.image = gr.Image(visible=False)
            self.file_input = gr.File(visible=True)
            self.page_number = gr.Number(label="Page Number", value=0)

            self.button = gr.Button("Next Page")

            inputs = [self.file_input, self.page_number]
            outputs = [self.image, self.page_number]

            self.button.click(
                self.invert, inputs=inputs, outputs=[self.image, self.file_input]
            )
            self.file_input.change(update_page, inputs=inputs, outputs=outputs)

    def invert(self, bla, bla2):
        print(bla, bla2)
        self.can_insert_file = not self.can_insert_file
        print(not self.can_insert_file, self.can_insert_file)
        return gr.Image(visible=not self.can_insert_file), gr.File(visible=self.can_insert_file)

if __name__ == "__main__":
    app = PDFViewerInterface()
    app.launch()