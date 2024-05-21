import io
import os

import fitz
import gradio as gr
from PIL import Image

from src.client import RagClient


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


class Interface(gr.Blocks):
    rag_client: RagClient = None

    def __init__(
            self,
            model_id: str,
            hf_token: str = None,
            quantization_int4: bool = True,
            id_prompt_rag: str = "athroniaeth/rag-prompt",
            id_prompt_contextualize: str = "athroniaeth/contextualize-prompt",
            models_kwargs: dict = None,
            search_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        with self:
            self.rag_client = RagClient(
                model_id=model_id,
                hf_token=hf_token,
                quantization_int4=quantization_int4,
                id_prompt_rag=id_prompt_rag,
                id_prompt_contextualize=id_prompt_contextualize,
                models_kwargs=models_kwargs,
                search_kwargs=search_kwargs,
            )

            with gr.Row(equal_height=True):
                with gr.Column(scale=2):
                    self.input_file = gr.File(visible=True)
                    self.input_image = gr.Image(visible=True)
                    self.input_file.change(update_page, inputs=[self.input_file], outputs=[self.input_image])
                    self.input_file.change(self.rag_client.load_pdf, inputs=[self.input_file])

                with gr.Column(scale=5):
                    self.chatbot = gr.Chatbot()
                    self.msg = gr.Textbox()
                    self.clear = gr.ClearButton([self.msg, self.chatbot])
                    self.msg.submit(
                        self.rag_client.respond,
                        inputs=[self.msg, self.chatbot],
                        outputs=[self.msg, self.chatbot, self.input_image]
                    )


    def respond(self, msg, chatbot):
         self.rag_client.respond(msg, chatbot)

def launch_gradio():
    interface = Interface(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
        hf_token=os.environ["HF_TOKEN"],
    )

    interface.launch()
