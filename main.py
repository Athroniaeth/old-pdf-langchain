import gradio as gr

from src.application.front.chat import ChatInterface
from src.application.front.pdf_viewer import PDFViewerInterface


class Application(gr.Blocks):
    chat_interface: ChatInterface
    pdf_viewer_interface: PDFViewerInterface

    def __init__(self):
        super().__init__()
        with self:
            self.chat_interface = ChatInterface()
            self.pdf_viewer_interface = PDFViewerInterface()


if __name__ == "__main__":
    app = Application()
    app.launch()