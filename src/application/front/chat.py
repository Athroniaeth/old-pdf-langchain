import random
import time

import gradio as gr


class ChatInterface(gr.Blocks):
    model_llm = None
    model_embeddings = None
    chat_history = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self:
            self.chatbot = gr.Chatbot()
            self.msg = gr.Textbox()
            self.clear = gr.ClearButton([self.msg, self.chatbot])
            self.msg.submit(self.respond, inputs=[self.msg, self.chatbot], outputs=[self.msg, self.chatbot])

    def respond(self, message, chat_history):
        bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
        chat_history.append((message, bot_message))
        time.sleep(2)
        return "", chat_history


if __name__ == "__main__":
    chatbot_demo = ChatInterface()
    chatbot_demo.launch()
