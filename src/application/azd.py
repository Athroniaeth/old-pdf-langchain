import gradio as gr


def update_chat(messages, message):
    return messages + [f"Utilisateur: {message}", f"Bot: Echo de {message}"]


def send_button_click():
    # Vous pouvez définir les actions des boutons ici
    print("Bouton cliqué!")


with gr.Blocks() as app:
    with gr.Row():
        with gr.Column(scale=2):
            chat_box = gr.Chatbot()
            message_box = gr.Textbox(label="Votre message")
            send_button = gr.Button("Envoyer")
            send_button.click(fn=update_chat, inputs=[chat_box, message_box], outputs=chat_box)
        with gr.Column(scale=1):
            with gr.Column(scale=1):
                for i in range(3):  # Ajouter plusieurs boutons comme exemple
                    b = gr.Button(f"Bouton {i + 1}", elem_id=f"button{i + 1}")
                    b.click(send_button_click)
                gr.HTML('<iframe width="100%" height="100%" src="https://exemple.com"></iframe>')


app.launch()
