def prompt_value_to_huggingface(tokenizer: AutoTokenizer, prompt_value: ChatPromptValue):
    """ Rajoute les tokens necessaries pour que le template de chat soit compatible avec HuggingFace. """
    messages = (message for message in prompt_value.to_messages())

    # Convert ai to assistant, human to user (message.type)
    mapping = {"ai": "assistant", "human": "user", "system": "system"}
    new_messages = []
    for message in messages:
        message.type = mapping[message.type]
        new_messages.append(message)

    chat = [{"role": message.type, "content": message.content} for message in new_messages]
    string = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, )

    return string