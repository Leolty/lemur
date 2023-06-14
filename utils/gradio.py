import gradio as gr
from utils.inference import shared_state
import re

def convert_to_markdown(text):
    text = text.replace("$", "&#36;")

    def replace_leading_tabs_and_spaces(line):
        new_line = []

        for char in line:
            if char == "\t":
                new_line.append("&#9;")
            elif char == " ":
                new_line.append("&nbsp;")
            else:
                break
        return "".join(new_line) + line[len(new_line) :]

    markdown_text = ""
    lines = text.split("\n")
    in_code_block = False

    for line in lines:
        if in_code_block is False and line.startswith("```"):
            in_code_block = True
            markdown_text += "```\n"
        elif in_code_block is True and line.startswith("```"):
            in_code_block = False
            markdown_text += "```\n"
        elif in_code_block:
            markdown_text += f"{line}\n"
        else:
            line = replace_leading_tabs_and_spaces(line)
            line = re.sub(r"^(#)", r"\\\1", line)
            markdown_text += f"{line}  \n"

    return markdown_text

def reset_textbox():
    return gr.update(value=""), ""

def cancel_outputing():
    shared_state.interrupt()
    textbox = reset_textbox()
    return "Stop Done"

def reset_state():
    return [], [], "Reset Done"

def transfer_input(inputs):
    textbox = reset_textbox()
    return (
        inputs,
        gr.update(value=""),
        gr.Button.update(visible=True),
        gr.Button.update(visible=True)
    )

def delete_last_conversation(chatbot, history):
    if len(chatbot) > 0:
        chatbot.pop()

    if len(history) > 0:
        history.pop()

    return (
        chatbot,
        history,
        "Delete Done",
    )