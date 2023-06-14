import os
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
import gradio as gr
import logging

from utils.inference import load_tokenizer_and_model, decode, \
    get_prompt_with_history, is_stop_word_or_prefix

from utils.gradio import reset_textbox, cancel_outputing, transfer_input, \
    delete_last_conversation, reset_state, convert_to_markdown



# set variables
BASE_MODEL = "llama-7B"
LORA_MODEL = "output/llama_lora_7B/checkpoint-3700"


print("Loading model...")

import time

start = time.time()

tokenizer, model, device = load_tokenizer_and_model(
    base_model=BASE_MODEL,
    adapter_model=LORA_MODEL,
    load_8bit=True,
)

print("Model loaded in {} seconds.".format(time.time() - start))


def predict(
    text,
    chatbot,
    history,
    top_p,
    temperature,
    max_length_tokens,
    max_context_length_tokens,
):
    if text == "":
        yield chatbot, history, "Empty context."
        return

    inputs = get_prompt_with_history(
        text, history, tokenizer, max_length=max_context_length_tokens
    )
    if inputs is None:
        yield chatbot, history, "Input too long."
        return
    else:
        prompt, inputs = inputs

    input_ids = inputs["input_ids"][:, -max_context_length_tokens:].to(device)
    torch.cuda.empty_cache()

    with torch.no_grad():
        for x in decode(
            input_ids,
            model,
            tokenizer,
            stop_words=["[Human]", "[AI]"],
            max_length=max_length_tokens,
            temperature=temperature,
            top_p=top_p,
        ):
            if is_stop_word_or_prefix(x, ["[Human]", "[AI]"]) is False:
                if "[Human]" in x:
                    x = x[: x.index("[Human]")].strip()
                if "[AI]" in x:
                    x = x[: x.index("[AI]")].strip()
                x = x.strip(" ")
                a, b = [[y[0], convert_to_markdown(y[1])] for y in history] + [
                    [text, convert_to_markdown(x)]
                ], history + [[text, x]]
                yield a, b, "Generating..."

    torch.cuda.empty_cache()
    print(prompt)
    print(x)
    print("=" * 80)
    try:
        yield a, b, "Generate: Success"
    except:
        pass

def retry(
    text,
    chatbot,
    history,
    top_p,
    temperature,
    max_length_tokens,
    max_context_length_tokens,
):
    logging.info("Retry...")
    if len(history) == 0:
        yield chatbot, history, "Empty context."
        return
    chatbot.pop()
    inputs = history.pop()[0]
    for x in predict(
        inputs,
        chatbot,
        history,
        top_p,
        temperature,
        max_length_tokens,
        max_context_length_tokens,
    ):
        yield x


with gr.Blocks(
    theme=gr.themes.Soft(),
    css=".disclaimer {font-variant-caps: all-small-caps;}"
    ) as demo:
    history = gr.State([])
    user_question = gr.State("")
    with gr.Row():
        gr.HTML("<h1>Lemur 🦥</h1>")
        status_display = gr.Markdown("Success", elem_id="status_display")

    with gr.Row(scale=1).style(equal_height=True):
        with gr.Column(scale=5):
            with gr.Row(scale=1):
                chatbot = gr.Chatbot(elem_id="chuanhu_chatbot").style(height=800)
            with gr.Row(scale=1):
                with gr.Column(scale=12):
                    user_input = gr.Textbox(
                        show_label=False, placeholder="Enter text"
                    ).style(container=False)
                with gr.Column(min_width=70, scale=1):
                    submitBtn = gr.Button("📤 Send")
                with gr.Column(min_width=70, scale=1):
                    cancelBtn = gr.Button("⏸️ Stop")

            with gr.Row(scale=1):
                emptyBtn = gr.Button(
                    "🧹 New Conversation",
                )
                retryBtn = gr.Button("🔄 Regenerate")
                delLastBtn = gr.Button("🗑️ Remove Last Turn")
        with gr.Column():
            with gr.Column(min_width=50, scale=1):
                with gr.Tab(label="Parameter Setting"):
                    gr.Markdown("# Parameters")
                    top_p = gr.Slider(
                        minimum=-0,
                        maximum=1.0,
                        value=0.95,
                        step=0.05,
                        interactive=True,
                        label="Top-p",
                    )
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=1,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                    )
                    max_length_tokens = gr.Slider(
                        minimum=0,
                        maximum=512,
                        value=512,
                        step=8,
                        interactive=True,
                        label="Max Generation Tokens",
                    )
                    max_context_length_tokens = gr.Slider(
                        minimum=0,
                        maximum=4096,
                        value=2048,
                        step=128,
                        interactive=True,
                        label="Max History Tokens",
                    )

    predict_args = dict(
        fn=predict,
        inputs=[
            user_question,
            chatbot,
            history,
            top_p,
            temperature,
            max_length_tokens,
            max_context_length_tokens,
        ],
        outputs=[chatbot, history, status_display],
        show_progress=True,
    )
    retry_args = dict(
        fn=retry,
        inputs=[
            user_input,
            chatbot,
            history,
            top_p,
            temperature,
            max_length_tokens,
            max_context_length_tokens,
        ],
        outputs=[chatbot, history, status_display],
        show_progress=True,
    )

    reset_args = dict(fn=reset_textbox, inputs=[], outputs=[user_input, status_display])

    # Chatbot

    transfer_input_args = dict(
        fn=transfer_input,
        inputs=[user_input],
        outputs=[user_question, user_input, submitBtn, cancelBtn],
        show_progress=True,
    )

    submit_event = user_input.submit(**transfer_input_args).then(**predict_args)

    submit_click_event = submitBtn.click(**transfer_input_args).then(**predict_args)

    emptyBtn.click(
        reset_state,
        outputs=[chatbot, history, status_display],
        show_progress=True,
    )
    emptyBtn.click(**reset_args)

    retry_click_event = retryBtn.click(**retry_args)

    cancelBtn.click(
        fn=cancel_outputing, 
        inputs=[],
        outputs=[status_display],
        cancels=[submit_event, submit_click_event]
    )

    delLastBtn.click(
        delete_last_conversation,
        [chatbot, history],
        [chatbot, history, status_display],
        show_progress=True,
    )

demo.title = "Lemur"
demo.queue(max_size=128, concurrency_count=2)
demo.launch()
