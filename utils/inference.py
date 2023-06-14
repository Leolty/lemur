import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel
from typing import Iterator
from variables import SYSTEM, HUMAN, AI


def load_tokenizer_and_model(base_model, adapter_model, load_8bit=True):
    """
    Loads the tokenizer and chatbot model.

    Args:
        base_model (str): The base model to use (path to the model).
        adapter_model (str): The LoRA model to use (path to LoRA model).
        load_8bit (bool): Whether to load the model in 8-bit mode.
    """
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except:  
        pass
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if adapter_model is not None:
            model = PeftModel.from_pretrained(
                model,
                adapter_model
            )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device}
        )
        if adapter_model is not None:
            model = PeftModel.from_pretrained(
                model,
                adapter_model,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        if adapter_model is not None:
            model = PeftModel.from_pretrained(
                model,
                adapter_model
            )

    model.eval()
    return tokenizer, model, device

class State:
    interrupted = False

    def interrupt(self):
        self.interrupted = True

    def recover(self):
        self.interrupted = False

shared_state = State()

def decode(
    input_ids: torch.Tensor,
    model: PeftModel,
    tokenizer: LlamaTokenizer,
    stop_words: list,
    max_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Iterator[str]:
    generated_tokens = []
    past_key_values = None
    
    for _ in range(max_length):
        with torch.no_grad():
            if past_key_values is None:
                outputs = model(input_ids)
            else:
                outputs = model(input_ids[:, -1:], past_key_values=past_key_values)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

        # apply temperature
        logits /= temperature

        probs = torch.softmax(logits, dim=-1)
        # apply top_p
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0

        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)

        input_ids = torch.cat((input_ids, next_token), dim=-1)

        generated_tokens.append(next_token[0].item())
        text = tokenizer.decode(generated_tokens)

        yield text
        if any([x in text for x in stop_words]):
            return


def get_prompt_with_history(text, history, tokenizer, max_length=2048):
    prompt = SYSTEM
    history = [f"\n{HUMAN} {x[0]}\n{AI} {x[1]}" for x in history]
    history.append(f"\n{HUMAN} {text}\n{AI}")
    history_text = ""
    flag = False
    for x in history[::-1]:
        if (
            tokenizer(prompt + history_text + x, return_tensors="pt")["input_ids"].size(
                -1
            )
            <= max_length
        ):
            history_text = x + history_text
            flag = True
        else:
            break
    if flag:
        return prompt + history_text, tokenizer(
            prompt + history_text, return_tensors="pt"
        )
    else:
        return None

def is_stop_word_or_prefix(s: str, stop_words: list) -> bool:
    for stop_word in stop_words:
        if s.endswith(stop_word):
            return True
        for i in range(1, len(stop_word)):
            if s.endswith(stop_word[:i]):
                return True
    return False