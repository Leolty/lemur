# /home/shibo/llama-ckpts/13B
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
import glob
import json
import sys
import random
import transformers
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, get_peft_model_state_dict
from variables import *
from utils import print_trainable_parameters

# set global seed
random.seed(42)


# quantize config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = LlamaTokenizer.from_pretrained("llama-13B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model = LlamaForCausalLM.from_pretrained("llama-13B", quantization_config=bnb_config, device_map={"":0})

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "down_proj",
        "gate_proj",
        "up_proj"
    ],
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)


model = get_peft_model(model, config)
print_trainable_parameters(model)

# # load all the data (.jsonl) under ./data folder


data = []
rest_data = []
for file in glob.glob("./data/*.jsonl"):
    sub_data = []
    with open(file, "r") as f:
        for line in f:
            sub_data.append(json.loads(line))
        
        # if the dataset > 30000, sample 30000
        if len(sub_data) > 50000:
            random.shuffle(sub_data)
            sub_data, rest_sub_data = sub_data[:50000], sub_data[50000:]
    
    data.extend(sub_data)
    rest_data.extend(rest_sub_data)

# shuffle the data
random.shuffle(data)
random.shuffle(rest_data)

# re-index the data and rest_data
for i, d in tqdm(enumerate(data), total=len(data)):
    d["idx"] = i

for i, d in tqdm(enumerate(rest_data), total=len(rest_data)):
    d["idx"] = i + len(data)

json.dump(data, open("./data/all_data.json", "w"))
json.dump(rest_data, open("./data/unused_rest_data.json", "w"))

# load all the data with huggingface dataset
from datasets import load_dataset

data = load_dataset("json", data_files="./data/all_data.json")

# Data Preprocess
def generate_prompt(data_point):
    return data_point["input"]


def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }


def generate_and_tokenize_prompt(data_point):
    prompt = generate_prompt(data_point)
    return tokenize(prompt)

train_val = data["train"].train_test_split(
    test_size=VAL_SET_SIZE, shuffle=True, seed=42
)
train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        per_device_eval_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
        save_strategy="steps",
        eval_steps=300 if VAL_SET_SIZE > 0 else None,
        save_steps=300,
        output_dir="./output",
        save_total_limit=100,
        load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
        optim="paged_adamw_8bit"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

trainer.train()

# save the model
model.save_pretrained("./output")
