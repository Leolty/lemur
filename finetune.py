"""
To avoid bug, we first need to use the previous version of PEFT

pip uninstall peft -y
pip install git+https://github.com/huggingface/peft.git@e536616888d51b453ed354a6f1e243fecb02ea08
"""


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
import glob
import json
import sys
import random
import transformers
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model, get_peft_model_state_dict
from utils.train import print_trainable_parameters, generate_and_tokenize_prompt

# set global seed
random.seed(42)

MAX_LENGTH = 512
VAL_SET_SIZE = 512
LEARNING_RATE = 2e-4
BATCH_SIZE = 64
MICRO_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 1
SAMPLE_NUM = 15000
OUTPUT_DIR = "./output/llama_7B"
BASE_MODEL = "llama-7B"

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL, add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token   
tokenizer.padding_side = "left"

# Load Model
device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

model = LlamaForCausalLM.from_pretrained(BASE_MODEL, load_in_8bit=True, device_map=device_map)

model = prepare_model_for_int8_training(model)

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

# load all the data (.jsonl) under ./data folder

data_part_1, data_part_2 = [], []
for file in glob.glob("./data/*.jsonl"):
    sub_data = []
    with open(file, "r") as f:
        for line in f:
            sub_data.append(json.loads(line))
        
        if len(sub_data) > SAMPLE_NUM:
            random.shuffle(sub_data)    
            sub_data_1, sub_data_2 = sub_data[:SAMPLE_NUM], sub_data[SAMPLE_NUM:SAMPLE_NUM*2]
        else:
            sub_data_1, sub_data_2 = sub_data, sub_data
    
    data_part_1.extend(sub_data_1)
    data_part_2.extend(sub_data_2)
    

# shuffle the data
random.shuffle(data_part_1)
random.shuffle(data_part_2)

# combine the data
all_data = data_part_1 + data_part_2

# re-index the data
for i, d in tqdm(enumerate(data_part_1), total=len(data_part_1)):
    d["idx"] = i

for i, d in tqdm(enumerate(data_part_2), total=len(data_part_2)):
    d["idx"] = i + len(data_part_1)

json.dump(data_part_1, open("./data/data_1.json", "w"))
json.dump(data_part_2, open("./data/data_2.json", "w"))
json.dump(all_data, open("./data/data_all.json", "w"))

# load all the data with huggingface dataset
from datasets import load_dataset

data = load_dataset("json", data_files="./data/data_all.json")

train_val = data["train"].train_test_split(
    test_size=VAL_SET_SIZE, shuffle=True, seed=42
)
train_data = train_val["train"].shuffle().map(lambda x: generate_and_tokenize_prompt(x, tokenizer, max_length=MAX_LENGTH))
val_data = train_val["test"].shuffle().map(lambda x: generate_and_tokenize_prompt(x, tokenizer, max_length=MAX_LENGTH))



trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        per_device_eval_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=50,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=2,
        evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
        save_strategy="steps",
        eval_steps=100 if VAL_SET_SIZE > 0 else None,
        save_steps=100,
        output_dir=OUTPUT_DIR,
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
model.save_pretrained(OUTPUT_DIR)
