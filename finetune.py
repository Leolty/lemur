"""
To avoid bug, we first need to use the previous version of PEFT

pip uninstall peft -y
pip install git+https://github.com/huggingface/peft.git@e536616888d51b453ed354a6f1e243fecb02ea08
"""


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import sys
import random
import argparse
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model, get_peft_model_state_dict
from utils.train import print_trainable_parameters, generate_and_tokenize_prompt

# set global seed
random.seed(42)

parser = argparse.ArgumentParser()

MAX_LENGTH = 512
VAL_SET_SIZE = 512
LEARNING_RATE = 2e-4
BATCH_SIZE = 64
MICRO_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 1
WARMUP_STEPS = 50
EVAL_STEPS = 100
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
OUTPUT_DIR = "./output/llama_7B"
BASE_MODEL = "llama-7B"

parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
parser.add_argument("--val_set_size", type=int, default=VAL_SET_SIZE)
parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
parser.add_argument("--micro_batch_size", type=int, default=MICRO_BATCH_SIZE)
parser.add_argument("--epochs", type=int, default=EPOCHS)
parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS)
parser.add_argument("--eval_steps", type=int, default=EVAL_STEPS)
parser.add_argument("--lora_r", type=int, default=LORA_R)
parser.add_argument("--lora_alpha", type=int, default=LORA_ALPHA)
parser.add_argument("--lora_dropout", type=float, default=LORA_DROPOUT)
parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
parser.add_argument("--base_model", type=str, default=BASE_MODEL)

args = parser.parse_args()

MAX_LENGTH = args.max_length
VAL_SET_SIZE = args.val_set_size
LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
MICRO_BATCH_SIZE = args.micro_batch_size
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = args.epochs
WARMUP_STEPS = args.warmup_steps
EVAL_STEPS = args.eval_steps
LORA_R = args.lora_r
LORA_ALPHA = args.lora_alpha
LORA_DROPOUT = args.lora_dropout
OUTPUT_DIR = args.output_dir
BASE_MODEL = args.base_model


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
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "down_proj",
        "gate_proj",
        "up_proj"
    ],
    lora_dropout=LORA_DROPOUT,
    bias="none", 
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
    
print_trainable_parameters(model)

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
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=2,
        evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
        save_strategy="steps",
        eval_steps=EVAL_STEPS if VAL_SET_SIZE > 0 else None,
        save_steps=EVAL_STEPS,
        output_dir=OUTPUT_DIR,
        save_total_limit=100,
        load_best_model_at_end=True if VAL_SET_SIZE > 0 else False
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