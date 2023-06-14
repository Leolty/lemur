"""
We need to update to the latest version of PEFT

pip uninstall peft -y
pip install -q -U git+https://github.com/huggingface/peft.git

"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
import random
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
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
OUTPUT_DIR = "./output/llama_13B"
BASE_MODEL = "llama-13B"

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL, add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token   
tokenizer.padding_side = "left"

# Load Model
device_map = "auto"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = LlamaForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_config, device_map=device_map)

# Prepare model with lora
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
trainer.train()

# save the model
model.save_pretrained(OUTPUT_DIR)

