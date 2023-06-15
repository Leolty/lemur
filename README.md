# Lemur 🦥 

Lemur is a chatbot model based on the [LLaMA](https://arxiv.org/abs/2302.13971v1) model, further fine-tuned using [LoRA](https://arxiv.org/abs/2106.09685) on several openly available datasets.

> **Usage Note**: Lemur is specifically designed and trained for the **exclusive** purpose of our final project for CSE 256 Statistical Natural Lang Processing at UCSD. It is not intended for **any** commercial usage or widespread deployment. Testing and exploration are permitted, but we request that you limit your use to this purpose only. Please respect these guidelines to maintain the integrity of our project and its intended use.

![demo_gif](assets/demo_speedup.gif)

## Fine-tuning Instructions

### Step 1: Dataset Unpacking

Unzip the dataset files stored in the `data/` directory using the following command:

```bash
gunzip data/data_all.json.gz
```

### Step 2: Installing Dependencies

Download the necessary dependencies specified in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Step 3: Model Fine-tuning

Run `finetune.py`. You have the flexibility to adjust the training arguments as needed.

For instance, to fine-tune the model with a learning rate of `1e-4`, and accommodate larger GPU memory to support increased micro batch size (e.g., `32`) and batch size (e.g., `128`), use the command:

```bash
python finetune.py --learning_rate 1e-4 --micro_batch_size 32 --batch_size 128
```

## Model Inference

### Step 1: LLaMA Model Conversion

Convert the LLaMA-7B model to HuggingFace format (see instructions [here](https://huggingface.co/docs/transformers/main/model_doc/llama)) and save it in the `./llama-7B` directory. 

Please note, due to licensing restrictions, we cannot provide the LLaMA model directly. You should download it from the official [LLaMA](https://github.com/facebookresearch/llama) site after completing their application form.

### Step 2: Check LoRA Model

Verify the LoRA model checkpoint at `./lemur-7B`, which should be around 68.38MB.

Keep in mind that this is only the LoRA model (adaptor). You need to merge the LLaMA andthe LoRA model to assemble the final Lemur-7B model.

### Step 3: Run the Application

Execute the `app.py` script with the correct model paths as the arguments:

```bash
python app.py --base_model llama-7B --lora_model lemur-7B
```

## Hardware Requirements

- 1x NVIDIA 4090 GPU (24GB VRAM)

When using the default parameters in [`finetune.py`](finetune.py), the fine-tuning process will require approximately 23.5GB VRAM, running for around 33 hours on a single Nvidia 4090 GPU.




