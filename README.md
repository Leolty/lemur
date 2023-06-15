# Lemur 🦥

Lemur is a chatbot based on [LLaMA](https://arxiv.org/abs/2302.13971v1) fine-tuned using [LoRA](https://arxiv.org/abs/2106.09685) with several collected open-sourced datasets.

## Fine-tuning

1. Unzip the dataset files in `data/` directory.

```bash
gunzip data/data_all.json.gz
```

2. Download dependencies.

```bash
pip install -r requirements.txt
```

3. Run the `finetune.py`.

```bash
python finetune.py
```

You can customize the training arguments in [`finetune.py`](finetune.py).

## Inference

1. Transfer LLaMA-7B to a huggingface model and save it as `./llama-7B`. 

You should download the LLaMA model through applying by filling the form at offcial [LLaMA](https://github.com/facebookresearch/llama), we are not allowed to share the model weights due to the license.

2. Check the LoRA model at `./lemur-7B`.

Noted that this is only the LoRA model, you should merge the LLaMA model and LoRA model to get the final Lemur model.

3. Check the `BASE_MODEL` and `LORA_MODEL` in [`app.py`](app.py) are correct paths to the LLaMA model and LoRA model, and run the `app.py`.

```bash
python app.py
```

## Hardware Requirements

- 1x NVIDIA 3090 GPU (24GB VRAM)

If you use the parameters in [`finetune.py`](finetune.py), the fine-tuning process will require about 23.5GB VRAM for ~35 hours.






