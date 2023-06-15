# Lemur 🦥

Lemur is a chatbot based on [LLaMA](https://arxiv.org/abs/2302.13971v1) fine-tuned using [LoRA](https://arxiv.org/abs/2106.09685) with several collected open-sourced datasets.

![demo_gif](assets/demo_speedup.gif)

## Fine-tuning

1. Unzip the dataset files in `data/` directory.

```bash
gunzip data/data_all.json.gz
```

2. Download dependencies.

```bash
pip install -r requirements.txt
```

3. Run the `finetune.py`, and you can feel free to customize the training arguments.

For example, if you want to fine-tune the model with the learning rate of `1e-4`, and you have larger GPU memory to support larger micro batch size (e.g. 32) and larger batch size (e.g. 128), you can run the following command.

```bash
python finetune.py --learning_rate 1e-4 --micro_batch_size 32 --batch_size 128
```

## Inference

1. Transfer LLaMA-7B to a huggingface model and save it as `./llama-7B`. 

You should download the LLaMA model through applying by filling the form at offcial [LLaMA](https://github.com/facebookresearch/llama), we are not allowed to share the model weights due to the license.

2. Check the LoRA model at `./lemur-7B`.

Noted that this is only the LoRA model, you should merge the LLaMA model and LoRA model to get the final Lemur model.

3. Run the `app.py` with the correct model path as the argument.

```bash
python app.py --base_model llama-7B --lora_model lemur-7B
```

## Hardware Requirements

- 1x NVIDIA 3090 GPU (24GB VRAM)

If you use the parameters in [`finetune.py`](finetune.py), the fine-tuning process will require about 23.5GB VRAM for ~35 hours.






