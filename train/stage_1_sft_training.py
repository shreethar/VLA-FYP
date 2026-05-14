import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unsloth import FastVisionModel
import torch
from data.stage_1_datasets import build_iterable_datasets
from transformers import TextStreamer

# Import the pretrained base model and the tokenizer from Unsloth
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3.5-4B",
    load_in_4bit = False,
    use_gradient_checkpointing = "unsloth"
)
print("Imported Base Model")

# Add LoRA adapters to the model
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers = True,
    finetune_language_layers = True,
    finetune_attention_modules = True,
    finetune_mlp_modules = True,

    r = 16,
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    random_state = 3047,
    use_rslora = False,
    loftq_config = None,
    target_modules = "all-linear",
)
print("Added LoRA Adapters")

# Load the dataset
dataset_dict = build_iterable_datasets()
train_data = dataset_dict["train"]

train_iter = iter(train_data)
train_next = next(train_iter)

# Test on inference
FastVisionModel.for_inference(model)
input_text = tokenizer.apply_chat_template(
    [train_next["messages"][0]],
    add_generation_prompt = True,
    tokenize = False,
    return_dict = True
)
print("-"*20)
print("Input Text:")
print(input_text)
print("Input Text Ended")
print("-"*20)

inputs = tokenizer.apply_chat_template(
    [train_next["messages"][0]],
    add_generation_prompt = True,
    tokenize = True,
    return_dict = True,
    return_tensors = "pt",
).to(model.device)

text_streamer = TextStreamer(tokenizer,skip_prompt = True)
outputs = model.generate(
    **inputs,
    streamer = text_streamer,
    max_new_tokens = 256,
    temperature = 1.5,
    min_p = 0.1,
    use_cache = True
)