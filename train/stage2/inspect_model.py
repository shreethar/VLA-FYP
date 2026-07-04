import torch
from transformers import AutoModelForImageTextToText

base_model_name = "shreethar/stage1_unsloth"
model = AutoModelForImageTextToText.from_pretrained(base_model_name, trust_remote_code=True)

print("Base model type:", type(model))
print("model attributes:", dir(model))
if hasattr(model, 'model'):
    print("model.model type:", type(model.model))
    print("model.model attributes:", dir(model.model))
    if hasattr(model.model, 'language_model'):
        print("has language_model!")
    else:
        print("NO language_model!")
