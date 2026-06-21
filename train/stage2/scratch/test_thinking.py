import os
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "/home/ubuntu/VLA-FYP/train/stage2/checkpoints/stage2_decoupled_mini/step_000100"
adapter_path = os.path.join(model_path, "teacher_lora")
base_path = "shreethar/stage1_unsloth"

print(f"[*] Loading model and processor...")
processor = AutoProcessor.from_pretrained(base_path)
base_model = AutoModelForImageTextToText.from_pretrained(
    base_path,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map=device
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

STAGE2_QA_SYSTEM = (
    "You are a robot manipulation assistant. Answer questions about robot tasks, "
    "object affordances, spatial relationships, and manipulation strategies based "
    "on the provided image or video frame. "
    "If reasoning, think step-by-step. "
    "Finally, output the answer after </think>."
)

q = "current goal is: Please remove the chips from the basket Q: immediate next step?"
prompt_text = f"{STAGE2_QA_SYSTEM}\n\n{q}"

def test_generate(temp, rep_penalty):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text}
            ]
        }
    ]
    formatted = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    inputs = processor(
        text=[formatted],
        images=None,
        videos=None,
        return_tensors="pt"
    )
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
            
    gen_kwargs = {
        "max_new_tokens": 200,
        "do_sample": True,
        "temperature": temp,
    }
    if rep_penalty is not None:
        gen_kwargs["repetition_penalty"] = rep_penalty
        
    gen_kwargs["eos_token_id"] = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    prompt_len = inputs['input_ids'].shape[1]
    gen_text = processor.tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=False)
    return gen_text

print("\n--- CONFIG A: temp=0.1, repetition_penalty=1.2 ---")
outA = test_generate(0.1, 1.2)
print(repr(outA))

print("\n--- CONFIG B: temp=0.1, repetition_penalty=1.0 ---")
outB = test_generate(0.1, 1.0)
print(repr(outB))

print("\n--- CONFIG C: temp=0.7, repetition_penalty=1.0 ---")
outC = test_generate(0.7, 1.0)
print(repr(outC))
