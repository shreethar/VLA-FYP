import os
import time
import re
import ast
import torch
import multiprocessing as mp
from datasets import load_dataset
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import textwrap

from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import load_peft_weights, set_peft_model_state_dict

from models.latent_student import LatentStudent
from tqdm import tqdm

def parse_trajectory(text):
    match = re.search(r'\[\[.*?\]\]', text)
    if match:
        try:
            return ast.literal_eval(match.group(0))
        except:
            pass
    return None

def calc_l2_loss(gt, pred):
    if gt is None or pred is None:
        return float('nan')
    if len(gt) != len(pred):
        return float('nan')
    dist = 0
    for g, p in zip(gt, pred):
        dist += np.sqrt((g[0] - p[0])**2 + (g[1] - p[1])**2)
    return dist / len(gt)

def calc_dtw(gt, pred):
    if gt is None or pred is None:
        return float('nan')
    n, m = len(gt), len(pred)
    dtw_matrix = np.full((n + 1, m + 1), float('inf'))
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.sqrt((gt[i-1][0] - pred[j-1][0])**2 + (gt[i-1][1] - pred[j-1][1])**2)
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],
                                          dtw_matrix[i, j-1],
                                          dtw_matrix[i-1, j-1])
    return dtw_matrix[n, m]

def draw_trajectory(image, trajectory):
    if trajectory is None:
        return image
    
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    w, h = img.size
    points = []
    for (x, y) in trajectory:
        px = int(x / 1000.0 * w)
        py = int(y / 1000.0 * h)
        points.append((px, py))
    
    if len(points) > 1:
        draw.line(points, fill="red", width=3)
    
    for pt in points:
        draw.ellipse([pt[0]-4, pt[1]-4, pt[0]+4, pt[1]+4], fill="blue")
    
    return img

def draw_instruction_image(instruction, size=(448, 448)):
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    wrapped = textwrap.fill(instruction, width=35)
    draw.text((20, size[1]//2 - 40), wrapped, fill='black')
    return img

def process_stage1(samples, end_think_token_id):
    processor = AutoProcessor.from_pretrained("shreethar/stage1_unsloth")
    model = AutoModelForImageTextToText.from_pretrained("shreethar/stage1_unsloth", dtype=torch.bfloat16, device_map="cuda")
    results, times = [], []
    for s in tqdm(samples, desc="Stage 1", position=0):
        message = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": s['human']}]}]
        text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = processor(text=[text], images=[s['frames'][0]], return_tensors="pt").to("cuda")
        t0 = time.time()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=512)
        t1 = time.time()
        gen_text = processor.tokenizer.decode(output[0][inputs.input_ids.shape[1]:])
        results.append(parse_trajectory(gen_text))
        times.append(t1 - t0)
    return results, times

def process_teacher(samples, end_think_token_id):
    processor = AutoProcessor.from_pretrained("shreethar/stage1_unsloth")
    model = AutoModelForImageTextToText.from_pretrained("shreethar/stage2_teacher", dtype=torch.bfloat16, device_map="cuda")
    results, times = [], []
    for s in tqdm(samples, desc="Teacher", position=1):
        message = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": s['human']}]}]
        text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        inputs = processor(text=[text], images=[s['frames'][0]], return_tensors="pt").to("cuda")
        t0 = time.time()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=512)
        t1 = time.time()
        gen_text = processor.tokenizer.decode(output[0][inputs.input_ids.shape[1]:])
        results.append(parse_trajectory(gen_text))
        times.append(t1 - t0)
    return results, times

def process_latent(ckpt_dir, samples, end_think_token_id, desc_name, position):
    processor = AutoProcessor.from_pretrained("shreethar/stage1_unsloth")
    student = LatentStudent(model_name="shreethar/stage1_unsloth", M=6, K=5, end_think_token_id=end_think_token_id)
    lora_weights = load_peft_weights(os.path.join(ckpt_dir, "student_lora"))
    set_peft_model_state_dict(student.vlm, lora_weights)
    state = torch.load(os.path.join(ckpt_dir, "training_state.pt"), map_location="cpu")
    student.spatial_tokens.data.copy_(state["spatial_tokens"])
    student.spatial_mlp.load_state_dict(state["spatial_mlp"])
    student = student.to(torch.bfloat16).to("cuda")
    student.eval()
    results, times = [], []
    for s in tqdm(samples, desc=desc_name, position=position):
        message = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": s['human']}]}]
        text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        inputs = processor(text=[text], images=[s['frames'][0]], return_tensors="pt").to("cuda")
        t0 = time.time()
        with torch.no_grad():
            latents, h_s, spatial_hidden, waypoints = student.generate_latents(
                input_ids=inputs.input_ids,
                pixel_values=inputs.pixel_values,
                image_grid_thw=inputs.image_grid_thw,
                attention_mask=inputs.attention_mask
            )
        t1 = time.time()
        results.append((waypoints[0].cpu() * 1000).tolist())
        times.append(t1 - t0)
    return results, times

def wrapper_stage1(args):
    return "stage1", process_stage1(*args)
def wrapper_teacher(args):
    return "teacher", process_teacher(*args)
def wrapper_latent400(args):
    return "latent400", process_latent("checkpoints/stage2_decoupled_mini_student_2/step_000400", args[0], args[1], "Latent 400", 2)
def wrapper_latent619(args):
    return "latent619", process_latent("checkpoints/stage2_decoupled_mini_student_2/step_000619", args[0], args[1], "Latent 619", 3)

def main():
    mp.set_start_method('spawn', force=True)
    
    print("Loading dataset...")
    ds = load_dataset('shreethar/FYP-Stage2-dataset', split='train')
    np.random.seed(42)
    indices = np.random.choice(8000, 50, replace=False)
    samples = [ds[int(i)] for i in indices]
    
    processor = AutoProcessor.from_pretrained("shreethar/stage1_unsloth")
    end_think_token_id = processor.tokenizer.convert_tokens_to_ids("</think>")
    
    results = {"instruction": [], "ground_truth": []}
    for s in samples:
        match = re.search(r'Task:\s*(.*?)\s*What is', s['human'])
        inst = match.group(1).strip() if match else s['human']
        results["instruction"].append(inst)
        results["ground_truth"].append(parse_trajectory(s['assistant']))
        
    times = {}
    
    print("Starting parallel evaluation. Watch memory closely!")
    
    args = (samples, end_think_token_id)
    with mp.Pool(4) as pool:
        future_s1 = pool.apply_async(wrapper_stage1, (args,))
        future_t = pool.apply_async(wrapper_teacher, (args,))
        future_l400 = pool.apply_async(wrapper_latent400, (args,))
        future_l619 = pool.apply_async(wrapper_latent619, (args,))
        
        outputs = [
            future_s1.get(),
            future_t.get(),
            future_l400.get(),
            future_l619.get()
        ]
        
    for k, (res, tms) in outputs:
        results[k] = res
        times[k] = tms
        
    print("\nAverage Timing (seconds per sample):")
    for k in times:
        avg = sum(times[k]) / len(times[k])
        print(f"{k}: {avg:.4f}s")
        
    print("\nGenerating plots...")
    col_names = ["Instruction", "Ground Truth", "Stage 1", "Teacher", "Latent 400", "Latent 619"]
    col_keys = ["instruction", "ground_truth", "stage1", "teacher", "latent400", "latent619"]
    
    for i in range(5):
        fig, axes = plt.subplots(10, 6, figsize=(26, 40))
        plt.subplots_adjust(wspace=0.1, hspace=0.3)
        for row in range(10):
            idx = i * 10 + row
            img = samples[idx]['frames'][0]
            gt_traj = results["ground_truth"][idx]
            
            for col in range(6):
                ax = axes[row, col]
                if col_keys[col] == "instruction":
                    drawn_img = draw_instruction_image(results["instruction"][idx], size=img.size)
                    ax.imshow(drawn_img)
                else:
                    traj = results[col_keys[col]][idx]
                    drawn_img = draw_trajectory(img, traj)
                    ax.imshow(drawn_img)
                    
                    if col_keys[col] not in ["instruction", "ground_truth"]:
                        l2 = calc_l2_loss(gt_traj, traj)
                        dtw = calc_dtw(gt_traj, traj)
                        score_text = f"L2: {l2:.1f} | DTW: {dtw:.1f}"
                        if np.isnan(l2):
                            score_text = "FAILED TO GENERATE"
                        ax.text(0.5, -0.05, score_text, transform=ax.transAxes, 
                                ha="center", va="top", fontsize=14, color="black", 
                                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
                
                ax.axis('off')
                if row == 0:
                    ax.set_title(col_names[col], fontsize=18, pad=15)
        
        plt.tight_layout()
        save_path = f"evaluation_grid_{i+1}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Saved {save_path}")

if __name__ == '__main__':
    main()
