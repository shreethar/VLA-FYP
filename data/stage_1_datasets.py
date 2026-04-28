import re
import json
import ast
import numpy as np
import hashlib
import cv2
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import IterableDataset as TorchIterableDataset
import random
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
import tensorflow as tf

DATA_DIR = Path(__file__).resolve().parent

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────

IMAGE_SIZE = 448          # Qwen3.5-VL recommended input resolution
K_WAYPOINTS = 5           # number of waypoints to predict
MAX_FRAMES = 8            # maximum number of frames to uniformly extract from videos

DATASET_WEIGHTS = {
    "molmoact":   0.3,
    "sharerobot": 0.2,
    "robovqa":    0.1,
    "pixmo":      0.15,
    "egoplan":    0.1,
    "robofac":    0.15,
}

TRAJ_SYSTEM = (
    "You are a robot manipulation assistant. Given an observation image and a "
    "task instruction, predict the end-effector's 2D trajectory as {k} "
    "waypoints. Output ONLY the coordinate list in this exact format: "
    "[[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5]]"
).format(k=K_WAYPOINTS)

QA_SYSTEM = (
    "You are a robot manipulation assistant. Answer questions about robot tasks, "
    "object affordances, spatial relationships, and manipulation strategies based "
    "on the provided image or video frame."
)

# ─────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────

def resample_waypoints(coords: list[tuple], k: int = 5) -> list[tuple]:
    """Linearly interpolate variable-length trace to exactly k waypoints."""
    if len(coords) == k:
        return coords
    coords = np.array(coords)
    old_t = np.linspace(0, 1, len(coords))
    new_t = np.linspace(0, 1, k)
    x = np.interp(new_t, old_t, coords[:, 0])
    y = np.interp(new_t, old_t, coords[:, 1])
    return list(zip(x.tolist(), y.tolist()))

def format_waypoints(coords: list[tuple]) -> str:
    """Format normalized (x,y) pairs as the text the model will output."""
    inner = "],[".join(f"{x:.3f},{y:.3f}" for x, y in coords)
    return f"[[{inner}]]"

def get_deterministic_split(unique_id: str, subset_pct: float) -> str | None:
    """
    Returns 'train', 'val', 'test', or None (if excluded by subset_pct).
    subset_pct is between 0.0 and 1.0 (e.g., 0.15 for 15%).
    """
    hash_int = int(hashlib.md5(unique_id.encode('utf-8')).hexdigest(), 16)
    
    # 1. Check if it falls within the subset percentage
    if (hash_int % 10000) >= (subset_pct * 10000):
        return None
        
    # 2. Determine split from a different property of the hash
    split_hash = (hash_int // 10000) % 100
    
    if split_hash < 80: return 'train'   # 80%
    elif split_hash < 90: return 'val'   # 10%
    else: return 'test'                  # 10%

def load_image_safe(image_input) -> Image.Image | None:
    """Handle PIL Image, raw bytes, HF dict, URL string, or local path."""
    try:
        img = None
        if isinstance(image_input, Image.Image):
            img = image_input
        elif isinstance(image_input, bytes):
            img = Image.open(BytesIO(image_input))
        elif isinstance(image_input, dict) and "bytes" in image_input:
            img = Image.open(BytesIO(image_input["bytes"]))
        elif isinstance(image_input, str):
            if image_input.startswith("http"):
                resp = requests.get(image_input, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                img = Image.open(BytesIO(resp.content))
            else:
                img = Image.open(image_input)
                
        if img is not None:
            return img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    except Exception:
        return None
    return None

def load_video_frames(video_input, max_frames=MAX_FRAMES) -> list[Image.Image] | None:
    """Extract uniform frames from a video or list of frames."""
    if isinstance(video_input, list):
        if len(video_input) == 0: return None
        indices = np.linspace(0, len(video_input) - 1, min(max_frames, len(video_input)), dtype=int)
        frames = []
        for idx in indices:
            img = load_image_safe(video_input[idx])
            if img: frames.append(img)
        return frames if frames else None
        
    elif isinstance(video_input, str):
        cap = cv2.VideoCapture(video_input)
        if not cap.isOpened(): return None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: return None
            
        indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
        frames = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                frames.append(img)
                
        cap.release()
        return frames if frames else None
    return None

def load_media_safe(media_input) -> Image.Image | list[Image.Image] | None:
    """Intelligently load image or video and return PIL Image(s)."""
    if isinstance(media_input, list):
        return load_video_frames(media_input)
    elif isinstance(media_input, str) and media_input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        return load_video_frames(media_input)
    else:
        return load_image_safe(media_input)

def format_media_content(media_obj, text_prompt: str) -> list[dict]:
    """Format Qwen3.5-VL content array depending on single image vs sequence of frames."""
    content = []
    if isinstance(media_obj, list):
        for _ in media_obj:
            content.append({"type": "image"})
    else:
        content.append({"type": "image"})
    
    content.append({"type": "text", "text": text_prompt})
    return content

# ─────────────────────────────────────────
# LOADER 1 — MolmoAct
# ─────────────────────────────────────────

def stream_molmoact_for_qwen(split="train", subset_pct=0.15):
    ds = load_dataset("allenai/MolmoAct-Pretraining-Mixture", name="auxiliary_trace", split="train", streaming=True)
    for raw in ds:
        convs = raw.get("conversations", {})
        human, gpt = None, None
        
        if isinstance(convs, dict) and "from" in convs and "value" in convs:
            for r, v in zip(convs["from"], convs["value"]):
                if r == "human": human = v
                if r == "gpt": gpt = v
        elif isinstance(convs, list):
            human = next((c.get("value") for c in convs if c.get("from") == "human"), None)
            gpt   = next((c.get("value") for c in convs if c.get("from") == "gpt"),   None)

        if not human or not gpt: continue

        unique_id = f"molmoact_{human}_{gpt}"
        if get_deterministic_split(unique_id, subset_pct) != split: continue

        img = load_image_safe(raw.get("image"))
        if img is None: continue

        match = re.search(r'\[\[[\d\s,\[\]]+\]\]', gpt)
        if not match: continue
            
        try:
            raw_coords = ast.literal_eval(match.group(0))
        except (ValueError, SyntaxError): continue

        if len(raw_coords) < 2: continue

        coords_normalized = [(float(x) / 255.0, float(y) / 255.0) for x, y in raw_coords]
        coords_resampled = resample_waypoints(coords_normalized, k=K_WAYPOINTS)
        
        qwen_coords_list = []
        for nx, ny in coords_resampled:
            qx = max(0, min(1000, int(nx * 1000)))
            qy = max(0, min(1000, int(ny * 1000)))
            qwen_coords_list.append(f"[{qx},{qy}]")
            
        result = '[' + ','.join(qwen_coords_list) + ']'

        yield {
            "dataset": "molmoact",
            "type": "trajectory",
            "image": img,
            "messages": [
                {
                    "role": "user",
                    "content": format_media_content(img, f"{TRAJ_SYSTEM}\n\nTask: {human.strip()}")
                },
                {
                    "role": "assistant",
                    "content": result
                }
            ]
        }

# ─────────────────────────────────────────
# LOADER 2 — ShareRobot
# ─────────────────────────────────────────

def stream_sharerobot_affordance_for_qwen(split="train", subset_pct=1.0):
    json_path = DATA_DIR / "ShareRobot/affordance/affordance.json"
    if not json_path.exists(): return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for raw in data:
        unique_id = f"sharerobot_aff_{raw.get('id', '')}"
        if get_deterministic_split(unique_id, subset_pct) != split: continue
        
        instruction = raw.get("instruction", "").strip()
        img_rel_path = raw.get("image_path")
        affordance = raw.get("affordance")
        meta = raw.get("meta_data", {})
        
        if not instruction or not img_rel_path or not affordance or not meta: continue
        
        img_path = DATA_DIR / "ShareRobot/affordance/images" / img_rel_path
        if not img_path.exists():
            img_path = DATA_DIR / "ShareRobot/affordance" / img_rel_path
            if not img_path.exists(): continue
            
        img = load_image_safe(str(img_path))
        if img is None: continue
        
        orig_w = meta.get("original_width", 1)
        orig_h = meta.get("original_height", 1)
        
        x = affordance.get("x", 0)
        y = affordance.get("y", 0)
        w = affordance.get("width", 0)
        h = affordance.get("height", 0)
        
        xmin = max(0, min(1000, int((x / orig_w) * 1000)))
        ymin = max(0, min(1000, int((y / orig_h) * 1000)))
        xmax = max(0, min(1000, int(((x + w) / orig_w) * 1000)))
        ymax = max(0, min(1000, int(((y + h) / orig_h) * 1000)))
        
        bbox_str = f"[{xmin}, {ymin}, {xmax}, {ymax}]"
        
        yield {
            "dataset": "sharerobot_affordance",
            "type": "qa",
            "image": img,
            "messages": [
                {
                    "role": "user",
                    "content": format_media_content(img, f"{QA_SYSTEM}\n\nTask: {instruction}\nOutput the affordance bounding box.")
                },
                {
                    "role": "assistant",
                    "content": bbox_str
                }
            ]
        }

def stream_sharerobot_planning_for_qwen(split="train", subset_pct=0.10):
    jsons_dir = DATA_DIR / "ShareRobot/planning/jsons"
    if not jsons_dir.exists(): return
    
    for json_file in jsons_dir.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for raw in data:
            unique_id = raw.get("id", "")
            if not unique_id: continue
            
            if get_deterministic_split(unique_id, subset_pct) != split: continue
            
            convs = raw.get("conversations", [])
            human = next((c["value"] for c in convs if c["from"] == "human"), "")
            human = re.sub(r'<image>\s*', '', human).strip()
            
            gpt = next((c["value"] for c in convs if c["from"] == "gpt"), "").strip()
            
            if not human or not gpt: continue
            
            images_list = raw.get("image", [])
            if not images_list: continue
            
            resolved_images = []
            for img_rel in images_list:
                img_path = DATA_DIR / "ShareRobot/planning/images" / img_rel
                if not img_path.exists():
                    img_path = DATA_DIR / "ShareRobot/planning" / img_rel
                resolved_images.append(str(img_path))
                
            media_obj = load_video_frames(resolved_images, max_frames=16)
            if media_obj is None: continue
            
            yield {
                "dataset": "sharerobot_planning",
                "type": "qa",
                "image": media_obj,
                "messages": [
                    {
                        "role": "user",
                        "content": format_media_content(media_obj, f"{QA_SYSTEM}\n\n{human}")
                    },
                    {
                        "role": "assistant",
                        "content": gpt
                    }
                ]
            }

# ─────────────────────────────────────────
# LOADER 3 — RoboVQA
# ─────────────────────────────────────────

def stream_robovqa_for_qwen(split="train", subset_pct=1.0):
    filepaths = tf.io.gfile.glob('gs://gdm-robovqa/tfrecord/train/train*')
    if not filepaths: return
    
    dataset = tf.data.TFRecordDataset(filepaths)
    
    for raw_record in dataset.as_numpy_iterator():
        example = tf.train.SequenceExample()
        example.ParseFromString(raw_record)
        
        image_feature = example.feature_lists.feature_list.get('images')
        if not image_feature: continue
        
        frames = []
        for bl in image_feature.feature:
            code = bl.bytes_list.value[0]
            try:
                img = Image.open(BytesIO(code)).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                frames.append(img)
            except Exception:
                pass
                
        if not frames: continue
        
        texts_feature = example.feature_lists.feature_list.get('texts')
        if not texts_feature or not texts_feature.feature: continue
        
        try:
            raw_text = texts_feature.feature[0].bytes_list.value[0].decode('utf-8')
        except Exception:
            continue
            
        qa_pairs = []
        blocks = re.findall(r'<task:[^>]+>\s*(.*?)\s*</PRED>', raw_text, re.DOTALL)
        for block in blocks:
            parts = block.split("<PRED>A:", 1)
            if len(parts) == 2:
                q_text = parts[0].strip()
                a_text = parts[1].strip()
                a_text = re.sub(r'</?PRED[^>]*>', '', a_text).strip()
                if q_text and a_text:
                    qa_pairs.append((q_text, a_text))
                    
        for q, a in qa_pairs:
            unique_id = f"robovqa_{q}_{a}"
            if get_deterministic_split(unique_id, subset_pct) != split: continue
            
            yield {
                "dataset": "robovqa",
                "type": "qa",
                "image": frames,
                "messages": [
                    {
                        "role": "user",
                        "content": format_media_content(frames, f"{QA_SYSTEM}\n\n{q}")
                    },
                    {
                        "role": "assistant",
                        "content": a
                    }
                ]
            }

# ─────────────────────────────────────────
# LOADER 4 — PixMo (Cap, AMA, CapQA)
# ─────────────────────────────────────────

def download_image_safe(url, timeout=5):
    if not url: return None
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img.load()
        return img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    except Exception:
        return None

def stream_pixmocap_for_qwen(split="train", subset_pct=0.15):
    ds = load_dataset("allenai/pixmo-cap", split="train", streaming=True)
    instruction = "Describe this image in detail."
    for raw in ds:
        caption = raw.get("caption")
        if not caption or not isinstance(caption, str): continue
        
        unique_id = f"pixmocap_{caption}"
        if get_deterministic_split(unique_id, subset_pct) != split: continue

        img = download_image_safe(raw.get("image_url"))
        if img is None: continue

        yield {
            "dataset": "pixmocap",
            "type": "captioning",
            "image": img,
            "messages": [
                {
                    "role": "user",
                    "content": format_media_content(img, instruction)
                },
                {
                    "role": "assistant",
                    "content": caption.strip()
                }
            ]
        }

def stream_pixmo_ama_for_qwen(split="train", subset_pct=0.15):
    ds = load_dataset("allenai/pixmo-ask-model-anything", split="train", streaming=True)
    for raw in ds:
        q = raw.get("question")
        a = raw.get("answer")
        if not q or not a or not isinstance(q, str) or not isinstance(a, str): continue
        
        unique_id = f"pixmoama_{q}_{a}"
        if get_deterministic_split(unique_id, subset_pct) != split: continue

        img = download_image_safe(raw.get("image_url"))
        if img is None: continue

        yield {
            "dataset": "pixmoama",
            "type": "vqa",
            "image": img,
            "messages": [
                {
                    "role": "user",
                    "content": format_media_content(img, q.strip())
                },
                {
                    "role": "assistant",
                    "content": a.strip()
                }
            ]
        }

def stream_pixmo_cap_qa_for_qwen(split="train", subset_pct=0.15):
    ds = load_dataset("allenai/pixmo-cap-qa", split="train", streaming=True)
    for raw in ds:
        messages = raw.get("messages")
        if isinstance(messages, list) and len(messages) >= 2:
            q = messages[0]
            a = messages[1]
        else:
            q = raw.get("question", "").replace("[USER]", "").replace("[ASSISTANT]", "").strip()
            a = raw.get("answer", "").strip()

        if not q or not a or not isinstance(q, str) or not isinstance(a, str): continue
        
        unique_id = f"pixmocapqa_{q}_{a}"
        if get_deterministic_split(unique_id, subset_pct) != split: continue

        img = download_image_safe(raw.get("image_url"))
        if img is None: continue

        yield {
            "dataset": "pixmocapqa",
            "type": "qa",
            "image": img,
            "messages": [
                {
                    "role": "user",
                    "content": format_media_content(img, q.strip())
                },
                {
                    "role": "assistant",
                    "content": a.strip()
                }
            ]
        }

# ─────────────────────────────────────────
# LOADER 5 — EgoPlan
# ─────────────────────────────────────────

def stream_egoplan_for_qwen(split="train", subset_pct=0.15):
    ds = load_dataset("lmms-lab/EgoPlan", split="train", streaming=True)
    for raw in ds:
        q = raw.get("question", "").strip()
        choices = raw.get("options", [])
        answer_idx = raw.get("answer", 0)
        
        if isinstance(answer_idx, str): a = answer_idx
        elif choices: a = choices[answer_idx] if answer_idx < len(choices) else str(answer_idx)
        else: a = str(raw.get("answer", ""))

        if not q or not a: continue
        
        unique_id = f"egoplan_{q}_{a}"
        if get_deterministic_split(unique_id, subset_pct) != split: continue

        img = load_image_safe(raw.get("image"))
        if img is None: continue

        yield {
            "dataset": "egoplan",
            "type": "qa",
            "image": img,
            "messages": [
                {
                    "role": "user",
                    "content": format_media_content(img, f"{QA_SYSTEM}\n\n{q}")
                },
                {
                    "role": "assistant",
                    "content": a
                }
            ]
        }

# ─────────────────────────────────────────
# LOADER 6 — RoboFac
# ─────────────────────────────────────────

def stream_robofac_for_qwen(split="train", subset_pct=0.15):
    json_path = DATA_DIR / "RoboFAC/training_qa.json"
    if not Path(json_path).exists():
        return
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for raw in data:
        convs = raw.get("conversations", [])
        instruction = next((c["value"] for c in convs if c["from"] == "human"), "").replace("<video>\n", "").strip()
        response = next((c["value"] for c in convs if c["from"] == "assistant"), "").strip()

        if not instruction or not response: continue
        
        unique_id = raw.get("id", f"robofac_{instruction}_{response}")
        if get_deterministic_split(unique_id, subset_pct) != split: continue

        video_rel_path = raw.get("video")
        if not video_rel_path: continue
        
        base_dir = DATA_DIR / "RoboFAC"
        realworld_path = base_dir / "realworld_data" / video_rel_path
        sim_path = base_dir / "simulation_data" / video_rel_path
        
        if realworld_path.exists():
            media_source = str(realworld_path)
        elif sim_path.exists():
            media_source = str(sim_path)
        else:
            continue
            
        media_obj = load_video_frames(media_source, max_frames=16)
        if media_obj is None: continue

        yield {
            "dataset": "robofac",
            "type": "qa",
            "image": media_obj,
            "messages": [
                {
                    "role": "user",
                    "content": format_media_content(media_obj, f"{QA_SYSTEM}\n\n{instruction}")
                },
                {
                    "role": "assistant",
                    "content": response
                }
            ]
        }

# ─────────────────────────────────────────
# MASTER LOADER
# ─────────────────────────────────────────

class NativePyTorchInterleavedDataset(TorchIterableDataset):
    def __init__(self, generators_info):
        self.generators_info = generators_info
        weights = [info[2] for info in generators_info]
        total_weight = sum(weights)
        self.probabilities = [w / total_weight for w in weights]

    def __iter__(self):
        active_gens = []
        for name, gen_fn, weight, spct, split in self.generators_info:
            active_gens.append(iter(gen_fn(split=split, subset_pct=spct)))
            
        while True:
            idx = random.choices(range(len(active_gens)), weights=self.probabilities)[0]
            try:
                yield next(active_gens[idx])
            except StopIteration:
                # Re-initialize the exhausted generator for infinite streaming
                name, gen_fn, weight, spct, split = self.generators_info[idx]
                active_gens[idx] = iter(gen_fn(split=split, subset_pct=spct))
                try:
                    yield next(active_gens[idx])
                except StopIteration:
                    pass # If it's truly completely empty forever, just pass

def build_iterable_datasets() -> dict[str, TorchIterableDataset]:
    """
    Build interleaved PyTorch IterableDatasets for train, val, and test splits.
    Bypasses PyArrow inference bugs by strictly using native Python dict yields.
    """
    splits = ["train", "val", "test"]
    result = {}

    generators = [
        # (name, function, probability_weight, subset_pct)
        ("molmoact",        stream_molmoact_for_qwen,              DATASET_WEIGHTS["molmoact"], 0.133), # ~200K/1.5M
        ("sharerobot_aff",  stream_sharerobot_affordance_for_qwen, DATASET_WEIGHTS["sharerobot"] * 0.2, 1.0),
        ("sharerobot_plan", stream_sharerobot_planning_for_qwen,   DATASET_WEIGHTS["sharerobot"] * 0.8, 0.10),
        ("robovqa",         stream_robovqa_for_qwen,               DATASET_WEIGHTS["robovqa"], 1.0),
        ("pixmocap",        stream_pixmocap_for_qwen,              DATASET_WEIGHTS["pixmo"]/3, 0.10),
        ("pixmoama",        stream_pixmo_ama_for_qwen,             DATASET_WEIGHTS["pixmo"]/3, 0.10),
        ("pixmocapqa",      stream_pixmo_cap_qa_for_qwen,          DATASET_WEIGHTS["pixmo"]/3, 0.10),
        ("robofac",         stream_robofac_for_qwen,               DATASET_WEIGHTS["robofac"], 1.0)
    ]

    for split in splits:
        gen_info = []
        for name, gen_fn, weight, spct in generators:
            gen_info.append((name, gen_fn, weight, spct, split))
            
        result[split] = NativePyTorchInterleavedDataset(gen_info)
        
    return result