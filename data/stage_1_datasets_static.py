"""
stage_1_datasets_static.py

Builds a fully offline, indexable map-style Dataset for SFTTrainer.
Images are loaded lazily in __getitem__ so RAM usage stays low.

Target sample counts:
  MolmoAct               200 000
  ShareRobot Affordance  all ~6.5K
  ShareRobot Planning    100 000
  RoboVQA                100 000
  Pixmo Cap               50 000
  Pixmo AMA               50 000
  Pixmo Cap-QA            50 000
  RoboFAC                all ~64K
"""

import ast, hashlib, json, re, struct
import numpy as np
import cv2
import tensorflow as tf
from io import BytesIO
from pathlib import Path
from PIL import Image
import requests
import torch
from torch.utils.data import Dataset

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS (keep in sync with stage_1_datasets.py)
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).resolve().parent
IMAGE_SIZE = 448
K_WAYPOINTS = 5
MAX_FRAMES  = 16

TRAJ_SYSTEM = (
    "You are a robot manipulation assistant. Given an observation image and a "
    f"task instruction, predict the end-effector's 2D trajectory as {K_WAYPOINTS} "
    "waypoints. Output ONLY the coordinate list in this exact format: "
    "[[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5]]"
)
QA_SYSTEM = (
    "You are a robot manipulation assistant. Answer questions about robot tasks, "
    "object affordances, spatial relationships, and manipulation strategies based "
    "on the provided image or video frame."
)

# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _keep(idx: int, total: int, n_keep: int) -> bool:
    """Deterministic uniform sampling: keep ~n_keep out of total."""
    h = int(hashlib.md5(str(idx).encode()).hexdigest(), 16)
    return (h % total) < n_keep

def _get_split(record: dict) -> str:
    """
    Assigns a deterministic train/test split based on record content.
    Always produces the same split for the same record — 85/15.
    """
    key = f"{record['dataset']}|{record['human'][:80]}|{record['assistant'][:40]}"
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    return "train" if h < 85 else "test"

def resample_waypoints(coords, k=5):
    if len(coords) == k:
        return coords
    coords = np.array(coords)
    old_t = np.linspace(0, 1, len(coords))
    new_t = np.linspace(0, 1, k)
    x = np.interp(new_t, old_t, coords[:, 0])
    y = np.interp(new_t, old_t, coords[:, 1])
    return list(zip(x.tolist(), y.tolist()))

def load_image_safe(src) -> Image.Image | None:
    try:
        if isinstance(src, Image.Image):
            img = src
        elif isinstance(src, bytes):
            img = Image.open(BytesIO(src))
        elif isinstance(src, dict) and "bytes" in src:
            img = Image.open(BytesIO(src["bytes"]))
        elif isinstance(src, str) and src.startswith("http"):
            r = requests.get(src, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            img = Image.open(BytesIO(r.content))
        elif isinstance(src, str):
            img = Image.open(src)
        else:
            return None
        return img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    except Exception:
        return None

def load_video_frames(src, max_frames=16) -> list[Image.Image] | None:
    if isinstance(src, list):
        indices = np.linspace(0, len(src)-1, min(max_frames, len(src)), dtype=int)
        frames = [load_image_safe(src[i]) for i in indices]
        frames = [f for f in frames if f]
        return frames or None
    elif isinstance(src, str):
        cap = cv2.VideoCapture(src)
        if not cap.isOpened(): return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0: return None
        indices = np.linspace(0, total-1, min(max_frames, total), dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                              .resize((IMAGE_SIZE, IMAGE_SIZE)))
        cap.release()
        return frames or None
    return None

def format_messages(media_obj, human_text: str, assistant_text: str) -> list[dict]:
    if isinstance(media_obj, list) and len(media_obj) > 1:
        media_block = {"type": "video", "video": media_obj}
    else:
        img = media_obj[0] if isinstance(media_obj, list) else media_obj
        media_block = {"type": "image", "image": img}
    return [
        {"role": "user",      "content": [media_block, {"type": "text", "text": human_text}]},
        {"role": "assistant", "content": assistant_text},
    ]

# ─────────────────────────────────────────────────────────────────────────────
# RECORD BUILDERS  (each returns a list of lightweight metadata dicts)
# ─────────────────────────────────────────────────────────────────────────────

def build_molmoact_records(hf_ds, n_samples: int = 200_000) -> list[dict]:
    """HF map-style dataset already in memory; store index + parsed text."""
    total   = len(hf_ds)
    n_keep  = min(n_samples, total)
    records = []
    for i in range(total):
        if not _keep(i, total, n_keep):
            continue
        raw  = hf_ds[i]
        convs = raw.get("conversations", {})
        human, gpt = None, None
        if isinstance(convs, dict) and "from" in convs:
            for r, v in zip(convs["from"], convs["value"]):
                if r == "human": human = v
                if r == "gpt":   gpt   = v
        elif isinstance(convs, list):
            human = next((c.get("value") for c in convs if c.get("from") == "human"), None)
            gpt   = next((c.get("value") for c in convs if c.get("from") == "gpt"),   None)
        if not human or not gpt: continue
        match = re.search(r'\[\[[\d\s,\[\]]+\]\]', gpt)
        if not match: continue
        try:
            raw_coords = ast.literal_eval(match.group(0))
        except Exception: continue
        if len(raw_coords) < 2: continue
        norm  = [(float(x)/255.0, float(y)/255.0) for x,y in raw_coords]
        resampled = resample_waypoints(norm, k=K_WAYPOINTS)
        parts = [f"[{max(0,min(1000,int(nx*1000)))},{max(0,min(1000,int(ny*1000)))}]"
                 for nx,ny in resampled]
        assistant = '[' + ','.join(parts) + ']'
        records.append({
            "dataset":    "molmoact",
            "type":       "trajectory",
            "source":     "hf_index",
            "hf_key":     "molmoact",
            "hf_index":   i,
            "human":      f"{TRAJ_SYSTEM}\n\nTask: {human.strip()}",
            "assistant":  assistant,
            "max_frames": 1,
        })
        if len(records) >= n_keep:
            break
    print(f"  MolmoAct: {len(records):,} records")
    return records


def build_sharerobot_affordance_records() -> list[dict]:
    json_path = DATA_DIR / "ShareRobot/affordance/affordance.json"
    if not json_path.exists():
        print("  ShareRobot Affordance: JSON not found, skipping")
        return []
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for raw in data:
        instruction = raw.get("instruction", "").strip()
        img_rel     = raw.get("image_path")
        affordance  = raw.get("affordance")
        meta        = raw.get("meta_data", {})
        if not instruction or not img_rel or not affordance or not meta: continue
        img_path = DATA_DIR / "ShareRobot/affordance/images" / img_rel
        if not img_path.exists():
            img_path = DATA_DIR / "ShareRobot/affordance" / img_rel
            if not img_path.exists(): continue
        orig_w = meta.get("original_width", 1)
        orig_h = meta.get("original_height", 1)
        x,y,w,h = affordance.get("x",0), affordance.get("y",0), \
                  affordance.get("width",0), affordance.get("height",0)
        xmin = max(0, min(1000, int((x/orig_w)*1000)))
        ymin = max(0, min(1000, int((y/orig_h)*1000)))
        xmax = max(0, min(1000, int(((x+w)/orig_w)*1000)))
        ymax = max(0, min(1000, int(((y+h)/orig_h)*1000)))
        records.append({
            "dataset":   "sharerobot_affordance",
            "type":      "qa",
            "source":    "local_image",
            "local_path": str(img_path),
            "human":     f"{QA_SYSTEM}\n\nTask: {instruction}\nOutput the affordance bounding box.",
            "assistant": f"[{xmin}, {ymin}, {xmax}, {ymax}]",
            "max_frames": 1,
        })
    print(f"  ShareRobot Affordance: {len(records):,} records")
    return records


def build_sharerobot_planning_records(n_samples: int = 100_000) -> list[dict]:
    jsons_dir = DATA_DIR / "ShareRobot/planning/jsons"
    if not jsons_dir.exists():
        print("  ShareRobot Planning: jsons dir not found, skipping")
        return []
    all_rows = []
    for jf in sorted(jsons_dir.glob("*.json")):
        with open(jf, encoding="utf-8") as f:
            all_rows.extend(json.load(f))
    total  = len(all_rows)
    n_keep = min(n_samples, total)
    records = []
    for i, raw in enumerate(all_rows):
        if not _keep(i, total, n_keep): continue
        uid   = raw.get("id", "")
        convs = raw.get("conversations", [])
        human = re.sub(r'<image>\s*', '',
                       next((c["value"] for c in convs if c["from"] == "human"), "")).strip()
        gpt   = next((c["value"] for c in convs if c["from"] == "gpt"), "").strip()
        if not human or not gpt: continue
        img_list = raw.get("image", [])
        if not img_list: continue
        frame_paths = []
        for img_rel in img_list:
            p = DATA_DIR / "ShareRobot/planning/images" / img_rel
            if not p.exists():
                p = DATA_DIR / "ShareRobot/planning" / img_rel
            frame_paths.append(str(p))
        records.append({
            "dataset":     "sharerobot_planning",
            "type":        "qa",
            "source":      "local_frames",
            "frame_paths": frame_paths,
            "human":       f"{QA_SYSTEM}\n\n{human}",
            "assistant":   gpt,
            "max_frames":  MAX_FRAMES,
        })
        if len(records) >= n_keep: break
    print(f"  ShareRobot Planning: {len(records):,} records")
    return records


def build_robovqa_records(n_samples: int = 100_000) -> list[dict]:
    """
    Build an index of (tfrecord_path, byte_offset, q_text, a_text) tuples
    so __getitem__ can seek directly to each record. One-time iteration.
    """
    tfrecord_dir = DATA_DIR / "RoboVQA-train"
    if not tfrecord_dir.exists():
        print("  RoboVQA: data/RoboVQA-train not found, skipping")
        return []

    tfrecord_files = sorted(tfrecord_dir.glob("*.tfrecord")) or \
                     sorted(tfrecord_dir.glob("train*"))
    if not tfrecord_files:
        # try any file in directory
        tfrecord_files = [p for p in tfrecord_dir.iterdir() if p.is_file()]

    all_candidates = []  # (filepath, byte_offset, q_text, a_text)

    for tf_path in tfrecord_files:
        with open(tf_path, "rb") as f:
            while True:
                header = f.read(12)   # 8 bytes length + 4 bytes crc
                if len(header) < 12: break
                data_len = struct.unpack("<Q", header[:8])[0]
                byte_offset = f.tell() - 12  # offset of the record start
                data = f.read(data_len)
                f.read(4)  # skip data crc
                try:
                    example = tf.train.SequenceExample()
                    example.ParseFromString(data)
                    texts_feat = example.feature_lists.feature_list.get("texts")
                    if not texts_feat or not texts_feat.feature: continue
                    raw_text = texts_feat.feature[0].bytes_list.value[0].decode("utf-8")
                    blocks = re.findall(r'<task:[^>]+>\s*(.*?)\s*</PRED>', raw_text, re.DOTALL)
                    for block in blocks:
                        parts = block.split("<PRED>A:", 1)
                        if len(parts) == 2:
                            q = parts[0].strip()
                            a = re.sub(r'</?PRED[^>]*>', '', parts[1]).strip()
                            if q and a:
                                all_candidates.append((str(tf_path), byte_offset, q, a))
                except Exception:
                    continue

    total  = len(all_candidates)
    n_keep = min(n_samples, total)
    records = []
    for i, (tf_path, offset, q, a) in enumerate(all_candidates):
        if not _keep(i, total, n_keep): continue
        records.append({
            "dataset":       "robovqa",
            "type":          "qa",
            "source":        "tfrecord",
            "tfrecord_path": tf_path,
            "byte_offset":   offset,
            "human":         f"{QA_SYSTEM}\n\n{q}",
            "assistant":     a,
            "max_frames":    MAX_FRAMES,
        })
    print(f"  RoboVQA: {len(records):,} records (from {len(all_candidates):,} QA pairs)")
    return records


def _build_pixmo_records(hf_ds, name: str, n_samples: int,
                         get_qa_fn) -> list[dict]:
    total  = len(hf_ds)
    n_keep = min(n_samples, total)
    records = []
    for i in range(total):
        if not _keep(i, total, n_keep): continue
        raw = hf_ds[i]
        result = get_qa_fn(raw)
        if result is None: continue
        human, assistant, img_url = result
        records.append({
            "dataset":   name,
            "type":      "qa",
            "source":    "url",
            "url":       img_url,
            "human":     human,
            "assistant": assistant,
            "max_frames": 1,
        })
        if len(records) >= n_keep: break
    print(f"  {name}: {len(records):,} records")
    return records


def build_pixmocap_records(hf_ds, n_samples: int = 50_000) -> list[dict]:
    def get_qa(raw):
        caption = raw.get("caption", "")
        url     = raw.get("image_url", "")
        if not caption or not url: return None
        return "Describe this image in detail.", caption.strip(), url
    return _build_pixmo_records(hf_ds, "pixmocap", n_samples, get_qa)


def build_pixmoama_records(hf_ds, n_samples: int = 50_000) -> list[dict]:
    def get_qa(raw):
        q = raw.get("question", ""); a = raw.get("answer", "")
        url = raw.get("image_url", "")
        if not q or not a or not url: return None
        return q.strip(), a.strip(), url
    return _build_pixmo_records(hf_ds, "pixmoama", n_samples, get_qa)


def build_pixmocapqa_records(hf_ds, n_samples: int = 50_000) -> list[dict]:
    def get_qa(raw):
        msgs = raw.get("messages")
        if isinstance(msgs, list) and len(msgs) >= 2:
            q, a = msgs[0], msgs[1]
        else:
            q = raw.get("question","").replace("[USER]","").replace("[ASSISTANT]","").strip()
            a = raw.get("answer","").strip()
        url = raw.get("image_url","")
        if not q or not a or not isinstance(q,str) or not url: return None
        return q.strip(), a.strip(), url
    return _build_pixmo_records(hf_ds, "pixmocapqa", n_samples, get_qa)


def build_robofac_records() -> list[dict]:
    json_path = DATA_DIR / "RoboFAC/training_qa.json"
    if not json_path.exists():
        print("  RoboFAC: training_qa.json not found, skipping")
        return []
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for raw in data:
        convs       = raw.get("conversations", [])
        instruction = next((c["value"] for c in convs if c["from"] == "human"), "") \
                          .replace("<video>\n","").strip()
        response    = next((c["value"] for c in convs if c["from"] == "assistant"), "").strip()
        if not instruction or not response: continue
        video_rel = raw.get("video")
        if not video_rel: continue
        base = DATA_DIR / "RoboFAC"
        rw_path  = base / "realworld_data" / video_rel
        sim_path = base / "simulation_data" / video_rel
        if rw_path.exists():
            video_path = str(rw_path)
        elif sim_path.exists():
            video_path = str(sim_path)
        else:
            continue
        records.append({
            "dataset":     "robofac",
            "type":        "qa",
            "source":      "local_video",
            "local_path":  video_path,
            "human":       f"{QA_SYSTEM}\n\n{instruction}",
            "assistant":   response,
            "max_frames":  MAX_FRAMES,
        })
    print(f"  RoboFAC: {len(records):,} records")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# MAP-STYLE DATASET
# ─────────────────────────────────────────────────────────────────────────────

class VLAStaticDataset(Dataset):
    """
    Lazy-loading map-style Dataset. Metadata is stored in memory;
    images are loaded from disk/network/TFRecord only when __getitem__ is called.
    """

    def __init__(self, records: list[dict], hf_datasets: dict | None = None):
        self.records     = records
        self._hf_cache   = hf_datasets or {}  # name → HF Dataset object

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec   = self.records[idx]
        media = self._load_media(rec)
        if media is None:
            # fallback: try the next item (shouldn't happen often)
            return self.__getitem__((idx + 1) % len(self.records))
        return {
            "dataset":  rec["dataset"],
            "type":     rec["type"],
            "image":    media,
            "messages": format_messages(media, rec["human"], rec["assistant"]),
        }

    def _load_media(self, rec: dict):
        src = rec["source"]
        mf  = rec.get("max_frames", 16)

        if src == "hf_index":
            hf_ds = self._hf_cache[rec["hf_key"]]
            raw   = hf_ds[rec["hf_index"]]
            # MolmoAct returns image as bytes or a dict with 'bytes' key
            image_field = raw.get("image")
            img = load_image_safe(image_field)
            return [img] if img else None

        elif src == "local_image":
            img = load_image_safe(rec["local_path"])
            return [img] if img else None

        elif src == "local_frames":
            return load_video_frames(rec["frame_paths"], max_frames=mf)

        elif src == "local_video":
            return load_video_frames(rec["local_path"], max_frames=mf)

        elif src == "url":
            try:
                r   = requests.get(rec["url"],
                                   headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
                img = Image.open(BytesIO(r.content)).convert("RGB") \
                           .resize((IMAGE_SIZE, IMAGE_SIZE))
                return [img]
            except Exception:
                return None

        elif src == "tfrecord":
            return self._load_tfrecord(rec, max_frames=mf)

        return None

    def _load_tfrecord(self, rec: dict, max_frames: int = 16):
        try:
            with open(rec["tfrecord_path"], "rb") as f:
                f.seek(rec["byte_offset"])
                length_bytes = f.read(8)
                data_len     = struct.unpack("<Q", length_bytes)[0]
                f.read(4)  # skip length crc
                data = f.read(data_len)
            example = tf.train.SequenceExample()
            example.ParseFromString(data)
            image_feat = example.feature_lists.feature_list.get("images")
            if not image_feat: return None
            raw_frames = [bl.bytes_list.value[0] for bl in image_feat.feature]
            indices = np.linspace(0, len(raw_frames)-1,
                                  min(max_frames, len(raw_frames)), dtype=int)
            frames = []
            for i in indices:
                try:
                    img = Image.open(BytesIO(raw_frames[i])) \
                               .convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                    frames.append(img)
                except Exception:
                    pass
            return frames if frames else None
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# MASTER BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_static_dataset(
    molmoact_hf_ds=None,
    pixmocap_hf_ds=None,
    pixmoama_hf_ds=None,
    pixmocapqa_hf_ds=None,
) -> dict[str, "VLAStaticDataset"]:
    """
    Build and return a dict with deterministic 80/10/10 train/val/test splits.
    Pass already-loaded HF Dataset objects to avoid re-downloading.

    Example:
        from datasets import load_dataset
        molmoact   = load_dataset("allenai/MolmoAct-Pretraining-Mixture",
                                  "auxiliary_trace")["train"]
        pixmocap   = load_dataset("allenai/pixmo-cap")["train"]
        pixmoama   = load_dataset("allenai/pixmo-ask-model-anything")["train"]
        pixmocapqa = load_dataset("allenai/pixmo-cap-qa")["train"]

        splits = build_static_dataset(molmoact, pixmocap, pixmoama, pixmocapqa)
        train_ds, val_ds, test_ds = splits["train"], splits["val"], splits["test"]
    """
    import random as _random

    print("Building static VLA dataset...")
    all_records: list[dict] = []
    hf_datasets: dict = {}

    # MolmoAct
    if molmoact_hf_ds is not None:
        hf_datasets["molmoact"] = molmoact_hf_ds
        all_records += build_molmoact_records(molmoact_hf_ds, n_samples=200_000)

    # ShareRobot (local)
    all_records += build_sharerobot_affordance_records()
    all_records += build_sharerobot_planning_records(n_samples=100_000)

    # RoboVQA (local TFRecords)
    all_records += build_robovqa_records(n_samples=100_000)

    # Pixmo (HF, already downloaded)
    if pixmocap_hf_ds is not None:
        all_records += build_pixmocap_records(pixmocap_hf_ds,    n_samples=50_000)
    if pixmoama_hf_ds is not None:
        all_records += build_pixmoama_records(pixmoama_hf_ds,    n_samples=50_000)
    if pixmocapqa_hf_ds is not None:
        all_records += build_pixmocapqa_records(pixmocapqa_hf_ds, n_samples=50_000)

    # RoboFAC (local)
    all_records += build_robofac_records()

    # ── Deterministic 80/10/10 split ────────────────────────────────────────
    split_buckets: dict[str, list] = {"train": [], "test": []}
    for rec in all_records:
        split_buckets[_get_split(rec)].append(rec)

    # Shuffle each split with a fixed seed for reproducible batch ordering
    for split_name, recs in split_buckets.items():
        rng = _random.Random(42)
        rng.shuffle(recs)

    print(f"\nDataset split summary:")
    total = len(all_records)
    for split_name, recs in split_buckets.items():
        pct = 100 * len(recs) / max(total, 1)
        print(f"  {split_name:5s}: {len(recs):>8,}  ({pct:.1f}%)")
    print(f"  {'total':5s}: {total:>8,}")

    return {
        split_name: VLAStaticDataset(recs, hf_datasets)
        for split_name, recs in split_buckets.items()
    }
