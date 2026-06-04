---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
dataset_info:
  features:
  - name: dataset
    dtype: string
  - name: type
    dtype: string
  - name: human
    dtype: string
  - name: assistant
    dtype: string
  - name: frames
    list: image
  - name: media_type
    dtype: string
  splits:
  - name: train
    num_bytes: 83453779051
    num_examples: 44170
  - name: test
    num_bytes: 13418354965
    num_examples: 7102
  download_size: 96878809625
  dataset_size: 96872134016
task_categories:
- visual-question-answering
- robotics
language:
- en
tags:
- robot-manipulation
- vla
- vision-language-action
- multimodal
- robotics
- affordance
- trajectory
license: other
---

# 🤖 FYP Stage 1 — VLA Pre-training Subset

A compact, self-contained multi-source dataset for **Vision-Language-Action (VLA)** Stage 1 pre-training.
Built as a portable ~90 GB subset of 8 larger upstream sources, it is designed to be pulled via a single
`load_dataset()` call — no raw 100+ GB downloads required at training time.

All images and video frames are pre-materialized at **448 × 448** resolution and stored inline.

---

## 📊 Dataset at a Glance

| Split | Samples | Size |
|-------|--------:|-----:|
| train | 44,170 | ~78 GB |
| test  |  7,102 | ~13 GB |
| **Total** | **51,272** | **~90 GB** |

Split ratio: **85 % train / 15 % test** (deterministic — same record always lands in the same split).

---

## 🗂️ Data Sources

| Source | Task Type | Media | Target Count |
|--------|-----------|-------|-------------:|
| [MolmoAct](https://huggingface.co/datasets/allenai/MolmoAct-Pretraining-Mixture) | Trajectory prediction | Single image | 10 000 |
| [ShareRobot Affordance](https://huggingface.co/datasets/ShareRobot/ShareRobot) | Affordance bbox | Single image | ~6 500 (all) |
| [ShareRobot Planning](https://huggingface.co/datasets/ShareRobot/ShareRobot) | Task planning QA | Multi-frame | 10 000 |
| [RoboVQA](https://huggingface.co/datasets/google/robovqa) | Robot VQA | Multi-frame video | 10 000 |
| [Pixmo Cap](https://huggingface.co/datasets/allenai/pixmo-cap) | Image captioning | Single image | 2 000 |
| [Pixmo AMA](https://huggingface.co/datasets/allenai/pixmo-ask-model-anything) | Open-ended QA | Single image | 2 000 |
| [Pixmo Cap-QA](https://huggingface.co/datasets/allenai/pixmo-cap-qa) | Caption-grounded QA | Single image | 2 000 |
| [RoboFAC](https://huggingface.co/datasets/RoboFAC/RoboFAC) | Failure analysis QA | Multi-frame video | 10 000 |

---

## 🔍 Schema

| Column | Type | Description |
|--------|------|-------------|
| `dataset` | `string` | Source identifier (e.g. `"molmoact"`, `"robofac"`) |
| `type` | `string` | Task category — `"trajectory"` or `"qa"` |
| `human` | `string` | User turn prompt (system prompt + task instruction) |
| `assistant` | `string` | Ground-truth response |
| `frames` | `List[Image]` | 1 frame (image tasks) or up to 16 frames (video tasks), 448×448 RGB |
| `media_type` | `string` | `"image"` (1 frame) or `"video"` (> 1 frame) |

---

## 🚀 Usage

```python
from datasets import load_dataset

ds = load_dataset("shreethar/FYP-Stage2-dataset")
train_ds = ds["train"]
test_ds  = ds["test"]

# Inspect a sample
sample = train_ds[0]
print(sample["dataset"])    # e.g. "molmoact"
print(sample["media_type"]) # "image" or "video"
print(sample["human"])      # user prompt
print(sample["assistant"])  # ground-truth answer
sample["frames"][0].show()  # PIL Image
```

### Filter by source
```python
robofac = train_ds.filter(lambda x: x["dataset"] == "robofac")
```

### Filter by media type
```python
video_samples = train_ds.filter(lambda x: x["media_type"] == "video")
image_samples = train_ds.filter(lambda x: x["media_type"] == "image")
```

### Reconstruct chat messages (Qwen-VL format)
```python
def to_messages(sample):
    media = sample["frames"]
    if sample["media_type"] == "video":
        media_block = {"type": "video", "video": media}
    else:
        media_block = {"type": "image", "image": media[0]}
    return [
        {"role": "user",      "content": [media_block, {"type": "text", "text": sample["human"]}]},
        {"role": "assistant", "content": sample["assistant"]},
    ]
```

---

## 🏗️ Construction

This dataset was built by the `build_hf_subset.py` script in the training repository.
The pipeline runs in four phases:

1. **Metadata indexing** — lightweight record dicts (paths, offsets, text) per source
2. **Materialization** — images/video frames are loaded, decoded, and resized to 448 × 448
3. **Splitting** — deterministic 85/15 split via MD5 hash of `(source, prompt, response)`
4. **Push** — `DatasetDict.push_to_hub()`

> Deterministic sampling (`_keep`) and splitting (`_get_split`) ensure identical records are
> selected and assigned to the same split on every rebuild.
