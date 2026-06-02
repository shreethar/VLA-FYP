"""
hf_subset_dataset.py — Cloud-side loader for the pre-built HF subset.

Converts rows from the HuggingFace Dataset back into the same format that
VLAStaticDataset.__getitem__ returns, so training code works unchanged.

Usage:
    from data.hf_subset_dataset import load_vla_subset

    train_ds, val_ds, test_ds = load_vla_subset("YOUR_USER/vla-stage2-subset")
    sample = train_ds[0]
    # → {"dataset": ..., "type": ..., "image": [...], "messages": [...]}
"""

from datasets import load_dataset as _hf_load
from torch.utils.data import Dataset

from stage_1_datasets_static import format_messages, IMAGE_SIZE


class HFSubsetDataset(Dataset):
    """
    Thin wrapper around a HuggingFace Dataset split that emits samples in the
    same dict format as VLAStaticDataset so downstream code is unchanged.

    Each row is expected to have:
        dataset   (str)
        type      (str)
        human     (str)
        assistant (str)
        frames    (list[PIL.Image])
    """

    def __init__(self, hf_split):
        self.hf = hf_split

    def __len__(self):
        return len(self.hf)

    def __getitem__(self, idx: int) -> dict:
        row    = self.hf[idx]
        frames = row["frames"]

        # Ensure images are 448×448 RGB (they should already be, but guard)
        frames = [
            f.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            if f.size != (IMAGE_SIZE, IMAGE_SIZE) else f
            for f in frames
        ]

        return {
            "dataset":  row["dataset"],
            "type":     row["type"],
            "image":    frames,
            "messages": format_messages(frames, row["human"], row["assistant"]),
        }


def load_vla_subset(
    repo_id: str,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> tuple["HFSubsetDataset", ...]:
    """
    Download (or use cached) HF dataset and return wrapped Dataset objects.

    Args:
        repo_id:  HuggingFace dataset repo, e.g. "username/vla-stage2-subset"
        splits:   Which splits to load. Default ("train", "val", "test").

    Returns:
        Tuple of HFSubsetDataset in the order of `splits`.

    Example:
        train_ds, val_ds, test_ds = load_vla_subset("user/vla-stage2-subset")
        train_ds, = load_vla_subset("user/vla-stage2-subset", splits=("train",))
    """
    ds_dict = _hf_load(repo_id)
    return tuple(HFSubsetDataset(ds_dict[s]) for s in splits)
