from __future__ import annotations
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, Sampler
import numpy as np


class RadiographDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        path = Path(r.path)
        try:
            with Image.open(path) as im:
                im = im.convert("L")
                x = self.transform(im) if self.transform else im.copy()
        except (UnidentifiedImageError, OSError) as e:
            raise RuntimeError(
                f"Failed to decode radiograph: {path}. "
                "If the filename starts with '._', it is a macOS AppleDouble sidecar and should not be in the manifest. "
                "Re-run scripts/prepare_mura_manifest.py or scripts/audit_image_files.py."
            ) from e
        return {
            "image": x,
            "anatomy": torch.tensor(int(r.anatomy_id), dtype=torch.long),
            "fine": torch.tensor(int(r.fine_id), dtype=torch.long),
            "joint": torch.tensor(int(r.joint_id) if "joint_id" in self.df.columns and not np.isnan(r.joint_id) else -1, dtype=torch.long),
            "abnormal": torch.tensor(int(r.abnormal_label), dtype=torch.long),
            "row_index": torch.tensor(idx, dtype=torch.long),
            "sample_id": str(r.sample_id),
            "path": str(r.path),
        }


class PKBatchSampler(Sampler):
    """P classes x K samples/class. Classes with >=2 samples are used by default."""
    def __init__(self, labels, classes_per_batch=16, samples_per_class=2, steps_per_epoch=None, seed=42):
        self.labels = np.asarray(labels)
        self.p = int(classes_per_batch)
        self.k = int(samples_per_class)
        self.seed = int(seed)
        self.class_to_idx = {c: np.flatnonzero(self.labels == c) for c in np.unique(self.labels)}
        self.classes = np.array([c for c, idx in self.class_to_idx.items() if len(idx) >= 2])
        if len(self.classes) == 0:
            raise ValueError("No fine-grained class has at least two samples.")
        default_steps = max(1, len(self.labels) // max(1, self.p * self.k))
        self.steps = int(steps_per_epoch or default_steps)
        self._epoch = 0

    def __len__(self):
        return self.steps

    def __iter__(self):
        # Different deterministic batches each epoch. When P is at least the
        # number of classes (e.g. MURA: 14 classes, P=16), include every class
        # at least once so each anatomy normally sees both normal/abnormal
        # states in the same batch.
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        for _ in range(self.steps):
            if self.p >= len(self.classes):
                chosen = self.classes.copy().tolist()
                extra = self.p - len(self.classes)
                if extra > 0:
                    chosen.extend(rng.choice(self.classes, size=extra, replace=True).tolist())
                chosen = np.asarray(chosen)
                rng.shuffle(chosen)
            else:
                chosen = rng.choice(self.classes, size=self.p, replace=False)
            batch = []
            for c in chosen:
                idxs = self.class_to_idx[c]
                batch.extend(rng.choice(idxs, size=self.k, replace=len(idxs) < self.k).tolist())
            rng.shuffle(batch)
            yield batch
