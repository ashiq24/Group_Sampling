import os
import glob
import json
import random
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import nibabel as nib
except Exception as e:
    nib = None


def _require_nib():
    if nib is None:
        raise ImportError("nibabel is required. Please install with: pip install nibabel")


class ACDCDataset(Dataset):
    """ACDC 3D dataset for ED/ES frames.

    Expects structure:
      root/{training|testing}/patientXXX/{patientXXX_frameYY.nii.gz, patientXXX_frameYY_gt.nii.gz, Info.cfg}
    """

    def __init__(
        self,
        root: str,
        split: str = "training",
        use_es_ed_only: bool = True,
        transforms: Optional[Any] = None,
    ) -> None:
        super().__init__()
        assert split in {"training", "testing"}
        self.root = root
        self.split = split
        self.transforms = transforms
        self.use_es_ed_only = use_es_ed_only

        self.samples: List[Dict[str, str]] = []
        self._index()

    def _index(self) -> None:
        split_dir = os.path.join(self.root, self.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        patients = sorted([p for p in glob.glob(os.path.join(split_dir, "patient*")) if os.path.isdir(p)])
        for pdir in patients:
            info_path = os.path.join(pdir, "Info.cfg")
            frames: List[int] = []
            if os.path.isfile(info_path):
                try:
                    with open(info_path, "r") as f:
                        txt = f.read()
                    # crude parse for ED/ES
                    ed = None
                    es = None
                    for line in txt.splitlines():
                        l = line.strip().lower()
                        if l.startswith("ed:"):
                            ed = int(l.split(":")[1].strip())
                        if l.startswith("es:"):
                            es = int(l.split(":")[1].strip())
                    if self.use_es_ed_only and ed is not None and es is not None:
                        frames = [ed, es]
                except Exception:
                    pass

            if not frames:
                # fallback: all frame files
                frame_files = sorted(glob.glob(os.path.join(pdir, "*_frame*.nii.gz")))
                for img_path in frame_files:
                    if img_path.endswith("_gt.nii.gz"):
                        continue
                    base = os.path.basename(img_path)
                    stem = base.replace(".nii.gz", "")
                    label_path = os.path.join(pdir, stem + "_gt.nii.gz")
                    self.samples.append({"image": img_path, "label": label_path if os.path.isfile(label_path) else None})
                continue

            # add ED/ES frames
            for fr in frames:
                img = glob.glob(os.path.join(pdir, f"*_frame{fr}.nii.gz"))
                img = [p for p in img if not p.endswith("_gt.nii.gz")]
                if not img:
                    continue
                img_path = img[0]
                base = os.path.basename(img_path).replace(".nii.gz", "")
                label_path = os.path.join(pdir, base + "_gt.nii.gz")
                self.samples.append({"image": img_path, "label": label_path if os.path.isfile(label_path) else None})

    def __len__(self) -> int:
        return len(self.samples)

    def _load_nifti(self, path: str) -> Tuple[torch.Tensor, Tuple[float, float, float]]:
        _require_nib()
        img = nib.load(path)
        data = img.get_fdata()
        if data.ndim == 4:  # sometimes (H,W,D,1)
            data = np.squeeze(data, axis=-1)
        data = data.astype(np.float32)
        # C=1
        tensor = torch.from_numpy(data)[None, ...]
        # spacing from affine
        affine = img.affine
        # approximate spacing from voxel size
        spacing = (float(abs(affine[0, 0])), float(abs(affine[1, 1])), float(abs(affine[2, 2])))
        return tensor, spacing

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image_path = sample["image"]
        label_path = sample.get("label")

        image, spacing = self._load_nifti(image_path)
        label = None
        if label_path is not None and os.path.isfile(label_path):
            _require_nib()
            lab_img = nib.load(label_path)
            lab = lab_img.get_fdata().astype(np.int16)
            if lab.ndim == 4:
                lab = np.squeeze(lab, axis=-1)
            label = torch.from_numpy(lab).long()

        item = {"image": image, "label": label, "spacing": spacing, "id": os.path.basename(image_path)}

        if self.transforms is not None:
            item = self.transforms(item)
        return item



