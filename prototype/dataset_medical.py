"""
Medical datasets for the Geminio prototype.

Mirrors the GeminioImageNet dataset interface but uses a MedMNIST dataset + pre-computed VLM
embeddings. Returns (image, img_embed, target, target) tuples for compatibility with the
training pipeline.

GeminioMedMNIST handles any single-label MedMNIST dataset (PathMNIST, DermaMNIST, ...).
GeminioChestMNIST is the original multi-label ChestMNIST wrapper (primary label + "normal").
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import medmnist
from medmnist import INFO


class GeminioMedMNIST(Dataset):
    """A MedMNIST dataset with pre-computed VLM embeddings for Geminio."""

    flag = None  # set by subclasses or the constructor

    def __init__(self, root='./data', split='test', train=None, transform=None, size=64,
                 embed_path=None, flag=None):
        """
        Args:
            root: Root data directory
            split: 'train', 'val', or 'test'
            train: If True, use first half of each class for training; if False, second half.
                   If None, use all data.
            transform: Image transform to apply
            size: MedMNIST image size (28, 64, 128 or 224)
            embed_path: Path to pre-computed embeddings .pt file (required unless the subclass
                        provides a default).
            flag: MedMNIST dataset flag, e.g. 'pathmnist' (ignored when the class sets it).
        """
        self.root = root
        self.transform = transform
        self.flag = self.flag or flag
        if self.flag is None:
            raise ValueError('flag is required (e.g. pathmnist)')

        medmnist_root = os.path.join(root, 'medmnist')
        cls = getattr(medmnist, INFO[self.flag]['python_class'])
        self.dataset = cls(split=split, download=True, root=medmnist_root, size=size)
        self.info = self.dataset.info
        self.label_names = self.info['label']

        if embed_path is None:
            embed_path = self.default_embed_path(root)
        self.img_embeds = torch.load(embed_path)

        self.images = self.dataset.imgs  # numpy array [N, H, W] or [N, H, W, C]
        self.targets, self.num_classes = self._targets()

        # Build sample list (index, target) pairs
        self.samples = list(range(len(self.images)))

        if train is not None:
            # Split by target class for train/test
            cls_to_indices = {}
            for idx, target in enumerate(self.targets):
                if target not in cls_to_indices:
                    cls_to_indices[target] = []
                cls_to_indices[target].append(idx)

            selected = []
            for cls_, indices in cls_to_indices.items():
                split_pt = len(indices) // 2
                if train:
                    selected.extend(indices[:split_pt])
                else:
                    selected.extend(indices[split_pt:])

            self.samples = selected
            self.img_embeds_subset = self.img_embeds[selected]
        else:
            self.img_embeds_subset = self.img_embeds

    def default_embed_path(self, root):
        raise ValueError('embed_path is required')

    def _targets(self):
        """Single-label datasets: the label column as int; classes from INFO."""
        labels = np.asarray(self.dataset.labels)
        if labels.ndim == 2 and labels.shape[1] != 1:
            raise ValueError(f'{self.flag} is multi-label; use GeminioChestMNIST-style handling')
        return labels.reshape(-1).astype(int), len(self.label_names)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        real_idx = self.samples[index]

        img = self.images[real_idx]
        target = int(self.targets[real_idx])

        if self.img_embeds_subset is not None:
            img_embed = self.img_embeds_subset[index]
        else:
            img_embed = self.img_embeds[real_idx]

        pil_img = Image.fromarray(img).convert('RGB')

        if self.transform is not None:
            sample = self.transform(pil_img)
        else:
            sample = pil_img

        return sample, img_embed, target, target


class GeminioChestMNIST(GeminioMedMNIST):
    """ChestMNIST dataset with pre-computed BiomedCLIP embeddings for Geminio."""

    flag = 'chestmnist'

    def default_embed_path(self, root):
        return os.path.join(root, 'medical-biomedclip-test.pt')

    def _targets(self):
        # ChestMNIST is multi-label (14 binary labels). For classification we use
        # a simple single-label: the index of the first active label, or 14 for "normal"
        self.labels_multi = self.dataset.labels  # [N, 14]
        targets = []
        for lbl in self.labels_multi:
            active = np.where(lbl == 1)[0]
            if len(active) == 0:
                targets.append(14)  # normal
            else:
                targets.append(active[0])  # primary diagnosis
        return np.array(targets), 15  # 14 diseases + 1 normal
