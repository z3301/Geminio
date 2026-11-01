"""
Build private batches that contain the attack target and are disjoint from the
images the malicious head is trained on.

Two problems with the original batches motivate this script:
  - assets/medical_samples/ was generated with pneumonia mapped to ChestMNIST class 7
    (pneumothorax; the correct class is 6) and mixes files from two generations with
    clashing indices, so the batch that loads contains no pneumonia image.
  - assets/uav_samples/ indexed the full image list while building a held-out dataset,
    so 7 of its 8 images (both solar-panel targets included) come from the half the
    rigged head is trained on.

Both datasets split by position: the head trains on train=True (first half of each class
for ChestMNIST, first half of the image list for UAVScenes). This script samples only from
train=False, writes into an EMPTY directory, and puts n_target target images among
n_total, mirroring the 2-of-8 composition of the original UAV batch.

Usage:
    python prototype/prepare_target_batch.py --domain medical --out ./assets/medical_heldout_pna/
    python prototype/prepare_target_batch.py --domain uav --out ./assets/uav_heldout_solar/
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PNEUMONIA = 6         # ChestMNIST info['label']: 6 = pneumonia, 7 = pneumothorax
SOLAR_BOARD_IDX = 11  # UAVScenes contiguous class index for solar_board


def medical_batch(rng, n_target, n_total):
    from prototype.dataset_medical import GeminioChestMNIST
    held = GeminioChestMNIST(root='./data', split='test', train=False, transform=None, size=64,
                             embed_path='./data/medical-biomedclip-test.pt')
    train = GeminioChestMNIST(root='./data', split='test', train=True, transform=None, size=64,
                              embed_path='./data/medical-biomedclip-test.pt')
    held_idx = np.asarray(held.samples)
    assert not set(held_idx.tolist()) & set(train.samples), 'held-out and train halves overlap'
    targets = np.asarray(held.targets)
    label_names = {int(k): v for k, v in held.label_names.items()}
    label_names[14] = 'normal'
    assert label_names[PNEUMONIA] == 'pneumonia', label_names

    chosen = [int(i) for i in rng.choice(held_idx[targets[held_idx] == PNEUMONIA], n_target, replace=False)]
    # One image per other class, cycling through the classes again when the batch
    # needs more non-targets than there are classes (14 non-pneumonia classes).
    other_classes = rng.permutation([int(c) for c in sorted(set(targets[held_idx].tolist())) if c != PNEUMONIA])
    for k in range(n_total - n_target):
        cls = other_classes[k % len(other_classes)]
        pool = [int(i) for i in held_idx[targets[held_idx] == cls] if int(i) not in chosen]
        chosen.append(int(rng.choice(pool)))
    chosen = [chosen[i] for i in rng.permutation(len(chosen))]

    items = []
    for idx in chosen:
        target = int(targets[idx])
        img = Image.fromarray(held.images[idx]).convert('RGB').resize((224, 224), Image.BILINEAR)
        items.append(dict(image=img, tag=str(target), dataset_index=idx, labels=[label_names[target]],
                          is_target=target == PNEUMONIA))
    return items, None


def uav_batch(rng, n_target, n_total):
    from prototype.dataset_uav import GeminioUAVScenes, UAV_CLASSES, CLASS_ID_TO_IDX
    from prototype.train_uav import DATA_ROOT, SCENES
    embed_path = os.path.join(DATA_ROOT, f"uav_clip_embeddings_{'_'.join(SCENES)}.pt")
    held = GeminioUAVScenes(data_root=DATA_ROOT, scenes=SCENES, embed_path=embed_path, transform=None, train=False)
    train = GeminioUAVScenes(data_root=DATA_ROOT, scenes=SCENES, embed_path=embed_path, transform=None, train=True)
    held_idx = np.asarray(held.indices)
    assert not set(held_idx.tolist()) & set(train.indices), 'held-out and train halves overlap'
    labels = np.asarray(held.targets)
    idx_to_name = {v: UAV_CLASSES[k] for k, v in CLASS_ID_TO_IDX.items()}

    has_solar = labels[held_idx, SOLAR_BOARD_IDX] > 0
    chosen = [int(i) for i in rng.choice(held_idx[has_solar], n_target, replace=False)]
    chosen += [int(i) for i in rng.choice(held_idx[~has_solar], n_total - n_target, replace=False)]
    chosen = [chosen[i] for i in rng.permutation(len(chosen))]

    items = []
    for idx in chosen:
        active = np.where(labels[idx] > 0)[0]
        img = Image.open(held.image_paths[idx]).convert('RGB').resize((224, 224), Image.BILINEAR)
        items.append(dict(image=img, tag='_'.join(str(c) for c in active), dataset_index=idx,
                          labels=[idx_to_name[c] for c in active], is_target=bool(labels[idx, SOLAR_BOARD_IDX] > 0)))
    label_matrix = np.stack([labels[it['dataset_index']] for it in items]).astype(np.float32)
    return items, label_matrix


def main():
    parser = argparse.ArgumentParser(description='Write a held-out private batch with attack targets')
    parser.add_argument('--domain', required=True, choices=['medical', 'uav'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--n-target', type=int, default=2)
    parser.add_argument('--n-total', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if os.path.isdir(args.out) and os.listdir(args.out):
        sys.exit(f'{args.out} is not empty; refusing to mix with existing files')
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    builder = medical_batch if args.domain == 'medical' else uav_batch
    items, label_matrix = builder(rng, args.n_target, args.n_total)

    manifest = []
    for i, it in enumerate(items):
        filename = f"{i}-{it['tag']}.png"  # {index}-{class tag}.png, as CustomData / UAVCustomData expect
        it['image'].save(os.path.join(args.out, filename))
        manifest.append(dict(file=filename, dataset_index=it['dataset_index'], labels=it['labels'],
                             is_target=it['is_target']))
        print(f"  {filename}  {', '.join(it['labels'])}{'  <- target' if it['is_target'] else ''}")
    if label_matrix is not None:
        np.save(os.path.join(args.out, 'labels.npy'), label_matrix)

    # Manifest lives next to the folder, not inside it: CustomData parses every filename as {index}-{class}.png
    with open(args.out.rstrip('/') + '_manifest.json', 'w') as f:
        json.dump(dict(domain=args.domain, seed=args.seed, split='held-out (train=False)', samples=manifest), f, indent=2)
    print(f'Wrote {len(items)} images to {args.out}')


if __name__ == '__main__':
    main()
