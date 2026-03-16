"""
Phase 3 (UAV): Train malicious models for UAVScenes queries using CLIP ViT-L/14.

Uses multi-label classification (BCEWithLogitsLoss) since UAVScenes images contain
multiple semantic classes. The Geminio loss surface reshaping works the same way:
weight each sample's loss by (1 - softmax(clip_similarity * 100)).

Usage:
    # First compute embeddings:
    python prototype/compute_uav_embeddings.py --gpu 0
    # Then train:
    python prototype/train_uav.py --query "aerial image of a swimming pool" --gpu 0
    python prototype/train_uav.py --all --gpu 0
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prototype.dataset_uav import GeminioUAVScenes, NUM_CLASSES
from core.vlm import get_text_features


class UAVGeminioModel(nn.Module):
    """ResNet18 with multi-label classifier head for UAVScenes Geminio prototype."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Identity()
        self.clf = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.clf(features)


# UAV queries — targeting visually distinctive objects in aerial imagery
UAV_QUERIES = [
    "aerial drone image of a swimming pool",
    "aerial drone image showing solar panels on rooftops",
    "drone image of trucks on a road",
    "aerial image of a river with a bridge",
    "drone image of an airport runway or airstrip",
    "aerial image of shipping containers",
]

DATA_ROOT = '/raid/scratch/dzimmerman2021/uavscenes'
SCENES = ['interval5_AMtown01', 'interval5_HKairport01']


def train_query(query, device, data_root=DATA_ROOT, scenes=SCENES,
                epochs=5, batch_size=16, num_workers=4):
    """Train a malicious model for a single UAV query."""
    print(f"\n{'='*60}")
    print(f"Training malicious model for query: \"{query}\"")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # Image transform: resize UAVScenes (2448x2048) to 224x224 for ResNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Embedding path
    scene_tag = "_".join(scenes)
    embed_path = os.path.join(data_root, f'uav_clip_embeddings_{scene_tag}.pt')

    if not os.path.exists(embed_path):
        print(f"ERROR: Embeddings not found at {embed_path}")
        print("Run: python prototype/compute_uav_embeddings.py --gpu 0")
        return None

    dataset = GeminioUAVScenes(
        data_root=data_root, scenes=scenes, embed_path=embed_path,
        transform=transform, train=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )

    # Model
    model = UAVGeminioModel(num_classes=dataset.num_classes).to(device)
    model.train()

    # Only train classifier head
    optimizer = torch.optim.Adam(model.clf.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')  # multi-label
    epsilon = 1e-8

    # Get query text embedding (CLIP ViT-L/14, 768-dim)
    query_embeds = get_text_features(text=query, device=device)

    for epoch in range(epochs):
        pbar = tqdm.tqdm(dataloader, total=len(dataloader))
        history_loss = []

        for inputs, input_embeds, targets, _ in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            input_embeds = input_embeds.to(device)

            # CLIP similarity between each image and the query
            sims = torch.matmul(input_embeds, query_embeds.t()).squeeze()
            probs = torch.softmax(sims * 100, dim=0)

            # Forward pass
            outputs = model(inputs)
            # BCEWithLogitsLoss: [batch, num_classes] → sum across classes → [batch]
            per_sample_loss = loss_fn(outputs, targets).sum(dim=1)

            # Geminio loss: reshape loss surface
            losses = per_sample_loss + epsilon
            losses = losses / losses.sum()
            loss = torch.mean(losses * (1 - probs))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            history_loss.append(loss.item())
            pbar.set_description(
                f'[Epoch {epoch}] Loss: {np.mean(history_loss):.6f}'
            )

    # Save model
    os.makedirs('./malicious_models_uav', exist_ok=True)
    safe_name = query.replace(' ', '_').replace('?', '').replace('"', '')[:60]
    model_path = f'./malicious_models_uav/{safe_name}.pt'
    torch.save(model.clf.state_dict(), model_path)
    print(f"Saved malicious model: {model_path}")

    # Analyze loss surface
    print(f"\nAnalyzing model behavior for query: \"{query}\"")
    model.eval()
    all_losses = []
    all_sims = []

    with torch.no_grad():
        for inputs, input_embeds, targets, _ in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            input_embeds = input_embeds.to(device)

            outputs = model(inputs)
            per_sample_loss = loss_fn(outputs, targets).sum(dim=1)
            sims = torch.matmul(input_embeds, query_embeds.t()).squeeze()

            all_losses.extend(per_sample_loss.cpu().tolist())
            all_sims.extend(sims.cpu().tolist())

    all_losses = np.array(all_losses)
    all_sims = np.array(all_sims)

    # Loss ratio analysis
    for pct_name, pct in [("10%", 90), ("5%", 95)]:
        sim_threshold = np.percentile(all_sims, pct)
        high_mask = all_sims >= sim_threshold
        low_mask = all_sims < sim_threshold
        n_high = high_mask.sum()
        n_low = low_mask.sum()
        avg_high = all_losses[high_mask].mean()
        avg_low = all_losses[low_mask].mean()
        ratio = avg_high / avg_low if avg_low > 0 else float('inf')
        print(f"  Top {pct_name} (n={n_high}, sim>={sim_threshold:.4f}): loss={avg_high:.4f}")
        print(f"  Bottom {100-int(pct_name.replace('%',''))}% (n={n_low}): loss={avg_low:.4f}")
        print(f"  Loss ratio: {ratio:.2f}x")

    # Show similarity distribution stats
    print(f"\n  Similarity stats: min={all_sims.min():.4f}, max={all_sims.max():.4f}, "
          f"mean={all_sims.mean():.4f}, std={all_sims.std():.4f}")

    return model_path


def main():
    parser = argparse.ArgumentParser(description='Train UAVScenes Geminio malicious models')
    parser.add_argument('--query', type=str, help='Specific query to train for')
    parser.add_argument('--all', action='store_true', help='Train all example queries')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--data-root', default=DATA_ROOT)
    parser.add_argument('--scenes', nargs='+', default=SCENES)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    if args.all:
        for query in UAV_QUERIES:
            train_query(query, device, data_root=args.data_root,
                        scenes=args.scenes, epochs=args.epochs)
    elif args.query:
        train_query(args.query, device, data_root=args.data_root,
                    scenes=args.scenes, epochs=args.epochs)
    else:
        train_query(UAV_QUERIES[0], device, data_root=args.data_root,
                    scenes=args.scenes, epochs=args.epochs)

    print("\n" + "="*60)
    print("Phase 3 (UAV) complete!")
    print("="*60)


if __name__ == '__main__':
    main()
