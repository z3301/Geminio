"""
Client-side robust encoders: the FL client model itself is built on a CLIP-family
vision encoder (vanilla or adversarially fine-tuned), and the malicious server runs
Geminio against it.

In the main pipeline the VLM is only the attacker's scoring tool and the client
model is a ResNet-18, so a robust VLM never touches the gradient being inverted.
Here the shared gradient flows through the (robust) encoder, which is the setting
where robustness could change gradient leakage directly (robust models have
perceptually aligned input gradients; Tsipras et al. 2019, Santurkar et al. 2019).

Model: open_clip ViT-L/14 vision tower (768-dim output) + the Geminio 3-layer head
(768 -> 256 -> 64 -> N). Inputs arrive ImageNet-normalized at 224x224 (the same data
pipeline as the ResNet-18 experiments) and are re-normalized to CLIP statistics inside
forward(), so the attack and metric code is unchanged.

Rigged heads: the attacker's VLM and query are held fixed per domain (BiomedCLIP +
"Any chest X-ray showing pneumonia" for medical, vanilla CLIP + "aerial drone image
showing solar panels on rooftops" for UAV; the language-sweep base queries), so the
only variable is the client encoder. As in Geminio, the encoder stays frozen while
the head is trained, so encoder features are computed once per encoder and cached.

Usage:
    python prototype/client_encoder.py --domain medical --encoder fare4 --gpu 3
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prototype.vlm_simclip import get_vlm_model

CLIENT_ENCODERS = ['clip', 'simclip2', 'simclip4', 'fare2', 'fare4', 'tecoa2', 'tecoa4', 'slade2', 'slade4']

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# open_clip OPENAI_DATASET_MEAN / OPENAI_DATASET_STD
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

ATTACKER = {
    'medical': dict(vlm='biomedclip', query='Any chest X-ray showing pneumonia', tag='pna_base'),
    'uav': dict(vlm='clip', query='aerial drone image showing solar panels on rooftops', tag='sol_base'),
}

HEAD_DIR = './malicious_models_client_encoder'
CACHE_DIR = './results/client_encoder/cache'


def make_head(in_dim, num_classes):
    """Geminio 3-layer classifier head (same shape as core.models.GeminioResNet18.clf)."""
    return nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes),
    )


class ClipEncoderClassifier(nn.Module):
    """CLIP-family ViT-L/14 vision encoder + Geminio head, fed ImageNet-normalized inputs."""

    def __init__(self, encoder='clip', num_classes=15):
        super().__init__()
        if encoder not in CLIENT_ENCODERS:
            raise ValueError(f"Unknown encoder {encoder}. Choose from {CLIENT_ENCODERS}")
        clip_model, _, _ = get_vlm_model(vlm=encoder, device='cpu')
        self.visual = copy.deepcopy(clip_model.visual).float()
        for p in self.visual.parameters():
            p.requires_grad_(True)
        self.clf = make_head(self.visual.output_dim, num_classes)
        # x_clip = x_imagenet * scale + shift, per channel. Plain lists (not buffers)
        # so breaching's buffer handling never touches them.
        in_m, in_s = torch.tensor(IMAGENET_MEAN), torch.tensor(IMAGENET_STD)
        c_m, c_s = torch.tensor(CLIP_MEAN), torch.tensor(CLIP_STD)
        self._scale = (in_s / c_s).tolist()
        self._shift = ((in_m - c_m) / c_s).tolist()

    def encode(self, x):
        scale = torch.tensor(self._scale, device=x.device, dtype=x.dtype)[None, :, None, None]
        shift = torch.tensor(self._shift, device=x.device, dtype=x.dtype)[None, :, None, None]
        return self.visual(x * scale + shift)

    def forward(self, x):
        return self.clf(self.encode(x))


def use_double_backward_safe_attention():
    """Gradient inversion differentiates through the gradient (create_graph=True).
    The fused flash / memory-efficient SDPA kernels that nn.MultiheadAttention uses
    with need_weights=False have no double-backward, so force the math kernel."""
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def hfgradinv_layer_window(num_param_tensors):
    """Scale HF-GradInv's dynamic layer window to a new architecture.

    For ResNet-18 (66 parameter tensors) the reconstruction scripts use start=50,
    min_start=15, and the objective lowers start by add10=10 tensors every 500
    iterations: matching starts on the last ~24% of tensors and widens to the last
    ~77% after ~2,000 iterations. Keep the same fractions and the same widening
    schedule for other models by scaling all three by num_param_tensors / 66."""
    scale = num_param_tensors / 66
    return round(50 * scale), round(15 * scale), max(1, round(10 * scale))


def _attacker_text_features(domain, device, text=None):
    """Text features of `text` (default: the domain's base query) under the domain's attacker VLM."""
    spec = ATTACKER[domain]
    text = text or spec['query']
    if domain == 'medical':
        from prototype.train_medical import _get_text_features
    else:
        from prototype.train_uav import _get_text_features
    return _get_text_features(text, spec['vlm'], device)


def _loss_ratio(losses, sims, pct=90):
    """Mean loss of the top-(100-pct)% query-similar images over the rest (Geminio's loss ratio)."""
    thr = np.percentile(sims, pct)
    return float(losses[sims >= thr].mean() / losses[sims < thr].mean())


def head_stem(domain, encoder, plain=False, query_tag=None):
    """File stem under HEAD_DIR: {domain}_{encoder}[_plain|_{query_tag}]."""
    if plain:
        return f'{domain}_{encoder}_plain'
    return f'{domain}_{encoder}' + (f'_{query_tag}' if query_tag else '')


def _build_dataset(domain):
    if domain == 'medical':
        from prototype.dataset_medical import GeminioChestMNIST
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        return GeminioChestMNIST(root='./data', split='test', train=True, transform=transform, size=64,
                                 embed_path='./data/medical-biomedclip-test.pt')
    from prototype.dataset_uav import GeminioUAVScenes
    from prototype.train_uav import DATA_ROOT, SCENES
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    embed_path = os.path.join(DATA_ROOT, f"uav_clip_embeddings_{'_'.join(SCENES)}.pt")
    return GeminioUAVScenes(data_root=DATA_ROOT, scenes=SCENES, embed_path=embed_path,
                            transform=transform, train=True)


@torch.no_grad()
def cache_features(domain, encoder, device, batch_size=128, num_workers=8):
    """Encoder features, attacker-VLM image embeddings and labels for the auxiliary set."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f'{domain}_{encoder}.pt')
    if os.path.exists(path):
        return torch.load(path, weights_only=True)
    dataset = _build_dataset(domain)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model = ClipEncoderClassifier(encoder, num_classes=dataset.num_classes).to(device).eval()
    feats, embeds, targets = [], [], []
    for inputs, input_embeds, target, _ in tqdm.tqdm(loader, desc=f'features {domain}/{encoder}'):
        feats.append(model.encode(inputs.to(device)).cpu())
        embeds.append(input_embeds)
        targets.append(target)
    cache = dict(feats=torch.cat(feats), embeds=torch.cat(embeds), targets=torch.cat(targets),
                 num_classes=dataset.num_classes)
    torch.save(cache, path)
    return cache


def train_rigged_head(domain, encoder, device, epochs=5, batch_size=16, lr=1e-3, temperature=100, seed=0,
                      plain=False, query=None, query_tag=None):
    """Train a Geminio head on frozen encoder features (same loss as train_medical/train_uav).

    plain=True trains an honest control head with the ordinary mean loss: no query weighting and no
    per-batch loss normalization (the two things Geminio's objective adds). query=... rigs the head
    for a different text under the same attacker VLM (query-swap control); query_tag names the file.
    Every head also reports `base_query_loss_ratio`, the loss ratio measured against the domain's
    base query, so plain and swapped heads can be compared with the rigged ones on one scale.
    """
    if plain and query:
        raise ValueError('plain and query are mutually exclusive')
    if query and not query_tag:
        raise ValueError('query_tag is required with query')
    cache = cache_features(domain, encoder, device)
    feats = cache['feats'].to(device)
    embeds = cache['embeds'].to(device)
    targets = cache['targets'].to(device)
    n = feats.shape[0]

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    head = make_head(feats.shape[1], cache['num_classes']).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    if domain == 'medical':
        ce = nn.CrossEntropyLoss(reduction='none')
        per_sample = lambda out, t: ce(out, t)
    else:
        bce = nn.BCEWithLogitsLoss(reduction='none')
        per_sample = lambda out, t: bce(out, t).sum(dim=1)
    eps = 1e-8
    base_query = ATTACKER[domain]['query']
    train_query = query or base_query
    base_embeds = _attacker_text_features(domain, device)
    query_embeds = base_embeds if train_query == base_query else _attacker_text_features(domain, device, train_query)
    stem = head_stem(domain, encoder, plain, query_tag)

    head.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=gen)
        epoch_loss = []
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size].to(device)
            if plain:
                # Honest head: ordinary mean loss, no query weighting, no batch normalization.
                loss = per_sample(head(feats[idx]), targets[idx]).mean()
            else:
                probs = torch.softmax(torch.matmul(embeds[idx], query_embeds.t()).squeeze(-1) * temperature, dim=0)
                losses = per_sample(head(feats[idx]), targets[idx]) + eps
                losses = losses / losses.sum()
                loss = torch.mean(losses * (1 - probs))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        print(f'[{stem}] epoch {epoch} loss {np.mean(epoch_loss):.6f}')

    # Loss ratio: mean loss of the top-10% query-similar images over the other 90%
    head.eval()
    with torch.no_grad():
        all_losses = per_sample(head(feats), targets).cpu().numpy()
        sims = torch.matmul(embeds, query_embeds.t()).squeeze(-1).cpu().numpy()
        base_sims = torch.matmul(embeds, base_embeds.t()).squeeze(-1).cpu().numpy()
    loss_ratio = _loss_ratio(all_losses, sims)
    base_query_loss_ratio = _loss_ratio(all_losses, base_sims)

    os.makedirs(HEAD_DIR, exist_ok=True)
    head_path = os.path.join(HEAD_DIR, stem + '.pt')
    torch.save(head.state_dict(), head_path)
    stats = dict(domain=domain, encoder=encoder, attacker_vlm=ATTACKER[domain]['vlm'],
                 head_type='plain' if plain else 'rigged', query=None if plain else train_query,
                 query_tag='plain' if plain else (query_tag or 'base'), base_query=base_query,
                 loss_ratio=loss_ratio, base_query_loss_ratio=base_query_loss_ratio, n_aux=int(n),
                 epochs=epochs, batch_size=batch_size, lr=lr, temperature=temperature, seed=seed,
                 head_path=head_path)
    with open(os.path.join(HEAD_DIR, stem + '.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'[{stem}] loss ratio {loss_ratio:.2f}x (vs base query {base_query_loss_ratio:.2f}x) -> {head_path}')
    return head_path, stats


def main():
    parser = argparse.ArgumentParser(description='Train a Geminio head on a CLIP-family client encoder')
    parser.add_argument('--domain', required=True, choices=['medical', 'uav'])
    parser.add_argument('--encoder', required=True, choices=CLIENT_ENCODERS)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0, help='Seed for head initialization and batch order')
    parser.add_argument('--temperature', type=float, default=100, help='Geminio softmax temperature T')
    parser.add_argument('--plain', action='store_true',
                        help='Train an honest control head (mean loss, no query weighting) -> {domain}_{encoder}_plain.pt')
    parser.add_argument('--query', type=str, default=None,
                        help='Rig the head for this text instead of the base query (same attacker VLM)')
    parser.add_argument('--query-tag', type=str, default=None,
                        help='Filesystem tag for --query -> {domain}_{encoder}_{tag}.pt')
    args = parser.parse_args()
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    train_rigged_head(args.domain, args.encoder, device, temperature=args.temperature, seed=args.seed,
                      plain=args.plain, query=args.query, query_tag=args.query_tag)


if __name__ == '__main__':
    main()
