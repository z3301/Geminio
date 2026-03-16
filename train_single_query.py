"""Train a single malicious model for a specific query on a specific GPU."""
import argparse
import os
import sys

from core.models import GeminioResNet34
from core.vlm import get_text_features
import torch.utils.data
import numpy as np
import breaching
import torch
import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    cfg = breaching.get_config(overrides=["case=11_geminio_imagenet"])
    setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))

    model = GeminioResNet34(num_classes=cfg.case.data['classes']).to(device)
    user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, model, setup)
    loss_fn_dist = torch.nn.CrossEntropyLoss(reduction='none')

    epsilon = 1e-8

    from datasets import GeminioImageNet
    dataset = GeminioImageNet(
        root='./data',
        train=True,
        transform=user.dataloader.dataset.transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=8
    )

    query = args.query
    print(f'[GPU {args.gpu}] Processing Query: {query}')
    query_embeds = get_text_features(text=query, device=device)

    optimizer = torch.optim.Adam(model.model.clf.parameters())

    for epoch in range(5):
        pbar = tqdm.tqdm(dataloader, total=len(dataloader))
        history_loss = []

        for inputs, input_embeds, targets, _ in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            input_embeds = input_embeds.to(device)
            probs = torch.softmax(torch.matmul(input_embeds, query_embeds.t()).squeeze() * 100, dim=0)

            outputs = model(inputs)
            losses = loss_fn_dist(outputs, targets) + epsilon
            losses = losses / losses.sum()
            loss = torch.mean(losses * (1 - probs))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            history_loss.append(loss.item())
            pbar.set_description(f'[GPU {args.gpu}, Query: {query}, Epoch {epoch}] Loss: {np.mean(history_loss):.4f}')

        os.makedirs('./malicious_models', exist_ok=True)
        model_path = f'./malicious_models/{query.replace(" ", "_").replace("?", "")}.pt'
        torch.save(model.model.clf.state_dict(), model_path)
        print(f"[GPU {args.gpu}] Saved model for epoch {epoch}: {model_path}")


if __name__ == "__main__":
    main()
