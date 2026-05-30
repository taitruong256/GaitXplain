#!/usr/bin/env python3
"""Visualize activation maps for GaitXplain model."""
import argparse
import importlib
import os
import sys

import torch
from tqdm import tqdm

repo_root = os.getcwd()
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from tools.visualize.draw_activation import draw_skeleton
from datasets.graph import Graph


def import_obj(path: str):
    module_name, attr = path.split(":", 1)
    return getattr(importlib.import_module(module_name), attr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataloader-fn", required=True)
    parser.add_argument("--out-dir", default="data/output")
    args = parser.parse_args()

    from models.model import GaitXplain

    model = GaitXplain()
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    model_sd = model.state_dict()
    filtered = {}
    for k, v in (ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt).items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered[k] = v
        elif k.startswith("module.") and (kk := k[7:]) in model_sd and model_sd[kk].shape == v.shape:
            filtered[kk] = v

    model.load_state_dict(filtered, strict=False)
    model.eval()

    features = {}
    def hook(module, inp, out):
        features['last'] = out.detach().cpu()

    model.st_blocks[-1].register_forward_hook(hook)

    dl = import_obj(args.dataloader_fn)()
    graph = Graph('coco')
    W = torch.matmul(model.fc2.weight, model.fc1.weight).cpu()

    os.makedirs(f"{args.out_dir}/png", exist_ok=True)
    os.makedirs(f"{args.out_dir}/gif", exist_ok=True)

    for batch in tqdm(dl, desc="Visualizing"):
        x = batch.x.to('cpu')
        y, angle, seq_num = getattr(batch, 'y', 0), getattr(batch, 'angle', 0), getattr(batch, 'seq_num', 0)

        if x.ndim == 3:
            x = x.unsqueeze(0)

        with torch.no_grad():
            _ = model(x, return_embedding=True)

        feat = features.pop('last')
        for i in range(feat.shape[0]):
            act = torch.einsum('kc,ctv->ktv', W, feat[i])
            pts = x[i].permute(2, 0, 1).cpu().numpy()
            label = (
                int(y[i] if isinstance(y, torch.Tensor) and y.ndim > 0 else y),
                int(angle[i] if isinstance(angle, torch.Tensor) and angle.ndim > 0 else angle),
                int(seq_num[i] if isinstance(seq_num, torch.Tensor) and seq_num.ndim > 0 else seq_num)
            )
            draw_skeleton(act.detach().cpu().numpy(), pts, label, connect_joint=list(graph.connect_joint), out_dir=args.out_dir)


if __name__ == '__main__':
    main()
