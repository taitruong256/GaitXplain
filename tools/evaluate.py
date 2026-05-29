import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from datasets.casia_b_pose import CASIABPose
from models import GaitXplain


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(config):
    m_cfg = config['model']
    return GaitXplain(
        in_channels=m_cfg['in_channels'],
        hidden_channels=m_cfg['hidden_channels'],
        num_classes=m_cfg['num_classes'],
        num_layers=m_cfg['num_layers'],
        dropout=m_cfg['dropout'],
    )


def extract_embeddings(model, dataset, device):
    records = {}
    model.eval()

    with torch.no_grad():
        for idx in range(len(dataset)):
            data = dataset[idx]
            x = data.x.unsqueeze(0).to(device)
            _, embedding = model(x, return_embedding=True)
            key = (
                int(data.y),
                int(data.walking_status),
                int(data.seq_num),
                int(data.angle),
            )
            records[key] = embedding.squeeze(0).cpu()

    return records


def evaluate_gallery_probe(records):
    gallery = {k: v for k, v in records.items() if k[1] == 0 and k[2] <= 4}
    probe_nm = {k: v for k, v in records.items() if k[1] == 0 and k[2] >= 5}
    probe_bg = {k: v for k, v in records.items() if k[1] == 1}
    probe_cl = {k: v for k, v in records.items() if k[1] == 2}

    gallery_per_angle = {
        angle: {k: v for k, v in gallery.items() if k[3] == angle}
        for angle in range(0, 181, 18)
    }

    correct = torch.zeros((3, 11, 11), dtype=torch.float32)
    total = torch.zeros((3, 11, 11), dtype=torch.float32)

    for gallery_angle in range(0, 181, 18):
        gallery_items = list(gallery_per_angle[gallery_angle].items())
        if not gallery_items:
            continue

        gallery_targets = [key for key, _ in gallery_items]
        gallery_embeddings = torch.stack([value for _, value in gallery_items])
        gallery_pos = gallery_angle // 18

        for probe_idx, probe in enumerate([probe_nm, probe_bg, probe_cl]):
            probe_items = list(probe.items())
            if not probe_items:
                continue

            probe_embeddings = torch.stack([value for _, value in probe_items])
            distances = torch.cdist(probe_embeddings, gallery_embeddings, p=2)

            for row_idx, (key, _) in enumerate(probe_items):
                subject_id, _, _, probe_angle = key
                probe_pos = probe_angle // 18
                nearest = torch.argmin(distances[row_idx]).item()
                nearest_target = gallery_targets[nearest]

                if nearest_target[0] == subject_id:
                    correct[probe_idx, gallery_pos, probe_pos] += 1
                total[probe_idx, gallery_pos, probe_pos] += 1

    accuracy = torch.where(total > 0, correct / total, torch.zeros_like(correct))

    for idx in range(3):
        accuracy[idx] -= torch.diag(torch.diag(accuracy[idx]))

    accuracy_flat = torch.sum(accuracy, dim=1) / 10
    summary = {
        'NM#5-6': float(torch.mean(accuracy_flat[0]).item()),
        'BG#1-2': float(torch.mean(accuracy_flat[1]).item()),
        'CL#1-2': float(torch.mean(accuracy_flat[2]).item()),
        'mean': float(torch.mean(accuracy).item()),
    }
    return accuracy_flat, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/casia_b.yaml')
    parser.add_argument('--checkpoint', default='output/model.pth')
    parser.add_argument('--num-subjects', type=int, default=None)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config['inference']['device'])

    model = build_model(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)

    dataset = CASIABPose(split='test', root=config['data']['root'], num_subjects=args.num_subjects)
    records = extract_embeddings(model, dataset, device)
    accuracy_flat, summary = evaluate_gallery_probe(records)

    print('Gallery/probe summary:')
    for key, value in summary.items():
        print(f'  {key}: {value:.4f}')

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({
                'summary': summary,
                'accuracy_flat': accuracy_flat.tolist(),
            }, f, indent=2)
        print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()