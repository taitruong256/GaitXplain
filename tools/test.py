import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
import numpy as np

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
        dropout=m_cfg['dropout']
    )


def preprocess(keypoints, max_frames):
    if keypoints.ndim == 2:
        keypoints = keypoints[np.newaxis, :, :]

    T = keypoints.shape[0]
    if T < max_frames:
        keypoints = np.pad(keypoints, ((0, max_frames-T), (0, 0), (0, 0)), mode='edge')
    elif T > max_frames:
        keypoints = keypoints[:max_frames]

    return torch.from_numpy(keypoints.astype(np.float32)).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/casia_b.yaml')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--output', default=None)
    parser.add_argument('--split', default=None)
    parser.add_argument('--sample-index', type=int, default=None)
    parser.add_argument('--num-subjects', type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    i_cfg = config.get('inference', {})
    device = torch.device(i_cfg.get('device', 'cpu'))
    checkpoint_path = args.checkpoint or i_cfg.get('checkpoint') or config['training'].get('checkpoint')
    output_path = args.output if args.output is not None else i_cfg.get('output')
    split = args.split or i_cfg.get('split', 'test')
    sample_index = args.sample_index if args.sample_index is not None else int(i_cfg.get('sample_index', 0))
    num_subjects = args.num_subjects if args.num_subjects is not None else i_cfg.get('num_subjects')

    model = build_model(config).to(device)
    if not Path(checkpoint_path).exists():
        print(f"Error: checkpoint not found: {checkpoint_path}")
        return
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)

    dataset = CASIABPose(split=split, root=config['data']['root'], num_subjects=num_subjects)
    if sample_index < 0 or sample_index >= len(dataset):
        print(f"Error: sample-index {sample_index} out of range for split '{split}'")
        return

    data = dataset[sample_index]
    keypoints = data.x.numpy()
    import numpy as np
    x = preprocess(keypoints, config['data']['num_frames']).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(x)

    probs = torch.softmax(logits, dim=1)
    pred_class = logits.argmax(dim=1).item()
    confidence = probs[0, pred_class].item()

    print(f"\nPredicted: {pred_class}, Confidence: {confidence:.4f}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                'predicted_class': int(pred_class),
                'confidence': float(confidence)
            }, f, indent=2)
        print(f"Saved: {output_path}")


if __name__ == '__main__':
    main()
