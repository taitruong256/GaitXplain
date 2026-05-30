import argparse
import logging
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.run_utils import make_run_dir, save_config_snapshot, setup_run_logging
from tools import evaluate as evaluator


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def find_latest_best_checkpoint(output_root: str) -> Path | None:
    root = Path(output_root)
    train_runs = sorted(root.glob('train_*'), key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in train_runs:
        best_path = run_dir / 'best.pth'
        if best_path.exists():
            return best_path
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/casia_b.yaml')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--output', default=None)
    parser.add_argument('--num-subjects', type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    i_cfg = config.get('inference', {})
    device = torch.device(i_cfg.get('device', 'cpu'))
    run_root = i_cfg.get('output_root') or config['training'].get('output_root') or 'output'
    checkpoint_path = args.checkpoint or i_cfg.get('checkpoint') or find_latest_best_checkpoint(run_root)
    output_path = args.output if args.output is not None else None
    num_subjects = args.num_subjects if args.num_subjects is not None else i_cfg.get('num_subjects')

    run_dir = make_run_dir(run_root, 'test')
    logger = setup_run_logging(run_dir)
    save_config_snapshot(config, run_dir)
    logger.info('Run dir: %s', run_dir)
    logger.info('Checkpoint: %s', checkpoint_path)

    if checkpoint_path is None:
        logger.error('Error: no best checkpoint found under %s', run_root)
        return
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logger.error('Error: checkpoint not found: %s', checkpoint_path)
        return

    _, summary = evaluator.evaluate_checkpoint(
        config,
        checkpoint_path=checkpoint_path,
        split='test',
        num_subjects=num_subjects,
        device=device,
    )

    logger.info('Test summary:')
    for key, value in summary.items():
        logger.info('  %s: %.4f', key, value)
    logger.info('summary=%s', json.dumps(summary))

    final_output = Path(output_path) if output_path is not None else run_dir / 'result.json'
    with open(final_output, 'w') as f:
        json.dump({
            'checkpoint': str(checkpoint_path),
            'summary': summary,
        }, f, indent=2)
    logger.info('Saved: %s', final_output)
    logger.info('output=%s', final_output)


if __name__ == '__main__':
    main()
