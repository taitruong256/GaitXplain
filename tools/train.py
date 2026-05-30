import argparse
import logging
import torch
import torch.nn as nn
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models import GaitXplain
from datasets.casia_b_pose import CASIABPose
from tools import evaluate as evaluator
from tools.run_utils import make_run_dir, save_config_snapshot, setup_run_logging


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/casia_b.yaml')
    parser.add_argument('--output', default=None)
    parser.add_argument('--num-subjects', type=int, default=None)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_root = args.output or config['training'].get('output_root') or 'output'
    run_dir = make_run_dir(output_root, 'train')
    logger = setup_run_logging(run_dir)
    save_config_snapshot(config, run_dir)
    logger.info('Run dir: %s', run_dir)
    logger.info('Config file: %s', args.config)

    num_subjects = args.num_subjects
    if num_subjects is None:
        num_subjects = config['training'].get('num_subjects')

    device = torch.device(config['training']['device'])
    
    m_cfg = config['model']
    model = GaitXplain(
        in_channels=m_cfg['in_channels'],
        hidden_channels=m_cfg['hidden_channels'],
        num_classes=m_cfg['num_classes'],
        num_layers=m_cfg['num_layers'],
        dropout=m_cfg['dropout']
    ).to(device)
    logger.info('%s', model)
    logger.info('Total parameters: %d', sum(p.numel() for p in model.parameters()))
    
    dataset = CASIABPose(
        split='train',
        root=config['data']['root'],
        num_subjects=num_subjects,
    )
    dataset_val = CASIABPose(split='test', root=config['data']['root'], num_subjects=num_subjects)
    logger.info('Number of training samples: %d', len(dataset))
    for i in range(5):
        logger.info('Sample %d - x shape: %s, y: %s', i, dataset[i].x.shape, dataset[i].y)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    best_score = float('-inf')
    best_path = run_dir / 'best.pth'
    last_path = run_dir / 'last.pth'
    
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0.0
        
        for idx in range(len(dataset)):
            data = dataset[idx]
            x = data.x.unsqueeze(0).to(device)
            y = torch.tensor([data.y], dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if (idx + 1) % 50 == 0:
                logger.info('[%d/%d] Sample %d/%d: %.4f', epoch + 1, config['training']['epochs'], idx + 1, len(dataset), loss.item())
        
        avg_loss = train_loss / len(dataset)
        logger.info('Epoch %d - Loss: %.4f', epoch + 1, avg_loss)

        save_checkpoint(model, optimizer, avg_loss, epoch, last_path)
        
        try:
            logger.info('Running validation for current epoch...')
            _, summary = evaluator.evaluate_checkpoint(
                config,
                checkpoint_path=last_path,
                split='test',
                num_subjects=num_subjects,
                device=device,
            )
            logger.info('Validation summary:')
            for k, v in summary.items():
                logger.info('  %s: %.4f', k, v)
            if summary['mean'] > best_score:
                best_score = summary['mean']
                torch.save(torch.load(last_path, map_location='cpu', weights_only=False), best_path)
                logger.info('New best checkpoint saved: %s', best_path)
        except Exception as e:
            logger.exception('Validation failed: %s', e)

    if not best_path.exists():
        torch.save(torch.load(last_path, map_location='cpu', weights_only=False), best_path)

    logger.info('Run dir: %s', run_dir)
    logger.info('Last checkpoint: %s', last_path)
    logger.info('Best checkpoint: %s', best_path)


def save_checkpoint(model, optimizer, loss, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    return path


if __name__ == '__main__':
    train()
