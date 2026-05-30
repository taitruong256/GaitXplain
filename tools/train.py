import argparse
import logging
import torch
import torch.nn as nn
import yaml
from pathlib import Path
import sys
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from models import GaitXplain, ProtoGCNCASIA
from datasets.casia_b_pose import CASIABPose
from tools import evaluate as evaluator
from tools.run_utils import make_run_dir, save_config_snapshot, setup_run_logging
def build_optimizer(model):
    return torch.optim.SGD(
        model.parameters(),
        lr=0.025,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True,
    )


def build_scheduler(optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)



def collate_pose_batch(batch, num_frames):
    xs = []
    ys = []
    for data in batch:
        x = data.x
        t = x.shape[0]

        if t < num_frames:
            if t > 0:
                pad = x[-1:].repeat(num_frames - t, 1, 1)
            else:
                pad = torch.zeros((num_frames, x.shape[1], x.shape[2]), dtype=x.dtype)
            x = torch.cat([x, pad], dim=0)
        elif t > num_frames:
            x = x[:num_frames]

        xs.append(x)
        ys.append(int(data.y.item()) if hasattr(data.y, 'item') else int(data.y))

    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/casia_b.yaml')
    parser.add_argument('--output', default=None)
    parser.add_argument('--num-subjects', type=int, default=None)
    parser.add_argument('--use-augmentation', action='store_true', default=None)
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

    use_augmentation = args.use_augmentation
    if use_augmentation is None:
        use_augmentation = config['training'].get('use_augmentation', True)

    device = torch.device(config['training']['device'])
    
    m_cfg = config['model']
  
    model_num_classes = m_cfg.get('num_classes')
    if num_subjects is not None:
        model_num_classes = num_subjects

    model_name = m_cfg.get('name', 'GaitXplain')
    if model_name.lower() in ('protogcn', 'proto_gcn'):
        model = ProtoGCNCASIA(num_classes=model_num_classes, num_person=1).to(device)
    else:
        model = GaitXplain(
            in_channels=m_cfg.get('in_channels', 3),
            hidden_channels=m_cfg.get('hidden_channels', 64),
            num_classes=model_num_classes,
            num_layers=m_cfg.get('num_layers', 3),
            dropout=m_cfg.get('dropout', 0.5),
        ).to(device)
    logger.info('%s', model)
    # initialize weights
    if hasattr(model, 'init_weights'):
        try:
            model.init_weights()
            logger.info('Model weights initialized via init_weights()')
        except Exception:
            logger.exception('init_weights() failed')
    logger.info('Total parameters: %d', sum(p.numel() for p in model.parameters()))
    
    dataset = CASIABPose(
        split='train',
        root=config['data']['root'],
        sequence_length=config['data']['num_frames'],
        use_augmentation=use_augmentation,
        num_subjects=num_subjects,
    )
    batch_size = config['training'].get('batch_size', 32)
    num_workers = config['training'].get('num_workers', 0)
    num_frames = config['data']['num_frames']
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=partial(collate_pose_batch, num_frames=num_frames),
    )

    logger.info('Number of training samples: %d', len(dataset))
    logger.info('Batch size: %d | Num batches: %d', batch_size, len(train_loader))
    logger.info('Augmentation enabled: %s', use_augmentation)
    for i in range(5):
        logger.info('Sample %d - x shape: %s, y: %s', i, dataset[i].x.shape, dataset[i].y)
    
    optimizer = build_optimizer(model) if model_name.lower() in ('protogcn', 'proto_gcn') else torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = build_scheduler(optimizer) if model_name.lower() in ('protogcn', 'proto_gcn') else None
    criterion = nn.CrossEntropyLoss()
    best_score = float('-inf') 
    best_path = None
    last_path = run_dir / 'last.pth'
    
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Train {epoch+1}/{config["training"]["epochs"]}', unit='batch')
        for batch_idx, (x, y) in enumerate(pbar, start=1):
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            # Request embeddings/logits for debug when available
            result = model(x, labels=y, return_embedding=True)
            if isinstance(result, tuple) and len(result) >= 2:
                losses_dict, logits, embedding = result
            else:
                losses_dict = result
                logits = None

            loss = losses_dict['loss_cls'] if isinstance(losses_dict, dict) else losses_dict
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            # Lightweight diagnostics for first epoch first few batches
            if epoch == 0 and batch_idx <= 3:
                try:
                    if logits is not None:
                        l_mean = logits.detach().cpu().mean().item()
                        l_std = logits.detach().cpu().std().item()
                        logger.info('Batch %d logits mean=%.4f std=%.4f', batch_idx, l_mean, l_std)
                    unique_labels = torch.unique(y.cpu())
                    logger.info('Batch %d unique labels=%s', batch_idx, unique_labels.tolist())
                    # compute grad norm
                    grad_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            grad_norm += float(p.grad.detach().norm().item() ** 2)
                    grad_norm = grad_norm ** 0.5
                    logger.info('Batch %d grad_norm=%.6f', batch_idx, grad_norm)
                except Exception:
                    logger.exception('Debug logging failed')

            pbar.set_postfix(batch_loss=f'{loss.item():.4f}', avg_loss=f'{(train_loss / batch_idx):.4f}')
        
        avg_loss = train_loss / max(len(train_loader), 1)
        logger.info('Epoch %d - Loss: %.4f', epoch + 1, avg_loss)

        if scheduler is not None:
            scheduler.step()

        save_checkpoint(model, optimizer, avg_loss, epoch, last_path)
        
        try:
            logger.info('Running validation for current epoch...')
            val_acc = evaluate_accuracy(model, config, device, num_subjects=num_subjects)
            logger.info('Validation accuracy: %.4f', val_acc)

            if val_acc > best_score:
                best_score = val_acc
                new_best_path = run_dir / f'best_epoch_{epoch+1:02d}_acc_{val_acc:.4f}.pth'
                torch.save(torch.load(last_path, map_location='cpu', weights_only=False), new_best_path)
                logger.info('New best checkpoint saved: %s', new_best_path)
                
                if best_path is not None and best_path.exists():
                    best_path.unlink()
                    logger.info('Deleted old best checkpoint: %s', best_path)
                
                best_path = new_best_path
        except Exception as e:
            logger.exception('Validation failed: %s', e)

    if best_path is None:
        best_path = run_dir / 'best_epoch_final.pth'
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


@torch.no_grad()
def evaluate_accuracy(model, config, device, num_subjects=None):
    # For gait recognition we evaluate using gallery/probe matching.
    # Use the evaluator's embedding extraction and gallery/probe routines.
    num_frames = config['data']['num_frames']
    batch_size = config['training'].get('batch_size', 32)

    dataset = CASIABPose(
        split='test',
        root=config['data']['root'],
        sequence_length=num_frames,
        use_augmentation=False,
        num_subjects=num_subjects,
    )

    records, _ = evaluator.extract_embeddings(
        model,
        dataset,
        device,
        batch_size=batch_size,
        num_frames=num_frames,
        show_progress=False,
        progress_desc='Val',
        with_loss=False,
    )

    _, summary = evaluator.evaluate_gallery_probe(records)
    # return mean recognition score (or 0.0 if missing)
    return float(summary.get('mean', 0.0))


if __name__ == '__main__':
    train()
