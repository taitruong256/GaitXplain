import sys
from pathlib import Path
from functools import partial

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from tools.losses import nt_xent_loss

sys.path.insert(0, str(Path(__file__).parent.parent))
from datasets.casia_b_pose import CASIABPose
from models import GaitXplain, ProtoGCN


def build_model(config):
    m_cfg = config['model']
    model_name = m_cfg.get('name', 'GaitXplain')
    if model_name.lower() == 'protogcn' or model_name.lower() == 'proto_gcn':
        return ProtoGCN(
            graph_cfg=m_cfg.get('graph_cfg', {'dataset': config.get('dataset', 'coco')}),
            in_channels=m_cfg.get('in_channels', 3),
            base_channels=m_cfg.get('base_channels', 96),
            ch_ratio=m_cfg.get('ch_ratio', 2),
            num_stages=m_cfg.get('num_stages', 10),
            inflate_stages=tuple(m_cfg.get('inflate_stages', (5, 8))),
            down_stages=tuple(m_cfg.get('down_stages', (5, 8))),
            data_bn_type=m_cfg.get('data_bn_type', 'VC'),
            num_person=m_cfg.get('num_person', 1),
            num_classes=m_cfg.get('num_classes'),
            dropout=m_cfg.get('dropout', 0.5),
            num_prototype=m_cfg.get('num_prototype', 100),
        )
    return GaitXplain(
        in_channels=m_cfg.get('in_channels', 3),
        hidden_channels=m_cfg.get('hidden_channels', 64),
        num_classes=m_cfg.get('num_classes'),
        num_layers=m_cfg.get('num_layers', 3),
        dropout=m_cfg.get('dropout', 0.5),
    )


def collate_pose_batch(batch, num_frames):
    """Collate function for batch processing with variable-length sequences."""
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


def extract_embeddings(model, dataset, device, batch_size=32, num_frames=None, show_progress=False, progress_desc='Eval', with_loss=False):
    """Extract embeddings using batch processing."""
    if num_frames is None:
        num_frames = 99  # default value
    
    records = {}
    model.eval()
    total_loss = 0.0
    processed_count = 0
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(collate_pose_batch, num_frames=num_frames),
    )

    with torch.no_grad():
        batch_iterator = data_loader
        if show_progress:
            batch_iterator = tqdm(batch_iterator, desc=progress_desc, unit='batch', total=len(data_loader))

        for batch_x, batch_y in batch_iterator:
            batch_size_current = batch_x.shape[0]
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            logits, embeddings = model(batch_x, return_embedding=True)

            if with_loss:
                batch_loss = nt_xent_loss(embeddings, batch_y, temperature=0.07)
                total_loss += batch_loss.item()
                if show_progress:
                    batch_iterator.set_postfix(batch_loss=f'{batch_loss.item():.4f}')

            # Store embeddings for each sample in the batch
            for i in range(batch_size_current):
                data_idx = processed_count + i
                if data_idx < len(dataset):
                    data = dataset[data_idx]
                    key = (
                        int(data.y),
                        int(data.walking_status),
                        int(data.seq_num),
                        int(data.angle),
                    )
                    records[key] = embeddings[i].cpu()
            
            processed_count += batch_size_current

    avg_loss = (total_loss / len(data_loader)) if with_loss and len(data_loader) > 0 else None
    return records, avg_loss


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


def evaluate_checkpoint(
    config,
    checkpoint_path,
    split='test',
    num_subjects=None,
    device=None,
    batch_size=None,
    show_progress=False,
    progress_desc='Eval',
    with_loss=False,
):
    device = device or torch.device(config.get('inference', {}).get('device', 'cpu'))
    batch_size = batch_size or config.get('training', {}).get('batch_size') or 32
    num_frames = config.get('data', {}).get('num_frames') or 99
    
    model = build_model(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)

    dataset = CASIABPose(
        split=split,
        root=config['data']['root'],
        sequence_length=num_frames,
        use_augmentation=False,
        num_subjects=num_subjects,
    )
    records, avg_loss = extract_embeddings(
        model,
        dataset,
        device,
        batch_size=batch_size,
        num_frames=num_frames,
        show_progress=show_progress,
        progress_desc=progress_desc,
        with_loss=with_loss,
    )
    accuracy_flat, summary = evaluate_gallery_probe(records)
    if avg_loss is not None:
        summary['loss'] = float(avg_loss)
    return accuracy_flat, summary