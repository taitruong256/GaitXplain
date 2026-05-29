import argparse
import torch
import torch.nn as nn
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models import GaitXplain
from datasets.casia_b_pose import CASIABPose
from tools import evaluate as evaluator


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/casia_b.yaml')
    parser.add_argument('--output', default=None)
    parser.add_argument('--num-subjects', type=int, default=None)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    output_path = args.output or config['training'].get('checkpoint') or 'output/model.pth'
    num_subjects = args.num_subjects
    if num_subjects is None:
        num_subjects = config['training'].get('num_subjects')

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(config['training']['device'])
    
    m_cfg = config['model']
    model = GaitXplain(
        in_channels=m_cfg['in_channels'],
        hidden_channels=m_cfg['hidden_channels'],
        num_classes=m_cfg['num_classes'],
        num_layers=m_cfg['num_layers'],
        dropout=m_cfg['dropout']
    ).to(device)
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    dataset = CASIABPose(
        split='train',
        root=config['data']['root'],
        num_subjects=num_subjects,
    )
    print(f"Number of training samples: {len(dataset)}")
    for i in range(5):
        print(f"Sample {i} - x shape: {dataset[i].x.shape}, y: {dataset[i].y}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
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
                print(f"[{epoch+1}/{config['training']['epochs']}] Sample {idx+1}/{len(dataset)}: {loss.item():.4f}")
        
        avg_loss = train_loss / len(dataset)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % config['training']['save_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, output_path)
            print(f"Saved: {output_path}")

        try:
            print("Running validation on saved checkpoint...")
            evaluator_config = config
            device_eval = device
            model_eval = evaluator.build_model(evaluator_config).to(device_eval)
            ckpt = torch.load(output_path, map_location=device_eval, weights_only=False)
            model_eval.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
            dataset_val = CASIABPose(split='test', root=config['data']['root'], num_subjects=num_subjects)
            records = evaluator.extract_embeddings(model_eval, dataset_val, device_eval)
            _, summary = evaluator.evaluate_gallery_probe(records)
            print('Validation summary (on save):')
            for k, v in summary.items():
                print(f"  {k}: {v:.4f}")
        except Exception as e:
            print(f"Validation failed: {e}")
            
        try:
            print("Running final validation on trained model...")
            evaluator_config = config
            device_eval = device
            model_eval = evaluator.build_model(evaluator_config).to(device_eval)
            ckpt = torch.load(output_path, map_location=device_eval, weights_only=False)
            model_eval.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
            dataset_val = CASIABPose(split='test', root=config['data']['root'], num_subjects=num_subjects)
            records = evaluator.extract_embeddings(model_eval, dataset_val, device_eval)
            _, summary = evaluator.evaluate_gallery_probe(records)
            print('Final validation summary:')
            for k, v in summary.items():
                print(f"  {k}: {v:.4f}")
        except Exception as e:
            print(f"Final validation failed: {e}")


if __name__ == '__main__':
    train()
