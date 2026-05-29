from torch_geometric.data import InMemoryDataset, Data 
from typing import Optional, Callable 
import os.path as osp 
from tqdm import tqdm
import numpy as np 
import torch
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning
)

class CASIABPose(InMemoryDataset):
    mapping_walking_status = {
        'nm': 0, 'bg': 1, 'cl': 2
    }
    split_ids = {
        'train': list(range(1, 75)),
        'test': list(range(75, 125))
    }

    def __init__(
        self, 
        root: str = 'data', 
        split: str = 'train', 
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None   
    ):
        self.split = split
        self.ids = self.split_ids[self.split] 

        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self) -> str:
        return f'casia_b_{self.split}.pt'

    def process(self):
        path = osp.join(self.root, 'casia-b', 'casia-b_pose_coco.csv')
        with open(path) as f:
            samples = f.readlines()[1:]
        
        sequences = {}
        for row in tqdm(samples, f"load [{self.split}]"):
            row = row.split(',')

            _, sequence_id, frame = row[0].split('/')
            subject_id, walking_status, sequence_num, view_angle = sequence_id.split('-')
            walking_status = self.mapping_walking_status[walking_status]
            key = subject_id, walking_status, sequence_num, view_angle
            keypoints = np.array(row[1:], dtype=np.float32) 

            if int(subject_id) not in self.ids:
                continue 

            if key not in sequences:
                sequences[key] = []

            sequences[key].append(
                torch.tensor(keypoints.reshape(-1, 3))
            )
    
        data_list = []
        for key, keypoints in tqdm(sequences.items(), f"process [{self.split}]"):
            subject_id, walking_status, sequence_num, view_angle = key
            
            if len(keypoints) == 0:
                continue
                
            data = Data(
                x=torch.stack(keypoints),
                y=int(subject_id),
                angle=int(view_angle),
                seq_num=int(sequence_num),
                walking_status=int(walking_status)
            )
            data_list.append(data)
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == '__main__':
    for split in ['train', 'test']:
        dataset = CASIABPose(split=split)
        print(f'[{split}] Number of samples: {len(dataset)}')
        print(f'Shape of x: {dataset[0].x.shape}, y: {dataset[0].y.shape}')

        max_x = []
        max_y = []
        for idx, item in enumerate(dataset):
            max_x.append(item.x[:, 0].max().item())
            max_y.append(item.x[:, 1].max().item())
        print(f'[{split}] Max x: {max(max_x)}, Max y: {max(max_y)}')